import math
from typing import Any, Mapping, Sequence
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

from ..core import ModelConfig
from ..core.context_parallel import split_sequence_indices
from ..core.device.npu_compatible_device import get_device_type
from ..diffusion import FlowMatchScheduler
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit
from ..models.minimax_h3_dit import MiniMaxH3DiT, patchify_video, unpatchify_video, pack_audio, unpack_audio
from ..models.minimax_h3_text_encoder import (
    MiniMaxH3TextEncoder, presentation_t2va, presentation_fl2va, presentation_ref2va,
    sample_qwen_video_frames, image_token_counts, video_token_counts,
)
from ..models.minimax_h3_video_vae import MiniMaxH3VideoVAE
from ..models.minimax_h3_audio_vae import MiniMaxH3AudioVAE
from ..utils.data.audio import convert_to_stereo, resample_waveform
from ..utils.lora.minimax_h3 import MiniMaxH3LoRALoader
from .minimax_h3_continuation import (
    H3AudioLatentContinuation,
    H3PipelineResult,
    H3VideoLatentContinuation,
    apply_audio_latent_continuation,
    apply_video_latent_continuation,
    normalize_audio_to_timeline,
    project_video_motion_anchor,
    video_motion_anchor,
)
# ---------------------------------------------------------------------------
# Long-horizon memory slots
# ---------------------------------------------------------------------------
# A memory slot is a clean conditioning block packed into the sequence next to
# the prediction window.  Slots are described by plain dicts so that one
# description travels through the latent cache, the manifest and the pipeline:
#
#     {"name": "stm",                    # free-form label (logging / ablation)
#      "tensor": Tensor[1,24,t,h,w],     # latent slot; patchified inside the DiT
#      "lead_steps": None}               # None = contiguous, right before the window
#
#     {"name": "ltm",                    # compressed slot (memory compressor output)
#      "tensor": Tensor[m, 96],
#      "kind": "rows",
#      "lead_steps": 36}
#
# ``lead_steps`` is the distance, in *video latent steps* on the 17n+5 grid, from
# the first token of the prediction window back to the first token of the slot.
# ``None`` means "end exactly where the window begins", which is the M1-fast STM
# semantics.  A fixed ``lead_steps`` is the LTM anchor: it keeps a sequence-head
# slot at a constant, bounded relative distance no matter how many shots have
# already been generated.  Without such an anchor the relative distance grows
# without bound and the slot drifts out of the trained position range.

MEMORY_SLOT_KEYS = ("name", "kind", "tensor", "latent_t", "lead_steps", "rows")


def normalize_memory_slots(memory_latents, *, latent_h=None, latent_w=None):
    """Normalise a memory input into an ordered list of slot dicts.

    Accepted inputs:

      * ``None``                    -> ``[]``
      * ``Tensor[1,24,t,h,w]``      -> one contiguous latent slot (M1-fast)
      * a list whose items are either ``Tensor[1,24,t,h,w]`` or a slot dict with
        ``tensor`` plus optional ``name`` / ``kind`` / ``lead_steps``.
    """
    if memory_latents is None:
        return []
    items = memory_latents if isinstance(memory_latents, (list, tuple)) else [memory_latents]
    slots = []
    for index, item in enumerate(items):
        if isinstance(item, torch.Tensor):
            spec = {"tensor": item}
        elif isinstance(item, Mapping):
            spec = dict(item)
        else:
            raise TypeError(f"unsupported memory slot {index}: {type(item)!r}")
        tensor = spec.get("tensor")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"memory slot {index} must carry a torch.Tensor under 'tensor'")
        kind = spec.get("kind") or ("latent" if tensor.dim() == 5 else "rows")
        if kind == "latent":
            if tensor.dim() != 5:
                raise ValueError(
                    f"latent memory slot {index} must be [1,24,t,h,w], got {tuple(tensor.shape)}"
                )
            if latent_h is not None and tuple(tensor.shape[-2:]) != (int(latent_h), int(latent_w)):
                raise ValueError(
                    f"memory slot {index} spatial shape {tuple(tensor.shape[-2:])} must match the "
                    f"window's {(int(latent_h), int(latent_w))}; memory tokens share the window's spatial grid"
                )
            latent_t = int(tensor.shape[-3])
            rows = latent_t * int(tensor.shape[-2]) // 2 * int(tensor.shape[-1]) // 2
        elif kind == "rows":
            if tensor.dim() != 2:
                raise ValueError(f"compressed memory slot {index} must be [m,96], got {tuple(tensor.shape)}")
            if spec.get("lead_steps") is None:
                raise ValueError(f"compressed memory slot {index} must declare lead_steps")
            latent_t = None
            rows = int(tensor.shape[0])
        else:
            raise ValueError(f"unknown memory slot kind {kind!r}")
        lead = spec.get("lead_steps")
        slots.append({
            "name": str(spec.get("name") or f"mem{index}"),
            "kind": kind,
            "tensor": tensor,
            "latent_t": latent_t,
            "rows": rows,
            "lead_steps": None if lead is None else int(lead),
        })
    return slots


def memory_slot_rows(slots):
    """Patchify every latent slot; compressed slots are already row-shaped."""
    blocks = []
    for slot in slots:
        blocks.append(slot["tensor"] if slot["kind"] == "rows" else patchify_video(slot["tensor"]))
    return blocks


def _encode_audio_waveform(pipe: "MiniMaxH3Pipeline", waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Encode an AI2V driving waveform using H3's audio VAE contract."""
    if waveform.dim() == 3:
        waveform = waveform.squeeze(0)
    if waveform.dim() != 2:
        raise ValueError(f"expected AI2V waveform [C, L], got {list(waveform.shape)}")
    waveform = convert_to_stereo(waveform.float())
    if int(sample_rate) != pipe.audio_vae.sample_rate:
        waveform = resample_waveform(
            waveform, int(sample_rate), int(pipe.audio_vae.sample_rate)
        )
    pipe.load_models_to_device(("audio_vae",))
    return pipe.audio_vae.encode_audio(
        waveform[:2].to(pipe.device), dtype=pipe.torch_dtype
    )


class MiniMaxH3Pipeline(BasePipeline):

    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(device=device, torch_dtype=torch_dtype, height_division_factor=32, width_division_factor=32, time_division_factor=17, time_division_remainder=5)
        self.scheduler = FlowMatchScheduler("MiniMax-H3")
        self.scheduler_audio = FlowMatchScheduler("MiniMax-H3")
        self.text_encoder: MiniMaxH3TextEncoder = None
        self.dit: MiniMaxH3DiT = None
        self.video_vae: MiniMaxH3VideoVAE = None
        self.audio_vae: MiniMaxH3AudioVAE = None
        self.tokenizer = None
        self.processor = None
        self.imgvid_cond_noise_aug = 0.999
        self.audio_cond_noise_aug = 1.0
        # v3 windows share one physical temporal RoPE origin.  It is learned
        # from the first window's positive prompt and reused for later calls.
        self._h3_v3_temporal_origin = None
        self.in_iteration_models = ("dit",)
        self.units = [
            MiniMaxH3Unit_ShapeChecker(),
            MiniMaxH3Unit_NoiseInitializer(),
            MiniMaxH3Unit_InputVideoEmbedder(),
            MiniMaxH3Unit_InputAudioEmbedder(),
            MiniMaxH3Unit_VideoLatentContinuationEmbedder(),
            MiniMaxH3Unit_AudioLatentContinuationEmbedder(),
            MiniMaxH3Unit_VideoRetakeEmbedder(),
            MiniMaxH3Unit_AudioRetakeEmbedder(),
            MiniMaxH3Unit_KeyframeEncoder(),
            MiniMaxH3Unit_ReferenceEncoder(),
            MiniMaxH3Unit_PromptEmbedder(),
            MiniMaxH3Unit_PackedSequenceBuilder(),
        ]
        self.model_fn = model_fn_minimax_h3
        self.compilable_models = ["dit"]
        self.lora_loader = MiniMaxH3LoRALoader

    def _rebuild_dense_wws_packed(
        self,
        inputs_shared: dict[str, Any],
        inputs_posi: dict[str, Any],
        inputs_nega: dict[str, Any],
        *,
        video_start: int,
        audio_start: int,
        total_video_steps: int,
        total_audio_steps: int,
    ) -> None:
        """Bind one dense WWS window's H3 positions to the global timeline."""
        if inputs_shared.get("ref_blocks") is not None:
            if video_start or audio_start:
                raise ValueError(
                    "joint MultiDiffusion absolute WWS positions are not supported "
                    "with reference blocks outside the first window"
                )
            return

        video_latents = inputs_shared["video_latents"]
        audio_latents = inputs_shared["audio_latents"]
        video_source_indices = tuple(range(video_start, video_start + video_latents.shape[2]))
        audio_source_indices = tuple(range(audio_start, audio_start + audio_latents.shape[-1]))
        packed_builder = self.units[-1]
        for condition in (inputs_posi, inputs_nega):
            condition["packed"] = packed_builder.process(
                self,
                prompt_embeds=condition["prompt_embeds"],
                video_latents=video_latents,
                audio_latents=audio_latents,
                text_token_tags=condition.get("text_token_tags"),
                keyframe_cond_anchor=inputs_shared.get("keyframe_cond_anchor"),
                keyframe_indices=inputs_shared.get("keyframe_indices"),
                ref_blocks=None,
                video_source_indices=video_source_indices,
                audio_source_indices=audio_source_indices,
                global_video_latent_length=total_video_steps,
                global_audio_latent_length=total_audio_steps,
            )["packed"]

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = get_device_type(),
        model_configs: list[ModelConfig] = [],
        processor_config: ModelConfig = ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/processor/"),
        vram_limit: float = None,
        redirect_common_files: bool = True,
    ):
        if redirect_common_files:
            redirect_dict = {
                "MiniMax/MiniMax-H3": "MiniMaxAI/MiniMax-H3",
            }
            for model_config in model_configs:
                if model_config.require_downloading() and model_config.parse_download_source() == "huggingface":
                    if model_config.model_id is not None and model_config.model_id in redirect_dict:
                        print(f"The model is detected to be downloading from HuggingFace. {model_config.model_id} is redirected to {redirect_dict[model_config.model_id]}. You can use `redirect_common_files=False` to disable file redirection.")
                        model_config.model_id = redirect_dict[model_config.model_id]
            if processor_config is not None and processor_config.require_downloading() and processor_config.parse_download_source() == "huggingface":
                if processor_config.model_id is not None and processor_config.model_id in redirect_dict:
                    print(f"The model is detected to be downloading from HuggingFace. {processor_config.model_id} is redirected to {redirect_dict[processor_config.model_id]}. You can use `redirect_common_files=False` to disable file redirection.")
                    processor_config.model_id = redirect_dict[processor_config.model_id]
        pipe = MiniMaxH3Pipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)
        pipe.text_encoder = model_pool.fetch_model("minimax_h3_text_encoder")
        pipe.dit = model_pool.fetch_model("minimax_h3_dit")
        pipe.video_vae = model_pool.fetch_model("minimax_h3_video_vae")
        pipe.audio_vae = model_pool.fetch_model("minimax_h3_audio_vae")
        if processor_config is not None:
            processor_config.download_if_necessary()
            pipe.processor = AutoProcessor.from_pretrained(processor_config.path)
            pipe.tokenizer = pipe.processor.tokenizer
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe

    @torch.no_grad()
    def __call__(
        self,
        prompt: str = None,
        negative_prompt: str = " ",
        height: int = 768,
        width: int = 1344,
        num_frames: int = 124,
        num_inference_steps: int = 50,
        seed: int = 42,
        rand_device: str = "cpu",
        cfg_scale: float = 1.0,
        flow_shift: float = 12.0,
        audio_flow_shift: float = 3.0,
        tiled: bool = True,
        tile_size: int = 256,
        tile_overlap: int = 64,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        # Keyframe to Video
        keyframes: list[Image.Image] = None,
        keyframe_indices: list[int] = None,
        # Reference to Video
        references: list[dict] = None,
        ref_image_short_edge: int = 2048,
        ref_video_short_edge: int = 768,
        ref_video_max_pixels: int = 768 * 1344,
        # Video / Audio Retake
        retake_video: list[Image.Image] = None,
        frame_regions_to_retake: list[tuple[float, float]] = None,
        retake_audio: torch.Tensor = None,
        retake_audio_sample_rate: int = 32000,
        seconds_regions_to_retake: list[tuple[float, float]] = None,
        # Latent continuation (mutually exclusive with same-modality Retake)
        continuation_video_latents: H3VideoLatentContinuation = None,
        continuation_audio_latents: H3AudioLatentContinuation = None,
        # Long-horizon memory: a compact clean video-latent block placed right
        # before the current window (see 长时记忆模块训练方案.md §13/§16).
        memory_latents: torch.Tensor = None,
        # AI2V driving audio. The audio latent is kept clean during denoising
        # and serves as the audio-side condition for video generation.
        ai2v_audio: tuple[torch.Tensor, int] | None = None,
        video_source_indices: Sequence[int] | None = None,
        audio_source_indices: Sequence[int] | None = None,
        global_video_latent_length: int | None = None,
        global_audio_latent_length: int | None = None,
        global_temporal_position_origin: float | None = None,
        return_latents: bool = False,
        decode_output: bool = True,
        progress_bar_cmd=tqdm,
        # Template inputs
        text_embedding: torch.Tensor = None,
    ):
        """Generate a joint video + audio sample.

        `num_frames` is snapped up to the nearest 17n+5 and that aligned value
        drives every downstream shape (video/audio latent lengths, reference video
        capping), so the returned clip may be slightly longer than requested.

        `keyframes` (FL2AV) is a list of PIL images with `keyframe_indices` in
        {0, -1}; both are resized onto the target canvas.

        `references` (Ref2AV) is a list of dicts in request order:

            {"type": "image",       "image": PIL.Image}
            {"type": "video",       "video": list[PIL.Image]}   # silent
            {"type": "audio",       "audio": Tensor[C, L], "sample_rate": int}
            {"type": "video_audio", "video": list[PIL.Image],
                                    "audio": Tensor[C, L], "sample_rate": int}

        Input contract: `video` frame lists must ALREADY. be 24fps the pipeline never resamples frame rate.
        """
        if continuation_video_latents is not None and retake_video is not None:
            raise ValueError("continuation_video_latents and retake_video are mutually exclusive")
        if continuation_audio_latents is not None and retake_audio is not None:
            raise ValueError("continuation_audio_latents and retake_audio are mutually exclusive")
        if ai2v_audio is not None and continuation_audio_latents is not None:
            raise ValueError("ai2v_audio and continuation_audio_latents are mutually exclusive")
        if video_source_indices is not None and len(video_source_indices) > 0 and int(video_source_indices[0]) == 0:
            # A new v3 timeline starts at source index zero.  Reset the
            # auto-derived origin so reusing a pipeline cannot leak state from
            # a previous continuation run.
            self._h3_v3_temporal_origin = None
        if global_temporal_position_origin is not None:
            self._h3_v3_temporal_origin = float(global_temporal_position_origin)
        self.scheduler.set_timesteps(num_inference_steps, shift=flow_shift)
        self.scheduler_audio.set_timesteps(num_inference_steps, shift=audio_flow_shift)

        ai2v_audio_latent = None
        if ai2v_audio is not None:
            waveform, sample_rate = ai2v_audio
            ai2v_audio_latent = _encode_audio_waveform(
                self, waveform, int(sample_rate)
            ).to(self.torch_dtype)

        inputs_posi = {"prompt": prompt}
        inputs_nega = {"negative_prompt": negative_prompt}
        inputs_shared = {
            "cfg_scale": cfg_scale,
            "height": height, "width": width, "num_frames": num_frames,
            "seed": seed, "rand_device": rand_device,
            "tiled": tiled, "tile_size": tile_size, "tile_overlap": tile_overlap,
            "use_gradient_checkpointing": use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": use_gradient_checkpointing_offload,
            "keyframes": keyframes, "keyframe_indices": keyframe_indices,
            "references": references, "ref_image_short_edge": ref_image_short_edge, "ref_video_short_edge": ref_video_short_edge, "ref_video_max_pixels": ref_video_max_pixels,
            "retake_video": retake_video, "frame_regions_to_retake": frame_regions_to_retake,
            "retake_audio": (retake_audio, retake_audio_sample_rate) if retake_audio is not None else None, "seconds_regions_to_retake": seconds_regions_to_retake,
            "continuation_video_latents": continuation_video_latents,
            "continuation_audio_latents": continuation_audio_latents,
            "memory_latents": memory_latents,
            "ai2v_audio": ai2v_audio,
            "video_source_indices": video_source_indices,
            "audio_source_indices": audio_source_indices,
            "global_video_latent_length": global_video_latent_length,
            "global_audio_latent_length": global_audio_latent_length,
            "global_temporal_position_origin": global_temporal_position_origin,
            "imgvid_cond_noise_aug": self.imgvid_cond_noise_aug, "audio_cond_noise_aug": self.audio_cond_noise_aug,
            "text_embedding": text_embedding,
        }

        # 3. Unit chain
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # 4. Denoise loop
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep_video in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep_video = timestep_video.unsqueeze(0).to(dtype=torch.float32, device=self.device)
            timestep_audio = self.scheduler_audio.timesteps[progress_id].unsqueeze(0).to(dtype=torch.float32, device=self.device)
            if ai2v_audio_latent is not None:
                target_audio_steps = inputs_shared["audio_latents"].shape[-1]
                if ai2v_audio_latent.shape[-1] < target_audio_steps:
                    raise ValueError(
                        "ai2v_audio is shorter than the resolved video window: "
                        f"audio_latent_steps={ai2v_audio_latent.shape[-1]}, "
                        f"required={target_audio_steps}"
                    )
                # Keep the driving audio clean at every denoise iteration.
                inputs_shared["audio_latents"] = ai2v_audio_latent[..., :target_audio_steps]
                timestep_audio = torch.ones_like(timestep_audio)
            inputs_shared["video_latents"] = project_video_motion_anchor(
                inputs_shared["video_latents"], self.scheduler, timestep_video,
                inputs_shared.get("motion_anchor_latents_video"),
                inputs_shared.get("motion_anchor_noise_video"),
                inputs_shared.get("motion_anchor_weights_video"),
                start_step=int(inputs_shared.get("motion_anchor_start_video", 0)),
            )
            noise_pred_video, noise_pred_audio = self.cfg_guided_model_fn(
                self.model_fn, cfg_scale, inputs_shared, inputs_posi, inputs_nega,
                **models, timestep_video=timestep_video, timestep_audio=timestep_audio,
            )
            inputs_shared["video_latents"] = self.step(
                self.scheduler, inputs_shared["video_latents"], progress_id, noise_pred=noise_pred_video,
                inpaint_mask=inputs_shared.get("denoise_mask_video"), input_latents=inputs_shared.get("input_latents_video"),
            )
            if ai2v_audio_latent is None:
                inputs_shared["audio_latents"] = self.step(
                    self.scheduler_audio, inputs_shared["audio_latents"], progress_id, noise_pred=noise_pred_audio,
                    inpaint_mask=inputs_shared.get("denoise_mask_audio"), input_latents=inputs_shared.get("input_latents_audio"),
                )
            else:
                inputs_shared["audio_latents"] = ai2v_audio_latent[..., :inputs_shared["audio_latents"].shape[-1]]


        # 5. Decode.  Continuation callers can defer decoding when they will
        # use ``decode_continuation_suffix`` with the previous latent context.
        if not decode_output and not return_latents:
            raise ValueError("decode_output=False requires return_latents=True")
        if not decode_output:
            video_latents = inputs_shared["video_latents"].detach().clone()
            audio_latents = inputs_shared["audio_latents"].detach().clone()
            return H3PipelineResult(
                video=None,
                audio=None,
                video_latents=video_latents,
                audio_latents=audio_latents,
                resolved_video_frames=inputs_shared["num_frames"],
                video_fps=24,
                audio_sample_rate=self.audio_vae.sample_rate,
                video_latent_steps=video_latents.shape[2],
                audio_latent_steps=audio_latents.shape[-1],
            )

        self.load_models_to_device(["video_vae"])
        frames = self.video_vae.decode_video(inputs_shared["video_latents"], dtype=self.torch_dtype, tiled=tiled, tile_size=tile_size, tile_overlap=tile_overlap)
        video = self.vae_output_to_video(frames, min_value=0, max_value=1)

        self.load_models_to_device(["audio_vae"])
        waveform = self.audio_vae.decode_audio(inputs_shared["audio_latents"], dtype=self.torch_dtype)
        audio = self.output_audio_format_check(waveform)
        if return_latents:
            video_latents = inputs_shared["video_latents"].detach().clone()
            audio_latents = inputs_shared["audio_latents"].detach().clone()
            resolved_frames = inputs_shared["num_frames"]
            return H3PipelineResult(
                video=video,
                audio=audio,
                video_latents=video_latents,
                audio_latents=audio_latents,
                resolved_video_frames=resolved_frames,
                video_fps=24,
                audio_sample_rate=self.audio_vae.sample_rate,
                video_latent_steps=video_latents.shape[2],
                audio_latent_steps=audio_latents.shape[-1],
            )
        return video, audio

    @torch.no_grad()
    def decode_continuation_suffix(
        self,
        *,
        previous_video_latents: torch.Tensor | None,
        current_video_latents: torch.Tensor | None,
        previous_audio_latents: torch.Tensor | None,
        current_audio_latents: torch.Tensor | None,
        overlap_video_steps: int,
        overlap_audio_steps: int,
        overlap_video_frames: int,
        overlap_audio_samples: int,
        tiled: bool = True,
        tile_size: int = 256,
        tile_overlap: int = 64,
    ):
        """Decode a continuation suffix with the previous latent tail as context.

        The current window's overlap latents are used for denoising, but are
        omitted from this decode sequence.  Replacing them with the previous
        clean tail gives the temporal VAE the same left context as the emitted
        history, while only the newly owned suffix is returned.
        """
        video = audio = None
        if current_video_latents is not None:
            if previous_video_latents is None:
                raise ValueError("previous_video_latents is required for video context decoding")
            video_decode_latents = torch.cat(
                [previous_video_latents, current_video_latents[:, :, overlap_video_steps:]], dim=2
            )
            self.load_models_to_device(["video_vae"])
            frames = self.video_vae.decode_video(
                video_decode_latents, dtype=self.torch_dtype, tiled=tiled,
                tile_size=tile_size, tile_overlap=tile_overlap,
            )
            frames = frames[:, :, overlap_video_frames:]
            video = self.vae_output_to_video(frames, min_value=0, max_value=1)

        if current_audio_latents is not None:
            if previous_audio_latents is None:
                raise ValueError("previous_audio_latents is required for audio context decoding")
            audio_decode_latents = torch.cat(
                [previous_audio_latents, current_audio_latents[..., overlap_audio_steps:]], dim=-1
            )
            self.load_models_to_device(["audio_vae"])
            waveform = self.audio_vae.decode_audio(audio_decode_latents, dtype=self.torch_dtype)
            waveform = waveform[..., overlap_audio_samples:]
            audio = self.output_audio_format_check(waveform)
        return video, audio

    @torch.no_grad()
    def decode_continuation_bridge(
        self,
        *,
        previous_video_latents: torch.Tensor | None,
        current_video_latents: torch.Tensor | None,
        previous_audio_latents: torch.Tensor | None = None,
        current_audio_latents: torch.Tensor | None = None,
        overlap_video_steps: int,
        overlap_audio_steps: int,
        overlap_video_frames: int,
        overlap_audio_samples: int,
        bridge_video_frames: int,
        bridge_audio_samples: int = 0,
        tiled: bool = True,
        tile_size: int = 256,
        tile_overlap: int = 64,
    ):
        """Decode a short post-boundary bridge with historical context."""
        if bridge_video_frames < 0 or bridge_audio_samples < 0:
            raise ValueError("bridge lengths must be non-negative")
        video, audio = self.decode_continuation_suffix(
            previous_video_latents=previous_video_latents,
            current_video_latents=current_video_latents,
            previous_audio_latents=previous_audio_latents,
            current_audio_latents=current_audio_latents,
            overlap_video_steps=overlap_video_steps,
            overlap_audio_steps=overlap_audio_steps,
            overlap_video_frames=overlap_video_frames,
            overlap_audio_samples=overlap_audio_samples,
            tiled=tiled,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
        )
        if video is not None:
            video = list(video[:bridge_video_frames])
        if audio is not None:
            audio = audio[..., :bridge_audio_samples]
        return video, audio

    @torch.no_grad()
    def decode_latent_timeline(
        self,
        *,
        video_latents: torch.Tensor | None,
        audio_latents: torch.Tensor | None,
        tiled: bool = True,
        tile_size: int = 256,
        tile_overlap: int = 64,
    ):
        """Decode an already assembled latent timeline exactly once."""
        video = audio = None
        if video_latents is not None:
            self.load_models_to_device(["video_vae"])
            frames = self.video_vae.decode_video(
                video_latents, dtype=self.torch_dtype, tiled=tiled,
                tile_size=tile_size, tile_overlap=tile_overlap,
            )
            video = self.vae_output_to_video(frames, min_value=0, max_value=1)
        if audio_latents is not None:
            self.load_models_to_device(["audio_vae"])
            waveform = self.audio_vae.decode_audio(audio_latents, dtype=self.torch_dtype)
            audio = self.output_audio_format_check(waveform)
        return video, audio

    @torch.no_grad()
    def export_text_embedding(self, prompt: str):
        inputs_posi = {"prompt": prompt}
        inputs_nega = {"negative_prompt": ""}
        inputs_shared = {
            # These parameters are placeholders
            "cfg_scale": 1,
            "height": 480, "width": 480, "num_frames": 5,
            "seed": 0, "rand_device": "cpu",
            "imgvid_cond_noise_aug": self.imgvid_cond_noise_aug, "audio_cond_noise_aug": self.audio_cond_noise_aug,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)
        return inputs_posi["prompt_embeds"]


class MiniMaxH3Unit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames"),
            output_params=("height", "width", "num_frames"),
        )

    def process(self, pipe: MiniMaxH3Pipeline, height, width, num_frames):
        height, width, num_frames = pipe.check_resize_height_width(height, width, num_frames)
        return {"height": height, "width": width, "num_frames": num_frames}


class MiniMaxH3Unit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("seed", "num_frames", "height", "width", "rand_device"),
            output_params=("video_latents", "audio_latents"),
        )

    def process(self, pipe: MiniMaxH3Pipeline, seed, num_frames, height, width, rand_device):
        video_latent_t, latent_h, latent_w = ((num_frames - 5) // 17) * 5 + 2, height // 16, width // 16
        audio_latent_t = round(num_frames / 24.0 * 40.0)
        video_latents = pipe.generate_noise((1, 24, video_latent_t, latent_h, latent_w), seed=seed, rand_device=rand_device, rand_torch_dtype=pipe.torch_dtype)
        audio_latents = pipe.generate_noise((2, 32, audio_latent_t), seed=seed, rand_device=rand_device, rand_torch_dtype=pipe.torch_dtype)
        return {"video_latents": video_latents, "audio_latents": audio_latents}


class MiniMaxH3Unit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt"},
            input_params_nega={"prompt": "negative_prompt"},
            input_params=("keyframes", "ref_blocks", "height", "width", "text_embedding"),
            output_params=("prompt_embeds", "text_token_tags"),
            onload_model_names=("text_encoder",),
        )

    def preprocess_ref_blocks(self, pipe: MiniMaxH3Pipeline, prompt, ref_blocks):
        pixel_values = image_grid_thw = pixel_values_videos = video_grid_thw = None
        counters = {"image": 0, "audio": 0, "video": 0}
        images, videos, timestamps_per_video, condition_labels = [], [], [], []
        for block in ref_blocks:
            kind = block["kind"]
            if kind == "image":
                counters["image"] += 1
                condition_labels.append(("image", counters["image"]))
                images.append(block["prepared_image"])
            elif kind == "audio":
                counters["audio"] += 1
                condition_labels.append(("audio", counters["audio"]))
            elif kind in ("video", "video_audio"):
                if int(block["ref_audio_t"]) > 0:
                    counters["audio"] += 1
                    condition_labels.append(("audio", counters["audio"]))
                counters["video"] += 1
                condition_labels.append(("video", counters["video"]))
                sampled, timestamps = sample_qwen_video_frames(block["prepared_frames"])
                videos.append(np.stack([np.asarray(f) for f in sampled]))
                timestamps_per_video.append(timestamps)
            else:
                raise ValueError(f"unknown reference kind: {kind}")

        image_counts, video_counts, video_timestamps = [], [], []
        if len(images) > 0:
            pixel_values, image_grid_thw, image_counts = image_token_counts(pipe.processor, images)
        if len(videos) > 0:
            pixel_values_videos, video_grid_thw, video_counts, video_timestamps = video_token_counts(pipe.processor, videos, timestamps_per_video)
        input_ids, text_token_tags = presentation_ref2va(pipe.tokenizer, prompt, condition_labels, image_counts, video_counts, video_timestamps)
        return input_ids, text_token_tags, pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw

    def process(self, pipe: MiniMaxH3Pipeline, prompt, keyframes=None, ref_blocks=None, height=None, width=None, text_embedding=None):
        if text_embedding is not None and prompt is None:
            return {"prompt_embeds": text_embedding.to(pipe.device, pipe.torch_dtype), "text_token_tags": torch.ones((text_embedding.shape[0],), device=pipe.device, dtype=torch.long)}
        pipe.load_models_to_device(self.onload_model_names)
        pixel_values = image_grid_thw = pixel_values_videos = video_grid_thw = None
        if ref_blocks:
            input_ids, text_token_tags, pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw = self.preprocess_ref_blocks(pipe, prompt, ref_blocks)
        elif keyframes:
            keyframes = [img.convert("RGB").resize((width, height), Image.LANCZOS) for img in keyframes]
            pixel_values, image_grid_thw, image_counts = image_token_counts(pipe.processor, keyframes)
            input_ids, text_token_tags = presentation_fl2va(pipe.tokenizer, prompt, image_counts)
        else:
            input_ids, text_token_tags = presentation_t2va(pipe.tokenizer, prompt)

        ids = input_ids.unsqueeze(0).to(pipe.device)
        kwargs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        if pixel_values is not None:
            kwargs["pixel_values"] = pixel_values.to(pipe.device, pipe.torch_dtype)
            kwargs["image_grid_thw"] = image_grid_thw.to(pipe.device, torch.long)
        if pixel_values_videos is not None:
            kwargs["pixel_values_videos"] = pixel_values_videos.to(pipe.device, pipe.torch_dtype)
            kwargs["video_grid_thw"] = video_grid_thw.to(pipe.device, torch.long)
        hidden = pipe.text_encoder(**kwargs)

        if text_embedding is not None:
            hidden = torch.concat([text_embedding.to(hidden.device, hidden.dtype), hidden], dim=0)
            extra_tags = torch.ones((text_embedding.shape[0],), device=text_token_tags.device, dtype=text_token_tags.dtype)
            text_token_tags = torch.concat([extra_tags, text_token_tags], dim=0)

        return {"prompt_embeds": hidden.to(pipe.device, pipe.torch_dtype), "text_token_tags": text_token_tags.view(-1).to(pipe.device, torch.long)}


class MiniMaxH3Unit_InputVideoEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_video",),
            output_params=("video_latents", "input_latents"),
            onload_model_names=("video_vae",)
        )

    def process(self, pipe: MiniMaxH3Pipeline, input_video):
        if input_video is None or not pipe.scheduler.training:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        frames_tensor = pipe.preprocess_video(input_video, torch_dtype=torch.float32, min_value=0, device=pipe.device)
        latents = pipe.video_vae.encode_video(frames_tensor, dtype=pipe.torch_dtype).to(pipe.torch_dtype)
        return {"video_latents": latents, "input_latents": latents}


class MiniMaxH3Unit_InputAudioEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_audio",),
            output_params=("audio_latents", "audio_input_latents"),
            onload_model_names=("audio_vae",)
        )

    def process(self, pipe: MiniMaxH3Pipeline, input_audio):
        if input_audio is None or not pipe.scheduler.training:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        waveform, sample_rate = input_audio
        waveform = waveform.squeeze(0) if waveform.dim() == 3 else waveform
        assert waveform.dim() == 2, "waveform must be in shape (C, T)"
        waveform = resample_waveform(convert_to_stereo(waveform).float(), sample_rate, pipe.audio_vae.sample_rate) # [C, T]
        latents = pipe.audio_vae.encode_audio(waveform[:2].to(pipe.device), dtype=pipe.torch_dtype).to(pipe.torch_dtype)  # [C, 32, T]
        return {"audio_latents": latents, "audio_input_latents": latents}


class MiniMaxH3Unit_VideoLatentContinuationEmbedder(PipelineUnit):
    """Apply an exported video-latent tail as a fixed Retake prefix."""

    def __init__(self):
        super().__init__(
            input_params=("continuation_video_latents", "video_latents", "num_frames", "seed", "rand_device"),
            output_params=("input_latents_video", "denoise_mask_video", "motion_anchor_latents_video", "motion_anchor_noise_video", "motion_anchor_weights_video", "motion_anchor_start_video"),
        )

    def process(self, pipe: MiniMaxH3Pipeline, continuation_video_latents, video_latents, num_frames, seed, rand_device):
        if continuation_video_latents is None:
            return {}
        input_latents, mask = apply_video_latent_continuation(
            continuation_video_latents, video_latents, resolved_video_frames=num_frames
        )
        output = {
            "input_latents_video": input_latents,
            "denoise_mask_video": mask.view(1, 1, -1, 1, 1).expand(-1, -1, -1, *video_latents.shape[-2:]),
        }
        motion = video_motion_anchor(
            continuation_video_latents, video_latents, resolved_video_frames=num_frames,
        )
        if motion is not None:
            motion_latents, motion_weights = motion
            prefix_steps = continuation_video_latents.latents.shape[2]
            start = prefix_steps
            output.update(
                motion_anchor_latents_video=motion_latents,
                motion_anchor_noise_video=video_latents[:, :, start:start + motion_latents.shape[2]].detach().clone(),
                motion_anchor_weights_video=motion_weights,
                motion_anchor_start_video=start,
            )
        return output


class MiniMaxH3Unit_AudioLatentContinuationEmbedder(PipelineUnit):
    """Apply an exported audio-latent tail as a fixed Retake prefix."""

    def __init__(self):
        super().__init__(
            input_params=("continuation_audio_latents", "audio_latents", "num_frames", "seed", "rand_device"),
            output_params=("input_latents_audio", "denoise_mask_audio"),
        )

    def process(self, pipe: MiniMaxH3Pipeline, continuation_audio_latents, audio_latents, num_frames, seed, rand_device):
        if continuation_audio_latents is None:
            return {}
        input_latents, mask = apply_audio_latent_continuation(
            continuation_audio_latents, audio_latents, resolved_video_frames=num_frames
        )
        output = {
            "input_latents_audio": input_latents,
            "denoise_mask_audio": mask.view(1, 1, -1).expand(audio_latents.shape[0], -1, -1),
        }
        return output


class MiniMaxH3Unit_VideoRetakeEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("retake_video", "frame_regions_to_retake", "video_latents", "height", "width", "num_frames", "tiled", "tile_size", "tile_overlap"),
            output_params=("input_latents_video", "denoise_mask_video"),
            onload_model_names=("video_vae",)
        )

    def process(self, pipe: MiniMaxH3Pipeline, retake_video, frame_regions_to_retake, video_latents, height, width, num_frames, tiled, tile_size, tile_overlap):
        if retake_video is None:
            return {}
        assert len(retake_video) > 0, "retake_video must contain at least one frame"
        pipe.load_models_to_device(self.onload_model_names)
        frames = [f.convert("RGB").resize((width, height), Image.LANCZOS) for f in retake_video[:num_frames]]
        padded_frames = num_frames - len(frames)
        frames += [frames[-1]] * padded_frames
        frames_tensor = pipe.preprocess_video(frames, torch_dtype=torch.float32, min_value=0, device=pipe.device)
        latents = pipe.video_vae.encode_video(frames_tensor, dtype=pipe.torch_dtype, tiled=tiled, tile_size=tile_size, tile_overlap=tile_overlap).to(device=pipe.device, dtype=pipe.torch_dtype)
        assert latents.shape == video_latents.shape, f"retake_video latents {tuple(latents.shape)} do not match the target shape {tuple(video_latents.shape)}"

        regions = list(frame_regions_to_retake or [])
        if padded_frames > 0:
            regions.append((num_frames - padded_frames, num_frames))
        clip_frames, latents_per_clip = pipe.video_vae.clip_length, pipe.video_vae.tokens_chunk_size
        mask = torch.zeros(latents.shape[2], dtype=pipe.torch_dtype, device=pipe.device)  # 1 = regenerate
        for start, end in regions:
            if end > start:
                mask[max(0, math.floor(start / clip_frames)) * latents_per_clip: math.ceil(end / clip_frames) * latents_per_clip] = 1
        return {"input_latents_video": latents, "denoise_mask_video": mask.view(1, 1, -1, 1, 1).expand(-1, -1, -1, *latents.shape[-2:])}


class MiniMaxH3Unit_AudioRetakeEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("retake_audio", "seconds_regions_to_retake", "audio_latents"),
            output_params=("input_latents_audio", "denoise_mask_audio"),
            onload_model_names=("audio_vae",)
        )

    def process(self, pipe: MiniMaxH3Pipeline, retake_audio, seconds_regions_to_retake, audio_latents):
        if retake_audio is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        waveform, sample_rate = retake_audio
        waveform = waveform.squeeze(0) if waveform.dim() == 3 else waveform
        assert waveform.dim() == 2, "waveform must be in shape (C, T)"
        waveform = resample_waveform(convert_to_stereo(waveform).float(), sample_rate, pipe.audio_vae.sample_rate)
        latents = pipe.audio_vae.encode_audio(waveform[:2].to(pipe.device), dtype=pipe.torch_dtype).to(pipe.torch_dtype)  # [C, 32, T]
        audio_latent_t = audio_latents.shape[-1]
        padded_latents = 0
        if latents.shape[-1] > audio_latent_t:
            latents = latents[..., :audio_latent_t]
        elif latents.shape[-1] < audio_latent_t:
            padded_latents = audio_latent_t - latents.shape[-1]
            latents = torch.cat([latents, latents[..., -1:].repeat(1, 1, padded_latents)], dim=-1)

        latent_fps = pipe.audio_vae.sample_rate / pipe.audio_vae.hop_length
        mask = torch.zeros(audio_latent_t, dtype=pipe.torch_dtype, device=pipe.device)  # 1 = regenerate
        for start, end in seconds_regions_to_retake or []:
            if end > start:
                # Continuation boundaries are derived from video time and may
                # lie between 40 Hz audio-VAE ticks.  Use one deterministic
                # nearest-tick rule for both the retained prefix and suffix.
                mask[max(0, round(start * latent_fps)): round(end * latent_fps)] = 1
        if padded_latents > 0:
            mask[audio_latent_t - padded_latents:] = 1
        return {"input_latents_audio": latents, "denoise_mask_audio": mask.view(1, 1, -1).expand(latents.shape[0], -1, -1)}


class MiniMaxH3Unit_KeyframeEncoder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("keyframes", "keyframe_indices", "video_latents", "rand_device", "seed", "height", "width"),
            output_params=("keyframe_cond_anchor",),
            onload_model_names=("video_vae",)
        )

    def process(self, pipe: MiniMaxH3Pipeline, keyframes, keyframe_indices, video_latents, rand_device, seed, height, width):
        if keyframes is None:
            return {}
        assert keyframe_indices is not None and len(keyframes) == len(keyframe_indices), "keyframe_indices must be provided when keyframes is not None"
        assert all(idx in (0, -1) for idx in keyframe_indices), "keyframe_indices must be within the range of keyframes (0 or -1)"
        pipe.load_models_to_device(self.onload_model_names)
        # Encode keyframes
        all_cond_rows = []
        for img in keyframes:
            img_tensor = pipe.preprocess_image(img.convert("RGB").resize((width, height), Image.LANCZOS), torch_dtype=torch.float32, min_value=0)
            z_norm = pipe.video_vae.encode_video(img_tensor, dtype=pipe.torch_dtype, process_image=True)  # [1,24,1,H',W']
            rows = patchify_video(z_norm)
            all_cond_rows.append(rows)
        clean_cond_rows = torch.cat(all_cond_rows, dim=0).to(device=pipe.device, dtype=pipe.torch_dtype)
        if pipe.imgvid_cond_noise_aug == 1.0:
            keyframe_cond_anchor = clean_cond_rows
        else:
            video_latent_t, latent_h, latent_w = (int(x) for x in video_latents.shape[2:])
            ts = torch.tensor(pipe.imgvid_cond_noise_aug, dtype=pipe.torch_dtype, device=pipe.device)
            noise = pipe.generate_noise((1, 24, video_latent_t + len(keyframes), latent_h, latent_w), seed, rand_device, pipe.torch_dtype)[:,:,:1]
            noise_rows = patchify_video(noise).to(device=pipe.device, dtype=pipe.torch_dtype)
            frame_rows = (latent_h // 2) * (latent_w // 2)
            parts = [
                ts * clean_cond_rows[i * frame_rows:(i + 1) * frame_rows] + (1.0 - ts) * noise_rows
                for i in range(len(keyframes))
            ]
            keyframe_cond_anchor = torch.cat(parts, dim=0)
        return {"keyframe_cond_anchor": keyframe_cond_anchor.contiguous()}


class MiniMaxH3Unit_ReferenceEncoder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("references", "seed", "video_latents", "num_frames", "ref_image_short_edge", "ref_video_short_edge", "ref_video_max_pixels"),
            output_params=("ref_blocks", "ref_visual_anchor", "ref_audio_anchor"),
            onload_model_names=("video_vae", "audio_vae")
        )

    @staticmethod
    def _nearest_multiple(value: float, multiple: int) -> int:
        return max(multiple, int(round(float(value) / multiple)) * multiple)

    def _resolve_reference_image_shape(self, pipe: MiniMaxH3Pipeline, width: int, height: int, short_edge: int):
        scale = short_edge * 1.0 / min(width, height)
        return self._nearest_multiple(width * scale, pipe.width_division_factor), self._nearest_multiple(height * scale, pipe.height_division_factor)

    def _resolve_reference_video_shape(self, pipe: MiniMaxH3Pipeline, width: int, height: int, short_edge: int, max_pixels: int):
        scale = min(short_edge * 1.0 / min(width, height), float(np.sqrt(max_pixels / (width * height))))
        return self._nearest_multiple(width * scale, pipe.width_division_factor), self._nearest_multiple(height * scale, pipe.height_division_factor)

    def _trim_reference_video_length(self, pipe: MiniMaxH3Pipeline, frame_count: int) -> int:
        frame_count = int(frame_count)
        factor, remainder = pipe.time_division_factor, pipe.time_division_remainder
        chunks = max(1, (frame_count - remainder) // factor)
        used = chunks * factor + remainder
        assert used <= frame_count, f"cannot trim {frame_count} reference video frames down to a valid " f"length: at least {used} frames are required"
        return used

    def _encode_image_ref(self, pipe, img: Image.Image, short_edge: int):
        target_w, target_h = self._resolve_reference_image_shape(pipe, *img.size, short_edge)
        img = img.convert("RGB").resize((target_w, target_h), Image.LANCZOS)
        img_tensor = pipe.preprocess_image(img, torch_dtype=torch.float32, min_value=0)
        z = pipe.video_vae.encode_video(img_tensor.to(pipe.device), dtype=pipe.torch_dtype, process_image=True,)  # [1,24,1,H',W']
        return patchify_video(z), z.shape[-2], z.shape[-1], img

    def _encode_video_ref(self, pipe, frames, target_frame_count: int, short_edge: int, max_pixels: int):
        target_w, target_h = self._resolve_reference_video_shape(pipe, *frames[0].size, short_edge, max_pixels)
        frames = [f.resize((target_w, target_h), Image.LANCZOS).convert("RGB") for f in frames]
        frames = frames[: self._trim_reference_video_length(pipe, min(len(frames), target_frame_count))]
        frames_tensor = pipe.preprocess_video(frames, torch_dtype=torch.float32, min_value=0, device=pipe.device)
        z = pipe.video_vae.encode_video(frames_tensor, dtype=pipe.torch_dtype, process_image=False,)  # [1,24,T',H',W']
        return patchify_video(z), int(z.shape[2]), int(z.shape[3]), int(z.shape[4]), frames

    def _encode_audio_ref(self, pipe, waveform, sample_rate: int):
        waveform = waveform.squeeze(0) if waveform.dim() == 3 else waveform
        assert waveform.dim() == 2, "waveform must be in shape (C, T)"
        waveform = resample_waveform(convert_to_stereo(waveform).float(), sample_rate, pipe.audio_vae.sample_rate) # [C, T]
        latent = pipe.audio_vae.encode_audio(waveform[:2].to(pipe.device), dtype=pipe.torch_dtype)  # [C, 32, T]
        return pack_audio(latent), latent.shape[-1]

    @staticmethod
    def _require(ref, key, kind):
        value = ref.get(key)
        assert value is not None, f"reference type {kind!r} requires field {key!r}"
        return value

    def _build_block(self, pipe, ref, target_frame_count, ref_image_short_edge, ref_video_short_edge, ref_video_max_pixels):
        kind = ref["type"]
        if kind == "image":
            rows, lh, lw, prepared = self._encode_image_ref(pipe, self._require(ref, "image", kind), ref_image_short_edge)
            return {"kind": kind, "visual_clean": rows, "latent_t": 1, "latent_h": lh, "latent_w": lw, "prepared_image": prepared, "ref_audio_t": 0}
        if kind in ("video", "video_audio"):
            if kind == "video" and ref.get("audio") is not None:
                raise ValueError("reference type 'video' is silent; use 'video_audio' to pass a soundtrack")
            frames = self._require(ref, "video", kind)
            rows, lt, lh, lw, prepared = self._encode_video_ref(pipe, frames, target_frame_count, ref_video_short_edge, ref_video_max_pixels)
            block = {"kind": kind, "visual_clean": rows, "latent_t": lt, "latent_h": lh, "latent_w": lw, "prepared_frames": prepared, "ref_audio_t": 0}
            if kind == "video_audio":
                audio_waveform = self._require(ref, "audio", kind)
                audio_sample_rate = int(self._require(ref, "sample_rate", kind))
                audio_rows, ref_audio_t = self._encode_audio_ref(pipe, audio_waveform, audio_sample_rate)
                block["audio_clean"], block["ref_audio_t"] = audio_rows, ref_audio_t
            return block
        if kind == "audio":
            rows, ref_audio_t = self._encode_audio_ref(pipe, self._require(ref, "audio", kind), int(self._require(ref, "sample_rate", kind)))
            return {"kind": kind, "audio_clean": rows, "ref_audio_t": ref_audio_t}

    def process(self, pipe: MiniMaxH3Pipeline, references, seed, video_latents, num_frames, ref_image_short_edge, ref_video_short_edge, ref_video_max_pixels):
        if not references:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        seed = int(seed) if seed is not None else 42
        device = pipe.device

        ref_blocks_out = [self._build_block(pipe, ref, num_frames, ref_image_short_edge, ref_video_short_edge, ref_video_max_pixels) for ref in references]
        visual_blocks = [b for b in ref_blocks_out if "visual_clean" in b]
        imgvid_cond_num_frames = len(visual_blocks)
        visual_parts = []
        for block in visual_blocks:
            clean = block.pop("visual_clean").to(device=device, dtype=pipe.torch_dtype)
            if pipe.imgvid_cond_noise_aug == 1.0:
                anchor = clean
            else:
                full_t = video_latents.shape[2] + imgvid_cond_num_frames
                noise_shape = (1, 24, full_t, int(block["latent_h"]), int(block["latent_w"]))
                noise = pipe.generate_noise(noise_shape, seed=seed, rand_device="cpu", rand_torch_dtype=pipe.torch_dtype, device="cpu", torch_dtype=pipe.torch_dtype)
                noise_rows = patchify_video(noise[:, :, : int(block["latent_t"])]).to(device=device, dtype=pipe.torch_dtype)
                ts = torch.tensor(pipe.imgvid_cond_noise_aug, dtype=pipe.torch_dtype, device=device)
                anchor = ts * clean + (1.0 - ts) * noise_rows
            visual_parts.append(anchor)

        audio_parts = []
        for block in ref_blocks_out:
            if "audio_clean" not in block or int(block["ref_audio_t"]) <= 0:
                block.pop("audio_clean", None)
                continue
            clean = block.pop("audio_clean").to(device=device, dtype=pipe.torch_dtype)
            if pipe.audio_cond_noise_aug == 1.0:
                anchor = clean
            else:
                noise = pipe.generate_noise(clean.shape, seed=seed + 1, rand_device="cpu", rand_torch_dtype=pipe.torch_dtype, device=device, torch_dtype=pipe.torch_dtype)
                ts = torch.tensor(pipe.audio_cond_noise_aug, dtype=pipe.torch_dtype, device=device)
                anchor = ts * clean + (1.0 - ts) * noise
            audio_parts.append(anchor)

        return {
            "ref_blocks": ref_blocks_out,
            "ref_visual_anchor": torch.cat(visual_parts, dim=0) if visual_parts else None,
            "ref_audio_anchor": torch.cat(audio_parts, dim=0) if audio_parts else None,
        }


class MiniMaxH3Unit_PackedSequenceBuilder(PipelineUnit):
    _INTERP = 32
    _T_GROUP = 5
    _FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
    _FRAME_RESCALE = 5.0 / 3.0
    _SEQ_ALIGN = 64

    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt_embeds": "prompt_embeds", "text_token_tags": "text_token_tags"},
            input_params_nega={"prompt_embeds": "prompt_embeds", "text_token_tags": "text_token_tags"},
            input_params=("video_latents", "audio_latents", "memory_latents", "keyframe_cond_anchor", "keyframe_indices", "ref_blocks", "video_source_indices", "audio_source_indices", "global_video_latent_length", "global_audio_latent_length", "global_temporal_position_origin"),
            output_params=("packed",)
        )

    @staticmethod
    def _to_device(packed: dict, device) -> dict:
        return {k: v.to(device) if torch.is_tensor(v) else v for k, v in packed.items()}

    def _aligned_seq_len(self, used: int) -> int:
        return ((used + self._SEQ_ALIGN - 1) // self._SEQ_ALIGN) * self._SEQ_ALIGN

    def _axis_from_sqrt_area(self, dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
        ratio = dim / sqrt_area
        left = (1.0 - ratio) * 0.5
        right = left + ratio
        grid = np.linspace(left, right, dim // patch, endpoint=False) * self._INTERP
        return torch.from_numpy(grid).to(torch.float64)

    def _video_t_grid(self, n: int, origin: float) -> torch.Tensor:
        spans = torch.tensor([self._FRAME_RESCALE * self._FRAME_PER_TOKEN[k % self._T_GROUP] for k in range(n)], dtype=torch.float64)
        return origin + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])

    def _temporal_position_span(self, temporal_length: int) -> float:
        spans = np.ones(int(temporal_length), dtype=np.float64) * self._FRAME_RESCALE
        for token_index in range(self._T_GROUP):
            spans[token_index::self._T_GROUP] *= self._FRAME_PER_TOKEN[token_index]
        return float(spans.sum())

    def _video_t_span(self, n: int) -> float:
        return sum(self._FRAME_RESCALE * self._FRAME_PER_TOKEN[k % self._T_GROUP] for k in range(n))

    def _resolve_memory_slots(self, memory_slots, memory_latent_t, frame_rows):
        """Resolve the memory layout into per-slot row counts and temporal leads.

        Every slot ends up as ``[window_origin - S(lead_steps), ...)`` where
        ``S`` is the video temporal span.  ``lead_steps`` is measured in video
        latent steps and must be at least the slot's own length, so that no slot
        ever reaches into the prediction window.  Slots must also be pairwise
        disjoint on the temporal axis, otherwise two conditioning blocks would
        claim the same position and the model could not tell them apart.
        """
        slots = list(memory_slots or [])
        if not slots and memory_latent_t:
            slots = [{"name": "mem0", "kind": "latent", "latent_t": int(memory_latent_t), "lead_steps": None}]
        specs = []
        for index, slot in enumerate(slots):
            kind = slot.get("kind", "latent")
            lead = slot.get("lead_steps")
            if kind == "latent":
                latent_t = slot.get("latent_t")
                if latent_t is None:
                    tensor = slot.get("tensor")
                    if not isinstance(tensor, torch.Tensor) or tensor.dim() != 5:
                        raise ValueError(
                            f"latent memory slot {index} needs latent_t, or a [1,24,t,h,w] tensor to infer it"
                        )
                    latent_t = int(tensor.shape[-3])
                latent_t = int(latent_t)
                if latent_t <= 0:
                    raise ValueError(f"latent memory slot {index} has an empty temporal extent")
                n_steps = latent_t
                rows = latent_t * frame_rows
                # No lead = the slot ends exactly where the window begins, which
                # is the M1-fast STM semantics.
                lead_steps = latent_t if lead is None else int(lead)
            elif kind == "rows":
                rows = int(slot.get("rows") or slot["tensor"].shape[0])
                if lead is None:
                    raise ValueError(f"compressed memory slot {index} must declare lead_steps; its length is not implied by a 17n+5 grid")
                latent_t = None
                n_steps = rows
                lead_steps = int(lead)
            else:
                raise ValueError(f"unknown memory slot kind {kind!r}")
            if lead_steps < 0:
                raise ValueError(f"memory slot {index} lead_steps must be >= 0, got {lead_steps}")
            if lead_steps < n_steps:
                raise ValueError(
                    f"memory slot {index} would reach into the window: lead_steps={lead_steps} < length={n_steps}"
                )
            specs.append({
                "name": str(slot.get("name") or f"mem{index}"), "kind": kind, "latent_t": latent_t,
                "rows": rows, "n_steps": n_steps, "lead_steps": lead_steps,
            })
        spans = sorted(
            (-self._video_t_span(spec["lead_steps"]), -self._video_t_span(spec["lead_steps"]) + self._video_t_span(spec["n_steps"]))
            for spec in specs
        )
        for (_, previous_end), (start, _) in zip(spans, spans[1:]):
            if start < previous_end - 1e-9:
                raise ValueError(
                    "memory slots overlap on the temporal axis; give each slot a distinct lead_steps"
                )
        return specs

    def _frame_grid(self, latent_h: int, latent_w: int, sqrt_area):
        h_grid = self._axis_from_sqrt_area(latent_h, 2, sqrt_area)
        w_grid = self._axis_from_sqrt_area(latent_w, 2, sqrt_area)
        hh, ww = torch.meshgrid(h_grid, w_grid, indexing="ij")
        return torch.stack([hh.reshape(-1), ww.reshape(-1)], dim=-1), w_grid

    def _video_grid(self, latent_t: int, frame: torch.Tensor, origin: float) -> torch.Tensor:
        video_g = torch.empty(latent_t, frame.shape[0], 3, dtype=torch.float64)
        video_g[:, :, 0] = self._video_t_grid(latent_t, origin)[:, None]
        video_g[:, :, 1:] = frame[None]
        return video_g.reshape(-1, 3)

    def _video_grid_from_source_indices(
        self, source_indices: Sequence[int], global_length: int, frame: torch.Tensor, origin: float
    ) -> torch.Tensor:
        if global_length <= 0 or any(int(index) < 0 or int(index) >= global_length for index in source_indices):
            raise ValueError("video source indices must be inside the global latent timeline")
        full = self._video_t_grid(global_length, origin)
        positions = full[torch.tensor(tuple(int(index) for index in source_indices), dtype=torch.long)]
        video_g = torch.empty(len(source_indices), frame.shape[0], 3, dtype=torch.float64)
        video_g[:, :, 0] = positions[:, None]
        video_g[:, :, 1:] = frame[None]
        return video_g.reshape(-1, 3)

    def _audio_w_axis(self, w_grid: torch.Tensor, audio_t: int, audio_rows: int) -> torch.Tensor:
        return torch.cat([torch.full((audio_t,), float(w_grid[0]), dtype=torch.float64),
                          torch.full((audio_rows - audio_t,), float(w_grid[-1]), dtype=torch.float64)])

    def _build_packed_fl2va(
        self, text_len, latent_t, latent_h, latent_w, audio_t, keyframe_indices,
        audio_channel=2, video_source_indices=None, audio_source_indices=None,
        global_video_latent_length=None, global_audio_latent_length=None,
        temporal_origin=None, memory_latent_t=None, memory_slots=None,
    ):
        """fl2va layout: [text | memory_0 .. memory_k | cond | audio | video | pad]."""
        frame_rows = (latent_h // 2) * (latent_w // 2)
        video_rows = latent_t * frame_rows
        # ``memory_latent_t`` is the M1-fast shorthand for a single contiguous
        # latent slot; both forms end up as an ordered slot list.
        slot_specs = self._resolve_memory_slots(memory_slots, memory_latent_t, frame_rows)
        memory_rows = sum(spec["rows"] for spec in slot_specs)
        audio_rows = audio_t * audio_channel
        num_keyframes = len(keyframe_indices)
        cond_rows = num_keyframes * frame_rows
        used = text_len + memory_rows + cond_rows + audio_rows + video_rows
        seq_len = self._aligned_seq_len(used)

        text_sl = slice(0, text_len)
        cursor = text_len
        slot_slices = []
        for spec in slot_specs:
            slot_slices.append(slice(cursor, cursor + spec["rows"]))
            cursor += spec["rows"]
        memory_sl = slice(text_len, text_len + memory_rows)
        cond_sl = slice(cursor, cursor + cond_rows)
        audio_sl = slice(cond_sl.stop, cond_sl.stop + audio_rows)
        video_sl = slice(audio_sl.stop, audio_sl.stop + video_rows)

        # Keep the temporal origin independent from the local prompt length in
        # v3.  Text still occupies [0, text_len), while image/audio positions
        # use the first window's cached origin across all windows.
        origin = float(text_len) if temporal_origin is None else float(temporal_origin)

        # A memory slot is a clean condition sitting *before* the window on the
        # temporal axis, so the window's origin moves back by exactly the
        # deepest slot lead.  Slots then land in what was previously the empty
        # gap between the text island and the window and never consume any of
        # the window's own frame budget.  With no slots (or the legacy
        # ``memory_latent_t == 0``) this reproduces the previous layout bit for
        # bit.
        memory_lead = self._video_t_span(max((spec["lead_steps"] for spec in slot_specs), default=0))
        origin = origin + memory_lead

        # img_pos covers both cond AND video rows, conditions first
        img_pos = torch.cat([torch.arange(cond_sl.start, cond_sl.stop), torch.arange(video_sl.start, video_sl.stop)])
        audio_pos = torch.arange(audio_sl.start, audio_sl.stop)

        g = torch.zeros(seq_len, 3, dtype=torch.float64)
        g[text_sl, 0] = torch.arange(text_len, dtype=torch.float64)

        sqrt_area = np.sqrt(latent_h * latent_w)
        frame, w_grid = self._frame_grid(latent_h, latent_w, sqrt_area)

        # Condition rows: temporal position depends on frame_index
        # A TES sparse window can refer to arbitrary positions in the complete
        # latent timeline.  Its final FL2VA condition must stay at the original
        # global endpoint rather than the endpoint of the gathered subsequence.
        position_length = global_video_latent_length if video_source_indices is not None else latent_t
        temporal_span = self._temporal_position_span(int(position_length))
        for i, idx in enumerate(keyframe_indices):
            sl = slice(i * frame_rows, (i + 1) * frame_rows)
            if idx == 0:
                cond_t = origin
            else:  # idx == -1
                cond_t = origin + temporal_span - self._FRAME_RESCALE
            cond_g = torch.empty(frame_rows, 3, dtype=torch.float64)
            cond_g[:, 0] = cond_t
            cond_g[:, 1:] = frame
            g[cond_sl.start + sl.start: cond_sl.start + sl.stop] = cond_g

        if video_source_indices is not None:
            if len(video_source_indices) != latent_t or global_video_latent_length is None:
                raise ValueError("video_source_indices must match local video latent length and provide global length")
            g[video_sl] = self._video_grid_from_source_indices(
                video_source_indices, int(global_video_latent_length), frame, origin
            )
        else:
            g[video_sl] = self._video_grid(latent_t, frame, origin)
        for spec, slot_sl in zip(slot_specs, slot_slices):
            # Each slot carries its own anchor.  A latent slot gets the ordinary
            # video grid; a compressed slot occupies a virtual temporal axis of
            # ``m`` tokens at its anchor (its tokens have no spatial extent of
            # their own, so they borrow the window's leading patch coordinates
            # and stay distinct from the window through their temporal position).
            anchor = origin - self._video_t_span(spec["lead_steps"])
            if spec["kind"] == "latent":
                g[slot_sl] = self._video_grid(spec["latent_t"], frame, anchor)
            else:
                slot_g = torch.empty(spec["rows"], 3, dtype=torch.float64)
                slot_g[:, 0] = self._video_t_grid(spec["rows"], anchor)
                slot_g[:, 1:] = frame[: spec["rows"]]
                g[slot_sl] = slot_g
        if audio_source_indices is not None:
            if len(audio_source_indices) != audio_t or global_audio_latent_length is None:
                raise ValueError("audio_source_indices must match local audio latent length and provide global length")
            audio_positions = torch.tensor(tuple(int(index) for index in audio_source_indices), dtype=torch.float64)
            g[audio_sl, 0] = (origin + audio_positions).repeat(audio_channel)
        else:
            g[audio_sl, 0] = (origin + torch.arange(audio_t, dtype=torch.float64)).repeat(audio_channel)
        g[audio_sl, 2] = self._audio_w_axis(w_grid, audio_t, audio_rows)

        token_tags = torch.full((seq_len,), -1, dtype=torch.long)
        token_tags[text_sl] = 1
        token_tags[audio_sl] = 2
        token_tags[img_pos] = 0  # both cond and video rows are tagged as video (0)
        # Memory rows carry real video latents, so they take the video modality
        # like the cond rows do.  "Clean, not a prediction target" is expressed
        # by the timestep (see model_fn_minimax_h3), not by the tag.
        token_tags[memory_sl] = 0

        return {
            "img_pos": img_pos, "audio_pos": audio_pos, "text_pos": torch.arange(0, text_len),
            "mem_pos": torch.arange(memory_sl.start, memory_sl.stop),
            "mem_slot_slices": [(sl.start, sl.stop) for sl in slot_slices],
            "mem_slot_names": [spec["name"] for spec in slot_specs],
            "img_position_ids": g[None], "token_tags": token_tags,
            "cu_seqlens": torch.tensor([0, used, seq_len], dtype=torch.int32), "seq_len": seq_len,
        }

    def _build_packed_ref2va(self, text_len, latent_t, latent_h, latent_w, audio_t, ref_blocks, audio_channel=2):
        """ref2va layout: [text | ref_0 | ref_1 | ... | target_audio | target_video | pad]"""
        ph, pw = latent_h // 2, latent_w // 2
        target_frame_rows = ph * pw
        target_video_rows = latent_t * target_frame_rows
        target_audio_rows = audio_t * audio_channel

        block_dims, total_ref_visual_rows, total_ref_audio_rows = [], 0, 0
        for b in ref_blocks:
            kind = b["kind"]
            info = {"kind": kind, "visual_rows": 0, "audio_rows": 0, "ref_audio_t": int(b.get("ref_audio_t", 0))}
            if kind in ("image", "video", "video_audio"):
                lt_r, lh_r, lw_r = int(b["latent_t"]), int(b["latent_h"]), int(b["latent_w"])
                info.update(visual_rows=lt_r * (lh_r // 2) * (lw_r // 2), latent_t=lt_r, latent_h=lh_r, latent_w=lw_r)
                total_ref_visual_rows += info["visual_rows"]
                info["audio_rows"] = info["ref_audio_t"] * audio_channel
                total_ref_audio_rows += info["audio_rows"]
            elif kind == "audio":
                info["audio_rows"] = info["ref_audio_t"] * audio_channel
                total_ref_audio_rows += info["audio_rows"]
            else:
                raise ValueError(f"unknown ref kind: {kind}")
            block_dims.append(info)

        used = text_len + total_ref_visual_rows + total_ref_audio_rows + target_audio_rows + target_video_rows
        seq_len = self._aligned_seq_len(used)

        g = torch.zeros(seq_len, 3, dtype=torch.float64)
        g[0:text_len, 0] = torch.arange(text_len, dtype=torch.float64)
        token_tags = torch.full((seq_len,), -1, dtype=torch.long)
        token_tags[0:text_len] = 1

        # Iterate through ref blocks, placing them contiguously.
        cursor, t_cursor = text_len, float(text_len)
        ref_visual_pos_parts, ref_audio_pos_parts = [], []
        # Target w_grid used for audio W-axis (channel separation)
        target_sqrt_area = float(np.sqrt(latent_h * latent_w))
        target_w_grid = self._axis_from_sqrt_area(latent_w, 2, target_sqrt_area)

        for info in block_dims:
            kind = info["kind"]
            if kind == "image":
                v_rows, lh_r, lw_r = info["visual_rows"], info["latent_h"], info["latent_w"]
                # Own spatial grid
                sqrt_area = float(np.sqrt(lh_r * lw_r))
                frame, w_grid = self._frame_grid(lh_r, lw_r, sqrt_area)
                sl = slice(cursor, cursor + v_rows)
                g[sl, 0] = t_cursor
                g[sl, 1:] = frame
                token_tags[sl] = 0
                ref_visual_pos_parts.append(torch.arange(sl.start, sl.stop))
                cursor += v_rows
                t_cursor += 1.0

            elif kind in ("video", "video_audio"):
                a_rows, v_rows, ref_at = info["audio_rows"], info["visual_rows"], info["ref_audio_t"]
                lt_r, lh_r, lw_r = info["latent_t"], info["latent_h"], info["latent_w"]
                sqrt_area = float(np.sqrt(lh_r * lw_r))
                frame, rv_w_grid = self._frame_grid(lh_r, lw_r, sqrt_area)

                audio_sl = slice(cursor, cursor + a_rows)
                visual_sl = slice(audio_sl.stop, audio_sl.stop + v_rows)

                a_t_grid = t_cursor + torch.arange(ref_at, dtype=torch.float64)
                g[audio_sl, 0] = a_t_grid.repeat(audio_channel)
                if ref_at:
                    # W-axis uses THIS reference video's own grid, not the target's.
                    g[audio_sl, 2] = self._audio_w_axis(rv_w_grid, ref_at, a_rows)
                    token_tags[audio_sl] = 2
                    ref_audio_pos_parts.append(torch.arange(audio_sl.start, audio_sl.stop))

                g[visual_sl] = self._video_grid(lt_r, frame, t_cursor)
                token_tags[visual_sl] = 0
                ref_visual_pos_parts.append(torch.arange(visual_sl.start, visual_sl.stop))

                cursor = visual_sl.stop
                t_cursor += max(float(ref_at), self._video_t_span(lt_r))

            elif kind == "audio":
                a_rows, ref_at = info["audio_rows"], info["ref_audio_t"]
                sl = slice(cursor, cursor + a_rows)
                a_t_grid = t_cursor + torch.arange(ref_at, dtype=torch.float64)
                g[sl, 0] = a_t_grid.repeat(audio_channel)
                if ref_at:
                    g[sl, 2] = self._audio_w_axis(target_w_grid, ref_at, a_rows)
                    token_tags[sl] = 2
                    ref_audio_pos_parts.append(torch.arange(sl.start, sl.stop))
                cursor += a_rows
                t_cursor += float(ref_at)

        # Target audio + target video after ref blocks
        target_audio_sl = slice(cursor, cursor + target_audio_rows)
        target_video_sl = slice(target_audio_sl.stop, target_audio_sl.stop + target_video_rows)

        # Target spatial grid (own)
        frame_t, w_grid_t = self._frame_grid(latent_h, latent_w, target_sqrt_area)
        g[target_video_sl] = self._video_grid(latent_t, frame_t, t_cursor)

        target_audio_t_grid = t_cursor + torch.arange(audio_t, dtype=torch.float64)
        g[target_audio_sl, 0] = target_audio_t_grid.repeat(audio_channel)
        g[target_audio_sl, 2] = self._audio_w_axis(w_grid_t, audio_t, target_audio_rows)

        # Build combined img_pos / audio_pos
        target_video_pos = torch.arange(target_video_sl.start, target_video_sl.stop)
        target_audio_pos = torch.arange(target_audio_sl.start, target_audio_sl.stop)

        if ref_visual_pos_parts:
            img_pos = torch.cat(ref_visual_pos_parts + [target_video_pos])
        else:
            img_pos = target_video_pos
        if ref_audio_pos_parts:
            audio_pos = torch.cat(ref_audio_pos_parts + [target_audio_pos])
        else:
            audio_pos = target_audio_pos

        token_tags[img_pos] = 0
        token_tags[target_audio_sl] = 2

        text_pos = torch.arange(0, text_len)
        cu = torch.tensor([0, used, seq_len], dtype=torch.int32)

        return {
            "img_pos": img_pos, "audio_pos": audio_pos, "text_pos": text_pos,
            "img_position_ids": g[None], "token_tags": token_tags, "cu_seqlens": cu,
            "seq_len": seq_len,
        }

    def process(
        self, pipe: MiniMaxH3Pipeline, prompt_embeds, video_latents, audio_latents,
        text_token_tags=None, keyframe_cond_anchor=None, keyframe_indices=None, ref_blocks=None,
        video_source_indices=None, audio_source_indices=None,
        global_video_latent_length=None, global_audio_latent_length=None,
        global_temporal_position_origin=None, memory_latents=None,
    ):
        text_len = prompt_embeds.shape[0]
        video_latent_t, latent_h, latent_w = video_latents.shape[2:]
        audio_latent_t = audio_latents.shape[-1]
        # Memory may arrive as a single latent block (M1-fast STM) or as an
        # ordered slot list (STM + anchored LTM).  Both normalise to the same
        # slot description, which is what the packer and the DiT both read.
        memory_slots = normalize_memory_slots(memory_latents, latent_h=latent_h, latent_w=latent_w)
        if memory_slots and ref_blocks is not None:
            raise ValueError("memory_latents is not supported together with ref_blocks")
        if ref_blocks is not None:
            if video_source_indices is not None or audio_source_indices is not None:
                raise ValueError("TES source indices are currently supported only for FL2VA packed sequences")
            packed = self._build_packed_ref2va(text_len, video_latent_t, latent_h, latent_w, audio_latent_t, ref_blocks)
        else:
            temporal_origin = global_temporal_position_origin
            if video_source_indices is not None or audio_source_indices is not None:
                if temporal_origin is None:
                    temporal_origin = getattr(pipe, "_h3_v3_temporal_origin", None)
                if temporal_origin is None and video_source_indices and int(video_source_indices[0]) == 0:
                    temporal_origin = float(text_len)
                    pipe._h3_v3_temporal_origin = temporal_origin
            packed = self._build_packed_fl2va(
                text_len, video_latent_t, latent_h, latent_w, audio_latent_t,
                keyframe_indices if keyframe_cond_anchor is not None else [],
                video_source_indices=video_source_indices,
                audio_source_indices=audio_source_indices,
                global_video_latent_length=global_video_latent_length,
                global_audio_latent_length=global_audio_latent_length,
                temporal_origin=temporal_origin,
                memory_slots=memory_slots,
            )

        packed["token_tags"][packed["text_pos"]] = text_token_tags.cpu()
        if pipe.device == "mps":
            packed["img_position_ids"] = packed["img_position_ids"].to(torch.float32)
        return {"packed": self._to_device(packed, pipe.device)}


def model_fn_minimax_h3(
    dit,
    video_latents,
    audio_latents,
    packed,
    prompt_embeds,
    timestep_video,
    timestep_audio,
    keyframe_cond_anchor=None,
    ref_visual_anchor=None,
    ref_audio_anchor=None,
    input_latents_video=None,
    denoise_mask_video=None,
    input_latents_audio=None,
    denoise_mask_audio=None,
    memory_latents=None,
    imgvid_cond_noise_aug=0.999,
    audio_cond_noise_aug=1.0,
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
    **kwargs,
):
    t_video = 1.0 - float(timestep_video) / 1000
    t_audio = 1.0 - float(timestep_audio) / 1000

    dtype, device = video_latents.dtype, video_latents.device
    f, h, w = video_latents.shape[2:]
    audio_channel, audio_t = audio_latents.shape[0], audio_latents.shape[-1]

    if input_latents_video is not None:
        video_latents = video_latents * denoise_mask_video + input_latents_video * (1.0 - denoise_mask_video)
        denoise_mask_video = patchify_video(denoise_mask_video)[:, 0]
    if input_latents_audio is not None:
        audio_latents = audio_latents * denoise_mask_audio + input_latents_audio * (1.0 - denoise_mask_audio)
        denoise_mask_audio = pack_audio(denoise_mask_audio)[:, 0]
    video_rows = patchify_video(video_latents)
    audio_rows = pack_audio(audio_latents)

    img_pos, audio_pos, text_pos = packed["img_pos"], packed["audio_pos"], packed["text_pos"]
    mem_pos = packed.get("mem_pos")
    cu, seq_len, text_len = packed["cu_seqlens"], packed["seq_len"], text_pos.shape[0]
    # A memory block is a clean visual condition, not a prediction target: it is
    # packed as real video rows at their own sequence positions and its rows are
    # deliberately *not* part of ``img_pos``, so the loss never sees them.
    memory_rows = None
    if memory_latents is not None:
        if mem_pos is None or mem_pos.numel() == 0:
            raise ValueError("memory_latents was provided but the packed sequence has no memory rows")
        # Slots are concatenated in the packer's order, which is the only thing
        # the DiT needs: every memory row goes through the same ``video_patch_proj``
        # reading path as the window, so a compressed slot with ``m`` rows is
        # read exactly like a latent slot with ``m`` patch rows.
        memory_rows = torch.cat(memory_slot_rows(normalize_memory_slots(memory_latents)), dim=0)
        if memory_rows.shape[0] != int(mem_pos.numel()):
            raise ValueError(
                f"memory block has {memory_rows.shape[0]} patch rows but the packed sequence "
                f"reserved {int(mem_pos.numel())}"
            )
    # Video Sequence
    cond_anchor = ref_visual_anchor if ref_visual_anchor is not None else keyframe_cond_anchor
    cond_rows_count = 0 if cond_anchor is None else cond_anchor.shape[0]
    x = torch.zeros(1, seq_len, 96, dtype=dtype, device=device)
    x[0].index_copy_(0, img_pos[cond_rows_count:], video_rows)
    if cond_anchor is not None:
        x[0].index_copy_(0, img_pos[:cond_rows_count], cond_anchor)
    if memory_rows is not None:
        # A slot only has to be a latent block of the right geometry; where it was
        # stored before it got here is not the packer's business.  The swap arm
        # replays donor slots loaded from CPU, so align the device here rather
        # than making every slot producer responsible for it.
        x[0].index_copy_(0, mem_pos, memory_rows.to(device=device, dtype=dtype))
    # Audio Sequence
    ref_audio_rows_count = 0 if ref_audio_anchor is None else ref_audio_anchor.shape[0]
    expected_audio_rows = int(audio_pos.numel()) - ref_audio_rows_count
    if audio_rows.shape[0] != expected_audio_rows:
        raise ValueError(
            f"packed sequence expects {expected_audio_rows} target audio rows "
            f"({expected_audio_rows // audio_channel} steps) but the audio latent has "
            f"{audio_rows.shape[0]} rows ({audio_rows.shape[0] // audio_channel} steps); "
            "the audio latent must be conformed to the frame-derived audio raster "
            "(round(window_frames / 24 * 40) steps)"
        )
    audio_x = torch.zeros(1, seq_len, 32, dtype=dtype, device=device)
    audio_x[0].index_copy_(0, audio_pos[ref_audio_rows_count:], audio_rows)
    if ref_audio_anchor is not None:
        audio_x[0].index_copy_(0, audio_pos[:ref_audio_rows_count], ref_audio_anchor)

    timesteps = torch.full((seq_len,), t_video, dtype=torch.float32, device=device)
    timesteps[audio_pos] = t_audio

    if input_latents_video is not None:
        timesteps[img_pos[cond_rows_count:][denoise_mask_video == 0]] = 1.0
    if input_latents_audio is not None:
        timesteps[audio_pos[ref_audio_rows_count:][denoise_mask_audio == 0]] = 1.0

    timesteps[img_pos[:cond_rows_count]] = max(t_video, imgvid_cond_noise_aug)
    timesteps[audio_pos[:ref_audio_rows_count]] = max(t_audio, audio_cond_noise_aug)
    # Memory rows are fed at t=1.0 (fully clean), exactly like the clean
    # continuation prefix, so the DiT treats them as conditioning rather than as
    # something to denoise.
    if mem_pos is not None and mem_pos.numel():
        timesteps[mem_pos] = 1.0
    unique_timesteps, inverse_indices = torch.unique(timesteps, sorted=True, return_inverse=True)

    refiner_cu = torch.tensor([0, text_len, text_len], dtype=torch.int32, device=device)
    cp_rank = int(kwargs.get("cp_rank", 0))
    cp_world_size = int(kwargs.get("cp_world_size", 1))
    cp_group = kwargs.get("cp_group")
    if cp_world_size > 1 and cp_group is not None:
        spans = split_sequence_indices(seq_len, cp_world_size)
        local_start, local_end = spans[cp_rank]
        local_len = local_end - local_start

        img_pos_arr = img_pos.view(-1).to(torch.long)
        audio_pos_arr = audio_pos.view(-1).to(torch.long)
        text_pos_arr = text_pos.view(-1).to(torch.long)
        img_mask = (img_pos_arr >= local_start) & (img_pos_arr < local_end)
        audio_mask = (audio_pos_arr >= local_start) & (
            audio_pos_arr < local_end
        )
        text_mask = (text_pos_arr >= local_start) & (
            text_pos_arr < local_end
        )
        local_img_pos = img_pos_arr[img_mask] - local_start
        local_audio_pos = audio_pos_arr[audio_mask] - local_start
        local_text_pos = text_pos_arr[text_mask] - local_start

        mem_mask = None
        local_mem_pos = None
        if mem_pos is not None and mem_pos.numel():
            mem_pos_arr = mem_pos.view(-1).to(torch.long)
            mem_mask = (mem_pos_arr >= local_start) & (mem_pos_arr < local_end)
            local_mem_pos = mem_pos_arr[mem_mask] - local_start

        source_video = (
            torch.cat([cond_anchor, video_rows], dim=0)
            if cond_anchor is not None
            else video_rows
        )
        x_local = torch.zeros(
            1, local_len, 96, dtype=dtype, device=device
        )
        x_local[0].index_copy_(0, local_img_pos, source_video[img_mask])
        if mem_mask is not None:
            x_local[0].index_copy_(
                0, local_mem_pos, memory_rows[mem_mask].to(device=device, dtype=dtype)
            )

        source_audio = (
            torch.cat([ref_audio_anchor, audio_rows], dim=0)
            if ref_audio_anchor is not None
            else audio_rows
        )
        audio_x_local = torch.zeros(
            1, local_len, 32, dtype=dtype, device=device
        )
        audio_x_local[0].index_copy_(
            0, local_audio_pos, source_audio[audio_mask]
        )

        v_video_rows, v_audio_rows = dit(
            x=x_local,
            audio_x=audio_x_local,
            img_position_ids=packed["img_position_ids"][
                :, local_start:local_end
            ],
            unique_timesteps=unique_timesteps,
            inverse_indices=inverse_indices[local_start:local_end],
            token_tags=packed["token_tags"][local_start:local_end],
            prompt_embeds=prompt_embeds,
            img_pos_info={"position_ids": local_img_pos},
            audio_pos_info={"position_ids": local_audio_pos},
            text_pos_info={"position_ids": local_text_pos},
            img_pos_for_infer_output_info={"position_ids": local_img_pos},
            mem_pos_info=None if local_mem_pos is None else {"position_ids": local_mem_pos},
            packed_seq_params={
                "cu_seqlens_q": cu,
                "max_seqlen_q": int(cu[1]),
                "seq_len": seq_len,
            },
            refiner_packed_seq_params={
                "cu_seqlens_q": refiner_cu,
                "max_seqlen_q": text_len,
            },
            update_mask=None,
            skip_mask_out_condition=True,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            cp_rank=cp_rank,
            cp_world_size=cp_world_size,
            cp_group=cp_group,
            text_embed_select=text_pos_arr[text_mask],
        )

        local_img_indices = torch.nonzero(img_mask).squeeze(1)
        local_cond_count = int(
            (local_img_indices < cond_rows_count).sum()
        )
        v_video_rows = v_video_rows[local_cond_count:]
        local_audio_indices = torch.nonzero(audio_mask).squeeze(1)
        local_ref_count = int(
            (local_audio_indices < ref_audio_rows_count).sum()
        )
        v_audio_rows = v_audio_rows[local_ref_count:]
        return -v_video_rows, -v_audio_rows

    v_video_rows, v_audio_rows = dit(
        x=x,
        audio_x=audio_x,
        img_position_ids=packed["img_position_ids"],
        unique_timesteps=unique_timesteps,
        inverse_indices=inverse_indices,
        token_tags=packed["token_tags"],
        prompt_embeds=prompt_embeds,
        img_pos_info={"position_ids": img_pos},
        audio_pos_info={"position_ids": audio_pos},
        text_pos_info={"position_ids": text_pos},
        img_pos_for_infer_output_info={"position_ids": img_pos},
        mem_pos_info=None if mem_pos is None else {"position_ids": mem_pos},
        packed_seq_params={"cu_seqlens_q": cu, "max_seqlen_q": int(cu[1])},
        refiner_packed_seq_params={"cu_seqlens_q": refiner_cu, "max_seqlen_q": text_len},
        update_mask=None,
        skip_mask_out_condition=True,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        cp_rank=cp_rank,
        cp_world_size=cp_world_size,
    )

    v_video_rows = v_video_rows[cond_rows_count:]
    v_audio_rows = v_audio_rows[ref_audio_rows_count:]

    v_video = unpatchify_video(v_video_rows, f, h, w)
    v_audio = unpack_audio(v_audio_rows, audio_channel, audio_t)
    return -v_video, -v_audio
