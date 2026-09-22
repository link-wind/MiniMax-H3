import argparse
import glob
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

from diffsynth.core.context_parallel import create_cp_process_group
from diffsynth.models.minimax_h3_dit import (
    unpack_audio,
    unpatchify_video,
)
from diffsynth.pipelines.minimax_h3_audio_video import (
    MiniMaxH3Pipeline,
    ModelConfig,
    model_fn_minimax_h3,
)
from diffsynth.pipelines.minimax_h3_continuation import (
    H3ContinuationConfig,
    H3ContinuationRunner,
    cp_window_metadata,
    is_designated_cp_writer,
    is_masked_av_context_frames,
    load_h3_segment_plan,
    validate_cp_window_metadata,
    write_continuation_evaluation,
)
from diffsynth.pipelines.h3_appearance_memory import H3AppearanceMemoryConfig
from diffsynth.utils.data.audio_video import write_video_audio


# Keep every FL2VA component on the model tree used by the reviewed 8-step
# masked-continuation reference videos.  Do not mix a DiT from one snapshot
# with encoders/VAEs from another snapshot.
FL2VA_BASE = "/gemini/platform/public/aigc/human_guozz2/code/songqy/diffsynth_h3/models/MiniMax/MiniMax-H3/FL2VA"


def _cpu_offload_vram_config():
    return {
        "offload_dtype": torch.bfloat16,
        "offload_device": "cpu",
        "onload_dtype": torch.bfloat16,
        "onload_device": "cpu",
        "preparing_dtype": torch.bfloat16,
        "preparing_device": "cuda",
        "computation_dtype": torch.bfloat16,
        "computation_device": "cuda",
    }


def _gather_rows(local_rows, cp_rank, cp_world_size, cp_group):
    if cp_world_size <= 1:
        return local_rows

    local_count = torch.tensor(
        [local_rows.shape[0]], dtype=torch.long, device=local_rows.device
    )
    counts = [
        torch.zeros(1, dtype=torch.long, device=local_rows.device)
        for _ in range(cp_world_size)
    ]
    dist.all_gather(counts, local_count, group=cp_group)
    counts = [int(c.item()) for c in counts]

    max_len = max(counts)
    padded = local_rows
    if max_len > local_rows.shape[0]:
        padded = F.pad(local_rows, (0, 0, 0, max_len - local_rows.shape[0]))

    gathered = [
        torch.empty_like(padded) for _ in range(cp_world_size)
    ]
    dist.all_gather(gathered, padded, group=cp_group)
    return torch.cat(
        [gathered[i][:counts[i]] for i in range(cp_world_size)], dim=0
    )


def make_cp_model_fn(cp_rank, cp_world_size, cp_group):
    def cp_model_fn(**kwargs):
        kwargs["cp_rank"] = cp_rank
        kwargs["cp_world_size"] = cp_world_size
        kwargs["cp_group"] = cp_group
        local_video_rows, local_audio_rows = model_fn_minimax_h3(**kwargs)

        # With CP disabled, model_fn_minimax_h3 already returns unpacked
        # latent tensors ([B, C, T, H, W] and [C, D, T]).  The gather path
        # below receives packed row tensors from each CP rank and therefore
        # must not be applied to the single-rank return value.  Re-unpacking
        # here silently scrambles the latent layout and decodes as noise.
        if cp_world_size <= 1:
            return local_video_rows, local_audio_rows

        video_latents = kwargs["video_latents"]
        audio_latents = kwargs["audio_latents"]
        f, h, w = (int(x) for x in video_latents.shape[2:])
        audio_channel, audio_t = (
            int(audio_latents.shape[0]),
            int(audio_latents.shape[-1]),
        )

        video_rows = _gather_rows(
            local_video_rows, cp_rank, cp_world_size, cp_group
        )
        audio_rows = _gather_rows(
            local_audio_rows, cp_rank, cp_world_size, cp_group
        )

        video = unpatchify_video(video_rows, f, h, w)
        audio = unpack_audio(audio_rows, audio_channel, audio_t)
        return video, audio

    return cp_model_fn


class _WriterOnlyDecodePipeline:
    """Keep VAE decode on the designated CP writer.

    Every rank still executes the same DiT call and receives identical gathered
    latents. Non-writer ranks skip VAE decode to avoid duplicating the large
    temporal decode peak, while the continuation runner keeps their latent tail
    state for the next window.
    """

    def __init__(self, pipeline, is_writer):
        self.pipeline = pipeline
        self.is_writer = is_writer

    def __getattr__(self, name):
        return getattr(self.pipeline, name)

    def __call__(self, **kwargs):
        if not self.is_writer:
            kwargs["return_latents"] = True
            kwargs["decode_output"] = False
        return self.pipeline(**kwargs)

    def decode_continuation_suffix(self, **kwargs):
        if not self.is_writer:
            return None, None
        return self.pipeline.decode_continuation_suffix(**kwargs)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Optional H3 DiT checkpoint. If omitted, use the original FL2VA "
            "transformer shards under FL2VA_BASE."
        ),
    )
    parser.add_argument(
        "--lora",
        default=None,
        help="Optional acceleration LoRA (for example an 8-step H3 LoRA).",
    )
    parser.add_argument(
        "--lora-scale", type=float, default=1.0,
        help="Scale applied to --lora (default: 1.0).",
    )
    parser.add_argument(
        "--prompt_file",
        default=None,
        help="Path to a text file containing the full 30s prompt.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Inline prompt string. Ignored when --prompt_file is provided.",
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=50
    )
    parser.add_argument(
        "--num_frames", type=int, default=719
    )
    parser.add_argument(
        "--height", type=int, default=480
    )
    parser.add_argument(
        "--width", type=int, default=832
    )
    parser.add_argument(
        "--cp_world_size", type=int, default=4
    )
    parser.add_argument(
        "--output_path",
        default="models/inference/latest_30s.mp4",
    )
    parser.add_argument(
        "--seed", type=int, default=0
    )
    parser.add_argument("--continuation-plan", default=None, help="JSON segment plan; omitted keeps existing single-window inference.")
    parser.add_argument("--continuation-window-frames", type=int, default=243)
    parser.add_argument("--continuation-overlap-frames", type=int, default=34)
    parser.add_argument(
        "--continuation-mode",
        choices=("retake-hard", "latent-handoff", "masked-av-v14"),
        default="retake-hard",
    )
    parser.add_argument("--audio-crossfade-ms", type=float, default=0.0)
    parser.add_argument(
        "--appearance-memory-mode",
        choices=("disabled", "static", "dynamic"),
        default="disabled",
        help="Optional decoded-frame reference bank for later masked-av-v14 windows.",
    )
    parser.add_argument("--appearance-trusted-anchor-frames", type=int, default=2)
    parser.add_argument("--appearance-memory-frames", type=int, default=0)
    parser.add_argument("--appearance-boundary-reference", type=int, choices=(0, 1), default=0)
    parser.add_argument("--appearance-max-visual-references", type=int, default=4)
    parser.add_argument(
        "--output-video-frames", type=int, default=0,
        help="Keep at most this many final video frames; 0 preserves the full assembled timeline.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.continuation_plan:
        if args.continuation_overlap_frames <= 0:
            raise ValueError("--continuation-overlap-frames must be positive")
        if args.continuation_mode == "masked-av-v14" and not is_masked_av_context_frames(
            args.continuation_overlap_frames
        ):
            raise ValueError(
                "masked-av-v14 requires a native Masked-AV overlap (39, 90, 141, ...), "
                f"got {args.continuation_overlap_frames}"
            )
        if not 0 <= args.audio_crossfade_ms <= 200:
            raise ValueError("--audio-crossfade-ms must be between 0 and 200")
        if args.appearance_memory_mode != "disabled" and args.continuation_mode != "masked-av-v14":
            raise ValueError("appearance memory requires --continuation-mode masked-av-v14")
        H3AppearanceMemoryConfig(
            mode=args.appearance_memory_mode,
            trusted_anchor_frames=args.appearance_trusted_anchor_frames,
            memory_frame_budget=args.appearance_memory_frames,
            boundary_reference_frames=bool(args.appearance_boundary_reference),
            max_visual_references=args.appearance_max_visual_references,
        )
    if args.output_video_frames < 0:
        raise ValueError("--output-video-frames must be non-negative")
    if args.checkpoint is None:
        checkpoint_path = sorted(
            glob.glob(f"{FL2VA_BASE}/transformer/model*.safetensors")
        )
        if not checkpoint_path:
            raise FileNotFoundError(
                f"No FL2VA transformer shards found under {FL2VA_BASE}; "
                "pass --checkpoint explicitly."
            )
    else:
        if "," in args.checkpoint:
            checkpoint_path = [part.strip() for part in args.checkpoint.split(",") if part.strip()]
        elif os.path.isdir(args.checkpoint):
            checkpoint_path = sorted(glob.glob(os.path.join(args.checkpoint, "*.safetensors")))
            if not checkpoint_path:
                raise FileNotFoundError(
                    f"No .safetensors checkpoint shards found in directory: {args.checkpoint}"
                )
        else:
            checkpoint_path = args.checkpoint
        checkpoint_files = checkpoint_path if isinstance(checkpoint_path, list) else [checkpoint_path]
        missing = [path for path in checkpoint_files if not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError(f"H3 checkpoint file not found: {missing}")
        invalid = [path for path in checkpoint_files if not path.endswith(".safetensors")]
        if invalid:
            raise ValueError(
                "--checkpoint must point to merged/base .safetensors weights; "
                f"training checkpoints such as .pt are not valid: {invalid}"
            )
    if args.lora is not None:
        if not os.path.isfile(args.lora):
            raise FileNotFoundError(f"LoRA file not found: {args.lora}")
        if args.lora_scale < 0:
            raise ValueError("--lora-scale must be non-negative")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        device_id=local_rank,
    )

    world_size = dist.get_world_size()
    if world_size != args.cp_world_size:
        raise RuntimeError(
            f"Launched {world_size} processes but requested CP size "
            f"{args.cp_world_size}. Launch one rank per GPU with "
            "--nproc_per_node equal to --cp_world_size."
        )

    cp_rank = dist.get_rank()
    cp_group = create_cp_process_group(cp_rank, args.cp_world_size)
    device = f"cuda:{local_rank}"
    if dist.get_rank() == 0:
        print(
            f"Distributed inference: world_size={world_size}, "
            f"cp_world_size={args.cp_world_size}, device={device}, "
            f"checkpoint={checkpoint_path}"
        )

    vram_config = _cpu_offload_vram_config()
    vram_config["preparing_device"] = device
    vram_config["computation_device"] = device
    text_encoder_paths = sorted(
        glob.glob(f"{FL2VA_BASE}/text_encoder/model*.safetensors")
    )
    video_vae_path = f"{FL2VA_BASE}/video_vae/source/model.safetensors"
    audio_vae_path = f"{FL2VA_BASE}/audio_vae/model.safetensors"

    pipe = MiniMaxH3Pipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(path=text_encoder_paths, **vram_config),
            # This is the exact loading policy used for the reviewed 8-step
            # masked-av-v14 reference.  The full merged DiT stays resident
            # during denoising; only the VAEs use the low-VRAM policy.
            # Match MiniMax-H3-Continuation.py: the DiT also uses the
            # standard CPU-offload policy.  Keeping it as a dense, always-GPU
            # module changes memory paging and is not the reference path.
            ModelConfig(path=checkpoint_path, **vram_config),
            ModelConfig(path=video_vae_path, **vram_config),
            ModelConfig(path=audio_vae_path, **vram_config),
        ],
        processor_config=ModelConfig(
            path=f"{FL2VA_BASE}/processor"
        ),
        vram_limit=torch.cuda.mem_get_info(device)[1] / (1024 ** 3) - 4,
    )
    if args.lora is not None:
        # H3 acceleration weights are LoRA adapters, not replacement DiT checkpoints.
        pipe.load_lora(pipe.dit, args.lora, alpha=args.lora_scale)
        if dist.get_rank() == 0:
            print(f"Loaded acceleration LoRA: {args.lora} (scale={args.lora_scale})")
    # A single-rank run must use the pipeline's native model function.  The
    # CP wrapper is only needed when rows are partitioned across ranks; using
    # it for world_size=1 used to make the inference path diverge from the
    # known-good continuation generator and could corrupt decoded latents.
    if args.cp_world_size > 1:
        pipe.model_fn = make_cp_model_fn(cp_rank, args.cp_world_size, cp_group)

    # Match the reviewed continuation path: retain DiT on the accelerator for
    # denoising, then move it off only while the writer reconstructs AV.  This
    # avoids changing DiT parameter paging semantics at eight-step sampling.
    if args.cp_world_size > 1:
        _original_load_models_to_device = pipe.load_models_to_device

        def _load_models_to_device(model_names):
            if "dit" in model_names and pipe.dit is not None:
                for vae_name in ("video_vae", "audio_vae"):
                    vae = getattr(pipe, vae_name, None)
                    if vae is not None:
                        vae.to("cpu")
                torch.cuda.empty_cache()
                pipe.dit = pipe.dit.to(device)
                torch.cuda.empty_cache()
            if any(name in model_names for name in ("video_vae", "audio_vae")):
                if pipe.dit is not None:
                    pipe.dit = pipe.dit.to("cpu")
                    torch.cuda.empty_cache()
            return _original_load_models_to_device(model_names)

        pipe.load_models_to_device = _load_models_to_device

    dist.barrier()

    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompt = f.read().strip()
    elif args.prompt:
        prompt = args.prompt
    else:
        prompt = (
            "A girl is very happy, she is speaking in English: "
            "\"I enjoy working with DiffSynth-Studio, it's a perfect framework.\""
        )
    if dist.get_rank() == 0:
        if args.num_inference_steps < 50:
            print(
                "Warning: MiniMax-H3 base/T2VA quality is tuned for 50 steps; "
                "20 steps often produces noisy video."
            )
        print(f"Using prompt file: {args.prompt_file}")
        print(f"Prompt length: {len(prompt)} characters")
        print(f"Using num_inference_steps={args.num_inference_steps}")
    if args.continuation_plan:
        plan = load_h3_segment_plan(args.continuation_plan)
        config = H3ContinuationConfig(
            requested_window_frames=args.continuation_window_frames,
            overlap_frames=args.continuation_overlap_frames,
            audio_crossfade_ms=args.audio_crossfade_ms,
        )
        config.validate()

        def _synchronize_window(*, window, seed, controls):
            local = cp_window_metadata(
                window, seed=seed, num_inference_steps=args.num_inference_steps,
                continuation_mode=args.continuation_mode,
            )
            gathered = [None for _ in range(args.cp_world_size)]
            dist.all_gather_object(gathered, local, group=cp_group)
            validate_cp_window_metadata(local, gathered)

        runner_pipeline = (
            _WriterOnlyDecodePipeline(pipe, is_writer=True)
            if args.cp_world_size > 1
            else pipe
        )
        runner = H3ContinuationRunner(
            runner_pipeline, config,
            model_identity=f"{checkpoint_path}|lora={args.lora}|scale={args.lora_scale}",
            continuation_mode=args.continuation_mode,
            prefer_latent_handoff=args.continuation_mode in (
                "latent-handoff", "masked-av-v14"
            ),
            appearance_memory_config=H3AppearanceMemoryConfig(
                mode=args.appearance_memory_mode,
                trusted_anchor_frames=args.appearance_trusted_anchor_frames,
                memory_frame_budget=args.appearance_memory_frames,
                boundary_reference_frames=bool(args.appearance_boundary_reference),
                max_visual_references=args.appearance_max_visual_references,
            ),
            manifest_directory=(os.path.splitext(args.output_path)[0] + ".state") if is_designated_cp_writer(cp_rank) else None,
            window_synchronizer=_synchronize_window,
        )
        result = runner.run(
            plan, base_seed=args.seed,
            pipeline_kwargs={
                "height": args.height, "width": args.width,
                "num_inference_steps": args.num_inference_steps, "cfg_scale": 1.0,
            },
        )
        video, audio = result.video, result.audio
    else:
        video, audio = pipe(
            prompt=prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            seed=args.seed,
            cfg_scale=1.0,
        )

    if args.output_video_frames and video is not None and audio is not None:
        target_frames = min(args.output_video_frames, len(video))
        target_audio_samples = round(target_frames / 24.0 * pipe.audio_vae.sample_rate)
        video = list(video[:target_frames])
        audio = audio[..., :target_audio_samples]
        if dist.get_rank() == 0:
            print(
                f"Trimmed assembled output to {target_frames} frames "
                f"({target_audio_samples} audio samples)."
            )

    dist.barrier()
    if dist.get_rank() == 0:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
        write_video_audio(
            video=video,
            audio=audio,
            output_path=args.output_path,
            fps=24,
            audio_sample_rate=pipe.audio_vae.sample_rate,
        )
        print(
            f"saved {args.output_path}, frames={len(video)}, "
            f"audio={tuple(audio.shape)}"
        )
        if args.continuation_plan:
            report_path = os.path.splitext(args.output_path)[0] + ".json"
            write_continuation_evaluation(result, report_path)
            print(f"saved continuation report {report_path}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
