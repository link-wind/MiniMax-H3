import torch, os, argparse, json, accelerate, sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from diffsynth.core.context_parallel import (
    create_cp_process_group,
    validate_cp_setup,
    validate_cp_topology,
)
from diffsynth.diffusion import *
from diffsynth.utils.continuation_lora import (
    ContinuationRegionConfig,
    ContinuationLatentDataset,
    audit_memory_slots,
    check_memory_only_sample,
    select_memory_slots,
)
from diffsynth.diffusion.loss import ContinuationFlowMatchSFTMiniMaxH3AudioVideoLoss
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MINIMAX_H3_FRAME_RATE = 24
MINIMAX_H3_TIME_DIVISION_FACTOR = 17
MINIMAX_H3_TIME_DIVISION_REMAINDER = 5


class MiniMaxH3TrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        processor_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        quant_options=None,
        template_model_id_or_path=None,
        resume_from_checkpoint=None, remove_prefix_in_ckpt=None,
        silent_on_missing_audio=False,
        training_cfg_scale=1.0,
        continuation_memory_mode="off",
        continuation_memory_expect_slots=None,
        device="cpu",
        task="sft",
        cp_rank=0,
        cp_world_size=1,
        cp_group=None,
    ):
        super().__init__()
        if training_cfg_scale < 1.0:
            raise ValueError("training_cfg_scale must be at least 1.0")
        self.training_cfg_scale = training_cfg_scale
        from diffsynth.pipelines.minimax_h3_audio_video import (
            MiniMaxH3Pipeline,
            ModelConfig,
        )
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, quant_options=quant_options, device=device)
        pipe_kwargs = {}
        if processor_path is not None:
            processor_config = self.parse_path_or_model_id(processor_path)
            if processor_config.path is not None:
                pipe_kwargs["processor_config"] = processor_config
            else:
                pipe_kwargs["processor_config"] = ModelConfig(
                    model_id=processor_config.model_id,
                    origin_file_pattern=processor_config.origin_file_pattern,
                )
        self.pipe = MiniMaxH3Pipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, **pipe_kwargs)
        self.pipe = self.load_training_template_model(self.pipe, template_model_id_or_path, use_gradient_checkpointing, use_gradient_checkpointing_offload)
        self.pipe = self.split_pipeline_units(
            task, self.pipe, trainable_models, lora_base_model,
            remove_unnecessary_params=True,
            force_remove_params_shared=("video_latents", "audio_latents"),
            force_remove_params_nega=("prompt_embeds", "text_token_tags", "packed") if training_cfg_scale == 1.0 else (),
        )
        self.resume_from_checkpoint(resume_from_checkpoint, remove_prefix_in_ckpt)
        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )
        self.pipe.scheduler_audio.set_timesteps(1000, training=True)

        # Store other configs
        self.silent_on_missing_audio = silent_on_missing_audio
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.continuation_memory_mode = continuation_memory_mode
        # Memory-only training requires the memory slots to be the *only*
        # cross-shot conditioning, so every sample must be prefix-free.  The
        # expected slot names make a mis-built cache (for example an STM-only
        # cache fed to the dual-slot layout) fail on the first step instead of
        # silently training a different objective.
        self.continuation_memory_expect_slots = tuple(continuation_memory_expect_slots or ())
        self.fp8_models = fp8_models
        self.task = task
        self.cp_rank = cp_rank
        self.cp_world_size = cp_world_size
        self.cp_group = cp_group
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTMiniMaxH3AudioVideoLoss(
                pipe, training_cfg_scale=self.training_cfg_scale, inputs_nega=inputs_nega,
                **inputs_shared, **inputs_posi,
                cp_rank=self.cp_rank, cp_world_size=self.cp_world_size, cp_group=self.cp_group,
            ),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTMiniMaxH3AudioVideoLoss(
                pipe, training_cfg_scale=self.training_cfg_scale, inputs_nega=inputs_nega,
                **inputs_shared, **inputs_posi,
                cp_rank=self.cp_rank, cp_world_size=self.cp_world_size, cp_group=self.cp_group,
            ),
            "continuation_sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: ContinuationFlowMatchSFTMiniMaxH3AudioVideoLoss(
                pipe, training_cfg_scale=training_cfg_scale, inputs_nega=inputs_nega,
                region_config=ContinuationRegionConfig(
                    overlap_video_steps=getattr(self, "continuation_overlap_steps", 12),
                    hard_core_video_steps=getattr(self, "continuation_hard_core_steps", 12),
                    transition_video_steps=getattr(self, "continuation_transition_steps", 0),
                    first_suffix_clip_steps=getattr(self, "continuation_first_suffix_steps", 5),
                    transition_weight=getattr(self, "continuation_transition_weight", 0.5),
                    first_suffix_weight=getattr(self, "continuation_first_suffix_weight", 3.0),
                    suffix_weight=getattr(self, "continuation_suffix_weight", 1.0),
                    lambda_audio=getattr(self, "continuation_lambda_audio", 0.5),
                    conditioning_mode=getattr(self, "continuation_conditioning", "masked-av-v14"),
                    prefix_present_mode=getattr(self, "continuation_prefix_present_mode", "noised"),
                ),
                **inputs_shared, **inputs_posi,
                cp_rank=self.cp_rank, cp_world_size=self.cp_world_size, cp_group=self.cp_group,
            ),
        }

    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        # First/last-frame conditioning is derived from the training video itself, following the
        # `input_image` / `end_image` convention used by the wanvideo series. The H3 pipeline takes
        # them as a `keyframes` list plus `keyframe_indices` in {0, -1}.
        keyframes, keyframe_indices = [], []
        for extra_input in extra_inputs:
            if extra_input == "input_image":
                keyframes.append(data["video"][0])
                keyframe_indices.append(0)
            elif extra_input == "end_image":
                keyframes.append(data["video"][-1])
                keyframe_indices.append(-1)
            else:
                inputs_shared[extra_input] = data[extra_input]
        if keyframes:
            inputs_shared["keyframes"] = keyframes
            inputs_shared["keyframe_indices"] = keyframe_indices
        # If no audio tracks in the video file, we use a silence tensor for training.
        if self.silent_on_missing_audio:
            if "input_audio" in extra_inputs and "input_audio" in inputs_shared and inputs_shared["input_audio"] is None:
                inputs_shared["input_audio"] = (torch.zeros((2, 800 * round(len(data["video"]) / 24 * 40))), 32000)
        return inputs_shared

    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {"negative_prompt": " "}
        cached = "input_latents" in data
        if cached:
            video_latents = data["input_latents"]
            audio_latents = data.get("audio_input_latents")
            memory_latents = data.get("continuation_memory_video_latents")
            memory_on = self.continuation_memory_mode in ("on", "memory-only")
            if memory_on and not memory_latents and self.continuation_memory_mode != "memory-only":
                # The ``on`` arm is the A/B control for the M1-fast block, so a
                # cache without slots would silently make it identical to ``off``.
                # ``memory-only`` allows a slot-free sample (the first shot of a
                # sequence has nothing to remember); the run-level audit in
                # ``main`` catches a cache that has no slots at all.
                raise ValueError(
                    f"continuation_memory_mode={self.continuation_memory_mode} but this cache entry "
                    "has no memory slot; rebuild the cache with --memory-frames/--ltm-frames"
                )
            if self.continuation_memory_mode == "memory-only":
                # The memory slots must be the *only* cross-shot conditioning.
                check_memory_only_sample(data.get("continuation_prefix_mode", "masked"), memory_latents)
                # ``--continuation_memory_expect_slots`` is the ablation switch:
                # naming a subset drops the other slots, so one cache serves the
                # stm+ltm / stm / ltm arms without re-encoding.  A shot that ends
                # up with no slot at all (first shot of the stm-only arm) is a
                # plain text-to-shot sample, which is the right baseline.
                memory_latents = select_memory_slots(memory_latents, self.continuation_memory_expect_slots) or None
            inputs_shared = {
                "input_video": None,
                "input_audio": None,
                "input_latents": video_latents,
                "audio_input_latents": audio_latents,
                # Long-horizon memory (M1-fast): a clean latent block packed in
                # front of the window.  It never enters img_pos, so the loss is
                # untouched; the switch only gates whether it is fed at all.
                "memory_latents": memory_latents if memory_on else None,
                "continuation_history_video_latents": data.get("continuation_history_video_latents", video_latents),
                "continuation_history_audio_latents": data.get("continuation_history_audio_latents", audio_latents),
                "continuation_prefix_mode": data.get("continuation_prefix_mode", "masked"),
                # Shot-level samples each carry their own overlap; the loss reads
                # it from the cache metadata instead of the run-level default.
                "continuation_cache_metadata": data.get("cache_metadata"),
                "height": data["height"], "width": data["width"], "num_frames": data["num_frames"],
                "keyframes": None, "keyframe_indices": None, "references": None,
                "ref_image_short_edge": 2048, "ref_video_short_edge": 768, "ref_video_max_pixels": 768 * 1344,
                "imgvid_cond_noise_aug": self.pipe.imgvid_cond_noise_aug,
                "audio_cond_noise_aug": self.pipe.audio_cond_noise_aug,
                "cfg_scale": self.training_cfg_scale, "seed": data.get("seed", 42),
                "rand_device": "cpu", "use_gradient_checkpointing": self.use_gradient_checkpointing,
                "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            }
            return inputs_shared, inputs_posi, inputs_nega
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            "keyframes": None,
            "keyframe_indices": None,
            "references": None,
            "ref_image_short_edge": 2048,
            "ref_video_short_edge": 768, "ref_video_max_pixels": 768 * 1344,
            "imgvid_cond_noise_aug": self.pipe.imgvid_cond_noise_aug,
            "audio_cond_noise_aug": self.pipe.audio_cond_noise_aug,
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            # Reuse the pipeline's CFG preprocessing path to build unconditional
            # embeddings when CFG-aware training is enabled.
            "cfg_scale": self.training_cfg_scale,
            "seed": 42,
            "rand_device": "cpu",
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega

    def forward(self, data, inputs=None):
        # The runner may pass either preprocessed pipeline inputs or a cached
        # sample dictionary.  Normalize both forms before positional expansion
        # into PipelineUnitRunner; expanding the cache dict itself yields keys.
        if inputs is None or isinstance(inputs, dict):
            inputs = self.get_pipeline_inputs(data if inputs is None else inputs)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        # rank0-only diagnostic: report the audio_input_latents the loss
        # actually receives (after pipeline unit forward). Off by default.
        if os.environ.get("DIFFSYNTH_AUDIO_DEBUG") == "1":
            try:
                is_mp = (not torch.distributed.is_initialized()) or (torch.distributed.get_rank() == 0)
            except Exception:
                is_mp = True
            if is_mp:
                _aud = inputs[0].get("audio_input_latents") if isinstance(inputs, (list, tuple)) else None
                print(f"[audiodbg-loss] inputs_audio_input_latents="
                      f"{None if _aud is None else tuple(_aud.shape)}", flush=True)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss


def minimax_h3_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.add_argument("--processor_path", type=str, default=None, help="Path or `model_id:pattern` of the Qwen3-VL processor.")
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true", help="Whether to initialize models on CPU.")
    parser.add_argument("--silent_on_missing_audio", default=False, action="store_true", help="Whether to use silent audio as a fallback when no audio track is present in the video data.")
    parser.add_argument("--training_cfg_scale", type=float, default=1.0, help="Inverse-CFG scale for preserving MiniMax-H3 guidance distillation during fine-tuning. Values greater than 1 enable a no-grad unconditional branch; 1 keeps the standard flow-matching loss.")
    parser.add_argument("--cp_world_size", type=int, default=1, help="Context parallel size. process_count must be divisible by this value.")
    parser.add_argument("--cp_seed", type=int, default=42, help="Seed used to keep sample order consistent inside each CP group.")
    parser.add_argument("--cp_gradient_reduction", type=str, default="global-group", choices=["global-group", "dp-only"], help="Parameter gradient strategy for CP. Use dp-only only with a DP-only ZeRO/FSDP parameter state.")
    parser.add_argument("--validate_cp_setup", default=False, action="store_true", help="Validate the CP launch shape and exit without loading models.")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for deterministic training and resume.")
    parser.add_argument("--bf16", default=False, action="store_true", help="Use bf16 mixed precision. H3 models already use bf16 by default; this flag also records the intent in training args and Accelerator.")
    parser.add_argument("--continuation_overlap_steps", type=int, default=12, help="Video latent overlap tokens for mask-v14 (39 frames = 12 tokens).")
    parser.add_argument("--continuation_memory_mode", choices=("off", "on", "memory-only"), default="off",
                        help=(
                            "How the cached long-horizon memory is used. 'off' reproduces the previous "
                            "packed layout bit for bit. 'on' feeds the cached slots when present. "
                            "'memory-only' additionally requires every sample to be prefix-free and to "
                            "carry the expected slots, i.e. the whole window is generated from "
                            "memory + text with no context rows at all."
                        ))
    parser.add_argument("--continuation_memory_expect_slots", type=str, default=None,
                        help="Comma-separated slot names the cache must provide in memory-only mode (e.g. stm,ltm).")
    parser.add_argument("--continuation_hard_core_steps", type=int, default=12)
    parser.add_argument("--continuation_transition_steps", type=int, default=0)
    parser.add_argument("--continuation_first_suffix_steps", type=int, default=5)
    parser.add_argument("--continuation_transition_weight", type=float, default=0.5)
    parser.add_argument("--continuation_first_suffix_weight", type=float, default=3.0)
    parser.add_argument("--continuation_suffix_weight", type=float, default=1.0)
    parser.add_argument("--continuation_lambda_audio", type=float, default=0.5)
    parser.add_argument("--continuation_prefix_present_mode", choices=("noised", "clean", "mixed"), default="noised",
                        help="Prefix input form in masked-av-v14 training: noised-at-t (v3), clean injection with timestep=1 (inference form), or a 50/50 mix.")
    parser.add_argument("--continuation_conditioning", choices=("masked-av-v14",), default="masked-av-v14")
    parser.add_argument("--validate_continuation_config", action="store_true", help="Validate continuation region and LoRA target configuration without loading models.")
    parser.add_argument("--continuation_manifest", type=str, default=None, help="Cache manifest or cache directory for continuation_sft.")
    parser.add_argument("--continuation_split", type=str, default="train", choices=("train", "validation", "test"))
    parser.add_argument("--continuation_max_items", type=int, default=None)
    parser.add_argument("--continuation_checkpoint_save_path", type=str, default=None, help="Optional full resume checkpoint path for continuation_sft.")
    parser.add_argument("--continuation_checkpoint_interval", type=int, default=None, help="Save a full continuation checkpoint every N optimizer steps; defaults to --save_steps.")
    parser.add_argument("--continuation_resume_path", type=str, default=None, help="Resume a full continuation training checkpoint (LoRA + optimizer + scheduler + global step + config metadata).")
    return parser


if __name__ == "__main__":
    parser = minimax_h3_parser()
    args = parser.parse_args()
    if args.task == "continuation_sft":
        if args.lora_base_model is None:
            args.lora_base_model = "dit"
        if args.lora_target_modules == "q,k,v,o,ffn.0,ffn.2":
            args.lora_target_modules = "attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2"
    if args.validate_continuation_config:
        config = ContinuationRegionConfig(
            overlap_video_steps=args.continuation_overlap_steps,
            hard_core_video_steps=args.continuation_hard_core_steps,
            transition_video_steps=args.continuation_transition_steps,
            first_suffix_clip_steps=args.continuation_first_suffix_steps,
            transition_weight=args.continuation_transition_weight,
            first_suffix_weight=args.continuation_first_suffix_weight,
            suffix_weight=args.continuation_suffix_weight,
            lambda_audio=args.continuation_lambda_audio,
            conditioning_mode=args.continuation_conditioning,
            prefix_present_mode=args.continuation_prefix_present_mode,
        )
        config.validate(video_steps=20)
        targets = [name.strip() for name in args.lora_target_modules.split(",") if name.strip()]
        if targets and any(target not in {"attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"} for target in targets):
            raise ValueError("continuation LoRA targets must be attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2")
        print(json.dumps({"continuation": True, "region_config": config.__dict__, "lora_targets": targets or ["default H3 DiT targets"]}, ensure_ascii=False, indent=2))
        raise SystemExit(0)
    if args.num_frames % MINIMAX_H3_TIME_DIVISION_FACTOR != MINIMAX_H3_TIME_DIVISION_REMAINDER:
        raise ValueError(
            f"--num_frames must be {MINIMAX_H3_TIME_DIVISION_FACTOR}n+{MINIMAX_H3_TIME_DIVISION_REMAINDER} "
            f"(e.g. 39, 56, 124) so it lands on the video VAE's temporal grouping, got {args.num_frames}."
        )
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
        mixed_precision="bf16" if args.bf16 else None,
    )
    topology = validate_cp_topology(accelerator.num_processes, args.cp_world_size)
    platform_node_count = os.environ.get(
        "GEMINI_TASK_ROLE_TASK_COUNT_taskrole1"
    )
    if platform_node_count is not None:
        local_gpu_count = max(1, torch.cuda.device_count())
        expected_process_count = int(platform_node_count) * local_gpu_count
        if accelerator.num_processes != expected_process_count:
            raise RuntimeError(
                f"Gemini platform reports {platform_node_count} nodes, "
                f"expected {expected_process_count} accelerator processes "
                f"({local_gpu_count} GPUs per node), but this job started "
                f"{accelerator.num_processes}. Use one torchrun command per "
                "node with --nnodes/--nproc_per_node, or use accelerate launch with --deepspeed_multinode_launcher standard and --num_processes <expected>, instead of two "
                "independent local 8-rank launches."
            )
    args.cp_world_size = topology["cp_world_size"]
    args.dp_world_size = topology["dp_world_size"]
    args.cp_rank = accelerator.process_index % args.cp_world_size
    args.dp_rank = accelerator.process_index // args.cp_world_size
    args.cp_group = create_cp_process_group(accelerator.process_index, args.cp_world_size)
    if accelerator.is_main_process:
        print(
            f"CP topology: process_count={args.cp_world_size * args.dp_world_size}, "
            f"cp_world_size={args.cp_world_size}, dp_world_size={args.dp_world_size}"
        )
        if args.cp_world_size > 1:
            if args.cp_gradient_reduction == "dp-only":
                print("Using DP-only parameter state with explicit CP gradient reduction.")
            else:
                print("Using DeepSpeed/global-group parameter gradient reduction; no extra CP allreduce is applied.")
    if args.validate_cp_setup:
        deepspeed_config = None
        if getattr(accelerator.state, "deepspeed_plugin", None) is not None:
            deepspeed_config = (
                accelerator.state.deepspeed_plugin.deepspeed_config
            )
        report = validate_cp_setup(
            process_count=accelerator.num_processes,
            cp_world_size=args.cp_world_size,
            deepspeed_config=deepspeed_config,
        )
        if accelerator.is_main_process:
            print(
                json.dumps(report, indent=2, ensure_ascii=False, default=str)
            )
        accelerator.end_training()
        raise SystemExit(0)
    from diffsynth.core import UnifiedDataset
    from diffsynth.core.data.operators import (
        LoadAudioWithTorchaudio,
        ToAbsolutePath,
    )
    from diffsynth.utils.data.minimax_h3 import MiniMaxH3ReferenceLoader

    if args.task == "continuation_sft" and args.continuation_manifest is None:
        raise ValueError("continuation_sft requires --continuation_manifest pointing to a latent cache manifest")

    video_processor = UnifiedDataset.default_video_operator(
        base_path=args.dataset_base_path,
        max_pixels=args.max_pixels,
        height=args.height,
        width=args.width,
        height_division_factor=32,
        width_division_factor=32,
        num_frames=args.num_frames,
        time_division_factor=MINIMAX_H3_TIME_DIVISION_FACTOR,
        time_division_remainder=MINIMAX_H3_TIME_DIVISION_REMAINDER,
        frame_rate=MINIMAX_H3_FRAME_RATE,
        fix_frame_rate=True,
    )
    if args.task == "continuation_sft" and args.continuation_memory_mode == "memory-only":
        # A memory-only run must not be silently memory-free: per-sample, a
        # missing slot is legal, so the "was this cache built with slots at all"
        # question is answered once here.
        audit = audit_memory_slots(args.continuation_manifest)
        if not audit["records_with_slots"]:
            raise ValueError(
                f"continuation_memory_mode=memory-only but no entry in {audit['manifest']} carries a "
                "memory slot; rebuild the cache with --memory-frames/--ltm-frames"
            )
        print(f"[memory-only] {json.dumps(audit, ensure_ascii=False)}", flush=True)
    dataset = ContinuationLatentDataset(args.continuation_manifest, split=args.continuation_split, max_items=args.continuation_max_items) if args.task == "continuation_sft" else UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=video_processor,
        special_operator_map={
            "input_audio": ToAbsolutePath(args.dataset_base_path) >> LoadAudioWithTorchaudio(
                num_frames=args.num_frames,
                time_division_factor=MINIMAX_H3_TIME_DIVISION_FACTOR,
                time_division_remainder=MINIMAX_H3_TIME_DIVISION_REMAINDER,
                frame_rate=MINIMAX_H3_FRAME_RATE,
                fix_frame_rate=True,
            ),
            "references": MiniMaxH3ReferenceLoader(
                base_path=args.dataset_base_path,
                height=args.height,
                width=args.width,
                max_pixels=args.max_pixels,
                num_frames=args.num_frames,
                frame_rate=MINIMAX_H3_FRAME_RATE,
            ),
        }
    )
    model = MiniMaxH3TrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        processor_path=args.processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        quant_options=args.quant_options,
        template_model_id_or_path=args.template_model_id_or_path,
        resume_from_checkpoint=args.resume_from_checkpoint,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        silent_on_missing_audio=args.silent_on_missing_audio,
        training_cfg_scale=args.training_cfg_scale,
        continuation_memory_mode=args.continuation_memory_mode,
        continuation_memory_expect_slots=(
            tuple(name.strip() for name in args.continuation_memory_expect_slots.split(",") if name.strip())
            if args.continuation_memory_expect_slots else None
        ),
        task=args.task,
        device="cpu" if (args.initialize_model_on_cpu or args.enable_model_cpu_offload) else accelerator.device,
        cp_rank=args.cp_rank,
        cp_world_size=args.cp_world_size,
        cp_group=args.cp_group,
    )
    model.continuation_overlap_steps = args.continuation_overlap_steps
    model.continuation_hard_core_steps = args.continuation_hard_core_steps
    model.continuation_transition_steps = args.continuation_transition_steps
    model.continuation_first_suffix_steps = args.continuation_first_suffix_steps
    model.continuation_transition_weight = args.continuation_transition_weight
    model.continuation_first_suffix_weight = args.continuation_first_suffix_weight
    model.continuation_suffix_weight = args.continuation_suffix_weight
    model.continuation_lambda_audio = args.continuation_lambda_audio
    model.continuation_conditioning = args.continuation_conditioning
    model.continuation_prefix_present_mode = args.continuation_prefix_present_mode
    if args.initialize_model_on_cpu:
        # Keep the module-level device pointer consistent with the accelerator
        # after DeepSpeed ZeRO-3 partitions the CPU-initialized model.
        model.pipe.device = accelerator.device
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        enable_tensorboard_log=args.enable_tensorboard_log,
        enable_swanlab_log=args.enable_swanlab_log,
        swanlab_project=args.swanlab_project,
        enable_wandb_log=args.enable_wandb_log,
        wandb_project=args.wandb_project,
        enable_csv_log=args.enable_csv_log,
    )
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
        "continuation_sft": launch_training_task,
        "continuation_sft:data_process": launch_data_process_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
