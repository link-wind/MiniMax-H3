"""MiniMax-H3 multi-window continuation example.

The default shape is 345 frames with a 34-frame overlap.  The first window is
normal H3 generation; later windows use Retake-hard, latent handoff, or an
experimental latent continuation mode.
"""

import argparse
import json
from pathlib import Path
import time

import torch

from diffsynth.core import ModelConfig
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline
from diffsynth.pipelines.h3_appearance_memory import H3AppearanceMemoryConfig
from diffsynth.pipelines.minimax_h3_continuation import (
    H3ContinuationConfig,
    H3ContinuationRunner,
    is_masked_av_context_frames,
    load_h3_segment_plan,
    write_continuation_evaluation,
)
from diffsynth.utils.data.audio_video import write_video_audio
from diffsynth.utils.continuation_lora import apply_continuation_lora


def resolve_checkpoint_paths(checkpoint: str) -> str | list[str]:
    """Accept a single checkpoint, comma-separated shards, or a shard directory."""
    path = Path(checkpoint)
    if path.is_dir():
        shards = [str(item) for item in sorted(path.glob("*.safetensors"))]
        if not shards:
            raise ValueError(f"checkpoint directory contains no .safetensors files: {path}")
        return shards
    if "," in checkpoint:
        shards = [part.strip() for part in checkpoint.split(",") if part.strip()]
        if not shards or not all(Path(item).is_file() for item in shards):
            raise ValueError("every comma-separated --checkpoint shard must be an existing file")
        return shards
    if not path.is_file():
        raise ValueError(f"--checkpoint must be a file, shard directory, or comma-separated existing files: {checkpoint}")
    return str(path)


def cpu_offload_vram_config() -> dict:
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


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a long MiniMax-H3 video/audio timeline.")
    parser.add_argument("--segment-plan", type=Path, default=Path(__file__).with_name("h3_continuation_plan.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/h3_continuation.mp4"))
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--evaluation-variant", type=str, default=None, help="Optional fixed-matrix variant name recorded in the report.")
    parser.add_argument("--checkpoint", required=True, help="H3 transformer checkpoint or comma-separated shards")
    parser.add_argument("--h3-base", type=Path, required=True, help="FL2VA root containing text_encoder/, video_vae/, audio_vae/, and processor/")
    parser.add_argument("--lora", type=Path, default=None, help="Optional continuation LoRA safetensors loaded on top of the base H3 checkpoint.")
    parser.add_argument("--lora-scale", type=float, default=1.0, help="LoRA scale. Use 0 for an exact base-model run.")
    parser.add_argument("--turbo-lora", type=Path, default=None, help="Optional MiniMax-H3 Turbo 8-step distillation LoRA (lightx2v/Minimax-h3-Turbo, e.g. minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors). Loaded on top of the base checkpoint and auto-converted from the lightx2v layout.")
    parser.add_argument("--turbo-lora-scale", type=float, default=1.0, help="Scale for --turbo-lora (default: 1.0).")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--window-frames", type=int, default=345, help="H3 continuation window; 345 frames is the shot-safe default for this corpus.")
    parser.add_argument("--overlap-frames", type=int, default=39, help="Continuation context. Use 39/90/141/... for native Masked AV; legacy modes may use 17-frame multiples.")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--audio-sample-rate", type=int, default=32000)
    parser.add_argument("--audio-crossfade-ms", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--flow-shift", type=float, default=None, help="Video flow shift. Defaults to the pipeline value (12.0); few-step runs (8) usually want ~6.0.")
    parser.add_argument("--audio-flow-shift", type=float, default=None, help="Audio flow shift. Defaults to the pipeline value (3.0).")
    parser.add_argument("--mode", choices=("retake-hard", "latent-handoff", "masked-av-v14"), default="retake-hard")
    parser.add_argument(
        "--shared-noise", action="store_true",
        help="FreeNoise-style shared noise: slice every window from one deterministic "
        "global noise timeline at its temporal offset instead of independent noise.",
    )
    parser.add_argument("--global-latent-decode", action="store_true", help="Assemble all window latents and decode the timeline once.")
    parser.add_argument("--global-reference-frames", type=int, default=0, help="Use 1-4 middle frames from the first window as image references for later windows.")
    parser.add_argument("--boundary-reference-frames", type=int, choices=(0, 1), default=0, help="Also pass the previous window's final decoded frame as a local image reference.")
    parser.add_argument("--appearance-memory-mode", choices=("disabled", "static", "dynamic"), default=None, help="Enable the appearance memory bank. Static keeps first-window anchors; dynamic also selects diverse memory frames.")
    parser.add_argument("--appearance-trusted-anchor-frames", type=int, default=2, help="Number of immutable first-window anchor frames used by the appearance memory bank.")
    parser.add_argument("--appearance-memory-frames", type=int, default=1, help="Number of dynamically selected memory frames kept by the appearance memory bank.")
    parser.add_argument("--appearance-max-visual-references", type=int, default=4, help="Maximum visual references emitted by the appearance memory bank in one H3 request.")
    args = parser.parse_args()
    if args.window_frames <= 0:
        parser.error("--window-frames must be positive")
    if args.overlap_frames <= 0 or args.overlap_frames >= args.window_frames:
        parser.error("--overlap-frames must be positive and smaller than --window-frames")
    if args.overlap_frames % 17 and not is_masked_av_context_frames(args.overlap_frames):
        parser.error("--overlap-frames must be divisible by 17 or an exact native Masked AV length (39, 90, 141, ...)")
    if args.fps != 24:
        parser.error("MiniMax-H3 continuation currently requires --fps 24")
    if args.audio_sample_rate != 32000:
        parser.error("MiniMax-H3 continuation currently requires --audio-sample-rate 32000")
    if not 0 <= args.audio_crossfade_ms <= 200:
        parser.error("--audio-crossfade-ms must be between 0 and 200")
    if args.num_inference_steps <= 0:
        parser.error("--num-inference-steps must be positive")
    if args.lora_scale < 0:
        parser.error("--lora-scale must be non-negative")
    if args.lora is not None and not args.lora.exists():
        parser.error(f"--lora path does not exist: {args.lora}")
    if args.turbo_lora_scale < 0:
        parser.error("--turbo-lora-scale must be non-negative")
    if args.turbo_lora is not None and not args.turbo_lora.exists():
        parser.error(f"--turbo-lora path does not exist: {args.turbo_lora}")
    if not 0 <= args.global_reference_frames <= 4:
        parser.error("--global-reference-frames must be between 0 and 4")
    if args.global_reference_frames and args.global_latent_decode:
        parser.error("--global-reference-frames cannot be combined with --global-latent-decode")
    if args.appearance_memory_mode is not None:
        if args.mode != "masked-av-v14":
            parser.error("--appearance-memory-mode is currently only supported with --mode masked-av-v14")
        if args.global_latent_decode:
            parser.error("--appearance-memory-mode cannot be combined with --global-latent-decode")
        if args.global_reference_frames:
            parser.error("--appearance-memory-mode cannot be combined with --global-reference-frames")
        if args.appearance_memory_mode == "static" and args.appearance_memory_frames:
            parser.error("--appearance-memory-frames must be 0 in static appearance memory mode")
        if args.appearance_memory_mode == "dynamic" and args.appearance_trusted_anchor_frames < 1:
            parser.error("dynamic appearance memory mode requires at least one trusted anchor")
        try:
            H3AppearanceMemoryConfig(
                mode=args.appearance_memory_mode,
                trusted_anchor_frames=args.appearance_trusted_anchor_frames,
                memory_frame_budget=args.appearance_memory_frames,
                boundary_reference_frames=bool(args.boundary_reference_frames),
                max_visual_references=args.appearance_max_visual_references,
            )
        except ValueError as error:
            parser.error(str(error))
    required_assets = ("text_encoder", "video_vae", "audio_vae", "processor")
    missing = [name for name in required_assets if not (args.h3_base / name).exists()]
    if missing:
        parser.error(f"--h3-base is missing required FL2VA assets: {', '.join(missing)}")
    try:
        args.checkpoint = resolve_checkpoint_paths(args.checkpoint)
    except ValueError as error:
        parser.error(str(error))
    return args


def main():
    args = parse_args()
    plan = load_h3_segment_plan(args.segment_plan)
    config = H3ContinuationConfig(
        requested_window_frames=args.window_frames,
        overlap_frames=args.overlap_frames,
        video_fps=args.fps,
        audio_sample_rate=args.audio_sample_rate,
        audio_crossfade_ms=args.audio_crossfade_ms,
    )
    config.validate()
    if not torch.cuda.is_available():
        raise RuntimeError("MiniMax-H3 continuation inference requires CUDA")
    vram_config = cpu_offload_vram_config()
    pipe = MiniMaxH3Pipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(path=[str(path) for path in sorted((args.h3_base / "text_encoder").glob("model*.safetensors"))], **vram_config),
            ModelConfig(path=args.checkpoint, **vram_config),
            ModelConfig(path=str(args.h3_base / "video_vae" / "source" / "model.safetensors"), **vram_config),
            ModelConfig(path=str(args.h3_base / "audio_vae" / "model.safetensors"), **vram_config),
        ],
        processor_config=ModelConfig(path=str(args.h3_base / "processor")),
        vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 4,
    )
    if args.lora is not None:
        print(json.dumps(apply_continuation_lora(pipe, args.lora, args.lora_scale), ensure_ascii=False))
    if args.turbo_lora is not None:
        pipe.load_lora(pipe.dit, str(args.turbo_lora), alpha=args.turbo_lora_scale)
        print(json.dumps({"applied": True, "scale": args.turbo_lora_scale, "turbo_lora_path": str(args.turbo_lora)}, ensure_ascii=False))
    model_identity = str(args.checkpoint)
    if args.lora is not None:
        model_identity += f"|continuation_lora={args.lora}@{args.lora_scale}"
    if args.turbo_lora is not None:
        model_identity += f"|turbo_lora={args.turbo_lora}@{args.turbo_lora_scale}"
    legacy_global_reference_frames = args.global_reference_frames
    legacy_boundary_reference_frames = args.boundary_reference_frames
    appearance_memory_config = None
    if args.appearance_memory_mode is not None:
        appearance_memory_config = H3AppearanceMemoryConfig(
            mode=args.appearance_memory_mode,
            trusted_anchor_frames=args.appearance_trusted_anchor_frames,
            memory_frame_budget=args.appearance_memory_frames,
            boundary_reference_frames=bool(args.boundary_reference_frames),
            max_visual_references=args.appearance_max_visual_references,
        )
        legacy_global_reference_frames = 0
        legacy_boundary_reference_frames = 0
    runner = H3ContinuationRunner(
        pipe, config, model_identity=model_identity,
        continuation_mode=args.mode,
        prefer_latent_handoff=args.mode in ("latent-handoff", "masked-av-v14"),
        global_latent_decode=args.global_latent_decode,
        global_reference_frames=legacy_global_reference_frames,
        boundary_reference_frames=legacy_boundary_reference_frames,
        appearance_memory_config=appearance_memory_config,
        manifest_directory=args.output.parent / (args.output.stem + ".state"),
    )
    torch.cuda.reset_peak_memory_stats()
    started_at = time.perf_counter()
    pipeline_kwargs = {
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.num_inference_steps,
        "cfg_scale": 1.0,
    }
    if args.flow_shift is not None:
        pipeline_kwargs["flow_shift"] = args.flow_shift
    if args.audio_flow_shift is not None:
        pipeline_kwargs["audio_flow_shift"] = args.audio_flow_shift
    result = runner.run(
        plan, base_seed=args.seed,
        pipeline_kwargs=pipeline_kwargs,
    )
    elapsed_seconds = time.perf_counter() - started_at
    metadata = dict(result.state.experiment_metadata)
    metadata["runtime"] = {
        "elapsed_seconds": elapsed_seconds,
        "cuda_peak_memory_gib": torch.cuda.max_memory_allocated() / (1024 ** 3),
    }
    if args.evaluation_variant is not None:
        metadata["evaluation_variant"] = args.evaluation_variant
    result.state.experiment_metadata = metadata
    runner._persist_manifest(result.state)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_video_audio(result.video, result.audio, str(args.output), fps=args.fps, audio_sample_rate=args.audio_sample_rate)
    report_path = args.report or args.output.with_suffix(".json")
    write_continuation_evaluation(result, report_path)
    print(json.dumps({"output": str(args.output), "report": str(report_path), "frames": len(result.video)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
