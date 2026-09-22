"""Appearance-drift A/B evaluation for MiniMax-H3 appearance memory bank.

This entrypoint runs the same segment plan with no bank, static global
references, and the dynamic appearance memory bank, then writes one JSON
report with per-window reference composition and appearance metrics.
"""

import argparse
import json
from pathlib import Path

import torch

from diffsynth.core import ModelConfig
from diffsynth.pipelines.h3_appearance_evaluation import (
    format_appearance_metric,
    write_appearance_drift_report,
)
from diffsynth.pipelines.h3_appearance_memory import H3AppearanceMemoryConfig
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline
from diffsynth.pipelines.minimax_h3_continuation import (
    H3ContinuationConfig,
    H3ContinuationRunner,
    is_masked_av_context_frames,
    load_h3_segment_plan,
)
from diffsynth.utils.data.audio_video import write_video_audio


def _resolve_checkpoint(checkpoint: str) -> str | list[str]:
    path = Path(checkpoint)
    if path.is_dir():
        shards = [str(item) for item in sorted(path.glob("*.safetensors"))]
        if not shards:
            raise ValueError(f"checkpoint directory contains no .safetensors files: {path}")
        return shards
    if "," in checkpoint:
        shards = [part.strip() for part in checkpoint.split(",") if part.strip()]
        if not shards or not all(Path(item).is_file() for item in shards):
            raise ValueError("every comma-separated --checkpoint shard must exist")
        return shards
    if not path.is_file():
        raise ValueError(f"--checkpoint must be a file, shard directory, or comma-separated shards: {checkpoint}")
    return str(path)


def _cpu_offload_vram_config() -> dict:
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
    parser = argparse.ArgumentParser(description="Run appearance-drift A/B for MiniMax-H3.")
    parser.add_argument("--segment-plan", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--h3-base", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/h3_appearance_memory_eval"))
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--window-frames", type=int, default=362)
    parser.add_argument("--overlap-frames", type=int, default=90)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--modes", type=str, default="no-bank,static,dynamic")
    parser.add_argument("--appearance-metric", choices=("histogram", "clip", "lpips"), default="histogram")
    parser.add_argument("--trusted-anchor-frames", type=int, default=2)
    parser.add_argument("--memory-frames", type=int, default=1)
    parser.add_argument("--boundary-reference", type=int, choices=(0, 1), default=1)
    parser.add_argument("--max-visual-references", type=int, default=4)
    args = parser.parse_args()
    if args.window_frames <= 0:
        parser.error("--window-frames must be positive")
    if not 0 < args.overlap_frames < args.window_frames:
        parser.error("--overlap-frames must be positive and smaller than --window-frames")
    if args.overlap_frames % 17 and not is_masked_av_context_frames(args.overlap_frames):
        parser.error("--overlap-frames must be a legacy 17-frame multiple or exact Masked AV head")
    if args.trusted_anchor_frames < 0 or args.trusted_anchor_frames > 4:
        parser.error("--trusted-anchor-frames must be between 0 and 4")
    if args.memory_frames < 0 or args.memory_frames > 4:
        parser.error("--memory-frames must be between 0 and 4")
    if args.max_visual_references < 1 or args.max_visual_references > 16:
        parser.error("--max-visual-references must be between 1 and 16")
    requested_modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    allowed = {"no-bank", "static", "dynamic"}
    if not requested_modes or not set(requested_modes).issubset(allowed):
        parser.error(f"--modes must be a comma-separated subset of {sorted(allowed)}")
    args.modes = requested_modes
    args.checkpoint = _resolve_checkpoint(args.checkpoint)
    return args


def _run_mode(pipeline, args, plan, mode):
    config = H3ContinuationConfig(
        requested_window_frames=args.window_frames,
        overlap_frames=args.overlap_frames,
    )
    kwargs = {
        "continuation_mode": "masked-av-v14",
        "prefer_latent_handoff": True,
    }
    if mode == "no-bank":
        kwargs.update(global_reference_frames=0, boundary_reference_frames=0)
    elif mode == "static":
        kwargs.update(
            global_reference_frames=args.trusted_anchor_frames,
            boundary_reference_frames=args.boundary_reference,
        )
    else:
        kwargs.update(
            global_reference_frames=0,
            boundary_reference_frames=0,
            appearance_memory_config=H3AppearanceMemoryConfig(
                mode="dynamic",
                trusted_anchor_frames=args.trusted_anchor_frames,
                memory_frame_budget=args.memory_frames,
                boundary_reference_frames=bool(args.boundary_reference),
                max_visual_references=args.max_visual_references,
            ),
        )
    runner = H3ContinuationRunner(pipeline, config, model_identity=str(args.checkpoint), **kwargs)
    return runner.run(
        plan,
        base_seed=args.seed,
        pipeline_kwargs={
            "height": args.height,
            "width": args.width,
            "num_inference_steps": args.num_inference_steps,
            "cfg_scale": 1.0,
        },
    )


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("appearance drift evaluation requires CUDA")
    plan = load_h3_segment_plan(args.segment_plan)
    pipe = MiniMaxH3Pipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(
                path=[str(path) for path in sorted((args.h3_base / "text_encoder").glob("model*.safetensors"))],
                **_cpu_offload_vram_config(),
            ),
            ModelConfig(path=args.checkpoint, **_cpu_offload_vram_config()),
            ModelConfig(
                path=str(args.h3_base / "video_vae" / "source" / "model.safetensors"),
                **_cpu_offload_vram_config(),
            ),
            ModelConfig(
                path=str(args.h3_base / "audio_vae" / "model.safetensors"),
                **_cpu_offload_vram_config(),
            ),
        ],
        processor_config=ModelConfig(path=str(args.h3_base / "processor")),
        vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 4,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for mode in args.modes:
        row = {
            "mode": mode,
            "plan_id": plan.plan_id,
            "seed": args.seed,
            "window_frames": args.window_frames,
            "overlap_frames": args.overlap_frames,
            "output": None,
            "references_per_window": [],
            "appearance_memory_metadata": None,
            "appearance_metric": None,
            "error": None,
        }
        try:
            result = _run_mode(pipe, args, plan, mode)
            output = args.output_dir / f"{mode}.mp4"
            write_video_audio(result.video, result.audio, str(output), fps=24, audio_sample_rate=32_000)
            first_frames = result.state.records[0].window.resolved_video_frames
            reference_frames = result.video[:first_frames]
            target_frames = result.video[first_frames:]
            row.update(
                output=str(output),
                references_per_window=[
                    record.appearance_memory_frames
                    for record in result.state.records[1:]
                ],
                appearance_memory_metadata=result.manifest.appearance_memory_metadata,
                appearance_metric=format_appearance_metric(
                    reference_frames,
                    target_frames,
                    metric=args.appearance_metric,
                ),
            )
        except Exception as error:
            row["error"] = str(error)
        rows.append(row)
    report_path = args.report or args.output_dir / "appearance_drift_report.json"
    write_appearance_drift_report(
        report_path,
        rows,
        metadata={
            "checkpoint": str(args.checkpoint),
            "appearance_metric": args.appearance_metric,
            "segment_plan": str(args.segment_plan),
        },
    )
    print(json.dumps({"report": str(report_path), "rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
