"""Run paired base/LoRA continuation ablation reports for MiniMax-H3.

The CLI is GPU-only for actual generation.  ``--validate-only`` plans and
writes a deferred-validation manifest without loading a model, so CI can check
the evaluation topology on CPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from diffsynth.metrics.h3_continuation_lora import (
    ContinuationEvaluationCase,
    RegressionGates,
    run_paired_continuation_evaluation,
    write_ablation_report,
)
from diffsynth.pipelines.minimax_h3_continuation import (
    H3ContinuationConfig,
    H3ContinuationRunner,
    load_h3_segment_plan,
    write_continuation_evaluation,
)
from diffsynth.utils.continuation_lora import apply_continuation_lora


def _float_list(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def _int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def _vram_config() -> dict:
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


def _load_pipeline(checkpoint: str, h3_base: Path, *, lora_path: str | None, lora_scale: float) -> MiniMaxH3Pipeline:
    from diffsynth.core import ModelConfig
    from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline

    pipe = MiniMaxH3Pipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(path=[str(path) for path in sorted((h3_base / "text_encoder").glob("model*.safetensors"))], **_vram_config()),
            ModelConfig(path=checkpoint, **_vram_config()),
            ModelConfig(path=str(h3_base / "video_vae" / "source" / "model.safetensors"), **_vram_config()),
            ModelConfig(path=str(h3_base / "audio_vae" / "model.safetensors"), **_vram_config()),
        ],
        processor_config=ModelConfig(path=str(h3_base / "processor")),
        vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 4,
    )
    if lora_path and lora_scale > 0:
        apply_continuation_lora(pipe, lora_path, lora_scale)
    return pipe


def _parse_args():
    parser = argparse.ArgumentParser(description="Paired continuation LoRA evaluation.")
    parser.add_argument("--segment-plan", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True, help="H3 transformer checkpoint or shard directory.")
    parser.add_argument("--h3-base", type=Path, required=True)
    parser.add_argument("--lora", type=Path, default=None)
    parser.add_argument("--lora-scales", type=str, default="0.0,1.0")
    parser.add_argument("--transition-weights", type=str, default="0.5")
    parser.add_argument("--suffix-weights", type=str, default="1.0")
    parser.add_argument("--overlap-frames", type=str, default="34")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-category", type=str, default="single-shot")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/h3_continuation_lora_evaluation"))
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--min-boundary-improvement", type=float, default=0.05)
    parser.add_argument("--max-video-regression", type=float, default=0.25)
    parser.add_argument("--max-audio-regression", type=float, default=0.25)
    parser.add_argument("--max-overall-regression", type=float, default=0.15)
    parser.add_argument("--validate-only", action="store_true", help="Plan and write deferred validation without loading a model.")
    return parser.parse_args()


def _build_cases(args) -> list[ContinuationEvaluationCase]:
    scales = _float_list(args.lora_scales)
    transitions = _float_list(args.transition_weights)
    suffixes = _float_list(args.suffix_weights)
    overlaps = _int_list(args.overlap_frames)
    cases = []
    for overlap in overlaps:
        for transition in transitions:
            for suffix in suffixes:
                for scale in scales:
                    if scale == 0:
                        cases.append(ContinuationEvaluationCase(
                            config_id=f"base_overlap{overlap}_t{transition}_s{suffix}",
                            method="base",
                            checkpoint=args.checkpoint,
                            lora_path=str(args.lora) if args.lora else None,
                            lora_scale=0.0,
                            transition_weight=transition,
                            suffix_weight=suffix,
                            overlap_frames=overlap,
                            seed=args.seed,
                            sample_category=args.sample_category,
                        ))
                    cases.append(ContinuationEvaluationCase(
                        config_id=f"lora{scale}_overlap{overlap}_t{transition}_s{suffix}",
                        method="lora",
                        checkpoint=args.checkpoint,
                        lora_path=str(args.lora) if args.lora else None,
                        lora_scale=scale,
                        transition_weight=transition,
                        suffix_weight=suffix,
                        overlap_frames=overlap,
                        seed=args.seed,
                        sample_category=args.sample_category,
                    ))
    return cases


def _write_deferred_validation(args, cases) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "planned_not_executed",
        "reason": "GPU, long-video, CP, and subjective quality validation deferred",
        "plan": str(args.segment_plan),
        "checkpoint": args.checkpoint,
        "h3_base": str(args.h3_base),
        "lora": str(args.lora) if args.lora else None,
        "cases": [case.to_dict() for case in cases],
        "commands": [
            "python examples/minimax_h3/model_evaluation/continuation_lora_evaluation.py "
            f"--segment-plan {args.segment_plan} --checkpoint {args.checkpoint} "
            f"--h3-base {args.h3_base} --lora {args.lora} --validate-only"
        ],
    }
    target = args.output_dir / "deferred_validation.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main():
    args = _parse_args()
    cases = _build_cases(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.validate_only:
        _write_deferred_validation(args, cases)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("continuation LoRA evaluation requires CUDA")
    if not args.lora:
        raise ValueError("--lora is required for GPU evaluation")
    plan = load_h3_segment_plan(args.segment_plan)
    reports = {}
    for case in cases:
        pipe = _load_pipeline(
            case.checkpoint,
            args.h3_base,
            lora_path=case.lora_path,
            lora_scale=case.lora_scale,
        )
        runner = H3ContinuationRunner(
            pipe,
            H3ContinuationConfig(
                requested_window_frames=plan.segments[0].requested_frames or 345,
                overlap_frames=case.overlap_frames,
            ),
            model_identity=case.checkpoint,
            continuation_mode="masked-av-v14",
            prefer_latent_handoff=True,
            global_latent_decode=True,
            manifest_directory=args.output_dir / case.config_id,
        )
        result = runner.run(
            plan,
            base_seed=case.seed,
            pipeline_kwargs={
                "height": args.height,
                "width": args.width,
                "num_inference_steps": args.num_inference_steps,
                "cfg_scale": 1.0,
            },
        )
        result.state.experiment_metadata = {
            **result.state.experiment_metadata,
            "evaluation_case": case.to_dict(),
        }
        report_path = args.output_dir / f"{case.config_id}.json"
        reports[case.config_id] = write_continuation_evaluation(result, report_path)
    for overlap in set(case.overlap_frames for case in cases):
        base_id = next(
            case.config_id
            for case in cases
            if case.method == "base" and case.overlap_frames == overlap
        )
        lora_ids = [
            case.config_id
            for case in cases
            if case.method == "lora" and case.lora_scale > 0 and case.overlap_frames == overlap
        ]
        for lora_id in lora_ids:
            write_ablation_report(
                {base_id: reports[base_id], lora_id: reports[lora_id]},
                args.output_dir / f"ablation_{base_id}__{lora_id}.json",
                base_config_id=base_id,
                regression_gates=RegressionGates(
                    min_boundary_improvement=args.min_boundary_improvement,
                    max_video_seam_regression=args.max_video_regression,
                    max_audio_seam_regression=args.max_audio_regression,
                    max_overall_quality_regression=args.max_overall_regression,
                ),
            )
    print(json.dumps({"output_dir": str(args.output_dir), "reports": len(reports)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
