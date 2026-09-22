"""CPU-safe seam and regression diagnostics for continuation LoRA.

The module deliberately keeps model loading outside its public functions.  A
caller can feed either a real ``H3ContinuationResult`` from a CUDA run or a
synthetic result built by a fake pipeline during unit tests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from diffsynth.pipelines.minimax_h3_continuation import (
    H3ContinuationResult,
    evaluate_continuation_joins,
)


@dataclass(frozen=True)
class ContinuationEvaluationCase:
    """One row in a reproducible ablation matrix."""

    config_id: str
    method: str = "lora"
    checkpoint: str = ""
    lora_path: str | None = None
    lora_scale: float = 0.0
    transition_weight: float = 0.5
    suffix_weight: float = 1.0
    overlap_frames: int = 34
    seed: int = 42
    sample_category: str = "default"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RegressionGates:
    """Thresholds used to fail a LoRA ablation that overfits the seam."""

    min_boundary_improvement: float = 0.05
    max_video_seam_regression: float = 0.25
    max_audio_seam_regression: float = 0.25
    max_overall_quality_regression: float = 0.15


def frame_to_tensor(frame: Any) -> torch.Tensor | None:
    """Convert common decoded frame containers to a float CPU tensor."""
    if isinstance(frame, torch.Tensor):
        return frame.detach().float().cpu()
    try:
        if hasattr(frame, "convert") and hasattr(frame, "getpixel"):
            import numpy as np
            frame = np.asarray(frame)
        return torch.from_numpy(np.asarray(frame).copy()).float()
    except (TypeError, ValueError, RuntimeError):
        return None


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def brightness_color_difference(first: Any, second: Any) -> dict[str, float | None]:
    first_tensor = frame_to_tensor(first)
    second_tensor = frame_to_tensor(second)
    if first_tensor is None or second_tensor is None or first_tensor.shape != second_tensor.shape:
        return {
            "mean_absolute_difference": None,
            "luminance_absolute_difference": None,
            "channel_mean_absolute_difference": None,
        }
    diff = (first_tensor - second_tensor).abs()
    channel_mean = diff.mean(dim=tuple(range(1, diff.ndim)))
    if isinstance(channel_mean, torch.Tensor):
        channel_mean = channel_mean.tolist()
    return {
        "mean_absolute_difference": float(diff.mean()),
        "luminance_absolute_difference": float(diff.mean(dim=tuple(range(1, diff.ndim))).mean()),
        "channel_mean_absolute_difference": channel_mean,
    }


def person_box_displacement(
    before_frames: Sequence[Any],
    after_frames: Sequence[Any],
    detector: Callable[[Any], Mapping[str, float] | tuple[float, float, float, float] | None] | None = None,
) -> dict[str, Any]:
    """Report displacement of the best person box when a detector is available."""
    if detector is None:
        return {"status": "unavailable", "reason": "no person-box detector configured"}
    if not before_frames or not after_frames:
        return {"status": "unavailable", "reason": "missing frames"}
    before = detector(before_frames[-1])
    after = detector(after_frames[0])
    if before is None or after is None:
        return {"status": "not_detected"}
    if isinstance(before, Mapping):
        before_box = (before["x"], before["y"], before["w"], before["h"])
        after_box = (after["x"], after["y"], after["w"], after["h"])
    else:
        before_box = tuple(before)
        after_box = tuple(after)
    displacement = (
        abs(after_box[0] - before_box[0]),
        abs(after_box[1] - before_box[1]),
    )
    return {
        "status": "detected",
        "before_box": list(before_box),
        "after_box": list(after_box),
        "center_displacement": displacement,
        "center_displacement_norm": float(np.hypot(*displacement)),
    }


def motion_difference(
    before_frames: Sequence[Any],
    after_frames: Sequence[Any],
    *,
    latent_velocity_difference: float | None = None,
    optical_flow: Callable[[Sequence[Any], Sequence[Any]], float] | None = None,
) -> dict[str, Any]:
    """Estimate motion continuity using frames or an injected flow function."""
    if optical_flow is not None:
        flow = float(optical_flow(before_frames, after_frames))
        return {
            "optical_flow_difference": flow,
            "frame_mean_absolute_difference": None,
            "latent_velocity_difference": latent_velocity_difference,
        }
    before_tensors = [frame_to_tensor(frame) for frame in before_frames]
    after_tensors = [frame_to_tensor(frame) for frame in after_frames]
    before_tensors = [item for item in before_tensors if item is not None]
    after_tensors = [item for item in after_tensors if item is not None]
    if len(before_tensors) < 2 or len(after_tensors) < 2:
        return {
            "optical_flow_difference": None,
            "frame_mean_absolute_difference": None,
            "latent_velocity_difference": latent_velocity_difference,
        }
    before_delta = (before_tensors[-1] - before_tensors[-2]).abs().mean()
    after_delta = (after_tensors[1] - after_tensors[0]).abs().mean()
    return {
        "optical_flow_difference": None,
        "frame_mean_absolute_difference": float((after_delta - before_delta).abs()),
        "latent_velocity_difference": latent_velocity_difference,
    }


def flicker_statistics(frames: Sequence[Any], *, high_jump_abs: float = 0.02) -> dict[str, Any]:
    tensors = [frame_to_tensor(frame) for frame in frames]
    tensors = [item for item in tensors if item is not None]
    if len(tensors) < 2:
        return {
            "mean_absolute_difference": None,
            "std_absolute_difference": None,
            "high_jump_fraction": None,
            "first_clip_mean_absolute_difference": None,
        }
    deltas = [
        (tensors[index] - tensors[index - 1]).abs().mean().item()
        for index in range(1, len(tensors))
    ]
    first_clip = deltas[:17] if len(deltas) >= 17 else deltas
    return {
        "mean_absolute_difference": float(np.mean(deltas)),
        "std_absolute_difference": float(np.std(deltas)),
        "high_jump_fraction": float(np.mean([value > high_jump_abs for value in deltas])),
        "first_clip_mean_absolute_difference": float(np.mean(first_clip)),
    }


def audio_energy_difference(before: torch.Tensor, after: torch.Tensor) -> dict[str, float | None]:
    before_float = before.float()
    after_float = after.float()
    if before_float.numel() == 0 or after_float.numel() == 0:
        return {"rms_before": None, "rms_after": None, "rms_absolute_difference": None}
    rms_before = float(before_float.square().mean().sqrt())
    rms_after = float(after_float.square().mean().sqrt())
    return {
        "rms_before": rms_before,
        "rms_after": rms_after,
        "rms_absolute_difference": abs(rms_after - rms_before),
        "loudness_absolute_difference": abs(20 * np.log10(max(rms_after, 1e-8)) - 20 * np.log10(max(rms_before, 1e-8))),
    }


def spectral_discontinuity(before: torch.Tensor, after: torch.Tensor) -> float | None:
    before_float = before.float()
    after_float = after.float()
    if before_float.numel() == 0 or after_float.numel() == 0:
        return None
    before_spectrum = torch.fft.rfft(before_float, dim=-1).abs().mean(dim=0)
    after_spectrum = torch.fft.rfft(after_float, dim=-1).abs().mean(dim=0)
    return float((after_spectrum - before_spectrum).abs().mean())


def phase_waveform_jump(before: torch.Tensor, after: torch.Tensor) -> dict[str, float | None]:
    before_float = before.float()
    after_float = after.float()
    if before_float.numel() == 0 or after_float.numel() == 0:
        return {"phase_jump": None, "waveform_absolute_difference": None}
    before_complex = torch.fft.rfft(before_float, dim=-1).mean(dim=0)
    after_complex = torch.fft.rfft(after_float, dim=-1).mean(dim=0)
    phase_delta = torch.angle(after_complex) - torch.angle(before_complex)
    magnitude = (before_complex.abs() + after_complex.abs()) * 0.5
    phase_jump = (
        (torch.atan2(torch.sin(phase_delta), torch.cos(phase_delta)).abs() * magnitude).sum()
        / magnitude.sum().clamp_min(torch.finfo(magnitude.dtype).eps)
    )
    seam_length = min(before_float.shape[-1], after_float.shape[-1])
    waveform_jump = (
        (after_float[..., :seam_length] - before_float[..., -seam_length:]).abs().mean()
        if seam_length
        else None
    )
    return {"phase_jump": float(phase_jump), "waveform_absolute_difference": _safe_float(waveform_jump)}


def estimate_av_boundary_offset_ms(
    video_frames: Sequence[Any],
    audio: torch.Tensor,
    *,
    sample_rate: int = 32000,
    video_fps: int = 24,
    max_offset_frames: int = 5,
) -> float | None:
    """Estimate audio-video boundary offset from onset-energy correlation."""
    video_energy = []
    for frame in video_frames:
        tensor = frame_to_tensor(frame)
        video_energy.append(float(tensor.abs().mean()) if tensor is not None else None)
    video_energy = [value for value in video_energy if value is not None]
    if len(video_energy) < 2 or audio is None or audio.numel() == 0:
        return None
    audio_energy = audio.float().abs().mean(dim=0)
    frame_samples = sample_rate // video_fps
    audio_frames = [
        float(audio_energy[start:start + frame_samples].mean())
        for start in range(0, len(audio_energy) - frame_samples + 1, frame_samples)
    ]
    if len(audio_frames) < len(video_energy):
        audio_frames += [0.0] * (len(video_energy) - len(audio_frames))
    length = min(len(video_energy), len(audio_frames))
    video_np = np.asarray(video_energy[:length], dtype=np.float32)
    audio_np = np.asarray(audio_frames[:length], dtype=np.float32)
    video_np = (video_np - video_np.mean()) / max(video_np.std(), 1e-8)
    audio_np = (audio_np - audio_np.mean()) / max(audio_np.std(), 1e-8)
    best_offset = 0
    best_correlation = -1.0
    for offset in range(-max_offset_frames, max_offset_frames + 1):
        if offset < 0:
            shifted_video = video_np[-offset:]
            shifted_audio = audio_np[:len(shifted_video)]
        elif offset > 0:
            shifted_video = video_np[:-offset]
            shifted_audio = audio_np[offset:]
        else:
            shifted_video = video_np
            shifted_audio = audio_np
        if len(shifted_video) < 2:
            continue
        correlation = float(np.corrcoef(shifted_video, shifted_audio)[0, 1])
        if correlation > best_correlation:
            best_correlation = correlation
            best_offset = offset
    return best_offset * 1000.0 / video_fps


def augment_seam_report(result: H3ContinuationResult, *, person_box_detector=None, optical_flow=None) -> dict[str, Any]:
    """Extend the existing continuation report with task-specific diagnostics."""
    report = evaluate_continuation_joins(result)
    if result.video is None:
        return report
    video = list(result.video)
    audio = result.audio
    for join in report.get("joins", []):
        frame_boundary = int(join["video_frame"])
        if 0 < frame_boundary < len(video):
            before_frames = video[max(0, frame_boundary - 17):frame_boundary]
            after_frames = video[frame_boundary:min(len(video), frame_boundary + 34)]
            join["brightness_color_difference"] = brightness_color_difference(
                video[frame_boundary - 1], video[frame_boundary]
            )
            join["person_box"] = person_box_displacement(
                before_frames, after_frames, detector=person_box_detector
            )
            join["motion_difference"] = motion_difference(
                before_frames,
                after_frames,
                latent_velocity_difference=None,
                optical_flow=optical_flow,
            )
            join["flicker"] = flicker_statistics(after_frames)
        sample_boundary = int(join.get("audio_sample") or 0)
        if audio is not None and 0 < sample_boundary < audio.shape[-1]:
            radius = min(round(0.05 * int(result.state.config.audio_sample_rate)), sample_boundary, audio.shape[-1] - sample_boundary)
            before_audio = audio[..., sample_boundary - radius:sample_boundary]
            after_audio = audio[..., sample_boundary:sample_boundary + radius]
            join["audio_energy"] = audio_energy_difference(before_audio, after_audio)
            join["audio_spectral_discontinuity"] = spectral_discontinuity(before_audio, after_audio)
            join["audio_phase_waveform"] = phase_waveform_jump(before_audio, after_audio)
            join["crossfade_ms"] = float(result.state.config.audio_crossfade_ms)
            join["av_boundary_offset_ms"] = estimate_av_boundary_offset_ms(
                video[max(0, frame_boundary - 17):min(len(video), frame_boundary + 17)],
                audio[max(0, sample_boundary - radius):min(audio.shape[-1], sample_boundary + radius)],
                sample_rate=int(result.state.config.audio_sample_rate),
            )
    return report


def run_paired_continuation_evaluation(
    plan: Any,
    base_case: ContinuationEvaluationCase,
    lora_case: ContinuationEvaluationCase,
    runner_factory: Callable[[ContinuationEvaluationCase], Any],
    *,
    pipeline_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Run base and LoRA on the same plan/seed policy and return paired reports."""
    if base_case.seed != lora_case.seed:
        raise ValueError("paired base/LoRA cases must use the same seed policy")
    if base_case.sample_category != lora_case.sample_category:
        raise ValueError("paired base/LoRA cases must use the same sample category")
    if base_case.overlap_frames != lora_case.overlap_frames:
        raise ValueError("paired base/LoRA cases must use the same overlap")
    pipeline_kwargs = dict(pipeline_kwargs or {})
    base_runner = runner_factory(base_case)
    lora_runner = runner_factory(lora_case)
    base_result = base_runner.run(plan, base_seed=base_case.seed, pipeline_kwargs=pipeline_kwargs)
    lora_result = lora_runner.run(plan, base_seed=lora_case.seed, pipeline_kwargs=pipeline_kwargs)
    return {
        "base": augment_seam_report(base_result),
        "lora": augment_seam_report(lora_result),
    }


def check_regression_gates(
    paired: Mapping[str, Mapping[str, Any]],
    *,
    base_name: str = "base",
    lora_name: str = "lora",
    gates: RegressionGates = RegressionGates(),
) -> dict[str, Any]:
    """Flag LoRA improvements that are outweighed by quality regressions."""
    if base_name not in paired or lora_name not in paired:
        raise ValueError("regression gates require both base and LoRA reports")
    base = paired[base_name]
    lora = paired[lora_name]
    base_joins = base.get("joins", [])
    lora_joins = lora.get("joins", [])
    if len(base_joins) != len(lora_joins):
        raise ValueError("paired reports must contain the same number of joins")
    rows = []
    for index, (base_join, lora_join) in enumerate(zip(base_joins, lora_joins)):
        base_video = _safe_float(base_join.get("video_mean_absolute_difference"))
        lora_video = _safe_float(lora_join.get("video_mean_absolute_difference"))
        base_audio = _safe_float(base_join.get("audio_energy_jump"))
        lora_audio = _safe_float(lora_join.get("audio_energy_jump"))
        base_motion = _safe_float(base_join.get("motion_difference", {}).get("frame_mean_absolute_difference"))
        lora_motion = _safe_float(lora_join.get("motion_difference", {}).get("frame_mean_absolute_difference"))
        improvements = []
        regressions = []
        if base_video is not None and lora_video is not None:
            improvement = (base_video - lora_video) / max(base_video, 1e-8)
            improvements.append(improvement)
            regressions.append(-improvement)
        if base_audio is not None and lora_audio is not None:
            improvement = (base_audio - lora_audio) / max(base_audio, 1e-8)
            improvements.append(improvement)
            regressions.append(-improvement)
        if base_motion is not None and lora_motion is not None:
            improvement = (base_motion - lora_motion) / max(base_motion, 1e-8)
            improvements.append(improvement)
            regressions.append(-improvement)
        passed = bool(improvements) and all(
            improvement >= gates.min_boundary_improvement
            for improvement in improvements
        )
        if regressions:
            passed = passed and max(regressions) <= gates.max_video_seam_regression
        base_overall = _safe_float(
            base.get("video_metrics", {}).get("sampled_motion_mean_absolute_difference")
        )
        lora_overall = _safe_float(
            lora.get("video_metrics", {}).get("sampled_motion_mean_absolute_difference")
        )
        if base_overall is not None and lora_overall is not None and base_overall > 0:
            overall_regression = (lora_overall - base_overall) / base_overall
            if overall_regression > gates.max_overall_quality_regression:
                passed = False
        rows.append({
            "segment_index": base_join.get("segment_index"),
            "boundary_improvements": improvements,
            "boundary_regressions": regressions,
            "passed": passed,
            "reason": None if passed else "LoRA did not improve the seam or exceeded the regression threshold",
        })
    passed = all(row["passed"] for row in rows)
    return {"passed": passed, "gates": asdict(gates), "rows": rows}


def write_ablation_report(
    reports: Mapping[str, Mapping[str, Any]],
    output_path: str | Path,
    *,
    base_config_id: str | None = None,
    regression_gates: RegressionGates | None = None,
) -> dict[str, Any]:
    """Write a machine-readable JSON report and a readable Markdown summary."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "plan_id": next(iter(reports.values())).get("plan_id") if reports else None,
        "base_config_id": base_config_id,
        "runs": reports,
    }
    if regression_gates is not None and base_config_id is not None:
        paired = {
            "base": reports[base_config_id],
            "lora": next(
                report
                for config_id, report in reports.items()
                if config_id != base_config_id
            ),
        }
        summary["regression"] = check_regression_gates(
            paired, gates=regression_gates
        )
    destination.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown = _ablation_markdown(summary)
    destination.with_suffix(".md").write_text(markdown, encoding="utf-8")
    return summary


def _ablation_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Continuation LoRA Ablation",
        "",
        f"- Plan: `{summary.get('plan_id')}`",
        f"- Base config: `{summary.get('base_config_id')}`",
        "",
        "| Config | Method | Scale | Joins | Seam video | Seam audio | Passed |",
        "|---|---|---|---|---|---|---|",
    ]
    for config_id, report in summary.get("runs", {}).items():
        config = report.get("configuration", {})
        joins = report.get("joins", [])
        seam_video = ", ".join(
            str(join.get("video_mean_absolute_difference"))
            for join in joins[:3]
        )
        seam_audio = ", ".join(
            str(join.get("audio_energy_jump"))
            for join in joins[:3]
        )
        passed = summary.get("regression", {}).get("passed")
        lines.append(
            f"| {config_id} | {config.get('method', '')} | "
            f"{config.get('lora_scale', 0)} | {len(joins)} | "
            f"{seam_video} | {seam_audio} | {passed} |"
        )
    lines.append("")
    regression = summary.get("regression")
    if regression is not None:
        lines.append(f"Regression gates passed: **{regression.get('passed')}**")
        lines.append("")
    return "\n".join(lines)
