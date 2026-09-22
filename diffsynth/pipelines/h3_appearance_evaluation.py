"""Appearance-drift report helpers for H3 continuation evaluation."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import importlib.util
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _frame_gray(frame: Any, size: int = 8) -> np.ndarray | None:
    converter = getattr(frame, "convert", None)
    resizer = getattr(frame, "resize", None)
    if callable(converter) and callable(resizer):
        try:
            return np.asarray(converter("L").resize((size, size)), dtype=np.float32) / 255.0
        except Exception:
            return None
    if isinstance(frame, np.ndarray):
        try:
            if frame.ndim == 2:
                data = frame.astype(np.float32)
            else:
                data = frame.astype(np.float32).mean(axis=-1)
            return np.asarray(data, dtype=np.float32) / max(1.0, float(np.abs(data).max()))
        except Exception:
            return None
    return None


def histogram_appearance_score(
    reference_frames: Sequence[Any],
    target_frames: Sequence[Any],
    *,
    size: int = 8,
) -> float | None:
    """Return a deterministic histogram similarity in [0, 1]; 1 is identical."""
    refs = [_frame_gray(frame, size) for frame in reference_frames]
    targets = [_frame_gray(frame, size) for frame in target_frames]
    refs = [item for item in refs if item is not None]
    targets = [item for item in targets if item is not None]
    if not refs or not targets:
        return None
    distances = []
    for index, target in enumerate(targets):
        reference = refs[index % len(refs)]
        distances.append(float(np.mean(np.abs(reference - target))))
    return max(0.0, 1.0 - min(1.0, float(np.mean(distances))))


def _optional_metric_unavailable(metric: str, reason: str) -> dict[str, Any]:
    return {"metric": metric, "status": "unavailable", "reason": reason}


def clip_appearance_score(
    reference_frames: Sequence[Any],
    target_frames: Sequence[Any],
) -> dict[str, Any]:
    if importlib.util.find_spec("clip") is None:
        return _optional_metric_unavailable("clip", "clip package is not installed")
    try:
        import clip
        import torch

        model, preprocess = clip.load("ViT-B/32", device="cpu")
        model.eval()
        frames = [*reference_frames, *target_frames]
        if not frames:
            return _optional_metric_unavailable("clip", "no frames supplied")
        images = torch.stack([preprocess(frame) for frame in frames])
        with torch.no_grad():
            features = model.encode_image(images)
        features = features / features.norm(dim=-1, keepdim=True)
        ref_count = len(reference_frames)
        target_count = max(1, len(target_frames))
        scores = []
        for target_index in range(target_count):
            ref_index = target_index % ref_count
            score = float((features[ref_index] * features[ref_count + target_index]).sum())
            scores.append(score)
        return {
            "metric": "clip",
            "status": "measured",
            "value": float(np.mean(scores)),
            "details": {"frames": len(frames)},
        }
    except Exception as error:
        return _optional_metric_unavailable("clip", str(error))


def lpips_appearance_score(
    reference_frames: Sequence[Any],
    target_frames: Sequence[Any],
) -> dict[str, Any]:
    if importlib.util.find_spec("lpips") is None:
        return _optional_metric_unavailable("lpips", "lpips package is not installed")
    return _optional_metric_unavailable("lpips", "lpips evaluation requires aligned tensors; not enabled by default")


def format_appearance_metric(
    reference_frames: Sequence[Any],
    target_frames: Sequence[Any],
    *,
    metric: str = "histogram",
) -> dict[str, Any]:
    if metric == "histogram":
        value = histogram_appearance_score(reference_frames, target_frames)
        if value is None:
            return _optional_metric_unavailable("histogram", "frames do not expose a supported image representation")
        return {"metric": "histogram", "status": "measured", "value": value}
    if metric == "clip":
        return clip_appearance_score(reference_frames, target_frames)
    if metric == "lpips":
        return lpips_appearance_score(reference_frames, target_frames)
    raise ValueError(f"unsupported appearance metric: {metric}")


def write_appearance_drift_report(
    report_path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized_rows = []
    for row in rows:
        if is_dataclass(row):
            serialized_rows.append(asdict(row))
        else:
            serialized_rows.append(dict(row))
    report = {
        "schema_version": 1,
        "metadata": dict(metadata or {}),
        "rows": serialized_rows,
    }
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return path
