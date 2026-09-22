"""CPU-safe appearance memory bank for MiniMax-H3 continuation.

The bank deliberately works with H3's existing decoded-image reference
conditioning path.  It does not touch RoPE positions, global source indices,
KV caches, or the H3 DiT implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

import numpy as np


H3AppearanceMemoryMode = Literal["disabled", "static", "dynamic"]


@dataclass(frozen=True)
class H3AppearanceMemoryConfig:
    """Opt-in appearance memory settings for one continuation run."""

    mode: H3AppearanceMemoryMode = "disabled"
    trusted_anchor_frames: int = 0
    memory_frame_budget: int = 0
    boundary_reference_frames: bool = False
    max_visual_references: int = 4
    diversity_threshold: float = 0.08
    feature_size: int = 8
    candidate_frames_per_window: int = 3

    def __post_init__(self) -> None:
        if self.mode not in ("disabled", "static", "dynamic"):
            raise ValueError(f"appearance memory mode must be disabled/static/dynamic, got {self.mode!r}")
        if not isinstance(self.trusted_anchor_frames, int) or isinstance(self.trusted_anchor_frames, bool):
            raise ValueError("trusted_anchor_frames must be an integer")
        if not 0 <= self.trusted_anchor_frames <= 4:
            raise ValueError(f"trusted_anchor_frames must be between 0 and 4, got {self.trusted_anchor_frames}")
        if not isinstance(self.memory_frame_budget, int) or isinstance(self.memory_frame_budget, bool):
            raise ValueError("memory_frame_budget must be an integer")
        if not 0 <= self.memory_frame_budget <= 4:
            raise ValueError(f"memory_frame_budget must be between 0 and 4, got {self.memory_frame_budget}")
        if self.mode == "static" and self.memory_frame_budget:
            raise ValueError("static appearance memory mode cannot use memory_frame_budget")
        if self.mode == "dynamic" and self.trusted_anchor_frames < 1:
            raise ValueError("dynamic appearance memory mode requires at least one trusted anchor")
        if not isinstance(self.max_visual_references, int) or isinstance(self.max_visual_references, bool):
            raise ValueError("max_visual_references must be an integer")
        if not 1 <= self.max_visual_references <= 16:
            raise ValueError(f"max_visual_references must be between 1 and 16, got {self.max_visual_references}")
        if not 0.0 <= self.diversity_threshold <= 1.0:
            raise ValueError("diversity_threshold must be in [0, 1]")
        if self.feature_size <= 0:
            raise ValueError("feature_size must be positive")
        if self.candidate_frames_per_window <= 0:
            raise ValueError("candidate_frames_per_window must be positive")
        bank_count = self.trusted_anchor_frames + self.memory_frame_budget
        if self.boundary_reference_frames:
            bank_count += 1
        if bank_count > self.max_visual_references:
            raise ValueError(
                "appearance memory bank exceeds max_visual_references: "
                f"configured={bank_count}, max={self.max_visual_references}"
            )


def appearance_memory_config_from_legacy(
    global_reference_frames: int = 0,
    boundary_reference_frames: int = 0,
    *,
    explicit: H3AppearanceMemoryConfig | None = None,
) -> H3AppearanceMemoryConfig:
    """Map legacy runner flags to a bank config while allowing explicit overrides."""
    if explicit is not None:
        return explicit
    mode = "static" if global_reference_frames or boundary_reference_frames else "disabled"
    return H3AppearanceMemoryConfig(
        mode=mode,
        trusted_anchor_frames=int(global_reference_frames),
        memory_frame_budget=0,
        boundary_reference_frames=bool(boundary_reference_frames),
        max_visual_references=max(
            4,
            int(global_reference_frames) + int(bool(boundary_reference_frames)),
        ),
    )


@dataclass(frozen=True)
class H3AppearanceFrame:
    """One frame retained by the appearance memory bank."""

    frame: Any
    segment_index: int
    frame_index: int
    role: str = "memory"
    feature: Any = field(default=None, repr=False, compare=False)

    def as_reference(self) -> dict[str, Any]:
        return {"type": "image", "image": _copy_frame(self.frame)}

    def provenance(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "segment_index": self.segment_index,
            "frame_index": self.frame_index,
        }


def _copy_frame(frame: Any) -> Any:
    copier = getattr(frame, "copy", None)
    return frame.copy() if callable(copier) else frame


def _image_feature(frame: Any, size: int) -> np.ndarray | None:
    converter = getattr(frame, "convert", None)
    resizer = getattr(frame, "resize", None)
    if callable(converter) and callable(resizer):
        try:
            gray = converter("L").resize((size, size))
            return np.asarray(gray, dtype=np.float32) / 255.0
        except Exception:
            return None
    if isinstance(frame, np.ndarray):
        try:
            if frame.ndim == 2:
                data = frame.astype(np.float32)
            else:
                data = frame.astype(np.float32).mean(axis=-1)
            return np.asarray(data, dtype=np.float32) / max(255.0, float(np.abs(data).max()))
        except Exception:
            return None
    return None


def _feature_distance(left: np.ndarray | None, right: np.ndarray | None) -> float | None:
    if left is None or right is None or left.shape != right.shape:
        return None
    return float(np.mean(np.abs(left - right)))


def _candidate_indices(
    frame_count: int,
    *,
    segment_index: int,
    overlap_frames: int,
    candidate_count: int,
) -> list[int]:
    if frame_count <= 0 or candidate_count <= 0:
        return []
    start = 0
    end = frame_count
    if segment_index > 0 and 0 < overlap_frames < frame_count:
        start = overlap_frames
        end = frame_count - overlap_frames
    if end <= start:
        start = 0
        end = frame_count
    if end <= start:
        return []
    span = end - start
    count = min(candidate_count, span)
    if count <= 0:
        return []
    return [
        start + ((span * (index + 1)) // (count + 1))
        for index in range(count)
    ]


@dataclass
class H3AppearanceMemoryBank:
    """Maintains trusted anchors, selected memory frames, and a boundary frame."""

    config: H3AppearanceMemoryConfig
    anchors: list[H3AppearanceFrame] = field(default_factory=list)
    memory_frames: list[H3AppearanceFrame] = field(default_factory=list)
    boundary: H3AppearanceFrame | None = field(default=None, compare=False)
    initialized: bool = False

    def is_active(self) -> bool:
        return self.config.mode != "disabled"

    def initialize_anchors(
        self,
        video_frames: Sequence[Any],
        *,
        segment_index: int = 0,
    ) -> None:
        if self.initialized or not self.is_active():
            return
        self.initialized = True
        frame_count = len(video_frames)
        anchor_count = self.config.trusted_anchor_frames
        if anchor_count <= 0:
            return
        if frame_count < anchor_count:
            raise ValueError(
                "first window has fewer decoded frames than trusted_anchor_frames: "
                f"frames={frame_count}, anchors={anchor_count}"
            )
        feature_size = self.config.feature_size
        self.anchors = [
            H3AppearanceFrame(
                frame=_copy_frame(video_frames[(frame_count * (index + 1)) // (anchor_count + 1)]),
                segment_index=segment_index,
                frame_index=(frame_count * (index + 1)) // (anchor_count + 1),
                role="trusted",
                feature=_image_feature(
                    video_frames[(frame_count * (index + 1)) // (anchor_count + 1)],
                    feature_size,
                ),
            )
            for index in range(anchor_count)
        ]

    def update_from_window(
        self,
        video_frames: Sequence[Any],
        *,
        segment_index: int,
        overlap_frames: int = 0,
    ) -> None:
        if not self.is_active() or self.config.mode != "dynamic":
            return
        self.initialize_anchors(video_frames, segment_index=segment_index)
        budget = self.config.memory_frame_budget
        if budget <= 0:
            return
        candidates = _candidate_indices(
            len(video_frames),
            segment_index=segment_index,
            overlap_frames=overlap_frames,
            candidate_count=self.config.candidate_frames_per_window,
        )
        feature_size = self.config.feature_size
        for frame_index in candidates:
            if len(self.memory_frames) >= budget:
                break
            frame = video_frames[frame_index]
            feature = _image_feature(frame, feature_size)
            if self._is_near_duplicate(frame, feature):
                continue
            self.memory_frames.append(
                H3AppearanceFrame(
                    frame=_copy_frame(frame),
                    segment_index=segment_index,
                    frame_index=frame_index,
                    role="memory",
                    feature=feature,
                )
            )

    def set_boundary(
        self,
        frame: Any,
        *,
        segment_index: int,
        frame_index: int,
    ) -> None:
        if not self.is_active() or not self.config.boundary_reference_frames:
            return
        self.boundary = H3AppearanceFrame(
            frame=_copy_frame(frame),
            segment_index=segment_index,
            frame_index=frame_index,
            role="boundary",
            feature=_image_feature(frame, self.config.feature_size),
        )

    def _is_near_duplicate(self, frame: Any, feature: np.ndarray | None) -> bool:
        for existing in [*self.anchors, *self.memory_frames]:
            if feature is not None and existing.feature is not None:
                distance = _feature_distance(feature, existing.feature)
                if distance is not None and distance < self.config.diversity_threshold:
                    return True
            elif existing.frame == frame:
                return True
        return False

    def build_references(
        self,
        user_references: Sequence[Mapping[str, Any]] | None = None,
    ) -> list[dict[str, Any]] | None:
        if not self.is_active():
            return None
        continuation_references: list[dict[str, Any]] = []
        if self.config.boundary_reference_frames and self.boundary is not None:
            continuation_references.append(self.boundary.as_reference())
        continuation_references.extend(anchor.as_reference() for anchor in self.anchors)
        continuation_references.extend(memory.as_reference() for memory in self.memory_frames)
        if not continuation_references:
            return None
        if user_references is None:
            return continuation_references
        if isinstance(user_references, (list, tuple)):
            return [*user_references, *continuation_references]
        raise TypeError("references must be a list or tuple when appearance memory bank references are enabled")

    def active_reference_count(self) -> int:
        count = len(self.anchors) + len(self.memory_frames)
        if self.config.boundary_reference_frames and self.boundary is not None:
            count += 1
        return count

    def provenance_metadata(self) -> dict[str, Any]:
        return {
            "mode": self.config.mode,
            "trusted_anchor_frames": len(self.anchors),
            "memory_frame_budget": len(self.memory_frames),
            "boundary_reference_frames": self.config.boundary_reference_frames,
            "anchors": [anchor.provenance() for anchor in self.anchors],
            "memory": [memory.provenance() for memory in self.memory_frames],
            "boundary": self.boundary.provenance() if self.boundary is not None else None,
        }
