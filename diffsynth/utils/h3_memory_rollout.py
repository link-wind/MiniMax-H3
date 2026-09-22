"""Inference-side long-horizon memory for H3 continuation rollouts.

Training conditions each window on two memory slots built by
``diffsynth.utils.continuation_lora.encode_memory_slot``:

* **STM** -- the ``stm_frames`` frames immediately before the window, encoded as
  an independent ``17n+5`` VAE clip and placed flush against the window
  (``lead_steps=None``).
* **LTM** -- the first ``ltm_frames`` frames of the sequence, encoded the same
  way but pinned to a constant ``lead_steps`` lead, so its temporal distance does
  not grow with rollout length.

Both slots are dropped rather than faked when their source frames do not exist
or would overlap the window: STM on the sequence head, LTM while the window
still covers the sequence head.

A rollout has to rebuild those two slots from frames *it generated itself*.  Any
other source -- the reference video, or the previous window's decoded tail --
is a different conditioning signal than the one the model was trained on, which
is exactly the exposure bias a rollout exists to measure.  ``H3MemorySlotBuffer``
retains just enough of the decoded timeline to reproduce the training cut frame
for frame, and re-encodes it through the window's own VAE.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch

from .continuation_lora import DEFAULT_LTM_LEAD_STEPS, resolve_h3_video_frames


#: Slot order is part of the conditioning contract: training packs ``stm``
#: before ``ltm``, and the packed sequence lays slots out in that same order.
MEMORY_SLOT_ORDER = ("stm", "ltm")


@dataclass(frozen=True)
class H3MemoryRolloutPlan:
    """Which long-horizon slots a rollout rebuilds, and where they sit."""

    stm_frames: int | None = None
    ltm_frames: int | None = None
    ltm_lead_steps: int = DEFAULT_LTM_LEAD_STEPS

    def __post_init__(self) -> None:
        for name, frames in (("stm_frames", self.stm_frames), ("ltm_frames", self.ltm_frames)):
            if frames is None:
                continue
            frames = int(frames)
            if frames <= 0:
                raise ValueError(f"{name} must be positive or None, got {frames}")
            if resolve_h3_video_frames(frames) != frames:
                raise ValueError(f"{name} must satisfy 17n+5, got {frames}")
        if int(self.ltm_lead_steps) < 0:
            raise ValueError(f"ltm_lead_steps must be >= 0, got {self.ltm_lead_steps}")

    @property
    def enabled(self) -> bool:
        return bool(self.stm_frames) or bool(self.ltm_frames)

    def slot_names(self) -> tuple[str, ...]:
        names = []
        if self.stm_frames:
            names.append("stm")
        if self.ltm_frames:
            names.append("ltm")
        return tuple(names)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stm_frames": None if self.stm_frames is None else int(self.stm_frames),
            "ltm_frames": None if self.ltm_frames is None else int(self.ltm_frames),
            "ltm_lead_steps": int(self.ltm_lead_steps),
        }


class H3MemorySlotBuffer:
    """Retain the decoded frames the trained STM/LTM slots need, and rebuild them.

    The buffer holds two disjoint slices of the rollout timeline: the sequence
    head (for LTM) and a trailing window of recent frames (for STM).  Nothing
    else is kept, because the slots only ever reach back ``stm_frames`` frames
    before the next window start.

    Two properties matter for framing the evidence a rollout produces:

    * the slots are rebuilt from generated frames, never from the reference
      video, so a slot's error is the rollout's own drift;
    * a slot is *omitted* exactly when training omitted it, so the conditioning
      distribution at inference keeps the same support as at training time.
    """

    def __init__(
        self,
        plan: H3MemoryRolloutPlan,
        *,
        overlap_frames: int = 0,
        retain_frames: int | None = None,
        observer: Callable[[str, torch.Tensor], None] | None = None,
    ) -> None:
        if not isinstance(plan, H3MemoryRolloutPlan):
            raise TypeError(f"plan must be H3MemoryRolloutPlan, got {type(plan).__name__}")
        self.plan = plan
        self.overlap_frames = int(overlap_frames)
        if self.overlap_frames < 0:
            raise ValueError(f"overlap_frames must be >= 0, got {self.overlap_frames}")
        # The next window starts at ``emitted - overlap``, and its STM reaches a
        # further ``stm_frames`` back, so that is the true retention lower bound.
        # A generous margin costs nothing: retained frames are references to
        # frames the rollout already holds, not copies.
        floor = int(plan.stm_frames or 0) + self.overlap_frames
        self._retain_frames = int(retain_frames) if retain_frames is not None else max(2 * floor + 8, 64)
        if self._retain_frames <= floor:
            raise ValueError(
                f"retain_frames={self._retain_frames} cannot cover stm_frames={plan.stm_frames} "
                f"plus overlap_frames={self.overlap_frames}"
            )
        self._head: list[Any] = []
        self._tail: deque = deque()
        self._tail_start = 0
        self._emitted = 0
        self._ltm_latents: torch.Tensor | None = None
        self._stm_encodes = 0
        self._ltm_encodes = 0
        # An observer sees every slot exactly as the model will, so a caller that
        # needs the produced content (a transplant target, a report) never has to
        # re-implement slot encoding and risk disagreeing with the model path.
        self._observer = observer

    # ------------------------------------------------------------------ state
    @property
    def emitted_frames(self) -> int:
        return self._emitted

    @property
    def head_frames(self) -> int:
        return len(self._head)

    def observe(self, frames: Sequence[Any], *, start_frame: int | None = None) -> None:
        """Append newly decoded frames that cover the next slice of the timeline.

        ``start_frame`` is the global timeline index of ``frames[0]``.  It must be
        contiguous with everything observed so far, otherwise a STM slot would be
        cut out of a gapped timeline that never existed during training.
        """
        frames = list(frames)
        if start_frame is None:
            start_frame = self._emitted
        start_frame = int(start_frame)
        if start_frame < 0:
            raise ValueError(f"start_frame must be >= 0, got {start_frame}")
        if start_frame > self._emitted:
            raise ValueError(
                f"observed frames start at {start_frame} but only {self._emitted} frames exist so far"
            )
        if not frames:
            return
        end_frame = start_frame + len(frames)
        if end_frame <= self._emitted:
            # Fully redundant re-observation (e.g. a resumed window); keep the copy
            # already held instead of growing the tail twice.
            return
        if start_frame < self._emitted:
            frames = frames[self._emitted - start_frame:]
            start_frame = self._emitted

        wanted_head = int(self.plan.ltm_frames or 0)
        if wanted_head and len(self._head) < wanted_head:
            self._head.extend(frames[: wanted_head - len(self._head)])
        if not self._tail:
            self._tail_start = start_frame
        self._tail.extend(frames)
        self._emitted = end_frame
        self._prune()

    def _prune(self) -> None:
        keep = self._retain_frames
        while len(self._tail) > keep:
            self._tail.popleft()
            self._tail_start += 1

    def reset(self) -> None:
        self._head.clear()
        self._tail.clear()
        self._tail_start = 0
        self._emitted = 0
        self._ltm_latents = None
        self._stm_encodes = 0
        self._ltm_encodes = 0

    # ------------------------------------------------------------------ slots
    def slots(
        self,
        *,
        window_start_frame: int,
        window_end_frame: int,
        encode: Callable[[Sequence[Any]], torch.Tensor],
    ) -> list[dict[str, Any]]:
        """Rebuild the ordered slot list for one window position.

        ``window_start_frame`` / ``window_end_frame`` are the window's bounds on
        the global timeline; they decide which slots exist, exactly as
        ``encode_memory_slot`` did over the source media at training time.
        """
        window_start_frame = int(window_start_frame)
        window_end_frame = int(window_end_frame)
        if window_end_frame <= window_start_frame:
            raise ValueError("window_end_frame must exceed window_start_frame")
        ordered = []
        for name in MEMORY_SLOT_ORDER:
            if name == "stm":
                slot = self._stm_slot(window_start_frame, encode)
            else:
                slot = self._ltm_slot(window_start_frame, window_end_frame, encode)
            if slot is not None:
                ordered.append(slot)
        return ordered

    def _stm_slot(self, window_start_frame: int, encode) -> dict[str, Any] | None:
        frames = int(self.plan.stm_frames or 0)
        if frames <= 0:
            return None
        start_frame = window_start_frame - frames
        if start_frame < 0:
            # Sequence head: no frames precede the window, so there is no recency
            # slot to build.  Training drops it here too.
            return None
        block = self._frames_between(start_frame, window_start_frame)
        if block is None:
            return None
        tensor = self._encode(encode, block)
        self._stm_encodes += 1
        self._notify("stm", tensor)
        return {"name": "stm", "kind": "latent", "tensor": tensor, "lead_steps": None}

    def _ltm_slot(self, window_start_frame: int, window_end_frame: int, encode) -> dict[str, Any] | None:
        frames = int(self.plan.ltm_frames or 0)
        if frames <= 0:
            return None
        if window_start_frame < frames:
            # The slot's frames are still inside this window, so it would be a
            # clean block of the target rather than a memory.
            return None
        if len(self._head) < frames:
            return None
        if self._ltm_latents is None:
            self._ltm_latents = self._encode(encode, self._head[:frames])
            self._ltm_encodes += 1
            self._notify("ltm", self._ltm_latents)
        return {
            "name": "ltm",
            "kind": "latent",
            "tensor": self._ltm_latents,
            "lead_steps": int(self.plan.ltm_lead_steps),
        }

    def _notify(self, name: str, tensor: torch.Tensor) -> None:
        if self._observer is not None:
            self._observer(name, tensor)

    def _encode(self, encode, frames: Sequence[Any]) -> torch.Tensor:
        tensor = encode(list(frames))
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"memory slot encoder must return a torch.Tensor, got {type(tensor).__name__}")
        if tensor.dim() != 5:
            raise ValueError(f"memory slot encoder must return [1,24,t,h,w], got {tuple(tensor.shape)}")
        return tensor

    def _frames_between(self, start_frame: int, end_frame: int) -> list[Any] | None:
        tail_end = self._tail_start + len(self._tail)
        if start_frame < self._tail_start or end_frame > tail_end:
            return None
        offset = start_frame - self._tail_start
        return list(self._tail)[offset:offset + (end_frame - start_frame)]

    # --------------------------------------------------------------- metadata
    def provenance_metadata(self) -> dict[str, Any]:
        """JSON-safe record of what the rollout actually conditioned on."""
        return {
            "plan": self.plan.to_dict(),
            "slot_names": list(self.plan.slot_names()),
            "overlap_frames": int(self.overlap_frames),
            "retain_frames": int(self._retain_frames),
            "emitted_frames": int(self._emitted),
            "head_frames_held": len(self._head),
            "stm_encodes": int(self._stm_encodes),
            "ltm_encodes": int(self._ltm_encodes),
            "source": "rollout-generated-frames",
        }
