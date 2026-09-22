#!/usr/bin/env python3
"""Generative-rollout ablation for the H3 long-horizon memory slots.

Why a rollout is the only test that settles the question
-------------------------------------------------------
The teacher-forced ablation (``eval_memory_ablation.py``) showed that removing
the memory slots hurts, but that swapping their *content* hurts just as much --
the model was reading "a slot exists", not "this slot".  That result is an
artifact of the objective, not of the model: flow matching feeds
``x_t = (1 - t) * x_0 + t * noise`` and ``t`` is drawn uniformly, so roughly half
of the target's information is still sitting in the input.  The model can follow
``x_t`` to the answer and treat the slots as redundant.

A rollout removes the leak.  The target no longer exists, the slots are rebuilt
from the rollout's own frames, and cross-shot consistency becomes a quantity that
accumulates over the whole sequence.  Three things become measurable that teacher
forcing cannot see: whether a slot is the only carrier of identity once the
target is gone, whether the *content* of the slot steers the output (the causal
test), and whether the slots' own error compounds across shots.

Arms
----
both   stm + ltm                  the trained configuration
stm    stm only                   near-history recency
ltm    ltm only                   sequence-head identity anchor
none   no memory at all           the trained LoRA without slots
swap   another sequence's slots   content transplant: right distribution, wrong content
base   no LoRA, both slots        separates "LoRA learned continuation" from "slots work"

Swap content comes from the validation cache rather than from a second rollout:
those are real, in-distribution latent slots from other sequences, so the arm
varies content while holding distribution, geometry, and count fixed.

Prompt redundancy
-----------------
``--prompt-mode no-subject`` strips the ``<SUBJECT>`` block, which is where the
appearance identity is restated in words.  In that setting the LTM slot is the
only remaining source of identity, so a model that truly uses slot content must
diverge much further from the ``full`` condition than one that does not.  It is
the sharpest available version of the same causal question.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from diffsynth.pipelines.minimax_h3_continuation import (  # noqa: E402
    H3ContinuationConfig,
    H3ContinuationRunner,
    H3Segment,
    H3SegmentPlan,
    evaluate_continuation_joins,
)
from diffsynth.utils.h3_memory_rollout import H3MemoryRolloutPlan  # noqa: E402


ARMS: dict[str, dict[str, Any]] = {
    "both": {"stm": True, "ltm": True, "swap": False, "lora_scale": 1.0},
    "stm": {"stm": True, "ltm": False, "swap": False, "lora_scale": 1.0},
    "ltm": {"stm": False, "ltm": True, "swap": False, "lora_scale": 1.0},
    "none": {"stm": False, "ltm": False, "swap": False, "lora_scale": 1.0},
    "swap": {"stm": True, "ltm": True, "swap": True, "lora_scale": 1.0},
    "base": {"stm": True, "ltm": True, "swap": False, "lora_scale": 0.0},
}

#: Identity drift is referenced to the sequence head, so the head block is the
#: natural unit to compare windows against.
HEAD_REFERENCE_FRAMES = 39


class _SlotObserver:
    """Record every slot the runner built, as the model saw it.

    The ``both`` arm's own slots are how the swap arm learns the exact latent
    geometry a donor has to match, without a second rollout and without
    re-implementing the encode path.
    """

    def __init__(self):
        self.calls: list[tuple[str, torch.Tensor]] = []

    def __call__(self, name, tensor):
        self.calls.append((str(name), tensor.detach().clone()))

    def shapes(self) -> list[tuple[int, ...]]:
        return [tuple(tensor.shape) for _name, tensor in self.calls]


class _ReplaySlotEncoder:
    """Serve donor slot content for every slot the runner asks for.

    Slot count and order are the runner's business; this encoder only asserts the
    transplant is shape-compatible, because a slot is a raw latent block and
    cannot cross resolutions.
    """

    def __init__(self, donors: Sequence[torch.Tensor]):
        if not donors:
            raise ValueError("swap arm needs at least one donor slot")
        self.donors = [tensor.detach() for tensor in donors]
        self.index = 0

    def __call__(self, frames, *, height=None, width=None):
        # Donors arrive from CPU-loaded cache files; the packer aligns the device,
        # so this stays a pure content substitution.
        donor = self.donors[self.index % len(self.donors)]
        self.index += 1
        return donor.clone()


def _load_evaluation_module():
    """Reuse the existing paired-evaluation loader instead of duplicating it."""
    path = _HERE.parent / "model_evaluation" / "continuation_lora_evaluation.py"
    spec = importlib.util.spec_from_file_location("h3_continuation_lora_evaluation_under_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_args():
    parser = argparse.ArgumentParser(description="Generative-rollout memory-slot ablation.")
    parser.add_argument("--segment-plan", type=Path, required=True)
    parser.add_argument("--checkpoint", nargs="+", required=True,
                        help="H3 transformer shard file(s); a shard set is one checkpoint.")
    parser.add_argument("--h3-base", type=Path, required=True)
    parser.add_argument("--lora", type=Path, default=None,
                        help="Continuation LoRA used by every arm in the built-in table.")
    parser.add_argument(
        "--lora-arms", type=str, default=None,
        help=(
            "Replace the built-in arm table with a LoRA comparison, e.g. "
            "'base=,v5=/path/step-1271.safetensors,v9=/path/step-3123.safetensors'. "
            "Each entry is NAME=PATH[@SCALE]; an empty PATH is the un-adapted base model and "
            "SCALE defaults to 1.0.  Every arm under --lora-arms runs with memory completely "
            "off, so the comparison isolates what the checkpoints learned rather than what the "
            "memory injector supplies."
        ),
    )
    parser.add_argument("--donor-cache", type=Path, default=None,
                        help="Cache root or split directory supplying swap-arm slot content.")
    parser.add_argument("--donor-split", type=str, default="validation",
                        help="Split to draw donors from when --donor-cache holds shards.")
    parser.add_argument("--arms", type=str, default="both,stm,ltm,none,swap,base")
    parser.add_argument("--prompt-mode", choices=("full", "no-subject"), default="full")
    parser.add_argument("--stm-frames", type=int, default=39)
    parser.add_argument("--ltm-frames", type=int, default=39)
    parser.add_argument("--overlap-frames", type=int, default=None,
                        help="Cut length; masked-av-v14 needs an exact joint H3 head (39 default).")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    # 8, not the pipeline default of 50: the continuation LoRA was trained on top
    # of ``transformer_lora600_v4``, the distilled base whose references are
    # generated at 8 steps, so 50 samples it off its training distribution.
    parser.add_argument("--num-inference-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    # Numbers decide the ablation; the video decides whether the numbers mean what
    # they say.  Drift metrics cannot show whether a shot change is coherent, only
    # whether pixels moved, so the rollout is written out whenever it is asked for.
    parser.add_argument("--save-video", action="store_true",
                        help="Write each arm's assembled rollout next to its metrics.")
    parser.add_argument("--fps", type=int, default=24,
                        help="H3's frame rate; the plan's durations and the audio "
                             "raster (round(frames / 24 * 40) steps) both assume it.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/memory_rollout_ablation"))
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the planned arms and exit without loading a model.")
    return parser.parse_args()


def _parse_lora_arms(spec: str) -> dict[str, dict[str, Any]]:
    """Parse ``NAME=PATH[@SCALE],...`` into an arm table.

    These arms deliberately mirror the ``none`` arm: memory off, no donor
    transplant.  What varies between them is only the checkpoint, which is the
    point -- a comparison of two training runs has to hold the inference-time
    conditioning fixed, or the difference stops being attributable to training.
    """
    table: dict[str, dict[str, Any]] = {}
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        name, sep, rest = chunk.partition("=")
        name = name.strip()
        if not sep or not name:
            raise ValueError(f"--lora-arms entry needs NAME=PATH[@SCALE], got {chunk!r}")
        if name in table:
            raise ValueError(f"--lora-arms repeats arm {name!r}")
        path_text, _at, scale_text = rest.partition("@")
        path_text = path_text.strip()
        scale_text = scale_text.strip()
        if path_text:
            resolved = Path(path_text)
            if not resolved.is_file():
                raise ValueError(f"--lora-arms arm {name!r}: no such LoRA file {resolved}")
            # Absolute, so the recorded config still says which checkpoint ran even
            # though the eval is normally launched from a different working directory.
            path_text = str(resolved.resolve())
        table[name] = {
            "stm": False, "ltm": False, "swap": False,
            "lora_scale": float(scale_text) if scale_text else 1.0,
            "lora_path": path_text or None,
        }
    if not table:
        raise ValueError("--lora-arms was given but names no arms")
    return table


def _strip_subject(prompt: str) -> str:
    """Drop the appearance restatement so the slot is the only identity source."""
    lines = prompt.splitlines()
    kept: list[str] = []
    skipping = False
    for line in lines:
        if line.startswith("<"):
            skipping = line.startswith("<SUBJECT>")
        if not skipping:
            kept.append(line)
    return "\n".join(kept).strip()


def _prompted_plan(plan: H3SegmentPlan, prompt_mode: str) -> H3SegmentPlan:
    if prompt_mode == "full":
        return plan
    global_prompt = _strip_subject(plan.global_prompt)
    if not global_prompt:
        raise ValueError("--prompt-mode no-subject removed the entire global prompt")
    segments = tuple(
        H3Segment(
            prompt=None if segment.prompt is None else _strip_subject(segment.prompt),
            requested_frames=segment.requested_frames,
            segment_id=segment.segment_id,
            controls=segment.controls,
            semantic_state=segment.semantic_state,
        )
        for segment in plan.segments
    )
    return H3SegmentPlan(
        global_prompt=global_prompt, segments=segments, plan_id=plan.plan_id,
        global_controls=plan.global_controls,
    )


def _donor_entries(donor_cache: Path, split: str) -> list[Path]:
    """Find donor caches under either the sharded split layout or a flat directory."""
    sharded = sorted(donor_cache.glob(f"**/{split}/*.pt"))
    if sharded:
        return sharded
    return sorted(donor_cache.rglob("*.pt"))


def _load_donor_slots(
    donor_cache: Path, *, window_count: int, reference_shape: tuple[int, ...] | None = None,
    split: str = "validation",
) -> list[torch.Tensor]:
    """Collect donor slot content with matching latent geometry, one per window.

    Distinct entries are preferred so the transplant is content-wrong *and*
    content-diverse; falling back to fewer entries still moves content away from
    the target, which is the property the arm needs.

    ``reference_shape`` is the rollout's own slot shape when a ``both`` arm has
    already run.  Standalone, the first usable donor defines the shape and the
    pipeline rejects any transplant that does not fit the window, because a slot
    is a raw latent block and cannot cross resolutions.
    """
    from diffsynth.utils.continuation_lora import load_latent_cache_full

    entries = _donor_entries(donor_cache, split)
    if not entries:
        raise ValueError(f"donor cache {donor_cache} contains no *.pt entries (split={split!r})")
    donors: list[torch.Tensor] = []
    unreadable = 0
    mismatch = 0
    for entry in entries:
        if len(donors) >= window_count:
            break
        try:
            _video, _audio, slots, _metadata = load_latent_cache_full(str(entry))
        except Exception:
            unreadable += 1
            continue
        for slot in slots or ():
            tensor = slot.get("tensor")
            if not isinstance(tensor, torch.Tensor):
                continue
            if reference_shape is not None and tuple(tensor.shape) != tuple(reference_shape):
                continue
            donors.append(tensor.detach().clone())
            break
        else:
            mismatch += 1
    if not donors:
        raise ValueError(
            f"donor cache {donor_cache} has no usable slot"
            + (f" matching the rollout's latent shape {tuple(reference_shape)}" if reference_shape else "")
            + f" ({mismatch} entries offered no slot, {unreadable} could not be read); "
            "rebuild the donor cache at --height/--width"
        )
    shapes = {tuple(donor.shape) for donor in donors}
    if len(shapes) > 1:
        raise ValueError(f"donor cache {donor_cache} mixes latent shapes {sorted(shapes)}")
    return donors


def _frames_to_tensor(frames: Sequence[Any]) -> torch.Tensor | None:
    try:
        tensors = [torch.from_numpy(np.asarray(frame).copy()).float() for frame in frames]
    except (TypeError, ValueError, RuntimeError):
        return None
    if not tensors or any(tensor.shape != tensors[0].shape for tensor in tensors):
        return None
    return torch.stack(tensors)


def _mean_abs_difference(first: torch.Tensor, second: torch.Tensor) -> float:
    count = min(first.shape[0], second.shape[0])
    if count == 0:
        return float("nan")
    left = first[:count].reshape(count, -1)
    right = second[:count].reshape(count, -1)
    return float((left - right).abs().mean())


def _memory_consistency_report(result) -> dict[str, Any]:
    """Measure drift of every window relative to the sequence head.

    Drift is read from the frames a window *generated*, not from its whole
    assembled range.  The first ``overlap_video_frames`` frames of a continuation
    window are inherited from the previous window by construction (the assembled
    timeline keeps the earlier copy), so measuring them would compare the head
    against frames this window never produced.

    The head itself is window 0 -- conditioned on text alone, so it is the only
    appearance the sequence has that no memory slot could have supplied.  Drift is
    reported next to the in-window motion scale, because an absolute frame
    difference is only interpretable relative to how much the shot moves.
    """
    if result.video is None:
        return {"available": False, "reason": "no decoded video"}
    video = _frames_to_tensor(result.video)
    if video is None:
        return {"available": False, "reason": "frames are not array-like"}

    head_end = min(HEAD_REFERENCE_FRAMES, video.shape[0])
    head = video[:head_end]
    windows = []
    for record in result.state.records:
        window = record.window
        generated_start = window.timeline_start_video_frame + window.overlap_video_frames
        end = min(window.timeline_end_video_frame, video.shape[0])
        if end - generated_start <= 1:
            continue
        block = video[generated_start:end]
        entry: dict[str, Any] = {
            "segment_index": window.segment_index,
            "generated_index": int(generated_start),
            "generated_frames": int(end - generated_start),
            "head_drift": None,
            "motion_scale": None,
        }
        if window.overlap_video_frames > 0:
            compare = min(HEAD_REFERENCE_FRAMES, block.shape[0])
            entry["head_drift"] = _mean_abs_difference(block[:compare], head[:compare])
        entry["motion_scale"] = _mean_abs_difference(block[1:], block[:-1])
        windows.append(entry)

    drifts = [entry["head_drift"] for entry in windows if entry["head_drift"] is not None]
    return {
        "available": True,
        "windows": windows,
        "head_drift_mean": sum(drifts) / len(drifts) if drifts else None,
        "head_drift_max": max(drifts) if drifts else None,
        "head_drift_last": drifts[-1] if drifts else None,
        "motion_scale_mean": (
            sum(entry["motion_scale"] for entry in windows if entry["motion_scale"] is not None)
            / max(1, sum(1 for entry in windows if entry["motion_scale"] is not None))
        ),
    }


class _WindowProgress:
    """Record each window's resolved shape and the previous window's GPU peak.

    A rollout that dies inside the DiT leaves nothing behind but a traceback, so
    the peak that killed it is invisible.  The runner calls ``window_synchronizer``
    at the top of every window, which is exactly the moment the previous window's
    peak is final and the next window's shape is known.  Writing that to JSONL
    makes "which window, how big, how much memory" answerable after the fact.
    """

    def __init__(self, path: Path):
        self.path = path
        self.records: list[dict[str, Any]] = []
        self._peak_allocated = 0
        self._peak_reserved = 0

    def __call__(self, *, window, seed, controls) -> None:
        torch.cuda.synchronize()
        allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
        free = torch.cuda.mem_get_info()[0] / (1024 ** 3)
        self._peak_allocated = max(self._peak_allocated, allocated)
        self._peak_reserved = max(self._peak_reserved, reserved)
        self._write({
            "event": "window_start",
            "segment_index": window.segment_index,
            "resolved_video_frames": window.resolved_video_frames,
            "overlap_video_frames": window.overlap_video_frames,
            "video_latent_steps": window.video_latent_steps,
            "previous_peak_allocated_gib": round(allocated, 2),
            "previous_peak_reserved_gib": round(reserved, 2),
            "free_gib_at_start": round(free, 2),
        })
        # Reset so the next record measures this window alone, not the maximum
        # ever seen.
        torch.cuda.reset_peak_memory_stats()

    def finish(self, *, video_frames: int | None) -> dict[str, Any]:
        torch.cuda.synchronize()
        peak = max(self._peak_allocated, torch.cuda.max_memory_allocated() / (1024 ** 3))
        reserved = max(self._peak_reserved, torch.cuda.max_memory_reserved() / (1024 ** 3))
        record = {
            "event": "rollout_done",
            "decoded_video_frames": video_frames,
            "peak_allocated_gib": round(peak, 2),
            "peak_reserved_gib": round(reserved, 2),
        }
        self._write(record)
        return record

    def _write(self, record: dict[str, Any]) -> None:
        self.records.append(record)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _run_arm(
    runner: H3ContinuationRunner,
    plan: H3SegmentPlan,
    *,
    pipeline_kwargs: Mapping[str, Any],
    base_seed: int,
) -> dict[str, Any]:
    # Resolution and step count have to be handed over explicitly.  The pipeline
    # carries its own defaults, they are not the training-time ones, and a memory
    # slot is a raw latent block that cannot cross spatial grids -- so a rollout
    # that silently runs at the default resolution both allocates a much larger
    # sequence and fails the slot shape check.  Passing them also merges them into
    # every per-window request, which is where the slot encoder reads them from.
    result = runner.run(plan, base_seed=base_seed, pipeline_kwargs=pipeline_kwargs)
    return {
        "result": result,
        "joins": evaluate_continuation_joins(result),
        "memory": _memory_consistency_report(result),
    }


def _divergence(left, right) -> float | None:
    if left.video is None or right.video is None:
        return None
    first, second = _frames_to_tensor(left.video), _frames_to_tensor(right.video)
    if first is None or second is None:
        return None
    return _mean_abs_difference(first, second)


def _summarize(payload: dict[str, dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"arms": {}, "divergence": {}}
    for name, entry in payload.items():
        memory = entry["memory"]
        joins = entry["joins"]
        summary["arms"][name] = {
            "head_drift_mean": memory.get("head_drift_mean"),
            "head_drift_last": memory.get("head_drift_last"),
            "motion_scale_mean": memory.get("motion_scale_mean"),
            "join_video_mad_mean": _join_mean(joins, "video_mean_absolute_difference"),
            "join_audio_energy_mean": _join_mean(joins, "audio_energy_jump"),
            "long_range_video_mad": joins.get("video_metrics", {}).get("long_range_mean_absolute_difference"),
        }
    for name, entry in payload.items():
        if "slots" in entry:
            summary["arms"][name]["slots"] = entry["slots"]
        if "video_path" in entry:
            summary["arms"][name]["video_path"] = entry["video_path"]
        progress = entry.get("progress")
        if progress is not None:
            summary["arms"][name]["peak_allocated_gib"] = progress["peak_allocated_gib"]
            summary["arms"][name]["peak_reserved_gib"] = progress["peak_reserved_gib"]
    names = [name for name in payload if "result" in payload[name]]
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            value = _divergence(payload[first]["result"], payload[second]["result"])
            if value is not None:
                summary["divergence"][f"{first}_vs_{second}"] = value
    return summary


def _join_mean(joins: dict[str, Any], key: str) -> float | None:
    values = [
        item[key] for item in joins.get("joins", [])
        if isinstance(item.get(key), (int, float))
    ]
    return sum(values) / len(values) if values else None


def main() -> None:
    args = _parse_args()
    lora_arms = _parse_lora_arms(args.lora_arms) if args.lora_arms else None
    if lora_arms is not None:
        # A LoRA comparison is a different experiment from the memory ablation:
        # the arms come from the command line and the built-in names mean nothing.
        arms = list(lora_arms)
        arm_table: dict[str, dict[str, Any]] = dict(lora_arms)
    else:
        if args.lora is None:
            raise ValueError("either --lora or --lora-arms is required")
        arm_table = {name: dict(arm) for name, arm in ARMS.items()}
        arms = [name.strip() for name in args.arms.split(",") if name.strip()]
        unknown = [name for name in arms if name not in arm_table]
        if unknown:
            raise ValueError(f"unknown arms {unknown}; known arms are {sorted(arm_table)}")
    # --lora-arms never sets swap, so this only ever fires for the built-in table.
    if "swap" in arms and args.donor_cache is None:
        raise ValueError("--donor-cache is required for the swap arm")

    # A sharded transformer is one model, so a single shard stays a plain string
    # and a shard set becomes a list; ``ModelConfig.path`` accepts either.
    checkpoint: str | list[str] = (
        args.checkpoint[0] if len(args.checkpoint) == 1 else list(args.checkpoint)
    )

    # Resolution and step count are decoupled from the plan, so they are carried
    # as explicit pipeline kwargs.  The pipeline has its own defaults and they are
    # not the training-time ones, so forgetting to forward these changes both the
    # sequence cost and the memory-slot spatial grid.
    pipeline_kwargs: dict[str, Any] = {
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.num_inference_steps,
        "cfg_scale": 1.0,
    }

    if args.dry_run:
        print(json.dumps({
            "plan": str(args.segment_plan), "arms": arms, "prompt_mode": args.prompt_mode,
            "checkpoint": checkpoint if isinstance(checkpoint, str) else list(checkpoint),
            "stm_frames": args.stm_frames, "ltm_frames": args.ltm_frames,
            "overlap_frames": args.overlap_frames,
            "donor_cache": str(args.donor_cache) if args.donor_cache else None,
            "donor_split": args.donor_split,
            "pipeline_kwargs": pipeline_kwargs,
            "seed": args.seed,
            "lora_arms": (
                None if lora_arms is None
                else {name: {"lora_path": arm["lora_path"], "lora_scale": arm["lora_scale"]}
                      for name, arm in lora_arms.items()}
            ),
        }, ensure_ascii=False, indent=2))
        return

    if not torch.cuda.is_available():
        raise RuntimeError("the rollout ablation requires CUDA")

    evaluation = _load_evaluation_module()
    from diffsynth.pipelines.minimax_h3_continuation import load_h3_segment_plan

    plan = _prompted_plan(load_h3_segment_plan(args.segment_plan), args.prompt_mode)
    window_count = len(plan.segments)
    overlap_frames = (
        args.overlap_frames if args.overlap_frames is not None
        else manifest_overlap(plan) if manifest_overlap(plan) is not None
        else args.stm_frames
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, dict[str, Any]] = {}
    donor_slots: list[torch.Tensor] | None = None
    reference_shape: tuple[int, ...] | None = None
    # A 14B model load is the most expensive thing here, so pipelines are shared
    # across every arm that resolves to the same (LoRA file, scale) pair.  The
    # key has to carry the path and not just the scale: under --lora-arms each
    # arm brings its own checkpoint, and several arms commonly sit at scale 1.0.
    pipelines: dict[tuple[str | None, float], Any] = {}
    for name in arms:
        arm = arm_table[name]
        scale = float(arm["lora_scale"])
        if "lora_path" in arm:
            lora_path = arm["lora_path"]
        else:
            lora_path = str(args.lora) if args.lora is not None else None
        key = (lora_path, scale)
        if key not in pipelines:
            pipelines[key] = evaluation._load_pipeline(
                checkpoint, args.h3_base, lora_path=lora_path, lora_scale=scale,
            )
        pipe = pipelines[key]
        memory_plan = H3MemoryRolloutPlan(
            stm_frames=args.stm_frames if arm["stm"] else None,
            ltm_frames=args.ltm_frames if arm["ltm"] else None,
        )
        encoder = None
        observer = None
        if name == "both":
            observer = _SlotObserver()
        elif arm["swap"]:
            if donor_slots is None:
                donor_slots = _load_donor_slots(
                    args.donor_cache, window_count=window_count,
                    reference_shape=reference_shape, split=args.donor_split,
                )
            encoder = _ReplaySlotEncoder(donor_slots)

        # Per-arm file: one process may run several arms, and a shared name would
        # let the last arm overwrite the only record of the arm that failed.
        progress = _WindowProgress(args.output_dir / f"progress_{name}.jsonl")
        runner = H3ContinuationRunner(
            pipe,
            H3ContinuationConfig(
                requested_window_frames=plan.segments[0].requested_frames or 345,
                overlap_frames=overlap_frames,
            ),
            continuation_mode="masked-av-v14",
            prefer_latent_handoff=True,
            model_identity=f"h3-memory-rollout-{name}",
            manifest_directory=args.output_dir / "manifests" / name,
            memory_rollout_plan=memory_plan,
            memory_slot_encoder=encoder,
            memory_slot_observer=observer,
            window_synchronizer=progress,
        )
        entry = _run_arm(
            runner, plan, pipeline_kwargs=pipeline_kwargs, base_seed=args.seed,
        )
        entry["progress"] = progress.finish(
            video_frames=None if entry["result"].video is None else len(entry["result"].video)
        )
        if args.save_video:
            video_path = _save_arm_video(
                entry["result"], args.output_dir, name, fps=args.fps,
            )
            if video_path is not None:
                entry["video_path"] = video_path
        payload[name] = entry
        if name == "both" and observer is not None and observer.calls:
            # The arm's own first slot is the strongest available statement of the
            # geometry a donor has to match.
            reference_shape = observer.shapes()[0]
            entry["slots"] = [{"name": slot_name, "shape": list(shape)}
                              for (slot_name, _), shape in zip(observer.calls, observer.shapes())]

    summary = _summarize(payload)
    summary["config"] = {
        "plan": str(args.segment_plan), "arms": arms, "prompt_mode": args.prompt_mode,
        "stm_frames": args.stm_frames, "ltm_frames": args.ltm_frames,
        "height": args.height, "width": args.width,
        "num_inference_steps": args.num_inference_steps, "seed": args.seed,
        "overlap_frames": overlap_frames,
        # Which checkpoint produced which row is the whole comparison, so it is
        # recorded rather than left to the reader to infer from the arm name.
        "lora_arms": (
            None if lora_arms is None
            else {name: {"lora_path": arm_table[name]["lora_path"],
                         "lora_scale": arm_table[name]["lora_scale"]}
                  for name in arms}
        ),
    }
    summary_path = _summary_path(args.output_dir, arms)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for name, entry in payload.items():
        (args.output_dir / f"joins_{name}.json").write_text(
            json.dumps(entry["joins"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _save_arm_video(result, output_dir: Path, arm: str, *, fps: int) -> str | None:
    """Write one arm's assembled rollout, or return ``None`` if it has no frames."""
    video = getattr(result, "video", None)
    if not video:
        return None
    from diffsynth.utils.data import save_video as _write_video

    path = output_dir / "videos" / f"{arm}.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_video(list(video), str(path), fps=fps)
    return str(path)


def _summary_path(output_dir: Path, arms: Sequence[str]) -> Path:
    """Summary file names carry the arm set.

    One process runs one arm subset and several processes share an output
    directory -- one per GPU, and on a two-node job one per node.  A single
    ``rollout_ablation.json`` would be overwritten by whichever process finished
    last, silently publishing an incomplete ablation.
    """
    return output_dir / f"rollout_ablation_{'-'.join(arms)}.json"


def manifest_overlap(plan: H3SegmentPlan) -> int | None:
    """A plan may carry its own cut length; if it does, it wins over the CLI."""
    value = plan.global_controls.get("overlap_frames")
    return None if value is None else int(value)


if __name__ == "__main__":
    main()
