#!/usr/bin/env python3
"""Analyze and build multi-window continuation manifests.

The input is the expanded 345-frame continuation index.  Records are grouped by
``sequence_id`` and ``shot_index``; windows are chained only when they belong to
the same shot and their starts differ by the continuation stride (within a
small tolerance).  Crossing an annotated shot boundary is intentionally
disabled because it would train the model to continue through a camera cut.

The output JSONL contains one record per eligible window in a chain.  Each
record carries the first-window anchor, the immediately preceding window, and
the chain position, so the training loader can use short-term latent handoff
and long-term appearance memory without looking ahead.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def _read_records(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"line {line_no} is not an object")
            required = ("sample_id", "sequence_id", "shot_index", "start_sec", "end_sec", "window_frames")
            missing = [key for key in required if key not in value]
            if missing:
                raise ValueError(f"line {line_no} missing: {', '.join(missing)}")
            records.append(value)
    return records


def _chain_group(records: list[dict], *, stride_frames: int, tolerance_sec: float) -> list[list[dict]]:
    """Split a group into chains of adjacent windows with the expected stride."""
    records = sorted(records, key=lambda item: (float(item["start_sec"]), str(item["sample_id"])))
    chains: list[list[dict]] = []
    current: list[dict] = []
    stride_sec = stride_frames / 24.0
    for record in records:
        if not current:
            current = [record]
            continue
        previous = current[-1]
        delta = float(record["start_sec"]) - float(previous["start_sec"])
        same_geometry = int(record["window_frames"]) == int(previous["window_frames"])
        if same_geometry and math.isclose(delta, stride_sec, abs_tol=tolerance_sec):
            current.append(record)
        else:
            chains.append(current)
            current = [record]
    if current:
        chains.append(current)
    return chains


def build_manifest(records: list[dict], *, overlap_frames: int, min_chain_length: int,
                   tolerance_sec: float) -> tuple[list[dict], dict]:
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for record in records:
        groups[(str(record["sequence_id"]), int(record["shot_index"]))].append(record)

    chains: list[list[dict]] = []
    chain_lengths = Counter()
    stride_frames = None
    for group in groups.values():
        if group:
            stride_frames = int(group[0]["window_frames"]) - overlap_frames
        if stride_frames is None or stride_frames <= 0:
            continue
        for chain in _chain_group(group, stride_frames=stride_frames, tolerance_sec=tolerance_sec):
            chain_lengths[len(chain)] += 1
            if len(chain) >= min_chain_length:
                chains.append(chain)

    output: list[dict] = []
    for chain_id, chain in enumerate(chains):
        anchor = chain[0]
        for position, target in enumerate(chain):
            history = chain[position - 1] if position else None
            output.append({
                "schema_version": "h3-multi-window-v1",
                "chain_id": f"{anchor['sequence_id']}:{anchor['shot_index']}:{chain_id}",
                "sequence_id": anchor["sequence_id"],
                "shot_index": anchor["shot_index"],
                "split": anchor.get("split", "train"),
                "window_index": position,
                "chain_length": len(chain),
                "anchor_sample_id": anchor["sample_id"],
                "history_sample_id": None if history is None else history["sample_id"],
                "target_sample_id": target["sample_id"],
                "history": history,
                "target": target,
            })

    stats = {
        "records": len(records),
        "groups": len(groups),
        "chain_lengths": {str(length): count for length, count in sorted(chain_lengths.items())},
        "chains_total": sum(chain_lengths.values()),
        "chains_eligible": len(chains),
        "windows_eligible": len(output),
        "groups_with_multiple_windows": sum(count for length, count in chain_lengths.items() if length >= 2),
        "max_chain_length": max(chain_lengths, default=0),
        "required_min_chain_length": min_chain_length,
        "overlap_frames": overlap_frames,
        "stride_frames": stride_frames or 0,
    }
    return output, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--overlap-frames", type=int, default=39)
    parser.add_argument("--min-chain-length", type=int, default=2)
    parser.add_argument("--tolerance-sec", type=float, default=1e-3)
    args = parser.parse_args()
    if args.overlap_frames <= 0 or args.min_chain_length <= 0:
        parser.error("overlap-frames and min-chain-length must be positive")
    records = _read_records(args.input_jsonl)
    output, stats = build_manifest(
        records,
        overlap_frames=args.overlap_frames,
        min_chain_length=args.min_chain_length,
        tolerance_sec=args.tolerance_sec,
    )
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_jsonl.with_suffix(args.output_jsonl.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for item in output:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    temporary.replace(args.output_jsonl)
    print(json.dumps({"output": str(args.output_jsonl), **stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
