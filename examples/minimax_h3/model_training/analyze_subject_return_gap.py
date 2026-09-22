#!/usr/bin/env python3
"""Measure the *return gap* Delta of the existing continuation training data.

Delta is the quantity that decides whether a memory module can buy anything at
all. For an adjacent shot pair ``(context = shot t-1, target = shot t)`` and a
subject ``S`` visible in the target,

    Delta_S(t) = t - (last shot < t in which S was visible)

``Delta = 1`` means S is still inside the one-shot context, so the *existing*
conditioning already carries it. Only ``Delta >= 2`` samples are ones whose
subject has left the context window -- those are the samples a memory module can
possibly help, and the share of them is an upper bound on memory's marginal
value over the current pipeline.

Two properties of the source annotation make the measurement well posed:

1. ``[subjectN]`` is a **sequence-global identity index assigned in order of
   first appearance**: across 20k records the first-appearance positions are
   monotone in N in 99.95% of records, and the gender cue carried by the
   ``<SUBJECT>`` description disagrees between two mentions of the same N in
   only 2.7% of cases. Same N therefore means same person.
2. Because the numbering is monotone by first appearance, a subject absent from
   all earlier shots of the *same record* has never been seen: it is a **new
   entrant**, not a returning one. Memory cannot help there -- there is nothing
   to retrieve. The two cases must be separated, and the monotonicity is what
   separates them.

So each adjacent pair is classified three ways: ``context_covered`` (Delta = 1),
``memory_eligible`` (Delta >= 2 within the record), and ``new_entrant`` (no
subject of the target appears earlier in the record).

Because a record is a ~5-shot excerpt of a longer episode, a subject returning
from *before* the excerpt also lands in ``new_entrant``. That biases the
``memory_eligible`` share **downward**, so the reported numbers are conservative
lower bounds.

The script also reports the ``Delta >= k`` survival curve, which separates the
two competing ways of covering a long gap -- growing the context window (costs
the 345-frame budget) versus a memory module (does not).
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import time
from pathlib import Path

# The header exists in several variants in the source data:
#   [Shot 2/5 | 3.6s-9.0s]
#   [Shot 5/5 | end | 9.0s-14.4s]
#   [Shot 1/5 | start | 0.0s-3.6s]
# The optional word between the two pipes marks a truncated first/last
# shot of the enclosing sequence; it is not a different block kind.
SHOT_HEADER = re.compile(
    r"\[Shot\s+(\d+)\s*/\s*(\d+)\s*\|\s*(?:\w+\s*\|\s*)?([0-9.]+)s\s*-\s*([0-9.]+)s\]")

SUBJECT_BLOCK = re.compile(r"<SUBJECT>(.*?)(?=<Scene>|<Event>|\[Shot\s+\d+\s*/|\Z)", re.S)
SUBJECT_ID = re.compile(r"\[subject\s*(\d+)\]", re.I)


def parse_shot_subjects(prompt: str) -> list[tuple[int, int, set[int]]]:
    """Return ``[(shot_index, duration_ms, subject_ids)]`` for one record prompt.

    Only the ``<SUBJECT>`` block counts as "visible in this shot": the ``<Scene>``
    and ``<Event>`` blocks mention people too, but those mentions include people
    the shot *refers to* without showing (e.g. an off-screen interlocutor).
    """
    shots: list[tuple[int, int, set[int]]] = []
    matches = list(SHOT_HEADER.finditer(prompt))
    for position, match in enumerate(matches):
        end = matches[position + 1].start() if position + 1 < len(matches) else len(prompt)
        block = prompt[match.end():end]
        subject_block = SUBJECT_BLOCK.search(block)
        ids: set[int] = set()
        if subject_block is not None:
            ids = {int(value) for value in SUBJECT_ID.findall(subject_block.group(1))}
        start = float(match.group(3))
        stop = float(match.group(4))
        shots.append((int(match.group(1)), int(round((stop - start) * 1000)), ids))
    return shots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-jsonl", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=0, help="0 = all records (probe with e.g. 20000)")
    parser.add_argument("--progress-every", type=int, default=50000)
    parser.add_argument(
        "--shot-identities", type=Path, default=None,
        help=(
            "Report the gap distribution under the **epoch-correct** view: drop the "
            "repeated copies of an adjacent shot pair using source-level shot "
            "identity (source_id, start, end). Points at "
            "record_shot_identities.jsonl from extract_record_shot_identities.py."
        ),
    )
    args = parser.parse_args()

    started = time.monotonic()
    identities: dict[str, tuple[str, list]] = {}
    if args.shot_identities is not None:
        with args.shot_identities.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    identities[row["sequence_id"]] = (row["source_id"], row["shots"])
        print(f"[ident] loaded {len(identities)} records from {args.shot_identities}", flush=True)
    seen_pairs: set = set()
    identity_stats = collections.Counter()
    gap_hist: collections.Counter[int] = collections.Counter()
    gap_by_target_shot: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    stats = collections.Counter()
    pair_class: collections.Counter[str] = collections.Counter()
    reentry_subjects: collections.Counter[int] = collections.Counter()
    next_report = args.progress_every

    with open(args.source_jsonl, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            stats["records"] += 1
            if args.max_records and stats["records"] > args.max_records:
                break
            record = json.loads(line)
            prompt = record.get("prompt") or ""
            prev_source = ""
            shots = parse_shot_subjects(prompt)
            stats["window_shots"] += len(shots)
            if len(shots) < 2:
                stats["records_without_pair"] += 1
                continue
            ranges = None
            if identities:
                entry = identities.get(record.get("sequence_id"))
                if entry is None:
                    identity_stats["record_missing"] += 1
                elif len(entry[1]) != len(shots):
                    identity_stats["shot_count_mismatch"] += 1
                else:
                    ranges = entry[1]
                    prev_source = entry[0]
                    identity_stats["records_ok"] += 1
                if ranges is None:
                    continue
            last_seen: dict[int, int] = {}
            for position, (shot_index, duration_ms, ids) in enumerate(shots):
                if position > 0:
                    if ranges is not None:
                        previous_range, target_range = ranges[position - 1], ranges[position]
                        key = (prev_source, previous_range[0], previous_range[1],
                               target_range[0], target_range[1])
                        if key in seen_pairs:
                            identity_stats["duplicate_pairs_skipped"] += 1
                            for subject in ids:
                                last_seen[subject] = position
                            continue
                        seen_pairs.add(key)
                    stats["pairs"] += 1
                    if not ids:
                        stats["target_without_subject"] += 1
                    else:
                        returning = [position - last_seen[s] for s in ids if s in last_seen]
                        new_entrants = [s for s in ids if s not in last_seen]
                        stats["subject_new_entrant"] += len(new_entrants)
                        if not returning:
                            stats["pair_all_new"] += 1
                            pair_class["new_entrant_only"] += 1
                        else:
                            sample_max = max(returning)
                            gap_hist[sample_max] += 1
                            gap_by_target_shot[len(ids)][sample_max] += 1
                            if sample_max == 1:
                                pair_class["context_covered"] += 1
                            else:
                                pair_class["memory_eligible"] += 1
                                stats["pairs_memory_eligible"] += 1
                                reentry_subjects[0] += len([g for g in returning if g >= 2])
                for subject in ids:
                    last_seen[subject] = position
            if args.progress_every and stats["records"] >= next_report:
                elapsed = max(time.monotonic() - started, 1e-6)
                print(f"[gap] records={stats['records']} pairs={stats['pairs']} rate={stats['records']/elapsed:.0f} rec/s", flush=True)
                next_report += args.progress_every

    scored = max(sum(gap_hist.values()), 1)
    with_subject = max(stats["pairs"] - stats["target_without_subject"], 1)
    survival = []
    for threshold in range(1, 8):
        covered = sum(count for gap, count in gap_hist.items() if gap >= threshold)
        survival.append({
            "threshold": threshold,
            "samples": covered,
            "share_of_scored": round(covered / scored, 6),
            "share_of_all_pairs": round(covered / with_subject, 6),
        })
    payload = {
        "source_jsonl": str(args.source_jsonl),
        "stats": dict(stats),
        "pairs_scored": scored,
        "pairs_with_subject": with_subject,
        "pair_classes": {k: {"samples": v, "share": round(v / with_subject, 6)} for k, v in sorted(pair_class.items())},
        "gap_histogram": {str(gap): count for gap, count in sorted(gap_hist.items())},
        "gap_share": {str(gap): round(count / scored, 6) for gap, count in sorted(gap_hist.items())},
        "survival_delta_ge_k": survival,
        "subject_count_by_target": {
            str(count): {"samples": sum(counter.values()), "delta_ge_2": sum(v for k, v in counter.items() if k >= 2)}
            for count, counter in sorted(gap_by_target_shot.items())
        },
        "dedup_by_shot_identity": bool(identities),
        "identity_stats": dict(identity_stats),
        "elapsed_sec": round(time.monotonic() - started, 1),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps({
        "pairs_with_subject": with_subject,
        "pair_classes": {k: round(v / with_subject, 4) for k, v in sorted(pair_class.items())},
        "delta_ge_2_of_scored": survival[1]["share_of_scored"],
        "delta_ge_2_of_all_pairs": survival[1]["share_of_all_pairs"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
