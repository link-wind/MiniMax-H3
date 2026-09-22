"""Run a reproducible MiniMax-H3 Diff-VF ablation matrix.

The script intentionally delegates each variant to the continuation entrypoint
in a separate process.  This releases model/decoder memory between runs and
makes every report independently replayable with its exact CLI configuration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


VARIANTS: dict[str, tuple[str, ...]] = {
    "hni-independent": ("--mode", "diff-vf", "--hni-weight", "0.0"),
    "hni-independent": ("--mode", "diff-vf", "--hni-weight", "0.0"),
    "hni-mixed": ("--mode", "diff-vf", "--hni-weight", "0.5"),
    "hni-shared": ("--mode", "diff-vf", "--hni-weight", "1.0"),
    "center-distance": (
        "--mode", "diff-vf", "--hni-weight", "0.5", "--wws-weighting", "center-distance",
    ),
    "video-tes": (
        "--mode", "diff-vf", "--hni-weight", "0.5", "--tes", "--tes-window-steps", "12",
        "--tes-stride", "2", "--fusion-local-start", "0.25", "--fusion-local-end", "0.85",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute a fixed MiniMax-H3 Diff-VF evaluation matrix.")
    parser.add_argument("--h3-base", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--segment-plan", type=Path, default=Path(__file__).with_name("h3_continuation_plan.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variants", default=",".join(VARIANTS), help="Comma-separated variants from the fixed matrix.")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--window-frames", type=int, default=243)
    parser.add_argument("--overlap-frames", type=int, default=34)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true", help="Write the resolved matrix without starting inference.")
    args = parser.parse_args()
    args.variants = tuple(item.strip() for item in args.variants.split(",") if item.strip())
    unknown = set(args.variants).difference(VARIANTS)
    if unknown:
        parser.error(f"unknown variants: {', '.join(sorted(unknown))}; choices: {', '.join(VARIANTS)}")
    if not args.variants:
        parser.error("at least one evaluation variant is required")
    return args


def main() -> None:
    args = parse_args()
    continuation = Path(__file__).with_name("MiniMax-H3-Continuation.py")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixed = (
        "--h3-base", str(args.h3_base), "--checkpoint", args.checkpoint,
        "--segment-plan", str(args.segment_plan), "--height", str(args.height), "--width", str(args.width),
        "--window-frames", str(args.window_frames), "--overlap-frames", str(args.overlap_frames),
        "--seed", str(args.seed), "--num-inference-steps", str(args.num_inference_steps),
    )
    resolved = []
    for name in args.variants:
        output = args.output_dir / f"{name}.mp4"
        report = args.output_dir / f"{name}.json"
        command = [
            sys.executable, str(continuation), *fixed, *VARIANTS[name],
            "--output", str(output), "--report", str(report), "--evaluation-variant", name,
        ]
        resolved.append({"name": name, "command": command, "output": str(output), "report": str(report)})
    matrix_path = args.output_dir / "matrix.json"
    matrix_path.write_text(json.dumps({"runs": resolved}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.dry_run:
        print(json.dumps({"matrix": str(matrix_path), "runs": len(resolved), "dry_run": True}, ensure_ascii=False))
        return
    results = []
    for run in resolved:
        started = time.perf_counter()
        completed = subprocess.run(run["command"], check=False)
        results.append({
            "name": run["name"], "exit_code": completed.returncode,
            "driver_elapsed_seconds": time.perf_counter() - started,
            "report": run["report"], "output": run["output"],
        })
        if completed.returncode:
            break
    matrix_path.write_text(json.dumps({"runs": resolved, "results": results}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if any(result["exit_code"] for result in results):
        raise SystemExit(1)
    print(json.dumps({"matrix": str(matrix_path), "runs": len(results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
