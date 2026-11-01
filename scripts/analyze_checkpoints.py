#!/usr/bin/env python3
"""Accuracy by reasoning position, fixed-stop baseline, and the oracle saving, per task.

    python scripts/analyze_checkpoints.py \\
        --checkpoints outputs/pilot/checkpoints_full/checkpoints.jsonl \\
        --out outputs/pilot/curve.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from thinkenough.checkpoints import summarize_curve
from thinkenough.scot import is_correct, load_attributes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze forced answers by reasoning position")
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    mask3d = load_attributes("scannet_mask3d_val_attributes.pt")
    human = load_attributes("scannet_val_attributes.pt")
    by_task: dict[str, list[dict]] = {}
    with open(args.checkpoints, encoding="utf-8") as f:
        for row in map(json.loads, f):
            correct = is_correct(row["answer"], row["metadata"], mask3d, human)
            if correct is not None:
                by_task.setdefault(row["metadata"]["task"], []).append({**row, "correct": correct})

    report = {task: summarize_curve(rows) for task, rows in sorted(by_task.items())}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for task, summary in report.items():
        print(f"{task}: {summary['n']} questions")
        for point in summary["curve"]:
            print(
                f"  at {point['fraction']:.0%}: accuracy {point['accuracy']:.3f} "
                f"[{point['ci_low']:.3f}, {point['ci_high']:.3f}], "
                f"tokens kept {point['tokens_kept']:.0%}"
            )
        oracle = summary["oracle"]
        print(f"  oracle: accuracy {oracle['accuracy']:.3f}, tokens kept {oracle['tokens_kept']:.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
