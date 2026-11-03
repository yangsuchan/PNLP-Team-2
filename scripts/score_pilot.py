#!/usr/bin/env python3
"""Summarize finished runs per task and arm. The AI judge is not included yet.

    python scripts/score_pilot.py \\
        --run thinking=outputs/pilot/thinking_text \\
        --run instruct=outputs/pilot/instruct_text \\
        --run direct=outputs/pilot/direct_text \\
        --out outputs/pilot/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from thinkenough.scot import answer_text, caption_scores, load_attributes, summarize_arm


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize pilot runs")
    parser.add_argument("--run", action="append", required=True, help="arm=directory")
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-caption", action="store_true", help="Skip ROUGE-L and METEOR")
    args = parser.parse_args(argv)

    mask3d = load_attributes("scannet_mask3d_val_attributes.pt")
    human = load_attributes("scannet_val_attributes.pt")

    table = []
    for spec in args.run:
        arm, _, directory = spec.partition("=")
        rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(directory).glob("*.json"))]
        if not rows:
            print(f"[error] no results in {directory}", file=sys.stderr)
            return 1
        summaries = summarize_arm(rows, arm, mask3d, human)
        if not args.no_caption:
            for summary in summaries:
                group = [r for r in rows if r["metadata"]["task"] == summary["task"]]
                if summary["task"] != "implicit_detection":
                    summary.update(
                        caption_scores(
                            {r["sample_id"]: answer_text(r, arm) for r in group},
                            {r["sample_id"]: r["metadata"]["references"] for r in group},
                        )
                    )
        table.extend(summaries)

    columns = list(dict.fromkeys(key for row in table for key in row))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(table)
    for row in table:
        print({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()})
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
