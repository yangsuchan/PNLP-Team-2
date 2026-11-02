#!/usr/bin/env python3
"""Compare stopping rules on collected checkpoints, per program-graded task.

Rules: fixed token caps, the oracle upper bound, and, with --detector, the trained
answer-readiness detector under two-fold, scene-separated evaluation.

    python scripts/analyze_checkpoints.py \\
        --collected outputs/pilot/checkpoints_full --detector --out outputs/pilot/stopping.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from thinkenough.checkpoints import Checkpoint, QuestionTrace, fixed_cap, oracle, summarize
from thinkenough.scot import is_correct, load_attributes


def load_traces(collected: Path, mask3d: dict, human: dict) -> tuple[dict[str, list[QuestionTrace]], dict]:
    """Group graded questions by task. Tasks without program grading are left out."""
    rows_by_sample: dict[str, list[dict]] = {}
    for line in (collected / "checkpoints.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        rows_by_sample.setdefault(row["sample_id"], []).append(row)

    by_task: dict[str, list[QuestionTrace]] = {}
    info: dict[str, dict] = {}
    for line in (collected / "samples.jsonl").read_text(encoding="utf-8").splitlines():
        sample = json.loads(line)
        meta = sample["metadata"]
        missing = sample["full_status"] != "ok" or not sample["full_answer"].strip()
        full_correct = is_correct(sample["full_answer"], meta, mask3d, human)
        if full_correct is None:
            continue
        checkpoints = [
            Checkpoint(
                tokens=row["prefix_tokens"],
                correct=bool(is_correct(row["answer"], meta, mask3d, human)),
                missing=not row["answer"].strip(),
                row=row,
            )
            for row in sorted(rows_by_sample.get(sample["sample_id"], []), key=lambda r: r["checkpoint_idx"])
        ]
        by_task.setdefault(meta["task"], []).append(
            QuestionTrace(
                sample_id=sample["sample_id"],
                scene_id=meta["scene_id"],
                full_tokens=sample["full_tokens"],
                full_correct=bool(full_correct) and not missing,
                full_missing=missing,
                checkpoints=checkpoints,
            )
        )
        info[sample["sample_id"]] = sample
    return by_task, info


def detector_report(traces: list[QuestionTrace], info: dict, collected: Path) -> dict:
    """Two-fold evaluation of the answer-readiness detector on one task."""
    import numpy as np
    import torch

    from thinkenough.detector import stop_at_threshold, two_fold_evaluate

    def load(relative: str) -> np.ndarray:
        return torch.load(collected / relative, map_location="cpu", weights_only=True).numpy()

    cache: dict[str, np.ndarray] = {}

    def states(path: str) -> np.ndarray:
        if path not in cache:
            cache[path] = load(path)
        return cache[path]

    def hidden_at_layer(layer: int):
        def hidden(trace: QuestionTrace, index: int | None) -> np.ndarray:
            path = (
                info[trace.sample_id]["start_hidden_path"]
                if index is None
                else trace.checkpoints[index].row["hidden_state_path"]
            )
            return states(path)[layer]

        return hidden

    with_states = next(t for t in traces if t.checkpoints)
    layer_count = len(states(with_states.checkpoints[0].row["hidden_state_path"]))
    budget = next(iter(info.values()))["budget"]
    directions = two_fold_evaluate(traces, hidden_at_layer, range(layer_count), budget)

    report = []
    combined: dict[str, object] = {}
    for result in directions:
        score_of = dict(zip((t.sample_id for t in result["test_traces"]), result["test_scores"]))
        rule = lambda t, threshold=result["threshold"], scores=score_of: stop_at_threshold(
            t, scores[t.sample_id], threshold
        )
        combined.update({t.sample_id: rule(t) for t in result["test_traces"]})
        report.append(
            {
                "test_fold": result["test_fold"],
                "layer": result["layer"],
                "validation_auroc": result["validation_auroc"],
                "test_auroc": result["test_auroc"],
                "threshold": result["threshold"],
                **summarize(result["test_traces"], rule),
            }
        )
    return {"directions": report, "all_questions": summarize(traces, lambda t: combined[t.sample_id])}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare stopping rules")
    parser.add_argument("--collected", required=True, help="Output directory of collect_checkpoints.py")
    parser.add_argument("--out", required=True)
    parser.add_argument("--detector", action="store_true", help="Also train and evaluate the detector")
    args = parser.parse_args(argv)

    collected = Path(args.collected)
    mask3d = load_attributes("scannet_mask3d_val_attributes.pt")
    human = load_attributes("scannet_val_attributes.pt")
    by_task, info = load_traces(collected, mask3d, human)

    report = {}
    for task, traces in sorted(by_task.items()):
        caps = sorted({c.tokens for t in traces for c in t.checkpoints})
        report[task] = {
            "fixed_caps": {str(cap): summarize(traces, lambda t, c=cap: fixed_cap(t, c)) for cap in caps},
            "oracle": summarize(traces, oracle),
        }
        if args.detector:
            task_info = {t.sample_id: info[t.sample_id] for t in traces}
            report[task]["detector"] = detector_report(traces, task_info, collected)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    def line(name: str, s: dict) -> None:
        low, high = s["accuracy_change_ci"]
        print(
            f"  {name}: accuracy {s['accuracy_full']:.3f} -> {s['accuracy']:.3f} "
            f"(change CI [{low:+.3f}, {high:+.3f}]), tokens {s['reasoning_tokens_full_mean']:.0f} -> "
            f"{s['reasoning_tokens_mean']:.0f} ({s['token_reduction']:.0%} less), missing "
            f"{s['missing_answers_full']} -> {s['missing_answers']}, first check {s['stopped_at_first_check']:.0%}"
        )

    for task, result in report.items():
        print(f"{task}: {result['oracle']['n']} questions")
        for cap, summary in result["fixed_caps"].items():
            line(f"cap {cap}", summary)
        line("oracle", result["oracle"])
        if "detector" in result:
            line("detector", result["detector"]["all_questions"])
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
