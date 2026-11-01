"""Run the ThinkEnough evaluation (defaults: synthetic example data).

요약: [4단계] 평가 전체 실행: 방식 비교 → validation에서 τ 선택 → test에 고정 적용. 기본값은 가짜 예시.

Steps
  1. online method comparison           (--results, baseline --baseline)
  2. probe classification on validation  -> choose threshold (validation only) -> save
  3. probe classification on test with the saved, fixed threshold
  4. offline early-exit simulation on test with the same threshold

Usage (from the repo root):
    .venv/bin/python scripts/evaluate_early_exit.py
    .venv/bin/python scripts/evaluate_early_exit.py --results <online.jsonl> --references data/processed/sqa3d/test.jsonl ...
    .venv/bin/python scripts/evaluate_early_exit.py --threshold-in outputs/evaluation/<frozen>.json   # reuse, no reselection
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thinkenough.evaluation.checkpoint_labeling import load_references  # noqa: E402
from thinkenough.evaluation.early_exit_evaluator import (  # noqa: E402
    evaluate_online, evaluate_probe, join_predictions, select_threshold, simulate_early_exit,
)

FIX = ROOT / "tests/fixtures"


def read_jsonl(path):
    return [json.loads(line) for line in open(path)] if path else []


def rel(p):
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=FIX / "fake_evaluation_results.jsonl")
    parser.add_argument("--references", type=Path, nargs="+", default=[FIX / "fake_references.jsonl"])
    parser.add_argument("--baseline", default="full_cot")
    parser.add_argument("--checkpoints", type=Path, default=ROOT / "data/processed/checkpoints/example_checkpoints.jsonl")
    parser.add_argument("--labels", type=Path, default=ROOT / "data/processed/checkpoint_labels/example_labels.jsonl")
    parser.add_argument("--predictions", type=Path, default=FIX / "fake_probe_predictions.jsonl")
    parser.add_argument("--criterion", choices=["f1", "target_precision"], default="f1")
    parser.add_argument("--target-precision", type=float, default=0.9)
    parser.add_argument("--threshold-in", type=Path, help="use a frozen threshold file instead of selecting")
    parser.add_argument("--threshold-out", type=Path, default=ROOT / "outputs/evaluation/example_threshold.json")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/evaluation/example_metrics.json")
    args = parser.parse_args()

    references = load_references(args.references)
    results = read_jsonl(args.results)
    labels = read_jsonl(args.labels)
    predictions = read_jsonl(args.predictions)
    checkpoints = read_jsonl(args.checkpoints)
    synthetic = any(r.get("synthetic") for r in results + labels + predictions)

    report = {
        "data_source": "synthetic" if synthetic else "real",
        "disclaimer": ("SYNTHETIC TEST DATA - these numbers only check that the code works; "
                       "they are not SQA3D / Qwen3 results") if synthetic else None,
        "inputs": {k: rel(v) for k, v in [("results", args.results), ("labels", args.labels),
                                           ("predictions", args.predictions), ("checkpoints", args.checkpoints)]},
    }

    # 1. online comparison
    report["online"] = evaluate_online(results, references, baseline=args.baseline) if results else None

    # 2-4. probe: select on validation, apply unchanged to eval split
    joined, no_pred = join_predictions(labels, predictions)
    report["probe_rows_without_prediction"] = no_pred
    if args.threshold_in:
        threshold = json.load(open(args.threshold_in))
    else:
        threshold = select_threshold([r for r in joined if r["split"] == "validation"],
                                     criterion=args.criterion, target_precision=args.target_precision)
        args.threshold_out.parent.mkdir(parents=True, exist_ok=True)
        args.threshold_out.write_text(json.dumps(threshold, indent=2))
    report["threshold"] = threshold
    report["probe_classification"] = {
        "validation (selection split, in-sample)": evaluate_probe(joined, "validation", threshold if threshold.get("threshold") is not None else None),
        args.eval_split: evaluate_probe(joined, args.eval_split, threshold if threshold.get("threshold") is not None else None),
    }
    report["offline_simulation"] = simulate_early_exit(
        checkpoints, labels, predictions, threshold if threshold.get("threshold") is not None else None, args.eval_split)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    print(json.dumps(report, indent=1, ensure_ascii=False, default=str)[:6000])
    print(f"\nSaved: {rel(args.output)}")


if __name__ == "__main__":
    main()
