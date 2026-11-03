#!/usr/bin/env python3
"""Compare stopping rules on collected checkpoints, per program-graded task.

Rules: fixed token caps, stopping once the answer repeats, the oracle upper bound, and, with
--detector, the answer-readiness detector under two-fold, scene-separated evaluation. With --runs,
detectors that also read scene-grounded progress in the reasoning text.

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

from thinkenough.checkpoints import Checkpoint, QuestionTrace, answer_stable, fixed_cap, oracle, oracle_give_up, summarize
from thinkenough.claims import char_positions, progress_features
from thinkenough.judge import ACCURACY_CORRECT, CRITERIA
from thinkenough.scot import is_correct, load_attributes, parse_answer_key


FORCED_ANSWER_LIMIT = 1024  # collect_checkpoints.py --answer-max-tokens: an answer this long was cut off.


def load_labels(path: Path) -> dict[tuple[str, str], tuple[bool, float | None]]:
    """Answer Accuracy rows as right or wrong. An ungraded answer counts as wrong."""
    rows = map(json.loads, path.read_text(encoding="utf-8").splitlines())
    return {(r["sample_id"], r["item"]): ((r.get("accuracy") or 0.0) >= ACCURACY_CORRECT, None) for r in rows}


def official_labels(path: Path, margin: float) -> dict[tuple[str, str], tuple[bool, float | None]]:
    """SCoT scores as labels: a forced answer is right if it is within ``margin`` of the full answer.

    Scores are the mean of the three criteria. Each pair shows the full answer again, so a forced
    answer's score is its gap in the pair added to the question's mean full score. An unreadable
    grade counts as the lowest score.
    """
    mean = lambda scores: sum(scores[c] for c in CRITERIA) / len(CRITERIA)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    fulls: dict[str, list[float]] = {}
    for row in rows:
        if row["status"] == "ok":
            fulls.setdefault(row["sample_id"], []).append(mean(row["scores"]["full"]))
    labels: dict[tuple[str, str], tuple[bool, float | None]] = {}
    for sample_id, values in fulls.items():
        labels[(sample_id, "full")] = (True, sum(values) / len(values))
    for row in rows:
        sample_id, item = row["sample_id"], row["pair"].split(":")[0]
        if row["status"] != "ok":
            labels[(sample_id, item)] = (False, 1.0)
            continue
        gap = mean(row["scores"]["forced"]) - mean(row["scores"]["full"])
        labels[(sample_id, item)] = (gap >= -margin, labels[(sample_id, "full")][1] + gap)
    return labels


def load_traces(
    collected: Path,
    mask3d: dict,
    human: dict,
    labels: dict[tuple[str, str], tuple[bool, float | None]] | None = None,
    runs: Path | None = None,
) -> tuple[dict[str, list[QuestionTrace]], dict]:
    """Group graded questions by task. Without labels, tasks without program grading are left out."""
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
        grade = (
            (lambda answer, item, s=sample["sample_id"]: labels[(s, item)])
            if labels
            else (lambda answer, item, m=meta: (is_correct(answer, m, mask3d, human), None))
        )
        full_correct, full_score = grade(sample["full_answer"], "full")
        forced_grade = lambda row, m=meta, g=grade: (
            g(row["answer"], f"forced{row['checkpoint_idx']}")
            if labels
            else (is_correct(row["answer"], m, mask3d, human, cut_off=row["answer_tokens"] >= FORCED_ANSWER_LIMIT), None)
        )
        if full_correct is None:
            continue
        checkpoints = [
            Checkpoint(
                tokens=row["prefix_tokens"],
                correct=bool(forced_grade(row)[0]),
                score=forced_grade(row)[1],
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
                full_score=full_score,
                full_answer_tokens=(
                    json.loads((runs / f"{sample['sample_id']}_qwen3_vl.json").read_text(encoding="utf-8"))["answer_token_count"]
                    if runs
                    else None
                ),
            )
        )
        info[sample["sample_id"]] = sample
    return by_task, info


def detector_report(
    traces: list[QuestionTrace],
    info: dict,
    collected: Path,
    layers: list[int] | None = None,
    runs: Path | None = None,
    folds: int = 2,
    strengths: tuple[float, ...] = (1.0,),
    score_drop: float | None = None,
) -> dict:
    """Two-fold evaluation of the answer-readiness detector and its variants on one task."""
    import numpy as np
    import torch

    from thinkenough.detector import checkpoint_features, stop_at_threshold, two_fold_evaluate

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

    def evaluate(**options) -> dict:
        directions = two_fold_evaluate(
            traces,
            hidden_at_layer,
            layers or range(layer_count),
            budget,
            folds=folds,
            strengths=strengths,
            allowed_score_drop=score_drop,
            **options,
        )
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
                    "strength": result["strength"],
                    "validation_auroc": result["validation_auroc"],
                    "test_auroc": result["test_auroc"],
                    "threshold": result["threshold"],
                    **summarize(result["test_traces"], rule),
                }
            )
        return {"directions": report, "all_questions": summarize(traces, lambda t: combined[t.sample_id])}

    base = evaluate()
    chosen = {d["test_fold"]: d["layer"] for d in base["directions"]}
    share = lambda t, b: np.array([[c.tokens / b] for c in t.checkpoints])
    variants = {
        "detector": base,
        # Same layers as the main detector, so only the features differ.
        "tokens_only": evaluate(features=lambda hidden, t, b: share(t, b), layer_by_fold=chosen),
        "hidden_only": evaluate(
            features=lambda hidden, t, b: np.array([hidden(t, i) for i in range(len(t.checkpoints))]),
            layer_by_fold=chosen,
        ),
    }
    if runs:
        from transformers import AutoTokenizer

        rows = {}
        tokenizer = None
        for trace in traces:
            run = json.loads((runs / f"{trace.sample_id}_qwen3_vl.json").read_text(encoding="utf-8"))
            tokenizer = tokenizer or AutoTokenizer.from_pretrained(run["model_name"], revision=run["model_revision"])
            text = run["reasoning_text"] or ""
            cuts = char_positions(tokenizer, text, [c.tokens for c in trace.checkpoints])
            rows[trace.sample_id] = np.array(progress_features(text, run["question"], cuts)).reshape(len(trace.checkpoints), -1)
        variants["progress_only"] = evaluate(
            features=lambda hidden, t, b: np.concatenate([rows[t.sample_id], share(t, b)], axis=1), layer_by_fold=chosen
        )
        variants["detector_plus_progress"] = evaluate(
            features=lambda hidden, t, b: np.concatenate([checkpoint_features(hidden, t, b), rows[t.sample_id]], axis=1),
            layer_by_fold=chosen,
        )
    return variants


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare stopping rules")
    parser.add_argument("--collected", required=True, help="Output directory of collect_checkpoints.py")
    parser.add_argument("--out", required=True)
    parser.add_argument("--detector", action="store_true", help="Also train and evaluate the detector")
    parser.add_argument("--labels", default=None, help="Answer Accuracy rows, to grade free-text tasks")
    parser.add_argument("--official-labels", default=None, help="SCoT score rows of forced answers, to grade free-text tasks")
    parser.add_argument("--margin", type=float, default=1.0, help="With --official-labels: a forced answer this close to the full answer is right")
    parser.add_argument("--score-drop", type=float, default=None, help="With --official-labels: allowed mean score drop when choosing the threshold")
    parser.add_argument("--layers", type=int, nargs="+", default=None, help="Layers to search. Default: all")
    parser.add_argument("--runs", default=None, help="Full reasoning results, to add text-based progress features")
    parser.add_argument("--folds", type=int, default=2, help="Scene folds for the detector")
    parser.add_argument("--strengths", type=float, nargs="+", default=[1.0], help="Inverse regularization values to try on validation")
    parser.add_argument("--caps", type=int, nargs="+", default=[256, 512, 1024, 2048, 3072, 4096, 6144, 8192], help="Fixed caps to compare")
    args = parser.parse_args(argv)

    collected = Path(args.collected)
    mask3d = load_attributes("scannet_mask3d_val_attributes.pt")
    human = load_attributes("scannet_val_attributes.pt")
    labels = load_labels(Path(args.labels)) if args.labels else None
    if args.official_labels:
        labels = official_labels(Path(args.official_labels), args.margin)
    by_task, info = load_traces(collected, mask3d, human, labels, Path(args.runs) if args.runs else None)

    report = {}
    for task, traces in sorted(by_task.items()):
        report[task] = {
            "fixed_caps": {str(cap): summarize(traces, lambda t, c=cap: fixed_cap(t, c)) for cap in args.caps},
            "oracle": summarize(traces, oracle),
            "oracle_give_up": summarize(traces, oracle_give_up),
        }
        keys = {
            t.sample_id: [
                parse_answer_key(c.row["answer"], info[t.sample_id]["metadata"], c.row["answer_tokens"] >= FORCED_ANSWER_LIMIT)
                for c in t.checkpoints
            ]
            for t in traces
        }
        # shortcut: free text has no comparable answer, so answer stability is skipped there.
        for repeats in () if labels else (2, 3):
            summary = summarize(traces, lambda t, r=repeats: answer_stable(t, keys[t.sample_id], r))
            # This rule must really generate an answer at every check, unlike a detector, so count those tokens.
            stops = [answer_stable(t, keys[t.sample_id], repeats).stopped_at for t in traces]
            extra = sum(
                sum(c.row["answer_tokens"] for c in (t.checkpoints if stop is None else t.checkpoints[: stop + 1]))
                for t, stop in zip(traces, stops)
            )
            summary["forced_tokens_mean"] = extra / len(traces)
            full_total = summary.get("total_tokens_full_mean", summary["reasoning_tokens_full_mean"])
            summary["token_reduction_with_forced"] = 1 - (summary["reasoning_tokens_mean"] + extra / len(traces)) / full_total
            report[task][f"answer_stable_{repeats}"] = summary
        if args.detector:
            task_info = {t.sample_id: info[t.sample_id] for t in traces}
            report[task]["detectors"] = detector_report(
                traces,
                task_info,
                collected,
                args.layers,
                Path(args.runs) if args.runs else None,
                args.folds,
                tuple(args.strengths),
                args.score_drop,
            )

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
            + (f", score {s['score_full_mean']:.2f} -> {s['score_mean']:.2f}" if "score_mean" in s else "")
        )

    for task, result in report.items():
        print(f"{task}: {result['oracle']['n']} questions")
        for cap, summary in result["fixed_caps"].items():
            line(f"cap {cap}", summary)
        for repeats in () if "answer_stable_2" not in result else (2, 3):
            line(f"answer stable {repeats}", result[f"answer_stable_{repeats}"])
            print(f"    with its forced answers counted: {result[f'answer_stable_{repeats}']['token_reduction_with_forced']:.0%} less")
        line("first right answer (accuracy bound)", result["oracle"])
        line("first right, else stop at once (savings bound)", result["oracle_give_up"])
        for name, variant in result.get("detectors", {}).items():
            line(name, variant["all_questions"])
            print("    test AUROC", [round(d["test_auroc"], 3) for d in variant["directions"]], "layers", [d["layer"] for d in variant["directions"]], "strengths", [d["strength"] for d in variant["directions"]])
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
