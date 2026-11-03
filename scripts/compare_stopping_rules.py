#!/usr/bin/env python3
"""Compare stopping rules on one task, with thresholds picked from all training questions.

Reads the hidden-state detector's layers from analyze_checkpoints.py's report. Each test fold's
threshold comes from inner cross-validation on the other scenes. Rules that must generate forced
answers are charged for them, both at full length and at a shorter limit, which is simulated by
cutting the saved answers with the tokenizer and grading them again.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from analyze_checkpoints import FORCED_ANSWER_LIMIT, load_traces  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from thinkenough.checkpoints import QuestionTrace, answer_stable, fixed_cap, run_rule, summarize  # noqa: E402
from thinkenough.detector import (  # noqa: E402
    ReadinessDetector,
    auroc,
    checkpoint_features,
    choose_threshold,
    fold_of,
    stop_at_threshold,
)
from thinkenough.evidence import answer_keys, evidence_features, load_run  # noqa: E402
from thinkenough.scot import is_correct, load_attributes  # noqa: E402

FOLDS = 5
INNER_FOLDS = 4
STABILITY_COLUMNS = [0, 1, 2, 3]
ALL_COLUMNS = list(range(8))


def labels(group: list[QuestionTrace]) -> np.ndarray:
    return np.array([c.correct for t in group for c in t.checkpoints])


def forced_tokens(trace: QuestionTrace, stop: int | None) -> int:
    used = trace.checkpoints if stop is None else trace.checkpoints[: stop + 1]
    return sum(c.row["answer_tokens"] for c in used)


def evaluate(traces: list[QuestionTrace], features, strength, charge_forced: bool) -> dict:
    """Train on the other scenes, pick the threshold by inner folds, and stop on the test fold."""
    stops: dict[str, int | None] = {}
    fold_aurocs, within = [], []
    for fold in range(FOLDS):
        test = [t for t in traces if fold_of(t.scene_id, FOLDS) == fold]
        train = [t for t in traces if fold_of(t.scene_id, FOLDS) != fold]
        rows = lambda t, f=fold: features(t, f)
        inner: dict[str, np.ndarray] = {}
        for part in range(INNER_FOLDS):
            fit = [t for t in train if fold_of(t.scene_id, INNER_FOLDS, "inner") != part]
            held = [t for t in train if fold_of(t.scene_id, INNER_FOLDS, "inner") == part]
            if not held or len(set(labels(fit).tolist())) < 2:
                continue
            detector = ReadinessDetector(strength=strength(fold)).fit(np.concatenate([rows(t) for t in fit]), labels(fit))
            inner.update({t.sample_id: detector.score(rows(t)) for t in held})
        known = [t for t in train if t.sample_id in inner]
        threshold = choose_threshold(known, [inner[t.sample_id] for t in known])
        detector = ReadinessDetector(strength=strength(fold)).fit(np.concatenate([rows(t) for t in train]), labels(train))
        scores = [detector.score(rows(t)) for t in test]
        for trace, score in zip(test, scores):
            stops[trace.sample_id] = stop_at_threshold(trace, score, threshold).stopped_at
            right = np.array([c.correct for c in trace.checkpoints])
            if right.any() and not right.all():
                within.append(auroc(right, np.array(score)))
        fold_aurocs.append(auroc(labels(test), np.concatenate(scores)))
    summary = summarize(traces, lambda t: run_rule(t, stops[t.sample_id]))
    report = {
        "accuracy": summary["accuracy"],
        "accuracy_full": summary["accuracy_full"],
        "ci": summary["accuracy_change_ci"],
        "saved": summary["token_reduction"],
        "auroc": float(np.mean(fold_aurocs)),
        "auroc_within_question": float(np.mean(within)),
        "stops": stops,
    }
    if charge_forced:
        extra = sum(forced_tokens(t, stops[t.sample_id]) for t in traces) / len(traces)
        report["saved_with_forced"] = 1 - (summary["reasoning_tokens_mean"] + extra) / summary["total_tokens_full_mean"]
    return report


def stable_rule(traces: list[QuestionTrace], keys: dict[str, list], repeats: int) -> dict:
    rule = lambda t: answer_stable(t, keys[t.sample_id], repeats)
    summary = summarize(traces, rule)
    extra = sum(forced_tokens(t, rule(t).stopped_at) for t in traces) / len(traces)
    return {
        "accuracy": summary["accuracy"],
        "ci": summary["accuracy_change_ci"],
        "saved": summary["token_reduction"],
        "saved_with_forced": 1 - (summary["reasoning_tokens_mean"] + extra) / summary["total_tokens_full_mean"],
        "stops": {t.sample_id: rule(t).stopped_at for t in traces},
    }


def paired_gap(traces: list[QuestionTrace], stops: dict, against: dict, seed: int = 0) -> dict:
    """Accuracy and with-forced saving against another rule, paired by question."""
    rows = []
    for t in traces:
        a, b = stops[t.sample_id], against[t.sample_id]
        right = lambda i: t.full_correct if i is None else t.checkpoints[i].correct
        cost = lambda i: (t.full_tokens if i is None else t.checkpoints[i].tokens) + forced_tokens(t, i)
        rows.append((float(right(a)) - float(right(b)), cost(b) - cost(a), t.full_tokens + t.full_answer_tokens))
    rng = random.Random(seed)
    accuracy, saving = [], []
    for _ in range(4000):
        draw = rng.choices(rows, k=len(rows))
        accuracy.append(sum(r[0] for r in draw) / len(draw))
        saving.append(sum(r[1] for r in draw) / sum(r[2] for r in draw))
    accuracy.sort()
    saving.sort()
    return {
        "accuracy": sum(r[0] for r in rows) / len(rows),
        "accuracy_ci": [accuracy[100], accuracy[3899]],
        "saving": sum(r[1] for r in rows) / sum(r[2] for r in rows),
        "saving_ci": [saving[100], saving[3899]],
    }


def with_answer_limit(traces, runs: Path, tokenizer, limit: int, mask3d, human) -> list[QuestionTrace]:
    """The same traces if forced answers stopped at ``limit`` tokens."""
    limited = []
    for trace in traces:
        copied = copy.copy(trace)
        meta = load_run(runs, trace.sample_id)["metadata"]
        checkpoints = []
        for c in trace.checkpoints:
            if c.row["answer_tokens"] > limit:
                answer = tokenizer.decode(tokenizer(c.row["answer"], add_special_tokens=False)["input_ids"][:limit])
                right = bool(is_correct(answer, meta, mask3d, human, cut_off=True))
                checkpoints.append(dataclasses.replace(c, correct=right, row={**c.row, "answer": answer, "answer_tokens": limit}))
            else:
                checkpoints.append(c)
        copied.checkpoints = checkpoints
        limited.append(copied)
    return limited


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collected", required=True, help="Folder from collect_checkpoints.py")
    parser.add_argument("--runs", required=True, help="Full reasoning results")
    parser.add_argument("--task", required=True, choices=["relationship_analysis", "implicit_detection"])
    parser.add_argument("--report", required=True, help="analyze_checkpoints.py output, for the detector's layers")
    parser.add_argument("--answer-limit", type=int, default=200, help="Shorter forced answers to simulate")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    collected, runs = Path(args.collected), Path(args.runs)
    mask3d = load_attributes("scannet_mask3d_val_attributes.pt")
    human = load_attributes("scannet_val_attributes.pt")
    by_task, info = load_traces(collected, mask3d, human, None, runs)
    traces = by_task[args.task]
    budget = max(json.loads(line)["budget"] for line in (collected / "samples.jsonl").read_text(encoding="utf-8").splitlines())
    base = json.loads(Path(args.report).read_text(encoding="utf-8"))[args.task]["detectors"]["detector"]["directions"]
    picked = {d["test_fold"]: (d["layer"], d["strength"]) for d in base}
    first = load_run(runs, traces[0].sample_id)
    tokenizer = AutoTokenizer.from_pretrained(first["model_name"], revision=first["model_revision"])

    cache: dict[tuple[str, int], np.ndarray] = {}

    def hidden_at(layer: int):
        def hidden(trace: QuestionTrace, index: int | None) -> np.ndarray:
            path = info[trace.sample_id]["start_hidden_path"] if index is None else trace.checkpoints[index].row["hidden_state_path"]
            if (path, layer) not in cache:
                cache[(path, layer)] = torch.load(collected / path, map_location="cpu", weights_only=True)[layer].float().numpy()
            return cache[(path, layer)]

        return hidden

    probe = lambda t, fold: checkpoint_features(hidden_at(picked[fold][0]), t, budget)
    probe_strength = lambda fold: picked[fold][1]
    report = {
        "hidden_probe": evaluate(traces, probe, probe_strength, False),
        "tokens_only": evaluate(traces, lambda t, f: np.array([[c.tokens / budget] for c in t.checkpoints]), lambda f: 1e-4, False),
        "caps": {
            str(cap): {
                k: v
                for k, v in summarize(traces, lambda t, c=cap: fixed_cap(t, c)).items()
                if k in ("accuracy", "accuracy_change_ci", "token_reduction")
            }
            for cap in (2048, 4096, 6144)
        },
    }
    limited = with_answer_limit(traces, runs, tokenizer, args.answer_limit, mask3d, human)
    for name, group, limit in (("full_answers", traces, FORCED_ANSWER_LIMIT), ("limited_answers", limited, args.answer_limit)):
        evidence = {t.sample_id: evidence_features(t, load_run(runs, t.sample_id), tokenizer, budget, limit) for t in group}
        keys = {t.sample_id: answer_keys(t, load_run(runs, t.sample_id)["metadata"], limit) for t in group}
        section = {
            "stable_2": stable_rule(group, keys, 2),
            "stable_3": stable_rule(group, keys, 3),
            "stability": evaluate(group, lambda t, f: evidence[t.sample_id][:, STABILITY_COLUMNS], lambda f: 1.0, True),
            "stability_and_evidence": evaluate(group, lambda t, f: evidence[t.sample_id][:, ALL_COLUMNS], lambda f: 1.0, True),
            "hidden_stability_and_evidence": evaluate(
                group, lambda t, f: np.c_[probe(t, f), evidence[t.sample_id]], probe_strength, True
            ),
        }
        section["evidence_vs_stable_2"] = paired_gap(group, section["stability_and_evidence"]["stops"], section["stable_2"]["stops"])
        report[name] = section

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    line = lambda n, r: print(
        f"  {n}: accuracy {r['accuracy']:.2f}, saved {r['saved']:.0%}"
        + (f", with forced {r['saved_with_forced']:.0%}" if "saved_with_forced" in r else "")
        + (f" | AUROC {r['auroc']:.2f}, within question {r['auroc_within_question']:.2f}" if "auroc" in r else "")
    )
    print(f"{args.task}: {len(traces)} questions, full accuracy {report['hidden_probe']['accuracy_full']:.2f}")
    line("hidden probe", report["hidden_probe"])
    line("tokens only", report["tokens_only"])
    for name in ("full_answers", "limited_answers"):
        print(f" {name}")
        for rule in ("stable_2", "stable_3", "stability", "stability_and_evidence", "hidden_stability_and_evidence"):
            line(rule, report[name][rule])
        gap = report[name]["evidence_vs_stable_2"]
        print(f"  evidence minus stable 2: accuracy {gap['accuracy']:+.3f}, with-forced saving {gap['saving']:+.3f} {[round(x, 3) for x in gap['saving_ci']]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
