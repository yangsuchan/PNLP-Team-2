"""Tests for metrics and the early-exit evaluator (SYNTHETIC data only).

요약: [4단계] 지표와 early-exit 평가 테스트. 숫자는 코드 확인용이며 실제 결과가 아니다.

These numbers only check code behaviour; they are not SQA3D / Qwen3 results.
Run from the repo root:
    .venv/bin/python -m pytest tests/ -v
"""

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thinkenough.evaluation import metrics as M  # noqa: E402
from thinkenough.evaluation.checkpoint_labeling import build_labels, load_references  # noqa: E402
from thinkenough.evaluation.early_exit_evaluator import (  # noqa: E402
    EvaluationInputError, ThresholdLeakageError, check_threshold_usage, evaluate_online, evaluate_probe,
    join_predictions, select_threshold, simulate_early_exit,
)

FIX = ROOT / "tests/fixtures"
read = lambda p: [json.loads(line) for line in open(p)]  # noqa: E731
RESULTS = read(FIX / "fake_evaluation_results.jsonl")
REFS = load_references([FIX / "fake_references.jsonl"])
CHECKPOINTS = read(FIX / "synthetic_checkpoints.jsonl")
LABELS, _ = build_labels(CHECKPOINTS, read(FIX / "fake_checkpoint_answers.jsonl"), REFS)
PREDS = read(FIX / "fake_probe_predictions.jsonl")


@pytest.fixture(scope="module")
def online():
    return evaluate_online(RESULTS, REFS, baseline="full_cot")


def value(d):
    return None if d.get("value") is None else round(d["value"], 2)


def test_fixtures_are_synthetic():
    assert all(r["synthetic"] for r in RESULTS + PREDS)


# ---------------------------------------------------------------- metric functions


def test_accuracy_formula():
    assert M.accuracy([1, 0, 1, 1])["value"] == 75.0
    assert M.accuracy([])["status"] == "undefined"


def test_retention_and_difference_are_separate():
    assert M.performance_retention(60.0, 80.0)["value"] == 75.0
    assert M.accuracy_difference(60.0, 80.0)["value"] == -20.0


def test_case15_full_accuracy_zero_is_undefined():
    assert M.performance_retention(10.0, 0.0)["status"] == "undefined"


def test_case15_full_tokens_zero_is_undefined():
    zero = {"value": 0, "status": "measured"}
    assert M.token_reduction(zero, {"value": 5, "status": "measured"})["status"] == "undefined"


def test_unrecorded_tokens_are_not_filled():
    m = M.mean_of([100, None], ["generated_ids", "generated_ids"], "reasoning tokens")
    assert m["status"] == "not_computable" and m["value"] is None


def test_token_provenance_status():
    assert M.mean_of([1, 2], ["generated_ids"] * 2, "t")["status"] == "measured"
    assert M.mean_of([1, 2], ["generated_ids", "retokenized"], "t")["status"] == "estimated"
    assert M.mean_of([1, 2], ["synthetic"] * 2, "t")["status"] == "synthetic"
    assert M.mean_of([1, 2], [None, None], "t")["status"] == "not_computable"


def test_synthetic_latency_is_flagged():
    lat = M.latency_summary([10.0, 20.0], ["synthetic", "synthetic"])
    assert lat["status"] == "synthetic" and "NOT a performance result" in lat["note"]


def test_probe_quality_single_class_auroc_undefined():
    q = M.probe_quality([1, 1, 1], [0.2, 0.5, 0.9])
    assert q["auroc"]["status"] == "undefined"


def test_case11_no_probe_probabilities():
    assert M.probe_quality([], [])["status"] == "not_computable"
    assert M.probe_quality([0, 1], [0.3, None])["status"] == "not_computable"


# ---------------------------------------------------------------- online comparison (cases 8-10, 13-15)


def test_case8_early_exit_lower_accuracy(online):
    ee = online["methods"]["probe_ee"]
    assert value(ee["accuracy_em"]) == 66.67 and value(online["methods"]["full_cot"]["accuracy_em"]) == 83.33
    assert value(ee["vs_baseline"]["performance_retention"]) == 80.0
    assert value(ee["vs_baseline"]["accuracy_difference"]) == -16.67
    assert value(ee["vs_baseline"]["reasoning_token_reduction"]) == 49.0


def test_case9_equal_accuracy(online):
    ee = online["methods"]["probe_ee_seed2"]
    assert value(ee["vs_baseline"]["performance_retention"]) == 100.0
    assert value(ee["vs_baseline"]["accuracy_difference"]) == 0.0


def test_case10_early_exit_uses_more_tokens(online):
    red = online["methods"]["probe_ee_seed2"]["vs_baseline"]["reasoning_token_reduction"]
    assert red["value"] < 0 and "MORE" in red["note"]


def test_missing_token_count_makes_reduction_not_computable(online):
    fixed = online["methods"]["fixed_2"]
    assert fixed["reasoning_tokens"]["status"] == "not_computable"
    assert fixed["vs_baseline"]["reasoning_token_reduction"]["status"] == "not_computable"


def test_synthetic_tokens_never_reported_as_measured(online):
    for res in online["methods"].values():
        assert res["reasoning_tokens"]["status"] in ("synthetic", "not_computable")


def test_official_metric_counts_empty_answer_but_em_does_not(online):
    fixed = online["methods"]["fixed_2"]
    assert value(fixed["accuracy_official"]) > value(fixed["accuracy_official_empty_as_wrong"])
    assert "not applied" in fixed["accuracy_official"]["unit_merge"]


def test_case13_duplicate_result_rows_rejected():
    with pytest.raises(EvaluationInputError):
        evaluate_online(RESULTS + [RESULTS[0]], REFS)


def test_case14_different_question_sets_rejected():
    rows = [r for r in RESULTS if not (r["method"] == "probe_ee" and r["question_id"] == 910000000006)]
    with pytest.raises(EvaluationInputError, match="same question set"):
        evaluate_online(rows, REFS)


def test_mixed_evaluation_modes_rejected():
    rows = copy.deepcopy(RESULTS)
    rows[-1]["evaluation_mode"] = "offline_simulation"
    with pytest.raises(EvaluationInputError, match="mixes evaluation modes"):
        evaluate_online(rows, REFS)


def test_case15_full_cot_accuracy_zero():
    rows = copy.deepcopy(RESULTS)
    for r in rows:
        if r["method"] == "full_cot":
            r["final_answer"] = "wrong"
    res = evaluate_online(rows, REFS)
    assert res["methods"]["probe_ee"]["vs_baseline"]["performance_retention"]["status"] == "undefined"


def test_case15_full_cot_zero_tokens():
    rows = copy.deepcopy(RESULTS)
    for r in rows:
        if r["method"] == "full_cot":
            r["reasoning_tokens"] = 0
    res = evaluate_online(rows, REFS)
    assert res["methods"]["probe_ee"]["vs_baseline"]["reasoning_token_reduction"]["status"] == "undefined"


def test_failed_generation_counts_as_wrong_in_same_denominator():
    rows = copy.deepcopy(RESULTS)
    for r in rows:
        if r["method"] == "full_cot" and r["question_id"] == 910000000001:
            r["generation_status"] = "error"
    acc = evaluate_online(rows, REFS)["methods"]["full_cot"]["accuracy_em"]
    assert acc["total"] == 6 and acc["correct"] == 4


# ---------------------------------------------------------------- probe threshold protocol (cases 11-12)


def test_threshold_selected_on_validation_only():
    joined, _ = join_predictions(LABELS, PREDS)
    rec = select_threshold([r for r in joined if r["split"] == "validation"])
    assert rec["selected_on_split"] == "validation" and rec["threshold"] is not None
    test_eval = evaluate_probe(joined, "test", rec)
    assert test_eval["at_threshold"]["threshold"] == rec["threshold"]


def test_case12_test_data_in_threshold_selection_rejected():
    joined, _ = join_predictions(LABELS, PREDS)
    with pytest.raises(ThresholdLeakageError):
        select_threshold([r for r in joined if r["split"] in ("validation", "test")])
    with pytest.raises(ThresholdLeakageError):
        select_threshold([r for r in joined if r["split"] == "test"])


def test_case12_threshold_not_from_validation_rejected():
    with pytest.raises(ThresholdLeakageError):
        check_threshold_usage({"threshold": 0.5, "selected_on_split": "test", "selection_question_ids": []},
                              [{"question_id": 1, "split": "test"}])


def test_case12_evaluating_on_selection_questions_rejected():
    joined, _ = join_predictions(LABELS, PREDS)
    rec = select_threshold([r for r in joined if r["split"] == "validation"])
    leaked = [dict(r, split="test") for r in joined if r["split"] == "validation"]
    with pytest.raises(ThresholdLeakageError):
        evaluate_probe(leaked, "test", rec)


def test_target_precision_criterion():
    joined, _ = join_predictions(LABELS, PREDS)
    rec = select_threshold([r for r in joined if r["split"] == "validation"], criterion="target_precision", target_precision=1.0)
    assert rec["metrics_at_threshold_on_validation"]["precision"] == 1.0


def test_case11_checkpoints_without_probe_predictions():
    joined, no_pred = join_predictions(LABELS, [])
    assert joined == [] and len(no_pred) > 0
    assert evaluate_probe(joined, "test")["status"] == "not_computable"


# ---------------------------------------------------------------- offline simulation


def test_simulation_is_labeled_and_excludes_incomplete_questions():
    joined, _ = join_predictions(LABELS, PREDS)
    rec = select_threshold([r for r in joined if r["split"] == "validation"])
    sim = simulate_early_exit(CHECKPOINTS, LABELS, PREDS, rec, "test")
    assert sim["mode"] == "offline_simulation" and "not an online" in sim["note"]
    assert set(sim["excluded_questions"]) == {900000000007, 900000000010}
    counts = {m: r["accuracy_em"]["total"] for m, r in sim["methods"].items()}
    assert len(set(counts.values())) == 1  # same denominator for every simulated method
    # q9 labels [0, 0, 1, 1], p = [0.2, 0.45, 0.75, 0.85]; tau from validation = 0.55 -> exit at 2
    assert rec["threshold"] == 0.55
    assert sim["methods"]["probe_sim"]["mean_exit_checkpoint"] == 2.0
    assert sim["methods"]["oracle_sim"]["mean_exit_checkpoint"] == 2.0


def test_simulation_rejects_threshold_selected_on_eval_questions():
    rec = {"threshold": 0.5, "selected_on_split": "validation", "selection_question_ids": [900000000009]}
    with pytest.raises(ThresholdLeakageError):
        simulate_early_exit(CHECKPOINTS, LABELS, PREDS, rec, "test")
