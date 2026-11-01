"""Tests for STOP/CONTINUE labeling and answer extraction (SYNTHETIC data only).

요약: [3단계] 답 추출과 STOP/CONTINUE 라벨 테스트 (가짜 데이터 사용).

Run from the repo root:
    .venv/bin/python -m pytest tests/ -v
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thinkenough.evaluation.answer_extraction import extract_final_answer  # noqa: E402
from thinkenough.evaluation.checkpoint_labeling import (  # noqa: E402
    LabelingInputError, build_labels, label_answer, load_references, training_labels,
)

FIX = ROOT / "tests/fixtures"
CHECKPOINTS = [json.loads(line) for line in open(FIX / "synthetic_checkpoints.jsonl")]
ANSWERS = [json.loads(line) for line in open(FIX / "fake_checkpoint_answers.jsonl")]
REFS = load_references([FIX / "fake_references.jsonl"])
Q = lambda n: 900000000000 + n  # noqa: E731  synthetic question ids


@pytest.fixture(scope="module")
def labeled():
    rows, report = build_labels(CHECKPOINTS, ANSWERS, REFS)
    return rows, report


def seq(rows, n, sample=0):
    """stop_label sequence of synthetic question n (one sample index)."""
    return [r["stop_label"] for r in sorted(rows, key=lambda r: r["checkpoint_index"])
            if r["question_id"] == Q(n) and (r.get("sample_index") in (sample, None))]


def test_fixtures_are_synthetic():
    assert all(a["synthetic"] for a in ANSWERS)
    assert all(r["synthetic"] for r in REFS.values())


# ---------------------------------------------------------------- answer extraction


@pytest.mark.parametrize("raw, expected", [
    ("\n\nbrown<|im_end|>", ("brown", "ok")),
    ("<think>\nreasoning\n</think>\n\nleft", ("left", "ok")),
    ("A: left.", ("left", "ok")),
    ("The answer is **two**.", ("two", "ok")),
    ("Final answer: table\nBecause the table ...", ("table", "ok")),
    ("a brown chair", ("a brown chair", "ok")),
    ("", ("", "empty")),
    ("<think>x</think>\n\n", ("", "empty")),
    ("<think>\nstill thinking", ("", "think_unclosed")),
    (None, ("", "no_output")),
])
def test_extract_final_answer(raw, expected):
    assert extract_final_answer(raw) == expected


def test_only_final_answer_is_scored():
    """Reasoning that mentions the gold answer must not make the label STOP."""
    a = {"generation_status": "ok", "generated_answer": "<think>maybe brown</think>\n\nwhite"}
    assert label_answer(a, ["brown"])["stop_label"] == 0


# ---------------------------------------------------------------- Part C cases 1-7


def test_case1_all_correct(labeled):
    assert seq(labeled[0], 2) == [1, 1]


def test_case2_all_wrong(labeled):
    assert seq(labeled[0], 4) == [0, 0, 0, 0, 0]


def test_case3_first_correct_in_the_middle(labeled):
    assert seq(labeled[0], 1) == [0, 0, 1, 1, 1, 1]


def test_case4_correct_then_wrong_is_not_propagated(labeled):
    """Checkpoint labels are independent: STOP at 1 does not force STOP at 2/3."""
    assert seq(labeled[0], 3) == [0, 1, 0, 0, 1]


def test_case5_correct_only_at_last(labeled):
    assert seq(labeled[0], 5) == [0, 1]


def test_case6_missing_answers_are_not_labels(labeled):
    rows = labeled[0]
    q7 = {r["checkpoint_index"]: r for r in rows if r["question_id"] == Q(7)}
    assert q7[1]["label_status"] == "pending" and q7[1]["stop_label"] is None
    assert q7[2]["label_status"] == "missing" and q7[2]["stop_label"] is None
    q10 = [r for r in rows if r["question_id"] == Q(10)]
    assert q10 and all(r["label_status"] == "pending" for r in q10)
    assert not any(r["question_id"] == Q(10) for r in training_labels(rows))


def test_case7_empty_and_invalid_answers_are_continue(labeled):
    q4 = {r["checkpoint_index"]: r for r in labeled[0] if r["question_id"] == Q(4)}
    assert q4[2]["stop_label"] == 0 and "empty_answer" in q4[2]["flags"]
    assert q4[3]["stop_label"] == 0 and "invalid_format" in q4[3]["flags"]


def test_empty_answer_is_wrong_even_for_official_match():
    r = label_answer({"generation_status": "ok", "generated_answer": ""}, ["brown"])
    assert r["stop_label"] == 0 and r["official_correct"] is False


def test_official_loose_match_is_recorded_but_not_the_label(labeled):
    q3 = {r["checkpoint_index"]: r for r in labeled[0] if r["question_id"] == Q(3)}
    assert q3[2]["extracted_answer"] == "not left"
    assert q3[2]["official_correct"] is True and q3[2]["stop_label"] == 0
    assert "official_only_match" in q3[2]["flags"]
    loose = label_answer({"generation_status": "ok", "generated_answer": "right side"}, ["left side"])
    assert loose["official_correct"] is True and loose["stop_label"] == 0


def test_checkpoint_zero_direct_answer(labeled):
    q6 = [r for r in labeled[0] if r["question_id"] == Q(6)]
    assert len(q6) == 1 and q6[0]["checkpoint_type"] == "no_reasoning" and q6[0]["stop_label"] == 1


def test_multiple_samples_kept_without_majority_vote(labeled):
    q9 = [r for r in labeled[0] if r["question_id"] == Q(9) and r["checkpoint_index"] == 1]
    assert sorted((r["sample_index"], r["stop_label"]) for r in q9) == [(0, 0), (1, 1)]
    assert {r["generation_config"]["seed"] for r in q9} == {1, 2}


def test_labels_carry_no_probe_outputs(labeled):
    assert not any(k in r for r in labeled[0] for k in ("p_stop", "probe_score", "threshold"))


def test_orphan_answer_is_reported(labeled):
    assert labeled[1]["orphan_answer_checkpoint_ids"] == ["synthetic_run01/900000000099/sentence/0"]


# ---------------------------------------------------------------- case 13 (labeling side)


def test_case13_duplicate_answer_records_rejected():
    with pytest.raises(LabelingInputError):
        build_labels(CHECKPOINTS, ANSWERS + [ANSWERS[0]], REFS)


def test_case13_duplicate_checkpoint_ids_rejected():
    with pytest.raises(LabelingInputError):
        build_labels(CHECKPOINTS + [CHECKPOINTS[0]], ANSWERS, REFS)


def test_case13_answer_disagreeing_with_checkpoint_rejected():
    bad = dict(ANSWERS[0], checkpoint_index=99)
    with pytest.raises(LabelingInputError):
        build_labels(CHECKPOINTS, [bad], REFS)


def test_missing_reference_is_not_labeled():
    rows, _ = build_labels(CHECKPOINTS, ANSWERS, {})
    assert not training_labels(rows)
    assert {r["label_status"] for r in rows} <= {"no_reference", "pending"}
