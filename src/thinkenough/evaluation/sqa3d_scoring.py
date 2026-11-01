"""SQA3D answer scoring.

요약: [1단계 데이터, 전 단계 공용] SQA3D 채점. 공식 채점 방식을 그대로 옮긴 것과 주 지표인 정규화 EM.

Two metrics:
- official: a faithful port of `clean_answer` + `compute_accuracy` from the
  official LLM evaluation script
  https://github.com/SilongYong/SQA3D/blob/master/LLM/eval_sqa3d_llm.py
  (the optional GPT-based metric is not ported because it needs a paid API).
- normalized EM: the same cleaning applied to BOTH prediction and reference,
  then case-insensitive exact match.

Known quirks of the official metric (kept as-is for faithfulness):
- only the prediction is cleaned; references are compared raw, so a pure-digit
  reference ("2") can never match because the prediction's "2" becomes "two"
  (no SQA3D reference is pure-digit; ones like "180 degrees" still match via
  the shared-word rule);
- an empty prediction counts as correct ("" is a substring of every string);
  pass `empty_pred_is_wrong=True` to disable this;
- any shared word counts as correct ("right side" vs "left side").
"""

import json
import re

_DIGIT_WORDS = [
    ("0", "zero"), ("none", "zero"), ("1", "one"), ("2", "two"), ("3", "three"),
    ("4", "four"), ("5", "five"), ("6", "six"), ("7", "seven"), ("8", "eight"),
    ("9", "nine"), ("10", "ten"), ("11", "eleven"), ("12", "twelve"),
    ("13", "thirteen"), ("14", "fourteen"), ("15", "fifteen"), ("16", "sixteen"),
    ("17", "seventeen"), ("18", "eighteen"), ("19", "nineteen"), ("20", "twenty"),
    ("23", "twenty-three"),
]


def clean_answer(text):
    """Port of the official `clean_answer`, applied to a single string."""
    s = text.lower()
    s = re.sub("[ ]+$", "", s)
    s = re.sub("^[ ]+", "", s)
    s = re.sub(" {2,}", " ", s)

    s = re.sub(r"\.[ ]{2,}", ". ", s)
    s = re.sub(r"[^a-zA-Z0-9,\'\s\-:]+", "", s)
    s = re.sub("ç", "c", s)
    s = re.sub("’", "'", s)
    s = re.sub(r"\bletf\b", "left", s)
    s = re.sub(r"\blet\b", "left", s)
    s = re.sub(r"\btehre\b", "there", s)
    s = re.sub(r"\brigth\b", "right", s)
    s = re.sub(r"\brght\b", "right", s)
    s = re.sub(r"\bbehine\b", "behind", s)
    s = re.sub(r"\btv\b", "TV", s)
    s = re.sub(r"\bchai\b", "chair", s)
    s = re.sub(r"\bwasing\b", "washing", s)
    s = re.sub(r"\bwaslked\b", "walked", s)
    s = re.sub(r"\boclock\b", "o'clock", s)
    s = re.sub(r"\bo\'[ ]+clock\b", "o'clock", s)

    # digit to word, only for answer
    for digit, word in _DIGIT_WORDS:
        s = re.sub(rf"\b{digit}\b", word, s)

    # misc
    # no1, mat2, etc
    s = re.sub(r"\b([a-zA-Z]+)([0-9])\b", r"\g<1>", s)
    s = re.sub(r"\ba\b ([a-zA-Z]+)", r"\g<1>", s)
    s = re.sub(r"\ban\b ([a-zA-Z]+)", r"\g<1>", s)
    s = re.sub(r"\bthe\b ([a-zA-Z]+)", r"\g<1>", s)

    s = re.sub(r"\bbackwards\b", "backward", s)
    return s


def official_match(pred, reference, empty_pred_is_wrong=False):
    """Official non-GPT matching rule for one (prediction, reference) pair."""
    p = clean_answer(pred)
    if empty_pred_is_wrong and not p.strip():
        return False
    if p == reference:
        return True
    if p in reference:
        return True
    if "".join(p.split()) in "".join(reference.split()):
        return True
    return len(set(p.split()).intersection(reference.split())) > 0


def normalized_em(pred, reference):
    """Same cleaning on both sides, then case-insensitive exact match."""
    p = clean_answer(pred).strip().lower()
    r = clean_answer(reference).strip().lower()
    return bool(p) and p == r


def _official_key(record):
    meta = record["meta"]
    return (
        record["input"]["scene_id"],
        json.dumps(meta["rotation"]),
        json.dumps(meta["position"]),
        record["input"]["question"],
        record["input"]["situation"],
    )


def official_accuracy(records, predictions, empty_pred_is_wrong=False):
    """Port of the official `compute_accuracy`.

    Like the original, items sharing (scene, rotation, position, question,
    situation) are merged into one unit scored against the LAST such item's
    reference, and a unit is correct if any of its predictions matches.

    records: processed JSONL records (in file order).
    predictions: {question_id: predicted answer string}.
    """
    reference_by_key = {}
    for r in records:
        reference_by_key[_official_key(r)] = r["reference_answers"][0]
    correct = {}
    for r in records:
        if r["question_id"] not in predictions:
            continue
        key = _official_key(r)
        correct.setdefault(key, 0)
        if official_match(predictions[r["question_id"]], reference_by_key[key], empty_pred_is_wrong):
            correct[key] += 1
    total = len(correct)
    hits = sum(1 for v in correct.values() if v >= 1)
    return {"accuracy": hits / total if total else 0.0, "correct": hits, "total": total}


def em_accuracy(records, predictions):
    """Per-question normalized EM over records that have a prediction."""
    scored = [normalized_em(predictions[r["question_id"]], r["reference_answers"][0])
              for r in records if r["question_id"] in predictions]
    return {"accuracy": sum(scored) / len(scored) if scored else 0.0,
            "correct": sum(scored), "total": len(scored)}
