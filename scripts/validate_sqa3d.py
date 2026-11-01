"""Validate processed SQA3D files written by scripts/prepare_sqa3d.py.

요약: [1단계] 전처리 결과 검사(개수, 중복, 장면 겹침, 정답 유출, 채점기 동작).

Checks counts vs raw files, question_id uniqueness, cross-split scene overlap,
missing answers, missing captions, answer leakage into the model input, pilot
consistency, and scorer sanity. Results are added to
outputs/preprocessing_summary.json under "validation"; exit code 1 on failure.

Usage (from the repo root):
    .venv/bin/python scripts/validate_sqa3d.py
"""

import copy
import json
import re
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thinkenough.data.sqa3d_prompt import INPUT_KEYS, build_prompt  # noqa: E402
from thinkenough.evaluation.sqa3d_scoring import (  # noqa: E402
    em_accuracy, normalized_em, official_accuracy, official_match,
)

RAW = ROOT / "data/raw/sqa3d/extracted/sqa_task/balanced"
OUT_DIR = ROOT / "data/processed/sqa3d"
SUMMARY = ROOT / "outputs/preprocessing_summary.json"
SPLITS = {"train": "train", "validation": "val", "test": "test"}
PILOT_SIZES = {"train": 300, "validation": 100, "test": 100}
SENTINEL = "__LEAK_SENTINEL_9f3a__"
ANSWER_KEYS = {"reference_answers", "answers", "answer"}

results = []


def check(name, ok, detail=None):
    results.append({"check": name, "pass": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -> {detail}" if detail is not None else ""))


def read_jsonl(path):
    return [json.loads(line) for line in open(path)]


def main():
    full = {s: read_jsonl(OUT_DIR / f"{s}.jsonl") for s in SPLITS}
    pilot = {s: read_jsonl(OUT_DIR / "pilot" / f"{s}.jsonl") for s in SPLITS}
    missing_list = json.load(open(ROOT / "outputs/missing_captions.json"))
    conflicts = json.load(open(ROOT / "outputs/label_conflicts.json"))

    # 1. counts vs raw official files
    for s, official in SPLITS.items():
        raw_q = len(json.load(open(RAW / f"v1_balanced_questions_{official}_scannetv2.json"))["questions"])
        raw_a = len(json.load(open(RAW / f"v1_balanced_sqa_annotations_{official}_scannetv2.json"))["annotations"])
        check(f"{s}: processed count == raw count", len(full[s]) == raw_q == raw_a,
              {"raw_questions": raw_q, "raw_annotations": raw_a, "processed": len(full[s])})
        check(f"pilot/{s}: size == {PILOT_SIZES[s]}", len(pilot[s]) == PILOT_SIZES[s], len(pilot[s]))

    # 2. question_id uniqueness (within and across splits)
    all_qids = [r["question_id"] for s in SPLITS for r in full[s]]
    check("question_id unique across all splits", len(all_qids) == len(set(all_qids)),
          {"total": len(all_qids), "unique": len(set(all_qids))})

    # 3. scene / question overlap across splits
    for a, b in combinations(SPLITS, 2):
        sa = {r["input"]["scene_id"] for r in full[a]}
        sb = {r["input"]["scene_id"] for r in full[b]}
        check(f"no scene_id overlap {a}-{b}", not (sa & sb), sorted(sa & sb)[:5])
        ta = {(r["input"]["scene_id"], r["input"]["situation"], r["input"]["question"]) for r in full[a]}
        tb = {(r["input"]["scene_id"], r["input"]["situation"], r["input"]["question"]) for r in full[b]}
        check(f"no (scene, situation, question) overlap {a}-{b}", not (ta & tb), len(ta & tb))

    # 4. scene_id consistency and missing answers
    for s in SPLITS:
        mismatch = sum(r["input"]["scene_id"] != r["meta"]["annotation_scene_id"] for r in full[s])
        check(f"{s}: question/annotation scene_id agree", mismatch == 0, mismatch)
        no_answer = [r["question_id"] for r in full[s] if not r["reference_answers"] or not r["reference_answers"][0].strip()]
        check(f"{s}: every record has a reference answer", not no_answer, len(no_answer))

    # 5. captions
    for s in SPLITS:
        miss = sorted({r["input"]["scene_id"] for r in full[s] if r["meta"]["caption_status"] == "missing"})
        listed = sorted(m["scene_id"] for m in missing_list if m["split"] == s)
        check(f"{s}: caption-missing scenes are all recorded", miss == listed, miss)
        empty_ok = [r["question_id"] for r in full[s] if r["meta"]["caption_status"] == "ok" and not r["input"]["scene_captions"]]
        check(f"{s}: no silent empty captions", not empty_ok, len(empty_ok))
        over = [r["question_id"] for r in full[s] if len(r["input"]["scene_captions"]) > 30]
        check(f"{s}: <= 30 captions per record", not over, len(over))
        per_scene = {}
        consistent = True
        for r in full[s]:
            key = json.dumps(r["input"]["scene_captions"])
            consistent &= per_scene.setdefault(r["input"]["scene_id"], key) == key
        check(f"{s}: same captions for every question of a scene (question-independent)", consistent)
        check(f"pilot/{s}: no caption-missing records",
              all(r["meta"]["caption_status"] == "ok" for r in pilot[s]))

    # 6. answer leakage into the model input
    leak_rates = {}
    for s in SPLITS:
        bad_keys = [r["question_id"] for r in full[s] if set(r["input"]) != INPUT_KEYS]
        check(f"{s}: input has exactly the allowed keys {sorted(INPUT_KEYS)}", not bad_keys, len(bad_keys))
        nested = [r["question_id"] for r in full[s]
                  if any(k in ANSWER_KEYS for c in r["input"]["scene_captions"] for k in c)]
        check(f"{s}: no answer keys inside captions", not nested, len(nested))
        changed = 0
        sentinel_hits = 0
        for r in full[s]:
            original = build_prompt(r)
            tampered = copy.deepcopy(r)
            tampered["reference_answers"] = [SENTINEL]
            tampered["meta"] = {k: SENTINEL for k in tampered["meta"]}
            tampered["source"] = {k: SENTINEL for k in tampered["source"]}
            p = build_prompt(tampered)
            changed += p != original
            sentinel_hits += SENTINEL in p
        check(f"{s}: prompt unchanged when answers/meta/source are tampered", changed == 0 and sentinel_hits == 0,
              {"changed": changed, "sentinel_in_prompt": sentinel_hits})
        # informational: the answer word naturally appearing in captions/situation/question is not leakage
        hits = Counter()
        for r in full[s]:
            ans = r["reference_answers"][0].lower()
            pat = re.compile(rf"\b{re.escape(ans)}\b")
            inp = r["input"]
            hits["captions"] += bool(pat.search(" ".join(c["caption"].lower() for c in inp["scene_captions"])))
            hits["situation_or_question"] += bool(pat.search((inp["situation"] + " " + inp["question"]).lower()))
        leak_rates[s] = {k: round(v / len(full[s]), 4) for k, v in hits.items()}
    check("info: answer word appears in input text (natural, not a leak)", True, leak_rates)

    # 7. pilot consistency
    for s in SPLITS:
        by_qid = {r["question_id"]: r for r in full[s]}
        check(f"pilot/{s}: subset of full {s} with identical records",
              all(r["question_id"] in by_qid and by_qid[r["question_id"]] == r for r in pilot[s]))
        check(f"pilot/{s}: no flagged records", all(not r["meta"]["flags"] for r in pilot[s]))
    for a, b in combinations(SPLITS, 2):
        sa = {r["input"]["scene_id"] for r in pilot[a]}
        sb = {r["input"]["scene_id"] for r in pilot[b]}
        check(f"pilot: no scene_id overlap {a}-{b}", not (sa & sb))

    # 8. label conflicts recorded, not modified
    flagged = sum("label_conflict" in r["meta"]["flags"] for s in SPLITS for r in full[s])
    listed = sum(len(c["items"]) for c in conflicts)
    check("label conflicts: groups recorded and records flagged", flagged == listed,
          {"groups": len(conflicts), "flagged_records": flagged})

    # 9. scorer sanity: references as predictions
    scorer = {}
    for s in SPLITS:
        preds = {r["question_id"]: r["reference_answers"][0] for r in full[s]}
        off = official_accuracy(full[s], preds)
        em = em_accuracy(full[s], preds)
        failing = sorted({r["reference_answers"][0] for r in full[s]
                          if not official_match(r["reference_answers"][0], r["reference_answers"][0])})
        scorer[s] = {"official_on_gold": round(off["accuracy"], 4), "official_units": off["total"],
                     "em_on_gold": round(em["accuracy"], 4), "gold_failing_official": failing}
        check(f"{s}: EM of gold vs gold == 1.0", em["accuracy"] == 1.0, scorer[s]["em_on_gold"])
    check("info: official metric on gold answers (quirks expected)", True, scorer)
    cases = [("left", "left", True, True), ("Left.", "left", True, True), ("2", "two", True, True),
             ("right side", "left side", True, False), ("", "brown", True, False),
             ("brown", "white", False, False), ("the table", "table", True, True)]
    case_out = [{"pred": p, "ref": g, "official": official_match(p, g), "em": normalized_em(p, g)}
                for p, g, _, _ in cases]
    check("scorer behaves as documented on known cases",
          all(c["official"] == o and c["em"] == e for c, (_, _, o, e) in zip(case_out, cases)), case_out)

    summary = json.load(open(SUMMARY))
    summary["validation"] = {
        "passed": sum(r["pass"] for r in results),
        "failed": sum(not r["pass"] for r in results),
        "checks": results,
    }
    SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n{summary['validation']['passed']} passed, {summary['validation']['failed']} failed")
    sys.exit(1 if summary["validation"]["failed"] else 0)


if __name__ == "__main__":
    main()
