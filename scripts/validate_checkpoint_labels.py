"""Validate a STOP/CONTINUE label file.

요약: [3단계] 라벨 파일 검사(중복, 재채점 결과 일치, 라벨이 다음 checkpoint로 번지지 않았는지 등).

Checks: every checkpoint has at least one row, unique (checkpoint_id, sample_index),
stop_label is set only for labeled rows, labels agree with a fresh EM re-score,
no probe outputs inside label rows, split consistency, and that labels were not
propagated (a later checkpoint is never STOP just because an earlier one was).
Results go to outputs/checkpoint_label_validation.json; exit code 1 on failure.

Usage (from the repo root):
    .venv/bin/python scripts/validate_checkpoint_labels.py
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thinkenough.identity import question_sort_key
from thinkenough.evaluation.answer_extraction import extract_final_answer  # noqa: E402
from thinkenough.evaluation.sqa3d_scoring import normalized_em  # noqa: E402

PROBE_FIELDS = {"p_stop", "probe_score", "probe_prediction", "threshold"}
STATUSES = {"labeled", "pending", "missing", "no_reference"}

results = []


def check(name, ok, detail=None):
    results.append({"check": name, "pass": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -> {detail}" if detail is not None else ""))


def read_jsonl(path):
    return [json.loads(line) for line in open(path)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=Path, default=ROOT / "data/processed/checkpoints/example_checkpoints.jsonl")
    parser.add_argument("--labels", type=Path, default=ROOT / "data/processed/checkpoint_labels/example_labels.jsonl")
    parser.add_argument("--report", type=Path, default=ROOT / "outputs/checkpoint_label_validation.json")
    args = parser.parse_args()

    checkpoints = {c["checkpoint_id"]: c for c in read_jsonl(args.checkpoints)}
    labels = read_jsonl(args.labels)

    check("every checkpoint has a label row", set(checkpoints) == {r["checkpoint_id"] for r in labels},
          {"checkpoints": len(checkpoints), "label_checkpoints": len({r["checkpoint_id"] for r in labels})})
    keys = Counter((r["checkpoint_id"], r.get("sample_index")) for r in labels)
    check("unique (checkpoint_id, sample_index)", all(v == 1 for v in keys.values()),
          [k for k, v in keys.items() if v > 1])
    check("label_status has a known value", all(r["label_status"] in STATUSES for r in labels))
    bad = [r["checkpoint_id"] for r in labels
           if (r["label_status"] == "labeled") != (r["stop_label"] in (0, 1))]
    check("stop_label set iff label_status == labeled", not bad, bad)
    probe = sorted({k for r in labels for k in r if k in PROBE_FIELDS})
    check("no probe outputs in label rows", not probe, probe)
    mismatch = [r["checkpoint_id"] for r in labels
                if r["checkpoint_id"] in checkpoints and (r["split"] != checkpoints[r["checkpoint_id"]]["split"]
                                                          or r["question_id"] != checkpoints[r["checkpoint_id"]]["question_id"])]
    check("split / question_id agree with checkpoints", not mismatch, mismatch)

    rescored = []
    for r in labels:
        if r["label_status"] != "labeled":
            continue
        answer, status = extract_final_answer(r["generated_answer"])
        expected = int(status == "ok" and normalized_em(answer, r["reference_answer"]))
        if expected != r["stop_label"]:
            rescored.append(r["checkpoint_id"])
    check("labels match an independent EM re-score", not rescored, rescored)
    empty_stop = [r["checkpoint_id"] for r in labels if r["label_status"] == "labeled" and not r.get("extracted_answer") and r["stop_label"] == 1]
    check("empty / invalid answers are never STOP", not empty_stop, empty_stop)

    # per-question label sequences (sample 0), to show labels are not monotone-propagated
    seqs = {}
    for r in sorted(labels, key=lambda r: (question_sort_key(r["question_id"]), r["checkpoint_index"])):
        if r.get("sample_index") in (0, None):
            seqs.setdefault(r["question_id"], []).append(r["stop_label"])
    check("info: label sequences per question (sample 0; None = not labeled)", True, seqs)
    stats = {
        "label_status": dict(Counter(r["label_status"] for r in labels)),
        "stop_label_by_split": {s: dict(Counter(r["stop_label"] for r in labels if r["split"] == s and r["label_status"] == "labeled"))
                                for s in sorted({r["split"] for r in labels})},
        "official_only_matches": sum("official_only_match" in r["flags"] for r in labels),
        "synthetic_rows": sum(bool(r.get("synthetic")) for r in labels),
    }
    check("info: label statistics", True, stats)

    report = {"labels": str(args.labels), "passed": sum(r["pass"] for r in results),
              "failed": sum(not r["pass"] for r in results), "checks": results}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    print(f"\n{report['passed']} passed, {report['failed']} failed")
    sys.exit(1 if report["failed"] else 0)


if __name__ == "__main__":
    main()
