"""STOP/CONTINUE labels for reasoning checkpoints (probe training ground truth).

요약: [3단계 라벨] checkpoint마다 강제로 낸 답을 정답과 비교해 STOP(1)/CONTINUE(0) 라벨을 붙인다. probe 학습용 정답 데이터.

For every checkpoint, each forced-answer record (model answer generated after
closing </think> at that prefix) is scored on its own:

    stop_label = 1 (STOP)      normalized EM with the SQA3D reference
    stop_label = 0 (CONTINUE)  otherwise, including empty / unparsable answers

Rules
- Each checkpoint and each sample is judged independently: a correct answer at
  checkpoint k never implies anything about checkpoint k+1, and several samples
  of one checkpoint are kept as separate rows (no majority vote).
- Checkpoints without a usable generation get stop_label = None and are not
  training labels: "pending" (no answer record yet), "missing" (generation
  failed / not run), "no_reference" (reference answer not found).
- The official SQA3D match is recorded alongside (empty predictions counted
  wrong) but never decides the label.
- These are ground-truth labels. Probe predictions (p_stop) live in separate files.
"""
from thinkenough.identity import question_sort_key


from collections import defaultdict

from thinkenough.evaluation.answer_extraction import EXTRACTION_VERSION, extract_final_answer
from thinkenough.evaluation.sqa3d_scoring import normalized_em, official_match

LABELING_VERSION = "em_v1"
LABEL_NAMES = {1: "STOP", 0: "CONTINUE"}
OK_GENERATION = {"ok", "length"}   # "length": answer hit its token limit; still scored, flagged
FAILED_GENERATION = {"error", "not_run", "timeout", "oom"}


class LabelingInputError(ValueError):
    """Duplicate or inconsistent input records."""


def load_references(paths):
    """{question_id: record} from processed SQA3D JSONL files (or synthetic ones of the same shape)."""
    import json

    refs = {}
    for path in paths:
        for line in open(path):
            r = json.loads(line)
            if r["question_id"] in refs:
                raise LabelingInputError(f"duplicate reference for question_id {r['question_id']}")
            refs[r["question_id"]] = r
    return refs


def index_answers(answers):
    """Group forced answers by checkpoint_id; reject duplicate (checkpoint_id, sample_index)."""
    by_cp, seen = defaultdict(list), set()
    for a in answers:
        key = (a["checkpoint_id"], a.get("sample_index", 0))
        if key in seen:
            raise LabelingInputError(f"duplicate answer record for {key}")
        seen.add(key)
        by_cp[a["checkpoint_id"]].append(a)
    return by_cp


def label_answer(answer, reference_answers):
    """Score one forced answer against the reference. Returns label fields."""
    status = answer.get("generation_status")
    if status not in OK_GENERATION:
        return {"label_status": "missing", "stop_label": None, "extracted_answer": None,
                "extraction_status": None, "em_correct": None, "official_correct": None,
                "flags": [f"generation_{status}"]}
    extracted, extraction_status = extract_final_answer(answer.get("generated_answer"))
    flags = []
    if extraction_status == "empty":
        flags.append("empty_answer")
    elif extraction_status != "ok":
        flags.append("invalid_format")
    if status == "length":
        flags.append("answer_truncated")
    ref = reference_answers[0]
    em = extraction_status == "ok" and normalized_em(extracted, ref)
    official = extraction_status == "ok" and official_match(extracted, ref, empty_pred_is_wrong=True)
    if official and not em:
        flags.append("official_only_match")
    return {"label_status": "labeled", "stop_label": int(em), "extracted_answer": extracted,
            "extraction_status": extraction_status, "em_correct": bool(em),
            "official_correct": bool(official), "flags": flags}


def build_labels(checkpoints, answers, references):
    """One label row per (checkpoint, sample); pending rows for checkpoints without answers.

    checkpoints: checkpoint records (scripts/build_checkpoints.py output)
    answers: forced-answer records (docs/labeling_evaluation_interface.md)
    references: {question_id: {"split": ..., "reference_answers": [...]}}
    Returns (label_rows, report).
    """
    ids = [c["checkpoint_id"] for c in checkpoints]
    if len(ids) != len(set(ids)):
        raise LabelingInputError("duplicate checkpoint_id in checkpoint file")
    by_cp = index_answers(answers)
    known = set(ids)
    orphans = sorted(cid for cid in by_cp if cid not in known)
    rows, split_mismatch = [], []

    for c in sorted(checkpoints, key=lambda c: (question_sort_key(c["question_id"]), c["checkpoint_index"])):
        ref = references.get(c["question_id"])
        base = {
            "checkpoint_id": c["checkpoint_id"],
            "run_id": c.get("run_id"),
            "question_id": c["question_id"],
            "split": c["split"],
            "checkpoint_index": c["checkpoint_index"],
            "checkpoint_type": c["checkpoint_type"],
            "is_final": c.get("is_final"),
            "prefix_token_count": c.get("prefix_token_count"),
            "prefix_token_count_source": c.get("token_count_source"),
            "synthetic": bool((c.get("source") or {}).get("synthetic")),
        }
        if ref is not None and ref.get("split") != c["split"]:
            split_mismatch.append(c["checkpoint_id"])
        cp_answers = sorted(by_cp.get(c["checkpoint_id"], []), key=lambda a: a.get("sample_index", 0))
        if not cp_answers:
            rows.append({**base, "sample_index": None, "label_status": "pending", "stop_label": None,
                         "label_name": None, "flags": ["no_answer_record"]})
            continue
        for a in cp_answers:
            if a.get("question_id") != c["question_id"] or a.get("checkpoint_index") != c["checkpoint_index"]:
                raise LabelingInputError(f"answer fields disagree with checkpoint {c['checkpoint_id']}")
            meta = {
                "sample_index": a.get("sample_index", 0),
                "generated_answer": a.get("generated_answer"),
                "generation_status": a.get("generation_status"),
                "answer_token_count": a.get("answer_token_count"),
                "answer_token_count_source": a.get("answer_token_count_source"),
                "answer_latency_ms": a.get("answer_latency_ms"),
                "model": a.get("model"),
                "prompt_version": a.get("prompt_version"),
                "answer_prompt_version": a.get("answer_prompt_version"),
                "generation_config": a.get("generation_config"),
            }
            if ref is None:
                rows.append({**base, **meta, "label_status": "no_reference", "stop_label": None,
                             "label_name": None, "flags": ["no_reference"]})
                continue
            result = label_answer(a, ref["reference_answers"])
            rows.append({**base, **meta, **result,
                         "label_name": LABEL_NAMES.get(result["stop_label"]),
                         "reference_answer": ref["reference_answers"][0],
                         "label_source": "normalized_em",
                         "labeling_version": LABELING_VERSION,
                         "extraction_version": EXTRACTION_VERSION,
                         "synthetic": bool(base["synthetic"] or a.get("synthetic") or ref.get("synthetic"))})

    report = {
        "checkpoints": len(checkpoints),
        "answer_records": len(answers),
        "label_rows": len(rows),
        "orphan_answer_checkpoint_ids": orphans,
        "split_mismatch_checkpoint_ids": split_mismatch,
    }
    return rows, report


def training_labels(rows):
    """Rows usable as probe training targets (label_status == "labeled")."""
    return [r for r in rows if r["label_status"] == "labeled"]
