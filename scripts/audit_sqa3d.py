"""SQA3D dataset audit (read-only).

요약: [1단계] SQA3D 원본 데이터 조사(개수, 분할 겹침, 정답 분포, 캡션 유무). 원본은 읽기만 한다. 결과 보고서: docs/dataset_audit.md

Checks split sizes, fields, test-answer availability, cross-split overlap
(scene / question / situation), answer distribution, and scene-caption
coverage. Raw files are only read; results go to outputs/audit/.

Usage (from the repo root):
    .venv/bin/python scripts/audit_sqa3d.py
    .venv/bin/python scripts/audit_sqa3d.py --no-tokenizer   # skip token counts
"""

import argparse
import hashlib
import json
import pickle
import statistics
from collections import Counter
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/sqa3d"
SQA_DIR = RAW / "extracted/sqa_task"
LLM_DIR = RAW / "github_llm"
OUT_DIR = ROOT / "outputs/audit"
SPLITS = ["train", "val", "test"]
CAPTION_FILES = {
    "scanrefer": LLM_DIR / "scanrefer_captions_by_scene.pkl",
    "nr3d": LLM_DIR / "nr3d_captions_by_scene.pkl",
}
NUM_CAPTIONS = 30  # captions sampled per scene in the official LLM eval script


class SafeUnpickler(pickle.Unpickler):
    """Only builtin containers/strings are allowed; any class lookup is refused."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"blocked global {module}.{name}")


def load_split(split):
    q = json.load(open(SQA_DIR / f"balanced/v1_balanced_questions_{split}_scannetv2.json"))
    a = json.load(open(SQA_DIR / f"balanced/v1_balanced_sqa_annotations_{split}_scannetv2.json"))
    return q["questions"], a["annotations"]


def md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def field_types(records):
    keys = Counter(k for r in records for k in r)
    return {k: {"count": c, "type": type(records[0].get(k)).__name__} for k, c in sorted(keys.items())}


def first_word(question):
    w = question.strip().split()
    return w[0].lower().strip("?,.") if w else ""


def audit_split(questions, annotations):
    ann_by_qid = {a["question_id"]: a for a in annotations}
    q_qids = [q["question_id"] for q in questions]
    a_qids = [a["question_id"] for a in annotations]
    scene_mismatch = sum(
        1 for q in questions
        if q["question_id"] in ann_by_qid and ann_by_qid[q["question_id"]]["scene_id"] != q["scene_id"]
    )
    answers = [a["answers"][0]["answer"] if a.get("answers") else None for a in annotations]
    triples = Counter((q["scene_id"], q["situation"], q["question"]) for q in questions)
    # same (scene, situation, question) but different answers -> label noise
    answer_sets = {}
    for q in questions:
        a = ann_by_qid.get(q["question_id"])
        if a and a.get("answers"):
            answer_sets.setdefault((q["scene_id"], q["situation"], q["question"]), set()).add(a["answers"][0]["answer"])
    return {
        "num_questions": len(questions),
        "num_annotations": len(annotations),
        "num_scenes": len({q["scene_id"] for q in questions}),
        "num_unique_situations": len({(q["scene_id"], q["situation"]) for q in questions}),
        "question_ids_unique": len(set(q_qids)) == len(q_qids),
        "question_annotation_qids_match": set(q_qids) == set(a_qids),
        "scene_id_mismatch_q_vs_a": scene_mismatch,
        "question_fields": field_types(questions),
        "annotation_fields": field_types(annotations),
        "answers_present": sum(1 for x in answers if x not in (None, "")),
        "answers_per_annotation": dict(Counter(len(a.get("answers", [])) for a in annotations)),
        "answer_confidence": dict(Counter(a["answers"][0].get("answer_confidence") for a in annotations if a.get("answers"))),
        "answer_type": dict(Counter(a.get("answer_type") for a in annotations)),
        "question_type": dict(Counter(a.get("question_type") for a in annotations)),
        "num_unique_answers": len(set(answers)),
        "top20_answers": Counter(answers).most_common(20),
        "answer_word_len": dict(sorted(Counter(len(str(x).split()) for x in answers).items())),
        "question_first_word_top10": Counter(first_word(q["question"]) for q in questions).most_common(10),
        "duplicate_triples_within_split": sum(c - 1 for c in triples.values() if c > 1),
        "conflicting_answers_same_triple": sum(1 for s in answer_sets.values() if len(s) > 1),
    }, answers


def cross_split(data):
    out = {}
    for s1, s2 in combinations(SPLITS, 2):
        q1, q2 = data[s1][0], data[s2][0]
        scenes1, scenes2 = {q["scene_id"] for q in q1}, {q["scene_id"] for q in q2}
        qid1, qid2 = {q["question_id"] for q in q1}, {q["question_id"] for q in q2}
        trip1 = {(q["scene_id"], q["situation"], q["question"]) for q in q1}
        trip2 = {(q["scene_id"], q["situation"], q["question"]) for q in q2}
        sit1 = {q["situation"] for q in q1} | {s for q in q1 for s in q.get("alternative_situation", [])}
        sit2 = {q["situation"] for q in q2} | {s for q in q2 for s in q.get("alternative_situation", [])}
        qtext1, qtext2 = {q["question"] for q in q1}, {q["question"] for q in q2}
        out[f"{s1}-{s2}"] = {
            "scene_overlap": len(scenes1 & scenes2),
            "scene_overlap_examples": sorted(scenes1 & scenes2)[:10],
            "question_id_overlap": len(qid1 & qid2),
            "exact_scene_situation_question_overlap": len(trip1 & trip2),
            "situation_text_overlap_incl_alternatives": len(sit1 & sit2),
            "question_text_overlap": len(qtext1 & qtext2),
            "question_text_overlap_ratio_of_second": round(len(qtext1 & qtext2) / len(qtext2), 4),
        }
    return out


def answer_coverage(answers_by_split, answer_dict):
    train_set = set(answers_by_split["train"])
    vocab = set(answer_dict[0])
    out = {}
    for s in ["val", "test"]:
        ans = answers_by_split[s]
        out[s] = {
            "in_train_answer_set": round(sum(a in train_set for a in ans) / len(ans), 4),
            "in_answer_dict": round(sum(a in vocab for a in ans) / len(ans), 4),
            "majority_train_answer_acc": None,
        }
    top = Counter(answers_by_split["train"]).most_common(1)[0][0]
    for s in ["val", "test"]:
        ans = answers_by_split[s]
        out[s]["majority_train_answer_acc"] = {"answer": top, "acc": round(sum(a == top for a in ans) / len(ans), 4)}
    out["answer_dict_size"] = len(vocab)
    return out


def caption_audit(data, tokenizer):
    out = {}
    for name, path in CAPTION_FILES.items():
        caps = SafeUnpickler(open(path, "rb")).load()
        per_scene = [len(v) for v in caps.values()]
        info = {
            "num_scenes_with_captions": len(caps),
            "captions_per_scene": {
                "min": min(per_scene), "median": statistics.median(per_scene), "max": max(per_scene),
            },
            "entry_format": "list of (object_name, caption) tuples",
            "coverage": {},
        }
        for s in SPLITS:
            qs = data[s][0]
            scenes = {q["scene_id"] for q in qs}
            info["coverage"][s] = {
                "scenes_covered": f"{len(scenes & caps.keys())}/{len(scenes)}",
                "questions_covered": round(sum(q["scene_id"] in caps for q in qs) / len(qs), 4),
            }
        if tokenizer is not None:
            # tokens if ALL captions of a scene are joined vs the official 30-sample setting
            full, sampled = [], []
            for v in caps.values():
                texts = [c[1] for c in v]
                full.append(len(tokenizer(". ".join(texts))["input_ids"]))
                sampled.append(len(tokenizer(". ".join(texts[:NUM_CAPTIONS]))["input_ids"]))
            info["qwen3_tokens_all_captions"] = {
                "median": statistics.median(full), "p95": sorted(full)[int(0.95 * len(full))], "max": max(full),
            }
            info[f"qwen3_tokens_{NUM_CAPTIONS}_captions"] = {
                "median": statistics.median(sampled), "max": max(sampled),
            }
        out[name] = info
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-tokenizer", action="store_true", help="skip Qwen3 token counts")
    args = parser.parse_args()

    tokenizer = None
    if not args.no_tokenizer:
        from transformers import AutoTokenizer  # tokenizer only; no model weights
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

    data = {s: load_split(s) for s in SPLITS}
    report = {"files": {}, "splits": {}}

    report["files"]["sqa_task.zip_md5"] = md5(RAW / "zenodo/sqa_task.zip")
    for kind in ["questions", "sqa_annotations"]:
        name = f"v1_balanced_{kind}_test_scannetv2.json"
        report["files"][f"llm_dir_{kind}_test_identical_to_zip"] = md5(LLM_DIR / name) == md5(SQA_DIR / "balanced" / name)

    answers_by_split = {}
    for s in SPLITS:
        report["splits"][s], answers_by_split[s] = audit_split(*data[s])
    report["cross_split"] = cross_split(data)
    report["answer_coverage"] = answer_coverage(answers_by_split, json.load(open(SQA_DIR / "answer_dict.json")))
    report["captions"] = caption_audit(data, tokenizer)
    report["examples"] = {s: {"question": data[s][0][0], "annotation": data[s][1][0]} for s in SPLITS}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "sqa3d_audit.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    print(json.dumps({k: v for k, v in report.items() if k != "examples"}, indent=1, ensure_ascii=False, default=str))
    print(f"\nSaved: {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
