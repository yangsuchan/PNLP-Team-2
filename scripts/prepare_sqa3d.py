"""Build processed SQA3D JSONL files (full + pilot) from the raw official data.

요약: [1단계] SQA3D 원본을 전처리해 train/validation/test.jsonl과 Pilot(300/100/100)을 만든다.

- Keeps the official train/val/test split; nothing is dropped from the full files.
- Joins questions and annotations by question_id, ScanRefer captions by scene_id.
- Captions: up to 30 per scene, chosen by a seeded RNG keyed on scene_id only
  (never on questions or answers), kept in their original order.
- Model input (`input`) and answers (`reference_answers`) live in separate fields.
- Label conflicts, duplicate questions and caption-missing scenes are flagged,
  never fixed; they are excluded only from the pilot sample.
- Pilot: 300/100/100 stratified by (question category, answer category), seed 42.

Raw files are only read. Usage (from the repo root):
    .venv/bin/python scripts/prepare_sqa3d.py
"""

import hashlib
import json
import pickle
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thinkenough.data.sqa3d_prompt import PROMPT_TEMPLATE_ID, build_prompt, captions_text  # noqa: E402

RAW = ROOT / "data/raw/sqa3d"
SQA_DIR = RAW / "extracted/sqa_task/balanced"
ZIP_PATH = RAW / "zenodo/sqa_task.zip"
CAPTION_PATH = RAW / "github_llm/scanrefer_captions_by_scene.pkl"
NR3D_PATH = RAW / "github_llm/nr3d_captions_by_scene.pkl"
OUT_DIR = ROOT / "data/processed/sqa3d"
REPORT_DIR = ROOT / "outputs"

SEED = 42
MAX_CAPTIONS = 30
TOKENIZER = "Qwen/Qwen3-8B"
# processed split name -> official file split name
SPLITS = {"train": "train", "validation": "val", "test": "test"}
PILOT_SIZES = {"train": 300, "validation": 100, "test": 100}
CAPTION_RULE = (
    f"ScanRefer captions; if a scene has more than {MAX_CAPTIONS}, pick {MAX_CAPTIONS} with "
    f"random.Random('{SEED}:<scene_id>').sample over caption indices, keep original order; "
    "selection depends on scene_id only"
)

QUESTION_CATEGORIES = ("what", "is", "how", "can", "which")
YES_NO = {"yes", "no"}
DIRECTIONS = {
    "left", "right", "forward", "backward", "front", "behind", "back", "up", "down",
    "straight", "ahead", "turn left", "turn right", "my left", "my right", "in front",
    "front left", "front right", "back left", "back right", "left side", "right side",
}
NUMBERS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
    "nineteen", "twenty", "even", "odd",
} | {str(i) for i in range(30)}
COLORS = {
    "white", "black", "brown", "gray", "grey", "blue", "red", "green", "yellow", "orange",
    "purple", "pink", "beige", "silver", "gold", "tan", "dark brown", "light brown",
    "black and white", "dark", "light", "wood", "wooden",
}


class SafeUnpickler(pickle.Unpickler):
    """Only builtin containers/strings are allowed; any class lookup is refused."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"blocked global {module}.{name}")


def question_category(question):
    words = question.strip().lower().split()
    first = words[0].strip("?,.") if words else ""
    return first if first in QUESTION_CATEGORIES else "other"


def answer_category(answer):
    a = answer.strip().lower()
    if a in YES_NO:
        return "yes_no"
    if a in NUMBERS:
        return "number"
    if a in DIRECTIONS:
        return "direction"
    if a in COLORS:
        return "color"
    return "other"


def select_captions(scene_id, captions):
    if len(captions) <= MAX_CAPTIONS:
        idx = list(range(len(captions)))
    else:
        idx = sorted(random.Random(f"{SEED}:{scene_id}").sample(range(len(captions)), MAX_CAPTIONS))
    return [{"object": captions[i][0], "caption": captions[i][1]} for i in idx]


def load_official(split):
    qf = SQA_DIR / f"v1_balanced_questions_{split}_scannetv2.json"
    af = SQA_DIR / f"v1_balanced_sqa_annotations_{split}_scannetv2.json"
    return json.load(open(qf))["questions"], json.load(open(af))["annotations"], qf, af


def build_split(name, official, captions, zip_md5):
    questions, annotations, qf, af = load_official(official)
    ann_by_qid = {a["question_id"]: a for a in annotations}
    records, unmatched = [], []
    for q in questions:
        a = ann_by_qid.get(q["question_id"])
        if a is None:
            unmatched.append(q["question_id"])
            continue
        scene_id = q["scene_id"]
        available = captions.get(scene_id, [])
        selected = select_captions(scene_id, available)
        records.append({
            "question_id": q["question_id"],
            "split": name,
            "input": {
                "scene_id": scene_id,
                "situation": q["situation"],
                "question": q["question"],
                "scene_captions": selected,
            },
            "reference_answers": [x["answer"] for x in a.get("answers", [])],
            "meta": {
                "caption_status": "ok" if available else "missing",
                "num_captions_available": len(available),
                "num_captions_used": len(selected),
                "caption_tokens": None,
                "input_tokens": None,
                "prompt_template": PROMPT_TEMPLATE_ID,
                "question_category": question_category(q["question"]),
                "answer_category": answer_category(a["answers"][0]["answer"]) if a.get("answers") else None,
                "flags": [],
                "alternative_situation": q.get("alternative_situation", []),
                # kept only for the official scorer's item key; never part of the model input
                "position": a["position"],
                "rotation": a["rotation"],
                "annotation_scene_id": a["scene_id"],
            },
            "source": {
                "dataset": "SQA3D v1.0",
                "official_split": official,
                "question_file": str(qf.relative_to(ROOT)),
                "annotation_file": str(af.relative_to(ROOT)),
                "zip_md5": zip_md5,
                "caption_file": str(CAPTION_PATH.relative_to(ROOT)),
                "caption_rule": CAPTION_RULE,
                "seed": SEED,
            },
        })
    return records, {"raw_questions": len(questions), "raw_annotations": len(annotations),
                     "unmatched_question_ids": unmatched}


def flag_duplicates(records):
    """Flag repeated (scene, situation, question); mark groups whose answers disagree."""
    groups = defaultdict(list)
    for r in records:
        groups[(r["input"]["scene_id"], r["input"]["situation"], r["input"]["question"])].append(r)
    conflicts = []
    for (scene_id, situation, question), rs in groups.items():
        if len(rs) < 2:
            continue
        answers = [r["reference_answers"][0] for r in rs]
        is_conflict = len(set(answers)) > 1
        for r in rs:
            r["meta"]["flags"].append("duplicate_question")
            if is_conflict:
                r["meta"]["flags"].append("label_conflict")
        if is_conflict:
            conflicts.append({
                "split": rs[0]["split"],
                "scene_id": scene_id,
                "situation": situation,
                "question": question,
                "items": [{
                    "question_id": r["question_id"],
                    "answer": r["reference_answers"][0],
                    "position": r["meta"]["position"],
                    "rotation": r["meta"]["rotation"],
                } for r in rs],
                "same_position_and_rotation": len({json.dumps([r["meta"]["position"], r["meta"]["rotation"]]) for r in rs}) == 1,
            })
    return conflicts


def add_token_counts(records, tokenizer):
    for r in records:
        r["meta"]["caption_tokens"] = len(tokenizer(captions_text(r["input"]["scene_captions"]))["input_ids"])
        r["meta"]["input_tokens"] = len(tokenizer(build_prompt(r))["input_ids"])


def stratified_sample(records, size, rng):
    """Proportional allocation over (question_category, answer_category), largest remainder."""
    strata = defaultdict(list)
    for r in sorted(records, key=lambda r: r["question_id"]):
        strata[(r["meta"]["question_category"], r["meta"]["answer_category"])].append(r)
    total = len(records)
    quotas = {k: size * len(v) / total for k, v in strata.items()}
    alloc = {k: int(q) for k, q in quotas.items()}
    for k in sorted(quotas, key=lambda k: (quotas[k] - alloc[k], len(strata[k])), reverse=True)[: size - sum(alloc.values())]:
        alloc[k] += 1
    picked = []
    for k in sorted(strata):
        picked.extend(rng.sample(strata[k], alloc[k]))
    return sorted(picked, key=lambda r: r["question_id"])


def pilot_exclusion_reason(r):
    if r["meta"]["caption_status"] != "ok":
        return "caption_missing"
    if "label_conflict" in r["meta"]["flags"]:
        return "label_conflict"
    if "duplicate_question" in r["meta"]["flags"]:
        return "duplicate_question"
    return None


def distribution(records, key):
    c = Counter(r["meta"][key] for r in records)
    n = len(records)
    return {k: {"count": v, "ratio": round(v / n, 4)} for k, v in sorted(c.items())}


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def stats(values):
    v = sorted(values)
    return {"min": v[0], "median": v[len(v) // 2], "p95": v[int(0.95 * (len(v) - 1))], "max": v[-1],
            "mean": round(sum(v) / len(v), 1)}


def main():
    from transformers import AutoTokenizer  # tokenizer only; no model weights, no GPU

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    zip_md5 = hashlib.md5(ZIP_PATH.read_bytes()).hexdigest()
    captions = SafeUnpickler(open(CAPTION_PATH, "rb")).load()
    nr3d = SafeUnpickler(open(NR3D_PATH, "rb")).load()

    summary = {
        "config": {"seed": SEED, "max_captions": MAX_CAPTIONS, "caption_rule": CAPTION_RULE,
                   "tokenizer": TOKENIZER, "prompt_template": PROMPT_TEMPLATE_ID,
                   "token_count_note": "raw prompt text, no chat template / thinking tokens",
                   "pilot_sizes": PILOT_SIZES, "sqa_task_zip_md5": zip_md5},
        "splits": {}, "pilot": {},
    }
    all_conflicts, missing = [], []

    for name, official in SPLITS.items():
        records, load_info = build_split(name, official, captions, zip_md5)
        conflicts = flag_duplicates(records)
        add_token_counts(records, tokenizer)
        all_conflicts.extend(conflicts)

        missing_scenes = sorted({r["input"]["scene_id"] for r in records if r["meta"]["caption_status"] == "missing"})
        for scene_id in missing_scenes:
            missing.append({
                "split": name,
                "scene_id": scene_id,
                "num_questions": sum(r["input"]["scene_id"] == scene_id for r in records),
                "question_ids": [r["question_id"] for r in records if r["input"]["scene_id"] == scene_id],
                "nr3d_captions_available": len(nr3d.get(scene_id, [])),
            })

        write_jsonl(OUT_DIR / f"{name}.jsonl", records)

        excluded = Counter(pilot_exclusion_reason(r) for r in records)
        eligible = [r for r in records if pilot_exclusion_reason(r) is None]
        pilot = stratified_sample(eligible, PILOT_SIZES[name], random.Random(f"{SEED}:{name}"))
        write_jsonl(OUT_DIR / "pilot" / f"{name}.jsonl", pilot)

        summary["splits"][name] = {
            "official_split": official,
            **{k: v for k, v in load_info.items() if k != "unmatched_question_ids"},
            "unmatched_questions": len(load_info["unmatched_question_ids"]),
            "processed_records": len(records),
            "dropped_records": load_info["raw_questions"] - len(records),
            "num_scenes": len({r["input"]["scene_id"] for r in records}),
            "flags": dict(Counter(f for r in records for f in r["meta"]["flags"])),
            "caption_missing_records": sum(r["meta"]["caption_status"] == "missing" for r in records),
            "captions_used": stats([r["meta"]["num_captions_used"] for r in records]),
            "caption_tokens": stats([r["meta"]["caption_tokens"] for r in records]),
            "input_tokens": stats([r["meta"]["input_tokens"] for r in records]),
            "question_category": distribution(records, "question_category"),
            "answer_category": distribution(records, "answer_category"),
        }
        summary["pilot"][name] = {
            "source_records": len(records),
            "excluded_before_sampling": {k: v for k, v in excluded.items() if k is not None},
            "eligible_records": len(eligible),
            "sampled": len(pilot),
            "num_scenes": len({r["input"]["scene_id"] for r in pilot}),
            "input_tokens": stats([r["meta"]["input_tokens"] for r in pilot]),
            "question_category": distribution(pilot, "question_category"),
            "answer_category": distribution(pilot, "answer_category"),
        }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "label_conflicts.json").write_text(json.dumps(all_conflicts, indent=2, ensure_ascii=False))
    (REPORT_DIR / "missing_captions.json").write_text(json.dumps(missing, indent=2, ensure_ascii=False))
    summary["label_conflicts"] = len(all_conflicts)
    summary["missing_caption_scenes"] = [{k: v for k, v in m.items() if k != "question_ids"} for m in missing]
    (REPORT_DIR / "preprocessing_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps({n: {k: s[k] for k in ("raw_questions", "processed_records", "flags", "caption_missing_records")}
                      for n, s in summary["splits"].items()}, indent=1))
    print(json.dumps({n: {k: p[k] for k in ("eligible_records", "sampled", "excluded_before_sampling")}
                      for n, p in summary["pilot"].items()}, indent=1))
    print("label conflicts:", len(all_conflicts), "| missing-caption scenes:", [m["scene_id"] for m in missing])


if __name__ == "__main__":
    main()
