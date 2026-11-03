#!/usr/bin/env python3
"""Grade answers in pairs with SCoT's prompt on a local judge model.

Check that the judge is usable, before trusting it:
    python scripts/judge_answers.py --runs outputs/main/planning --check 30 --out outputs/main/planning/judge_check.json

Grade each baseline against full thinking, resuming if stopped:
    python scripts/judge_answers.py --runs outputs/main/planning --out outputs/main/planning/judge.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from PIL import Image

from thinkenough.judge import (
    ACCURACY_MAX_TOKENS,
    CRITERIA,
    JUDGE_SYSTEM,
    accuracy_prompts,
    accuracy_rating,
    answer_accuracy,
    by_arm,
    clip_answer,
    is_swapped,
    judge_prompt,
    parse_scores,
    scores_complete,
)
from thinkenough.scot import answer_text

Image.MAX_IMAGE_PIXELS = None
# Third of 54 judges for agreement with human grading in arXiv 2510.09738. On 24 GB it must be the 4-bit copy.
DEFAULT_MODEL = "unsloth/gemma-3-27b-it-bnb-4bit"
DEFAULT_REVISION = "c08b2ba63738aa8cfc60f06741d6356ef4e60b3f"
PAIRS = (("direct", "full"), ("no_think", "full"))


class Judge:
    def __init__(self, model_id: str, revision: str, image_dir: Path) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.image_dir = image_dir
        self.model_id, self.revision = model_id, revision
        self.processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, revision=revision, device_map="auto", dtype=torch.bfloat16
        ).eval()

    def __call__(self, scene_id: str, prompt: str) -> str:
        with Image.open(self.image_dir / f"{scene_id}.png") as image:
            image = image.convert("RGB")
            messages = [
                {"role": "system", "content": [{"type": "text", "text": JUDGE_SYSTEM}]},
                {"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image", "image": image}]},
            ]
            # The explanation after the scores is unused, and without sampling stopping there cannot change them.
            return self._generate(messages, 1024, stop_when=scores_complete)

    def text(self, prompt: str, max_new_tokens: int) -> str:
        """A text-only call, as in the Answer Accuracy metric."""
        return self._generate([{"role": "user", "content": [{"type": "text", "text": prompt}]}], max_new_tokens)

    def _generate(self, messages: list[dict], max_new_tokens: int, stop_when=None) -> str:
        from transformers import StoppingCriteria, StoppingCriteriaList

        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(self.model.device)
        start = inputs["input_ids"].shape[1]
        decode = lambda ids: self.processor.decode(ids[start:], skip_special_tokens=True)

        class StopWhen(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return input_ids.new_full((input_ids.shape[0],), stop_when(decode(input_ids[0]))).bool()

        extra = {"stopping_criteria": StoppingCriteriaList([StopWhen()])} if stop_when else {}
        with self.torch.inference_mode():  # No sampling, so the same answers always get the same scores.
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, **extra)
        return decode(output[0])


def load_arm(root: Path, arm: str) -> dict[str, dict]:
    rows = (json.loads(p.read_text(encoding="utf-8")) for p in (root / arm).glob("*.json"))
    return {row["sample_id"]: row for row in rows}


def run_check(judge: Judge, full: dict[str, dict], direct: dict[str, dict], count: int, out: Path) -> int:
    """Reference against another question's reference, and whether the order changes scores."""
    ids = sorted(full)
    random.Random(1234).shuffle(ids)
    ids = ids[:count]
    rows = []
    for index, sample_id in enumerate(ids):
        meta = full[sample_id]["metadata"]
        reference = meta["references"][0]
        other = full[ids[(index + 1) % len(ids)]]["metadata"]["references"][0]
        ask = lambda a, b: parse_scores(judge(meta["scene_id"], judge_prompt(meta["prompt"], reference, clip_answer(a), clip_answer(b))))
        first, second = answer_text(full[sample_id], "full"), answer_text(direct[sample_id], "direct")
        rows.append(
            {
                "sample_id": sample_id,
                "right_first": ask(reference, other),
                "right_second": ask(other, reference),
                "order_a": ask(first, second),
                "order_b": ask(second, first),
            }
        )
        print(f"[check] {index + 1}/{len(ids)} {sample_id}", flush=True)
    mean = lambda scores: statistics.mean(scores[c] for c in CRITERIA)
    gaps = []
    for row in rows:
        if row["right_first"]:
            gaps.append(mean(row["right_first"][1]) - mean(row["right_first"][2]))
        if row["right_second"]:
            gaps.append(mean(row["right_second"][2]) - mean(row["right_second"][1]))
    drift = [
        abs(mean(r["order_a"][1]) - mean(r["order_b"][2]))
        for r in rows
        if r["order_a"] and r["order_b"]
    ]
    summary = {
        "questions": len(rows),
        "parsed": sum(v is not None for r in rows for k, v in r.items() if k != "sample_id"),
        "asked": 4 * len(rows),
        "reference_wins": sum(g > 0 for g in gaps) / max(len(gaps), 1),
        "mean_gap": statistics.mean(gaps) if gaps else None,
        "order_drift": statistics.mean(drift) if drift else None,
    }
    # Fixed before looking: the judge must separate right from wrong, and order must barely matter.
    summary["passed"] = bool(
        gaps and summary["reference_wins"] >= 0.8 and summary["mean_gap"] >= 2.0 and drift and summary["order_drift"] <= 1.0
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary))
    return 0 if summary["passed"] else 2


def run_accuracy(judge, arms: dict, forced_path: Path, out: Path) -> int:
    """Answer Accuracy of every answer against its reference, plus two sanity items for the same question."""
    done = set()
    if out.is_file():
        done = {(r["sample_id"], r["item"]) for r in map(json.loads, out.read_text(encoding="utf-8").splitlines())}
    forced: dict[str, dict[str, str]] = {}
    for row in map(json.loads, forced_path.read_text(encoding="utf-8").splitlines()):
        forced.setdefault(row["sample_id"], {})[f"forced{row['checkpoint_idx']}"] = row["answer"] or ""
    ids = sorted(arms["full"])
    for index, sample_id in enumerate(ids):
        meta = arms["full"][sample_id]["metadata"]
        reference = meta["references"][0]
        items = {
            "reference": reference,
            "other_full": answer_text(arms["full"][ids[(index + 1) % len(ids)]], "full") or "",
            **{arm: answer_text(arms[arm][sample_id], arm) or "" for arm in arms if sample_id in arms[arm]},
            **forced.get(sample_id, {}),
        }
        for item, answer in items.items():
            if (sample_id, item) in done:
                continue
            row = {"sample_id": sample_id, "task": meta["task"], "item": item, "model": judge.model_id, "revision": judge.revision}
            if not answer.strip():
                row.update(status="missing_answer", accuracy=0.0)
            else:
                raw = [judge.text(p, ACCURACY_MAX_TOKENS) for p in accuracy_prompts(meta["prompt"], clip_answer(answer), reference)]
                ratings = [accuracy_rating(r) for r in raw]
                value = answer_accuracy(*ratings)
                row.update(status="ok" if value is not None else "unparsed", raw=raw, ratings=ratings, accuracy=value)
            with out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[accuracy] {sample_id} {len(items)} items", flush=True)
    return 0


def grade_pair(judge, out: Path, args, meta: dict, sample_id: str, pair: str, first: str, second: str, answers: dict) -> None:
    """Grade one pair and append the row, so a stopped run can resume."""
    row = {"sample_id": sample_id, "task": meta["task"], "pair": pair, "model": args.model_id, "revision": args.revision}
    if not all(answers.values()):  # No answer to grade; the summary counts it as a failure.
        row.update(status="missing_answer", empty=[a for a, t in answers.items() if not t])
    else:
        swapped = is_swapped(sample_id, pair)
        shown = (answers[second], answers[first]) if swapped else (answers[first], answers[second])
        raw = judge(meta["scene_id"], judge_prompt(meta["prompt"], meta["references"][0], clip_answer(shown[0]), clip_answer(shown[1])))
        scores = parse_scores(raw)
        row.update(status="ok" if scores else "unparsed", swapped=swapped, raw=raw)
        if scores:
            row["scores"] = by_arm(scores, first, second, swapped)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[judge] {sample_id} {pair} {row['status']}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Grade answers with SCoT's prompt on a local judge")
    parser.add_argument("--runs", required=True, help="Folder holding direct/, no_think/, and full/ results")
    parser.add_argument("--image-dir", default="data/raw/scene_images")
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--check", type=int, default=0, help="Run the usability check on this many questions")
    parser.add_argument("--forced", default=None, help="checkpoints.jsonl: grade each forced answer against the full answer")
    parser.add_argument("--accuracy", action="store_true", help="Answer Accuracy labels instead of SCoT's scores. Needs --forced")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    root, out = _REPO_ROOT / args.runs, _REPO_ROOT / args.out
    arms = {arm: load_arm(root, arm) for arm in ("direct", "no_think", "full")}
    judge = Judge(args.model_id, args.revision, _REPO_ROOT / args.image_dir)
    if args.check:
        return run_check(judge, arms["full"], arms["direct"], args.check, out)

    out.parent.mkdir(parents=True, exist_ok=True)
    if args.accuracy:
        return run_accuracy(judge, arms, _REPO_ROOT / args.forced, out)
    done = set()
    if out.is_file():
        done = {(r["sample_id"], r["pair"]) for r in map(json.loads, out.read_text(encoding="utf-8").splitlines())}
    grade = lambda *pair_args: grade_pair(judge, out, args, *pair_args)
    if args.forced:
        for row in map(json.loads, (_REPO_ROOT / args.forced).read_text(encoding="utf-8").splitlines()):
            sample_id, pair = row["sample_id"], f"forced{row['checkpoint_idx']}:full"
            if (sample_id, pair) not in done:
                full_answer = answer_text(arms["full"][sample_id], "full") or ""
                grade(arms["full"][sample_id]["metadata"], sample_id, pair, "forced", "full", {"forced": row["answer"] or "", "full": full_answer})
        return 0
    for sample_id in sorted(arms["full"]):
        meta = arms["full"][sample_id]["metadata"]
        for first, second in PAIRS:
            pair = f"{first}:{second}"
            if (sample_id, pair) in done or sample_id not in arms[first]:
                continue
            grade(meta, sample_id, pair, first, second, {arm: answer_text(arms[arm][sample_id], arm) or "" for arm in (first, second)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
