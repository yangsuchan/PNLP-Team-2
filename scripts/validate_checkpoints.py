"""Validate a checkpoint JSONL against the generation JSONL it was built from.

요약: [2단계] checkpoint 파일이 원래 생성 결과와 정확히 맞는지 검사.

Checks: required fields, contiguous checkpoint indices, exact prefix
accumulation, final prefix == reasoning text (re-parsed from the raw output),
monotonic token counts, the null rule for token positions, recorded parse
status, and that no answer fields leak into checkpoints. Results are written
to outputs/checkpoint_validation.json; exit code 1 on failure.

Usage (from the repo root):
    .venv/bin/python scripts/validate_checkpoints.py
    .venv/bin/python scripts/validate_checkpoints.py --generations gen.jsonl --checkpoints out.jsonl
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thinkenough.data.reasoning_checkpoint import parse_generation  # noqa: E402

REQUIRED = ["checkpoint_id", "run_id", "question_id", "split", "checkpoint_index", "reasoning_prefix", "prefix_token_count",
            "reasoning_token_start", "reasoning_token_end", "checkpoint_type"]
FORBIDDEN = {"reference_answers", "answers", "answer", "final_answer"}
PARSE_STATUSES = {"ok", "missing_open_tag", "unclosed_think", "no_think_tags", "empty_reasoning", "empty_output"}

results = []


def check(name, ok, detail=None):
    results.append({"check": name, "pass": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -> {detail}" if detail is not None else ""))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=Path, default=ROOT / "tests/fixtures/fake_qwen3_outputs.jsonl")
    parser.add_argument("--checkpoints", type=Path, default=ROOT / "data/processed/checkpoints/example_checkpoints.jsonl")
    parser.add_argument("--report", type=Path, default=ROOT / "outputs/checkpoint_validation.json")
    args = parser.parse_args()
    if args.report.resolve() in {args.generations.resolve(), args.checkpoints.resolve()}:
        parser.error("Report must not overwrite input files")
    results.clear()

    def identity(row):
        return (row.get("run_id"), row.get("split"), row["question_id"])

    with args.generations.open(encoding="utf-8") as handle:
        rows = list(map(json.loads, handle))
    generations = {identity(r): r for r in rows}
    if len(generations) != len(rows):
        raise ValueError("Duplicate generation identity")
    by_qid = defaultdict(list)
    with args.checkpoints.open(encoding="utf-8") as handle:
        for c in map(json.loads, handle):
            by_qid[identity(c)].append(c)
    all_cps = [c for cs in by_qid.values() for c in cs]

    check("every generation record has checkpoints", set(generations) == set(by_qid),
          {"generations": len(generations), "with_checkpoints": len(by_qid)})
    missing = [k for c in all_cps for k in REQUIRED if k not in c]
    check("required fields present", not missing, sorted(set(missing)))
    ids = [c.get("checkpoint_id") for c in all_cps]
    check("checkpoint_id unique", len(ids) == len(set(ids)), {"total": len(ids), "unique": len(set(ids))})
    leaked = sorted({k for c in all_cps for k in c if k in FORBIDDEN})
    check("no answer fields in checkpoints", not leaked, leaked)
    check("parse_status recorded with a known value", all(c["parse_status"] in PARSE_STATUSES for c in all_cps))

    bad_index, bad_accum, bad_final, bad_tokens, bad_null, bad_offsets = [], [], [], [], [], []
    for qid, cps in by_qid.items():
        cps.sort(key=lambda c: c["checkpoint_index"])
        if qid not in generations:
            bad_index.append(qid)
            continue
        generation = generations[qid]
        reasoning = parse_generation(
            generation.get("generated_text"),
            think_tag_in_prompt=generation.get("generation_config", {}).get("think_tag_in_prompt", False),
        ).reasoning_text
        has_ids = bool(generations[qid].get("generated_token_ids"))
        if [c["checkpoint_index"] for c in cps] != list(range(len(cps))) or cps[0]["checkpoint_type"] != "no_reasoning":
            bad_index.append(qid)
        text_kept = cps[0]["reasoning_prefix"] is not None
        for prev, cur in zip(cps, cps[1:]):
            if cur["segment_char_start"] != prev["char_end"]:
                bad_offsets.append(qid)
            if text_kept and cur["reasoning_prefix"] != prev["reasoning_prefix"] + cur["segment_text"]:
                bad_accum.append(qid)
        if any(text_kept and c["reasoning_prefix"] != reasoning[: c["char_end"]] for c in cps):
            bad_accum.append(qid)
        if cps[-1]["char_end"] != len(reasoning) or (text_kept and cps[-1]["reasoning_prefix"] != reasoning):
            bad_final.append(qid)
        counts = [c["prefix_token_count"] for c in cps]
        if None not in counts and counts != sorted(counts):
            bad_tokens.append(qid)
        positions = [c[k] for c in cps for k in ("reasoning_token_start", "reasoning_token_end", "hidden_state_position")]
        if not has_ids and any(p is not None for p in positions):
            bad_null.append(qid)

    check("checkpoint indices contiguous from 0 (no_reasoning first)", not bad_index, bad_index)
    check("segment offsets contiguous", not bad_offsets, sorted(set(bad_offsets), key=str))
    check("reasoning_prefix accumulates exactly", not bad_accum, sorted(set(bad_accum), key=str))
    check("final checkpoint restores the full reasoning text", not bad_final, bad_final)
    check("prefix token counts non-decreasing", not bad_tokens, bad_tokens)
    check("token positions are null when original token ids are absent", not bad_null, bad_null)

    status_by_record = [{"run_id": key[0], "split": key[1], "question_id": key[2],
                         "parse_status": cps[0]["parse_status"], "checkpoints": len(cps)}
                        for key, cps in by_qid.items()]
    check("info: parse status and checkpoint count per record", True, status_by_record)

    report = {"generations": str(args.generations), "checkpoints": str(args.checkpoints),
              "passed": sum(r["pass"] for r in results), "failed": sum(not r["pass"] for r in results),
              "checks": results}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    print(f"\n{report['passed']} passed, {report['failed']} failed")
    sys.exit(1 if report["failed"] else 0)


if __name__ == "__main__":
    main()
