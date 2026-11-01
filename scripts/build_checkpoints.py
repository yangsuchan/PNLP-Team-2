"""Build reasoning checkpoints from Qwen3 generation JSONL.

요약: [2단계] Qwen3 생성 결과(JSONL)를 읽어 checkpoint 파일을 만든다. 인자 없이 실행하면 가짜 예시로 동작한다.

Input format: docs/qwen3_output_spec.md. Reference answers are not read.
Token positions are filled only when a record carries its original
`generated_token_ids`; otherwise they are null and token counts come from
re-tokenizing the prefix text (token_count_source = "retokenized").

Usage (from the repo root):
    .venv/bin/python scripts/build_checkpoints.py                       # synthetic example
    .venv/bin/python scripts/build_checkpoints.py --input gen.jsonl --output out.jsonl \
        --segmenter sentence [--no-prefix-text] [--no-tokenizer]
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thinkenough.data.reasoning_checkpoint import SEGMENTERS, QwenTokenAligner, build_checkpoints  # noqa: E402

DEFAULT_INPUT = ROOT / "tests/fixtures/fake_qwen3_outputs.jsonl"
DEFAULT_OUTPUT = ROOT / "data/processed/checkpoints/example_checkpoints.jsonl"
ANSWER_KEYS = ("reference_answers", "answers", "answer")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--segmenter", choices=sorted(SEGMENTERS), default="sentence")
    parser.add_argument("--tokenizer", help="Exact tokenizer ID@revision (required unless --no-tokenizer)")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-tokenizer", action="store_true", help="skip token counts entirely")
    parser.add_argument("--no-prefix-text", action="store_true",
                        help="store char offsets only (prefix = reasoning[:char_end]) to save space")
    args = parser.parse_args()

    tokenizer = aligner = None
    if not args.no_tokenizer:
        if not args.tokenizer or "@" not in args.tokenizer:
            parser.error("Provide --tokenizer ID@revision or --no-tokenizer for unaligned CPU smoke tests")
        from transformers import AutoTokenizer  # tokenizer only; no model weights, no GPU
        name, revision = args.tokenizer.rsplit("@", 1)
        tokenizer = AutoTokenizer.from_pretrained(name, revision=revision, local_files_only=args.local_files_only)
        aligner = QwenTokenAligner(tokenizer)

    parse_status, alignment, n_checkpoints, dropped_answer_keys = Counter(), Counter(), [], 0
    if args.input.resolve() == args.output.resolve():
        parser.error("Input and output paths must differ")
    # Validate all rows and identities before touching output.
    from thinkenough.identity import valid_question_id
    from thinkenough.generation.output_validation import read_jsonl
    records, seen = [], set()
    for line, record, problem in read_jsonl(args.input):
        if problem or not isinstance(record, dict):
            raise ValueError(f"Invalid input at line {line}: {problem}")
        qid = record.get("question_id")
        if not valid_question_id(qid):
            raise ValueError(f"Invalid question_id at line {line}")
        key = (record.get("run_id"), qid)
        if key in seen:
            raise ValueError(f"Duplicate run/question identity at line {line}")
        seen.add(key)
        if not args.no_tokenizer and record.get("tokenizer") != args.tokenizer:
            raise ValueError(f"Tokenizer identity mismatch at line {line}")
        records.append(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as fout:
        for record in records:
            # defensive: answers must never influence checkpoints
            for key in ANSWER_KEYS:
                if key in record:
                    record.pop(key)
                    dropped_answer_keys += 1
            checkpoints, parsed = build_checkpoints(
                record, segmenter=args.segmenter, tokenizer=tokenizer, aligner=aligner,
                keep_prefix_text=not args.no_prefix_text,
            )
            parse_status[parsed.parse_status] += 1
            alignment.update(c["alignment_status"] for c in checkpoints)
            n_checkpoints.append(len(checkpoints))
            for c in checkpoints:
                fout.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(json.dumps({
        "input": str(args.input.relative_to(ROOT)) if args.input.is_relative_to(ROOT) else str(args.input),
        "output": str(args.output.relative_to(ROOT)) if args.output.is_relative_to(ROOT) else str(args.output),
        "segmenter": args.segmenter,
        "records": len(n_checkpoints),
        "checkpoints": sum(n_checkpoints),
        "checkpoints_per_record": n_checkpoints,
        "parse_status": dict(parse_status),
        "alignment_status": dict(alignment),
        "answer_keys_dropped_from_input": dropped_answer_keys,
    }, indent=1))


if __name__ == "__main__":
    main()
