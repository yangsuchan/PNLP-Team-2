#!/usr/bin/env python3
"""CLI for ThinkEnough Full CoT inference.

Local (safe default):
    python scripts/run_inference.py --config configs/inference_default.yaml

GPU server (requires approval + transformers/torch):
    python scripts/run_inference.py \\
        --config configs/inference_default.yaml \\
        --backend qwen3_vl \\
        --input data/samples/example_inputs.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from uuid import uuid4
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow running without editable install: add src/ to sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from thinkenough.config import load_inference_config
from thinkenough.inference import create_runner
from thinkenough.io_utils import load_samples_jsonl, save_result_json
from thinkenough.identity import valid_question_id
from thinkenough.generation.handoff import generation_record
from thinkenough.generation.output_validation import GenerationValidator


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ThinkEnough multimodal Full CoT inference"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(_REPO_ROOT / "configs" / "inference_default.yaml"),
        help="Path to inference YAML config",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["mock", "qwen3_vl"],
        default=None,
        help="Override config backend (default: value from YAML, usually mock)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input JSONL (sample_id, image_path, question). Overrides config.io.input_jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Overrides config.io.output_dir",
    )
    parser.add_argument(
        "--sample-id",
        type=str,
        default=None,
        help="If set, only run this sample_id from the input JSONL",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit on number of samples",
    )
    parser.add_argument("--run-id", help="Unique run directory name; existing directories are rejected")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_inference_config(args.config, repo_root=_REPO_ROOT)

    if args.backend is not None:
        config.backend = args.backend
    if args.output_dir is not None:
        config.io.output_dir = args.output_dir

    input_path = (
        config.resolve_path(args.input)
        if args.input
        else config.resolve_path(config.io.input_jsonl)
    )

    # A missing server input must fail, never silently run a fake sample.
    samples = load_samples_jsonl(input_path)

    if args.sample_id:
        samples = [s for s in samples if s.sample_id == args.sample_id]
        if not samples:
            print(f"[error] sample_id not found: {args.sample_id}", file=sys.stderr)
            return 1

    if args.max_samples is not None:
        samples = samples[: max(0, args.max_samples)]

    if config.backend == "qwen3_vl":
        print(
            "[warn] backend=qwen3_vl will load Qwen3-VL-8B-Thinking weights. "
            "Ensure you are on a GPU server with enough VRAM and have approval "
            "for model download / inference.",
            file=sys.stderr,
        )

    if not samples:
        raise ValueError("No samples selected")
    if not config.io.save_token_ids:
        raise ValueError("Team handoff requires io.save_token_ids=true")
    seen_ids = set()
    for sample in samples:
        qid = sample.question_id if sample.question_id is not None else sample.sample_id
        if not valid_question_id(qid) or sample.split not in {"train", "validation", "test"}:
            raise ValueError("Invalid question_id or split")
        if qid in seen_ids:
            raise ValueError(f"Duplicate question_id in one run: {qid}")
        seen_ids.add(qid)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or f"{config.backend}_{stamp}_{uuid4().hex[:8]}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
        raise ValueError("run_id must use letters, digits, underscores, dots or hyphens")
    output_dir = config.resolve_path(config.io.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = output_dir / "run_manifest.json"
    manifest = {"run_id": run_id, "backend": config.backend, "status": "incomplete",
                "selected_samples": len(samples), "completed_samples": 0,
                "input_jsonl": str(input_path)}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    runner = create_runner(config)
    validator = GenerationValidator()
    # Stream each completed sample to disk so a later failure preserves progress.
    with (output_dir / "generations.jsonl").open("x", encoding="utf-8") as handoff_file, \
         (output_dir / "results.jsonl").open("x", encoding="utf-8") as results_file:
        for index, sample in enumerate(samples):
            print(f"[run] {sample.sample_id} ({config.backend})")
            result = runner.generate_cot(sample)
            record = generation_record(result, run_id)
            validator.validate(record, index + 1)
            if validator.report()["errors"]:
                raise ValueError(f"Inconsistent handoff run: {validator.issues}")
            # Numeric filenames avoid treating arbitrary dataset IDs as paths.
            save_result_json(result, output_dir / f"sample_{index:06d}.json")
            handoff_file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            handoff_file.flush()
            results_file.write(json.dumps(result.to_dict(), ensure_ascii=False, allow_nan=False) + "\n")
            results_file.flush()
            manifest["completed_samples"] = index + 1
            temporary_manifest = output_dir / ".run_manifest.tmp"
            temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            temporary_manifest.replace(manifest_path)
    manifest["status"] = "complete"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[done] wrote {len(samples)} result(s) to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
