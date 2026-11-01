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
from thinkenough.io_utils import load_samples_jsonl, save_result_json, save_results_jsonl
from thinkenough.schemas import InferenceSample


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

    if not input_path.is_file():
        # Fall back to a single built-in mock sample so CLI still works.
        print(
            f"[warn] Input JSONL not found: {input_path}. "
            "Using a built-in mock sample.",
            file=sys.stderr,
        )
        samples = [
            InferenceSample(
                sample_id="builtin_mock_0001",
                image_path=str(
                    _REPO_ROOT / "tests" / "fixtures" / "sample_image.png"
                ),
                question="What is in this image? (mock)",
            )
        ]
    else:
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

    runner = create_runner(config)
    output_dir = config.resolve_path(config.io.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    jsonl_path = output_dir / f"cot_{config.backend}_{stamp}.jsonl"

    results = []
    for sample in samples:
        print(f"[run] {sample.sample_id} ({config.backend})")
        result = runner.generate_cot(sample)
        results.append(result)

        per_sample_path = output_dir / f"{result.sample_id}_{config.backend}.json"
        save_result_json(
            result,
            per_sample_path,
            include_token_ids=config.io.save_token_ids,
        )
        print(
            f"  parse_status={result.parse_status} "
            f"reasoning_tokens={result.reasoning_token_count} "
            f"answer_tokens={result.answer_token_count} "
            f"latency={result.latency_seconds:.3f}s "
            f"-> {per_sample_path}"
        )

    save_results_jsonl(
        results,
        jsonl_path,
        include_token_ids=config.io.save_token_ids,
        overwrite=True,
    )
    print(f"[done] wrote {len(results)} result(s) to {jsonl_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
