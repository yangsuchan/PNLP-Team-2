#!/usr/bin/env python3
"""Force answers and save hidden states every N reasoning tokens of finished Full CoT runs.

    python scripts/collect_checkpoints.py --config configs/pilot.yaml \\
        --input data/processed/pilot_full.jsonl \\
        --runs outputs/pilot/thinking_full \\
        --output-dir outputs/pilot/checkpoints_full \\
        --tasks implicit_detection relationship_analysis

Writes samples.jsonl (one row per question), checkpoints.jsonl (one row per forced
answer), and hidden/*.pt, and resumes if interrupted. Truncated reasoning is kept: it
counts as a missing answer for full thinking, while earlier checkpoints stay valid.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from thinkenough.checkpoints import interval_checkpoints
from thinkenough.config import load_inference_config
from thinkenough.inference import create_runner
from thinkenough.io_utils import load_samples_jsonl


def _read_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _append(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect forced answers and hidden states")
    parser.add_argument("--config", default=str(_REPO_ROOT / "configs" / "pilot.yaml"))
    parser.add_argument("--backend", choices=["mock", "qwen3_vl"], default=None)
    parser.add_argument("--input", required=True, help="The same input JSONL as the Full CoT run")
    parser.add_argument("--runs", required=True, help="Directory of finished Full CoT results")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tasks", nargs="+", default=None, help="Default: every task")
    parser.add_argument("--interval", type=int, default=256, help="Reasoning tokens between checkpoints")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--answer-max-tokens",
        type=int,
        default=1024,
        help="Cap on one forced answer; the longest normal one seen is 641 tokens, and an answer that wanders on would cost minutes",
    )
    args = parser.parse_args(argv)

    import torch

    config = load_inference_config(args.config, repo_root=_REPO_ROOT)
    config.generation.max_new_tokens = args.answer_max_tokens
    if args.backend:
        config.backend = args.backend
    runner = create_runner(config)
    runs = config.resolve_path(args.runs)
    out = config.resolve_path(args.output_dir)
    (out / "hidden").mkdir(parents=True, exist_ok=True)
    sample_table, checkpoint_table = out / "samples.jsonl", out / "checkpoints.jsonl"

    samples_done = {row["sample_id"] for row in _read_rows(sample_table)}
    checkpoints_done = {(r["sample_id"], r["checkpoint_idx"]) for r in _read_rows(checkpoint_table)}

    samples = load_samples_jsonl(config.resolve_path(args.input))
    if args.tasks:
        samples = [s for s in samples if s.metadata.get("task") in args.tasks]
    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    for sample in samples:
        path = runs / f"{sample.sample_id}_{config.backend}.json"
        if not path.is_file():
            print(f"[error] no Full CoT result for {sample.sample_id}: {path}", file=sys.stderr)
            return 1
        run = json.loads(path.read_text(encoding="utf-8"))
        think_end = run["metadata"].get("think_end_index", -1)
        output_ids = run["output_token_ids"]
        reasoning_ids = output_ids[:think_end] if think_end >= 0 else output_ids

        start_path = out / "hidden" / f"{sample.sample_id}_start.pt"
        if not start_path.is_file():
            torch.save(torch.as_tensor(runner.extract_hidden_state(sample, [])), start_path)

        lengths = interval_checkpoints(len(reasoning_ids), args.interval)
        done = frozenset(i for i in range(len(lengths)) if (sample.sample_id, i) in checkpoints_done)
        for index, length, states, forced in runner.checkpoints(sample, reasoning_ids, lengths, done):
            hidden = torch.as_tensor(states)
            hidden_path = out / "hidden" / f"{sample.sample_id}_cp{index}.pt"
            torch.save(hidden, hidden_path)
            _append(
                checkpoint_table,
                {
                    "sample_id": sample.sample_id,
                    "model_name": forced.model_name,
                    "model_revision": forced.model_revision,
                    "checkpoint_idx": index,
                    "prefix_tokens": length,
                    "answer": forced.final_answer,
                    "answer_tokens": forced.answer_token_count,
                    "latency_seconds": forced.latency_seconds,
                    "hidden_state_path": str(hidden_path.relative_to(out)),
                    "hidden_state_shape": list(hidden.shape),
                },
            )

        # Written last, so an interrupted question is finished on the next run.
        if sample.sample_id not in samples_done:
            _append(
                sample_table,
                {
                    "sample_id": sample.sample_id,
                    "full_tokens": len(reasoning_ids),
                    "full_answer": run["final_answer"],
                    "full_status": run["parse_status"],
                    "budget": run["generation_config"]["max_new_tokens"],
                    "interval": args.interval,
                    "start_hidden_path": str(start_path.relative_to(out)),
                    "metadata": sample.metadata,
                },
            )
        print(f"[done] {sample.sample_id}: {len(reasoning_ids)} reasoning tokens")
    print(f"[done] wrote {sample_table} and {checkpoint_table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
