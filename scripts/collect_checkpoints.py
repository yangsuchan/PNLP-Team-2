#!/usr/bin/env python3
"""Force answers and save hidden states at reasoning checkpoints of finished Full CoT runs.

    python scripts/collect_checkpoints.py --config configs/pilot.yaml \\
        --input data/processed/pilot_full.jsonl \\
        --runs outputs/pilot/thinking_full \\
        --output-dir outputs/pilot/checkpoints_full \\
        --tasks implicit_detection relationship_analysis

Writes checkpoints.jsonl and hidden/<sample_id>_cp<k>.pt, and resumes if interrupted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from thinkenough.checkpoints import pick_checkpoints
from thinkenough.config import load_inference_config
from thinkenough.inference import create_runner
from thinkenough.io_utils import load_samples_jsonl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect forced answers and hidden states")
    parser.add_argument("--config", default=str(_REPO_ROOT / "configs" / "pilot.yaml"))
    parser.add_argument("--backend", choices=["mock", "qwen3_vl"], default=None)
    parser.add_argument("--input", required=True, help="The same input JSONL as the Full CoT run")
    parser.add_argument("--runs", required=True, help="Directory of finished Full CoT results")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tasks", nargs="+", default=None, help="Default: every task")
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.2, 0.4, 0.6, 0.8, 1.0])
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args(argv)

    config = load_inference_config(args.config, repo_root=_REPO_ROOT)
    if args.backend:
        config.backend = args.backend
    runner = create_runner(config)
    runs = config.resolve_path(args.runs)
    out = config.resolve_path(args.output_dir)
    (out / "hidden").mkdir(parents=True, exist_ok=True)
    table = out / "checkpoints.jsonl"

    done: set[tuple[str, int]] = set()
    if table.is_file():
        with table.open(encoding="utf-8") as f:
            done = {(row["sample_id"], row["checkpoint_idx"]) for row in map(json.loads, f)}

    import torch

    samples = load_samples_jsonl(config.resolve_path(args.input))
    if args.tasks:
        samples = [s for s in samples if s.metadata.get("task") in args.tasks]
    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    skipped = 0
    for sample in samples:
        path = runs / f"{sample.sample_id}_{config.backend}.json"
        if not path.is_file():
            print(f"[error] no Full CoT result for {sample.sample_id}: {path}", file=sys.stderr)
            return 1
        run = json.loads(path.read_text(encoding="utf-8"))
        if run["parse_status"] != "ok":
            # Truncated reasoning has no true end, so positions would be meaningless.
            skipped += 1
            continue
        reasoning_ids = run["output_token_ids"][: run["metadata"]["think_end_index"]]

        chosen = pick_checkpoints(runner.decode_tokens(reasoning_ids), args.fractions)
        for index, (fraction, length) in enumerate(chosen):
            if (sample.sample_id, index) in done:
                continue
            prefix = reasoning_ids[:length]
            hidden = torch.as_tensor(runner.extract_hidden_state(sample, prefix))
            hidden_path = out / "hidden" / f"{sample.sample_id}_cp{index}.pt"
            torch.save(hidden, hidden_path)
            forced = runner.forced_answer(sample, prefix_token_ids=prefix)
            row = {
                "sample_id": sample.sample_id,
                "model_name": forced.model_name,
                "model_revision": forced.model_revision,
                "checkpoint_idx": index,
                "fraction": fraction,
                "prefix_tokens": length,
                "reasoning_tokens_total": len(reasoning_ids),
                "answer": forced.final_answer,
                "answer_tokens": forced.answer_token_count,
                "latency_seconds": forced.latency_seconds,
                "hidden_state_path": str(hidden_path.relative_to(out)),
                "hidden_state_shape": list(hidden.shape),
                "hidden_state_layer": "last",
                "metadata": sample.metadata,
            }
            with table.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[done] {sample.sample_id}: {len(chosen)} checkpoints")
    print(f"[done] wrote {table}; skipped {skipped} truncated samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
