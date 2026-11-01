"""Build STOP/CONTINUE labels from checkpoints + forced answers + references.

요약: [3단계] checkpoint + 강제 답변 + 정답으로 STOP/CONTINUE 라벨 파일을 만든다. 인자 없이 실행하면 가짜 예시로 동작한다.

Defaults run on the synthetic example. For real runs pass the A-side forced
answer file and the processed SQA3D splits as references.

Usage (from the repo root):
    .venv/bin/python scripts/build_checkpoint_labels.py
    .venv/bin/python scripts/build_checkpoint_labels.py \
        --checkpoints data/processed/checkpoints/<run>.jsonl --answers <forced_answers>.jsonl \
        --references data/processed/sqa3d/train.jsonl data/processed/sqa3d/validation.jsonl \
        --output data/processed/checkpoint_labels/<run>_labels.jsonl
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thinkenough.evaluation.checkpoint_labeling import build_labels, load_references  # noqa: E402


def read_jsonl(path):
    return [json.loads(line) for line in open(path)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=Path, default=ROOT / "data/processed/checkpoints/example_checkpoints.jsonl")
    parser.add_argument("--answers", type=Path, default=ROOT / "tests/fixtures/fake_checkpoint_answers.jsonl")
    parser.add_argument("--references", type=Path, nargs="+", default=[ROOT / "tests/fixtures/fake_references.jsonl"])
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/checkpoint_labels/example_labels.jsonl")
    args = parser.parse_args()

    rows, report = build_labels(read_jsonl(args.checkpoints), read_jsonl(args.answers), load_references(args.references))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    report.update({
        "output": str(args.output.relative_to(ROOT)) if args.output.is_relative_to(ROOT) else str(args.output),
        "label_status": dict(Counter(r["label_status"] for r in rows)),
        "stop_label": dict(Counter(str(r["stop_label"]) for r in rows)),
        "flags": dict(Counter(f for r in rows for f in r["flags"])),
        "labeled_by_split": dict(Counter(r["split"] for r in rows if r["label_status"] == "labeled")),
    })
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
