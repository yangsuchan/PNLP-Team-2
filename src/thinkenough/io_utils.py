"""JSON / JSONL helpers for inference inputs and results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

from thinkenough.schemas import InferenceResult, InferenceSample


def load_samples_jsonl(path: str | Path) -> list[InferenceSample]:
    """Load multimodal samples from a JSONL file.

    Each line must contain at least:
        sample_id, image_path, question

    Optional ``metadata`` object is preserved.

    Args:
        path: Path to a JSONL file.

    Returns:
        List of InferenceSample objects.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If a line is missing required fields.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Input JSONL not found: {file_path}")

    samples: list[InferenceSample] = []
    with file_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc

            missing = [k for k in ("sample_id", "image_path", "question") if k not in obj]
            if missing:
                raise ValueError(
                    f"Line {line_no} missing required fields: {', '.join(missing)}"
                )

            samples.append(
                InferenceSample(
                    sample_id=str(obj["sample_id"]),
                    image_path=str(obj["image_path"]),
                    question=str(obj["question"]),
                    metadata=dict(obj.get("metadata") or {}),
                )
            )
    return samples


def save_result_json(
    result: InferenceResult,
    path: str | Path,
    *,
    include_token_ids: bool = True,
) -> Path:
    """Save a single InferenceResult as a pretty-printed JSON file.

    Args:
        result: Inference result to save.
        path: Destination file path.
        include_token_ids: Whether to store raw token ID lists.

    Returns:
        Absolute path of the written file.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(
            result.to_dict(include_token_ids=include_token_ids),
            f,
            ensure_ascii=False,
            indent=2,
        )
        f.write("\n")
    return out.resolve()


def append_result_jsonl(
    result: InferenceResult,
    path: str | Path,
    *,
    include_token_ids: bool = True,
) -> Path:
    """Append one InferenceResult as a JSONL line.

    Args:
        result: Inference result to append.
        path: Destination JSONL path (created if missing).
        include_token_ids: Whether to store raw token ID lists.

    Returns:
        Absolute path of the JSONL file.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                result.to_dict(include_token_ids=include_token_ids),
                ensure_ascii=False,
            )
        )
        f.write("\n")
    return out.resolve()


def save_results_jsonl(
    results: Iterable[InferenceResult],
    path: str | Path,
    *,
    include_token_ids: bool = True,
    overwrite: bool = True,
) -> Path:
    """Write multiple InferenceResult objects to a JSONL file.

    Args:
        results: Iterable of results.
        path: Destination JSONL path.
        include_token_ids: Whether to store raw token ID lists.
        overwrite: If True, replace an existing file; else append.

    Returns:
        Absolute path of the JSONL file.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "a"
    with out.open(mode, encoding="utf-8") as f:
        for result in results:
            f.write(
                json.dumps(
                    result.to_dict(include_token_ids=include_token_ids),
                    ensure_ascii=False,
                )
            )
            f.write("\n")
    return out.resolve()


def iter_results_jsonl(path: str | Path) -> Iterator[InferenceResult]:
    """Yield InferenceResult objects from a JSONL file."""
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield InferenceResult.from_dict(json.loads(line))
