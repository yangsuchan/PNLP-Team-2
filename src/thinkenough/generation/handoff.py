"""Export original generation artifacts to the Dataset team's JSONL contract."""
from __future__ import annotations

import re

from thinkenough.generation.output_validation import GenerationValidator
from thinkenough.schemas import InferenceResult


def generation_record(result: InferenceResult, run_id: str) -> dict:
    """Build an answer-free handoff; never reconstruct token IDs from text.

    Real runs require immutable model/tokenizer revisions. Mock records are
    explicitly synthetic and only validate the offline plumbing.
    """
    synthetic = result.backend == "mock"
    if not synthetic:
        for revision in (result.model_revision, result.tokenizer_revision):
            if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
                raise ValueError("Real handoff requires resolved 40-character model/tokenizer commit hashes")
    ids = result.output_token_ids
    close = ids.index(151668) if 151668 in ids else None
    start = ids.index(151667) + 1 if 151667 in ids else 0
    config = dict(result.generation_config)
    config.update(
        eos_token_ids=result.eos_token_ids,
        think_tag_in_prompt=151667 in result.input_token_ids and 151667 not in ids,
        batch_size=1,
    )
    record = {
        "question_id": result.question_id if result.question_id is not None else result.sample_id,
        "split": result.split,
        "run_id": run_id,
        "model": f"{result.model_name}@{result.model_revision}",
        "tokenizer": f"{result.tokenizer_name}@{result.tokenizer_revision}",
        "quantization": result.dtype,
        "prompt_version": result.prompt_version,
        "enable_thinking": True,
        "prompt_text": result.prompt_text,
        "prompt_token_ids": result.input_token_ids,
        "prompt_token_count": len(result.input_token_ids),
        "generated_token_ids": ids,
        "generated_token_count": len(ids),
        "generated_text": result.raw_output,
        "reasoning_token_count": close - start if close is not None else None,
        "think_close_token_index": close,
        "final_answer_text": result.raw_output.split("</think>", 1)[1]
        if close is not None and "</think>" in result.raw_output else None,
        "finish_reason": result.finish_reason,
        "generation_config": config,
        "hidden_state_ref": None,
        "timing": {"generation_ms": result.latency_seconds * 1000},
        "synthetic": synthetic,
    }
    # Metadata is intentionally not copied: it may contain reference answers.
    validator = GenerationValidator()
    validator.validate(record)
    if validator.report()["errors"]:
        raise ValueError(f"Invalid generation handoff: {validator.issues}")
    return record
