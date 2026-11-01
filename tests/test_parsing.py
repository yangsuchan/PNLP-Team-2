"""Unit tests for Thinking / Answer parsing (no model required)."""

from __future__ import annotations

import json
from pathlib import Path

from thinkenough.parsing import (
    DEFAULT_THINK_END_TOKEN_ID,
    count_reasoning_tokens,
    decode_segments_from_token_ids,
    split_thinking_and_answer,
)

FIXTURES = Path(__file__).parent / "fixtures" / "mock_outputs.json"


def _fixtures() -> dict:
    return json.loads(FIXTURES.read_text(encoding="utf-8"))


def test_split_prefix_format():
    text = _fixtures()["prefix_format"]
    parsed = split_thinking_and_answer(text)
    assert parsed.parse_status == "ok"
    assert "Step 1" in parsed.reasoning_text
    assert parsed.final_answer == "The answer is blue."


def test_split_full_tag_format():
    # Opening tag may appear in text; we still split on the first think_end.
    text = _fixtures()["full_tag_format"]
    parsed = split_thinking_and_answer(text)
    assert parsed.parse_status == "ok"
    assert "reason step by step" in parsed.reasoning_text or "I reason" in parsed.reasoning_text
    assert parsed.final_answer == "Final: 42"


def test_split_missing_think_end():
    text = _fixtures()["missing_think_end"]
    parsed = split_thinking_and_answer(text)
    assert parsed.parse_status == "missing_think_end"
    assert parsed.final_answer == ""
    assert "Still thinking" in parsed.reasoning_text


def test_split_empty_generation():
    parsed = split_thinking_and_answer(_fixtures()["empty"])
    assert parsed.parse_status == "empty_generation"
    assert parsed.reasoning_text == ""
    assert parsed.final_answer == ""


def test_count_reasoning_tokens_with_end():
    # reasoning tokens + think_end + answer tokens
    ids = [10, 11, 12, DEFAULT_THINK_END_TOKEN_ID, 20, 21]
    counts = count_reasoning_tokens(ids)
    assert counts.think_end_index == 3
    assert counts.reasoning_token_count == 4  # includes think_end
    assert counts.answer_token_count == 2
    assert counts.total_generated_token_count == 6


def test_count_reasoning_tokens_missing_end():
    ids = [10, 11, 12]
    counts = count_reasoning_tokens(ids)
    assert counts.think_end_index == -1
    assert counts.reasoning_token_count == 3
    assert counts.answer_token_count == 0


def test_decode_segments_from_token_ids():
    ids = [1, 2, DEFAULT_THINK_END_TOKEN_ID, 3, 4]

    def decode_fn(token_ids: list[int]) -> str:
        return " ".join(str(t) for t in token_ids)

    parsed = decode_segments_from_token_ids(ids, decode_fn)
    assert parsed.parse_status == "ok"
    assert parsed.reasoning_text == "1 2"
    assert parsed.final_answer == "3 4"
