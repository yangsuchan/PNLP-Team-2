"""Tests for multimodal message construction."""

from __future__ import annotations

import pytest

from thinkenough.prompts import build_qwen3_vl_messages, validate_messages
from thinkenough.schemas import InferenceSample


def test_build_qwen3_vl_messages_structure():
    sample = InferenceSample(
        sample_id="s1",
        image_path="data/samples/sample_image.png",
        question="What color?",
    )
    messages = build_qwen3_vl_messages(sample)
    validate_messages(messages)

    assert messages[0]["role"] == "user"
    content = messages[0]["content"]
    types = [c["type"] for c in content]
    assert types == ["image", "text"]
    assert content[0]["image"] == "data/samples/sample_image.png"
    assert content[1]["text"] == "What color?"


def test_validate_messages_requires_image_and_text():
    with pytest.raises(ValueError, match="image"):
        validate_messages(
            [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        )
