"""Build multimodal chat messages for Qwen3-VL processors.

Message format follows the official Hugging Face Qwen3-VL example:
https://huggingface.co/Qwen/Qwen3-VL-8B-Thinking
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from thinkenough.schemas import InferenceSample


def build_qwen3_vl_messages(
    sample: InferenceSample,
    *,
    image_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Build a single-turn multimodal chat for Qwen3-VL.

    Args:
        sample: Inference sample containing question (and default image path).
        image_path: Optional override path/URL. If None, uses sample.image_path.

    Returns:
        A messages list compatible with ``processor.apply_chat_template``:
        ``[{"role": "user", "content": [{"type": "image", ...}, {"type": "text", ...}]}]``
    """
    resolved_image = str(image_path if image_path is not None else sample.image_path)
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": resolved_image},
                {"type": "text", "text": sample.question},
            ],
        }
    ]


def validate_messages(messages: list[dict[str, Any]]) -> None:
    """Validate that messages contain at least one image and one text item.

    Raises:
        ValueError: If the structure is incomplete.
    """
    if not messages:
        raise ValueError("messages must be non-empty")

    user = messages[0]
    if user.get("role") != "user":
        raise ValueError("first message role must be 'user'")

    content = user.get("content")
    if not isinstance(content, list) or not content:
        raise ValueError("user content must be a non-empty list")

    types = {item.get("type") for item in content if isinstance(item, dict)}
    if "image" not in types:
        raise ValueError("messages must include an image content item")
    if "text" not in types:
        raise ValueError("messages must include a text content item")
