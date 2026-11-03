"""Build multimodal chat messages for Qwen vision-language processors.

Message format follows the Hugging Face model cards for these models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from thinkenough.schemas import InferenceSample


def build_qwen3_vl_messages(
    sample: InferenceSample,
    *,
    image_path: str | Path | None = None,
    video_frames: list[str] | None = None,
    extra_images: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Build a single-turn multimodal chat.

    Args:
        sample: Inference sample containing question (and default image path).
        image_path: Optional override path/URL. If None, uses sample.image_path.
            An empty path leaves the image out (text-only sample).
        video_frames: Optional frame paths, shown to the model as one video.
        extra_images: Optional paths shown after the image, before the question.

    Returns:
        A messages list compatible with ``processor.apply_chat_template``:
        ``[{"role": "user", "content": [{"type": "image", ...}, {"type": "text", ...}]}]``
    """
    resolved_image = str(image_path if image_path is not None else sample.image_path)
    content: list[dict[str, Any]] = [{"type": "text", "text": sample.question}]
    if resolved_image:  # An empty path means a text-only sample.
        content.insert(0, {"type": "image", "image": resolved_image})
    for extra in extra_images or []:
        content.insert(-1, {"type": "image", "image": extra})
    if video_frames:
        content.insert(0, {"type": "video", "video": video_frames})
    messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
    if sample.system:
        messages.insert(0, {"role": "system", "content": [{"type": "text", "text": sample.system}]})
    return messages


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
