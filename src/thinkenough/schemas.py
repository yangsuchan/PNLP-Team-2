"""Data schemas for multimodal CoT inference.

These dataclasses define the interface between the Model & Inference team
and Dataset / Probe teams. Fields follow AGENTS.md metadata guidelines.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class InferenceSample:
    """One multimodal inference request.

    Inputs:
        sample_id: Stable identifier used across teams.
        image_path: Local path or URL to the input image.
        question: Text question paired with the image.
        metadata: Optional free-form fields (dataset split, etc.).
    """

    sample_id: str
    image_path: str
    question: str
    metadata: dict[str, Any] = field(default_factory=dict)
    question_id: int | str | None = None
    split: str = "test"
    prompt_version: str = "image_question_v1"


@dataclass
class ParsedOutput:
    """Result of splitting Thinking / Final Answer from model text.

    Inputs originate from `split_thinking_and_answer`.
    Outputs:
        reasoning_text: Content before the think-end marker.
        final_answer: Content after the think-end marker.
        parse_status: ok | missing_think_end | empty_generation
    """

    reasoning_text: str
    final_answer: str
    parse_status: str


@dataclass
class TokenCounts:
    """Token counts derived from generated token IDs.

    Inputs: generated token ID list + think_end token id.
    Outputs:
        reasoning_token_count: Tokens between think tags; None if think_end is missing.
        answer_token_count: Tokens after think_end (0 if missing).
        total_generated_token_count: len(output_token_ids).
        think_end_index: Index of think_end in output_token_ids, or -1.
    """

    reasoning_token_count: int | None
    answer_token_count: int
    total_generated_token_count: int
    think_end_index: int


@dataclass
class InferenceResult:
    """Full CoT generation result for one sample.

    This is the primary artifact saved as JSON/JSONL.
    Raw generation fields support Dataset-team handoff. Hidden-state extraction
    and forced-answer generation are separate, unimplemented extensions.
    """

    sample_id: str
    model_name: str
    model_revision: str | None
    dtype: str
    image_path: str
    question: str
    generation_config: dict[str, Any]
    raw_output: str
    reasoning_text: str
    final_answer: str
    input_token_ids: list[int]
    output_token_ids: list[int]
    reasoning_token_count: int | None
    answer_token_count: int
    total_generated_token_count: int
    latency_seconds: float
    parse_status: str
    backend: str
    created_at: str = field(default_factory=_utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    prompt_text: str | None = None
    tokenizer_name: str | None = None
    tokenizer_revision: str | None = None
    finish_reason: str | None = None
    eos_token_ids: list[int] = field(default_factory=list)
    question_id: int | str | None = None
    split: str = "test"
    prompt_version: str = "image_question_v1"

    def to_dict(self, *, include_token_ids: bool = True) -> dict[str, Any]:
        """Convert to a JSON-serializable dict.

        Args:
            include_token_ids: If False, omit raw token ID lists (smaller files).

        Returns:
            Flat dictionary suitable for json.dump / JSONL.
        """
        data = asdict(self)
        if not include_token_ids:
            data["input_token_ids"] = []
            data["output_token_ids"] = []
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InferenceResult:
        """Rebuild InferenceResult from a saved dictionary."""
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)
