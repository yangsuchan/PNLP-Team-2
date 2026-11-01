"""ModelRunner interface and factory.

Hidden-State extraction and Forced-Answer will plug into this interface
in later phases without changing the Full CoT result schema.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from thinkenough.config import InferenceConfig
from thinkenough.schemas import InferenceResult, InferenceSample


@runtime_checkable
class ModelRunner(Protocol):
    """Protocol for multimodal Full CoT generation.

    Implementations must keep the base model frozen (no training).
    """

    def generate_cot(self, sample: InferenceSample) -> InferenceResult:
        """Generate a full Chain-of-Thought for one multimodal sample.

        Args:
            sample: Image path + question (+ optional metadata).

        Returns:
            InferenceResult with raw output, parsed reasoning/answer,
            token IDs, token counts, and latency.
        """
        ...

    def forced_answer(
        self,
        sample: InferenceSample,
        reasoning_prefix: str = "",
        prefix_token_ids: list[int] | None = None,
    ) -> InferenceResult:
        """Answer immediately from a reasoning prefix. Empty prefix = direct answer."""
        ...

    def extract_hidden_state(
        self, sample: InferenceSample, prefix_token_ids: list[int]
    ) -> Any:
        """Last-layer hidden state at the last token of a reasoning prefix."""
        ...

    def decode_tokens(self, token_ids: list[int]) -> list[str]:
        """Decode each token separately, for finding sentence ends."""
        ...


def create_runner(config: InferenceConfig) -> ModelRunner:
    """Instantiate a runner from config.backend.

    Args:
        config: Loaded InferenceConfig.

    Returns:
        MockRunner or Qwen3VLRunner.

    Raises:
        ValueError: Unknown backend.
        ImportError: Real backend requested but dependencies missing.
    """
    if config.backend == "mock":
        from thinkenough.inference.mock_runner import MockRunner

        return MockRunner(config)

    if config.backend == "qwen3_vl":
        from thinkenough.inference.qwen3_vl_runner import Qwen3VLRunner

        return Qwen3VLRunner(config)

    raise ValueError(f"Unknown backend: {config.backend}")
