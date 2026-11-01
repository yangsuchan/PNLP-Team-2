"""ModelRunner interface and factory.

Hidden-State extraction and Forced-Answer will plug into this interface
in later phases without changing the Full CoT result schema.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

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

    # --- Phase 0+ extension points (not implemented yet) ---
    #
    # def extract_hidden_states(
    #     self,
    #     sample: InferenceSample,
    #     reasoning_prefix_token_ids: list[int],
    #     checkpoint_indices: list[int],
    # ) -> ...:
    #     """Extract last-layer hidden states at reasoning checkpoints."""
    #
    # def forced_answer(
    #     self,
    #     sample: InferenceSample,
    #     reasoning_prefix: str,
    # ) -> ...:
    #     """Generate a final answer from a truncated reasoning prefix."""


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
