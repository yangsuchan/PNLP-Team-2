"""Mock runner for local pipeline validation without model weights."""

from __future__ import annotations

import time
from pathlib import Path

from thinkenough.config import InferenceConfig
from thinkenough.parsing import count_reasoning_tokens, split_thinking_and_answer
from thinkenough.schemas import InferenceResult, InferenceSample

class MockRunner:
    """Return deterministic fake CoT outputs for offline testing.

    Does not load torch, transformers, or any model weights.
    """

    def __init__(self, config: InferenceConfig) -> None:
        self.config = config
        self.think_end_token_id = config.thinking.think_end_token_id
        self.think_end_text = config.thinking.think_end_text

    def generate_cot(self, sample: InferenceSample) -> InferenceResult:
        """Produce a fake Full CoT result for ``sample``.

        Args:
            sample: Multimodal sample (image path is recorded, not opened).

        Returns:
            InferenceResult with mock reasoning/answer and synthetic token IDs.
        """
        started = time.perf_counter()

        reasoning = (
            f"[mock reasoning for {sample.sample_id}] "
            f"I inspect the image and the question: {sample.question}"
        )
        answer = f"[mock answer] sample={sample.sample_id}"
        raw_output = f"{reasoning}\n{self.think_end_text}\n\n{answer}"

        if self.config.generation.max_new_tokens < 10:
            raise ValueError("Mock fixture requires max_new_tokens >= 10")
        # Synthetic IDs: a few placeholder tokens, then think_end, then answer tokens.
        input_token_ids = [1, 2, 3, 4, 151667]
        output_token_ids = [10, 11, 12, 13, 14, self.think_end_token_id, 20, 21, 22, 151645]

        parsed = split_thinking_and_answer(
            raw_output, think_end_text=self.think_end_text
        )
        counts = count_reasoning_tokens(
            output_token_ids, think_end_token_id=self.think_end_token_id
        )

        latency = time.perf_counter() - started
        image_path = str(sample.image_path)
        # Prefer absolute path when the file exists under repo_root.
        resolved = self.config.resolve_path(sample.image_path)
        if resolved.exists():
            image_path = str(resolved)

        return InferenceResult(
            sample_id=sample.sample_id,
            model_name=f"mock::{self.config.model.model_id}",
            model_revision="synthetic-v1",
            dtype=self.config.model.dtype,
            image_path=image_path,
            question=sample.question,
            generation_config=self.config.generation.to_dict(),
            raw_output=raw_output,
            reasoning_text=parsed.reasoning_text,
            final_answer=parsed.final_answer,
            input_token_ids=input_token_ids,
            output_token_ids=output_token_ids,
            reasoning_token_count=counts.reasoning_token_count,
            answer_token_count=counts.answer_token_count,
            total_generated_token_count=counts.total_generated_token_count,
            latency_seconds=latency,
            parse_status=parsed.parse_status,
            backend="mock",
            prompt_text="[synthetic mock prompt]<think>",
            tokenizer_name="mock-tokenizer",
            tokenizer_revision="synthetic-v1",
            finish_reason="stop",
            eos_token_ids=[151645],
            question_id=sample.question_id,
            split=sample.split,
            prompt_version=sample.prompt_version,
            metadata={
                **sample.metadata,
                "mock": True,
                "image_exists": Path(image_path).exists(),
            },
        )
