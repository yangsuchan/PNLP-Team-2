"""Qwen3-VL-8B-Thinking Full CoT runner (Hugging Face Transformers).

Implements the official quickstart from:
https://huggingface.co/Qwen/Qwen3-VL-8B-Thinking

WARNING:
- Do not run this on the local 8GB VRAM machine for full BF16 inference.
- Model download and GPU inference require explicit user approval.
- This module keeps the base model frozen (eval + inference_mode only).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from thinkenough.config import InferenceConfig
from thinkenough.parsing import (
    count_reasoning_tokens,
    decode_segments_from_token_ids,
    split_thinking_and_answer,
)
from thinkenough.prompts import build_qwen3_vl_messages
from thinkenough.schemas import InferenceResult, InferenceSample, ParsedOutput


def _resolve_torch_dtype(dtype_name: str):
    import torch

    mapping = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(
            f"Unsupported dtype '{dtype_name}'. "
            f"Choose one of: {', '.join(sorted(mapping))}"
        )
    return mapping[dtype_name]


class Qwen3VLRunner:
    """Frozen Qwen3-VL Thinking runner for Full CoT generation.

    Extension hooks for Hidden State / Forced-Answer are documented on
    ``ModelRunner`` but intentionally not implemented here yet.
    """

    def __init__(self, config: InferenceConfig) -> None:
        self.config = config
        self.model = None
        self.processor = None
        self._loaded = False

    def load(self) -> None:
        """Load processor and model weights.

        Raises:
            ImportError: If transformers / torch are not installed.
        """
        if self._loaded:
            return

        try:
            import torch
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
            from transformers import set_seed
        except ImportError as exc:
            raise ImportError(
                "Qwen3VLRunner requires torch and transformers>=4.57.0. "
                "Install server dependencies only after approval. "
                "See requirements.txt."
            ) from exc

        set_seed(self.config.generation.seed)

        model_kwargs: dict[str, Any] = {
            "dtype": _resolve_torch_dtype(self.config.model.dtype),
            "device_map": self.config.model.device_map,
        }
        if self.config.model.revision:
            model_kwargs["revision"] = self.config.model.revision
        if self.config.model.attn_implementation:
            model_kwargs["attn_implementation"] = self.config.model.attn_implementation

        self.processor = AutoProcessor.from_pretrained(
            self.config.model.model_id,
            revision=self.config.model.revision,
        )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.config.model.model_id,
            **model_kwargs,
        )
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        self._torch = torch
        self._loaded = True

    def generate_cot(self, sample: InferenceSample) -> InferenceResult:
        """Run Full CoT generation for one multimodal sample.

        Args:
            sample: Image + question sample.

        Returns:
            InferenceResult with parsed reasoning/answer and token metadata.
        """
        self.load()
        assert self.model is not None and self.processor is not None

        image_path = self.config.resolve_path(sample.image_path)
        if not image_path.exists() and not str(sample.image_path).startswith(
            ("http://", "https://")
        ):
            # Allow remote URLs as in the official example; otherwise require a file.
            raise FileNotFoundError(f"Image not found: {image_path}")

        image_ref: str | Path
        if str(sample.image_path).startswith(("http://", "https://")):
            image_ref = sample.image_path
        else:
            image_ref = image_path

        messages = build_qwen3_vl_messages(sample, image_path=image_ref)

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        # Match official docs / HF model_doc guidance.
        inputs.pop("token_type_ids", None)

        device = self.model.device
        inputs = inputs.to(device)

        gen_cfg = self.config.generation
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": gen_cfg.max_new_tokens,
            "do_sample": gen_cfg.do_sample,
            "temperature": gen_cfg.temperature,
            "top_p": gen_cfg.top_p,
            "top_k": gen_cfg.top_k,
            "repetition_penalty": gen_cfg.repetition_penalty,
        }
        # presence_penalty: supported in recent transformers GenerationConfig.
        # If the installed version rejects it, retry without it and record a note.
        generate_kwargs["presence_penalty"] = gen_cfg.presence_penalty

        started = time.perf_counter()
        presence_penalty_applied = True
        with self._torch.inference_mode():
            try:
                generated = self.model.generate(**inputs, **generate_kwargs)
            except TypeError as exc:
                if "presence_penalty" not in str(exc):
                    raise
                generate_kwargs.pop("presence_penalty", None)
                presence_penalty_applied = False
                generated = self.model.generate(**inputs, **generate_kwargs)
        latency = time.perf_counter() - started

        input_ids = inputs["input_ids"][0].tolist()
        full_ids = generated[0].tolist()
        output_token_ids = full_ids[len(input_ids) :]

        # Prefer token-ID based segmentation so think tags are not lost when
        # skip_special_tokens=True strips them from decoded text.
        def _decode(ids: list[int]) -> str:
            return self.processor.decode(
                ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

        parsed: ParsedOutput = decode_segments_from_token_ids(
            output_token_ids,
            _decode,
            think_end_token_id=self.config.thinking.think_end_token_id,
        )

        # Also keep a raw decode of the full generation for auditing.
        raw_output = _decode(output_token_ids)

        # If token-ID path missed the end marker but text still contains it
        # (e.g. unexpected decode), fall back to text split.
        if parsed.parse_status == "missing_think_end":
            text_parsed = split_thinking_and_answer(
                raw_output, think_end_text=self.config.thinking.think_end_text
            )
            if text_parsed.parse_status == "ok":
                parsed = text_parsed

        counts = count_reasoning_tokens(
            output_token_ids,
            think_end_token_id=self.config.thinking.think_end_token_id,
        )

        return InferenceResult(
            sample_id=sample.sample_id,
            model_name=self.config.model.model_id,
            model_revision=self.config.model.revision,
            dtype=self.config.model.dtype,
            image_path=str(image_ref),
            question=sample.question,
            generation_config={
                **gen_cfg.to_dict(),
                "presence_penalty_applied": presence_penalty_applied,
            },
            raw_output=raw_output,
            reasoning_text=parsed.reasoning_text,
            final_answer=parsed.final_answer,
            input_token_ids=input_ids,
            output_token_ids=output_token_ids,
            reasoning_token_count=counts.reasoning_token_count,
            answer_token_count=counts.answer_token_count,
            total_generated_token_count=counts.total_generated_token_count,
            latency_seconds=latency,
            parse_status=parsed.parse_status,
            backend="qwen3_vl",
            metadata={
                **sample.metadata,
                "think_end_index": counts.think_end_index,
            },
        )
