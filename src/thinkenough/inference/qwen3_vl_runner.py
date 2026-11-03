"""Runner for Qwen vision-language models with thinking (Hugging Face Transformers).

The experiments use Qwen/Qwen3.5-9B. The backend keeps the name qwen3_vl, which also appears in
result file names.

WARNING:
- Do not run this on the local 8GB VRAM machine for full BF16 inference.
- Model download and GPU inference require explicit user approval.
- This module keeps the base model frozen (eval + inference_mode only).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterator

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


class PresencePenalty:
    """Lower the score of every token already generated, as vLLM's presence_penalty does.

    The model card recommends this for thinking mode, but transformers has no such option.
    Prompt tokens do not count. It runs before temperature and top-k, also as in vLLM.
    """

    def __init__(self, penalty: float, prompt_length: int) -> None:
        self.penalty = penalty
        self.prompt_length = prompt_length

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        seen = input_ids[:, self.prompt_length :]
        if seen.shape[1] == 0:
            return scores
        return scores.scatter(1, seen, scores.gather(1, seen) - self.penalty)


class Qwen3VLRunner:
    """Frozen thinking-model runner: Full CoT, forced answers, and hidden states."""

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
            from transformers import AutoModelForImageTextToText, AutoProcessor
            from transformers import set_seed
        except ImportError as exc:
            raise ImportError(
                "Qwen3VLRunner requires torch and transformers. "
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
        # The auto class also loads Qwen3.5, which the model comparison needs.
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.config.model.model_id,
            **model_kwargs,
        )
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        self._torch = torch
        self._loaded = True

    def _think_end_token_id(self) -> int:
        """The think-end token id differs between Qwen generations, so ask the tokenizer."""
        return self.processor.tokenizer.convert_tokens_to_ids(self.config.thinking.think_end_text)

    def _prepare_inputs(
        self, sample: InferenceSample, *, enable_thinking: bool = True
    ) -> tuple[Any, str | Path | None]:
        """Tokenize one sample. An empty ``image_path`` means text-only input."""
        assert self.model is not None and self.processor is not None

        image_ref: str | Path | None = None
        if sample.image_path:
            if str(sample.image_path).startswith(("http://", "https://")):
                # Allow remote URLs as in the official example.
                image_ref = sample.image_path
            else:
                image_ref = self.config.resolve_path(sample.image_path)
                if not image_ref.exists():
                    raise FileNotFoundError(f"Image not found: {image_ref}")

        video_frames = None
        processor_options: dict[str, Any] = {}
        if sample.video_path:
            folder = self.config.resolve_path(sample.video_path)
            video_frames = sorted(str(p) for p in folder.glob("*.jpg"))
            if not video_frames:
                raise FileNotFoundError(f"No video frames in: {folder}")
            # Frames are already sampled; the processor must keep all of them.
            self.processor.video_processor.do_sample_frames = False
            # Without real timing the processor writes wrong frame times into the prompt.
            timing = json.loads((folder / "frames.json").read_text(encoding="utf-8"))
            if len(timing["frames_indices"]) != len(video_frames):
                raise ValueError(f"Frame count does not match frames.json in: {folder}")
            processor_options = {"processor_kwargs": {"video_metadata": [timing]}}

        extra_images = [str(self.config.resolve_path(path)) for path in sample.extra_image_paths]
        missing = [path for path in extra_images if not Path(path).exists()]
        if missing:
            raise FileNotFoundError(f"Image not found: {missing[0]}")

        messages = build_qwen3_vl_messages(
            sample, image_path=image_ref or "", video_frames=video_frames, extra_images=extra_images
        )
        template_options = {} if enable_thinking else {"enable_thinking": False}
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            **template_options,
            **processor_options,
        )
        # Match official docs / HF model_doc guidance.
        inputs.pop("token_type_ids", None)
        if not enable_thinking and self._think_end_token_id() not in inputs["input_ids"][0][-4:].tolist():
            # Some templates ignore this option and would think anyway.
            raise ValueError(
                f"{self.config.model.model_id} has no non-thinking mode. "
                "Use its Instruct model instead."
            )
        return inputs.to(self.model.device), image_ref

    def _generate(self, inputs: Any) -> tuple[list[int], list[int], float, dict[str, Any]]:
        """Run generation; return input ids, new ids, latency, and run metadata."""
        gen_cfg = self.config.generation
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": gen_cfg.max_new_tokens,
            "do_sample": gen_cfg.do_sample,
            "temperature": gen_cfg.temperature,
            "top_p": gen_cfg.top_p,
            "top_k": gen_cfg.top_k,
            "repetition_penalty": gen_cfg.repetition_penalty,
            # The repository has no generation_config.json, so generate would stop only at the second.
            "eos_token_id": self.processor.tokenizer.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"]),
        }
        if gen_cfg.num_beams > 1:
            generate_kwargs["num_beams"] = gen_cfg.num_beams
        if gen_cfg.presence_penalty:
            from transformers import LogitsProcessorList

            generate_kwargs["logits_processor"] = LogitsProcessorList(
                [PresencePenalty(gen_cfg.presence_penalty, inputs["input_ids"].shape[1])]
            )

        has_cuda = self._torch.cuda.is_available()
        if has_cuda:
            self._torch.cuda.reset_peak_memory_stats()

        started = time.perf_counter()
        with self._torch.inference_mode():
            generated = self.model.generate(**inputs, **generate_kwargs)
        latency = time.perf_counter() - started

        input_ids = inputs["input_ids"][0].tolist()
        output_token_ids = generated[0].tolist()[len(input_ids) :]
        run_meta = {
            "presence_penalty_applied": bool(gen_cfg.presence_penalty),
            "peak_vram_gb": (
                self._torch.cuda.max_memory_allocated() / 1e9 if has_cuda else None
            ),
        }
        return input_ids, output_token_ids, latency, run_meta

    def _append_tokens(self, inputs: Any, token_ids: list[int]) -> None:
        """Extend tokenized inputs in place with plain text tokens."""
        torch = self._torch
        extra = torch.tensor(
            [token_ids], dtype=inputs["input_ids"].dtype, device=self.model.device
        )
        for key, values in (
            ("input_ids", extra),
            ("attention_mask", torch.ones_like(extra)),
            ("mm_token_type_ids", torch.zeros_like(extra)),
        ):
            if key in inputs:
                inputs[key] = torch.cat([inputs[key], values], dim=1)

    def extract_hidden_state(
        self, sample: InferenceSample, prefix_token_ids: list[int]
    ) -> Any:
        """Hidden states of every text layer at the last token of a reasoning prefix.

        The input stops at the prefix, so no later reasoning or answer can leak in.
        Hooks keep only the last token of each layer, because holding whole layers
        could exceed GPU memory on inputs of about 10,000 tokens.

        Args:
            sample: The sample whose reasoning is being read.
            prefix_token_ids: Generated reasoning tokens up to the checkpoint.

        Returns:
            float32 CPU tensor of shape [layers + 1, hidden size]. The last row is the
            normalized final output, the state AGENTS.md 5.2 names as the default.
        """
        self.load()
        inputs, _ = self._prepare_inputs(sample)
        if prefix_token_ids:
            self._append_tokens(inputs, prefix_token_ids)

        states = []

        def keep_last_token(_module: Any, _args: Any, output: Any) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            states.append(tensor[0, -1].float().cpu())

        layers = self.model.model.language_model.layers
        handles = [layer.register_forward_hook(keep_last_token) for layer in layers]
        try:
            with self._torch.inference_mode():
                outputs = self.model.model(**inputs, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        states.append(outputs.last_hidden_state[0, -1].float().cpu())
        return self._torch.stack(states)

    def forced_answer(
        self,
        sample: InferenceSample,
        reasoning_prefix: str = "",
        prefix_token_ids: list[int] | None = None,
    ) -> InferenceResult:
        """Answer immediately from a reasoning prefix (AGENTS.md 5.3).

        The chat template already ends with ``<think>\\n``, so closing the tag
        here makes the model answer from this prefix only. An empty prefix is
        the direct-answer baseline.

        Args:
            sample: Multimodal or text-only sample.
            reasoning_prefix: Reasoning text up to the checkpoint. Must not
                contain later reasoning or the original final answer.
            prefix_token_ids: The same prefix as token ids. Takes priority, so a
                checkpoint maps exactly onto the original generation.

        Returns:
            InferenceResult whose ``final_answer`` is the forced answer.
        """
        self.load()
        inputs, image_ref = self._prepare_inputs(sample)
        tokenizer = self.processor.tokenizer
        if prefix_token_ids is None:
            prefix_token_ids = (
                tokenizer(reasoning_prefix, add_special_tokens=False).input_ids
                if reasoning_prefix
                else []
            )
        else:
            reasoning_prefix = self.processor.decode(prefix_token_ids)
        thinking = self.config.thinking
        closing = tokenizer(
            f"\n{thinking.think_end_text}\n\n{thinking.answer_cue}", add_special_tokens=False
        ).input_ids
        self._append_tokens(inputs, list(prefix_token_ids) + closing)

        input_ids, output_token_ids, latency, run_meta = self._generate(inputs)
        return self._forced_result(
            sample, image_ref, reasoning_prefix, len(prefix_token_ids), input_ids, output_token_ids, latency, run_meta
        )

    def _forced_result(
        self,
        sample: InferenceSample,
        image_ref: Any,
        reasoning_prefix: str,
        prefix_length: int,
        input_ids: list[int],
        output_token_ids: list[int],
        latency: float,
        run_meta: dict[str, Any],
    ) -> InferenceResult:
        answer = self.processor.decode(
            output_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

        return InferenceResult(
            sample_id=sample.sample_id,
            model_name=self.config.model.model_id,
            model_revision=self.config.model.revision,
            dtype=self.config.model.dtype,
            image_path=str(image_ref or ""),
            question=sample.question,
            generation_config={
                **self.config.generation.to_dict(),
                "presence_penalty_applied": run_meta["presence_penalty_applied"],
            },
            raw_output=answer,
            reasoning_text=reasoning_prefix,
            final_answer=answer,
            input_token_ids=input_ids,
            output_token_ids=output_token_ids,
            reasoning_token_count=prefix_length,
            answer_token_count=len(output_token_ids),
            total_generated_token_count=len(output_token_ids),
            latency_seconds=latency,
            parse_status="ok" if answer else "empty_generation",
            backend="qwen3_vl",
            metadata={
                **sample.metadata,
                "forced_answer": True,
                "peak_vram_gb": run_meta["peak_vram_gb"],
            },
        )

    def checkpoints(
        self,
        sample: InferenceSample,
        token_ids: list[int],
        lengths: list[int],
        done: frozenset[int] = frozenset(),
        chunk_size: int = 512,
    ) -> Iterator[tuple[int, int, Any, InferenceResult]]:
        """Walk one reasoning trace once, with hidden states and a forced answer at each length.

        Reading the whole prompt and prefix at once for every checkpoint runs out of memory on
        long traces, so the prompt is read once and the reasoning is fed in small chunks through
        the cache. After each forced answer the cache is rolled back: full-attention layers are
        cropped, and linear-attention layers get their saved states back.

        Args:
            sample: The sample whose reasoning is being read.
            token_ids: Generated reasoning tokens.
            lengths: Increasing prefix lengths to stop at.
            done: Indices of ``lengths`` that are already saved, so they are only passed through.
            chunk_size: Tokens per forward pass.

        Yields:
            Index into ``lengths``, the length, [layers + 1, hidden size] float32 CPU states as in
            ``extract_hidden_state``, and the forced answer.
        """
        self.load()
        torch = self._torch
        inputs, image_ref = self._prepare_inputs(sample)
        thinking = self.config.thinking
        closing = self.processor.tokenizer(
            f"\n{thinking.think_end_text}\n\n{thinking.answer_cue}", add_special_tokens=False
        ).input_ids
        device = self.model.device
        ids, mask = inputs["input_ids"], inputs["attention_mask"]
        states: list[Any] = []

        def keep_last_token(_module: Any, _args: Any, output: Any) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            states.append(tensor[0, -1].float().cpu())

        def feed(new_ids: list[int], cache: Any) -> tuple[Any, Any, Any]:
            extra = torch.tensor([new_ids], dtype=ids.dtype, device=device)
            longer_ids = torch.cat([current["ids"], extra], dim=1)
            longer_mask = torch.cat([current["mask"], torch.ones_like(extra)], dim=1)
            # No mask here: with one the model builds positions for the whole sequence, not this chunk.
            output = self.model.model(input_ids=extra, past_key_values=cache, use_cache=True)
            return output, longer_ids, longer_mask

        layers = self.model.model.language_model.layers
        with torch.inference_mode():
            cache = self.model.model(**inputs, use_cache=True).past_key_values
            current = {"ids": ids, "mask": mask}
            position = 0
            for index, length in enumerate(lengths):
                while position < length:
                    step = min(length - position, chunk_size)
                    last = position + step == length
                    handles = [layer.register_forward_hook(keep_last_token) for layer in layers] if last else []
                    states.clear()
                    try:
                        output, current["ids"], current["mask"] = feed(token_ids[position : position + step], cache)
                    finally:
                        for handle in handles:
                            handle.remove()
                    position += step
                if index in done:
                    continue
                hidden = torch.stack([*states, output.last_hidden_state[0, -1].float().cpu()])

                saved = [
                    (layer, layer.conv_states.clone(), layer.recurrent_states.clone())
                    for layer in cache.layers
                    if getattr(layer, "is_conv_states_initialized", False)
                ]
                kept_length = cache.get_seq_length()
                walked = dict(current)
                if len(closing) > 1:  # generate needs one token it has not seen.
                    _, current["ids"], current["mask"] = feed(closing[:-1], cache)
                extra = torch.tensor([closing[-1:]], dtype=ids.dtype, device=device)
                forced_inputs = {
                    "input_ids": torch.cat([current["ids"], extra], dim=1),
                    "attention_mask": torch.cat([current["mask"], torch.ones_like(extra)], dim=1),
                    "past_key_values": cache,
                }
                input_ids, output_ids, latency, run_meta = self._generate(forced_inputs)
                result = self._forced_result(
                    sample, image_ref, self.processor.decode(token_ids[:length]), length, input_ids, output_ids, latency, run_meta
                )
                cache.crop(kept_length)
                for layer, conv_state, recurrent_state in saved:
                    layer.conv_states.copy_(conv_state)
                    layer.recurrent_states.copy_(recurrent_state)
                current.update(walked)
                yield index, length, hidden, result

    def generate_cot(
        self, sample: InferenceSample, *, enable_thinking: bool = True
    ) -> InferenceResult:
        """Run Full CoT generation for one multimodal or text-only sample.

        Args:
            sample: Question sample, with an image path or an empty path.
            enable_thinking: False uses the model's official non-thinking mode, in
                which it can still explain itself in the visible answer.

        Returns:
            InferenceResult with parsed reasoning/answer and token metadata.
        """
        self.load()
        inputs, image_ref = self._prepare_inputs(sample, enable_thinking=enable_thinking)
        input_ids, output_token_ids, latency, run_meta = self._generate(inputs)

        # Prefer token-ID based segmentation so think tags are not lost when
        # skip_special_tokens=True strips them from decoded text.
        def _decode(ids: list[int]) -> str:
            return self.processor.decode(
                ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

        think_end_token_id = self._think_end_token_id()
        parsed: ParsedOutput = decode_segments_from_token_ids(
            output_token_ids,
            _decode,
            think_end_token_id=think_end_token_id,
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
            think_end_token_id=think_end_token_id,
        )

        return InferenceResult(
            sample_id=sample.sample_id,
            model_name=self.config.model.model_id,
            model_revision=self.config.model.revision,
            dtype=self.config.model.dtype,
            image_path=str(image_ref or ""),
            question=sample.question,
            generation_config={
                **self.config.generation.to_dict(),
                "presence_penalty_applied": run_meta["presence_penalty_applied"],
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
                "peak_vram_gb": run_meta["peak_vram_gb"],
            },
        )
