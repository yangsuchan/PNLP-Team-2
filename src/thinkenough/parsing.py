"""Parse Qwen3-VL Thinking model outputs into reasoning and final answer.

Confirmed format (Hugging Face tokenizer / chat template):
- Thinking models prepend the think-start marker in the *prompt*.
- Generated text typically looks like: ``reasoning...</think>\\n\\nanswer``
- Special token IDs: think_start=151667, think_end=151668

This module only implements the officially documented Qwen3 Thinking
delimiter. Do not assume other models share the same markers.
"""

from __future__ import annotations

from thinkenough.schemas import ParsedOutput, TokenCounts

# Official Qwen3-VL-8B-Thinking think-end text / token id.
DEFAULT_THINK_END_TEXT = "</think>"
DEFAULT_THINK_END_TOKEN_ID = 151668


def split_thinking_and_answer(
    text: str,
    *,
    think_end_text: str = DEFAULT_THINK_END_TEXT,
) -> ParsedOutput:
    """Split raw decoded generation into reasoning and final answer.

    Args:
        text: Decoded model generation (may omit the opening think tag).
        think_end_text: Closing delimiter from the tokenizer vocabulary.

    Returns:
        ParsedOutput with reasoning_text, final_answer, and parse_status:
        - ``ok``: think_end found; answer may still be empty.
        - ``missing_think_end``: no delimiter; entire text treated as reasoning.
        - ``empty_generation``: blank / whitespace-only generation.
    """
    if text is None or not str(text).strip():
        return ParsedOutput(
            reasoning_text="",
            final_answer="",
            parse_status="empty_generation",
        )

    raw = str(text)
    if think_end_text not in raw:
        return ParsedOutput(
            reasoning_text=raw.strip(),
            final_answer="",
            parse_status="missing_think_end",
        )

    reasoning, _, answer = raw.partition(think_end_text)
    return ParsedOutput(
        reasoning_text=reasoning.strip(),
        final_answer=answer.lstrip("\n").strip(),
        parse_status="ok",
    )


def count_reasoning_tokens(
    output_token_ids: list[int],
    *,
    think_end_token_id: int = DEFAULT_THINK_END_TOKEN_ID,
) -> TokenCounts:
    """Count reasoning / answer tokens using the think-end token ID.

    Token-ID based counting is preferred over string splitting so that
    future Hidden-State checkpoints can share the same index mapping.

    Args:
        output_token_ids: Newly generated token IDs (prompt excluded).
        think_end_token_id: Vocabulary id of the think-end marker.

    Returns:
        TokenCounts with reasoning/answer/total counts and think_end_index.
        If think_end is missing, all generated tokens are counted as reasoning.
    """
    total = len(output_token_ids)
    try:
        end_idx = output_token_ids.index(think_end_token_id)
    except ValueError:
        return TokenCounts(
            reasoning_token_count=total,
            answer_token_count=0,
            total_generated_token_count=total,
            think_end_index=-1,
        )

    # Tokens before think_end are reasoning; tokens after are the answer.
    # The think_end token itself is counted with reasoning (end of think span).
    reasoning_count = end_idx + 1
    answer_count = total - reasoning_count
    return TokenCounts(
        reasoning_token_count=reasoning_count,
        answer_token_count=answer_count,
        total_generated_token_count=total,
        think_end_index=end_idx,
    )


def decode_segments_from_token_ids(
    output_token_ids: list[int],
    decode_fn,
    *,
    think_end_token_id: int = DEFAULT_THINK_END_TOKEN_ID,
) -> ParsedOutput:
    """Decode reasoning / answer separately from token IDs.

    Prefer this path when the tokenizer may strip think tags under
    ``skip_special_tokens=True``.

    Args:
        output_token_ids: Newly generated token IDs.
        decode_fn: Callable ``list[int] -> str`` (e.g. processor.decode).
        think_end_token_id: Vocabulary id of the think-end marker.

    Returns:
        ParsedOutput built from decoded reasoning and answer segments.
    """
    if not output_token_ids:
        return ParsedOutput(
            reasoning_text="",
            final_answer="",
            parse_status="empty_generation",
        )

    counts = count_reasoning_tokens(
        output_token_ids, think_end_token_id=think_end_token_id
    )
    if counts.think_end_index < 0:
        return ParsedOutput(
            reasoning_text=decode_fn(output_token_ids).strip(),
            final_answer="",
            parse_status="missing_think_end",
        )

    # Exclude the think_end token from the reasoning *text* when possible.
    reasoning_ids = output_token_ids[: counts.think_end_index]
    answer_ids = output_token_ids[counts.think_end_index + 1 :]
    return ParsedOutput(
        reasoning_text=decode_fn(reasoning_ids).strip() if reasoning_ids else "",
        final_answer=decode_fn(answer_ids).strip() if answer_ids else "",
        parse_status="ok",
    )
