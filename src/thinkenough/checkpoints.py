"""Reasoning checkpoints and the analysis of answers forced at those checkpoints."""

from __future__ import annotations

import random
from typing import Any, Iterable, Sequence

SENTENCE_END_MARKS = (".", "!", "?", "\n")


def sentence_ends(token_texts: Sequence[str]) -> list[int]:
    """Token counts at which a sentence has just ended (ReProbe's signal concentrates there)."""
    return [
        index + 1
        for index, text in enumerate(token_texts)
        if text.rstrip(" ").endswith(SENTENCE_END_MARKS)
    ]


def pick_checkpoints(
    token_texts: Sequence[str], fractions: Iterable[float] = (0.2, 0.4, 0.6, 0.8, 1.0)
) -> list[tuple[float, int]]:
    """Choose prefix lengths near each fraction of the reasoning, at sentence ends.

    Args:
        token_texts: Decoded reasoning tokens, without the closing think tag.
        fractions: Target positions, where 1.0 is the complete reasoning.

    Returns:
        (fraction, prefix length) pairs, unique by prefix length, ascending.
    """
    total = len(token_texts)
    if total == 0:
        return []
    ends = sentence_ends(token_texts) or [total]
    chosen: list[tuple[float, int]] = []
    for fraction in sorted(fractions):
        target = total * fraction
        # The complete reasoning stays exact even when it does not end a sentence.
        length = total if fraction >= 1.0 else min(ends, key=lambda end: abs(end - target))
        if length not in (kept for _, kept in chosen):
            chosen.append((fraction, length))
    return chosen


def bootstrap_mean_ci(
    values: Sequence[float], *, samples: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """95% percentile bootstrap interval of a mean, resampling questions."""
    rng = random.Random(seed)
    count = len(values)
    means = sorted(sum(rng.choices(values, k=count)) / count for _ in range(samples))
    return means[int(0.025 * samples)], means[int(0.975 * samples) - 1]


def summarize_curve(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Accuracy by reasoning position, plus the best possible saving (the oracle).

    Stopping at one fixed position is the fixed-stop baseline, so each row of the
    curve is that baseline. The oracle stops at each question's first correct
    checkpoint, an upper bound that no real stopping rule can beat.

    Args:
        rows: One row per checkpoint with ``sample_id``, ``fraction``,
            ``prefix_tokens``, ``reasoning_tokens_total``, and boolean ``correct``.

    Returns:
        ``curve`` list, ``oracle`` summary, and question count ``n``.
    """
    by_sample: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_sample.setdefault(row["sample_id"], []).append(row)
    for group in by_sample.values():
        group.sort(key=lambda row: row["prefix_tokens"])

    curve = []
    for fraction in sorted({row["fraction"] for row in rows}):
        group = [row for row in rows if row["fraction"] == fraction]
        correct = [float(row["correct"]) for row in group]
        low, high = bootstrap_mean_ci(correct)
        curve.append(
            {
                "fraction": fraction,
                "n": len(group),
                "accuracy": sum(correct) / len(correct),
                "ci_low": low,
                "ci_high": high,
                "tokens_kept": sum(r["prefix_tokens"] for r in group)
                / sum(r["reasoning_tokens_total"] for r in group),
            }
        )

    kept = spent = 0
    answered = 0
    for group in by_sample.values():
        total = group[-1]["reasoning_tokens_total"]
        first = next((row for row in group if row["correct"]), None)
        spent += total
        # A question never answered correctly keeps its full reasoning.
        kept += first["prefix_tokens"] if first else total
        answered += first is not None
    oracle = {
        "accuracy": answered / len(by_sample),
        "tokens_kept": kept / spent if spent else 0.0,
    }
    return {"n": len(by_sample), "curve": curve, "oracle": oracle}
