"""Reasoning checkpoints and stopping rules evaluated on recorded reasoning.

Checkpoints sit every fixed number of tokens, as in Answer-Readiness-Guided Early
Stopping (arXiv 2609.34809). A position such as "40% of the reasoning" cannot be used
while generating, because the total length is unknown until the reasoning ends.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence


def interval_checkpoints(total_tokens: int, interval: int = 256) -> list[int]:
    """Prefix lengths to check, strictly before the reasoning ends on its own."""
    return list(range(interval, total_tokens, interval))


@dataclass
class Checkpoint:
    """One forced answer during a question's reasoning."""

    tokens: int
    correct: bool
    missing: bool = False
    row: dict[str, Any] = field(default_factory=dict)


@dataclass
class QuestionTrace:
    """What full thinking did on a question, plus its forced answers in token order."""

    sample_id: str
    scene_id: str
    full_tokens: int
    full_correct: bool
    full_missing: bool
    checkpoints: list[Checkpoint] = field(default_factory=list)


@dataclass
class Outcome:
    """Result of a stopping rule on one question."""

    tokens: int
    correct: bool
    missing: bool
    stopped_at: int | None  # Checkpoint index, or None for running to the end.


def run_rule(trace: QuestionTrace, stop_index: int | None) -> Outcome:
    """Outcome of stopping at a checkpoint, or of full thinking when ``stop_index`` is None."""
    if stop_index is None:
        return Outcome(trace.full_tokens, trace.full_correct, trace.full_missing, None)
    checkpoint = trace.checkpoints[stop_index]
    return Outcome(checkpoint.tokens, checkpoint.correct, checkpoint.missing, stop_index)


def fixed_cap(trace: QuestionTrace, cap: int) -> Outcome:
    """Baseline: stop at the last checkpoint within a token cap, whatever the state."""
    index = None
    for position, checkpoint in enumerate(trace.checkpoints):
        if checkpoint.tokens == cap:
            index = position
    return run_rule(trace, index)


def oracle(trace: QuestionTrace) -> Outcome:
    """Upper bound: stop at the first correct checkpoint, which needs the true answer."""
    for position, checkpoint in enumerate(trace.checkpoints):
        if checkpoint.correct:
            return run_rule(trace, position)
    return run_rule(trace, None)


def bootstrap_mean_ci(
    values: Sequence[float], *, samples: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """95% percentile bootstrap interval of a mean, resampling questions."""
    rng = random.Random(seed)
    count = len(values)
    means = sorted(sum(rng.choices(values, k=count)) / count for _ in range(samples))
    return means[int(0.025 * samples)], means[int(0.975 * samples) - 1]


def summarize(traces: Sequence[QuestionTrace], rule: Callable[[QuestionTrace], Outcome]) -> dict[str, Any]:
    """Compare a rule with full thinking on the same questions.

    Accuracy differences are paired by question, so the interval reflects the change
    itself rather than two separate accuracies.
    """
    outcomes = [rule(trace) for trace in traces]
    correct = [float(o.correct) for o in outcomes]
    full = [float(t.full_correct) for t in traces]
    tokens = sum(o.tokens for o in outcomes)
    full_tokens = sum(t.full_tokens for t in traces)
    change_low, change_high = bootstrap_mean_ci([c - f for c, f in zip(correct, full)])
    return {
        "n": len(traces),
        "accuracy": sum(correct) / len(correct),
        "accuracy_full": sum(full) / len(full),
        "accuracy_change_ci": [change_low, change_high],
        "net_correct": int(sum(correct) - sum(full)),
        "reasoning_tokens_mean": tokens / len(traces),
        "reasoning_tokens_full_mean": full_tokens / len(traces),
        "token_reduction": 1 - tokens / full_tokens if full_tokens else 0.0,
        "missing_answers": sum(o.missing for o in outcomes),
        "missing_answers_full": sum(t.full_missing for t in traces),
        "stopped_at_first_check": sum(o.stopped_at == 0 for o in outcomes) / len(outcomes),
        "stopped_early": sum(o.stopped_at is not None for o in outcomes) / len(outcomes),
    }
