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
    score: float | None = None  # For tasks graded by score rather than right or wrong.


@dataclass
class QuestionTrace:
    """What full thinking did on a question, plus its forced answers in token order."""

    sample_id: str
    scene_id: str
    full_tokens: int
    full_correct: bool
    full_missing: bool
    checkpoints: list[Checkpoint] = field(default_factory=list)
    full_score: float | None = None
    full_answer_tokens: int | None = None


@dataclass
class Outcome:
    """Result of a stopping rule on one question."""

    tokens: int
    correct: bool
    missing: bool
    stopped_at: int | None  # Checkpoint index, or None for running to the end.
    score: float | None = None


def run_rule(trace: QuestionTrace, stop_index: int | None) -> Outcome:
    """Outcome of stopping at a checkpoint, or of full thinking when ``stop_index`` is None."""
    if stop_index is None:
        return Outcome(trace.full_tokens, trace.full_correct, trace.full_missing, None, trace.full_score)
    checkpoint = trace.checkpoints[stop_index]
    return Outcome(checkpoint.tokens, checkpoint.correct, checkpoint.missing, stop_index, checkpoint.score)


def fixed_cap(trace: QuestionTrace, cap: int) -> Outcome:
    """Baseline: stop at the last checkpoint within a token cap, whatever the state."""
    index = None
    for position, checkpoint in enumerate(trace.checkpoints):
        if checkpoint.tokens == cap:
            index = position
    return run_rule(trace, index)


def oracle_give_up(trace: QuestionTrace) -> Outcome:
    """Bound on savings: also stop questions that are never right at their first checkpoint."""
    outcome = oracle(trace)
    if not outcome.correct and trace.checkpoints:
        return run_rule(trace, 0)
    return outcome


def oracle(trace: QuestionTrace) -> Outcome:
    """Bound on accuracy: stop at the first correct checkpoint, which needs the true answer.

    Questions that are never right run to the end, so this does not bound token savings.
    """
    for position, checkpoint in enumerate(trace.checkpoints):
        if checkpoint.correct:
            return run_rule(trace, position)
    return run_rule(trace, None)


def answer_stable(trace: QuestionTrace, keys: Sequence[Any], repeats: int) -> Outcome:
    """Baseline without a Probe: stop once the forced answer has stayed the same ``repeats`` times.

    ``keys`` holds each checkpoint's answer in comparable form, or None if it has none.
    """
    for position in range(repeats - 1, len(keys)):
        window = keys[position - repeats + 1 : position + 1]
        if window[0] is not None and len(set(window)) == 1:
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
    scored = {}
    if traces and traces[0].full_score is not None:
        score_changes = [o.score - t.full_score for o, t in zip(outcomes, traces)]
        scored = {
            "score_mean": sum(o.score for o in outcomes) / len(outcomes),
            "score_full_mean": sum(t.full_score for t in traces) / len(traces),
            "score_change_ci": list(bootstrap_mean_ci(score_changes)),
        }
    if traces and all(t.full_answer_tokens is not None for t in traces):
        # Everything generated, so a stopped question also pays for the answer it ends with.
        total = sum(
            o.tokens + (t.checkpoints[o.stopped_at].row["answer_tokens"] if o.stopped_at is not None else t.full_answer_tokens)
            for o, t in zip(outcomes, traces)
        )
        full_total = sum(t.full_tokens + t.full_answer_tokens for t in traces)
        scored["total_tokens_mean"] = total / len(traces)
        scored["total_tokens_full_mean"] = full_total / len(traces)
        scored["total_reduction"] = 1 - total / full_total if full_total else 0.0
    return {
        **scored,
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
