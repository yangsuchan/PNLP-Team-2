"""Answer-readiness detector, following arXiv 2609.34809 (Eq. 6-7).

A linear detector reads [h, d, t/B] at each checkpoint: the hidden state, its change
since the previous checkpoint, and the share of the token budget used. Its label is
whether the answer forced at that checkpoint is correct. Questions are split by scene.
"""

from __future__ import annotations

import hashlib
from typing import Any, Callable, Sequence

import numpy as np

from thinkenough.checkpoints import Outcome, QuestionTrace, run_rule


def fold_of(scene_id: str, folds: int = 2, salt: str = "fold") -> int:
    """Stable fold for a scene, so one scene never crosses a train and test split."""
    return int(hashlib.sha1(f"{salt}:{scene_id}".encode()).hexdigest(), 16) % folds


def checkpoint_features(
    hidden: Callable[[QuestionTrace, int | None], np.ndarray],
    trace: QuestionTrace,
    budget: int,
) -> np.ndarray:
    """One row of [h, d, t/B] per checkpoint.

    Args:
        hidden: Returns the hidden state at a checkpoint index, or at the start of
            reasoning for None.
        trace: The question.
        budget: Token budget B.
    """
    rows = []
    previous = hidden(trace, None)
    for index, checkpoint in enumerate(trace.checkpoints):
        current = hidden(trace, index)
        rows.append(np.concatenate([current, current - previous, [checkpoint.tokens / budget]]))
        previous = current
    return np.array(rows) if rows else np.zeros((0, 0))


class ReadinessDetector:
    """Standardized logistic regression with class-weighted loss.

    Standardization and class weights come only from the training checkpoints.
    """

    def __init__(self, seed: int = 0) -> None:
        from sklearn.linear_model import LogisticRegression

        self.model = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "ReadinessDetector":
        self.mean = features.mean(0)
        self.scale = features.std(0) + 1e-8
        self.model.fit((features - self.mean) / self.scale, labels)
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        if len(features) == 0:
            return np.zeros(0)
        return self.model.predict_proba((features - self.mean) / self.scale)[:, 1]


def stop_at_threshold(trace: QuestionTrace, scores: Sequence[float], threshold: float) -> Outcome:
    """Stop at the first checkpoint whose score reaches the threshold."""
    for index, score in enumerate(scores):
        if score >= threshold:
            return run_rule(trace, index)
    return run_rule(trace, None)


def choose_threshold(
    traces: Sequence[QuestionTrace],
    scores: Sequence[Sequence[float]],
    *,
    allowed_accuracy_drop: float = 0.005,
) -> float:
    """Threshold with the fewest mean tokens within the allowed accuracy drop from full thinking."""
    full_accuracy = sum(t.full_correct for t in traces) / len(traces)
    candidates = sorted({float(s) for group in scores for s in group}) + [float("inf")]
    best_threshold, best_tokens = float("inf"), None
    for threshold in candidates:
        outcomes = [stop_at_threshold(t, s, threshold) for t, s in zip(traces, scores)]
        accuracy = sum(o.correct for o in outcomes) / len(outcomes)
        tokens = sum(o.tokens for o in outcomes)
        if accuracy >= full_accuracy - allowed_accuracy_drop and (best_tokens is None or tokens < best_tokens):
            best_threshold, best_tokens = threshold, tokens
    return best_threshold


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Chance that a correct checkpoint outscores an incorrect one. 0.5 when undefined."""
    positive, negative = scores[labels == 1], scores[labels == 0]
    if len(positive) == 0 or len(negative) == 0:
        return 0.5
    order = np.argsort(np.concatenate([positive, negative]), kind="stable")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[: len(positive)].sum() - len(positive) * (len(positive) + 1) / 2) / (len(positive) * len(negative)))


def labels_of(traces: Sequence[QuestionTrace]) -> np.ndarray:
    return np.array([float(c.correct) for t in traces for c in t.checkpoints])


def two_fold_evaluate(
    traces: Sequence[QuestionTrace],
    hidden_at_layer: Callable[[int], Callable[[QuestionTrace, int | None], np.ndarray]],
    layers: Sequence[int],
    budget: int,
) -> list[dict[str, Any]]:
    """Train on one scene fold and test on the other, in both directions.

    Within the training fold, a scene-separated validation part picks the layer and the
    threshold. Returns the per-question outcomes and choices of each direction.
    """
    results = []
    for test_fold in (0, 1):
        test = [t for t in traces if fold_of(t.scene_id) == test_fold]
        rest = [t for t in traces if fold_of(t.scene_id) != test_fold]
        validation = [t for t in rest if fold_of(t.scene_id, 5, "validation") == 0]
        train = [t for t in rest if fold_of(t.scene_id, 5, "validation") != 0]
        if not test or not train or not validation:
            raise ValueError("Each fold needs scenes for training, validation, and testing")

        def stacked(group: Sequence[QuestionTrace], hidden: Callable) -> np.ndarray:
            parts = [f for f in (checkpoint_features(hidden, t, budget) for t in group) if len(f)]
            return np.concatenate(parts) if parts else np.zeros((0, 0))

        y_train, y_validation = labels_of(train), labels_of(validation)
        if len(set(y_train.tolist())) < 2:
            raise ValueError("Training checkpoints need both correct and incorrect answers")

        best_layer, best_auroc, best_detector = None, -1.0, None
        for layer in layers:
            hidden = hidden_at_layer(layer)
            detector = ReadinessDetector().fit(stacked(train, hidden), y_train)
            value = auroc(y_validation, detector.score(stacked(validation, hidden)))
            if value > best_auroc:
                best_layer, best_auroc, best_detector = layer, value, detector

        hidden = hidden_at_layer(best_layer)
        scored = lambda group: [best_detector.score(checkpoint_features(hidden, t, budget)) for t in group]
        threshold = choose_threshold(validation, scored(validation))
        test_scores = scored(test)
        results.append(
            {
                "test_fold": test_fold,
                "layer": best_layer,
                "validation_auroc": best_auroc,
                "threshold": threshold,
                "test_traces": test,
                "test_scores": test_scores,
                "test_auroc": auroc(
                    labels_of(test), np.concatenate(test_scores) if test_scores else np.zeros(0)
                ),
            }
        )
    return results
