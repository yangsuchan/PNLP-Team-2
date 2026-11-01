"""Metric functions for ThinkEnough evaluation.

요약: [4단계 평가] 정확도, 토큰 감소율, Latency, AUROC/F1 등 지표 계산 함수. 값마다 측정값인지 가짜인지 status를 붙인다.

Every metric returns a dict with a `status` so an undefined or unmeasured value
is never printed as if it were real:
  ok / measured      computed from measured inputs
  estimated          some inputs are estimates (e.g. re-tokenized token counts)
  synthetic          inputs come from synthetic test data (not a performance result)
  undefined          mathematically undefined (e.g. division by zero, one class only)
  not_computable     required inputs are missing
"""

import math

# provenance of a numeric input -> metric status; worst one wins
_SOURCE_STATUS = {"generated_ids": "measured", "measured": "measured",
                  "retokenized": "estimated", "estimated": "estimated", "synthetic": "synthetic"}
_STATUS_RANK = {"measured": 0, "estimated": 1, "synthetic": 2, "not_computable": 3}


def undefined(reason):
    return {"value": None, "status": "undefined", "reason": reason}


def not_computable(reason):
    return {"value": None, "status": "not_computable", "reason": reason}


def combined_status(sources):
    """Worst provenance among inputs; unknown or None source -> not_computable."""
    statuses = [_SOURCE_STATUS.get(s, "not_computable") for s in sources]
    return max(statuses, key=_STATUS_RANK.__getitem__) if statuses else "not_computable"


# ---------------------------------------------------------------- task accuracy


def accuracy(correct_flags):
    """Accuracy (%) = correct / total * 100 over a fixed question set."""
    total = len(correct_flags)
    if total == 0:
        return undefined("no questions")
    correct = sum(bool(x) for x in correct_flags)
    return {"value": 100.0 * correct / total, "status": "ok", "correct": correct, "total": total}


def performance_retention(early_exit_acc, full_acc):
    """Retention (%) = early-exit accuracy / full-CoT accuracy * 100."""
    if early_exit_acc is None or full_acc is None:
        return not_computable("accuracy missing")
    if full_acc == 0:
        return undefined("full CoT accuracy is 0")
    return {"value": 100.0 * early_exit_acc / full_acc, "status": "ok", "unit": "% of full CoT accuracy"}


def accuracy_difference(early_exit_acc, full_acc):
    """Difference in percentage points = early-exit accuracy - full-CoT accuracy."""
    if early_exit_acc is None or full_acc is None:
        return not_computable("accuracy missing")
    return {"value": early_exit_acc - full_acc, "status": "ok", "unit": "percentage points"}


# ---------------------------------------------------------------- tokens / latency


def mean_of(values, sources, what):
    """Mean of per-question values; any missing value makes it not computable."""
    if not values:
        return not_computable(f"no {what} values")
    missing = sum(v is None for v in values)
    if missing:
        return not_computable(f"{missing}/{len(values)} {what} values not recorded")
    status = combined_status(sources)
    if status == "not_computable":
        return not_computable(f"{what} source unknown")
    return {"value": sum(values) / len(values), "total": sum(values), "n": len(values), "status": status}


def token_reduction(full_mean, early_exit_mean):
    """Reduction (%) = (full - early exit) / full * 100; negative means more tokens were used."""
    for m in (full_mean, early_exit_mean):
        if m["status"] in ("not_computable", "undefined"):
            return not_computable("token counts not available for both methods")
    if full_mean["value"] == 0:
        return undefined("full CoT reasoning tokens are 0")
    status = max(full_mean["status"], early_exit_mean["status"], key=_STATUS_RANK.__getitem__)
    value = 100.0 * (full_mean["value"] - early_exit_mean["value"]) / full_mean["value"]
    out = {"value": value, "status": status, "unit": "% of full CoT mean reasoning tokens"}
    if value < 0:
        out["note"] = "early exit used MORE reasoning tokens than full CoT"
    return out


def latency_summary(values, sources):
    """Mean / median / p90 of latencies in ms; only when every value is recorded."""
    base = mean_of(values, sources, "latency")
    if base["status"] in ("not_computable", "undefined"):
        return base
    v = sorted(values)
    base.update(median=v[len(v) // 2], p90=v[min(len(v) - 1, math.ceil(0.9 * len(v)) - 1)], unit="ms")
    if base["status"] == "synthetic":
        base["note"] = "synthetic timings: NOT a performance result"
    return base


# ---------------------------------------------------------------- probe quality


def binary_counts(labels, scores, threshold):
    tp = sum(1 for y, p in zip(labels, scores) if p >= threshold and y == 1)
    fp = sum(1 for y, p in zip(labels, scores) if p >= threshold and y == 0)
    fn = sum(1 for y, p in zip(labels, scores) if p < threshold and y == 1)
    tn = sum(1 for y, p in zip(labels, scores) if p < threshold and y == 0)
    return tp, fp, fn, tn


def threshold_metrics(labels, scores, threshold):
    """Precision / recall / F1 of STOP (positive class) at a fixed threshold."""
    tp, fp, fn, tn = binary_counts(labels, scores, threshold)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    if precision is None or recall is None:
        f1 = None  # undefined: no predicted or no actual positives
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"threshold": threshold, "precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": len(labels)}


def probe_quality(labels, scores, threshold=None):
    """AUROC / average precision (threshold-free) + metrics at `threshold` if given."""
    if not scores:
        return {"status": "not_computable", "reason": "no probe probabilities"}
    if any(p is None for p in scores):
        return {"status": "not_computable", "reason": "some probe probabilities missing"}
    from sklearn.metrics import average_precision_score, roc_auc_score

    out = {"status": "ok", "n": len(labels), "positives": sum(labels), "negatives": len(labels) - sum(labels)}
    if len(set(labels)) < 2:
        out["auroc"] = undefined("only one class present")
        out["average_precision"] = undefined("only one class present") if sum(labels) == 0 else {"value": 1.0, "status": "ok"}
    else:
        out["auroc"] = {"value": float(roc_auc_score(labels, scores)), "status": "ok"}
        out["average_precision"] = {"value": float(average_precision_score(labels, scores)), "status": "ok"}
    if threshold is not None:
        out["at_threshold"] = threshold_metrics(labels, scores, threshold)
    return out
