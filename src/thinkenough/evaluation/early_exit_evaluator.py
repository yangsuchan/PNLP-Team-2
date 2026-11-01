"""Compare Direct / Fixed-k / Full CoT / Probe early exit on the same question set.

요약: [4단계 평가] Direct / Fixed-k / Full CoT / Probe early exit을 같은 문항 집합에서 비교하고, 멈춤 기준값 τ는 validation에서만 고른다.

Three evaluation modes are kept apart and never mixed in one comparison:
  online              results of actual runs (one final answer per method/question)
  offline_simulation  "if we had stopped at checkpoint k": correctness = the forced
                      answer label at k, tokens = that checkpoint's prefix tokens
  probe_classification probe p_stop vs STOP labels, independent of any exit policy

Threshold protocol: the STOP threshold is chosen on VALIDATION predictions only,
saved with the question ids it was chosen on, and then applied unchanged to test.
Selecting on any other split, or evaluating on questions used for selection,
raises ThresholdLeakageError.
"""
from thinkenough.identity import question_sort_key


import hashlib
import json
from collections import defaultdict

from thinkenough.evaluation import metrics as M
from thinkenough.evaluation.answer_extraction import extract_final_answer
from thinkenough.evaluation.sqa3d_scoring import normalized_em, official_accuracy, official_match

METHOD_TYPES = {"direct", "fixed_k", "fixed_budget", "full_cot", "probe_early_exit", "trial_answer_early_exit", "oracle", "other"}


class EvaluationInputError(ValueError):
    """Duplicate rows, mismatched question sets, or mixed evaluation modes."""


class ThresholdLeakageError(ValueError):
    """Threshold selected on / applied to the wrong split."""


# ---------------------------------------------------------------- online method comparison


def _group_by_method(results):
    by_method, seen = defaultdict(list), set()
    for r in results:
        key = (r["method"], r["question_id"])
        if key in seen:
            raise EvaluationInputError(f"duplicate result row for method={key[0]} question_id={key[1]}")
        seen.add(key)
        by_method[r["method"]].append(r)
    return by_method


def check_same_question_set(by_method, baseline):
    """All methods must cover exactly the baseline's questions (same denominator)."""
    if baseline not in by_method:
        raise EvaluationInputError(f"baseline method '{baseline}' not found")
    base = {r["question_id"] for r in by_method[baseline]}
    problems = {}
    for m, rows in by_method.items():
        qs = {r["question_id"] for r in rows}
        if qs != base:
            problems[m] = {"missing": sorted(base - qs, key=question_sort_key)[:10], "extra": sorted(qs - base, key=question_sort_key)[:10]}
    if problems:
        raise EvaluationInputError(f"methods are not evaluated on the same question set: {problems}")
    return sorted(base, key=question_sort_key)


def _score_rows(rows, references):
    scored = []
    for r in rows:
        ref = references.get(r["question_id"])
        if ref is None:
            raise EvaluationInputError(f"no reference for question_id {r['question_id']}")
        answer, status = (extract_final_answer(r.get("final_answer"))
                          if r.get("generation_status", "ok") == "ok" else ("", "generation_failed"))
        gold = ref["reference_answers"][0]
        scored.append({
            "question_id": r["question_id"], "answer": answer, "extraction_status": status,
            "em": status == "ok" and normalized_em(answer, gold),
            "official": official_match(answer, gold) if status == "ok" else official_match("", gold),
            "official_empty_wrong": status == "ok" and official_match(answer, gold, empty_pred_is_wrong=True),
        })
    return scored


def _official_with_merge(rows, scored, references):
    """Official accuracy with the official item merge, when position/rotation are available."""
    recs = [references[r["question_id"]] for r in rows]
    if all("meta" in x and "position" in x["meta"] for x in recs):
        preds = {s["question_id"]: s["answer"] for s in scored}
        res = official_accuracy(recs, preds)
        return {"value": 100.0 * res["accuracy"], "status": "ok", "correct": res["correct"],
                "total_units": res["total"], "unit_merge": "applied (official compute_accuracy)"}
    flags = [s["official"] for s in scored]
    out = M.accuracy(flags)
    out["unit_merge"] = "not applied: references have no position/rotation (per-question official match)"
    return out


def evaluate_online(results, references, baseline="full_cot"):
    """Per-method metrics and comparisons vs `baseline` on one shared question set."""
    by_method = _group_by_method(results)
    questions = check_same_question_set(by_method, baseline)
    out = {"mode": "online", "baseline": baseline, "num_questions": len(questions), "methods": {}}

    for method, rows in sorted(by_method.items()):
        modes = {r.get("evaluation_mode", "online") for r in rows}
        if modes != {"online"}:
            raise EvaluationInputError(f"method {method} mixes evaluation modes {modes}")
        rows = sorted(rows, key=lambda r: question_sort_key(r["question_id"]))
        scored = _score_rows(rows, references)
        tok = lambda key, src="token_count_source": M.mean_of([r.get(key) for r in rows], [r.get(src) for r in rows], key)
        e2e = [None if any(r.get(k) is None for k in ("reasoning_tokens", "answer_tokens")) else
               r["reasoning_tokens"] + r["answer_tokens"] + (r.get("trial_answer_tokens") or 0) for r in rows]
        lat = lambda key: M.latency_summary([(r.get("latency") or {}).get(key) for r in rows],
                                            [r.get("latency_source") for r in rows])
        out["methods"][method] = {
            "method_type": rows[0].get("method_type"),
            "synthetic": any(r.get("synthetic") for r in rows),
            "accuracy_em": M.accuracy([s["em"] for s in scored]),
            "accuracy_official": _official_with_merge(rows, scored, references),
            "accuracy_official_empty_as_wrong": M.accuracy([s["official_empty_wrong"] for s in scored]),
            "failed_or_empty_answers": sum(s["extraction_status"] != "ok" for s in scored),
            "reasoning_tokens": tok("reasoning_tokens"),
            "answer_tokens": tok("answer_tokens"),
            "trial_answer_tokens": tok("trial_answer_tokens"),
            "end_to_end_generated_tokens": M.mean_of(e2e, [r.get("token_count_source") for r in rows], "end-to-end tokens"),
            "probe_calls": M.mean_of([r.get("probe_calls") for r in rows],
                                     ["synthetic" if r.get("synthetic") else "measured" for r in rows], "probe calls")
                           if any(r.get("probe_calls") is not None for r in rows) else M.not_computable("no probe"),
            "latency_total_ms": lat("total_ms"),
            "latency_reasoning_ms": lat("reasoning_ms"),
            "latency_answer_ms": lat("answer_ms"),
            "latency_probe_ms": lat("probe_ms"),
            "measurement": rows[0].get("measurement"),
        }

    base = out["methods"][baseline]
    for method, res in out["methods"].items():
        if method == baseline:
            continue
        acc, full = res["accuracy_em"].get("value"), base["accuracy_em"].get("value")
        res["vs_baseline"] = {
            "performance_retention": M.performance_retention(acc, full),
            "accuracy_difference": M.accuracy_difference(acc, full),
            "reasoning_token_reduction": M.token_reduction(base["reasoning_tokens"], res["reasoning_tokens"]),
            "end_to_end_token_reduction": M.token_reduction(base["end_to_end_generated_tokens"], res["end_to_end_generated_tokens"]),
        }
    return out


# ---------------------------------------------------------------- probe predictions


def join_predictions(labels, predictions, sample_index=0):
    """(checkpoint_id, split, question_id, label, p_stop) for labeled rows of one sample index.

    Several samples per checkpoint stay separate in the label file; evaluation
    uses `sample_index` (default 0) so each checkpoint counts once. No majority vote.
    """
    preds = {}
    for p in predictions:
        if p["checkpoint_id"] in preds:
            raise EvaluationInputError(f"duplicate prediction for {p['checkpoint_id']}")
        preds[p["checkpoint_id"]] = p
    joined, no_pred = [], []
    for r in labels:
        if r["label_status"] != "labeled" or (r.get("sample_index") or 0) != sample_index:
            continue
        p = preds.get(r["checkpoint_id"])
        if p is None or p.get("p_stop") is None:
            no_pred.append(r["checkpoint_id"])
            continue
        if p.get("split") != r["split"]:
            raise EvaluationInputError(f"split mismatch for {r['checkpoint_id']}")
        joined.append({"checkpoint_id": r["checkpoint_id"], "question_id": r["question_id"], "split": r["split"],
                       "checkpoint_index": r["checkpoint_index"], "label": r["stop_label"], "p_stop": p["p_stop"]})
    return joined, no_pred


def _selection_hash(rows):
    payload = json.dumps(sorted((r["checkpoint_id"], r["p_stop"], r["label"]) for r in rows))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def select_threshold(joined, criterion="f1", target_precision=0.9, probe_version=None):
    """Choose the STOP threshold on validation rows only.

    criterion="f1": maximize F1; "target_precision": highest recall with precision >= target.
    Ties go to the higher (more conservative) threshold.
    """
    splits = {r["split"] for r in joined}
    if splits != {"validation"}:
        raise ThresholdLeakageError(f"threshold must be selected on validation only, got splits {sorted(splits)}")
    labels = [r["label"] for r in joined]
    scores = [r["p_stop"] for r in joined]
    best = None
    for t in sorted(set(scores), reverse=True):
        m = M.threshold_metrics(labels, scores, t)
        if criterion == "f1":
            key = m["f1"] if m["f1"] is not None else -1
        elif criterion == "target_precision":
            ok = m["precision"] is not None and m["precision"] >= target_precision
            key = m["recall"] if ok and m["recall"] is not None else -1
        else:
            raise ValueError(f"unknown criterion {criterion}")
        if best is None or key > best[0]:
            best = (key, t, m)
    if best is None or best[0] < 0:
        return {"threshold": None, "status": "undefined", "reason": f"no threshold satisfies {criterion}"}
    return {
        "threshold": best[1], "status": "ok", "criterion": criterion,
        "target_precision": target_precision if criterion == "target_precision" else None,
        "selected_on_split": "validation",
        "selection_question_ids": sorted({r["question_id"] for r in joined}, key=question_sort_key),
        "selection_checkpoints": len(joined),
        "selection_hash": _selection_hash(joined),
        "metrics_at_threshold_on_validation": best[2],
        "probe_version": probe_version,
    }


def check_threshold_usage(threshold_record, eval_rows):
    """Refuse thresholds not chosen on validation, or evaluation on selection questions."""
    if threshold_record.get("selected_on_split") != "validation" or threshold_record.get("threshold") is None:
        raise ThresholdLeakageError("threshold record was not selected on validation")
    overlap = {r["question_id"] for r in eval_rows} & set(threshold_record["selection_question_ids"])
    eval_splits = {r["split"] for r in eval_rows}
    if eval_splits != {"validation"} and overlap:
        raise ThresholdLeakageError(f"evaluation questions were used for threshold selection: {sorted(overlap, key=question_sort_key)[:5]}")


def evaluate_probe(joined, split, threshold_record=None):
    """Offline probe classification metrics on one split (threshold fixed beforehand)."""
    rows = [r for r in joined if r["split"] == split]
    threshold = None
    if threshold_record is not None:
        check_threshold_usage(threshold_record, rows)
        threshold = threshold_record["threshold"]
    out = M.probe_quality([r["label"] for r in rows], [r["p_stop"] for r in rows], threshold)
    out.update(mode="probe_classification", split=split)
    return out


# ---------------------------------------------------------------- offline exit simulation


def simulate_early_exit(checkpoints, labels, predictions, threshold_record, split, fixed_ks=(1, 2), sample_index=0):
    """Offline simulation from forced-answer labels (NOT an online run).

    A question is included only if every checkpoint has a label (sample_index) and
    a probe probability, so all simulated methods share one denominator.
    """
    lab = {r["checkpoint_id"]: r for r in labels
           if r["label_status"] == "labeled" and (r.get("sample_index") or 0) == sample_index}
    preds = {p["checkpoint_id"]: p for p in predictions}
    by_q = defaultdict(list)
    for c in checkpoints:
        if c["split"] == split:
            by_q[c["question_id"]].append(c)
    if threshold_record is not None:
        check_threshold_usage(threshold_record, [{"question_id": q, "split": split} for q in by_q])
    tau = threshold_record["threshold"] if threshold_record else None

    excluded, per_method = {}, defaultdict(list)
    for q, cps in sorted(by_q.items(), key=lambda item: question_sort_key(item[0])):
        cps.sort(key=lambda c: c["checkpoint_index"])
        if any(c["checkpoint_id"] not in lab for c in cps):
            excluded[q] = "unlabeled checkpoint(s)"
            continue
        if tau is not None and any(preds.get(c["checkpoint_id"], {}).get("p_stop") is None for c in cps):
            excluded[q] = "missing probe probability"
            continue
        last = len(cps) - 1
        exits = {"direct_sim": 0, "full_cot_sim": last}
        for k in fixed_ks:
            exits[f"fixed_{k}_sim"] = min(k, last)
        if tau is not None:
            exits["probe_sim"] = next((i for i, c in enumerate(cps) if preds[c["checkpoint_id"]]["p_stop"] >= tau), last)
        exits["oracle_sim"] = next((i for i, c in enumerate(cps) if lab[c["checkpoint_id"]]["stop_label"] == 1), last)
        for method, i in exits.items():
            c = cps[i]
            per_method[method].append({"question_id": q, "exit_index": i, "correct": lab[c["checkpoint_id"]]["stop_label"],
                                       "reasoning_tokens": c.get("prefix_token_count"),
                                       "source": "synthetic" if (c.get("source") or {}).get("synthetic") else c.get("token_count_source"),
                                       "probe_calls": i + 1 if method == "probe_sim" else None})

    out = {"mode": "offline_simulation", "split": split, "threshold": tau,
           "note": "correctness = forced-answer label at the exit checkpoint; not an online early-exit run; "
                   "oracle_sim uses ground truth and is an upper bound, not a method",
           "included_questions": len({r["question_id"] for rows in per_method.values() for r in rows}),
           "excluded_questions": excluded, "methods": {}}
    for method, rows in per_method.items():
        out["methods"][method] = {
            "accuracy_em": M.accuracy([r["correct"] for r in rows]),
            "reasoning_tokens": M.mean_of([r["reasoning_tokens"] for r in rows], [r["source"] for r in rows], "reasoning tokens"),
            "mean_exit_checkpoint": sum(r["exit_index"] for r in rows) / len(rows) if rows else None,
            "probe_calls": (sum(r["probe_calls"] for r in rows) / len(rows)) if method == "probe_sim" and rows else None,
        }
    if "full_cot_sim" in out["methods"]:
        base = out["methods"]["full_cot_sim"]
        for method, res in out["methods"].items():
            if method != "full_cot_sim":
                res["vs_full_cot_sim"] = {
                    "performance_retention": M.performance_retention(res["accuracy_em"].get("value"), base["accuracy_em"].get("value")),
                    "accuracy_difference": M.accuracy_difference(res["accuracy_em"].get("value"), base["accuracy_em"].get("value")),
                    "reasoning_token_reduction": M.token_reduction(base["reasoning_tokens"], res["reasoning_tokens"]),
                }
    return out
