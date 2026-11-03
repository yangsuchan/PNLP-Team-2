"""What a stopping rule can read from the forced answer and the reasoning so far.

Nothing here uses the correct answer, the reasoning after the checkpoint, or its total length.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

import numpy as np

from thinkenough.checkpoints import QuestionTrace
from thinkenough.claims import _ID, char_positions, object_table
from thinkenough.scot import parse_answer_key

STABILITY = ("tokens_used", "same_as_last", "run_length", "has_answer")
DISTANCE_EVIDENCE = ("derived", "times_derived", "last_derived_matches", "values_derived")
OBJECT_EVIDENCE = ("mentions", "mention_share", "last_mentioned", "same_name_compared")
_DERIVED = re.compile(
    r"(?:sqrt\([^)]*\)|\\sqrt\{[^}]*\}|√[^=≈~\n]{0,40}|distance[^=≈~\n.]{0,50})"
    r"\s*(?:=|≈|~|\\approx|is|of|about)\s*\*{0,2}(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_DISTANCE_MATCH = 0.05


def answer_keys(trace: QuestionTrace, meta: dict[str, Any], limit: int) -> list[Any]:
    """Each forced answer in comparable form. An answer at ``limit`` tokens was cut off."""
    return [parse_answer_key(c.row["answer"], meta, c.row["answer_tokens"] >= limit) for c in trace.checkpoints]


def _distance_evidence(key: Any, derived: list[float]) -> list[float]:
    matches = [abs(value - key) <= _DISTANCE_MATCH for value in derived] if key is not None else []
    return [float(any(matches)), math.log1p(sum(matches)), float(bool(matches) and matches[-1]), math.log1p(len(derived))]


def _object_evidence(key: Any, seen: list[int], table: dict[int, dict[str, Any]]) -> list[float]:
    mentions = seen.count(key) if key is not None else 0
    name = table[key].get("name") if key is not None and key in table else None
    same_name = [i for i, entry in table.items() if entry.get("name") == name] if name else []
    return [
        math.log1p(mentions),
        mentions / max(len(seen), 1),
        float(bool(seen) and key == seen[-1]),
        len({i for i in seen if i in same_name}) / max(len(same_name), 1),
    ]


def evidence_features(trace: QuestionTrace, run: dict[str, Any], tokenizer: Any, budget: int, limit: int) -> np.ndarray:
    """One row per checkpoint: the STABILITY columns, then four evidence columns for the task."""
    text = run["reasoning_text"] or ""
    distance_task = run["metadata"]["task"] == "relationship_analysis"
    cuts = char_positions(tokenizer, text, [c.tokens for c in trace.checkpoints])
    table = object_table(run["question"])
    mentions = [(m.start(), int(m.group(1))) for m in re.finditer(_ID, text)]
    derived = [(m.end(), float(m.group(1))) for m in _DERIVED.finditer(text)]
    rows, previous, streak = [], None, 0
    for checkpoint, cut, key in zip(trace.checkpoints, cuts, answer_keys(trace, run["metadata"], limit)):
        same = key is not None and key == previous
        streak = streak + 1 if same else (1 if key is not None else 0)
        previous = key
        evidence = (
            _distance_evidence(key, [value for end, value in derived if end <= cut])
            if distance_task
            else _object_evidence(key, [i for start, i in mentions if start < cut], table)
        )
        rows.append([checkpoint.tokens / budget, float(same), math.log1p(streak), float(key is not None), *evidence])
    return np.array(rows)


def load_run(runs: Any, sample_id: str) -> dict[str, Any]:
    return json.loads((runs / f"{sample_id}_qwen3_vl.json").read_text(encoding="utf-8"))
