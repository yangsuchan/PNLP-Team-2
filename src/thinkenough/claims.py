"""Find spatial claims in reasoning text and check them against the object list the model saw.

Precision over recall: only claims with one clear answer are kept. "Left" and "right" are left
out because they depend on a viewpoint the question does not give. Relations such as "near" were
tried and dropped: read on real text, most matches tied two unrelated objects together.
"""

from __future__ import annotations

import math
import re
from typing import Any

_NUM = r"-?\d+(?:\.\d+)?"
_ID = r"<?OBJ(\d{3})>?`?"
_LINE = re.compile(
    rf"^<OBJ(\d{{3}})> (.+?): center \(({_NUM}), ({_NUM}), ({_NUM})\), size \(({_NUM}), ({_NUM}), ({_NUM})\)",
    re.MULTILINE,
)
# The gaps refuse words that would tie the numbers to something else: a size, or another object.
_CENTER = re.compile(rf"{_ID}(?:(?!size|OBJ)[^\n()\[\]]){{0,40}}?[(\[]({_NUM}),\s*({_NUM}),\s*({_NUM})[)\]]", re.IGNORECASE)
_SIZE = re.compile(rf"{_ID}(?:(?!OBJ)[^\n]){{0,70}}?\bsize\b[^\n(\d]{{0,8}}\(({_NUM}),\s*({_NUM}),\s*({_NUM})\)", re.IGNORECASE)
_AXIS = re.compile(rf"{_ID}(?:(?!OBJ|\bvs\b|\bif\b)[^\n.()]){{0,30}}?\b([xyz])\s*=\s*({_NUM})")
_COMPARE = re.compile(
    rf"{_ID}(?:(?!OBJ)[^\n.]){{0,40}}?\b(larger|bigger|smaller|taller|shorter|higher|lower)\b than[^\n.]{{0,25}}?{_ID}"
)
_ARITHMETIC = re.compile(rf"(?<![\d.\[(,])({_NUM})\s*([-+*])\s*({_NUM})\s*=\s*({_NUM})(?![\d.,]*\d)")
_SQUARE = re.compile(rf"(?<![\d.])({_NUM})\^2\s*=\s*({_NUM})")
_ROOT = re.compile(rf"sqrt\(\s*({_NUM})\s*\)\s*(?:=|≈|~)\s*({_NUM})")
_DISTANCE = re.compile(rf"\bdistance\b[^\n.]{{0,50}}?(?:=|≈|~|\bis\b|\bof\b|\babout\b|\bapprox\w*\b|\broughly\b)\s*\**\s*({_NUM})\s*\**\s*(?:m\b|meters?\b|metres?\b)")


def object_table(question: str) -> dict[int, dict[str, Any]]:
    """The objects as written in the prompt, keyed by shown id."""
    table = {}
    for match in _LINE.finditer(question):
        values = [float(v) for v in match.groups()[2:]]
        table[int(match.group(1))] = {"name": match.group(2), "center": values[:3], "size": values[3:]}
    return table


def find_claims(text: str, question: str, true_distance: float | None = None) -> list[dict[str, Any]]:
    """Every checkable claim, in text order, with whether it holds.

    Args:
        text: Reasoning text.
        question: The full prompt holding the object list.
        true_distance: Distance between the question's two points, for distance claims.
    """
    table = object_table(question)
    claims: list[dict[str, Any]] = []

    def add(kind: str, match: re.Match, holds: bool, **extra: Any) -> None:
        claims.append({"kind": kind, "start": match.start(), "end": match.end(), "holds": holds, "text": match.group(0)[:120], **extra})

    def quote(kind: str, match: re.Match) -> None:
        entry = table.get(int(match.group(1)))
        if entry:
            said = [float(v) for v in match.groups()[1:]]
            # A bare triple may be either value, so it counts as false only if it matches neither.
            gap = min(max(abs(a - b) for a, b in zip(said, entry[field])) for field in ("center", "size"))
            add(kind, match, gap <= 0.051, object=int(match.group(1)), gap=gap)

    for match in _CENTER.finditer(text):
        quote("center", match)
    for match in _SIZE.finditer(text):
        quote("size", match)
    for match in _AXIS.finditer(text):
        entry = table.get(int(match.group(1)))
        if entry:  # The model often renames axes, so a mismatch here need not be an error; kept apart.
            add("axis", match, abs(float(match.group(3)) - entry["center"]["xyz".index(match.group(2))]) <= 0.051, object=int(match.group(1)))
    for match in _COMPARE.finditer(text):
        a, b = table.get(int(match.group(1))), table.get(int(match.group(3)))
        if a and b:
            add("compare", match, _compare_holds(a, b, match.group(2)))
    for match in _ARITHMETIC.finditer(text):
        left, op, right, result = match.groups()
        value = {"-": float(left) - float(right), "+": float(left) + float(right), "*": float(left) * float(right)}[op]
        add("arithmetic", match, abs(value - float(result)) <= 0.051)
    for match in _SQUARE.finditer(text):
        add("arithmetic", match, abs(float(match.group(1)) ** 2 - float(match.group(2))) <= max(0.051, 0.02 * float(match.group(2))))
    for match in _ROOT.finditer(text):
        add("arithmetic", match, abs(math.sqrt(max(float(match.group(1)), 0.0)) - float(match.group(2))) <= 0.051)
    if true_distance is not None:
        for match in _DISTANCE.finditer(text):
            add("distance", match, abs(float(match.group(1)) - true_distance) <= 0.2)
    return sorted(claims, key=lambda c: c["start"])


def counts_as_false(claim: dict[str, Any]) -> bool:
    """Whether a claim is an error. The model often renames axes, so an axis mismatch is not one."""
    return not claim["holds"] and claim["kind"] != "axis"


def later_fixed(claims: list[dict[str, Any]], index: int) -> bool:
    """Whether a false quote of an object is followed by a true quote of the same object."""
    claim = claims[index]
    if claim.get("object") is None:
        return False
    return any(
        later["kind"] == claim["kind"] and later.get("object") == claim.get("object") and later["holds"]
        for later in claims[index + 1 :]
    )


def _compare_holds(a: dict[str, Any], b: dict[str, Any], word: str) -> bool:
    volume = lambda o: o["size"][0] * o["size"][1] * o["size"][2]
    if word in ("larger", "bigger"):
        return volume(a) > volume(b)
    if word == "smaller":
        return volume(a) < volume(b)
    if word == "taller":
        return a["size"][2] > b["size"][2]
    if word == "shorter":
        return a["size"][2] < b["size"][2]
    return a["center"][2] > b["center"][2] if word == "higher" else a["center"][2] < b["center"][2]


_RECHECK = re.compile(r"\b(wait|actually|however|re-?(?:read|check|evaluate|examine|consider)|double[- ]check|on second thought)\b", re.IGNORECASE)
PROGRESS_NAMES = ("claims", "false_claims", "new_false", "objects", "new_objects", "revisits", "new_rechecks")


def char_positions(tokenizer: Any, text: str, tokens: list[int]) -> list[int]:
    """Where the text stands after each token count, so nothing later is read at a checkpoint."""
    spans = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    return [spans[min(count, len(spans)) - 1][1] if spans and count > 0 else 0 for count in tokens]


def progress_features(text: str, question: str, cuts: list[int]) -> list[list[float]]:
    """How far the reasoning has got at each character position, from the text and the object list.

    Needs no correct answer and no later text, so a stopping rule could use it while running.
    Distance claims are left out because checking them computes the answer itself.
    """
    claims = [c for c in find_claims(text, question) if c["kind"] != "arithmetic"]
    mentions = [(m.start(), int(m.group(1))) for m in re.finditer(_ID, text)]
    rechecks = [m.start() for m in _RECHECK.finditer(text)]
    rows, last_cut = [], 0
    for cut in cuts:
        seen = [c for c in claims if c["end"] <= cut]
        false = [c for c in seen if counts_as_false(c)]
        said = [i for start, i in mentions if start < cut]
        before = {i for start, i in mentions if start < last_cut}
        rows.append(
            [
                math.log1p(len(seen)),
                float(len(false)),
                float(sum(c["start"] >= last_cut for c in false)),
                math.log1p(len(set(said))),
                float(len(set(said) - before)),
                len(said) / max(len(set(said)), 1),
                float(sum(last_cut <= start < cut for start in rechecks)),
            ]
        )
        last_cut = cut
    return rows
