"""Tests for reasoning checkpoint construction (synthetic CoT only).

요약: [2단계] checkpoint 만들기 테스트 (가짜 CoT 사용).

Run from the repo root:
    .venv/bin/python -m pytest tests/ -v
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thinkenough.data.reasoning_checkpoint import (  # noqa: E402
    SEGMENTERS, QwenTokenAligner, align_boundary, build_checkpoints, parse_generation,
)

FIXTURE = ROOT / "tests/fixtures/fake_qwen3_outputs.jsonl"
CASES = {r["case"]: r for r in map(json.loads, open(FIXTURE))}


def reasoning_segments(text, segmenter="sentence"):
    reasoning = parse_generation(text).reasoning_text
    return [reasoning[s:e] for s, e in SEGMENTERS[segmenter](reasoning)]


def assert_prefixes_accumulate(checkpoints, reasoning):
    assert checkpoints[0]["checkpoint_type"] == "no_reasoning"
    assert checkpoints[0]["reasoning_prefix"] == ""
    for prev, cur in zip(checkpoints, checkpoints[1:]):
        assert cur["checkpoint_index"] == prev["checkpoint_index"] + 1
        assert cur["reasoning_prefix"] == prev["reasoning_prefix"] + cur["segment_text"]
        assert cur["segment_char_start"] == prev["char_end"]
        assert cur["reasoning_prefix"] == reasoning[: cur["char_end"]]
    assert checkpoints[-1]["is_final"]
    assert checkpoints[-1]["reasoning_prefix"] == reasoning


# ---------------------------------------------------------------- parsing


@pytest.mark.parametrize("case, status, answer", [
    ("multi_sentence", "ok", "brown"),
    ("empty_reasoning", "empty_reasoning", "yes"),
    ("missing_open_tag", "missing_open_tag", "radiator"),
    ("no_think_tags", "no_think_tags", "brown"),
    ("unclosed_think", "unclosed_think", None),
])
def test_parse_status(case, status, answer):
    parsed = parse_generation(CASES[case]["generated_text"])
    assert parsed.parse_status == status
    assert parsed.final_answer == answer


def test_parse_empty_and_none_outputs():
    assert parse_generation("").parse_status == "empty_output"
    assert parse_generation(None).parse_status == "empty_output"


def test_parse_multiple_think_blocks_warns():
    parsed = parse_generation("<think>A.</think>\n<think>B.</think> ans")
    assert parsed.reasoning_text == "A."
    assert "multiple_think_blocks" in parsed.warnings


def test_reasoning_offset_points_into_original_text():
    for r in CASES.values():
        p = parse_generation(r["generated_text"])
        start = p.reasoning_char_offset
        assert r["generated_text"][start:start + len(p.reasoning_text)] == p.reasoning_text


# ---------------------------------------------------------------- segmentation


def test_multi_sentence():
    segs = reasoning_segments(CASES["multi_sentence"]["generated_text"])
    assert len(segs) == 5
    assert segs[2].strip().endswith("!")


def test_single_sentence():
    assert len(reasoning_segments(CASES["single_sentence"]["generated_text"])) == 1


def test_newlines_and_numbered_steps():
    text = CASES["numbered_steps"]["generated_text"]
    for segmenter in ("sentence", "line", "numbered_step"):
        segs = reasoning_segments(text, segmenter)
        assert [s.strip()[:2] for s in segs] == ["Le", "1.", "2.", "3."], segmenter


def test_decimals_and_abbreviations_are_not_boundaries():
    segs = reasoning_segments(CASES["decimals_abbrev"]["generated_text"])
    assert len(segs) == 4
    assert "approx. 3.5 m" in segs[0] and "1.2 m" in segs[0]
    assert "2.9, e.g. the table" in segs[1]
    assert segs[2].startswith("Fig. 2")


def test_unclear_boundaries_stay_one_segment():
    assert len(reasoning_segments(CASES["unclear_boundaries"]["generated_text"])) == 1


def test_ellipsis_followed_by_lowercase_is_not_a_boundary():
    segs = reasoning_segments(CASES["wait_and_newlines"]["generated_text"])
    assert any(s.startswith("Hmm... then") for s in segs)
    assert any(s.startswith("Wait,") for s in segs)


@pytest.mark.parametrize("segmenter", list(SEGMENTERS))
@pytest.mark.parametrize("case", list(CASES))
def test_segments_join_back_to_original(case, segmenter):
    reasoning = parse_generation(CASES[case]["generated_text"]).reasoning_text
    spans = SEGMENTERS[segmenter](reasoning)
    assert "".join(reasoning[s:e] for s, e in spans) == reasoning
    assert all(prev[1] == cur[0] for prev, cur in zip(spans, spans[1:]))
    assert all(reasoning[s:e].strip() for s, e in spans)


# ---------------------------------------------------------------- checkpoints


@pytest.mark.parametrize("segmenter", list(SEGMENTERS))
@pytest.mark.parametrize("case", list(CASES))
def test_checkpoint_prefixes_accumulate(case, segmenter):
    record = CASES[case]
    checkpoints, parsed = build_checkpoints(record, segmenter=segmenter)
    assert_prefixes_accumulate(checkpoints, parsed.reasoning_text)
    assert all(c["question_id"] == record["question_id"] for c in checkpoints)


@pytest.mark.parametrize("case", ["empty_reasoning", "no_think_tags"])
def test_no_reasoning_gives_only_checkpoint_zero(case):
    checkpoints, parsed = build_checkpoints(CASES[case])
    assert len(checkpoints) == 1
    assert checkpoints[0]["parse_status"] == parsed.parse_status


def test_missing_open_tag_still_builds_checkpoints():
    checkpoints, _ = build_checkpoints(CASES["missing_open_tag"])
    assert len(checkpoints) == 3
    assert checkpoints[-1]["parse_status"] == "missing_open_tag"


def test_token_positions_are_null_without_generated_ids():
    for record in CASES.values():
        checkpoints, _ = build_checkpoints(record)
        for c in checkpoints:
            assert c["reasoning_token_start"] is None
            assert c["reasoning_token_end"] is None
            assert c["hidden_state_position"] is None
            assert c["alignment_status"] == "no_token_ids"


def test_answers_are_not_needed():
    record = dict(CASES["multi_sentence"])
    tampered = dict(record, reference_answers=["__SENTINEL__"])
    a, _ = build_checkpoints(record)
    b, _ = build_checkpoints(tampered)
    assert a == b


def test_keep_prefix_text_false_drops_text_only():
    full, _ = build_checkpoints(CASES["multi_sentence"])
    lean, _ = build_checkpoints(CASES["multi_sentence"], keep_prefix_text=False)
    assert all(c["reasoning_prefix"] is None for c in lean)
    assert [c["char_end"] for c in full] == [c["char_end"] for c in lean]


def test_align_boundary():
    offsets = [(0, 7), (7, 8), (8, 11), (11, 13)]
    assert align_boundary(offsets, 8) == (2, True)
    assert align_boundary(offsets, 12) == (4, False)
    assert align_boundary(offsets, 0) == (0, True)


# ---------------------------------------------------------------- tokenizer-based (synthetic ids)


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", local_files_only=True)
    except OSError:
        pytest.skip("Qwen3 tokenizer not cached locally")


def test_retokenized_counts_are_monotonic(tokenizer):
    checkpoints, _ = build_checkpoints(CASES["multi_sentence"], tokenizer=tokenizer)
    counts = [c["prefix_token_count"] for c in checkpoints]
    assert counts[0] == 0
    assert counts == sorted(counts)
    assert all(c["token_count_source"] == "retokenized" for c in checkpoints)


@pytest.mark.parametrize("case", ["multi_sentence", "numbered_steps", "wait_and_newlines", "missing_open_tag"])
def test_alignment_with_synthetic_token_ids(tokenizer, case):
    """SYNTHETIC ids (re-tokenized fixture text) - real Qwen3 generations must be re-checked."""
    record = dict(CASES[case])
    ids = tokenizer(record["generated_text"], add_special_tokens=False)["input_ids"]
    record.update(generated_token_ids=ids, prompt_token_count=100)
    checkpoints, parsed = build_checkpoints(record, aligner=QwenTokenAligner(tokenizer))
    text = record["generated_text"]
    start = checkpoints[0]["reasoning_token_start"]
    for c in checkpoints:
        end = c["reasoning_token_end"]
        assert c["alignment_status"] in ("exact", "exact_up_to_whitespace")
        assert c["prefix_token_count"] == end - start
        assert c["hidden_state_position"] == 100 + end - 1
        # tokens[:end] cover the prefix content and at most extra whitespace, never next-segment text
        content = text[: c["alignment_char_end"]]
        covered = tokenizer.decode(ids[:end])
        assert covered.startswith(content)
        assert not covered[len(content):].strip()
        assert content.rstrip() == (text[: parsed.reasoning_char_offset] + c["reasoning_prefix"]).rstrip()


def test_newline_merged_token_is_flagged(tokenizer):
    """Qwen3 merges '.\\n\\n' into one token: the boundary falls inside it, but only whitespace is extra."""
    record = dict(CASES["wait_and_newlines"])
    record["generated_token_ids"] = tokenizer(record["generated_text"], add_special_tokens=False)["input_ids"]
    checkpoints, _ = build_checkpoints(record, aligner=QwenTokenAligner(tokenizer))
    assert any(c["alignment_status"] == "exact_up_to_whitespace" for c in checkpoints)
    assert not any(c["alignment_status"] == "boundary_inside_token" for c in checkpoints)


def test_mismatched_ids_are_not_aligned(tokenizer):
    record = dict(CASES["multi_sentence"])
    record["generated_token_ids"] = tokenizer("something else", add_special_tokens=False)["input_ids"]
    checkpoints, _ = build_checkpoints(record, aligner=QwenTokenAligner(tokenizer))
    assert all(c["alignment_status"] == "text_mismatch" for c in checkpoints)
    assert all(c["reasoning_token_end"] is None for c in checkpoints)
