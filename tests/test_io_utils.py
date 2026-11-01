"""Tests for JSON / JSONL IO helpers."""

from __future__ import annotations

from pathlib import Path

from thinkenough.io_utils import (
    iter_results_jsonl,
    load_samples_jsonl,
    save_result_json,
    save_results_jsonl,
)
from thinkenough.schemas import InferenceResult

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_JSONL = REPO_ROOT / "data" / "samples" / "example_inputs.jsonl"


def _make_result(sample_id: str = "t1") -> InferenceResult:
    return InferenceResult(
        sample_id=sample_id,
        model_name="mock::Qwen/Qwen3-VL-8B-Thinking",
        model_revision=None,
        dtype="bfloat16",
        image_path="data/samples/sample_image.png",
        question="q?",
        generation_config={"seed": 1234},
        raw_output="r\n</think>\n\na",
        reasoning_text="r",
        final_answer="a",
        input_token_ids=[1, 2],
        output_token_ids=[10, 151668, 20],
        reasoning_token_count=2,
        answer_token_count=1,
        total_generated_token_count=3,
        latency_seconds=0.01,
        parse_status="ok",
        backend="mock",
    )


def test_load_samples_jsonl():
    samples = load_samples_jsonl(SAMPLE_JSONL)
    assert len(samples) >= 1
    assert samples[0].sample_id == "sample_0001"
    assert samples[0].question
    assert samples[0].image_path


def test_save_and_reload_jsonl(tmp_path: Path):
    results = [_make_result("a"), _make_result("b")]
    out = tmp_path / "out.jsonl"
    save_results_jsonl(results, out, include_token_ids=True)
    loaded = list(iter_results_jsonl(out))
    assert len(loaded) == 2
    assert loaded[0].sample_id == "a"
    assert loaded[0].output_token_ids == [10, 151668, 20]
    assert loaded[1].final_answer == "a"


def test_save_result_json_omits_token_ids(tmp_path: Path):
    path = tmp_path / "one.json"
    save_result_json(_make_result(), path, include_token_ids=False)
    text = path.read_text(encoding="utf-8")
    assert '"input_token_ids": []' in text
    assert '"output_token_ids": []' in text
