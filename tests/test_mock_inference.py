"""End-to-end tests for MockRunner (no GPU / no weights)."""

from __future__ import annotations

from pathlib import Path

from thinkenough.config import load_inference_config
from thinkenough.inference import create_runner
from thinkenough.io_utils import load_samples_jsonl, save_results_jsonl
from thinkenough.schemas import InferenceSample

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "inference_default.yaml"
SAMPLE_JSONL = REPO_ROOT / "data" / "samples" / "example_inputs.jsonl"


def test_create_mock_runner_and_generate(tmp_path: Path):
    cfg = load_inference_config(DEFAULT_CONFIG, repo_root=REPO_ROOT)
    assert cfg.backend == "mock"
    runner = create_runner(cfg)

    sample = InferenceSample(
        sample_id="unit_mock_1",
        image_path="tests/fixtures/sample_image.png",
        question="Describe the image.",
    )
    result = runner.generate_cot(sample)

    assert result.backend == "mock"
    assert result.parse_status == "ok"
    assert result.reasoning_text
    assert result.final_answer
    assert result.total_generated_token_count == len(result.output_token_ids)
    assert result.reasoning_token_count > 0
    assert result.answer_token_count > 0
    assert 151668 in result.output_token_ids

    out = tmp_path / "mock.jsonl"
    save_results_jsonl([result], out)
    assert out.is_file()
    assert "unit_mock_1" in out.read_text(encoding="utf-8")


def test_mock_pipeline_from_example_inputs():
    cfg = load_inference_config(DEFAULT_CONFIG, repo_root=REPO_ROOT)
    runner = create_runner(cfg)
    samples = load_samples_jsonl(SAMPLE_JSONL)
    assert samples

    result = runner.generate_cot(samples[0])
    assert result.sample_id == samples[0].sample_id
    assert result.question == samples[0].question
    assert "generation_config" in result.to_dict()
