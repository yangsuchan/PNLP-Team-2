"""Tests for YAML config loading."""

from __future__ import annotations

from pathlib import Path

from thinkenough.config import load_inference_config

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "inference_default.yaml"


def test_load_default_config():
    cfg = load_inference_config(DEFAULT_CONFIG, repo_root=REPO_ROOT)
    assert cfg.backend == "mock"
    assert cfg.model.model_id == "Qwen/Qwen3-VL-8B-Thinking"
    assert cfg.model.dtype == "bfloat16"
    assert cfg.generation.max_new_tokens == 40960
    assert cfg.generation.temperature == 1.0
    assert cfg.generation.top_p == 0.95
    assert cfg.generation.top_k == 20
    assert cfg.thinking.think_end_token_id == 151668
    assert cfg.thinking.think_end_text == "</think>"
    assert cfg.repo_root == REPO_ROOT.resolve()


def test_resolve_path_relative():
    cfg = load_inference_config(DEFAULT_CONFIG, repo_root=REPO_ROOT)
    resolved = cfg.resolve_path("data/samples/example_inputs.jsonl")
    assert resolved == (REPO_ROOT / "data" / "samples" / "example_inputs.jsonl").resolve()


def test_generation_to_dict_roundtrip_fields():
    cfg = load_inference_config(DEFAULT_CONFIG, repo_root=REPO_ROOT)
    d = cfg.generation.to_dict()
    assert d["do_sample"] is True
    assert d["seed"] == 1234
    assert "presence_penalty" in d
