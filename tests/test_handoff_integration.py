"""Offline integration: generation -> handoff -> validation -> checkpoint selection."""
import json
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from thinkenough.config import InferenceConfig
from thinkenough.data.checkpoint_selection import select_checkpoints, verify_selection
from thinkenough.data.reasoning_checkpoint import build_checkpoints, checkpoint_id
from thinkenough.generation.handoff import generation_record
from thinkenough.generation.output_validation import GenerationValidator, load_references
from thinkenough.inference.mock_runner import MockRunner
from thinkenough.inference.qwen3_vl_runner import Qwen3VLRunner
from thinkenough.schemas import InferenceSample

ROOT = Path(__file__).resolve().parents[1]


def sample(qid="scene-A"):
    return InferenceSample("safe-sample", "tests/fixtures/sample_image.png", "Which object is behind?",
                           metadata={"reference_answers": ["SECRET"], "position": [1, 2, 3]},
                           question_id=qid, split="validation")


def test_mock_handoff_and_string_id_selection(tmp_path):
    result = MockRunner(InferenceConfig(repo_root=ROOT)).generate_cot(sample())
    record = generation_record(result, "offline-test")
    assert record["synthetic"] is True
    assert "SECRET" not in json.dumps(record)
    assert record["question_id"] == "scene-A"
    assert record["reasoning_token_count"] == 5
    assert record["final_answer_text"].startswith("\n\n")
    refs = tmp_path / "references.jsonl"
    refs.write_text(json.dumps({"question_id": "scene-A", "split": "validation"}) + "\n")
    validator = GenerationValidator(load_references([refs]))
    validator.validate(record)
    assert validator.report()["errors"] == 0
    checkpoints, _ = build_checkpoints(record)
    selected, _ = select_checkpoints(checkpoints, {"strategy": "uniform", "max_checkpoints": 3})
    assert verify_selection(checkpoints, selected, {"strategy": "uniform", "max_checkpoints": 3})
    assert checkpoint_id("r", 1, "sentence", 0) != checkpoint_id("r", "1", "sentence", 0)
    assert all(c["hidden_state_position"] is None for c in checkpoints)


class TokenRow:
    def __init__(self, ids):
        self.ids = ids
    def tolist(self):
        return self.ids


class Inputs(dict):
    def to(self, device):
        return self


class Tokenizer:
    vocab = {1: "prompt", 151667: "<think>", 151668: "</think>",
             10: "Left. ", 11: "Ahead.", 20: "\n\nB", 151645: "<|im_end|>"}
    def decode(self, ids, skip_special_tokens=False, **kwargs):
        return "".join(self.vocab[i] for i in ids if not skip_special_tokens or i < 151000)


def fake_qwen(output, limit=20):
    cfg = InferenceConfig(repo_root=ROOT)
    cfg.generation.max_new_tokens = limit
    runner = Qwen3VLRunner(cfg)
    runner._loaded = True
    runner.model_revision = "a" * 40
    runner._torch = SimpleNamespace(inference_mode=nullcontext,
                                    cuda=SimpleNamespace(is_available=lambda: False))
    tok = Tokenizer()
    runner.processor = SimpleNamespace(
        tokenizer=tok, decode=tok.decode,
        apply_chat_template=lambda *a, **kw: Inputs(input_ids=[TokenRow([1, 151667])]),
    )
    runner.model = SimpleNamespace(
        device="cpu", dtype="torch.bfloat16", generation_config=SimpleNamespace(eos_token_id=[151645]),
        generate=lambda **kw: [TokenRow([1, 151667] + output)],
    )
    return runner, tok


@pytest.mark.parametrize("output,limit,finish", [
    ([10, 11, 151668, 20, 151645], 20, "stop"),
    ([10, 11], 2, "length"),
    ([10, 151668, 20, 151645], 4, "stop"),  # EOS at the length limit is still stop
])
def test_qwen_artifacts_without_weights(output, limit, finish):
    runner, tokenizer = fake_qwen(output, limit)
    result = runner.generate_cot(sample(123))
    record = generation_record(result, "fake-qwen")
    assert record["finish_reason"] == finish
    assert record["generated_token_ids"] == output
    assert record["prompt_text"] == "prompt<think>"
    assert record["generated_text"] == tokenizer.decode(output, skip_special_tokens=False)
    validator = GenerationValidator(decode=tokenizer.decode, tokenizer_id=record["tokenizer"])
    validator.validate(record)
    assert validator.report()["errors"] == 0
    if finish == "length":
        assert result.reasoning_token_count is None
        assert record["final_answer_text"] is None
    else:
        assert record["final_answer_text"] == "\n\nB<|im_end|>"
        assert result.reasoning_token_count == output.index(151668)
        assert result.final_answer == "B"


@pytest.mark.parametrize("revision", [None, "main", "placeholder"])
def test_real_handoff_rejects_unresolved_revisions(revision):
    runner, _ = fake_qwen([10, 151668, 20, 151645])
    result = runner.generate_cot(sample())
    result.tokenizer_revision = revision
    with pytest.raises(ValueError, match="commit hashes"):
        generation_record(result, "bad")


def test_exact_checkpoint_positions_with_fake_aligner():
    runner, tok = fake_qwen([10, 11, 151668, 20, 151645])
    record = generation_record(runner.generate_cot(sample()), "aligned")
    class Aligner:
        def offsets(self, ids):
            text, offsets = "", []
            for tid in ids:
                start = len(text)
                text += tok.decode([tid])
                offsets.append((start, len(text)))
            return text, offsets
    checkpoints, _ = build_checkpoints(record, aligner=Aligner())
    assert checkpoints[0]["prefix_token_count"] == 0
    assert checkpoints[0]["hidden_state_position"] == 1  # prompt's last token
    assert checkpoints[-1]["reasoning_token_end"] == 2  # excludes </think>
    assert checkpoints[-1]["hidden_state_position"] == 3
    selected, _ = select_checkpoints(checkpoints, {"strategy": "token_interval", "interval": 1})
    assert selected[-1] == checkpoints[-1]


def run_script(name, *args):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / name), *map(str, args)],
                          cwd=ROOT, capture_output=True, text=True)


def test_cli_pipeline_and_overwrite_protection(tmp_path):
    inferred = run_script("run_inference.py", "--backend", "mock", "--output-dir", tmp_path,
                          "--run-id", "smoke")
    assert inferred.returncode == 0, inferred.stderr
    generations = tmp_path / "smoke/generations.jsonl"
    assert run_script("validate_generation_output.py", "--input", generations, "--strict").returncode == 0
    manifest = json.loads((tmp_path / "smoke/run_manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["completed_samples"] > 0
    original = generations.read_bytes()
    assert run_script("run_inference.py", "--output-dir", tmp_path, "--run-id", "smoke").returncode != 0
    assert generations.read_bytes() == original
    checkpoints, selected = tmp_path / "checkpoints.jsonl", tmp_path / "selected.jsonl"
    built = run_script("build_checkpoints.py", "--input", generations, "--output", checkpoints, "--no-tokenizer")
    assert built.returncode == 0, built.stderr
    checked = run_script("validate_checkpoints.py", "--generations", generations, "--checkpoints", checkpoints,
                         "--report", tmp_path / "validation.json")
    assert checked.returncode == 0, checked.stdout + checked.stderr
    flags = ("--input", checkpoints, "--strategy", "uniform", "--max-checkpoints", "3")
    assert run_script("select_checkpoints.py", *flags, "--output", selected).returncode == 0
    assert run_script("select_checkpoints.py", *flags, "--verify", selected).returncode == 0
    assert run_script("build_checkpoints.py", "--input", generations, "--output", checkpoints, "--no-tokenizer").returncode != 0


def test_missing_input_fails_without_creating_fake_outputs(tmp_path):
    result = run_script("run_inference.py", "--input", tmp_path / "missing.jsonl", "--output-dir", tmp_path / "out")
    assert result.returncode != 0
    assert not (tmp_path / "out").exists()


def test_prompt_think_truncation_preserves_reasoning_checkpoints():
    runner, _ = fake_qwen([10, 11], 2)
    record = generation_record(runner.generate_cot(sample()), "truncated")
    checkpoints, parsed = build_checkpoints(record)
    assert parsed.parse_status == "unclosed_think"
    assert parsed.final_answer is None
    assert len(checkpoints) > 1
    assert checkpoints[-1]["reasoning_prefix"] == "Left. Ahead."


@pytest.mark.parametrize("exception", [TypeError, ValueError])
def test_unsupported_presence_penalty_is_recorded(exception):
    runner, _ = fake_qwen([10, 151668, 20, 151645])
    original = runner.model.generate
    calls = []
    def generate(**kwargs):
        calls.append(kwargs)
        if "presence_penalty" in kwargs:
            raise exception("unused presence_penalty")
        return original(**kwargs)
    runner.model.generate = generate
    result = runner.generate_cot(sample())
    assert len(calls) == 2
    assert result.generation_config["presence_penalty_applied"] is False


def test_load_pins_processor_to_resolved_model_snapshot(monkeypatch):
    cfg = InferenceConfig()
    cfg.model.revision = "main"
    calls, frozen = {}, []
    model = SimpleNamespace(
        config=SimpleNamespace(_commit_hash="b" * 40),
        eval=lambda: calls.update(eval=True),
        parameters=lambda: [SimpleNamespace(requires_grad_=lambda value: frozen.append(value))],
    )
    def load_model(name, **kwargs):
        calls["model"] = kwargs
        return model
    def load_processor(name, **kwargs):
        calls["processor"] = kwargs
        return SimpleNamespace()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(bfloat16="bf16", float16="fp16", float32="fp32"))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoProcessor=SimpleNamespace(from_pretrained=load_processor),
        Qwen3VLForConditionalGeneration=SimpleNamespace(from_pretrained=load_model),
        set_seed=lambda seed: calls.update(seed=seed),
    ))
    runner = Qwen3VLRunner(cfg)
    runner.load()
    assert calls["model"]["revision"] == "main"
    assert calls["processor"]["revision"] == "b" * 40
    assert runner.model_revision == "b" * 40
    assert calls["eval"] and frozen == [False]


def test_synthetic_label_and_evaluation_cli(tmp_path):
    checkpoints = ROOT / "tests/fixtures/synthetic_checkpoints.jsonl"
    labels = tmp_path / "labels.jsonl"
    built = run_script("build_checkpoint_labels.py", "--checkpoints", checkpoints, "--output", labels)
    assert built.returncode == 0, built.stderr
    validated = run_script("validate_checkpoint_labels.py", "--checkpoints", checkpoints,
                           "--labels", labels, "--report", tmp_path / "labels_validation.json")
    assert validated.returncode == 0, validated.stdout + validated.stderr
    evaluated = run_script("evaluate_early_exit.py", "--checkpoints", checkpoints, "--labels", labels,
                           "--output", tmp_path / "metrics.json", "--threshold-out", tmp_path / "threshold.json")
    assert evaluated.returncode == 0, evaluated.stderr
    assert json.loads((tmp_path / "metrics.json").read_text())["data_source"] == "synthetic"


def test_mixed_string_integer_ids_in_labeling_and_evaluation():
    from thinkenough.evaluation.checkpoint_labeling import build_labels
    from thinkenough.evaluation.early_exit_evaluator import check_same_question_set
    records = []
    references = {}
    for qid in (1, "scene-A"):
        result = MockRunner(InferenceConfig(repo_root=ROOT)).generate_cot(sample(qid))
        checkpoints, _ = build_checkpoints(generation_record(result, "mixed"))
        records.extend(checkpoints)
        references[qid] = {"split": "validation", "reference_answers": ["yes"]}
    labels, _ = build_labels(records, [], references)
    assert {r["question_id"] for r in labels} == {1, "scene-A"}
    groups = {"full_cot": [{"question_id": 1}, {"question_id": "scene-A"}],
              "probe": [{"question_id": "scene-A"}, {"question_id": 1}]}
    assert check_same_question_set(groups, "full_cot") == [1, "scene-A"]
