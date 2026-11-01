"""Load and validate inference YAML configs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    """Base model loading settings (frozen inference only)."""

    model_id: str = "Qwen/Qwen3-VL-8B-Thinking"
    revision: str | None = None
    dtype: str = "bfloat16"
    device_map: str = "auto"
    attn_implementation: str | None = None


@dataclass
class GenerationConfig:
    """Decoding hyperparameters for Full CoT generation."""

    max_new_tokens: int = 40960
    do_sample: bool = True
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    seed: int = 1234

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "presence_penalty": self.presence_penalty,
            "seed": self.seed,
        }


@dataclass
class IOConfig:
    """Input / output path settings."""

    output_dir: str = "outputs"
    save_token_ids: bool = True
    input_jsonl: str = "data/samples/example_inputs.jsonl"


@dataclass
class ThinkingConfig:
    """Qwen3-VL Thinking special-token settings.

    Values come from the official tokenizer_config.json for
    Qwen/Qwen3-VL-8B-Thinking. Do not invent alternate markers.
    """

    think_start_token_id: int = 151667
    think_end_token_id: int = 151668
    think_end_text: str = "</think>"


@dataclass
class InferenceConfig:
    """Top-level inference configuration loaded from YAML."""

    backend: str = "mock"
    model: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    io: IOConfig = field(default_factory=IOConfig)
    thinking: ThinkingConfig = field(default_factory=ThinkingConfig)
    # Absolute repo root used to resolve relative paths.
    repo_root: Path = field(default_factory=lambda: Path.cwd())

    def resolve_path(self, path: str | Path) -> Path:
        """Resolve a path relative to the repository root if needed."""
        p = Path(path)
        if p.is_absolute():
            return p
        return (self.repo_root / p).resolve()


def _section(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"Config section '{key}' must be a mapping, got {type(value)}")
    return value


def load_inference_config(
    path: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> InferenceConfig:
    """Load an InferenceConfig from a YAML file.

    Args:
        path: Path to the YAML config file.
        repo_root: Optional repository root for relative path resolution.
            Defaults to the parent of `configs/` when the file lives there,
            otherwise the current working directory.

    Returns:
        Populated InferenceConfig.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        ValueError: If required structure is invalid.
    """
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ValueError("Inference config root must be a mapping")

    if repo_root is not None:
        root = Path(repo_root).resolve()
    elif config_path.parent.name == "configs":
        root = config_path.parent.parent.resolve()
    else:
        root = Path.cwd().resolve()

    backend = str(raw.get("backend", "mock"))
    if backend not in {"mock", "qwen3_vl"}:
        raise ValueError(
            f"Unsupported backend '{backend}'. Expected 'mock' or 'qwen3_vl'."
        )

    return InferenceConfig(
        backend=backend,
        model=ModelConfig(**_section(raw, "model")),
        generation=GenerationConfig(**_section(raw, "generation")),
        io=IOConfig(**_section(raw, "io")),
        thinking=ThinkingConfig(**_section(raw, "thinking")),
        repo_root=root,
    )
