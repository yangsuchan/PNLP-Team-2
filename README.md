# ThinkEnough

**Hidden-State Probing for Dynamic Early Exit in 3D-LLMs**

> Can a 3D-LLM know when it has reasoned enough?

ThinkEnough is a course project for **Practical Natural Language Processing** that explores whether a frozen 3D-LLM can dynamically stop Chain-of-Thought (CoT) reasoning once sufficient evidence has been accumulated.

Instead of retraining the base model with additional SFT or reinforcement learning, we train a lightweight probe on intermediate hidden states to predict whether the model can already produce the correct answer.

---

## Motivation

Recent 3D-LLMs use explicit Chain-of-Thought reasoning to solve complex spatial reasoning and planning tasks.

However, full CoT reasoning can increase:

- inference latency
- generated token cost
- redundant reasoning steps
- the risk of overthinking

Moreover, different 3D queries may require different amounts of reasoning.

This leads to our main question:

> **Can the hidden states of a frozen 3D-LLM reveal when its current reasoning is already sufficient to answer correctly?**

---

## Research Questions

### RQ1 — Reasoning Sufficiency

Do hidden states of a frozen 3D-LLM encode whether the current reasoning is sufficient to answer correctly?

### RQ2 — Efficient Early Exit

Can a lightweight probe use this signal to reduce reasoning cost while preserving task performance?

---

## Method Overview

The overall pipeline is:

```text
3D Scene + Question
        |
        v
Frozen Reasoning-capable 3D-LLM
        |
        v
Reasoning Checkpoint t
        |
        v
Hidden State h_t
        |
        v
ThinkEnough Probe
        |
   +----+----+
   |         |
CONTINUE    STOP
   |         |
   v         v
Next Step   Final Answer
```

---

## Model Inference (Phase 0)

Base model: `Qwen/Qwen3-VL-8B-Thinking` (frozen).  
Local machines (e.g. 8GB VRAM) should use the **mock** backend. Full model inference is intended for a GPU server (e.g. RTX 4090 24GB) after explicit approval.

### Setup (local, no model weights)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Mock inference (safe default)

```bash
python scripts/run_inference.py --config configs/inference_default.yaml --backend mock
```

Outputs are written under `outputs/` as per-sample JSON and a combined JSONL file.

### Unit tests

```bash
PYTHONPATH=src pytest tests/ -q
```

### GPU server checklist (do not run locally without approval)

1. Install server deps: `torch` (CUDA build), `transformers>=4.57.0`, `accelerate`
2. Confirm GPU / VRAM (`nvidia-smi`)
3. Download `Qwen/Qwen3-VL-8B-Thinking` weights
4. Run a single sample:

```bash
python scripts/run_inference.py \
  --config configs/inference_default.yaml \
  --backend qwen3_vl \
  --input data/samples/example_inputs.jsonl \
  --max-samples 1
```

5. Verify `</think>` / token id `151668` appears in outputs and that JSONL fields match the mock schema
