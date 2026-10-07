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
