# ThinkEnough

**When can long reasoning stop in 3D spatial question answering?**

> Can a frozen model's long reasoning about a 3D scene stop early without losing the answer, and how can we tell when?

ThinkEnough is a course project for **Practical Natural Language Processing**. We keep a reasoning model frozen, cut its reasoning every 256 tokens, ask for an answer at that point, and compare rules that decide when to stop.

Perception is out of scope. We assume other modules turn a scan into an object list, and we study the spatial reasoning that follows.

---

## Data

Questions, references, and splits come from [SCoT](https://huggingface.co/datasets/Orange-zZZ/SCoT) (ICLR 2026). The SCoT validation set is divided in half by scene into a tuning half and a final half. All results so far use the tuning half only.

| Task | Questions | Answer | Metrics |
|---|---:|---|---|
| Relationship Analysis | 100 | Distance and direction | Distance within 0.2 m (added by us), ROUGE-L, METEOR, SCoT scores |
| Implicit Detection | 100 | Object id | Acc@0.25, Acc@0.50 |
| Situated Planning | 50 | Free text | ROUGE-L, METEOR, SCoT scores |
| Un-situated Planning | 50 | Free text | ROUGE-L, METEOR, SCoT scores |

Relationship uses only the questions that ask for a distance.

Each question gets the Mask3D object list as text, scene video frames at 2 fps, a top-down scene image, and the question. Scene videos come from [VSI-Bench](https://huggingface.co/datasets/nyu-visionx/VSI-Bench), and object lists from [Chat-Scene](https://huggingface.co/ZzZZCHS/Chat-Scene).

## Model

`Qwen/Qwen3.5-9B`, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, with the model card's settings: temperature 1.0, top_p 0.95, top_k 20, presence_penalty 1.5, and up to 32,768 new tokens. `transformers` has no presence penalty, so we implement it as vLLM does.

## Method

```text
Object list + video + image + question
        |
        v
Frozen model, in three conditions: direct / thinking off / full thinking
        |
        v
Every 256 reasoning tokens: forced answer + hidden states of all layers
        |
        v
Stopping rules, evaluated on scenes they were not trained on
```

**Conditions:** `direct` answers right after the cue "Therefore, the answer is". `no_think` turns thinking off. `full` reasons until it finishes.

**Stopping rules:**

- Fixed token caps
- Answer stability: stop once the forced answer repeats 2 or 3 times
- First right answer: needs the true answer, so it only bounds accuracy
- Readiness detector: a linear classifier on the hidden state, its change, and the share of tokens used, following [arXiv 2609.34809](https://arxiv.org/abs/2609.34809)
- Variants of the detector that read tokens only, hidden states only, or text-based progress
- Evidence detector: reads whether the forced answer has settled and whether the reasoning so far backs it, with no hidden states

**Free-text grading:** SCoT's scoring prompt on a local judge, `unsloth/gemma-3-27b-it-bnb-4bit`, chosen from [arXiv 2510.09738](https://arxiv.org/abs/2510.09738). Scores are compared only within one judge. The judge also supports the RAGAS Answer Accuracy metric.

## Preliminary results

Tuning half, one seed. These are early numbers, not final results.

| Task | Direct | No think | Full |
|---|---:|---:|---:|
| Relationship, distance within 0.2 m | 21% | 22% | 88% |
| Implicit, Acc@0.25 | 21% | 19% | 31% |
| Situated, judge mean (1-10) | 4.20 | 7.19 | 8.6 |
| Un-situated, judge mean (1-10) | 5.64 | 7.37 | 8.8 |

**Known limits of these numbers:**

- They were produced before a fix in `to_scan_frame`, which left object sizes along the original axes. Rerunning Implicit with fixed sizes changed direct and no-think accuracy by less than repeat-run noise (25.5→23.0% and 15.0→12.0%, over two seeds).
- Implicit varies with the seed: direct gave 21% and 30%, and no-think 19% and 11%.
- Implicit uses a wider id parser than SCoT's evaluator, which gives 15%, 8%, and 19%.
- Answer parsers were corrected after results were seen. They never read the correct answer.
- The judge is not validated against human grading.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src pytest tests/ -q
```

This needs no GPU. To check the pipeline without model weights:

```bash
python scripts/run_inference.py --config configs/inference_default.yaml --backend mock
```

A real run needs a 24 GB GPU and the server dependencies listed in `requirements.txt`. METEOR needs Java.

## Running an experiment

```bash
# Build inputs (downloads SCoT data, frames, and scene images)
python scripts/build_scot_inputs.py --tasks implicit_detection --per-task 100 --mixed-order \
  --video-dir data/raw/frames_2fps --image-dir data/raw/scene_images --out data/processed/implicit.jsonl

# Three conditions
for mode in direct no_think full; do
  python scripts/run_inference.py --config configs/pilot.yaml --mode $mode --max-new-tokens 32768 \
    --input data/processed/implicit.jsonl --output-dir outputs/implicit/$mode
done

# Forced answers and hidden states along the full reasoning
python scripts/collect_checkpoints.py --config configs/pilot.yaml --input data/processed/implicit.jsonl \
  --runs outputs/implicit/full --output-dir outputs/implicit/checkpoints

# Stopping rules and detectors
python scripts/analyze_checkpoints.py --collected outputs/implicit/checkpoints --runs outputs/implicit/full \
  --detector --folds 5 --strengths 0.0001 0.001 0.01 0.1 1 --out outputs/implicit/stopping.json
```

For Relationship, add `--distance-only` when building inputs. For Planning, grade the answers and use the scores as labels:

```bash
python scripts/judge_answers.py --runs outputs/planning --out outputs/planning/judge.jsonl
python scripts/judge_answers.py --runs outputs/planning --forced outputs/planning/checkpoints/checkpoints.jsonl \
  --out outputs/planning/judge_forced.jsonl
python scripts/analyze_checkpoints.py --collected outputs/planning/checkpoints --runs outputs/planning/full \
  --official-labels outputs/planning/judge_forced.jsonl --margin 1 --score-drop 1 \
  --detector --folds 5 --strengths 0.0001 0.001 0.01 0.1 1 --out outputs/planning/stopping.json
```

To compare rules with thresholds picked from all training questions, including the evidence detector and forced-answer costs:

```bash
python scripts/compare_stopping_rules.py --collected outputs/implicit/checkpoints --runs outputs/implicit/full \
  --task implicit_detection --report outputs/implicit/stopping.json --out outputs/implicit/comparison.json
```

Runs resume from where they stopped. Use a new output folder when you change the question count.

## Repository

| Path | Purpose |
|---|---|
| `src/thinkenough/scot.py` | SCoT tasks, inputs, answer parsing, metrics |
| `src/thinkenough/inference/` | Model runners, including forced answers along a reasoning trace |
| `src/thinkenough/checkpoints.py` | Stopping rules and paired comparisons |
| `src/thinkenough/detector.py` | Readiness detector and scene-separated evaluation |
| `src/thinkenough/claims.py` | Checks spatial claims in reasoning against the object list |
| `src/thinkenough/evidence.py` | Features from the forced answer and the reasoning so far |
| `src/thinkenough/judge.py` | Scoring prompts for free-text answers |
| `scripts/` | Command-line steps above |
| `configs/pilot.yaml` | Settings for the experiments |

## Credits

Answer cleaning and the scoring prompt come from [SCoT](https://github.com/WHU-USI3DV/SCoT). The Answer Accuracy prompts come from [RAGAS](https://github.com/explodinggradients/ragas).
