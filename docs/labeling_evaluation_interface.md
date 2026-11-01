> 데이터팀 ZIP에서 가져온 원문입니다. 아래 실행 결과·설정·설치 설명은 작성 당시 기록이며 현재 통합본의 검증 결과가 아닙니다. 현재 실행 방법과 차이는 [integration.md](integration.md)를 따릅니다.

# 라벨링·평가 인터페이스 명세 (모델 담당 A ↔ Dataset & Evaluation)

- 작성일: 2026-10-16
- 작성: Dataset & Evaluation 담당
- 상태: **초안. 월요일 협의 후 확정**
- 관계: 기존 [`qwen3_output_spec.md`](qwen3_output_spec.md)(Full CoT 생성 결과 형식)는 그대로 두고, 그 위에 **강제 답변**, **방식별 결과**, **probe 예측** 형식을 추가한다.

```
[A] Full CoT 생성 (qwen3_output_spec.md)
      ↓ build_checkpoints.py
[D&E] checkpoint 파일 (checkpoint_id 포함)
      ↓ A에게 전달
[A] checkpoint마다 강제 답변 생성 (1절)
      ↓ build_checkpoint_labels.py
[D&E] STOP/CONTINUE 라벨 → [A] probe 학습 → probe 예측 (4절)
      ↓ evaluate_early_exit.py
[A] 방식별 온라인 실행 결과 (5절) → [D&E] 평가
```

---

## 1. Checkpoint별 강제 답변 결과 형식

A 담당은 checkpoint 파일의 각 `checkpoint_id`에 대해 "추론을 여기서 멈추고 답하라"는 생성을 수행한다.

**강제 답변 입력 구성 (제안 💭, 협의 필요)**
```
<원래 prompt_text (chat template 적용)>
<think>\n{reasoning_prefix}\n</think>\n\n     ← 모델이 여기서부터 최종 답을 생성
```
- **Checkpoint 0**: `<think>\n\n</think>\n\n` (추론 없이 바로 답). Qwen3의 non-thinking 형식과 같다.
- 권장 설정: `do_sample=False`(greedy), `max_new_tokens=32`. 여러 번 샘플링하면 `sample_index`를 0, 1, 2, ...로 매긴다.
- prefix는 checkpoint 파일의 `reasoning_prefix`를 그대로 쓰거나, `generated_token_ids[reasoning_token_start:reasoning_token_end]`를 그대로 이어 붙인다. **토큰 ID를 이어 붙이는 방식을 권장**한다. 텍스트를 다시 토큰화하면 원래 생성과 달라질 수 있다.

**레코드 형식** (`forced_answers_{split}_{run_id}.jsonl`, checkpoint 1개 × 샘플 1개 = 1줄)

| 필드 | 필수 | 설명 |
|---|:---:|---|
| `checkpoint_id` | ✅ | checkpoint 파일의 값 그대로 (`{run_id}/{question_id}/{segmenter}/{index}`) |
| `run_id` | ✅ | Full CoT 생성 run |
| `question_id`, `split`, `checkpoint_index` | ✅ | checkpoint와 같아야 한다 (다르면 라벨링이 오류로 멈춤) |
| `sample_index` | ✅ | 같은 checkpoint의 몇 번째 생성인지 (0부터) |
| `answer_run_id` | ✅ | 강제 답변 생성 실행 식별자 |
| `generated_answer` | ✅ | `</think>` 이후 모델이 생성한 텍스트 **원문** (특수 토큰 포함, 후처리 없음). 실패하면 null |
| `generation_status` | ✅ | `ok` / `length`(32토큰에 걸림) / `error` / `not_run` / `timeout` / `oom` |
| `model`, `prompt_version`, `answer_prompt_version` | ✅ | 모델 revision, 원래 프롬프트 버전, 강제 답변 형식 버전 |
| `generation_config` | ✅ | `do_sample`, `temperature`, `seed`, `max_new_tokens` 등 |
| `answer_token_ids` | ⭕ | 생성한 답의 토큰 ID |
| `answer_token_count` | ⭕ | 답 토큰 수. **실측값만** 넣고, 없으면 null |
| `answer_token_count_source` | ⭕ | `generated_ids`(실측) |
| `answer_latency_ms` | ⭕ | 실측한 경우에만 |
| `synthetic` | ✅ | 실제 실행이면 `false` |

예시:
```json
{"checkpoint_id": "pilot_run01/220602000182/sentence/3", "run_id": "pilot_run01",
 "question_id": 220602000182, "split": "train", "checkpoint_index": 3, "sample_index": 0,
 "answer_run_id": "force_run01", "generated_answer": "\n\nbrown<|im_end|>", "generation_status": "ok",
 "model": "Qwen/Qwen3-8B@<hash>", "prompt_version": "sqa3d_official_gpt_v0", "answer_prompt_version": "force_answer_v0",
 "generation_config": {"do_sample": false, "max_new_tokens": 32, "seed": 42},
 "answer_token_ids": [64899, 151645], "answer_token_count": 2, "answer_token_count_source": "generated_ids",
 "answer_latency_ms": 41.3, "synthetic": false}
```

## 2. STOP/CONTINUE 라벨에 필요한 정보

| 정보 | 출처 |
|---|---|
| checkpoint (`checkpoint_id`, prefix, 토큰 위치) | D&E (`build_checkpoints.py`) |
| 강제 답변 원문과 생성 상태 | A (1절) |
| 정답 | D&E (`data/processed/sqa3d/*.jsonl`의 `reference_answers`) |

- **라벨 = 정규화 EM**(정답이면 STOP=1, 아니면 CONTINUE=0). 공식 매칭 결과는 따로 기록한다.
- 답변 레코드가 없으면 `pending`, 생성 실패면 `missing`이고 둘 다 학습에서 제외한다. 빈 답이나 형식 오류는 STOP=0으로 학습에 포함하고 flag를 붙인다.
- 정답은 A 담당 쪽 파일에 넣지 않는다. 라벨링은 D&E 쪽에서 한다.

## 3. 실제 생성 토큰 수 기록 방법

- 모든 토큰 수는 **모델이 실제로 생성한 토큰 ID의 개수**로 센다. 텍스트를 다시 토큰화해서 세지 않는다.
- Full CoT 생성 결과에서
  - `reasoning_tokens` = `<think>`(151667) 다음부터 `</think>`(151668) 전까지의 토큰 수
  - `answer_tokens` = `</think>` 다음부터 EOS(`<|im_end|>` 151645)까지의 토큰 수. EOS 포함 여부를 정해서 일관되게 쓴다(제안: 포함)
  - 태그 토큰(`<think>`, `</think>`)은 어느 쪽에도 넣지 않는다
- Early exit 실행에서는 실제로 생성한 추론 토큰 수를 쓴다. checkpoint 개수를 쓰면 안 된다.
- 시험 답 토큰(DEER 방식처럼 추론 도중 답을 생성하는 경우)은 `trial_answer_tokens`에 따로 기록한다.
- 라벨링용 강제 답변 토큰은 1절의 `answer_token_count`에 기록한다. **추론 비용과는 분리**한다.
- 모르는 값은 null로 둔다. 추정값을 넣을 때는 반드시 `token_count_source: "estimated"`로 표시한다.

## 4. Probe 예측 결과 형식

`probe_predictions_{split}_{probe_version}.jsonl`, checkpoint 1개 = 1줄:
```json
{"checkpoint_id": "pilot_run01/220602000182/sentence/3", "question_id": 220602000182, "split": "validation",
 "checkpoint_index": 3, "p_stop": 0.83, "probe_version": "linear_L24_v1",
 "probe_latency_ms": 0.21, "synthetic": false}
```
- train / validation / test를 **별도 파일**로 준다. probe는 train으로만 학습한다.
- threshold는 D&E가 validation 예측으로 고르고 고정 파일로 공유한다. test 예측을 보고 τ를 바꾸면 안 된다.

## 5. Full CoT와 Early Exit 비교용 결과 파일 구조

`results_{split}_{method}_{run_id}.jsonl`, 방식 1개 × 문항 1개 = 1줄. **모든 방식이 같은 question_id 집합**을 가져야 한다(다르면 평가가 오류로 멈춤).

| 필드 | 설명 |
|---|---|
| `method` | 이름 (예: `full_cot`, `direct`, `fixed_2`, `budget_512`, `probe_ee_tau0.55`) |
| `method_type` | `direct` / `fixed_k` / `fixed_budget` / `full_cot` / `probe_early_exit` / `trial_answer_early_exit` / `oracle` / `other` |
| `evaluation_mode` | `online` (실제 실행) |
| `question_id`, `split` | |
| `final_answer` | 모델 출력 원문 (`</think>` 이후 포함 전체도 가능, 추출은 D&E가 함) |
| `generation_status` | `ok` / `length` / `error` ... (실패해도 행을 빼지 말 것: 오답으로 센다) |
| `exit_checkpoint_index` | probe가 멈춘 checkpoint (해당 없으면 null) |
| `reasoning_tokens`, `answer_tokens`, `trial_answer_tokens` | 3절 기준 실측값 |
| `token_count_source` | `generated_ids` / `estimated` |
| `probe_calls` | 이 문항에서 probe를 호출한 횟수 |
| `latency` | `{"total_ms", "reasoning_ms", "answer_ms", "probe_ms"}` (6절) |
| `latency_source` | `measured` |
| `measurement` | 측정 조건 (6절) |
| `threshold_ref` | probe 방식이면 사용한 threshold 파일 이름과 해시 |
| `synthetic` | `false` |

## 6. 실제 GPU Latency 기록 기준

- 모든 방식을 **같은 GPU, 같은 설정, 같은 batch size(1 권장)**로 측정한다.
- 모델 로딩 시간은 제외하고, 처음 몇 문항은 warm-up으로 버린다(예: 5문항).
- 구간을 잴 때 `torch.cuda.synchronize()`를 호출한 뒤 `time.perf_counter()`로 잰다.
- 기록할 값
  - `total_ms`: 입력 시작부터 최종 답 EOS까지
  - `reasoning_ms`: 추론 토큰 생성 구간
  - `answer_ms`: 최종 답 생성 구간
  - `probe_ms`: hidden state 추출·전송 + probe 계산의 합
  - (가능하면) `prefill_ms`, `decode_ms`
- `measurement`에 기록할 조건: GPU 모델, 양자화, dtype, batch size, `transformers`/`torch`/CUDA 버전, `max_new_tokens`, KV cache 사용 여부, 동시에 실행 중인 다른 작업 여부

## 7. 지금 가능한 평가와 모델 실행이 필요한 평가

| 평가 | 현재 코드 | 필요한 것 |
|---|:---:|---|
| 정답 채점, 정확도, retention, 정확도 차이 | ✅ 구현 완료 | 방식별 결과 파일 (5절) |
| STOP/CONTINUE 라벨 생성 | ✅ 구현 완료 | 강제 답변 파일 (1절) |
| 토큰 감소율 (실측) | ✅ 구현 완료 | 실측 토큰 수 (3절) |
| Latency 요약 | ✅ 구현 완료 | 실측 시간 (6절) |
| Probe 품질, threshold 선정 | ✅ 구현 완료 | probe 예측 (4절) |
| 오프라인 early exit 시뮬레이션 | ✅ 구현 완료 | 라벨 + probe 예측 |
| Hidden state 추출, probe 학습 | ❌ (A 담당) | GPU |
| 온라인 early exit (생성 루프에서 멈추기) | ❌ (A 담당) | probe를 생성 루프에 연결 |

## 8. 예외 상황 처리

| 상황 | 처리 |
|---|---|
| Checkpoint 0 | 빈 think 블록 뒤에 답 생성 → 일반 checkpoint와 같은 방식으로 라벨링 |
| 강제 답변 생성 실패 | `generation_status`에 사유 기록, `generated_answer: null` → 라벨 `missing` (학습 제외) |
| 아직 생성하지 않은 checkpoint | 레코드를 만들지 않음 → 라벨 `pending` |
| 답이 32토큰에 걸림 | `generation_status: "length"` → 채점하되 flag |
| Full CoT가 잘림 (`</think>` 없음) | checkpoint `parse_status: unclosed_think`. 라벨링은 가능하지만 Full CoT 최종 답이 없으므로, 평가에서는 생성 실패로 오답 처리 |
| 모델이 답 대신 다시 생각을 시작 (`<think>` 출력) | 답 추출 `think_unclosed` → STOP=0 + `invalid_format` |
| checkpoint_id 불일치나 중복 | 라벨링과 평가 모두 오류로 멈춤 → 파일을 다시 생성 |

## 9. 월요일 협의 안건

1. 강제 답변 입력 구성(1절)과 greedy 여부, `max_new_tokens`
2. prefix를 토큰 ID로 이어 붙일지 텍스트로 넣을지
3. 토큰 수의 EOS 포함 여부
4. hidden state 위치 규칙 (`qwen3_output_spec.md` 4절)과 probe 입력 layer
5. Pilot 실행 범위: train 300문항 × checkpoint 수 → 강제 답변 생성 횟수와 예상 GPU 시간
6. 온라인 early exit 구현 방식과 latency 측정 담당
