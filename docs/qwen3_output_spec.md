> 데이터팀 ZIP에서 가져온 원문입니다. 아래 실행 결과·설정·설치 설명은 작성 당시 기록이며 현재 통합본의 검증 결과가 아닙니다. 현재 실행 방법과 차이는 [integration.md](integration.md)를 따릅니다.

# Qwen3 생성 결과 요청 명세 (모델 담당 A → Dataset & Evaluation)

- 작성일: 2026-10-16
- 작성: Dataset & Evaluation 담당
- 목적: Qwen3 Long CoT 생성 결과를 checkpoint 분할(`scripts/build_checkpoints.py`)과 hidden state 정렬에 바로 쓸 수 있도록, 원본 출력 형식을 미리 맞춰둔다.
- 상태: **초안 (A 담당 검토 필요)**. 특히 4절 hidden state 위치 규칙은 협의해서 확정해야 한다.

---

## 1. 파일 형식

- **JSONL**: 한 줄 = 한 문항의 생성 결과 1개
- 인코딩 UTF-8, 파일 이름 예: `generations_{split}_{run_id}.jsonl`
- 같은 run에 대한 공통 설정은 별도 `run_config.json`으로 함께 전달해도 된다 (3절)
- **정답(`reference_answers`)은 넣지 않는다.** 채점은 Dataset & Evaluation 쪽에서 `question_id`로 연결해 따로 한다.

## 2. 레코드 필드

| 필드 | 타입 | 필수 | 설명 |
|---|---|:---:|---|
| `question_id` | int | ✅ | `data/processed/sqa3d/*.jsonl`의 `question_id` |
| `split` | str | ✅ | `train` / `validation` / `test` |
| `run_id` | str | ✅ | 생성 실행 식별자 (설정이 바뀌면 새 run_id) |
| `model` | str | ✅ | HF 모델 ID + revision(commit hash), 예: `Qwen/Qwen3-8B@<hash>` |
| `tokenizer` | str | ✅ | tokenizer ID + revision. 모델과 다르면 반드시 표기 |
| `quantization` | str | ✅ | `bf16` / `int8` / `nf4` 등 (hidden state 값에 영향) |
| `prompt_version` | str | ✅ | 예: `sqa3d_official_gpt_v0` (`src/thinkenough/data/sqa3d_prompt.py`) |
| `enable_thinking` | bool | ✅ | chat template의 thinking 설정 |
| `prompt_text` | str | ✅ | **chat template을 적용한 뒤** 실제로 모델에 들어간 전체 문자열 |
| `prompt_token_ids` | list[int] | ✅ | 실제 입력 토큰 ID (특수 토큰 포함) |
| `prompt_token_count` | int | ✅ | `len(prompt_token_ids)` |
| `generated_token_ids` | list[int] | ✅ | **모델이 생성한 원본 토큰 ID 그대로** (프롬프트 부분 제외, 특수 토큰 포함, 패딩 제거) |
| `generated_text` | str | ✅ | `tokenizer.decode(generated_token_ids, skip_special_tokens=False)` 결과 그대로 |
| `generated_token_count` | int | ✅ | `len(generated_token_ids)` |
| `reasoning_token_count` | int | ✅ | `<think>`(151667) 다음부터 `</think>`(151668) 전까지의 토큰 수. `</think>`가 없으면 null |
| `think_close_token_index` | int / null | ✅ | `generated_token_ids` 안에서 `</think>` 토큰의 위치. 없으면 null |
| `final_answer_text` | str / null | ✅ | `</think>` 이후 텍스트 (모델 출력 그대로, 후처리 전) |
| `finish_reason` | str | ✅ | `stop`(EOS로 종료) / `length`(max_new_tokens에 걸림) |
| `generation_config` | dict | ✅ | 3절 참고 (run_config.json으로 대신 전달 가능) |
| `hidden_state_ref` | dict / null | ⭕ | hidden state를 따로 저장했을 때 위치 정보 (4절) |
| `timing` | dict | ⭕ | `{"prefill_ms":..., "decode_ms":..., "tokens_per_s":...}` latency 지표용 |

**꼭 지켜야 할 것**
1. `generated_text`는 `generated_token_ids`를 **`skip_special_tokens=False`로** 디코딩한 결과와 정확히 같아야 한다. 텍스트를 다시 토큰화해서 ID를 만들면 안 된다. 다시 토큰화한 결과는 모델이 실제로 생성한 토큰과 다를 수 있다.
2. `generated_text`에 strip, 후처리, 답 추출을 하지 않는다. 원본 그대로 준다.
3. `<think>`가 생성 결과가 아니라 프롬프트(chat template) 쪽에 들어가는 모델이라면 그 사실을 `run_config.json`에 적는다. 이 경우 파서는 `missing_open_tag`로 기록하고 정상 처리한다.

## 3. 생성 설정 (`generation_config` 또는 `run_config.json`)

```json
{
  "do_sample": true,
  "temperature": 0.6,
  "top_p": 0.95,
  "top_k": 20,
  "min_p": 0.0,
  "max_new_tokens": 8192,
  "seed": 42,
  "repetition_penalty": 1.0,
  "batch_size": 1,
  "eos_token_ids": [151645, 151643],
  "library": "transformers==5.20.0 / torch==... / CUDA ...",
  "gpu": "RTX 5080 16GB",
  "think_tag_in_prompt": false
}
```
- 위 sampling 값은 예시다. Qwen3 모델 카드가 thinking 모드에 권장하는 값으로 알고 있지만, 실제 사용할 버전의 모델 카드에서 확인해야 한다.
- 문항마다 seed를 다르게 쓴다면 레코드마다 `seed`를 기록한다.

## 4. Hidden State와 연결하기 위한 위치 규칙 (협의 필요)

checkpoint 파일은 각 checkpoint에 아래 값을 기록한다.

| checkpoint 필드 | 의미 |
|---|---|
| `reasoning_token_start` | `generated_token_ids`에서 추론이 시작되는 토큰 index |
| `reasoning_token_end` | 이 checkpoint의 prefix를 덮는 마지막 토큰 다음 index (exclusive) |
| `hidden_state_position` | `prompt_token_count + reasoning_token_end - 1` = **전체 시퀀스(프롬프트+생성)** 안에서 prefix 마지막 토큰의 위치 |
| `alignment_status` | `exact` / `exact_up_to_whitespace` / `boundary_inside_token` / `no_token_ids` 등 |

**제안하는 hidden state 정의** 💭
- "위치 p의 hidden state" = 전체 시퀀스를 입력했을 때 **p번째 토큰을 읽은 직후** 각 layer의 출력
- 추출 방법(권장): 생성이 끝난 뒤 `prompt_token_ids + generated_token_ids`로 **forward pass를 한 번** 돌리고(`output_hidden_states=True`), 필요한 위치만 저장한다. `generate()` 도중의 hidden state는 KV cache 때문에 step마다 구조가 달라서, 위치를 맞추다 실수하기 쉽다.
- 저장할 layer, pooling 방식(마지막 토큰 / 평균), dtype은 A 담당과 정한다.

**`hidden_state_ref` 예시** (hidden state를 별도 파일로 저장할 때)
```json
{"file": "hidden/train_run01_000.safetensors", "key": "220602000182",
 "layers": [8, 16, 24, 32], "positions": "all_generated" , "dtype": "float16"}
```

## 5. 레코드 예시

```json
{"question_id": 220602000182, "split": "test", "run_id": "pilot_run01",
 "model": "Qwen/Qwen3-8B@<hash>", "tokenizer": "Qwen/Qwen3-8B@<hash>", "quantization": "nf4",
 "prompt_version": "sqa3d_official_gpt_v0", "enable_thinking": true,
 "prompt_text": "<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n",
 "prompt_token_ids": [151644, 872, ...], "prompt_token_count": 712,
 "generated_token_ids": [151667, 198, 785, ..., 151668, 271, 64899, 151645],
 "generated_text": "<think>\nThe refrigerator ...\n</think>\n\nbrown<|im_end|>",
 "generated_token_count": 1834, "reasoning_token_count": 1828, "think_close_token_index": 1830,
 "final_answer_text": "\n\nbrown<|im_end|>", "finish_reason": "stop",
 "generation_config": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_new_tokens": 8192, "seed": 42},
 "hidden_state_ref": null}
```
(숫자는 형식을 보여주기 위한 예시값이다.)

## 6. 받은 뒤 Dataset & Evaluation 쪽에서 하는 일

1. `generated_text == decode(generated_token_ids, skip_special_tokens=False)` 확인
2. `scripts/build_checkpoints.py --input <파일>` 실행 → checkpoint + 토큰 위치 계산
3. `scripts/validate_checkpoints.py`로 검증
4. 실제 출력 일부를 골라 checkpoint 경계 토큰과 hidden state 위치를 A 담당과 함께 직접 확인 (**실제 Qwen3 출력으로 하는 첫 정렬 검증**)

## 7. A 담당에게 확인할 질문

1. 사용할 정확한 모델 버전 (`Qwen3-8B` / thinking 전용 버전 등)과 `<think>`가 프롬프트 쪽에 들어가는지
2. 양자화 방식 (16GB GPU라 4-bit 예상)
3. `max_new_tokens`와 잘린 출력(`finish_reason=length`)을 어떻게 처리할지
4. hidden state 추출 방식 (생성 후 forward pass를 다시 돌리는 방식에 동의하는지), 저장할 layer와 용량
5. pilot 300문항 생성 일정
