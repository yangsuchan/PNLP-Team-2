> 데이터팀 ZIP에서 가져온 원문입니다. 아래 실행 결과·설정·설치 설명은 작성 당시 기록이며 현재 통합본의 검증 결과가 아닙니다. 현재 실행 방법과 차이는 [integration.md](integration.md)를 따릅니다.

# Reasoning Checkpoint Report

- 작성일: 2026-10-16 (같은 날 4·5단계 작업에 맞춰 갱신: `checkpoint_id`·`run_id` 추가, 가상 데이터 split 분배, 검증 12개 항목)
- 담당: Dataset & Evaluation
- 범위: Qwen3 Long CoT를 문장이나 단계 단위 checkpoint로 나누고, 각 지점의 누적 reasoning prefix를 만드는 코드
- 이번 단계에서 **하지 않은 것**: Qwen3 실행, GPU 사용, hidden state 추출, STOP/CONTINUE 라벨 생성, 유료 API
- 사용한 데이터: **가상 CoT 10개**(`tests/fixtures/fake_qwen3_outputs.jsonl`, 모두 `"synthetic": true`). 실제 SQA3D 원본과 전처리 결과는 읽지도, 수정하지도 않았다.

> ⚠️ **토큰 위치는 아직 실제 Qwen3 출력으로 검증하지 않았다.** 정렬 기능은 가상 텍스트를 tokenizer로 다시 토큰화한 **합성 토큰 ID**로만 테스트했다. 실제 생성 토큰 ID로 hidden state 위치가 맞는지는 A 담당의 출력을 받은 뒤 반드시 다시 확인해야 한다 (6절).

재현 방법:
```bash
.venv/bin/python -m pytest tests/ -v                 # 단위 테스트 88개 (tests/ 전체는 147개)
.venv/bin/python scripts/build_checkpoints.py        # 예시 checkpoint 생성
.venv/bin/python scripts/validate_checkpoints.py     # 검증 12개 항목
```

---

## 1. 생성 파일

| 파일 | 역할 |
|---|---|
| `src/thinkenough/data/reasoning_checkpoint.py` | 핵심 로직: `<think>` 파싱, 분할기, checkpoint 생성, 토큰 정렬 |
| `scripts/build_checkpoints.py` | 생성 결과 JSONL → checkpoint JSONL |
| `scripts/validate_checkpoints.py` | checkpoint 결과 검증 → `outputs/checkpoint_validation.json` |
| `tests/test_reasoning_checkpoint.py` | 단위 테스트 88개 |
| `tests/fixtures/fake_qwen3_outputs.jsonl` | 가상 CoT 10개 (question_id 9000000000xx, `run_id: synthetic_run01`, split: train 4 / validation 3 / test 3) |
| `data/processed/checkpoints/example_checkpoints.jsonl` | 출력 예시 (checkpoint 35개, git 제외) |
| `outputs/checkpoint_validation.json` | 검증 결과 (git 제외) |
| `docs/qwen3_output_spec.md` | A 담당에게 요청할 원본 출력 형식 |
| `docs/checkpoint_report.md` | 이 보고서 |

가상환경에 `pytest`를 추가로 설치했다. 기존 파일은 수정하지 않았다.

---

## 2. 처리 흐름

```
생성 결과 1개 (generated_text, 있으면 generated_token_ids)
  │
  ├─ ① parse_generation()   <think> 추론 / 최종 답 분리, 이상 형식이면 parse_status 기록
  ├─ ② SEGMENTERS[...]      추론 텍스트를 (시작, 끝) 글자 구간 목록으로 나눔
  ├─ ③ build_checkpoints()  checkpoint 0(추론 없음) + 조각마다 1개, 누적 prefix
  └─ ④ QwenTokenAligner     원본 토큰 ID가 있을 때만 토큰 위치 계산, 없으면 null
```

### 2.1 함수별 설명

| 함수 / 클래스 | 하는 일 |
|---|---|
| `parse_generation(text)` | 생성 결과에서 `<think>`와 `</think>` 사이의 추론 텍스트와 그 뒤의 최종 답을 분리한다. 형식이 이상해도 멈추지 않고 `parse_status`에 사유를 남긴다 |
| `segment_sentences(text)` | **기본 분할기.** 문장 끝(`. ! ?` 뒤 공백)과 줄바꿈에서 자른다 |
| `segment_lines(text)` | 줄바꿈마다 자른다 |
| `segment_numbered_steps(text)` | `1.`, `2)`, `Step 3:`, `- ` 같은 단계 표시가 있는 줄 앞에서 자른다 |
| `SEGMENTERS` | 분할기 목록. 새 방식을 추가하려면 함수를 만들어 여기에 등록하면 된다 |
| `build_checkpoints(record, ...)` | 한 문항의 checkpoint 목록을 만든다. 정답은 입력으로 받지 않는다 |
| `QwenTokenAligner` | 원본 토큰 ID마다 텍스트의 몇 번째 글자부터 몇 번째 글자까지인지 계산한다 (byte 단위라서 여러 토큰에 걸친 글자도 처리) |
| `align_boundary(offsets, pos)` | "글자 위치 → 토큰 번호" 변환. 경계가 토큰 중간에 걸리는지도 알려준다 |

### 2.2 `<think>` 파싱 규칙

| 출력 형태 | parse_status | 추론 | 최종 답 |
|---|---|---|---|
| `<think> R </think> A` | `ok` | R | A |
| `R </think> A` | `missing_open_tag` | R | A |
| `<think> R` (잘림) | `unclosed_think` | R | 없음 |
| `A` (태그 없음) | `no_think_tags` | 없음 | A |
| `<think> </think> A` | `empty_reasoning` | 없음 | A |
| 빈 출력 | `empty_output` | 없음 | 없음 |
| think 블록이 2개 이상 | (위 상태 중 하나) + 경고 `multiple_think_blocks` | 첫 블록 | 첫 `</think>` 이후 |

추론 텍스트는 앞뒤 공백을 제거하고, 원본에서 시작하는 위치를 `reasoning_char_offset`에 기록한다.

### 2.3 문장 분할 규칙 (`sentence/v1`)

- **자르는 곳**: `. ! ?`(뒤에 닫는 따옴표나 괄호가 와도 됨) 다음에 공백이 올 때, 줄바꿈
- **자르지 않는 곳**:
  - 소수점 `3.5 m`, `1.2`, `2.9` (마침표 바로 뒤에 공백이 없으므로)
  - 약어 `approx.`, `e.g.`, `i.e.`, `vs.`, `Fig.`, `Dr.`
  - 줄 맨 앞의 번호 `1.`
  - 마침표 뒤 단어가 소문자로 시작할 때 (`Hmm... then`, `approx. two`)
- 공백과 줄바꿈은 **앞 조각에 붙인다.** 그래서 모든 조각을 이어 붙이면 원본과 정확히 같다.

### 2.4 Checkpoint 레코드

| 필드 | 설명 |
|---|---|
| `checkpoint_id` | checkpoint 고유 식별자 `{run_id}/{question_id}/{segmenter}/{index}`. 강제 답변·라벨·probe 예측을 이 값으로 연결한다 |
| `run_id` | 이 checkpoint를 만든 Full CoT 생성 run (입력에 없으면 `norun`) |
| `question_id`, `split` | 문항 식별 |
| `checkpoint_index` | 0 = 추론 없음(Direct Answer 기준선용), 1부터는 조각 순서 |
| `checkpoint_type` | `no_reasoning` / `sentence` / `line` / `numbered_step` |
| `segment_text` | 이번 checkpoint에서 새로 추가된 조각 |
| `reasoning_prefix` | 처음부터 이 지점까지 누적된 추론 (`--no-prefix-text`이면 null) |
| `char_start`, `char_end` | 추론 텍스트 기준 prefix의 글자 범위 (`char_start`는 항상 0) |
| `segment_char_start` | 이번 조각의 시작 글자 위치 |
| `prefix_token_count` | prefix의 토큰 수 |
| `token_count_source` | `generated_ids`(원본 토큰 기준, 정확) / `retokenized`(prefix를 다시 토큰화, 근사치) |
| `reasoning_token_start`, `reasoning_token_end` | 원본 생성 토큰 기준 범위. **원본 토큰 ID가 없으면 null** |
| `hidden_state_position` | `prompt_token_count + reasoning_token_end - 1`. 정보가 없으면 null |
| `alignment_status` | 아래 표 참고 |
| `alignment_char_end` | 토큰 정렬에 쓴 원본 기준 글자 위치 (prefix 끝의 공백 제외) |
| `parse_status`, `parse_warnings` | 파싱 결과 |
| `segmenter`, `source` | 분할 방식과 버전, 모델·tokenizer·프롬프트 버전, `synthetic` 여부 |

**alignment_status 값**

| 값 | 의미 |
|---|---|
| `no_token_ids` | 원본 토큰 ID가 없어서 토큰 위치를 null로 둠 |
| `exact` | checkpoint 경계가 토큰 경계와 정확히 일치 |
| `exact_up_to_whitespace` | 경계가 토큰 중간이지만, 그 토큰에 더 들어 있는 것이 공백·줄바꿈뿐 (예: Qwen3는 `.\n\n`을 한 토큰으로 만든다) |
| `boundary_inside_token` | 마지막 토큰에 **다음 조각의 글자까지** 들어 있음. hidden state가 다음 내용을 일부 본 상태라 주의해야 함 |
| `start_inside_token` | 추론 시작 위치가 토큰 중간 |
| `text_mismatch` / `decode_failed` / `no_aligner` | 토큰 ID와 `generated_text`가 서로 맞지 않는 등의 이유로 정렬하지 않음 (위치는 null) |

---

## 3. 구현 중 발견하고 수정한 문제

**Qwen3 토큰화와 문장 경계**

처음에는 prefix 끝(공백 포함)을 기준으로 토큰을 맞췄다. 그랬더니 `"... the door. So the door ..."`에서 checkpoint의 마지막 토큰이 **다음 문장의 `' So'`**가 되었다. Qwen3 tokenizer는 공백을 다음 단어에 붙이기 때문이다(`' So'`가 한 토큰). 이대로 두면 hidden state를 다음 문장의 첫 단어까지 읽은 위치에서 추출하게 된다.

- 수정: 토큰 정렬은 **공백을 뺀 prefix 내용의 끝**(`alignment_char_end`)을 기준으로 한다. prefix 텍스트에는 원본 보존을 위해 공백을 그대로 둔다.
- 결과: 합성 토큰 테스트에서 모든 checkpoint의 마지막 토큰이 해당 문장의 마침표 토큰(`.` 또는 `.\n\n`)이 되었다. `boundary_inside_token`은 0건이다.

---

## 4. 테스트 결과

### 4.1 단위 테스트: 88개 통과, 0개 실패, 0개 건너뜀 (`pytest tests/test_reasoning_checkpoint.py`)

| 요청한 테스트 경우 | 테스트 | 결과 |
|---|---|:---:|
| 정상적인 여러 문장 CoT | `test_multi_sentence` (5문장, `!` 포함) | ✅ |
| 한 문장 CoT | `test_single_sentence` | ✅ |
| 줄바꿈과 번호가 있는 추론 | `test_newlines_and_numbered_steps` (분할기 3종 모두) | ✅ |
| 숫자와 소수점 | `test_decimals_and_abbreviations_are_not_boundaries` (`3.5`, `1.2`, `2.9`, `approx.`, `e.g.`, `Fig.`) | ✅ |
| 문장 구분이 불명확 | `test_unclear_boundaries_stay_one_segment` (구두점 없는 긴 문장 → 1조각) | ✅ |
| 빈 추론 | `test_parse_status[empty_reasoning]`, `test_no_reasoning_gives_only_checkpoint_zero` | ✅ |
| `<think>` 누락 | `test_parse_status[missing_open_tag]`, `test_missing_open_tag_still_builds_checkpoints` | ✅ |
| 이어 붙이면 원본 보존 | `test_segments_join_back_to_original` (10개 경우 × 분할기 3종 = 30개) | ✅ |
| prefix 정확히 누적 | `test_checkpoint_prefixes_accumulate` (30개): `prefix[k] == prefix[k-1] + segment[k]`, `prefix[k] == reasoning[:char_end]`, 마지막 prefix == 전체 추론 | ✅ |

**추가 테스트**
- 잘린 출력(`unclosed_think`), 빈 출력, None 입력, think 블록 여러 개
- 원본 토큰 ID가 없으면 토큰 위치가 전부 null인지
- 정답을 넣어도 결과가 똑같은지 (정답 미사용 확인)
- `--no-prefix-text` 모드
- 다시 토큰화한 토큰 수가 증가하는 방향으로만 변하는지
- **합성 토큰 ID 정렬** (4개 경우): 위치가 다음 조각을 침범하지 않는지, `hidden_state_position` 계산이 맞는지
- `.\n\n` 병합 토큰이 `exact_up_to_whitespace`로 표시되는지
- 다른 텍스트의 토큰 ID를 넣으면 정렬을 거부(`text_mismatch`)하는지

### 4.2 검증 스크립트: 12개 항목 통과 (`checkpoint_id` 중복 검사 포함)

| 분할기 | checkpoint 수 (10문항) | 검증 결과 |
|---|---:|---|
| `sentence` (기본, 예시 파일) | 35 | 12/12 통과 |
| `line` | 23 | 12/12 통과 |
| `numbered_step` | 21 | 12/12 통과 |
| `sentence` + `--no-prefix-text` | 35 | 12/12 통과 |

예시 파일의 parse_status 분포: ok 6, empty_reasoning 1, missing_open_tag 1, no_think_tags 1, unclosed_think 1

---

## 5. 설계상 결정 사항

1. **Checkpoint 0 (추론 없음)을 포함한다.** 제안서의 Depth 0 / Direct Answer 기준선과 맞추기 위해서다. 추론이 없는 출력은 checkpoint 0만 생긴다.
2. **토큰 위치는 지어내지 않는다.** 원본 생성 토큰 ID가 없으면 `reasoning_token_start/end`와 `hidden_state_position`은 null이다. 토큰 수만 다시 토큰화한 근사치로 채우고 `retokenized`로 표시한다.
3. **정답은 이 단계의 입력이 아니다.** `build_checkpoints.py`는 입력 레코드에 정답 관련 키가 있으면 처리 전에 지운다. 테스트로도 정답이 결과에 영향을 주지 않음을 확인했다.

---

## 6. 실제 Qwen3와 연결할 때 남은 문제

1. **토큰 정렬 실검증 (가장 중요)**: 실제 생성 토큰은 텍스트를 다시 토큰화한 결과와 다를 수 있다(모델이 같은 텍스트를 다른 토큰 조합으로 생성하는 경우). 정렬기는 원본 ID를 직접 쓰도록 만들었지만, **실제 출력 몇 개로 checkpoint 경계 토큰과 hidden state 위치를 A 담당과 함께 직접 확인해야 한다.**
2. **hidden state 위치 규칙 합의**: `hidden_state_position = prompt_token_count + reasoning_token_end - 1`로 정의했다. 생성 후 forward pass를 다시 돌려서 추출하는 방식을 제안했다 (`docs/qwen3_output_spec.md` 4절).
3. **checkpoint 수가 많다**: Long CoT는 수천 토큰, 문장 수백 개가 될 수 있다. checkpoint마다 답을 강제로 생성해서 라벨을 만들면 비용이 문장 수에 비례해 커진다. 짧은 조각 합치기(`min_segment_tokens`), 일정 간격으로만 checkpoint 사용하기, 생각 전환 표현("Wait,", "Hmm,") 기준 분할 같은 방법을 검토해야 한다. 분할기 목록에 추가할 자리는 마련해 두었다.
4. **Qwen3 추론 문체**: 실제 CoT에는 수식, 코드, 목록, 괄호 속 문장, 따옴표가 섞일 수 있다. 약어 목록도 영어 기준이다. 실제 출력에서 분할 품질을 표본 검사해야 한다.
5. **잘린 출력(`unclosed_think`) 처리**: 최종 답이 없으므로 라벨링과 평가에서 제외할지, 강제로 답을 생성할지 정해야 한다.
6. **`<think>` 위치는 모델 버전에 따라 다르다**: chat template이 `<think>`를 프롬프트에 넣는 버전이면 모든 출력이 `missing_open_tag`가 된다. 정상 처리되지만, 사용할 모델 버전을 확정해야 한다.
7. **최종 답 추출 규칙**: `</think>` 이후 텍스트에서 짧은 답을 뽑는 규칙(채점용)은 아직 구현하지 않았다.
8. **다시 토큰화한 토큰 수는 근사치다**: 원본 토큰 ID가 오면 `generated_ids` 기준의 정확한 값으로 바뀐다.
