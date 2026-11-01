> 데이터팀 ZIP에서 가져온 원문입니다. 아래 실행 결과·설정·설치 설명은 작성 당시 기록이며 현재 통합본의 검증 결과가 아닙니다. 현재 실행 방법과 차이는 [integration.md](integration.md)를 따릅니다.

# STOP/CONTINUE Labeling Report

- 작성일: 2026-10-16
- 담당: Dataset & Evaluation
- 범위: 4단계. reasoning checkpoint마다 모델이 강제로 낸 중간 답을 SQA3D 정답과 비교해서 probe 학습용 정답 라벨(STOP/CONTINUE)을 만든다.
- 사용 데이터: **가상 데이터만** (`tests/fixtures/`, 모든 레코드 `"synthetic": true`, question_id 9000000000xx). 실제 Qwen3 추론, GPU, 유료 API는 사용하지 않았다.

> ⚠️ 아래 라벨과 수치는 코드 동작을 확인하기 위한 **가상 예시**이며, SQA3D/Qwen3 실험 결과가 아니다.

재현 방법:
```bash
.venv/bin/python scripts/build_checkpoint_labels.py       # 라벨 생성
.venv/bin/python scripts/validate_checkpoint_labels.py    # 라벨 검증 (10개 항목)
.venv/bin/python -m pytest tests/test_checkpoint_labeling.py -v   # 테스트 29개
```

---

## 1. 생성·수정 파일

| 파일 | 역할 |
|---|---|
| `src/thinkenough/evaluation/answer_extraction.py` *(신규)* | 모델 출력에서 생각 과정을 빼고 최종 답만 추출. 라벨링과 평가가 같은 규칙을 쓴다 |
| `src/thinkenough/evaluation/checkpoint_labeling.py` *(신규)* | 라벨 생성 로직 |
| `scripts/build_checkpoint_labels.py` *(신규)* | checkpoint + 중간 답변 + 정답 → 라벨 JSONL |
| `scripts/validate_checkpoint_labels.py` *(신규)* | 라벨 검증 → `outputs/checkpoint_label_validation.json` |
| `tests/test_checkpoint_labeling.py` *(신규)* | 테스트 29개 |
| `tests/fixtures/fake_checkpoint_answers.jsonl` *(신규)* | 가상 중간 답변 30개 (그중 1개는 일부러 넣은 고아 레코드) |
| `tests/fixtures/fake_references.jsonl` *(신규)* | 가상 정답 (실제 SQA3D 정답과 분리) |
| `data/processed/checkpoint_labels/example_labels.jsonl` | 라벨 예시 36행 (git 제외) |
| `src/thinkenough/data/reasoning_checkpoint.py` *(수정)* | checkpoint에 `checkpoint_id`, `run_id` 필드 **추가만** 함 |
| `scripts/validate_checkpoints.py` *(수정)* | `checkpoint_id` 필수 필드 + 중복 검사 추가 (11 → 12개 항목) |
| `tests/fixtures/fake_qwen3_outputs.jsonl` *(수정)* | `run_id` 추가, split을 train 4 / validation 3 / test 3으로 분배 |

SQA3D 원본과 전처리 결과는 수정하지 않았다(md5 확인).

---

## 2. 라벨 규칙

| 상황 | `label_status` | `stop_label` | 학습 사용 |
|---|---|:---:|:---:|
| 생성 완료 + 정규화 EM 정답 | `labeled` | **1 (STOP)** | ✅ |
| 생성 완료 + 오답 | `labeled` | **0 (CONTINUE)** | ✅ |
| 생성 완료 + 빈 답 / 형식 오류 | `labeled` + flag `empty_answer` / `invalid_format` | 0 | ✅ |
| 답이 최대 길이에 걸려 잘림 (`generation_status=length`) | `labeled` + flag `answer_truncated` | EM 결과 | ✅ |
| 답변 레코드 없음 (아직 생성 안 함) | `pending` | null | ❌ |
| 생성 실패 (`error` / `not_run` / `timeout` / `oom`) | `missing` | null | ❌ |
| 정답을 찾을 수 없음 | `no_reference` | null | ❌ |

- **checkpoint마다 독립적으로 판정한다.** 앞 checkpoint의 결과를 뒤로 전파하지 않는다. 테스트로 확인했다: `[0, 1, 0, 0, 1]` 같은 순서가 그대로 보존된다.
- **여러 번 생성한 경우** `sample_index`별로 행을 따로 두고 생성 설정(seed 등)도 함께 보존한다. 다수결 라벨은 만들지 않는다.
- **최종 답만 채점한다.** 생각 과정에 정답 단어가 나와도 라벨에 영향이 없다 (`test_only_final_answer_is_scored`).
- **공식 SQA3D 매칭 결과**는 `official_correct`에 따로 기록한다(빈 답은 오답 처리). 공식 매칭으로만 정답이면 flag `official_only_match`를 붙인다. 라벨은 EM으로만 정한다.
- **Checkpoint 0** (추론 없이 바로 답한 경우)도 같은 규칙으로 라벨을 만든다.
- **라벨과 probe 예측은 분리한다.** 라벨 파일에는 `p_stop`, `threshold` 같은 probe 필드가 없다(검증 항목).
- 답변 레코드의 `question_id`나 `checkpoint_index`가 checkpoint와 다르면 오류로 멈춘다. 같은 `(checkpoint_id, sample_index)`가 두 번 나와도 오류다.

---

## 3. 최종 답 추출 규칙 (`answer_extraction.py`, v1)

1. `</think>`가 있으면 **마지막 `</think>` 뒤**만 사용한다. `<think>`만 있고 닫히지 않았으면 `think_unclosed`(답 없음).
2. `<|im_end|>`, `<|endoftext|>` 같은 특수 토큰을 제거한다.
3. 비어 있지 않은 **첫 줄**만 남긴다.
4. 머리말(`Answer:`, `A:`, `Final answer:`, `The answer is`)을 제거한다.
5. 마크다운 강조(`**`), 따옴표, 끝의 `. ! ;`를 제거한다.

| 원문 | 추출 결과 | 상태 |
|---|---|---|
| `\n\nbrown<\|im_end\|>` | `brown` | ok |
| `The answer is **two**.` | `two` | ok |
| `Final answer: table\nBecause ...` | `table` | ok |
| `<think>x</think>\n\n` | (빈 값) | empty |
| `<think>\nstill thinking` | (빈 값) | think_unclosed |

## 4. 정규화 규칙과 동의어 처리 범위

정규화 EM은 예측과 정답 **양쪽에** 공식 SQA3D `clean_answer`(`sqa3d_scoring.py`)를 적용한 뒤, 대소문자를 무시하고 완전히 일치하는지 본다.

| 처리 | 예 |
|---|---|
| 소문자화, 앞뒤 공백 제거, 연속 공백 합치기 | `Left ` → `left` |
| 영문·숫자·`, ' - :`·공백 외 문자 제거 | `left.` → `left` |
| 공식 코드의 오타 교정 | `letf`/`let`→`left`, `rigth`/`rght`→`right`, `behine`→`behind`, `chai`→`chair` 등 |
| 숫자 → 영단어 (0–20, 23), `none`→`zero` | `2` → `two` |
| 단어 끝 숫자 제거 | `mat2` → `mat` |
| 관사 제거 (a / an / the + 단어) | `the table` → `table` |
| `backwards` → `backward` | |

**동의어 처리 범위: 위 규칙 이외에는 없다.**
- `sofa`와 `couch`, `left`와 `left side`, `in front`와 `forward`는 서로 **다른 답**으로 채점한다.
- 💭 공식 정규화가 문자를 제거하는 방식이라 소수점이 사라진다(`3.5` → `35`). SQA3D 정답에는 소수가 없어서 영향이 없지만, 다른 데이터셋에 재사용할 때는 주의해야 한다.

---

## 5. 라벨 생성 예시 (가상 데이터)

**요약**: checkpoint 35개 → 라벨 36행 (labeled 28 / pending 7 / missing 1), STOP 14 / CONTINUE 14. 고아 답변 1개가 보고됨.

| 가상 문항 | 시나리오 (Part C) | checkpoint별 stop_label |
|---|---|---|
| q2 | 1. 모두 정답 | `[1, 1]` |
| q4 | 2. 모두 오답 (+7. 빈 답, 형식 오류) | `[0, 0, 0(empty), 0(invalid), 0]` |
| q1 | 3. 중간에 처음 정답 | `[0, 0, 1, 1, 1, 1]` |
| q3 | 4. 정답 이후 다시 오답 | `[0, 1, 0, 0, 1]`. 2번은 `not left`라서 공식 매칭으로는 정답, EM으로는 오답 |
| q5 | 5. 마지막에서만 정답 | `[0, 1]` |
| q7 | 6. 생성 답변 없음 | `[0, pending, missing]` |
| q10 | 6. 답변 레코드 전무 | 전부 `pending` |
| q6 | Checkpoint 0만 있음 | `[1]` |
| q9 | 같은 checkpoint 2회 생성 | cp1: sample0 = 0, sample1 = 1 (둘 다 보존) |

**라벨 행 예시** (q3, checkpoint 2)
```json
{"checkpoint_id": "synthetic_run01/900000000003/sentence/2", "question_id": 900000000003,
 "split": "validation", "checkpoint_index": 2, "sample_index": 0,
 "label_status": "labeled", "stop_label": 0, "label_name": "CONTINUE",
 "generated_answer": "\n\nnot left<|im_end|>", "extracted_answer": "not left",
 "reference_answer": "left", "em_correct": false, "official_correct": true,
 "flags": ["official_only_match"], "label_source": "normalized_em",
 "answer_token_count": null, "answer_latency_ms": null, "synthetic": true}
```

---

## 6. 검증 결과

- **단위 테스트**: `tests/test_checkpoint_labeling.py` 29개 통과
- **라벨 검증 스크립트**: 10개 항목 통과
  - 모든 checkpoint에 라벨 행이 있음
  - `(checkpoint_id, sample_index)` 중복 없음
  - `labeled`일 때만 `stop_label` 값이 있음
  - probe 필드 없음
  - split과 question_id가 checkpoint와 일치
  - **라벨을 독립적으로 EM 재채점한 결과와 일치**
  - 빈 답이나 형식 오류인데 STOP인 행 없음
- **전체 테스트**: 147개 통과 (checkpoint 88 + 라벨링 29 + 평가 30)

---

## 7. 남은 문제

1. **중간 답 생성 프롬프트 확정 (A 담당)**: prefix 뒤에 `</think>`를 붙여 강제로 답하게 하는 형식, 답 길이 제한, greedy 사용 여부 (`docs/labeling_evaluation_interface.md` 1절)
2. **라벨 정의 확정**: 지금은 checkpoint마다 "그 시점의 답이 맞는가"를 라벨로 쓴다. 제안서의 d*(처음으로 **안정적으로** 맞히는 지점) 기반 라벨을 쓸지는 팀 결정이 필요하다. 지금 라벨에서 d*를 계산할 수는 있지만 라벨에 섞지는 않았다.
3. **여러 번 생성한 경우 학습 방법**: 샘플별 행을 그대로 쓸지, 정답 비율(soft label)을 쓸지
4. **SQA3D 라벨 노이즈 5건**: 전처리 단계에서 표시한 `label_conflict` 문항을 라벨링에서 제외할지
5. **정답 클래스 불균형**: Long CoT 후반으로 갈수록 STOP 비율이 높아질 것으로 예상된다. probe 학습 시 가중치를 줄지 검토해야 한다.
