> 데이터팀 ZIP에서 가져온 원문입니다. 아래 실행 결과·설정·설치 설명은 작성 당시 기록이며 현재 통합본의 검증 결과가 아닙니다. 현재 실행 방법과 차이는 [integration.md](integration.md)를 따릅니다.

# 코드 안내서 (Dataset & Evaluation)

작성: 2026-10-29

처음 보는 사람을 위한 지도입니다. 파일마다 맨 위 설명(docstring)의 `요약:` 줄에 같은 내용이 한 줄로 적혀 있습니다.

> ⚠️ 2~4단계는 **가상(synthetic) CoT 데이터**로만 검증했습니다. 실제 Qwen3 출력으로는 아직 실행하지 않았습니다.

---

## 1. 전체 흐름

```text
[1단계 데이터]           [2단계 Checkpoint]          [3단계 라벨]                  [4단계 평가]
SQA3D 원본                Qwen3 생성 결과             checkpoint                    라벨 + probe 예측
  │ prepare_sqa3d.py        │ build_checkpoints.py      + 강제 답변 (모델 담당)       + 방식별 결과
  ▼                         ▼                           + 정답                         │ evaluate_early_exit.py
train/val/test.jsonl  ──►  checkpoint 파일   ──────►     │ build_checkpoint_labels.py   ▼
  (프롬프트 입력)          (추론을 조각낸 지점들)          ▼                            지표 JSON
                                                       STOP/CONTINUE 라벨 ─────────► (정확도, 토큰 감소율,
       ▲ 모델 담당: Qwen3로 Long CoT 생성,                                              AUROC/F1 …)
         checkpoint마다 강제 답변, hidden state 추출, probe 학습
```

- 단계마다 **만드는 스크립트(build/prepare)**와 **검사하는 스크립트(validate)**가 짝으로 있습니다.
- 모델을 돌리는 부분(생성, hidden state, probe)은 모델 담당 영역이라 비어 있습니다 (`generation/`, `probe/`).
- 모델 담당과 주고받는 파일 형식: `qwen3_output_spec.md`, `labeling_evaluation_interface.md`

## 2. 폴더별 역할

| 폴더 | 역할 |
|---|---|
| `src/thinkenough/data/` | 모델 입력 쪽: 프롬프트 만들기, 추론을 checkpoint로 자르기 |
| `src/thinkenough/evaluation/` | 채점과 평가 쪽: 답 추출, 채점, STOP/CONTINUE 라벨, 지표, 방식 비교 |
| `src/thinkenough/labeling/` | 비어 있음. 지금 라벨 코드는 `evaluation/checkpoint_labeling.py`에 있습니다 (채점 규칙을 함께 쓰기 때문). d* 방식 라벨을 쓰게 되면 여기에 둡니다. |
| `src/thinkenough/generation/`, `probe/` | 비어 있음. 모델 담당이 넣을 자리 |
| `scripts/` | 명령줄에서 실행하는 파일 (만들기 / 검사하기) |
| `tests/` | pytest 테스트, `fixtures/`는 가상 데이터 (`"synthetic": true`) |
| `docs/` | 단계별 보고서 |
| `data/`, `outputs/` | 데이터와 결과 (GitHub에는 올라가지 않음, `.gitignore`) |

## 3. 파일별 요약

### 1단계: SQA3D 데이터

| 파일 | 하는 일 | 입력 → 출력 |
|---|---|---|
| `src/thinkenough/data/sqa3d_prompt.py` | 문항의 `input`(캡션, 상황, 질문)만으로 프롬프트를 만든다. 정답은 들어가지 않는다. 템플릿은 임시. | 레코드 → 프롬프트 문자열 |
| `src/thinkenough/evaluation/sqa3d_scoring.py` | 공식 채점 방식을 그대로 옮긴 것 + 주 지표인 정규화 EM. 모든 단계가 이 채점을 쓴다. | 예측, 정답 → 맞음/틀림 |
| `scripts/audit_sqa3d.py` | 원본 조사 (개수, 분할 겹침, 정답 분포, 캡션 유무) | `data/raw/sqa3d/` → `outputs/audit/` |
| `scripts/prepare_sqa3d.py` | 전처리, Train/Val/Test + Pilot 300/100/100 | `data/raw/sqa3d/` → `data/processed/sqa3d/` |
| `scripts/validate_sqa3d.py` | 전처리 결과 검사 (중복, 장면 겹침, 정답 유출, 채점기 동작) | → `outputs/preprocessing_summary.json` |

보고서: `dataset_audit.md`, `preprocessing_report.md`

### 2단계: Reasoning Checkpoint

| 파일 | 하는 일 | 입력 → 출력 |
|---|---|---|
| `src/thinkenough/data/reasoning_checkpoint.py` | `<think>` 추론을 꺼내 문장/줄/번호 단위로 자르고, 각 지점까지의 추론(prefix)을 checkpoint로 만든다. 원본 토큰 ID가 있으면 토큰 위치도 계산한다. | 생성 레코드 → checkpoint 목록 |
| `scripts/build_checkpoints.py` | 생성 결과 파일 전체에 위 과정을 실행 | 생성 JSONL → `data/processed/checkpoints/` |
| `scripts/validate_checkpoints.py` | checkpoint가 원래 생성 결과와 정확히 맞는지 검사 | → `outputs/checkpoint_validation.json` |
| `tests/test_reasoning_checkpoint.py` | 위 코드 테스트 | `tests/fixtures/fake_qwen3_outputs.jsonl` |

보고서: `checkpoint_report.md`, 입력 형식: `qwen3_output_spec.md`

### 3단계: STOP/CONTINUE 라벨

| 파일 | 하는 일 | 입력 → 출력 |
|---|---|---|
| `src/thinkenough/evaluation/answer_extraction.py` | 모델 출력에서 짧은 최종 답만 뽑는다. 라벨과 평가가 같은 규칙을 쓴다. | 출력 문자열 → 답 |
| `src/thinkenough/evaluation/checkpoint_labeling.py` | checkpoint마다 강제 답변을 정답과 비교해 STOP(1)/CONTINUE(0)를 붙인다. 답이 없으면 `pending`/`missing`. | checkpoint + 강제 답변 + 정답 → 라벨 |
| `scripts/build_checkpoint_labels.py` | 라벨 파일 만들기 | → `data/processed/checkpoint_labels/` |
| `scripts/validate_checkpoint_labels.py` | 라벨 검사 (중복, 재채점 일치, 라벨이 다음 checkpoint로 번지지 않았는지) | → `outputs/checkpoint_label_validation.json` |
| `tests/test_checkpoint_labeling.py` | 위 코드 테스트 | `fake_checkpoint_answers.jsonl`, `fake_references.jsonl` |

보고서: `labeling_report.md`

### 4단계: 평가

| 파일 | 하는 일 | 입력 → 출력 |
|---|---|---|
| `src/thinkenough/evaluation/metrics.py` | 지표 계산 함수 (정확도, Retention, 토큰 감소율, Latency, AUROC/AP/F1). 값마다 `status`(measured / synthetic / not_computable …)를 붙인다. | 숫자 → 지표 dict |
| `src/thinkenough/evaluation/early_exit_evaluator.py` | Direct / Fixed-k / Full CoT / Probe early exit을 같은 문항 집합에서 비교. τ는 validation에서만 고르고, 어기면 `ThresholdLeakageError`. | 결과·라벨·예측 → 비교표 |
| `scripts/evaluate_early_exit.py` | 평가 전체 실행: 방식 비교 → validation에서 τ 선택 → test에 고정 적용 | → `outputs/evaluation/` |
| `tests/test_evaluation.py` | 위 코드 테스트 (숫자는 코드 확인용) | `fake_evaluation_results.jsonl`, `fake_probe_predictions.jsonl` |

보고서: `evaluation_report.md`, 모델 담당과의 형식: `labeling_evaluation_interface.md`

## 4. 실행 순서

저장소 최상위 폴더에서 실행합니다. 2~4단계는 인자 없이 실행하면 가짜 예시 데이터로 동작합니다.

```bash
.venv/bin/python scripts/audit_sqa3d.py              # 1단계 (선택)
.venv/bin/python scripts/prepare_sqa3d.py            # 1단계
.venv/bin/python scripts/validate_sqa3d.py
.venv/bin/python scripts/build_checkpoints.py        # 2단계
.venv/bin/python scripts/validate_checkpoints.py
.venv/bin/python scripts/build_checkpoint_labels.py  # 3단계
.venv/bin/python scripts/validate_checkpoint_labels.py
.venv/bin/python scripts/evaluate_early_exit.py      # 4단계
.venv/bin/python -m pytest tests/ -q                 # 전체 테스트 (147개)
```

주의: `tests/test_checkpoint_labeling.py`와 `tests/test_evaluation.py`는 `data/processed/checkpoints/example_checkpoints.jsonl`을 읽습니다. 이 파일이 없으면 `build_checkpoints.py`를 먼저 실행하세요.
