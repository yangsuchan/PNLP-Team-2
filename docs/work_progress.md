> 데이터팀 ZIP에서 가져온 원문입니다. 아래 실행 결과·설정·설치 설명은 작성 당시 기록이며 현재 통합본의 검증 결과가 아닙니다. 현재 실행 방법과 차이는 [integration.md](integration.md)를 따릅니다.

# ThinkEnough — Dataset & Evaluation 작업 진행 현황

- 기준일: 2026-10-16
- 담당: Dataset & Evaluation (GitHub `kmjshj0329-eng`)
- 저장소: `~/PNLP-Team-2` (원격 `yangsuchan/PNLP-Team-2`). **모든 작업은 로컬에만 있고 commit / push하지 않았다** (`main` = `origin/main` = `f34a57f`).
- 연구: Qwen3 (8B thinking) Long CoT의 hidden state에 probe를 붙여 STOP/CONTINUE를 예측하는 Dynamic Early Exit. 주력 데이터는 SQA3D, 보조 데이터는 SCoT.

> ⚠️ **실제 Qwen3는 아직 한 번도 실행하지 않았다.** 아래 checkpoint, 라벨, 평가 수치는 모두 **가상(synthetic) 데이터**로 코드 동작만 확인한 것이다. 6절을 참고한다.

---

## 1. 완료한 작업과 아직 진행하지 않은 작업

### 1.1 완료

| # | 작업 | 결과 | 데이터 종류 |
|---|---|---|---|
| 0 | 개발 환경 | WSL2 + uv + Python 3.12 가상환경(`.venv`), GitHub SSH 연동, 프로젝트 폴더 구조 | — |
| 1 | SQA3D 조사 | 공식 분할, 필드, 채점 방식, Test 정답 공개 여부, 분할 간 중복, 누출 위험 정리 | **실제 SQA3D** |
| 2 | 전처리 + 분할 | 공식 Train/Val/Test 전체 전처리본 + Pilot 300/100/100, 검증 59/59 | **실제 SQA3D** |
| 3 | Reasoning Checkpoint | `<think>` 파싱, 문장/줄/번호 단위 분할, 누적 prefix, 토큰 정렬 인터페이스 | 가상 CoT |
| 4 | STOP/CONTINUE 라벨 | 최종 답 추출, EM 기반 라벨, pending/missing 처리 | 가상 |
| 5 | 평가 코드 | 정확도, Retention, 정확도 차이, 토큰 감소율, Latency, Probe 품질, threshold 규칙, 오프라인 시뮬레이션 | 가상 |
| — | A 담당 협업 명세 | Qwen3 출력 형식, 강제 답변·probe 예측·방식별 결과 형식 (초안) | — |

### 1.2 아직 진행하지 않음

| 작업 | 이유 / 필요한 것 |
|---|---|
| 실제 Qwen3 Long CoT 생성 | A 담당의 GPU 실행 (월요일부터) |
| 실제 출력으로 토큰 정렬과 hidden state 위치 검증 | Qwen3 원본 생성 토큰 ID 필요 |
| 실제 checkpoint별 강제 답변 → 실제 STOP/CONTINUE 라벨 | A 담당의 강제 답변 결과 |
| Hidden state 추출, probe 학습 | A 담당 |
| 실제 threshold 선정, test 평가, 온라인 Early Exit | probe 예측과 온라인 실행 결과 |
| 실측 토큰 감소율, Latency | 실제 생성 토큰 수와 GPU 시간 |
| SCoT 보조 데이터셋 조사 | 아직 시작하지 않음 |
| PyTorch 설치 | GPU 테스트가 필요해지면 (RTX 5080은 cu128 이상) |
| GitHub 브랜치 업로드 | 사용자가 요청할 때 (5절) |

---

## 2. 생성·수정한 주요 파일과 역할

### 2.1 코드 (`src/thinkenough/`)

| 파일 | 역할 |
|---|---|
| `data/sqa3d_prompt.py` | 정답을 읽지 않는 프롬프트 생성기 (템플릿 `sqa3d_official_gpt_v0`는 **임시**) |
| `data/reasoning_checkpoint.py` | `<think>` 파싱, 분할기(sentence/line/numbered_step), checkpoint 생성(`checkpoint_id` 포함), Qwen3 토큰 정렬 |
| `evaluation/sqa3d_scoring.py` | 공식 SQA3D 채점 재현 + 정규화 EM |
| `evaluation/answer_extraction.py` | 모델 출력에서 생각 과정을 빼고 최종 답만 추출 (라벨링과 평가가 공통으로 사용) |
| `evaluation/checkpoint_labeling.py` | 중간 답을 정답과 비교해서 STOP(1) / CONTINUE(0) 라벨 생성 |
| `evaluation/metrics.py` | 지표 계산. 모든 값에 status(measured/estimated/synthetic/undefined/not_computable)가 붙는다 |
| `evaluation/early_exit_evaluator.py` | 방식 비교, threshold 선정(validation만)과 적용, probe 분류 평가, 오프라인 시뮬레이션 |
| `generation/`, `labeling/`, `probe/` | 빈 패키지 (자리만 마련) |

### 2.2 스크립트 (`scripts/`)

| 파일 | 역할 |
|---|---|
| `audit_sqa3d.py` | SQA3D 원본 감사 → `outputs/audit/sqa3d_audit.json` |
| `prepare_sqa3d.py` | 원본 → 전처리 JSONL (전체 + Pilot) |
| `validate_sqa3d.py` | 전처리 결과 검증 (59개 항목) |
| `build_checkpoints.py` / `validate_checkpoints.py` | 생성 결과 → checkpoint / 검증 (12개 항목) |
| `build_checkpoint_labels.py` / `validate_checkpoint_labels.py` | checkpoint + 강제 답변 + 정답 → 라벨 / 검증 (10개 항목) |
| `evaluate_early_exit.py` | 평가 실행 → `outputs/evaluation/` |

### 2.3 테스트 (`tests/`)

| 파일 | 역할 |
|---|---|
| `test_reasoning_checkpoint.py` | checkpoint 테스트 88개 |
| `test_checkpoint_labeling.py` | 라벨링·답 추출 테스트 29개 |
| `test_evaluation.py` | 지표·평가 테스트 30개 |
| `fixtures/fake_qwen3_outputs.jsonl` | 가상 Qwen3 출력 10개 |
| `fixtures/fake_checkpoint_answers.jsonl` | 가상 checkpoint별 강제 답변 30개 |
| `fixtures/fake_references.jsonl` | 가상 정답 (실제 SQA3D와 분리) |
| `fixtures/fake_evaluation_results.jsonl` | 가상 방식별 결과 (5개 방식 × 6문항) |
| `fixtures/fake_probe_predictions.jsonl` | 가상 probe 확률 29개 |

### 2.4 문서 (`docs/`)

| 파일 | 내용 |
|---|---|
| `dataset_audit.md` | SQA3D 조사 보고서 |
| `preprocessing_report.md` | 전처리, Pilot, 검증 결과 |
| `checkpoint_report.md` | Reasoning Checkpoint 구현 |
| `labeling_report.md` | STOP/CONTINUE 라벨 규칙, 정규화·동의어 범위 |
| `evaluation_report.md` | 평가 지표, threshold 규칙, 평가 모드 |
| `qwen3_output_spec.md` | A 담당 요청: Full CoT 생성 결과 형식 |
| `labeling_evaluation_interface.md` | A 담당 요청: 강제 답변·probe 예측·방식별 결과 형식, 협의 안건 |
| `work_progress.md` | 이 문서 |

### 2.5 설정과 데이터 (로컬)

| 경로 | 내용 | git |
|---|---|---|
| `.gitignore`, `requirements.txt` | 제외 규칙, 패키지 목록 | 올라감 |
| `CLAUDE.md` | Claude용 개인 작업 노트 | 로컬 제외 (`.git/info/exclude`) |
| `data/raw/sqa3d/` | 공식 원본 (`sqa_task.zip` md5 `0da8bb0e...`, 캡션 pkl) | 제외 |
| `data/processed/sqa3d/` | 전처리 전체본 + `pilot/` | 제외 |
| `data/processed/checkpoints/`, `checkpoint_labels/` | 가상 예시 출력 | 제외 |
| `outputs/` | 검증 요약, 라벨 충돌·캡션 누락 목록, 가상 평가 예시 | 제외 |

---

## 3. 현재 테스트 결과 (2026-10-16 재실행)

| 항목 | 결과 |
|---|---|
| `tests/test_reasoning_checkpoint.py` | **88 passed** |
| `tests/test_checkpoint_labeling.py` | **29 passed** |
| `tests/test_evaluation.py` | **30 passed** |
| **단위 테스트 합계** | **147 passed, 0 failed, 0 skipped** |
| SQA3D 전처리 검증 (`validate_sqa3d.py`) | 59 / 59 통과 |
| Checkpoint 검증 (`validate_checkpoints.py`) | 12 / 12 통과 (sentence, line, numbered_step, `--no-prefix-text` 모두) |
| 라벨 검증 (`validate_checkpoint_labels.py`) | 10 / 10 통과 |
| 전처리 재현성 | 두 번 실행한 결과 JSONL 6개의 md5 일치 |

실행 방법:
```bash
cd ~/PNLP-Team-2
.venv/bin/python -m pytest tests/ -q
```

---

## 4. 월요일 모델 담당 A와 협의할 내용

전달할 문서: `docs/qwen3_output_spec.md`, `docs/labeling_evaluation_interface.md` (바탕화면에도 복사본 있음)

1. **모델 설정**: 정확한 Qwen3 버전과 revision, 양자화 방식(16GB GPU라 4-bit 예상), `<think>`가 생성 결과에 나오는지 프롬프트에 들어가는지
2. **생성 결과 형식**: `generated_token_ids`를 원본 그대로 줄 것, `generated_text` = `decode(ids, skip_special_tokens=False)`, `prompt_token_count`
3. **hidden state 위치 규칙**: `hidden_state_position = prompt_token_count + reasoning_token_end - 1`, 생성 후 forward pass를 다시 돌려 추출하는 방식 제안, 저장할 layer
4. **강제 답변 형식**: `prompt + <think>\n{prefix}\n</think>\n\n` 뒤에 답 생성, greedy 여부, `max_new_tokens=32`, prefix를 토큰 ID로 이을지 텍스트로 넣을지
5. **토큰 수 규칙**: reasoning / answer / trial answer 구분, EOS 포함 여부
6. **잘린 출력**(`finish_reason=length`) 처리
7. **Pilot 실행 범위**: train 300문항 × checkpoint 수 → 강제 답변 생성 횟수와 GPU 시간 추정. Long CoT는 checkpoint가 많으므로 분할 단위(조각 합치기, 간격 두기)도 같이 정한다
8. **온라인 Early Exit 구현과 latency 측정** 담당 및 측정 조건
9. **첫 정렬 검증**: 실제 출력 몇 개로 checkpoint 경계 토큰과 hidden state 위치를 함께 확인

---

## 5. 다음에 Claude Code를 실행했을 때 이어서 할 작업

- 시작할 때 `~/PNLP-Team-2/CLAUDE.md`를 먼저 읽는다 (홈의 `~/CLAUDE.md`가 안내한다).

**A 담당 출력을 받은 경우**
1. `generated_text`와 토큰 ID가 일치하는지 확인 → `build_checkpoints.py --input <파일>` → `validate_checkpoints.py`
2. 실제 출력 일부로 토큰 정렬과 hidden state 위치를 직접 확인하고 결과를 `checkpoint_report.md`에 추가
3. 강제 답변을 받으면 → `build_checkpoint_labels.py --references data/processed/sqa3d/*.jsonl` → `validate_checkpoint_labels.py`
4. probe 예측과 방식별 결과를 받으면 → `evaluate_early_exit.py`로 실제 평가

**A 담당 출력과 무관하게 할 수 있는 작업**
- 미결정 사항 정리: 캡션 없는 장면 처리, 라벨 정의(시점별 vs d* 기반), τ 기준(F1 vs precision), 언어 prior 기준선 추가
- SCoT 보조 데이터셋 조사
- 팀 PPT의 실험 설계를 공유받으면 평가 방식과 맞추기
- Qwen3 프롬프트가 확정되면 `sqa3d_prompt.py`를 교체하고 `prepare_sqa3d.py`를 다시 실행 (토큰 수 갱신)

**GitHub 업로드 (사용자가 요청할 때만)**
1. `git fetch` → 최신 `origin/main`에서 새 브랜치 생성 (이름은 확인받음) → 테스트 통과 확인 → commit → push → PR 링크 전달
2. **데이터 포함 범위를 먼저 물어본다**: (a) 코드와 문서만 (추천) / (b) Pilot 등 작은 파일도 포함 / (c) 전체 데이터는 다른 경로로 공유. `train.jsonl`은 140MB라 GitHub 100MB 제한에 걸리고, ScanRefer 캡션의 재배포 조건도 확인이 필요하다.

---

## 6. 실제 Qwen3 실험과 가상 데이터 테스트의 구분

| 구분 | 실제 데이터 | 가상(synthetic) 데이터 |
|---|---|---|
| 해당 작업 | SQA3D 조사, 전처리, Pilot 구성, 캡션 선택, 입력 토큰 수(tokenizer만 사용) | Qwen3 출력, checkpoint, 강제 답변, 라벨, probe 확률, 방식별 결과, latency |
| 위치 | `data/raw/sqa3d/`, `data/processed/sqa3d/` | `tests/fixtures/`, `data/processed/checkpoints/example_*`, `data/processed/checkpoint_labels/example_*`, `outputs/evaluation/example_*` |
| 식별 방법 | 실제 SQA3D `question_id` (2206020xxxxx) | `"synthetic": true`, question_id `9000000000xx` / `9100000000xx`, `model: "synthetic"`, 출력의 `"data_source": "synthetic"` |
| 수치의 의미 | 실제 데이터 통계 | **코드가 의도대로 동작하는지만** 보여줌. 성능 결과가 아님 |

- 가상 평가 예시의 정확도, 토큰 감소율(예: 49%), latency 같은 숫자를 **연구 결과로 인용하면 안 된다.** 평가 코드는 가상 값에 `synthetic` 상태와 "NOT a performance result" 문구를 붙인다.
- 토큰 위치는 실제 생성 토큰 ID가 없으면 null이다. 정렬 테스트에는 가상 텍스트를 다시 토큰화한 **합성 ID**만 썼다. 실제 Qwen3 출력으로 다시 검증해야 한다.
- 실제 실험 결과는 A 담당의 GPU 출력이 들어온 뒤에만 나온다. 그때는 `synthetic: false`, `token_count_source: "generated_ids"`, `latency_source: "measured"`로 표시된다.
