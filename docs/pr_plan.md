> 데이터팀 ZIP에서 가져온 원문입니다. 아래 실행 결과·설정·설치 설명은 작성 당시 기록이며 현재 통합본의 검증 결과가 아닙니다. 현재 실행 방법과 차이는 [integration.md](integration.md)를 따릅니다.

# GitHub PR 나누기 계획

작성: 2026-10-29 · **아직 아무것도 올리지 않았습니다.**

한 번에 올리면 파일이 많아 읽기 어렵습니다. 그래서 단계별로 **PR 4개**로 나눕니다. 뒤 PR이 앞 PR의 코드를 쓰기 때문에 **1 → 2 → 3 → 4 순서**로 올리고 합칩니다.
파일별 설명은 `code_guide.md`에 있습니다.

| PR | 브랜치 이름 (예시) | 내용 | 파일 수 |
|---|---|---|---|
| 1 | `dataset-eval/1-sqa3d-data` | 기본 설정 + SQA3D 조사·전처리·채점 | 약 17 |
| 2 | `dataset-eval/2-checkpoints` | Reasoning Checkpoint | 7 |
| 3 | `dataset-eval/3-labels` | STOP/CONTINUE 라벨 | 8 |
| 4 | `dataset-eval/4-evaluation` | 평가 코드 + 안내 문서 | 11 |

`data/`, `outputs/`, `.venv/`, `CLAUDE.md`는 올리지 않습니다. 빈 폴더 구조를 지키는 `.gitkeep`만 올라갑니다.

---

## 올리기 전에 고칠 것

- [ ] `tests/test_checkpoint_labeling.py`와 `tests/test_evaluation.py`가 `data/processed/checkpoints/example_checkpoints.jsonl`을 읽습니다. `data/`는 GitHub에 올라가지 않으므로, 새로 받은 사람은 `build_checkpoints.py`를 먼저 돌리지 않으면 테스트가 실패합니다.
  → 해결안: 이 파일을 `tests/fixtures/example_checkpoints.jsonl`로 복사하고 테스트가 그것을 읽게 바꿉니다 (PR 3에 포함).
- [ ] 다른 팀원 작업과 합친 뒤 파일 목록을 다시 확인합니다.
- [ ] 각 PR마다 `pytest`가 통과하는지 확인합니다.

---

## PR 1 — 기본 설정 + SQA3D 데이터

**파일**
- `.gitignore`, `requirements.txt`, 폴더 구조(`.gitkeep`: `configs/`, `notebooks/`, `data/**`, `outputs/`)
- `src/thinkenough/__init__.py`, `src/thinkenough/{data,evaluation,generation,labeling,probe}/__init__.py`
- `src/thinkenough/data/sqa3d_prompt.py`
- `src/thinkenough/evaluation/sqa3d_scoring.py`
- `scripts/audit_sqa3d.py`, `scripts/prepare_sqa3d.py`, `scripts/validate_sqa3d.py`
- `docs/dataset_audit.md`, `docs/preprocessing_report.md`

**PR 설명 초안**
> SQA3D를 주력 데이터로 쓰기 위한 조사·전처리·채점 코드입니다.
> - 공식 분할 사용 (train 26,623 / val 3,261 / test 3,519), 분할 간 장면 겹침 없음
> - 문항은 지우지 않고 문제는 `meta.flags`로 표시, Pilot 300/100/100 (seed 42)
> - 장면 입력은 ScanRefer 캡션 30개 (장면 ID로만 고정 선택)
> - 채점: 공식 방식 이식 + 정규화 EM(주 지표)
> - 데이터는 올리지 않음: SQA3D 원본을 `data/raw/sqa3d/`에 받은 뒤 `prepare_sqa3d.py` → `validate_sqa3d.py` 실행
> - 프롬프트 템플릿은 임시

## PR 2 — Reasoning Checkpoint

**파일**
- `src/thinkenough/data/reasoning_checkpoint.py`
- `scripts/build_checkpoints.py`, `scripts/validate_checkpoints.py`
- `tests/test_reasoning_checkpoint.py`, `tests/fixtures/fake_qwen3_outputs.jsonl`
- `docs/checkpoint_report.md`, `docs/qwen3_output_spec.md`

**PR 설명 초안**
> Qwen3 Long CoT 출력을 checkpoint(멈출 수 있는 지점)로 나누는 코드입니다.
> - `<think>` 파싱, 형식이 깨진 출력도 상태(parse_status)로 기록
> - 문장 / 줄 / 번호 단위 분할, checkpoint 0 = 추론 없음
> - 원본 토큰 ID가 있을 때만 토큰 위치 계산 (없으면 null)
> - 모델 담당이 줄 출력 형식: `docs/qwen3_output_spec.md`
> - ⚠️ 가상 CoT로만 검증, 실제 Qwen3 출력은 미검증

## PR 3 — STOP/CONTINUE 라벨

**파일**
- `src/thinkenough/evaluation/answer_extraction.py`, `src/thinkenough/evaluation/checkpoint_labeling.py`
- `scripts/build_checkpoint_labels.py`, `scripts/validate_checkpoint_labels.py`
- `tests/test_checkpoint_labeling.py`, `tests/fixtures/fake_checkpoint_answers.jsonl`, `tests/fixtures/fake_references.jsonl`
- `docs/labeling_report.md`
- (위 "고칠 것"을 반영하면) `tests/fixtures/example_checkpoints.jsonl`

**PR 설명 초안**
> probe 학습용 정답 라벨을 만드는 코드입니다.
> - checkpoint마다 강제 답변이 정규화 EM으로 맞으면 STOP=1, 아니면 0
> - 답변이 없으면 `pending`, 생성 실패는 `missing` → 학습에서 제외
> - checkpoint마다 독립적으로 판정, 여러 번 생성한 결과는 다수결 없이 보존
> - 라벨 정의(지금 방식 vs d* 방식)는 팀 결정이 필요

## PR 4 — 평가 코드 + 안내 문서

**파일**
- `src/thinkenough/evaluation/metrics.py`, `src/thinkenough/evaluation/early_exit_evaluator.py`
- `scripts/evaluate_early_exit.py`
- `tests/test_evaluation.py`, `tests/fixtures/fake_evaluation_results.jsonl`, `tests/fixtures/fake_probe_predictions.jsonl`
- `docs/evaluation_report.md`, `docs/labeling_evaluation_interface.md`
- `docs/code_guide.md`, `docs/work_progress.md`, `docs/pr_plan.md`
- (선택) `docs/team_meeting_brief.md` — 회의 자료라 올릴지 따로 정합니다

**PR 설명 초안**
> Direct / Fixed-k / Full CoT / Probe early exit을 비교하는 평가 코드입니다.
> - 지표: 정확도, Retention, 정확도 차이, 토큰 감소율, Latency, AUROC/AP/P/R/F1
> - 평가 모드(online / offline_simulation / probe_classification)를 섞지 않음
> - τ는 validation에서만 선택하고 test에는 고정 (`ThresholdLeakageError`로 막음)
> - 측정하지 않은 값은 `not_computable`로 표시
> - ⚠️ 예시 수치는 모두 가상 데이터
