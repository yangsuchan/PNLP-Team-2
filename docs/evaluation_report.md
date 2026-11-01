> 데이터팀 ZIP에서 가져온 원문입니다. 아래 실행 결과·설정·설치 설명은 작성 당시 기록이며 현재 통합본의 검증 결과가 아닙니다. 현재 실행 방법과 차이는 [integration.md](integration.md)를 따릅니다.

# Evaluation Report

- 작성일: 2026-10-16
- 담당: Dataset & Evaluation
- 범위: 5단계. Direct Answer / Fixed-k / Full Long CoT / Probe 기반 Dynamic Early Exit을 같은 문항 집합에서 비교하는 평가 코드
- 사용 데이터: **가상 데이터만**. 실제 Qwen3 추론, GPU, 유료 API는 사용하지 않았다.

> ⚠️ **이 문서의 모든 수치는 가상 데이터로 코드 동작을 확인한 것이다.** 실제 성능 향상을 뜻하지 않는다. 출력 파일에도 `"data_source": "synthetic"`과 경고 문구가 들어간다.

> 참고: 요청서에 언급된 PPT의 실험 설계는 확인할 수 없었다. 제안서(Direct, Fixed-1/2/3, Full CoT, DEER-style, ThinkEnough, Oracle)와 이번 요청 내용을 기준으로 구현했다. 방식 이름은 자유롭게 추가할 수 있다.

재현 방법:
```bash
.venv/bin/python scripts/evaluate_early_exit.py           # → outputs/evaluation/example_metrics.json, example_threshold.json
.venv/bin/python -m pytest tests/test_evaluation.py -v    # 테스트 30개
```

---

## 1. 생성 파일

| 파일 | 역할 |
|---|---|
| `src/thinkenough/evaluation/metrics.py` | 지표 계산 함수. 모든 값에 `status`가 붙는다 |
| `src/thinkenough/evaluation/early_exit_evaluator.py` | 방식 비교(온라인), threshold 선정·적용, probe 분류 평가, 오프라인 시뮬레이션 |
| `scripts/evaluate_early_exit.py` | 평가 실행 |
| `tests/test_evaluation.py` | 테스트 30개 |
| `tests/fixtures/fake_evaluation_results.jsonl` | 가상 방식별 결과 (5개 방식 × 6문항) |
| `tests/fixtures/fake_probe_predictions.jsonl` *(추가)* | 가상 probe 확률 29개 |
| `outputs/evaluation/example_metrics.json`, `example_threshold.json` | 출력 예시 (git 제외) |

기존 `sqa3d_scoring.py`(채점)와 `answer_extraction.py`(답 추출)를 재사용했다. 새로 설치한 패키지는 없다(AUROC와 AP는 기존 `scikit-learn` 사용).

---

## 2. 값의 상태 표시 (`status`)

| status | 의미 |
|---|---|
| `ok` / `measured` | 실측 입력으로 계산 (토큰: 원본 생성 토큰 ID 기준) |
| `estimated` | 일부 입력이 추정값 (예: 다시 토큰화한 토큰 수) |
| `synthetic` | 가상 데이터. **성능 결과가 아님** |
| `undefined` | 수학적으로 정의되지 않음 (0으로 나누기, 클래스가 하나뿐) |
| `not_computable` | 필요한 값이 기록되지 않음. **임의로 채우지 않는다** |

입력이 섞여 있으면 가장 나쁜 상태를 따른다(`not_computable` > `synthetic` > `estimated` > `measured`).

---

## 3. 평가 모드 (서로 섞지 않음)

| 모드 | 입력 | 의미 |
|---|---|---|
| `online` | 방식별 실제 실행 결과 | 실제로 생성을 멈춘 결과. **최종 비교의 기준** |
| `offline_simulation` | 라벨 + checkpoint + probe 확률 | "checkpoint k에서 멈췄다면": 정답 여부는 k의 강제 답변 라벨, 토큰은 k의 prefix 토큰 수 |
| `probe_classification` | probe 확률 + 라벨 | exit 정책과 무관한 probe 분류 성능 |

- 한 방식 안에 서로 다른 모드의 행이 섞이면 오류로 멈춘다.
- 시뮬레이션 결과에는 "not an online early-exit run"이라는 문구가 붙는다. `oracle_sim`은 정답을 보고 멈추는 상한선이라 비교 방식이 아니다.

---

## 4. 지표

### 4.1 Task Accuracy
- **Accuracy (%) = 정답 문제 수 / 전체 평가 문제 수 × 100**
- **주 지표: 정규화 EM**. 공식 SQA3D 매칭(`accuracy_official`)과, 공식 매칭에서 빈 답을 오답으로 친 값(`accuracy_official_empty_as_wrong`)을 함께 저장한다.
- **같은 분모 보장**: 모든 방식이 기준(Full CoT)과 **똑같은 question_id 집합**이어야 한다. 다르면 오류로 멈춘다. 생성 실패나 빈 답은 분모에서 빼지 않고 오답으로 센다.
- **공식 방식의 문항 병합**: 정답 데이터에 position/rotation이 있으면(실제 전처리 데이터) 공식 `compute_accuracy`처럼 같은 문항을 합쳐서 계산하고, 분모를 `total_units`로 따로 표기한다. 없으면(가상 데이터) 문항별로 계산하고 "not applied"로 표시한다.

### 4.2 Performance Retention과 Accuracy Difference
- **Retention (%) = Early Exit 정확도 / Full CoT 정확도 × 100**. Full CoT 정확도가 0이면 `undefined`.
- **Accuracy Difference (%p) = Early Exit 정확도 − Full CoT 정확도**
- 두 값은 단위가 다르므로 따로 보고한다.

### 4.3 Reasoning Token Reduction
- **Reduction (%) = (Full 평균 추론 토큰 − EE 평균 추론 토큰) / Full 평균 추론 토큰 × 100**
- 문항마다 기록하는 토큰을 구분한다.
  - `reasoning_tokens`: `<think>` 안의 토큰
  - `answer_tokens`: 최종 답 토큰
  - `trial_answer_tokens`: 추론 도중 시험 삼아 생성한 답의 토큰 (DEER 방식)
- `end_to_end_generated_tokens` = 추론 + 최종 답 + 시험 답 토큰
- **probe 비용**은 토큰이 아니므로 `probe_calls`와 `latency_probe_ms`에 따로 기록한다.
- **라벨링 때 prefix마다 강제로 생성한 답의 토큰**은 학습 데이터를 만드는 비용이라 라벨 파일의 `answer_token_count`에 기록한다. 추론 비용과 섞지 않는다.
- 토큰 수가 하나라도 없으면 `not_computable`, 음수(오히려 더 씀)도 그대로 보고하고 경고를 붙인다. Full CoT 토큰이 0이면 `undefined`.
- **checkpoint 개수와 토큰 수를 혼동하지 않는다**: 시뮬레이션의 토큰은 checkpoint의 `prefix_token_count`이고, 지금은 다시 토큰화한 값이라 `estimated`로 표시된다(가상 데이터는 `synthetic`).

### 4.4 Latency
- `latency.total_ms / reasoning_ms / answer_ms / probe_ms`의 평균, 중앙값, p90을 낸다. 측정 조건은 `measurement`에 기록한다.
- `latency_source`가 `synthetic`이면 "NOT a performance result"가 붙는다. 하나라도 비어 있으면 `not_computable`.

### 4.5 Probe Quality
- 확률 기반(threshold 없음): **AUROC**, **Average Precision**
- threshold 기준: **Precision, Recall, F1** (STOP을 양성으로 봄), TP/FP/FN/TN
- 확률이 없으면 `not_computable`, 라벨이 한 종류뿐이면 AUROC는 `undefined`
- 같은 checkpoint를 여러 번 생성했으면 기본적으로 `sample_index=0`만 쓴다(다수결 없음)

### 4.6 Threshold 규칙
1. τ는 **validation 행으로만** 고른다. 다른 split이 하나라도 섞이면 `ThresholdLeakageError`.
2. 선정 기준: `f1`(기본, F1 최대) 또는 `target_precision`(precision ≥ 목표치 중 recall 최대). 같은 점수면 더 높은(보수적인) τ를 고른다.
3. 결과를 `example_threshold.json`에 **선정에 쓴 question_id 목록과 해시**와 함께 저장한다.
4. test 평가는 그 τ를 그대로 쓴다(`--threshold-in`으로 고정 파일 재사용 가능). 선정 split이 validation이 아니거나, 평가 문항이 선정 문항과 겹치면 오류로 멈춘다.

---

## 5. 계산 예시 (⚠️ 가상 데이터)

### 5.1 온라인 비교 (가상 6문항, 기준 = full_cot)

| 방식 | EM 정확도 | Retention | 정확도 차이 | 평균 추론 토큰 | 토큰 감소율 | 비고 |
|---|---:|---:|---:|---:|---:|---|
| full_cot | 83.33 | — | — | 751.7 *(synthetic)* | — | 기준 |
| probe_ee | 66.67 | 80.0% | −16.67%p | 383.3 *(synthetic)* | 49.0% | 8. EE 정확도가 더 낮음 |
| probe_ee_seed2 | 83.33 | 100.0% | 0.00%p | 805.0 *(synthetic)* | **−7.1%** | 9. 정확도 같음, 10. 토큰을 더 씀 (경고) |
| direct | 50.00 | 60.0% | −33.33%p | 0 *(synthetic)* | 100.0% | |
| fixed_2 | 50.00 | 60.0% | −33.33%p | `not_computable` | `not_computable` | 토큰 1개 미기록 → 채우지 않음 |

- `fixed_2`의 공식 매칭 정확도는 66.67이고, 빈 답을 오답으로 치면 50.00이다. 공식 매칭이 빈 답을 정답으로 처리하는 특성 때문이다.
- latency는 모두 `synthetic`이고 "NOT a performance result"가 붙어 있다.

### 5.2 Threshold와 Probe 분류 (가상)
- validation 8개 checkpoint로 선정: **τ = 0.55** (F1 기준). validation에서 P = 0.80, R = 1.00, F1 = 0.89. 이 값은 선정에 쓴 데이터라 in-sample이다.
- test (5개 checkpoint, 같은 τ): AUROC 1.00, AP 1.00, P = R = F1 = 1.00. 가상 데이터라 의미 없는 값이다.

### 5.3 오프라인 시뮬레이션 (test, τ = 0.55)
- 포함 1문항(q9). 제외 2문항: q7(라벨 없는 checkpoint), q10(전부 pending).
- `probe_sim`은 checkpoint 2에서 멈춤 → 정답, `full_cot_sim` 대비 retention 100%, 토큰 감소 11.1% (synthetic)
- 모든 시뮬레이션 방식이 같은 분모를 쓰는지 테스트로 확인했다.

---

## 6. 테스트 결과

**`tests/test_evaluation.py` 30개 통과** (전체 147개 통과)

| Part C 경우 | 테스트 |
|---|---|
| 8. EE 정확도 < Full CoT | `test_case8_early_exit_lower_accuracy` |
| 9. 정확도 같음 | `test_case9_equal_accuracy` |
| 10. EE가 토큰을 더 씀 | `test_case10_early_exit_uses_more_tokens` |
| 11. probe 확률 없음 | `test_case11_no_probe_probabilities`, `test_case11_checkpoints_without_probe_predictions` |
| 12. test로 threshold 선정 | `test_case12_*` 3개 (test 섞임, 선정 split 오류, 선정 문항으로 평가) |
| 13. 중복·누락 | `test_case13_duplicate_result_rows_rejected` (+ 라벨링 쪽 3개) |
| 14. 비교 문항 집합이 다름 | `test_case14_different_question_sets_rejected` |
| 15. Full 정확도 0 / 토큰 0 | `test_case15_*` 4개 |

추가 테스트: 측정 안 한 토큰을 채우지 않는지, 가상 토큰이 `measured`로 보고되지 않는지, 평가 모드 혼합 거부, 생성 실패도 같은 분모에서 오답으로 세는지, 시뮬레이션 분모가 같은지.

1–7번 경우는 `tests/test_checkpoint_labeling.py`에 있다 (`docs/labeling_report.md`).

---

## 7. 현재 가능한 것과 실제 Qwen3 연결 이후에만 가능한 것

| 기능 | 지금 | 비고 |
|---|:---:|---|
| 채점 (EM / 공식 매칭) | ✅ | 실제 SQA3D 정답으로 바로 사용 가능 |
| 최종 답 추출 | ✅ | 실제 Qwen3 출력 형식으로 재확인 필요 |
| 정확도, retention, 정확도 차이 | ✅ | 결과 파일만 있으면 계산 |
| 토큰 감소율 | ⏳ | 원본 생성 토큰 수(`generated_ids` 기준)가 있어야 `measured`가 된다 |
| Latency | ⏳ | 실제 GPU 측정값 필요 |
| Probe 품질, threshold | ⏳ | probe 학습과 hidden state 추출 이후 |
| 오프라인 시뮬레이션 | ⏳ | 실제 강제 답변 라벨 + probe 확률 필요 |
| 온라인 Early Exit | ⏳ | A 담당이 probe를 생성 루프에 연결해야 함 |

---

## 8. 발견된 문제와 향후 과제

1. **오프라인 시뮬레이션 ≠ 온라인 실행**: 멈춘 뒤 답을 생성하는 방식(프롬프트, sampling)이 라벨링 때와 다르면 결과가 달라진다. 최종 보고는 온라인 결과를 기준으로 하고, 시뮬레이션은 τ 탐색 등 보조 용도로 써야 한다.
2. **Fixed-k의 k 단위**: checkpoint 개수 기준(Fixed-k)과 토큰 예산 기준(Fixed Token Budget)이 다르다. 결과 파일의 `method_type`으로 구분하고, 비교 시 실제 토큰 수로 보고해야 한다.
3. **threshold 기준 결정**: F1과 precision 목표 중 무엇을 쓸지. Early exit에서는 잘못된 STOP이 곧 정확도 손실이라 precision 기준이 더 적합할 수 있다.
4. **probe 오버헤드**: probe 호출 수가 checkpoint 수에 비례하므로, 실제 latency에서 probe 비용과 hidden state 전송 비용을 따로 측정해야 한다.
5. **공식 매칭의 빈 답 문제**: 보고서에는 EM을 주 지표로 쓰고, 공식 매칭은 두 버전(원본 / 빈 답 오답)을 함께 표기하는 것을 권장한다.
6. **가상 시뮬레이션 표본이 작다**: test 1문항만 포함되었다. 실제 데이터에서는 pending/missing 비율을 함께 보고해야 한다.
