> 데이터팀 ZIP에서 가져온 원문입니다. 아래 실행 결과·설정·설치 설명은 작성 당시 기록이며 현재 통합본의 검증 결과가 아닙니다. 현재 실행 방법과 차이는 [integration.md](integration.md)를 따릅니다.

# SQA3D Preprocessing Report

- 작성일: 2026-10-16
- 담당: Dataset & Evaluation
- 근거: [docs/dataset_audit.md](dataset_audit.md) (1단계 조사 결과)
- 재현 방법:
  ```bash
  .venv/bin/python scripts/prepare_sqa3d.py    # 전처리 (약 40초, CPU)
  .venv/bin/python scripts/validate_sqa3d.py   # 검증 (59개 항목)
  ```
- 같은 입력이면 출력도 항상 같다. 두 번 실행한 결과 JSONL 6개의 md5가 일치했다.
- GPU, Qwen3 모델 가중치, 외부 유료 API는 사용하지 않았다. 토큰 수는 Qwen3 **tokenizer**만 CPU에서 사용해 셌다.

---

## 1. 생성 파일

| 파일 | 역할 |
|---|---|
| `scripts/prepare_sqa3d.py` | 원본 → 전처리 JSONL (전체 + pilot) 생성 |
| `scripts/validate_sqa3d.py` | 전처리 결과 검증. 결과를 요약 파일의 `validation` 항목에 기록하고, 실패하면 종료 코드 1 |
| `src/thinkenough/evaluation/sqa3d_scoring.py` | 공식 SQA3D 채점 재현 + 정규화 EM |
| `src/thinkenough/data/sqa3d_prompt.py` | 정답을 읽지 않는 프롬프트 생성기 (임시 템플릿) |
| `data/processed/sqa3d/{train,validation,test}.jsonl` | 전체 공식 데이터 전처리본 (git 제외) |
| `data/processed/sqa3d/pilot/{train,validation,test}.jsonl` | pilot 300 / 100 / 100 (git 제외) |
| `outputs/preprocessing_summary.json` | 개수, 분포, 토큰 통계, 제외 사유, 검증 결과 (git 제외) |
| `outputs/label_conflicts.json` | 라벨 충돌 5건 상세 (git 제외) |
| `outputs/missing_captions.json` | 캡션 누락 장면과 해당 question_id (git 제외) |

원본 파일(`data/raw/sqa3d/`)은 읽기만 했다. `sqa_task.zip` md5가 작업 전후로 같다(`0da8bb0e...`).

---

## 2. JSONL 레코드 구조

```jsonc
{
  "question_id": 220602000182,
  "split": "test",                       // train / validation / test
  "input": {                             // ← Qwen3에 들어가는 것은 이 안의 값뿐
    "scene_id": "scene0607_00",
    "situation": "I am facing the paper towel dispenser ...",
    "question": "What color is the refrigerator on my right?",
    "scene_captions": [{"object": "chair", "caption": "this chair is facing left. it is brown."}, ...]
  },
  "reference_answers": ["brown"],        // ← 채점 전용. 입력과 분리
  "meta": {
    "caption_status": "ok",              // ok / missing
    "num_captions_available": 72, "num_captions_used": 30,
    "caption_tokens": 544, "input_tokens": 629, "prompt_template": "sqa3d_official_gpt_v0",
    "question_category": "what", "answer_category": "color",
    "flags": [],                         // duplicate_question, label_conflict
    "alternative_situation": [...],
    "position": {...}, "rotation": {...},  // 공식 채점기의 문항 구분 키로만 사용. 입력에 넣지 않음
    "annotation_scene_id": "scene0607_00"
  },
  "source": {"dataset": "SQA3D v1.0", "official_split": "test",
             "question_file": "...", "annotation_file": "...", "zip_md5": "0da8...",
             "caption_file": "...scanrefer_captions_by_scene.pkl", "caption_rule": "...", "seed": 42}
}
```

- 질문과 정답은 `question_id`로, 캡션은 `scene_id`로 연결했다.
- `validation.jsonl`의 `source.official_split`은 공식 파일 이름에 맞춰 `"val"`이다.

---

## 3. 분할별 데이터 개수

### 3.1 전체 전처리본 (전처리 전후 개수 동일, 제외 0건)

| split | 원본 질문 | 원본 정답 | 전처리 결과 | 제외 | 장면 | 공식 채점 단위* |
|---|---:|---:|---:|---:|---:|---:|
| train | 26,623 | 26,623 | 26,623 | 0 | 518 | 26,607 |
| validation | 3,261 | 3,261 | 3,261 | 0 | 65 | 3,259 |
| test | 3,519 | 3,519 | 3,519 | 0 | 67 | 3,516 |

\* 공식 `compute_accuracy`는 (scene, rotation, position, question, situation)이 같은 문항을 **하나로 합쳐서** 채점한다. 그래서 공식 정확도의 분모가 문항 수보다 작다. 재현 코드도 같은 방식으로 동작한다.

전체 전처리본에서는 문항을 하나도 지우지 않았다. 문제가 있는 문항은 `meta.flags`와 `caption_status`로 **표시만** 했다.

### 3.2 Pilot (seed=42)

| split | 원본 | 제외: 캡션 누락 | 제외: 라벨 충돌 | 제외: 중복 질문 | 후보 | 추출 | 장면 |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 26,623 | 50 | 6 | 24 | 26,543 | **300** | 223 |
| validation | 3,261 | 49 | 2 | 2 | 3,208 | **100** | 47 |
| test | 3,519 | 0 | 2 | 4 | 3,513 | **100** | 53 |

- 제외 사유는 우선순위(캡션 누락 > 라벨 충돌 > 중복)대로 하나씩만 센다.
- 층화 기준: (질문 유형 × 정답 유형). 각 층의 크기에 비례해서 개수를 배정하고(최대 잔여법), 층 안에서는 `random.Random("42:<split>")`로 뽑는다.

**질문/정답 유형 정의** (SQA3D의 `question_type`과 `answer_type`은 전부 `N/A`/`other`라 직접 만들었다)
- 질문 유형: 첫 단어가 what / is / how / can / which이면 그 단어, 아니면 other (SQA3D 논문의 분류 방식)
- 정답 유형: yes_no / number (숫자, even, odd) / direction / color / other (정해진 단어 목록과 정확히 일치하는지로 판정)

**분포 비교 (전체 대비 pilot)**

| 질문 유형 | train 전체 | pilot train (n=300) | val 전체 | pilot val (n=100) | test 전체 | pilot test (n=100) |
|---|---:|---:|---:|---:|---:|---:|
| what | 30.7% | 92 (30.7%) | 32.4% | 33 | 32.5% | 32 |
| is | 19.0% | 57 (19.0%) | 18.8% | 18 | 18.5% | 18 |
| other | 17.3% | 51 (17.0%) | 17.1% | 17 | 16.2% | 17 |
| how | 15.1% | 46 (15.3%) | 15.7% | 16 | 13.2% | 13 |
| can | 9.5% | 28 (9.3%) | 8.6% | 9 | 9.6% | 10 |
| which | 8.5% | 26 (8.7%) | 7.4% | 7 | 10.0% | 10 |

| 정답 유형 | train 전체 | pilot train | val 전체 | pilot val | test 전체 | pilot test |
|---|---:|---:|---:|---:|---:|---:|
| other | 38.1% | 115 (38.3%) | 38.6% | 39 | 39.1% | 39 |
| yes_no | 20.3% | 61 (20.3%) | 19.1% | 20 | 19.0% | 20 |
| number | 19.5% | 58 (19.3%) | 20.8% | 20 | 18.7% | 18 |
| direction | 15.1% | 46 (15.3%) | 13.6% | 14 | 16.3% | 17 |
| color | 7.0% | 20 (6.7%) | 7.8% | 7 | 6.9% | 6 |

pilot의 분포가 전체와 거의 같다.

---

## 4. Scene Captions

- 출처: ScanRefer 캡션 (`LLM/scanrefer_captions_by_scene.pkl`)
- 선택 규칙: 캡션이 30개를 넘으면 `random.Random("42:<scene_id>")`로 30개를 뽑고, **원래 순서를 유지**한다. 30개 이하면 전부 쓴다.
- 이 규칙은 **scene_id만** 보고 정해진다. 검증 결과 같은 장면의 모든 질문에 똑같은 캡션이 들어간다(질문·정답과 무관).
- 공식 코드와의 차이: 공식 코드는 seed 없는 `random.sample`(문항마다 다른 30개)을 쓴다. 우리는 재현할 수 있도록 장면 단위로 고정했다.

### 4.1 토큰 수 (Qwen3-8B tokenizer, chat template 및 thinking 토큰 제외)

| split | 사용 캡션 수 (중앙값/평균) | 캡션 토큰 (중앙값/p95/max) | 전체 입력 토큰 (중앙값/p95/max) |
|---|---|---|---|
| train | 30 / 29.7 | 607 / 751 / 1,036 | 693 / 838 / 1,141 |
| validation | 30 / 29.4 | 596 / 755 / 979 | 683 / 851 / 1,091 |
| test | 30 / 29.9 | 608 / 746 / 897 | 700 / 843 / 998 |

- 입력 토큰은 **임시 템플릿**(`sqa3d_official_gpt_v0`, 공식 GPT 프롬프트 + 1-shot 예시) 기준이다. Qwen3 프롬프트가 확정되면 `sqa3d_prompt.py`를 바꾸고 `prepare_sqa3d.py`를 다시 실행하면 된다.
- 입력이 약 1.1k 토큰 이하라서, 메모리 부담은 대부분 Long CoT **출력** 쪽에서 생길 것이다.

### 4.2 캡션 누락 장면 (`outputs/missing_captions.json`)

| split | scene_id | 문항 수 | 분할 내 비율 | Nr3D 캡션 |
|---|---|---:|---:|---:|
| train | `scene0546_00` | 50 | 0.19% | 14개 있음 |
| validation | `scene0217_00` | 49 | 1.50% | 없음 |
| test | 없음 | 0 | 0% | — |

빈 문자열로 채우지 않았다. `caption_status: "missing"`, `scene_captions: []`로 표시하고 목록에 기록했다. pilot에서는 제외했다.

**처리 방안 (팀 결정 필요)**
- train `scene0546_00`: **probe 학습에서 제외 (추천)**. 전체의 0.19%라 영향이 거의 없다. Nr3D로 대체하면 캡션 출처가 섞이므로 권장하지 않는다.
- validation `scene0217_00`: 대체할 캡션이 없다. **주 평가에서 제외하고 제외 개수를 명시 (추천)**하거나, "캡션 없음" 조건으로 따로 평가하는 방법이 있다.
- test는 누락이 없으므로 최종 결과에는 영향이 없다.

---

## 5. 채점

| 지표 | 정의 | 용도 |
|---|---|---|
| **공식 (주 지표)** | 공식 `clean_answer` + `compute_accuracy`를 그대로 옮김: 예측만 정규화한 뒤 완전 일치 / 부분 문자열 / 공백 제거 후 부분 문자열 / 단어 하나라도 겹침이면 정답. 같은 문항은 합쳐서 채점 | 공식 기준선(GPT-3 + ScanRefer 41.0)과 비교 |
| 정규화 EM (보조) | 예측과 정답 **양쪽에** 같은 정규화를 적용한 뒤, 대소문자를 무시하고 완전히 일치하는지 | 엄격한 정답 판정, d* 라벨 후보 |

- 공식 GPT 판정 지표는 유료 API가 필요해서 제외했다.
- 정답끼리 비교한 self-check: 공식 지표와 EM 모두 세 분할에서 100%가 나왔다.

**공식 채점의 알려진 특성** (원본 그대로 재현함, 검증 스크립트에 테스트로 넣어둠)

| 예측 | 정답 | 공식 | EM | 비고 |
|---|---|:---:|:---:|---|
| `Left.` | `left` | ✅ | ✅ | 정규화 |
| `2` | `two` | ✅ | ✅ | 숫자를 영단어로 변환 |
| `right side` | `left side` | ✅ | ❌ | 단어("side")가 겹치면 정답 |
| (빈 문자열) | `brown` | ✅ | ❌ | 빈 예측은 모든 문자열의 부분 문자열이라 정답 |

- 💭 Early exit 실험에서는 답을 제대로 뽑지 못해 **빈 예측**이 나올 수 있다. 그러면 공식 지표가 그 문항을 정답으로 칠 위험이 있다. `official_match(..., empty_pred_is_wrong=True)` 옵션을 만들어 두었으니, 실험에서 이 옵션을 쓸지 팀에서 정해야 한다.
- 💭 Qwen3 출력에서 `<think>...</think>` 이후의 최종 답을 뽑는 규칙은 아직 정하지 않았다. 공식 코드는 GPT 응답의 마지막 줄을 답으로 쓴다.

---

## 6. 정답 누출 방지와 검증

**누출 검증** (`validate_sqa3d.py`, 3개 분할 모두 통과)
1. `input`에는 허용된 4개 키(`scene_id`, `situation`, `question`, `scene_captions`)만 있다.
2. 캡션 안에 정답 관련 키가 없다.
3. 정답, meta, source를 임의 문자열(sentinel)로 바꿔서 프롬프트를 다시 만들어도, 3만 3천여 문항 **전부 프롬프트가 똑같고**, sentinel이 프롬프트에 나타나지 않는다. 즉 프롬프트 생성기는 정답과 GT 위치·방향을 읽지 않는다.
4. pilot도 분할 사이에 장면이 겹치지 않는다.

**참고 수치 (누출 아님)**: 정답 단어가 입력 텍스트에 자연스럽게 등장하는 비율
- 캡션에 등장: 약 51% (예: 정답 "brown"이 캡션 "a brown chair"에 등장)
- situation이나 question에 등장: 약 26% (예: "Is the door open or closed?" → "closed")

선택형 질문이나 장면 설명의 특성일 뿐 데이터 누출은 아니다. 다만 모델이 문장을 그대로 베끼는 경향이 있는지 분석할 때 참고할 수 있다.

**전체 검증 결과: 59개 항목 통과, 0개 실패** (`outputs/preprocessing_summary.json` → `validation`)
- 분할별 개수 = 원본 개수, pilot 개수 = 300/100/100
- `question_id`가 전체 33,403개에서 모두 유일
- 분할 간 `scene_id`와 (scene, situation, question) 중복 0
- 정답 누락 0, 질문과 정답의 scene_id 불일치 0
- 캡션 누락 장면이 모두 기록됨, 정상 장면 중 빈 캡션 0, 문항당 캡션 30개 이하
- pilot은 전체 데이터의 부분집합이고 레코드 내용이 동일, 표시된 문항 없음

---

## 7. 라벨 충돌 5건 (`outputs/label_conflicts.json`)

5건 모두 **위치·방향(position/rotation)까지 같은 완전히 동일한 문항**인데 정답이 서로 다르다. 수정하지 않고 `label_conflict`로 표시만 했다.

| split | scene | question | question_id → 정답 | 관찰 💭 |
|---|---|---|---|---|
| train | scene0137_00 | How many beds are on the left? | 220602001395 → `one` / 220602006439 → `two` | 세는 개수가 다름 |
| train | scene0128_00 | Is the door in front of me open or closed? | 220602007145 → `closed` / 220602012505 → `move forward` | 다른 질문의 정답이 섞인 것으로 보임 |
| train | scene0247_00 | How many chairs are facing me that are front of me? | 220602018535 → `yes` / 220602020081 → `two` | 다른 질문의 정답이 섞인 것으로 보임 |
| validation | scene0144_00 | How many chairs are on the left? | 220602011401 → `one` / 220602014810 → `two` | 세는 개수가 다름 |
| test | scene0100_00 | What color is the bathtub? | 220602002937 → `backward` / 220602028420 → `brown` | 다른 질문의 정답이 섞인 것으로 보임 |

- 공식 채점기는 이런 쌍을 하나로 합치고, **파일에서 나중에 나오는 문항의 정답**으로 채점한다. 그래서 test 결과에 미치는 영향은 최대 1단위(0.03%)다.
- 중복 질문 중 정답까지 같은 경우: train 13그룹, validation 1그룹, test 2그룹. 공식 채점에서는 각각 하나로 합쳐진다.

---

## 8. 다음 단계에서 필요한 결정과 작업

1. **캡션 누락 장면 처리 확정** (4.2절 추천안)
2. **빈 예측 처리**: 공식 지표에서 빈 예측을 오답으로 칠지 (`empty_pred_is_wrong`)
3. **d* 라벨링에 쓸 채점 기준**: 공식 매칭은 느슨해서 STOP 라벨이 일찍 붙을 수 있다. EM을 쓸지, 두 기준을 모두 기록할지
4. **Qwen3 프롬프트 확정**: thinking 모드 chat template, 1-shot 예시 유지 여부, 답 형식 지시 문구. 확정되면 `sqa3d_prompt.py`를 교체하고 토큰 수를 다시 계산
5. **최종 답 추출 규칙**: `</think>` 이후 텍스트에서 짧은 답을 뽑는 방법
6. **pilot 실행 (모델 담당과 협의)**: pilot train 300문항에 대해 Long CoT 생성, step 분할, prefix별 강제 답변 생성
7. **언어 prior 기준선**: 캡션 없이 situation+question만 주는 조건을 같은 채점기로 평가
8. **SCoT 보조 데이터셋 조사** (별도 단계)
