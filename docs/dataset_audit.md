> 데이터팀 ZIP에서 가져온 원문입니다. 아래 실행 결과·설정·설치 설명은 작성 당시 기록이며 현재 통합본의 검증 결과가 아닙니다. 현재 실행 방법과 차이는 [integration.md](integration.md)를 따릅니다.

# SQA3D Dataset Audit

- 작성일: 2026-10-16
- 담당: Dataset & Evaluation
- 목적: ThinkEnough(Qwen3 + Long CoT) 실험의 주력 데이터셋인 SQA3D의 구조, 분할, 평가 방식, 데이터 누출 위험을 확인한다.
- 재현: `.venv/bin/python scripts/audit_sqa3d.py` → `outputs/audit/sqa3d_audit.json`

**표기 규칙**

| 표시 | 의미 |
|---|---|
| ✅ 공식 | 공식 문서·코드에 명시된 내용 (출처 링크 표기) |
| 🔍 실측 | 내려받은 원본 파일을 `scripts/audit_sqa3d.py` 등으로 직접 분석해 확인한 내용 |
| 💭 추정 | 확인되지 않은 추론. 실험 전에 검증 필요 |

---

## 1. 공식 자료와 내려받은 파일

### 1.1 출처

| 자료 | URL |
|---|---|
| [S1] GitHub README | https://github.com/SilongYong/SQA3D |
| [S2] 데이터 구조 설명 | https://github.com/SilongYong/SQA3D/blob/master/assets/dataset.md |
| [S3] LLM 평가 가이드 | https://github.com/SilongYong/SQA3D/blob/master/LLM/README.md |
| [S4] LLM 평가 코드 | https://github.com/SilongYong/SQA3D/blob/master/LLM/eval_sqa3d_llm.py |
| [S5] 벤치마크·리더보드 | https://github.com/SilongYong/SQA3D/blob/master/assets/benchmarking_leaderboard.md |
| [S6] ScanQA 실행 가이드 (QA 지표) | https://github.com/SilongYong/SQA3D/blob/master/ScanQA/README.md |
| [S7] Zenodo 데이터 | https://zenodo.org/records/7792397 (API: `https://zenodo.org/api/records/7792397`) |
| [S8] ScanNet 공식 split 목록 | https://github.com/ScanNet/ScanNet/tree/master/Tasks/Benchmark (`scannetv2_{train,val,test}.txt`) |
| [S9] 논문 | Ma et al., *SQA3D: Situated Question Answering in 3D Scenes*, ICLR 2023 (https://arxiv.org/abs/2210.07474) |

- ✅ 라이선스: 데이터 CC-BY-4.0, 코드 Apache [S1][S7]
- ✅ 규모: ScanNet 650개 장면, 6.8k situation, 20.4k situation 서술, 33.4k 질문 [S1][S7]

### 1.2 Zenodo 파일 목록과 다운로드 여부 ✅ [S7]

| 파일 | 용량 | 내용 | 받음 | 이유 |
|---|---:|---|:---:|---|
| `sqa_task.zip` | 4.6 MB | QA 질문·정답 (train/val/test) + `answer_dict.json` | ✅ | 주력 데이터 |
| `answer_counter.json` | 0.3 MB | 전체 정답 빈도 (ScanQA 학습용) | ❌ | 이번 단계에 불필요 |
| `ScanQA_format.zip` | 2.8 MB | ScanQA 모델 입력 형식으로 변환한 QA | ❌ | 같은 내용을 다른 형식으로 담은 파일 |
| `localization_task.zip` | 0.8 MB | situation 위치 추정 과제 | ❌ | QA 과제가 아님 |
| `bird.zip` | 484.4 MB | BEV(위에서 내려다본) 이미지 | ❌ | 이미지라 텍스트 전용 Qwen3에는 불필요 |
| `video.zip` | 425.2 MB | egocentric 비디오 | ❌ | 비디오, 다운로드 금지 대상 |

- 🔍 `sqa_task.zip` md5 = `0da8bb0ed16ea8094d433eeda397f45a` → Zenodo 공개값과 일치

### 1.3 Scene Captions (Qwen3 텍스트 입력용)

- ✅ 위치: Zenodo가 아니라 **GitHub 저장소 `LLM/` 폴더**에 들어 있다 [S3][S4]
  - `LLM/scanrefer_captions_by_scene.pkl` (4.7 MB): ScanRefer 기반
  - `LLM/nr3d_captions_by_scene.pkl` (2.7 MB): ReferIt3D(Nr3D) 기반
- ✅ 다운로드: 공개 저장소라 별도 승인 없이 받을 수 있다. raw.githubusercontent.com에서 받았다.
- ✅ 공식 LLM 평가 방식은 3D 장면을 여러 개의 scene caption으로 대체하고, 캡션 + situation + question을 LLM에 넣는다 [S3]
- 🔍 형식: `dict[scene_id] -> list[(object_name, caption)]`
  - 예: `('tv', 'a black tv, in the direction from the entrance ... on the right side of the blue curtain ...')`
- 🔍 pickle 파일 안전성: opcode를 분석한 결과 클래스나 함수를 불러오는 명령이 없고, 기본 자료형(dict, list, tuple, str)만 들어 있다. 스크립트는 클래스 로딩을 차단한 `SafeUnpickler`로 연다.

### 1.4 로컬 저장 위치 (모두 git 제외 대상)

```
data/raw/sqa3d/
├── zenodo/sqa_task.zip                 # 원본 zip (수정 금지)
├── extracted/sqa_task/                 # zip 압축 해제본 (수정 금지)
│   ├── answer_dict.json
│   └── balanced/v1_balanced_{questions,sqa_annotations}_{train,val,test}_scannetv2.json
└── github_llm/                         # GitHub LLM/ 폴더 원본
    ├── scanrefer_captions_by_scene.pkl
    ├── nr3d_captions_by_scene.pkl
    └── v1_balanced_{questions,sqa_annotations}_test_scannetv2.json
```

- 🔍 `github_llm/`의 test JSON 2개는 zip 안의 test 파일과 **md5가 똑같다**. 따라서 공식 LLM 평가도 같은 test split을 사용한다.

---

## 2. `sqa_task.zip` 구성과 분할

- ✅ 사용할 파일: `sqa_task/balanced/*`, `sqa_task/answer_dict.json` [S2]
- ✅ split마다 **질문 파일**(`v1_balanced_questions_{split}_scannetv2.json`)과 **정답 파일**(`v1_balanced_sqa_annotations_{split}_scannetv2.json`)이 따로 있고, `question_id`로 연결한다 [S2]
- ✅ 공식 분할은 train / val / test로 이미 고정되어 있다 [S2]. **우리가 새로 나눌 필요가 없다.**

### 2.1 분할 통계 🔍

| split | 질문 수 | 장면 수 | 고유 situation (scene, 텍스트) | 고유 정답 수 | ScanNet 원본 split [S8] |
|---|---:|---:|---:|---:|---|
| train | 26,623 | 518 | 5,438 | 1,277 | ScanNet **train** 518/518 |
| val | 3,261 | 65 | 668 | 396 | ScanNet **val** 65/65 |
| test | 3,519 | 67 | 719 | 403 | ScanNet **val** 67/67 |
| **합계** | **33,403** | **650** | **6,825** | | |

- 🔍 합계(33.4k 질문, 650개 장면, 6.8k situation)가 공식 수치 [S1]와 일치한다.
- 🔍 SQA3D의 val과 test는 둘 다 **ScanNet val 장면**에서 왔지만 서로 다른 장면이다. ScanNet의 test 장면(100개)은 사용하지 않았다.
- 🔍 모든 scene_id가 `sceneXXXX_00` 형식이다. 장면당 첫 번째 스캔만 사용한다.

---

## 3. 필드 정리

### 3.1 질문 파일 (`questions[]`)

| 필드 | 타입 | 공식 "useful" [S2] | 설명 / 우리 용도 |
|---|---|:---:|---|
| `question_id` | int | ✅ | 정답 파일과 연결하는 키. 🔍 split 안에서 유일하고, split 사이에도 겹치지 않는다 |
| `scene_id` | str | ✅ | ScanNet 장면 ID (`scene0380_00`). 캡션을 찾는 키 |
| `situation` | str | ✅ | 1인칭 상황 서술. **프롬프트 입력** |
| `question` | str | ✅ | 질문. **프롬프트 입력** |
| `alternative_situation` | list[str] | ❌ | 같은 상황을 다르게 표현한 서술. 데이터 증강 후보 |

### 3.2 정답 파일 (`annotations[]`)

| 필드 | 타입 | 공식 "useful" [S2] | 설명 / 우리 용도 |
|---|---|:---:|---|
| `question_id` | int | ✅ | 질문과 연결 |
| `scene_id` | str | ✅ | 🔍 질문 파일의 값과 100% 일치 |
| `answers[0]['answer']` | str | ✅ | **정답**. 🔍 모든 문항에 정답이 정확히 1개 있다 |
| `answers[0]['answer_confidence']` | str | ❌ | 🔍 전부 `"yes"`라 정보가 없다 |
| `position` {x,y,z} | dict | ❌ | 에이전트의 실제 위치 (ground truth) |
| `rotation` {_x,_y,_z,_w} | dict | ❌ | 에이전트의 실제 방향 (quaternion) |
| `answer_type` / `question_type` | str | ❌ | 🔍 각각 전부 `"other"` / `"N/A"`라 질문 유형 분류에 쓸 수 없다 |

- ✅ `answer_dict.json` = `[answer→class, class→answer]` 매핑, 706개 클래스 [S2]. 분류형 모델용이라 생성형인 Qwen3에는 필수가 아니다.
- ⚠️ `position`/`rotation`을 프롬프트에 넣으면 리더보드 기준으로 **"ground truth situation 사용"** 설정이 된다. 사용하면 반드시 보고서에 밝혀야 한다 ✅ [S5]

### 3.3 정답 분포 🔍

- 정답의 92.7%(train)가 한 단어다. 두 단어 6.2%, 세 단어 이상 1.1%.
- train 상위 정답: yes(2,818), no(2,576), right(1,485), left(1,414), one(1,256), two(1,242), even, odd, brown, backward ...
- 질문 첫 단어 (train): What 30.7%, Is 19.0%, How 15.1%, Can 9.5%, Which 8.5%, Are, If, Where ...
- val/test 정답 중 train에도 나온 정답 비율: val 96.3%, test 96.1%
- val/test 정답 중 `answer_dict`에 있는 정답 비율: val 97.6%, test 97.3%
- train 최빈 정답("yes")만 찍었을 때 정확도: val 9.3%, test 10.9%

---

## 4. 공식 평가 방식

| 항목 | 내용 | 근거 |
|---|---|---|
| QA 지표 | `Acc (top-1)` = 정답 완전 일치 비율 | ✅ [S5][S6] |
| LLM 기준선 | GPT-3 + ScanRefer captions: test 41.00 | ✅ [S5] |
| 기타 기준선 | Blind test (3D 없음) 43.65, ScanQA 46.58, 사람 90.06 | ✅ [S5][S6] |
| LLM 평가 데이터 | test split | ✅ [S4] |
| 캡션 사용 | 장면당 최대 **30개를 무작위 추출**(`random.sample`, seed 없음)해서 `. `로 연결 | ✅ [S4] |
| 프롬프트 | `Context: {captions}\nQ: {situation} {question}\nA:` (+ 1-shot 예시) | ✅ [S4] |
| 정답 매칭 | 아래 참고 | ✅ [S3][S4] |

**공식 LLM 매칭 규칙** (`compute_accuracy`) ✅ [S4]
1. 예측만 `clean_answer()`로 정규화한다: 소문자화, 특수문자 제거, 오타 교정, 숫자→영단어(`2`→`two`), 관사(a/an/the) 제거
2. 아래 중 하나라도 맞으면 정답으로 친다:
   - 완전 일치
   - 예측이 정답의 부분 문자열
   - 공백을 지우고 비교했을 때 부분 문자열
   - **단어가 하나라도 겹침**
3. 선택 사항으로 GPT 판정 지표(`--use_gpt_metric`)가 있다. 공식 문서도 이 방식을 권장한다.

- ✅ 원저자도 이 매칭이 "a bit aggressive"하다고 인정했다 [S3].
- 💭 단어가 하나만 겹쳐도 정답 처리하므로 정확도가 부풀려질 수 있다. 예를 들어 정답이 "left side"일 때 예측 "right side"도 정답이 된다.
- 💭 우리 연구에서는 d* 라벨 자체가 채점 결과로 정해지므로, 매칭 규칙이 느슨하면 STOP 라벨이 실제보다 일찍 붙는다. **엄격한 EM(정규화 후 완전 일치)을 주 지표로 쓰고, 공식 LLM 매칭은 비교용으로 함께 보고하는 것을 권장한다.**

---

## 5. Test 정답 공개 여부, 분할 간 중복, 데이터 누출

### 5.1 Test 정답 공개 여부
- 🔍 **공개되어 있다.** zip 안의 test 정답 파일에 3,519문항 모두 정답이 들어 있고, GitHub `LLM/` 폴더에도 같은 파일이 공개되어 있다.
- ✅ 리더보드는 결과를 이메일로 제출하는 방식이고 [S5], 별도의 비공개 test 서버는 없다.
- 💭 위험: 정답이 손에 있으니, 프롬프트나 τ, probe 구조를 test 결과를 보면서 고르면 사실상 test에 맞춘 튜닝이 된다. 연구 규칙으로 막아야 한다 (6절 참고).

### 5.2 분할 간 중복 🔍

| 비교 | scene_id 중복 | question_id 중복 | (scene, situation, question) 완전 중복 | situation 문장 중복* | question 문장 중복 |
|---|---:|---:|---:|---:|---:|
| train–val | **0** | 0 | 0 | 39 | 694 (val의 24.2%) |
| train–test | **0** | 0 | 0 | 37 | 741 (test의 24.4%) |
| val–test | **0** | 0 | 0 | 10 | 275 (test의 9.1%) |

\* `alternative_situation` 포함

- 🔍 **장면 단위로 완전히 분리되어 있다.** 같은 장면이 두 split에 동시에 나오지 않는다.
- 🔍 겹치는 situation과 question 문장은 "I am entering the room.", "What is in front of me?"처럼 흔한 표현이고, **서로 다른 장면**에 붙어 있다. 그래서 정답이 새어 나가는 누출은 아니다.
- 💭 다만 질문 문장이 24% 겹치고 정답 분포가 치우쳐 있어서 **언어 사전지식(prior)만으로도 점수가 높게 나온다.** 3D 없이 푼 Blind test가 43.65로 ScanQA(46.58)와 큰 차이가 없다 ✅ [S6]. 우리 실험에도 **캡션 없이 situation+question만 주는 기준선**을 넣어야, 캡션이 실제로 도움이 됐는지 보일 수 있다.

### 5.3 split 내부의 라벨 노이즈 🔍
- 같은 (scene, situation, question)이 split 안에 여러 번 나오는 경우: train 16, val 2, test 3
- 그중 **정답이 서로 다른 경우**: train 3, val 1, test 1
  - test 예: `scene0100_00 | I am washing my face at the sink. | What color is the bathtub?` → `['backward', 'brown']`
  - train 예: `How many beds are on the left?` → `['one', 'two']`
- 💭 수가 적어서 전체 정확도에는 큰 영향이 없다. 하지만 d* 라벨링 단계에서 이런 문항을 제외할지 정해야 한다.

### 5.4 Scene Captions 관련 누출 검토
- 🔍 ScanRefer 캡션의 장면 703개 = ScanNet train 562개 + val 141개. SQA3D의 val/test 장면은 ScanNet val에 속한다.
- 💭 캡션은 ScanRefer/Nr3D 작업자가 SQA3D와 **별개로** 쓴 물체 지시 문장이다. SQA3D 질문이나 정답을 보고 만든 것이 아니므로 **정답이 직접 새는 통로는 아니다.** 물체의 색, 모양, 위치 정보가 담긴 것은 과제가 의도한 장면 정보다.
- 💭 캡션의 방향 표현("on the left")은 캡션 작성자의 시점 기준이다. SQA3D situation의 1인칭 시점과 다르므로 모델이 혼동할 수 있다.

### 5.5 사전학습 오염 가능성 💭
- SQA3D는 2022–2023년부터 GitHub과 Zenodo에 test 정답까지 공개되어 있다. 따라서 Qwen3 사전학습 데이터에 포함됐을 가능성을 배제할 수 없다. 이건 검증할 수 없는 부분이므로, 보고서에 한계로 적는 것을 권장한다.

---

## 6. Qwen3 입력 관점의 확인 사항

### 6.1 캡션 커버리지 🔍

| 캡션 | 캡션이 있는 장면 | 장면당 캡션 수 (min/중앙값/max) | train 커버 | val 커버 | test 커버 |
|---|---:|---|---|---|---|
| ScanRefer | 703 | 9 / 60 / 257 | 517/518 (99.8%) | 64/65 (98.5%) | **67/67 (100%)** |
| Nr3D | 641 | 7 / 56 / 259 | 474/518 (92.1%) | 60/65 (92.5%) | 62/67 (93.1%) |

- 캡션이 없는 장면 (ScanRefer): train `scene0546_00`, val `scene0217_00`
- 공식 코드는 캡션이 없으면 빈 문자열을 넣는다 ✅ [S4]

### 6.2 캡션 길이 (Qwen3 tokenizer 기준) 🔍

| 캡션 | 전체 캡션 사용 시 (중앙값/p95/max) | 공식 30개 사용 시 (중앙값/max) |
|---|---|---|
| ScanRefer | 1,248 / 2,626 / 5,230 tokens | 609 / 1,044 tokens |
| Nr3D | 688 / 2,133 / 3,823 tokens | 351 / 713 tokens |

- 💭 전체 캡션을 넣어도 최대 약 5.2k 토큰이다. Qwen3의 기본 context(32k)에 충분히 들어가므로, **무작위 30개 추출 대신 전체 캡션을 정해진 순서로 넣으면 재현성을 확보할 수 있다.** 다만 입력이 길어지면 Long CoT 생성 시 메모리와 시간이 늘어나므로, 모델 담당과 함께 정해야 한다.

### 6.3 권장 사용 규칙 💭 (팀 합의 필요)
- **train**: probe 학습 데이터 (CoT 생성 + 라벨링)
- **val**: τ 선택, probe 구조 선택, 프롬프트 선택 등 **모든 튜닝**
- **test**: 최종 결과 보고에만 사용. 튜닝이 끝난 뒤 한 번만 평가한다
- 주 캡션은 ScanRefer (커버리지가 높고 공식 LLM 기준선과 같다), Nr3D는 보조 비교용
- `position`/`rotation`은 기본적으로 사용하지 않는다 (사용하면 별도 설정으로 표기)

---

## 7. 폴더 구조 제안 (전처리 단계)

현재 `data/raw/sqa3d/` 구조(1.4절)는 그대로 유지하고, 전처리 결과물은 아래처럼 추가할 것을 제안한다. 아직 만들지 않았으며, 승인을 받은 뒤 생성한다.

```
data/
├── raw/sqa3d/                    # 원본 (읽기 전용, 수정·삭제 금지)
│   ├── zenodo/  extracted/  github_llm/
├── interim/sqa3d/                # 중간 산출물 (다시 만들 수 있음)
│   ├── merged_{train,val,test}.jsonl      # 질문+정답을 question_id로 합친 파일
│   └── scene_text/{scanrefer,nr3d}.json   # scene_id → 장면 텍스트 (정해진 순서)
└── processed/sqa3d/              # 실험에 바로 쓰는 최종 입력
    ├── {train,val,test}.jsonl             # id, scene_id, situation, question, answer, scene_text
    └── README.md                          # 생성 스크립트, 버전, 필터링 규칙 기록
scripts/
├── audit_sqa3d.py                # (완료) 데이터 감사
└── prepare_sqa3d.py              # (예정) raw → interim → processed 변환
src/thinkenough/
├── data/sqa3d.py                 # (예정) 로딩·병합·장면 텍스트 생성 함수
└── evaluation/answer_match.py    # (예정) 정규화 + 엄격 EM + 공식 매칭 재현
```

---

## 8. 미해결 문제

1. **채점 기준 확정**: 엄격 EM을 주 지표로 할지, 공식 LLM 매칭이나 GPT 판정을 쓸지. d* 라벨이 이 기준에 따라 달라진다.
2. **캡션 사용 방식**: 전체 캡션을 고정 순서로 넣을지, 공식 방식처럼 30개를 추출할지(seed 고정). 캡션을 어떤 순서로 정렬할지.
3. **probe 학습 데이터 규모**: train 26.6k문항 × Long CoT × prefix마다 강제 답변 생성은 GPU 부담이 크다. 일부 문항만 뽑아 쓸지, 뽑는다면 몇 개를 어떤 기준으로 뽑을지(모델 담당과 협의).
4. **라벨 노이즈 처리**: 정답이 서로 다른 중복 문항 5건을 제외할지.
5. **캡션 없는 장면 2개**: 해당 문항을 제외할지, 빈 캡션으로 둘지.
6. **언어 prior 기준선**: 캡션 없이 situation+question만 주는 기준선을 실험표에 추가할지.
7. **SCoT 보조 데이터셋**: 별도로 감사가 필요하다 (용량, 형식, Long CoT 확장 방법).
8. **사전학습 오염**: 검증이 불가능하므로 한계로 적을지.
