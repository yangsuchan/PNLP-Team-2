> 데이터팀 ZIP에서 가져온 원문입니다. 아래 실행 결과·설정·설치 설명은 작성 당시 기록이며 현재 통합본의 검증 결과가 아닙니다. 현재 실행 방법과 차이는 [integration.md](integration.md)를 따릅니다.

# Checkpoint 선택 옵션 — 사용 설명서

## 설치와 변경 범위

ZIP 내부의 src, scripts, tests, configs, docs 폴더를 기존 ThinkEnough 프로젝트 루트에 병합한다. 이 패키지는 새 파일만 추가한다. 기존 모델 출력 검증기와 함께 사용할 수 있다. 기존 build_checkpoints.py와 validate_checkpoints.py는 수정하지 않는다.

Python 3.10 이상. 선택 스크립트는 표준 라이브러리만 사용한다. GPU/transformers 없이도 선택·검증 가능하다. 테스트 실행에만 pytest가 필요하다.

## 바로 시험하기 (실제 모델 출력 불필요)

VS Code 터미널에서 프로젝트 루트로 이동한다. 아래는 Windows Python launcher `py` 기준이다. 사용 중인 환경에 맞춰 `python`으로 바꿔도 된다.

```powershell
py scripts/build_checkpoints.py --no-tokenizer --output outputs/checkpoints_all_demo.jsonl
py scripts/select_checkpoints.py --input outputs/checkpoints_all_demo.jsonl --output outputs/checkpoints_selected_demo.jsonl --strategy uniform --max-checkpoints 3
py scripts/select_checkpoints.py --input outputs/checkpoints_all_demo.jsonl --verify outputs/checkpoints_selected_demo.jsonl --strategy uniform --max-checkpoints 3
```

첫 명령은 기존 synthetic fixture로 전체 checkpoint를 만든다. 두 번째는 최대 3개를 고르고, 세 번째는 선택 결과를 원본과 비교한다. 모두 가짜 데이터이며 성능 실험이 아니다. 정상 종료 시 select/verify 출력은 `status: pass`이다.

선택 JSONL 옆에 `출력파일명.selection.json` 보고서가 자동 생성된다. 문항별 원래 개수, 선택 개수, 선택된 원본 index, 설정, 원본/출력 SHA-256이 기록된다. `forced_answer_calls_saved`는 checkpoint당 강제 답변 1회 가정의 호출 감소량이며 측정 GPU 시간이나 온라인 추론 토큰 절감량이 아니다.

## 네 가지 전략

| 전략 | 옵션 | 의미 |
|---|---|---|
| all (기본) | 없음 | 전체 선택 |
| every_n | --interval 3 | 기존 checkpoint index 0,3,6,…와 마지막 선택 |
| uniform | --max-checkpoints 10 | 원본 checkpoint index를 기준으로 최대 10개 고르게 선택 |
| token_interval | --interval 128 | 직전 선택 위치부터 128개 이상 추론 토큰이 진행된 첫 구간 끝 선택 |

시작 index 0과 마지막 checkpoint는 항상 포함한다. 최대 개수에 둘 다 포함하며 최소값은 2이다. 추론이 비어 시작과 끝이 같은 경우에는 1개만 남는다. 마지막 구간은 간격보다 짧아도 포함한다. 원래 checkpoint가 제한 개수보다 적으면 전부 사용한다.

`every_n`은 기존 분할 단위를 따른다. sentence로 만든 입력이면 N문장, line이면 N구간이다. 분할 자체는 기존 `build_checkpoints.py --segmenter sentence|line|numbered_step`에서 지정한다.

`uniform`은 **토큰 수가 아니라 원본 checkpoint index 기준**이다. 선택 위치 공식은 i*(전체개수-1)//(선택개수-1)이다. 난수를 사용하지 않는다.

`token_interval`은 원본 생성 ID에 정렬된 token_count_source=generated_ids와 exact 또는 exact_up_to_whitespace가 모든 checkpoint에 있어야 한다. 재토큰화 추정값, missing ID, 문장 경계가 토큰 내부인 경우에는 오류로 멈춘다. 문장 경계를 새로 만들거나 토큰 중간에서 자르지 않는다. 데모의 --no-tokenizer 출력은 이 전략에 사용할 수 없다.

## 팀 논의 후 설정만 변경하기

다음 설정 파일들이 포함되어 있다.

- configs/checkpoint_selection/all.json
- configs/checkpoint_selection/every_3.json
- configs/checkpoint_selection/uniform_10.json
- configs/checkpoint_selection/tokens_128.json

```powershell
py scripts/select_checkpoints.py --input outputs/checkpoints_all.jsonl --output outputs/checkpoints_uniform10.jsonl --config configs/checkpoint_selection/uniform_10.json
```

uniform_10.json의 내용:

```json
{"checkpoint_selection": {"strategy": "uniform", "max_checkpoints": 10}}
```

설정 파일 대신 --strategy/--interval/--max-checkpoints를 써도 된다. 혼용은 오류로 처리해 우선순위 혼동을 방지한다. 설정에는 선택 관련 값만 넣는다. segmenter는 전체 checkpoint 생성 단계에서 정한다. include_initial/include_final은 생략하거나 true만 허용한다.

## 실제 실험 흐름

1. A의 원본 생성 결과를 validate_generation_output.py로 검사한다.
2. 기존 build_checkpoints.py로 전체 checkpoint를 생성한다.
3. 기존 validate_checkpoints.py로 **전체 파일**을 검증한다.
4. select_checkpoints.py로 선택 파일과 설정 보고서를 만든다.
5. 동일한 설정으로 --verify를 실행한다.
6. A에게 선택 파일을 전달해 해당 checkpoint_id에 대해서만 강제 답변을 요청한다.
7. 라벨링/평가에는 해당 선택 파일과 연결되는 답변·예측을 사용한다.

예:

```powershell
py scripts/select_checkpoints.py --input outputs/checkpoints_all.jsonl --verify outputs/checkpoints_uniform10.jsonl --config configs/checkpoint_selection/uniform_10.json
```

선택 파일은 checkpoint_index가 불연속이므로 기존 validate_checkpoints.py에 넣지 않는다. --verify는 전체 원본에서 선택을 다시 계산하고, 선택 레코드 전체를 비교한다. ID뿐 아니라 prefix와 모든 원본 필드 및 순서까지 같아야 통과한다. 원본 파일 자체의 토큰 정렬 정확성을 대신 검증하지는 않는다.

## 보존되는 정보와 평가 해석

원본 checkpoint_id, checkpoint_index, reasoning_prefix, segment_text, 토큰 위치, hidden_state_position, is_final을 모두 그대로 유지한다. num_checkpoints도 원래 전체 개수이다. 선택 후 개수는 별도 보고서의 selected_count를 확인한다.

건너뛴 문장까지 포함한 누적 prefix는 유지되지만 segment_text는 원래 해당 한 구간이다. 따라서 선택된 두 행의 prefix 차이가 바로 다음 행의 segment_text라고 가정하면 안 된다. 선택된 index를 0,1,2로 다시 붙이지 않는다. 이미 선택한 파일을 입력으로 다시 선택하는 것도 거부한다. 매번 전체 파일에서 시작한다.

동일한 checkpoint_id가 서로 다른 선택 설정에도 등장하는 것은 의도된 동작이다. 실험 설정별 결과 파일과 보고서를 따로 유지하고 서로 섞지 않는다. 기존 라벨링은 checkpoint_id를 연결하고 원본 index로 정렬하므로 선택된 부분집합을 전달할 수 있다.

기존 offline evaluator의 fixed_k_sim은 **선택된 목록에서 k번째 위치**를 뜻한다. 예를 들어 [0,3,6,9]에서 fixed_2_sim은 원본 index 6이다. mean_exit_checkpoint도 선택 목록의 위치를 평균낸다. 이를 원본 문장 수나 원본 checkpoint index로 해석하면 안 된다. 서로 다른 선택 전략의 fixed-k를 같은 추론량이라고 비교하지 않는다.

uniform은 전체 추론 길이를 알아야 하는 **오프라인 학습 데이터 선택** 방식이다. 실제 온라인 조기 종료 정책은 별도로 구현해야 한다. 매 N구간/토큰 간격 또한 여기서는 저장된 출력에 적용하는 기능이며 모델 생성 루프를 바꾸지 않는다.

## 오류와 테스트

정상 종료 코드는 0, 선택/검증 실패는 1, CLI 문법 오류는 2이다. 입력과 출력 경로가 같으면 거부한다. 전체 입력 검증에 실패하면 기존 출력 파일은 덮어쓰지 않는다. 성공 시 기존 출력 경로는 갱신되므로 다른 실험은 다른 파일명을 권장한다. 잘못된 JSON, 중복 ID, 부분 checkpoint 집합, 음수/0 간격, 잘못된 설정 이름 등을 거부한다.

```powershell
py -m pytest tests/test_checkpoint_selection.py -v
py -m pytest tests/ -q
```

실제 Qwen3 생성과 원본 토큰 정렬의 검증은 별도이며 synthetic 테스트 통과가 이를 보장하지 않는다.

2026-10-31 검증: 신규 테스트 30개 통과. 기존 테스트 포함 211 passed, 7 skipped. 기존 transformers 의존 토큰 정렬 테스트 7개는 환경에서 실행되지 않았다.
