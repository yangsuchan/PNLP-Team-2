# 모델·데이터팀 통합 실행 안내

이 문서는 현재 통합본 기준이다. 함께 들어온 데이터팀 보고서는 ZIP 원문에
가까운 과거 기록이며, 그 안의 테스트 결과는 이번 통합본의 실행 결과가 아니다.

## 포함 범위

- 기존 Qwen3-VL/mock 추론과 데이터팀의 출력 검증 → checkpoint 생성 → 선택·검증.
- 데이터팀의 SQA3D 전처리·채점·라벨링·offline 평가 도구와 synthetic fixtures.
- 모델 가중치, SQA3D 원본·가공 데이터, 팀의 outputs, 캐시, pickle은 포함하지 않았다.
- hidden state 추출, 실제 forced-answer 생성, probe 학습, 온라인 early exit는 미구현이다.
  라벨링 도구가 있다고 실제 forced-answer 추론까지 구현된 것은 아니다.
- 데이터셋은 아직 확정하지 않았다. SQA3D 입력을 자동으로 이미지 입력으로
  바꾸지 않는다. 후보 조사: [spatial_dataset_pilot.md](spatial_dataset_pilot.md).

## 설치

서버는 Python 3.12 가상환경을 권장한다. 로컬과 서버에서 같은 버전을 사용한다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,evaluation]'
python -m pytest tests/ -q
```

`evaluation` extra의 scikit-learn은 데이터팀 AUROC/AP 계산에 필요하다.
출력 검증·checkpoint 선택 자체는 표준 라이브러리만 사용한다.
데이터팀 ZIP의 requirements 전체를 설치하거나 기존 환경을 덮어쓰지 않는다.

## 입력 계약

JSONL 한 줄에 이미지 기반 문항 하나를 둔다. 기존 최소 입력도 읽을 수 있다.

```json
{"sample_id":"spatial_001","question_id":"scene-A-q1","split":"validation","image_path":"data/samples/sample_image.png","question":"Which object is behind the chair? A. table B. lamp","prompt_version":"spatial_mcq_v1"}
```

- `sample_id`, `image_path`, `question`은 필수이다.
- `question_id`는 음이 아닌 정수 또는 비어 있지 않은 문자열(`/` 제외)이다.
  생략하면 sample_id를 쓴다. 숫자 ID와 문자열 ID는 타입을 보존한다.
- split은 train / validation / test. 기존 입력과의 호환 기본값은 test이지만
  실제 실험에서는 반드시 지정한다. prompt_version의 기본값은 image_question_v1이다.
- 선택지는 question에 포함한다. 정답·정답 bbox·정답 위치를 넣지 않는다.
  metadata는 모델 프롬프트와 팀 전달 JSONL에 복사하지 않는다.
- 한 run에서 question_id는 유일해야 한다. 서로 다른 데이터셋의 ID가 겹치면
  데이터셋 이름으로 문자열 ID를 구분하고 원본 ID와 매핑을 별도 보관한다.
- 경로는 저장소 루트 기준이다. 현재 한 문항에 이미지 한 장만 지원한다.

## GPU 없이 연결 확인

```bash
python scripts/run_inference.py --backend mock --run-id mock_smoke
python scripts/validate_generation_output.py --input outputs/mock_smoke/generations.jsonl --strict --report outputs/mock_smoke/generation_validation.json
python scripts/build_checkpoints.py --input outputs/mock_smoke/generations.jsonl --output outputs/mock_smoke/checkpoints_all.jsonl --no-tokenizer
python scripts/validate_checkpoints.py --generations outputs/mock_smoke/generations.jsonl --checkpoints outputs/mock_smoke/checkpoints_all.jsonl --report outputs/mock_smoke/checkpoint_validation.json
python scripts/select_checkpoints.py --input outputs/mock_smoke/checkpoints_all.jsonl --output outputs/mock_smoke/checkpoints_selected.jsonl --strategy uniform --max-checkpoints 3
python scripts/select_checkpoints.py --input outputs/mock_smoke/checkpoints_all.jsonl --verify outputs/mock_smoke/checkpoints_selected.jsonl --strategy uniform --max-checkpoints 3
```

mock는 synthetic=true이고 ID와 텍스트가 실제 tokenizer에 대응하지 않는다.
따라서 실제 tokenizer의 decode 검증 대상으로 사용하지 않는다.
`--no-tokenizer`에서는 토큰 정렬·hidden state 위치가 null이며, 통과해도
실제 모델 정렬을 검증한 것이 아니다. token_interval 선택도 할 수 없다.
실행을 반복할 때 run-id와 출력 파일명을 바꾼다.

## 출력과 호환성 변경

각 실행은 `outputs/<run_id>/`에 저장한다.

| 파일 | 용도 |
| --- | --- |
| run_manifest.json | 실행 완료 여부·선택/완료 문항 수 |
| generations.jsonl | 데이터팀 전달용 원본 생성 기록 |
| results.jsonl | 기존 InferenceResult 필드와 추가 메타데이터 |
| sample_000000.json 등 | 문항별 상세 결과 |

- 기존 run 디렉터리는 재사용하지 않는다. 잘못된 입력 경로는 실패하며 mock로 대체하지 않는다.
- 완료된 문항은 즉시 저장한다. 중단된 run은 부분 결과이며 자동 재개하지 않는다.
  run_manifest.json의 status=complete와 문항 수를 확인한다.
- 기존 스크립트의 `cot_<backend>_<timestamp>.jsonl` 출력 경로는 위 구조로 변경되었다.
- raw_output은 이제 특수 토큰을 보존한다. 가독성을 위한 final_answer는 별도로 유지한다.
- reasoning_token_count는 `<think>`와 `</think>`를 제외한다. 미완결이면 null이다.
  과거 출력의 닫는 태그 포함 카운트와 직접 비교하면 안 된다.
- final_answer_text는 닫는 태그 뒤의 공백과 EOS 문자열을 그대로 보존한다.
- 문자열 ID의 checkpoint_id는 `run/s:<encoded-id>/segmenter/index`이다.
  기존 정수 ID 형식은 유지한다. checkpoint_id를 분해하지 말고 불투명한 키로 연결한다.
- save_token_ids=false는 전달 계약에 맞지 않으므로 CLI에서 거부한다.
- revision이 확인되지 않은 과거 결과 파일을 추정값으로 변환하지 않는다.

## GPU 서버에서 실제 실행

서버의 GPU/VRAM과 CUDA를 먼저 확인한다. 모델 다운로드·GPU 실행은 실제 서버에서
진행하며, 로컬 mock 명령과 구분한다. CUDA용 torch는 서버 환경에 맞는 빌드를 설치한다.

```bash
nvidia-smi
# 서버 CUDA에 맞는 torch를 설치한 뒤:
python -m pip install -e '.[server,dev,evaluation]'
python -m pip freeze > server_environment.txt
python scripts/run_inference.py --backend qwen3_vl --input data/samples/spatial_inputs.jsonl --max-samples 1 --run-id server_smoke
```

`spatial_inputs.jsonl`은 위 입력 계약대로 별도 준비하는 실제 파일이며 저장소에
들어 있는 데이터가 아니다. 설정의 model.revision은 가능하면 실제 HF commit hash로
고정한다. 로더는 실제 모델 commit에 processor를 고정하고 결과에 기록한다.
40자리 모델/tokenizer commit이 해결되지 않으면 실제 전달 파일 작성을 거부한다.
실제 dtype과 EOS를 기록하며 현재 runner는 양자화를 설정하지 않는다.

현재 기본 max_new_tokens=40960이다. 실행 전 GPU 예산을 확인하고, 변경이 필요하면
합의한 별도 설정을 만들어 `--config`로 지정한다. 이 통합에서는 값을 바꾸지 않았다.

검증과 정렬에는 generations.jsonl에 기록된 **tokenizer ID@commit 전체**를 사용한다.
아래 HASH는 실제 값으로 치환한다. 모델 가중치를 다시 로드하는 명령은 아니다.

```bash
python scripts/validate_generation_output.py --input outputs/server_smoke/generations.jsonl --tokenizer Qwen/Qwen3-VL-8B-Thinking@HASH --local-files-only --report outputs/server_smoke/generation_validation.json
python scripts/build_checkpoints.py --input outputs/server_smoke/generations.jsonl --output outputs/server_smoke/checkpoints_all.jsonl --tokenizer Qwen/Qwen3-VL-8B-Thinking@HASH --local-files-only
python scripts/validate_checkpoints.py --generations outputs/server_smoke/generations.jsonl --checkpoints outputs/server_smoke/checkpoints_all.jsonl --report outputs/server_smoke/checkpoint_validation.json
```

이후 선택은 mock와 같은 select/verify 명령을 사용한다.
실험에서는 `--references`에 같은 question_id/split의 참조 JSONL을 전달해 소속도 검증한다.
length/unclosed_think는 완결 CoT와 구분한다. 엄격한 완결 검증은 --strict를 사용한다.
토큰 위치는 원본 ID에 exact / exact_up_to_whitespace로 정렬된 것만 후속 추출에 쓴다.
boundary_inside_token·start_inside_token을 정확한 hidden state 경계로 쓰지 않는다.

## 라벨·평가와 Git

SQA3D의 normalized EM·공식 매칭은 SQA3D 전용이다. 다른 데이터셋의
MCQ·수치·bbox 채점에 그대로 사용하지 않는다. 새로운 채점 계약은 데이터팀과 별도로 정한다.
기본 synthetic fixture로 실행하는 라벨·평가는 데모이며 연구 결과가 아니다.

출력·모델·대용량 데이터는 기존 .gitignore로 제외한다.
server_environment.txt도 공유 방법을 확인하고 비밀값을 기록하지 않는다.
로컬 검증 환경은 Python 3.15이며 권장 서버 Python 3.12 실행은 별도 검증 대상이다.
이번 구현에서는 커밋·push·서버 접속을 수행하지 않는다.

## 가져온 코드의 출처와 이번 검증

- 출처: 사용자가 전달한 `ThinkEnough_공유.zip`.
- ZIP SHA-256: `af020031a264b56c8933c660991e295a789dd8799c8fa49752ab3a7f0144024d`.
- src/data·evaluation·generation, scripts, 테스트와 작은 synthetic fixtures,
  checkpoint 선택 설정 및 관련 문서를 선별했다. ZIP의 설치 지시는 실행하지 않았다.
- 라벨·평가 테스트가 제외된 data/processed 파일을 직접 읽고 있어, 필요한
  synthetic checkpoint 예시만 tests/fixtures/synthetic_checkpoints.jsonl로 옮겼다.
- 주요 수정: 전달용 exporter, 원본 특수 토큰 보존, 추론 토큰 경계·미완결 처리,
  문자열 ID 지원, tokenizer revision 일치 검사, run별 출력 보호.
- 검증: `python -m pytest tests/ -q` → **244 passed, 7 skipped**.
  7개는 transformers/tokenizer 캐시가 필요한 기존 정렬 테스트이다.
- 테스트에는 Qwen runner의 가짜 모델·tokenizer를 이용한 원문 보존, EOS/길이 제한,
  snapshot 고정 및 freeze 검사가 포함된다. 실제 Qwen 실행 결과가 아니다.
- CLI 연결: mock 생성 → 엄격한 출력 검증 → checkpoint 생성·검증 → uniform 선택·재검증,
  synthetic 라벨 생성·검증 → offline 평가를 임시 경로에서 실행했다.
- `compileall`, `pip check`, 기존 tracked 변경의 `git diff --check`도 통과했다.
- 실행 환경: Python 3.15, pytest 9.2.0, PyYAML 6.0.3, Pillow 13.0.0,
  scikit-learn 1.9.1. Python 3.12 서버·CUDA·실제 모델 정렬은 미검증이다.
