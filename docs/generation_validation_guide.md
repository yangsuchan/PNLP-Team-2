> 데이터팀 ZIP에서 가져온 원문입니다. 아래 실행 결과·설정·설치 설명은 작성 당시 기록이며 현재 통합본의 검증 결과가 아닙니다. 현재 실행 방법과 차이는 [integration.md](integration.md)를 따릅니다.

# Qwen3 원본 생성 결과 검증기

2026-10-31. `docs/qwen3_output_spec.md` 초안 기준. GPU와 모델 가중치는 필요하지 않음.

## 설치

이 ZIP을 기존 ThinkEnough 프로젝트 루트에 풀어 `src/`, `scripts/`, `tests/`, `docs/`를 합친다. 기존 파일 수정은 없다. Python 3.10 이상. 기본 실행은 표준 라이브러리만 사용한다. 테스트는 pytest, 선택적 tokenizer 검증은 transformers가 필요하다.

## 실행 (프로젝트 루트)

```bash
python scripts/validate_generation_output.py --input generations_train_run01.jsonl --references data/processed/sqa3d/pilot/train.jsonl --report outputs/generation_validation.json
```

Windows PowerShell에서도 위 한 줄을 그대로 실행할 수 있다. 입력 파일명은 A가 전달한 실제 이름으로 바꾼다. 여러 split 파일을 확인하려면 `--references` 뒤에 해당 파일들을 공백으로 나열한다. 전달받을 문항이 전체 데이터라면 pilot 대신 전체 파일을 지정한다.

정확한 tokenizer로 원본 ID와 텍스트까지 확인:

```bash
python scripts/validate_generation_output.py --input generations_train_run01.jsonl --references data/processed/sqa3d/pilot/train.jsonl --tokenizer Qwen/Qwen3-8B@ACTUAL_COMMIT_HASH --report outputs/generation_validation.json
```

`ACTUAL_COMMIT_HASH`를 레코드에 기록된 실제 revision으로 바꾼다. tokenizer만 내려받으며 모델은 로드하지 않는다. 캐시만 사용하려면 `--local-files-only`. 여러 tokenizer 버전이 섞였다면 파일을 나누어 검증한다. 디코딩은 명세대로 `skip_special_tokens=False`, 나머지는 해당 tokenizer의 기본값을 사용한다.

## 검사 범위

- 필수 키, 자료형, 음수/불리언 토큰 ID, 토큰 배열 길이
- 모델과 tokenizer의 `@revision` 표기 (revision이 실제 commit인지 원격 조회하지 않음)
- `(run_id, split, question_id)` 중복, 같은 run의 모델·tokenizer·양자화·프롬프트 버전·thinking·generation_config 일관성
- references 제공 시 question_id 존재와 split 일치. 전체 문항의 생성 완료 여부는 검사하지 않음
- `<think>`/`</think>`의 원본 ID 위치, reasoning 토큰 수, 텍스트 태그와 최종 답변 원문
- EOS/길이 제한은 관련 설정 제공 시 확인. `length`는 max_new_tokens 도달을 뜻하는 현재 초안을 따름
- 재귀적인 금지 정답 필드 탐지, timing의 유한한 비음수 수치, hidden_state_ref의 타입
- tokenizer 제공 시 prompt/generated 텍스트가 각각 원본 ID의 decode와 일치하는지 확인

`--run-config run_config.json`은 레코드에 generation_config가 없을 때만 대체한다. 파일은 설정 객체 자체 또는 `{"generation_config": {...}}` 형식이다. 레코드 설정이 있으면 우선한다. 레코드별 다른 seed는 최상위 `seed`에 두고, run 공통 generation_config는 유지한다. 외부 설정 파일이 다른 필수 메타데이터를 대신하지는 않는다.

## 판정과 종료 코드

- 0: 검사 범위에서 오류 없음. 경고 또는 생략 검사는 있을 수 있음
- 1: 검증 오류. `--strict`를 주면 경고도 실패 처리
- 2: 파일 읽기/설정/tokenizer 준비 실패 또는 잘못된 CLI 사용

JSON 보고서에는 status, records, errors, warnings, checks, issues가 있다. issues는 물리적 JSONL 줄 번호와 오류 코드를 포함한다. 빈 줄, 잘못된 JSON, 중복 JSON 키도 오류이며 이후 줄을 계속 검사한다. 파일 전체가 비어 있으면 오류다. 입력 파일 경로를 report로 지정하면 덮어쓰지 않고 거부한다.

길이 제한으로 잘린 CoT는 메타데이터가 맞으면 경고로 남긴다. `--strict`를 사용하거나 보고서에서 `truncated`, `unclosed_think`를 확인해 완결 CoT 평가에서 분리한다. 잘린 출력을 정상 완결 데이터로 간주하면 안 된다.

## 경계 규칙

생성 안에 `<think>`가 있으면 그 다음 토큰부터 `</think>` 직전까지 센다. 프롬프트에만 `<think>`가 있으면 `think_tag_in_prompt=true`를 설정하며 생성 index 0부터 센다. 닫는 태그가 없으면 reasoning_token_count, think_close_token_index, final_answer_text 모두 null이다. 마지막 답변은 공백과 EOS 문자열까지 그대로 보존한다.

현재 초안은 thinking handoff 중심이다. non-thinking Direct baseline은 답변 필드 정의를 별도 합의해야 한다. 이 검증기는 태그 없는 출력에 임의로 답변을 추정하지 않는다.

## 한계와 검증 결과

금지 키 탐지만으로 정답 유출을 보장할 수 없다. 특히 prompt_text에 문자열로 삽입된 정답을 완전히 검출하지 않는다. yes/no 같은 정답이 프롬프트에 나타났다는 이유만으로 유출 판정도 하지 않는다. 정답 없는 prompt builder 사용 및 실제 입력 검토가 별도로 필요하다. hidden-state 파일 존재·텐서 shape·위치와 모델 추론의 정확성도 별도 검증 대상이다.

토크나이저를 지정하지 않으면 decode 검사는 명시적으로 skipped이며, pass는 토큰 정렬 검증 완료를 의미하지 않는다. 테스트의 decoder는 가짜이며 실제 Qwen3 출력에 대한 검증은 A가 원본 출력을 전달한 후 실행해야 한다. 기존 `fake_qwen3_outputs.jsonl`은 checkpoint 테스트용 축약 자료이므로 이 엄격한 명세 검증을 통과하도록 만들지 않았다.

```bash
python -m pytest tests/test_generation_output_validation.py -q
python -m pytest tests/ -q
```

이번 환경의 실행 결과: 전체 181 passed, 7 skipped. 신규 검증기 테스트 41개 통과. 건너뛴 기존 7개는 transformers를 사용하는 tokenizer 정렬 테스트이며 실제 Qwen3 검증 결과가 아니다.
