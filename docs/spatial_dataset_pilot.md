# 3D 공간 추론 데이터셋 조사 및 pilot 제안

조사일: 2026-11-01. 상태: 공개 문서·데이터 뷰어·일부 평가 코드 조사 완료.
데이터 다운로드, 실제 표본 추출, GPU 추론, CoT 길이 측정은 수행하지 않았다.
아래 pilot 수량과 평가 절차는 제안이며 확정된 실험 설정이 아니다.

## 결론

3D 공간 추론을 필수 조건으로 유지한다. 현재 모델은 이미지 입력으로
3D 관계를 추론하는 Qwen3-VL-8B-Thinking이며, 포인트클라우드를 직접
처리하는 모델로 확인된 것은 아니다.

- 주 후보: SPAR-Bench의 spatial imagination / object relation inference.
- 비교군: 3DSRBench의 orientation / multi-object reasoning.
- 추가 후보: SAT-v2의 perspective-taking / dynamic reasoning.
- 조건부 후보: MultihopSpatial의 관점 전환·깊이 관계 문항.

이는 과제 구조에 근거한 우선순위이다. 어떤 후보도 현재 모델에서
Long CoT와 유용한 early exit 신호를 제공한다고 검증되지 않았다.

## 후보 비교와 확인된 제약

### SPAR-Bench / SPAR-Bench-Tiny

공식 저장소는 spatial imagination과 object relation inference를 고수준
공간 추론 과제로 구분한다. Tiny는 20개 과제별 50문항, 총 1,000문항이고
공개 뷰어에는 test split이 있다. 입력은 이미지 목록과 질문이며
`id`, `img_type`, `format_type`, `task`, `source`, `image`, `question`,
`answer` 필드가 보인다. 현재 우리 단일 이미지 스키마로 다중 시점을
표현할 수 없으므로, 초기 실행은 단일 이미지 문항에 한정하거나 별도
승인 후 다중 이미지 지원을 추가해야 한다.

뷰어의 깊이 문제는 기준 객체의 깊이를 주고 다른 객체의 깊이를 묻는다.
이런 수치 지각 과제와 공간적 상상 과제는 구분해서 집계한다.
정답 형식이 혼합되어 있으므로 수치 허용오차를 임의로 정하지 않고
공식 과제별 채점 구현을 추가 확인한다. Tiny를 probe 학습용으로 쓴 뒤
동일 문항을 독립적인 공식 평가 결과로 제시하지 않는다.

출처: [공식 저장소](https://github.com/LogosIntelligenceLab/SPAR),
[Tiny 데이터 카드·뷰어](https://huggingface.co/datasets/jasonzhango/SPAR-Bench-Tiny).

### 3DSRBench

height, location, orientation, multi-object의 3D 능력을 평가한다.
공개 HF 뷰어에는 test split, `index`, `question`, `A`~`D`, `answer`,
`category`, `image_source`, `image_url`이 있다. 선택지 일부는 null이다.
원본과 `-flip` 변형이 같은 이미지 URL을 사용하므로 URL만 읽고
이미지 변환을 무시하면 안 된다. FlipEval 변환 및 CircularEval의
선택지 순환 규칙은 실제 adapter 구현 전에 확인해야 한다.

단일 이미지 입력에 연결하기 쉽지만, 긴 추론을 요구하는지는 미확인이다.
pilot의 단일 순서 MCQ 정확도를 공식 CircularEval/FlipEval 점수라고
표기하지 않는다. 프로젝트 페이지와 논문·데이터 뷰어의 문항 수 표기가
서로 달라 다운로드 revision과 실제 고유 문항 수를 기준으로 기록한다.

출처: [공식 프로젝트](https://3dsrbench.github.io/),
[공개 데이터](https://huggingface.co/datasets/ccvl/3DSRBench).

### SAT-v2

관점 전환, 객체 관계, 동적 공간 추론을 포함한다. 데이터 카드의 분할은
train 172,384 / static 127,405 / val 4,001 / test 150이다.
입력 필드는 `images`, `question`, 선택지는 `answers`, 정답은
`correct_answer`이다. `answers`는 선택지이므로 정답 필드와 구분한다.
전체 데이터에는 단순 개수·2D 위치 문제도 있어 필터링이 필요하다.
공식 카드는 test에 선택지 위치를 바꾸는 circular evaluation을 요구한다.
장면 식별자·분할 간 장면 중복과 재사용 조건은 추가 확인 대상이다.

출처: [데이터 카드](https://huggingface.co/datasets/array/SAT-v2),
[공식 저장소](https://github.com/arijitray1993/SAT).

### MultihopSpatial

1~3 hop, ego/exo 관점을 구분하며 train 6,791 / test 4,500을 제공한다.
이미지·선택형 질문과 함께 정답 객체 bbox를 채점한다.
관점 전환 문제를 포함하지만 hop에 속성 탐색도 포함되므로
3-hop을 곧바로 3D 추론 세 단계라고 해석하지 않는다.
원문 문항을 검토하여 화면상의 2D 관계만으로 풀리는 문제를 분리한다.

원본 정답 bbox는 픽셀 단위 xywh이고 예측 좌표 변환이 필요하다.
MCQ와 grounding을 함께 쓰는 Acc@50IoU를 공식 지표와 구분 없이
MCQ 정확도로 대체하지 않는다. 코드와 문항의 모델 계열 호환성은
유망하지만 우리 Thinking 모델의 결과는 아직 없다.

출처: [데이터 카드](https://huggingface.co/datasets/etri-vilab/MultihopSpatial),
[공식 설명](https://github.com/youngwanLEE/multihopspatial),
[평가 코드](https://raw.githubusercontent.com/youngwanLEE/multihopspatial/main/eval/benchmark_qwen.py).

## 제안하는 1차 pilot

1. SPAR 고수준 과제 30문항, 3DSRBench orientation/multi-object
   30문항을 목표로 한다. 해당 조건의 실제 수량과 단일/다중 이미지
   구성은 표본을 확보한 뒤 확인한다. 부족하면 조용히 다른 과제로
   채우지 않고 수량을 기록한다.
2. 한 문항씩 원본 이미지와 질문을 확인해 3D 관점·방향·깊이·관계
   추론 여부, 모호함, 입력 누락을 점검한다. 정답으로 난도를 선별하지 않는다.
3. 원본 ID, 데이터 revision, 이미지/장면 그룹, 과제, 선택 규칙과 seed를
   manifest에 남긴다. 같은 장면과 flip·선택지 순환 변형은 같은 그룹이다.
   두 데이터셋이 같은 원천 장면을 쓸 가능성도 있으므로 교차 중복을 확인한다.
4. 기존 생성 설정을 임의로 바꾸지 않는다. 현재 기본 max_new_tokens는
   40,960이므로 GPU 시간·VRAM 예산을 검토한 별도 pilot 설정이 필요하다.
   긴 설명을 강제하지 않고 최종 답변 형식만 정한다.
5. reasoning 토큰 p25/중앙값/p75, 완결·길이 제한·파싱 실패 비율,
   정답률, checkpoint 수, latency를 과제별로 보고한다. 길이가 잘린
   출력도 분모에 포함하고 완결 출력과 분리한다.
6. 완결 문항에서 시작·중간·후반 checkpoint를 선택해 forced-answer를
   비교한다. 현재 저장소에는 실제 forced-answer와 hidden state 추출이
   구현되지 않았으므로 먼저 별도 구현·검증 계획이 필요하다.
7. 길이만으로 선정하지 않는다. 정답 가능한 문제, 이른 정답 가능 시점,
   이후 오답으로 바뀌는 비율을 함께 확인한다. 첫 정답 checkpoint는
   offline oracle 분석이며 온라인 STOP 정책의 성능이 아니다.

표본이 작으므로 본실험 성능 주장을 하지 않는다. pilot 문항은 개발용으로
표시하고 최종 평가와 분리한다. probe 학습·threshold 조정·최종 평가는
문항/장면 그룹 단위로 분리하며, 공식 test를 재분할했다면 자체 분할임을 명시한다.

## 데이터팀 코드와의 연결

SQA3D 전용 prompt builder와 채점기는 교체 대상이다. 원본 생성 출력
검증, checkpoint 생성·선택은 공통 부분으로 재사용을 검토한다.
질문 ID를 숫자로 강제하는 가정은 3DSRBench 문자열 ID와 맞지 않는다.
원본 ID와 데이터셋을 보존하는 매핑을 협의해야 하며, 아직 스키마를 바꾸지 않았다.
정답과 정답 bbox는 평가용으로 별도 보관하고 프롬프트/모델 전달 메타데이터에 넣지 않는다.

범위가 정해진 다음 adapter·출력 계약·테스트 계획을 제시해 확인받는다.
이번 조사는 모델 교체, 새 의존성, 다운로드, GPU 실행을 포함하지 않는다.
