# AGENTS.md — ThinkEnough

## 1. 프로젝트 개요

**프로젝트명:** ThinkEnough  
**연구 주제:** Hidden-State Probing for Dynamic Early Exit in Multimodal Long Chain-of-Thought Reasoning

ThinkEnough는 Multimodal Reasoning Model이 Long Chain-of-Thought(CoT)를 생성하는 과정에서, 현재까지의 추론만으로 정답을 도출할 수 있는지를 Hidden State를 통해 예측하고 불필요한 추론을 조기에 종료하는 연구 프로젝트이다.

핵심 연구 질문은 다음과 같다.

1. 모델의 중간 Hidden State에는 현재 추론의 충분성을 나타내는 정보가 존재하는가?
2. Lightweight Probe를 이용해 STOP/CONTINUE를 예측할 수 있는가?
3. Full CoT의 성능을 최대한 유지하면서 추론 토큰과 지연 시간을 줄일 수 있는가?

**기본 연구 원칙**
- Base Model은 Frozen 상태로 유지한다.
- Base Model의 추가 SFT나 RL은 기본 연구 범위에 포함하지 않는다.
- Hidden-State Probe만 별도로 학습한다.
- Long CoT의 길이와 실제 Early Exit 가능성을 먼저 검증한다.
- 실제로 확인되지 않은 성능 향상이나 구현 성공을 가정하지 않는다.

---

## 2. 현재 담당 범위: Model & Inference

현재 작업은 ThinkEnough 팀프로젝트의 **모델 및 추론 파트**에 해당한다.

### 주요 업무

1. Base Model 후보 검토 및 선정
2. 모델 실행 환경 구축
3. Multimodal 입력을 이용한 Full CoT 생성
4. Reasoning 중간 Hidden State 추출
5. Checkpoint별 Forced-Answer 추론 구현
6. GPU 메모리 사용량과 추론 속도 측정
7. 다른 팀원이 활용할 수 있는 추론 결과와 Hidden State 저장

### 담당하지 않는 업무

다음 작업은 다른 팀의 주된 담당 범위이다.

**Dataset & Evaluation**
- 데이터셋 전처리 및 분할
- Reasoning checkpoint 구성 규칙
- STOP/CONTINUE label 판정
- Task performance 및 efficiency metric 계산

**Probe & Early Exit**
- Linear Probe 및 MLP 학습
- STOP threshold 최적화
- Dynamic Early Exit 정책 구현
- 최종 성능 분석

단, 모델 추론이 필요한 기능은 다른 팀과 협의하여 공동으로 구현한다.

---

## 3. Base Model

현재 ThinkEnough의 1차 Base Model은
Qwen3-VL-8B-Thinking으로 선정한다.

- Hugging Face: Qwen/Qwen3-VL-8B-Thinking
- Architecture: Multimodal Vision-Language Model
- Parameters: 8B
- Reasoning: Thinking CoT
- Training: Frozen

### 모델 팀 주요 목표

1. Qwen3-VL-8B-Thinking 추론 환경 구축
2. 이미지 기반 Long CoT 생성 및 저장
3. Reasoning 구간 및 최종 답변 분리
4. Reasoning checkpoint별 hidden state 추출
5. Checkpoint별 forced-answer generation 구현
6. VRAM, reasoning tokens, latency 측정

### 구현 규칙

- Hugging Face Transformers 기반 구현을 우선한다.
- 모델 가중치는 수정하지 않는다.
- Raw token IDs와 생성 결과를 함께 저장한다.
- Reasoning과 Final Answer를 구분한다.
- Future token leakage가 발생하지 않도록 한다.
- 전체 hidden states를 GPU에 계속 저장하지 않는다.
- Base Model의 dtype과 checkpoint revision을 기록한다.

---

## 4. 개발 환경

### Local

- OS: Windows + WSL2 Ubuntu
- GPU: NVIDIA RTX 4060 Laptop, 8GB VRAM
- 주요 용도: 코드 개발, 데이터 처리, 경량 테스트, 디버깅

### GPU Server

- GPU 서버는 필요할 때마다 신청하여 사용한다.
- 대여한 서버의 실제 GPU 모델과 VRAM을 실행 전에 확인한다.
- 주요 용도: Base Model 추론, Long CoT 생성, Hidden State 추출, Forced-Answer 추론

### 환경 관리 원칙

- Python 및 PyTorch 버전은 프로젝트 설정 파일에 명시한다.
- 패키지 버전은 재현 가능하도록 관리한다.
- CUDA 관련 패키지를 변경하기 전에 기존 환경을 확인한다.
- GPU 메모리가 부족하면 batch size, 입력 크기, 정밀도 등을 검토한다.
- 양자화는 Hidden State에 영향을 줄 수 있으므로 실험 설정에 반드시 기록한다.
- Local과 Server에서 모델 정밀도가 다른 경우 결과를 동일한 조건으로 간주하지 않는다.

GPU 서버 접속, 대규모 다운로드, 장시간 추론 실행 등은 명시적인 요청 없이 임의로 수행하지 않는다.

---

## 5. 모델 추론 구현 원칙

### 5.1 Full CoT Generation

Base Model에서 reasoning 과정과 최종 답변을 생성하는 기능을 구현한다.

가능하면 다음 정보를 저장한다.

- Sample ID
- Model name 및 checkpoint revision
- 입력 데이터 식별자
- Prompt 및 generation configuration
- Raw model output
- Reasoning text
- Final answer
- Reasoning token count
- Generation latency

모델이 제공하는 reasoning 관련 special token이나 출력 형식이 있다면 이를 우선 활용한다.

모델의 출력 형식을 임의로 가정하지 않는다.

### 5.2 Hidden State Extraction

Reasoning 중간의 checkpoint에서 Hidden State를 추출할 수 있도록 구현한다.

기본적으로 다음 표현을 우선 검토한다.

- 마지막 Transformer layer
- 해당 checkpoint 마지막 token의 Hidden State

수식으로 표현하면 다음과 같다.

`h_t = Last-layer hidden state at checkpoint t`

이때 Hidden State는 마지막 문장만이 아니라 현재까지 누적된 입력 및 reasoning prefix를 반영해야 한다.

주의 사항:

- Tokenizer 기준 checkpoint 위치를 정확히 매핑한다.
- Hidden State 추출 시 불필요한 gradient를 계산하지 않는다.
- 최종 답변이나 미래 reasoning token의 정보가 checkpoint에 포함되지 않도록 한다.
- 대규모 Hidden State를 계속 GPU 메모리에 보관하지 않는다.
- 추출된 Tensor는 필요한 경우 CPU로 이동하여 저장한다.
- Hidden State의 shape, dtype, layer 정보를 기록한다.

### 5.3 Forced-Answer Generation

각 checkpoint에서 현재 reasoning prefix만을 이용해 즉시 최종 답변을 생성하는 기능을 구현한다.

입력:

- 원래의 Multimodal Input
- Question
- 해당 checkpoint까지의 Reasoning Prefix

출력:

- 생성된 Final Answer
- Token Count
- Generation Latency

미래 reasoning이나 원래 Full CoT의 최종 답변이 입력에 포함되지 않도록 주의한다.

Forced-Answer는 Probe 학습을 위한 **offline label construction**에 사용된다.

실제 Early Exit inference에서 매 checkpoint마다 Forced-Answer를 생성하는 방식과 혼동하지 않는다.

---

## 6. 팀 간 데이터 인터페이스

다른 팀원이 모델 내부 구현을 알지 못해도 데이터를 사용할 수 있도록 출력 형식을 명확히 정의한다.

가능하면 다음과 같은 metadata 구조를 사용한다.

```json
{
  "sample_id": "sample_0001",
  "model_name": "base_model",
  "checkpoint_idx": 2,
  "reasoning_prefix": "Reasoning up to checkpoint 2",
  "answer": "Generated answer",
  "reasoning_tokens": 128,
  "hidden_state_path": "path/to/hidden_state.pt"
}
```

이는 예시 형식이며 실제 스키마는 팀과 협의하여 확정한다.

- Hidden State Tensor는 별도 파일로 저장한다.
- Metadata는 JSON 또는 JSONL 등으로 관리한다.
- 경로는 특정 개발자의 컴퓨터에 종속되지 않도록 한다.
- Sample ID와 checkpoint index를 일관되게 관리한다.
- STOP/CONTINUE label은 Dataset 팀의 정답 판정 결과와 연결한다.
- Probe 팀이 사용할 입력 차원과 Tensor 구조를 문서화한다.

---

## 7. 재현성과 실험 관리

모든 실험은 가능한 한 재현 가능하도록 구성한다.

다음 정보를 기록한다.

- Base Model 및 checkpoint
- Dataset 및 split
- Random Seed
- Generation Parameters
- Precision 및 Quantization
- Checkpoint Definition
- GPU 정보
- 실행 명령어
- 실행 시간 및 메모리 사용량

실험 설정은 코드 내부의 상수보다 config 파일로 관리하는 것을 우선한다.

실험 결과와 코드를 분리하여 관리한다.

동일한 입력과 설정에서 결과를 재현할 수 있는지 검증한다. 확률적 decoding을 사용하는 경우 완전한 일치가 보장되지 않을 수 있음을 명시한다.

---

## 8. GitHub 협업 규칙

### 브랜치 관리

- `main` 브랜치는 안정적인 코드 유지용으로 사용한다.
- 새로운 기능은 별도 feature branch에서 개발한다.
- 기능 구현이 끝나면 Pull Request를 생성한다.
- 다른 팀원의 코드를 수정할 경우 변경 이유와 영향을 명확히 설명한다.

브랜치 이름 예시:

- `feat/model-inference`
- `feat/cot-generation`
- `feat/hidden-state`
- `feat/forced-answer`
- `fix/model-loading`

### 커밋 규칙

가능하면 다음 prefix를 사용한다.

- `feat:` 새로운 기능
- `fix:` 버그 수정
- `refactor:` 코드 구조 개선
- `docs:` 문서 수정
- `test:` 테스트 추가
- `chore:` 환경 및 설정 변경

### Git에 업로드하지 않을 파일

- 모델 가중치
- 대용량 데이터셋
- Hidden State Tensor
- GPU 실험 결과의 대용량 파일
- `.env` 및 API Key
- SSH Private Key
- 개인 환경에 종속된 설정

필요한 경우 `.gitignore`를 업데이트한다.

---

## 9. AI Coding Agent 작업 규칙

이 저장소에서 작업하는 AI Agent는 다음 원칙을 따른다.

1. **작업 전에 기존 코드와 프로젝트 구조를 먼저 확인한다.**
2. 기존 코드가 있다면 최대한 재사용한다.
3. 요청하지 않은 대규모 리팩터링을 수행하지 않는다.
4. 모델별 특수 구현과 공통 기능을 가능한 한 분리한다.
5. 실험 설정과 하이퍼파라미터를 임의로 변경하지 않는다.
6. 불명확한 모델 구조나 논문 구현은 추측으로 단정하지 않는다.
7. 기능 구현 후 가능한 범위에서 테스트를 수행한다.
8. GPU가 없어 테스트하지 못한 기능은 미검증이라고 명확히 표시한다.
9. 기존 결과 파일을 임의로 삭제하거나 덮어쓰지 않는다.
10. 코드 수정 후 변경된 파일, 수정 이유, 테스트 결과를 한국어로 정리한다.

### 코드 스타일

- Python을 주 언어로 사용한다.
- 함수와 클래스에는 이해하기 쉬운 이름을 사용한다.
- 복잡한 로직에는 Docstring과 주석을 작성한다.
- Type Hint 사용을 권장한다.
- 불필요한 추상화나 과도한 파일 분할은 피한다.
- 하드코딩된 절대 경로를 사용하지 않는다.

### 설명 원칙

사용자가 코드 이해를 중요하게 생각하므로 단순히 코드를 생성하는 것에 그치지 않는다.

주요 구현에서는 다음 내용을 함께 설명한다.

- 해당 코드가 필요한 이유
- 입력과 출력
- 사용된 핵심 알고리즘
- 모델 내부에서 어떤 정보가 처리되는지
- 실행 및 검증 방법

설명과 작업 결과는 기본적으로 한국어로 작성한다.

---

## 10. 현재 개발 우선순위

### Phase 0 — Feasibility Study

1. Base Model 후보 선정
2. Pretrained Model 로딩 및 추론 테스트
3. Full CoT 생성
4. CoT 길이 및 checkpoint 수 분석
5. Forced-Answer 기능 검증
6. Hidden State 추출 가능 여부 확인

### Phase 1 — Data Generation Pipeline

1. Batch CoT generation 구현
2. Checkpoint별 Hidden State extraction
3. Forced-Answer inference 구현
4. Metadata 및 Tensor 저장
5. Dataset 팀과 결과 연동

### Phase 2 — Integration

1. Probe 팀과 Hidden State 인터페이스 연결
2. STOP/CONTINUE prediction 연동
3. Dynamic Early Exit inference 지원
4. Full CoT 대비 추론 비용 분석

**가장 중요한 초기 목표는 모델을 단순히 실행하는 것이 아니라, 실제 Long CoT와 의미 있는 중간 checkpoint가 존재하는지 검증하는 것이다.**

---

## 11. 작업 완료 보고 형식

AI Agent는 작업을 마친 뒤 다음 사항을 간단히 보고한다.

1. 수행한 작업
2. 생성하거나 수정한 파일
3. 핵심 구현 내용
4. 실제 수행한 테스트와 결과
5. 확인된 문제 및 제한 사항
6. 다음에 진행할 작업

실제로 실행하지 않은 테스트를 성공했다고 보고하지 않는다.

연구 결과와 예상 결과를 명확히 구분한다.