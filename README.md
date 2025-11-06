# FinDER Benchmark with LLM Router

이 프로젝트는 [FinDER 데이터셋](https://huggingface.co/datasets/Linq-AI-Research/FinDER)을 사용하여 여러 LLM 모델의 성능을 벤치마크하고 결과를 JSON 형식으로 저장합니다.

## 기능

- FinDER 데이터셋 자동 로드
- 여러 LLM 제공자 지원 (OpenAI, Anthropic, Google, Together AI, XAI)
- Question과 Reference를 입력으로 사용하여 LLM 응답 생성
- 결과를 구조화된 JSON 형식으로 저장
- 비용 추적 및 토큰 사용량 기록
- 배치 처리 및 중간 결과 자동 저장

## 설치

1. 가상환경 생성 및 활성화:
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate  # Windows
```

2. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

3. 환경 변수 설정:
```bash
cp .env.example .env
# .env 파일을 열어서 실제 API 키를 입력하세요
```

## 사용법

### 기본 실행

```bash
python finder_benchmark.py
```

### 설정 커스터마이징

`finder_benchmark.py` 파일의 `main()` 함수에서 다음 설정을 변경할 수 있습니다:

```python
# Configuration
DATASET_SPLIT = "test"  # or "train", "validation"
NUM_SAMPLES = 10  # Set to None to process all samples
OUTPUT_DIR = "results"
```

### 모델 선택

`finder_benchmark.py`의 `models` 리스트에서 테스트할 모델을 활성화/비활성화할 수 있습니다:

```python
models = [
    # OpenAI
    OpenAIClient(model_id="gpt-4.1", temperature=0.6),
    
    # Google
    GeminiClient(model_id="gemini-2.5-flash", temperature=0.6),
    
    # 기타 모델들...
]
```

## 출력 형식

결과는 `results/` 디렉토리에 JSON 파일로 저장됩니다:

```json
{
  "metadata": {
    "model": "gemini-2.5-flash",
    "total_samples": 10,
    "total_cost": 0.0234,
    "average_cost": 0.00234
  },
  "results": [
    {
      "sample_id": 0,
      "Question": "What is the revenue growth rate?",
      "Reference": "Document context here...",
      "Response": "The revenue growth rate is...",
      "model": "gemini-2.5-flash",
      "input_tokens": 150,
      "output_tokens": 50,
      "cost": 0.00234,
      "ttft": 0.5
    }
  ]
}
```

## 시스템 프롬프트

기본 시스템 프롬프트는 다음과 같습니다:

```python
"""Answer the following question based on the provided documents:
Question: {query}
Documents:
{context}
Answer:
"""
```

프롬프트를 수정하려면 `finder_benchmark.py`의 `SYSTEM_PROMPT_TEMPLATE` 변수를 편집하세요.

## FinDER 데이터셋 필드

데이터셋의 필드명이 다를 경우, `run_benchmark()` 함수 호출 시 파라미터를 조정하세요:

```python
run_benchmark(
    model_client=model_client,
    dataset=dataset,
    output_file=output_file,
    question_field="question",  # 실제 필드명으로 변경
    reference_field="context"   # 실제 필드명으로 변경
)
```

## 지원 모델

### OpenAI
- gpt-4.1
- gpt-5

### Anthropic
- claude-sonnet-4-20250514

### Google
- gemini-2.5-flash
- gemini-2.5-pro

### Together AI
- deepseek-ai/DeepSeek-V3
- Qwen/Qwen3-235B-A22B-Instruct-2507-tput
- meta-llama/Llama-4-Scout-17B-16E-Instruct
- mistralai/Mistral-Small-24B-Instruct-2501

### XAI
- grok-beta
- grok-4-fast-non-reasoning

## 비용 추적

각 API 호출의 비용이 자동으로 계산되고 저장됩니다. 가격은 `llm_clients.py`의 `PRICING` 딕셔너리에서 확인할 수 있습니다.

## 문제 해결

### API 키 오류
`.env` 파일에 올바른 API 키가 설정되어 있는지 확인하세요.

### 데이터셋 로드 실패
인터넷 연결을 확인하고, Hugging Face 데이터셋에 접근할 수 있는지 확인하세요.

### 필드명 오류
FinDER 데이터셋의 실제 필드명을 확인하고 `question_field`와 `reference_field` 파라미터를 조정하세요.

## 라이센스

이 프로젝트는 MIT 라이센스를 따릅니다.
