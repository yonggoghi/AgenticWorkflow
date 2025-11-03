# 환경 변수 설정 가이드

## 필요한 환경 변수

product_crawler를 사용하려면 LLM API 키와 엔드포인트 URL이 필요합니다.

### 방법 1: 환경 변수 직접 설정

```bash
# 기본 설정 (필수)
export LLM_API_KEY="your-api-key-here"
export LLM_API_URL="https://api.openai.com/v1"
```

### 방법 2: .env 파일 사용

info_builder 디렉토리에 `.env` 파일을 생성:

```bash
# .env 파일 내용
LLM_API_KEY=your-api-key-here
LLM_API_URL=https://api.openai.com/v1
```

## config.py 설정

`config.py` 파일에서 다음을 설정할 수 있습니다:

```python
@dataclass
class ModelConfig:
    """모델 관련 설정"""
    # 모델 이름 (실제 API 엔드포인트에서 사용하는 형식)
    gemma_model: str = "skt/gemma3-12b-it"
    ax_model: str = "skt/ax4"
    claude_model: str = "amazon/anthropic/claude-sonnet-4-20250514"
    gemini_model: str = "gcp/gemini-2.5-flash"
    gpt_model: str = "azure/openai/gpt-4o-2024-08-06"
    
    # 기본 모델
    default_model: str = "skt/ax4"
    
    # 토큰 제한
    llm_max_tokens: int = 4000
    
    # 온도
    temperature: float = 0.0
    
    # Seed
    seed: int = 42
```

## 사용 가능한 모델

product_crawler는 다음 모델을 지원합니다:

| 모델명 | 실제 모델 ID | 사용 예시 |
|--------|--------------|-----------|
| gemma | skt/gemma3-12b-it | `--model gemma` |
| ax | skt/ax4 | `--model ax` (기본값) |
| claude | amazon/anthropic/claude-sonnet-4-20250514 | `--model claude` |
| gemini | gcp/gemini-2.5-flash | `--model gemini` |
| gpt | azure/openai/gpt-4o-2024-08-06 | `--model gpt` |

## API 엔드포인트 설정

### OpenAI API

```bash
export LLM_API_KEY="sk-..."
export LLM_API_URL="https://api.openai.com/v1"
```

### 커스텀 엔드포인트

```bash
export LLM_API_KEY="your-key"
export LLM_API_URL="https://your-custom-endpoint.com/v1"
```

### Anthropic API (직접 사용하는 경우)

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

## 확인

설정이 제대로 되었는지 확인:

```bash
echo $LLM_API_KEY
echo $LLM_API_URL
```

또는 Python에서:

```python
from config import settings

print(f"API Key: {settings.API_CONFIG.llm_api_key[:10]}...")
print(f"API URL: {settings.API_CONFIG.llm_api_url}")
print(f"Models: {settings.ModelConfig.claude_model}")
```

## 문제 해결

### "LLM API key not found"

```bash
# 환경 변수가 설정되어 있는지 확인
env | grep LLM

# 설정
export LLM_API_KEY="your-key"
```

### "config.py not found"

```bash
# info_builder 디렉토리에 있는지 확인
cd info_builder
ls config.py

# 없으면 생성 (기본 config.py 사용)
```

### langchain 오류

```bash
# langchain 설치
pip install langchain-openai langchain
```

