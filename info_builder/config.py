"""
Info Builder 설정 파일
"""
import os
from dotenv import load_dotenv
from dataclasses import dataclass

# 환경 변수 로드
load_dotenv()


@dataclass
class APIConfig:
    """API 관련 설정"""
    llm_api_key: str = os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    llm_api_url: str = os.getenv("LLM_API_URL", "https://api.openai.com/v1")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")


@dataclass
class ModelConfig:
    """모델 관련 설정"""
    # 모델 이름 (실제 API 엔드포인트에서 사용하는 형식)
    gemma_model: str = "skt/gemma3-12b-it"
    ax_model: str = "skt/ax4"
    claude_model: str = "amazon/anthropic/claude-sonnet-4-20250514"
    gemini_model: str = "gcp/gemini-2.5-flash"
    gpt_model: str = "azure/openai/gpt-4o-2024-08-06"
    
    # 기본 모델 (변경 가능)
    default_model: str = "skt/ax4"
    
    # 토큰 제한
    llm_max_tokens: int = 8000
    
    # 온도
    temperature: float = 0.0
    
    # Seed
    seed: int = 42


class Settings:
    """전역 설정"""
    API_CONFIG = APIConfig()
    ModelConfig = ModelConfig()


# 전역 settings 인스턴스
settings = Settings()

