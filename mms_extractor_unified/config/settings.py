"""
Configuration settings for MMS Extractor.
"""
import os
from dataclasses import dataclass
from typing import List
from pathlib import Path

# Set environment variable to suppress tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    # dotenv not available, skip loading .env file
    pass


@dataclass
class APIConfig:
    """API configuration settings."""
    llm_api_key: str = os.getenv("CUSTOM_API_KEY", "")
    llm_api_url: str = os.getenv("CUSTOM_BASE_URL", "https://api.platform.a15t.com/v1")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")


@dataclass
class ModelConfig:
    """Model configuration settings."""
    embedding_model: str = "jhgan/ko-sbert-nli"
    # Local model path - if provided, will load from local instead of downloading
    local_embedding_model_path: str = os.getenv("LOCAL_EMBEDDING_MODEL_PATH", "./models/ko-sbert-nli")
    # Model loading mode: 'auto', 'local', 'remote'
    model_loading_mode: str = os.getenv("MODEL_LOADING_MODE", "auto")
    gemma_model: str = "skt/gemma3-12b-it"
    gpt_model: str = "gpt-4o"
    claude_model: str = "claude-sonnet-4-20250514"
    llm_model: str = os.getenv("LLM_MODEL", "gemma")
    llm_max_tokens: int = 4000
    temperature: float = 0.0
    
    def __post_init__(self):
        """Validate model loading mode."""
        valid_modes = ['auto', 'local', 'remote']
        if self.model_loading_mode not in valid_modes:
            raise ValueError(f"model_loading_mode must be one of {valid_modes}, got: {self.model_loading_mode}")
    
    def get_loading_mode_description(self) -> str:
        """Get description of current loading mode."""
        descriptions = {
            'auto': 'Automatically use local model if available, otherwise download from internet',
            'local': 'Only use local models, fail if not found (offline mode)',
            'remote': 'Always download from internet, ignore local models'
        }
        return descriptions.get(self.model_loading_mode, 'Unknown mode')





@dataclass
class ProcessingConfig:
    """Processing configuration settings."""
    similarity_threshold: float = 0.7
    fuzzy_threshold: float = 0.4
    num_candidate_programs: int = 5
    batch_size: int = 100
    n_jobs: int = 6
    user_defined_entities: List[str] = None
    # Product information extraction mode
    product_info_extraction_mode: str = 'nlp'  # options: 'rag', 'llm', 'nlp'
    # Entity extraction mode
    entity_extraction_mode: str = 'logic'  # options: 'llm', 'logic'
    
    def __post_init__(self):
        if self.user_defined_entities is None:
            self.user_defined_entities = [
                'AIA Vitality', 
                '부스트 파크 건대입구', 
                'Boost Park 건대입구'
            ]
        
        # Validate product_info_extraction_mode
        valid_modes = ['rag', 'llm', 'nlp']
        if self.product_info_extraction_mode not in valid_modes:
            raise ValueError(f"product_info_extraction_mode must be one of {valid_modes}")
        
        # Validate entity_extraction_mode
        valid_entity_modes = ['llm', 'logic']
        if self.entity_extraction_mode not in valid_entity_modes:
            raise ValueError(f"entity_extraction_mode must be one of {valid_entity_modes}")

    @property
    def chain_of_thought(self) -> str:
        """Get chain of thought based on extraction mode."""
        if self.product_info_extraction_mode == 'nlp':
            return """1. 광고 목적을 먼저 파악한다.
2. 파악된 목적에 기반하여 Product 정보를 추출한다.
3. 주어진 name 정보에 기반하여, position과 action 필드의 정보를 추출한다.
4. 추출된 상품 정보를 고려하여 채널 정보를 제공한다."""
        else:
            return """1. 광고 목적을 먼저 파악한다.
2. 파악된 목적에 기반하여 Main 상품을 추출한다.
3. 추출한 Main 상품에 관련되는 Sub 상품을 추출한다.
4. 추출된 상품 정보를 고려하여 채널 정보를 제공한다."""

    def get_extraction_guide(self, candidate_items: List[str] = None) -> str:
        """Get extraction guide based on mode and candidate items."""
        base_guide = "* 상품 추출시 정확도(precision) 보다는 재현율(recall)에 중심을 두어라."
        
        if self.product_info_extraction_mode == 'rag' and candidate_items:
            return f"""{base_guide}
* 후보 상품 이름 목록에 포함된 상품 이름은 참고하여 Product 정보를 추출하라."""
        elif self.product_info_extraction_mode == 'nlp':
            return "* Product 정보에서 position, action 필드의 정보를 추출하라."
        else:
            return base_guide

# Global configuration instances
API_CONFIG = APIConfig()
MODEL_CONFIG = ModelConfig()
PROCESSING_CONFIG = ProcessingConfig() 