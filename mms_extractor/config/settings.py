"""
Configuration settings for MMS Extractor.
"""
import os
from dataclasses import dataclass
from typing import List, Dict, Any
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
    llm_api_key: str = os.getenv("CUSTOM_OPENAI_API_KEY", "")
    llm_api_url: str = os.getenv("CUSTOM_OPENAI_BASE_URL", "https://api.platform.a15t.com/v1")
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
    claude_model: str = "skt/claude-3-7-sonnet-20250219"
    gemma_model: str = "skt/gemma3-12b-it"
    gpt_model: str = "gpt-4.1"
    claude_sonnet_model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4000
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
class DataConfig:
    """Data configuration settings."""
    # Get the package root directory
    _package_root = Path(__file__).parent.parent.absolute()
    
    item_info_path: str = str(_package_root / "data" / "item_info_all_250527.csv")
    alias_rules_path: str = str(_package_root / "data" / "alias_rules.csv")
    stop_words_path: str = str(_package_root / "data" / "stop_words.csv")
    mms_data_path: str = str(_package_root / "data" / "mms_data_250408.csv")
    pgm_tag_path: str = str(_package_root / "data" / "pgm_tag_ext_250516.csv")
    org_info_path: str = str(_package_root / "data" / "org_info_all_250605.csv")
    
    # Embedding files
    item_embeddings_path: str = str(_package_root / "data" / "item_embeddings_250527.npz")
    org_all_embeddings_path: str = str(_package_root / "data" / "org_all_embeddings_250605.npz")
    org_nm_embeddings_path: str = str(_package_root / "data" / "org_nm_embeddings_250605.npz")


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


@dataclass
class ExtractionSchema:
    """Schema for information extraction."""
    
    def get_product_schema(self, product_elements: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get product schema, optionally with pre-filled product elements for NLP mode.
        
        Args:
            product_elements: Pre-filled product elements for NLP mode
            
        Returns:
            Product schema dictionary
        """
        base_schema = {
            "title": {
                "type": "string", 
                'description': '광고 제목. 광고의 핵심 주제와 가치 제안을 명확하게 설명할 수 있도록 생성'
            },
            'purpose': {
                'type': 'array', 
                'description': '광고의 주요 목적을 다음 중에서 선택(복수 가능): [상품 가입 유도, 대리점/매장 방문 유도, 웹/앱 접속 유도, 이벤트 응모 유도, 혜택 안내, 쿠폰 제공 안내, 경품 제공 안내, 수신 거부 안내, 기타 정보 제공]'
            },
            'product': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {
                        'name': {'type': 'string', 'description': '광고하는 제품이나 서비스 이름'},
                        'position': {'type': 'string', 'description': '광고 상품의 분류. [main, sub] 중에서 선택'},
                        'action': {'type': 'string', 'description': '고객에게 기대하는 행동: [구매, 가입, 사용, 방문, 참여, 코드입력, 쿠폰다운로드, 기타] 중에서 선택'}
                    }
                }
            },
            'channel': {
                'type': 'array', 
                'items': {
                    'type': 'object', 
                    'properties': {
                        'type': {'type': 'string', 'description': '채널 종류: [URL, 전화번호, 앱, 대리점] 중에서 선택'},
                        'value': {'type': 'string', 'description': '실제 URL, 전화번호, 앱 이름, 대리점 이름 등 구체적 정보'},
                        'action': {'type': 'string', 'description': '채널 목적: [가입, 추가 정보, 문의, 수신, 수신 거부] 중에서 선택'},
                    }
                }
            },
            'pgm':{
                'type': 'array', 
                'description': '아래 광고 분류 기준 정보에서 선택. 메세지 내용과 광고 분류 기준을 참고하여, 광고 메세지에 가장 부합하는 2개의 pgm_nm을 적합도 순서대로 제공'
            },
        }
        
        # Override product schema with pre-filled elements for NLP mode
        if product_elements:
            base_schema['product'] = product_elements
        
        return base_schema
    
    @property
    def product_schema(self) -> Dict[str, Any]:
        """Legacy property for backward compatibility."""
        return self.get_product_schema()

    @property
    def exclusion_tag_patterns(self) -> List[List[str]]:
        return [
            ['SN', 'NNB'], ['W_SERIAL'], ['JKO'], ['W_URL'], ['W_EMAIL'],
            ['XSV', 'EC'], ['VV', 'EC'], ['VCP', 'ETM'], ['XSA', 'ETM'],
            ['VV', 'ETN'], ['W_SERIAL'], ['W_URL'], ['JKO'], ['SSO'],
            ['SSC'], ['SW'], ['SF'], ['SP'], ['SS'], ['SE'], ['SO'],
            ['SB'], ['SH'], ['W_HASHTAG']
        ]


# Global configuration instances
API_CONFIG = APIConfig()
MODEL_CONFIG = ModelConfig()
DATA_CONFIG = DataConfig()
PROCESSING_CONFIG = ProcessingConfig()
EXTRACTION_SCHEMA = ExtractionSchema()


def get_device():
    """Determine the best available device for computation."""
    import torch
    
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu" 