"""
MMS Extractor Configuration Settings
=====================================

📋 개요
-------
MMS Extractor 시스템의 모든 설정을 관리하는 중앙 설정 모듈입니다.
Dataclass 기반으로 구조화되어 타입 안전성과 IDE 지원을 제공합니다.

🔗 의존성
---------
**사용되는 곳:**
- `core.mms_extractor`: MMSExtractor 초기화 시 설정 로드
- `utils.llm_factory`: LLM 모델 설정
- `services.*`: 각 서비스의 임계값 및 경로 설정
- `apps.*`: API/CLI 애플리케이션 설정

🏗️ 설정 그룹
------------

### 1. API_CONFIG (APIConfig)
**목적**: LLM API 키 및 엔드포인트 관리

**환경변수:**
- `CUSTOM_API_KEY`: 커스텀 LLM API 키
- `CUSTOM_BASE_URL`: 커스텀 LLM API URL
- `OPENAI_API_KEY`: OpenAI API 키
- `ANTHROPIC_API_KEY`: Anthropic API 키

**사용 예시:**
```python
from config.settings import API_CONFIG

# API 키 접근
api_key = API_CONFIG.llm_api_key
api_url = API_CONFIG.llm_api_url
```

---

### 2. MODEL_CONFIG (ModelConfig)
**목적**: AI 모델 설정 및 파라미터 관리

**주요 설정:**
- `embedding_model`: 임베딩 모델 (ko-sbert-nli)
- `llm_model`: 활성 LLM 모델 (ax, gpt, gemini 등)
- `llm_max_tokens`: 최대 토큰 수 (기본 4000)
- `temperature`: 생성 온도 (기본 0.0)
- `model_loading_mode`: 모델 로딩 전략 (auto/local/remote)

**모델 로딩 모드:**
| 모드 | 설명 | 사용 시나리오 |
|------|------|--------------|
| **auto** | 로컬 우선, 없으면 다운로드 | 일반적인 사용 (기본값) |
| **local** | 로컬만 사용, 없으면 실패 | 오프라인 환경 |
| **remote** | 항상 다운로드 | 최신 모델 강제 사용 |

**사용 예시:**
```python
from config.settings import MODEL_CONFIG

# 모델 설정 접근
llm_model = MODEL_CONFIG.llm_model  # 'skt/ax4'
max_tokens = MODEL_CONFIG.llm_max_tokens  # 4000
temperature = MODEL_CONFIG.temperature  # 0.0

# 로딩 모드 확인
mode_desc = MODEL_CONFIG.get_loading_mode_description()
```

---

### 3. PROCESSING_CONFIG (ProcessingConfig)
**목적**: 엔티티 추출 및 매칭 동작 제어

**임계값 설정:**
```python
# 엔티티 인식 임계값
entity_fuzzy_threshold: 0.5           # Fuzzy 매칭
entity_similarity_threshold: 0.2      # Sequence 유사도
entity_combined_similarity_threshold: 0.2  # 결합 유사도
entity_high_similarity_threshold: 1.0 # 최종 필터링
entity_llm_fuzzy_threshold: 0.6       # LLM 기반 추출

# 매장 매칭 임계값
store_matching_threshold: 0.5
similarity_threshold_for_store: 0.6
similarity_threshold_for_store_secondary: 0.3
```

**추출 모드:**
| 설정 | 옵션 | 설명 |
|------|------|------|
| `product_info_extraction_mode` | rag/llm/nlp | 상품 정보 추출 전략 |
| `entity_extraction_mode` | llm/logic | 엔티티 매칭 전략 |

**사용 예시:**
```python
from config.settings import PROCESSING_CONFIG

# 임계값 접근
fuzzy_threshold = PROCESSING_CONFIG.entity_fuzzy_threshold
extraction_mode = PROCESSING_CONFIG.entity_extraction_mode

# Chain of Thought 가져오기
cot = PROCESSING_CONFIG.chain_of_thought

# 추출 가이드 생성
guide = PROCESSING_CONFIG.get_extraction_guide(
    candidate_items=['아이폰 17', '갤럭시']
)
```

---

### 4. METADATA_CONFIG (METADATAConfig)
**목적**: 데이터 파일 경로 관리

**환경변수:**
- `ALIAS_RULE_PATH`: 별칭 규칙 CSV
- `STOP_ITEM_PATH`: 불용어 CSV
- `OFFER_DATA_PATH`: 상품 정보 CSV
- `ORG_INFO_PATH`: 조직 정보 CSV
- `PGM_INFO_PATH`: 프로그램 분류 CSV
- `MMS_MSG_PATH`: MMS 메시지 샘플 CSV

**사용 예시:**
```python
from config.settings import METADATA_CONFIG

# 파일 경로 접근
alias_path = METADATA_CONFIG.alias_rules_path
offer_path = METADATA_CONFIG.offer_data_path
```

---

### 5. EMBEDDING_CONFIG (EmbeddingConfig)
**목적**: 임베딩 및 모델 파일 경로 관리

**캐시 파일:**
- `item_embeddings_path`: 상품 임베딩 (.npz)
- `org_all_embeddings_path`: 조직 전체 임베딩
- `org_nm_embeddings_path`: 조직명 임베딩

**모델 경로:**
- `local_model_base_path`: 로컬 모델 기본 경로
- `ko_sbert_model_path`: 한국어 SBERT 모델 경로

---

### 6. STORAGE_CONFIG (StorageConfig)
**목적**: DAG 이미지 저장 및 URL 관리

**저장 모드:**
| 모드 | 설명 | URL 형식 |
|------|------|---------|
| **local** | API 서버에서 제공 | `http://{server_ip}:8000/dag_images/{filename}` |
| **nas** | NAS 서버에서 제공 | `http://172.27.7.58/dag_images/{filename}` |

**환경변수:**
- `DAG_STORAGE_MODE`: 저장 모드 (local/nas)
- `LOCAL_BASE_URL`: 로컬 서버 URL (자동 감지 가능)
- `LOCAL_PORT`: 로컬 서버 포트 (기본 8000)
- `NAS_BASE_URL`: NAS 서버 URL
- `NAS_URL_PATH`: NAS URL 경로

**사용 예시:**
```python
from config.settings import STORAGE_CONFIG

# DAG 이미지 URL 생성
dag_url = STORAGE_CONFIG.get_dag_image_url('dag_12345.png')
# local 모드: http://192.168.1.100:8000/dag_images/dag_12345.png
# nas 모드: http://172.27.7.58/dag_images/dag_12345.png

# 저장 디렉토리
dag_dir = STORAGE_CONFIG.get_dag_images_dir()  # 'dag_images'

# 모드 설명
desc = STORAGE_CONFIG.get_storage_description()
```

---

## 💡 전체 사용 예시

```python
from config.settings import (
    API_CONFIG,
    MODEL_CONFIG,
    PROCESSING_CONFIG,
    METADATA_CONFIG,
    EMBEDDING_CONFIG,
    STORAGE_CONFIG
)

# 1. LLM 초기화
from utils.llm_factory import LLMFactory

factory = LLMFactory(
    api_config=API_CONFIG,
    model_config=MODEL_CONFIG
)
llm = factory.create_model(MODEL_CONFIG.llm_model)

# 2. 데이터 로드
from services.item_data_loader import ItemDataLoader

loader = ItemDataLoader(data_source='local')
item_df, alias_df = loader.load_and_prepare_items(
    offer_data_path=METADATA_CONFIG.offer_data_path,
    alias_rules_path=METADATA_CONFIG.alias_rules_path,
    excluded_domains=PROCESSING_CONFIG.excluded_domain_codes_for_items,
    user_entities=PROCESSING_CONFIG.user_defined_entities
)

# 3. 엔티티 추출 설정
extraction_mode = PROCESSING_CONFIG.entity_extraction_mode
fuzzy_threshold = PROCESSING_CONFIG.entity_fuzzy_threshold

# 4. DAG 이미지 URL 생성
dag_url = STORAGE_CONFIG.get_dag_image_url('dag_example.png')
```

---

## ⚙️ 환경변수 우선순위

모든 설정은 다음 우선순위로 결정됩니다:
1. **환경변수** (`.env` 파일 또는 시스템 환경변수)
2. **기본값** (dataclass 필드 기본값)

### .env 파일 예시
```bash
# API 설정
CUSTOM_API_KEY=your_api_key_here
CUSTOM_BASE_URL=https://api.platform.a15t.com/v1

# 모델 설정
LLM_MODEL=skt/ax4
MODEL_LOADING_MODE=auto

# 처리 설정
ENTITY_EXTRACTION_MODE=llm
PRODUCT_INFO_EXTRACTION_MODE=llm

# 저장 설정
DAG_STORAGE_MODE=local
LOCAL_PORT=8000
```

---

## 📝 참고사항

- 모든 설정 클래스는 `@dataclass` 데코레이터 사용
- `__post_init__` 메서드로 초기화 후 검증 수행
- 환경변수는 `os.getenv()`로 안전하게 로드
- 글로벌 싱글톤 인스턴스로 제공 (API_CONFIG, MODEL_CONFIG 등)
- 타입 힌트로 IDE 자동완성 지원

"""
import os
import socket
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


def get_server_ip() -> str:
    """Get the server's IP address dynamically.
    
    Returns:
        str: Server's IP address (e.g., '192.168.1.100')
    """
    try:
        # Create a socket connection to get the actual network IP
        # We don't actually connect, just use it to determine the IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))  # Google DNS, doesn't actually send data
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        # Fallback to hostname-based resolution
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            # Ultimate fallback
            return "127.0.0.1"


@dataclass
class APIConfig:
    """API configuration settings for various LLM services."""
    
    # Custom LLM API configuration (e.g., local or hosted models)
    llm_api_key: str = os.getenv("CUSTOM_API_KEY", "")  # API key for custom LLM service
    llm_api_url: str = os.getenv("CUSTOM_BASE_URL", "https://api.platform.a15t.com/v1")  # Base URL for custom LLM API
    
    # OpenAI API configuration
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")  # OpenAI API key for GPT models
    
    # Anthropic API configuration  
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")  # Anthropic API key for Claude models

@dataclass
class METADATAConfig:
    """Data file path configuration settings.
    These paths point to various CSV/data files used by the system.
    """
    
    # Alias rules file for item name variations
    alias_rules_path: str = os.getenv("ALIAS_RULE_PATH", "./data/alias_rules.csv")  # CSV file containing item name aliases and variations
    
    # Stop words file for filtering unwanted terms
    stop_items_path: str = os.getenv("STOP_ITEM_PATH", "./data/stop_words.csv")  # CSV file with words to exclude from entity extraction
    
    # Main item/offer information database
    offer_data_path: str = os.getenv("OFFER_DATA_PATH", "./data/offer_master_data.csv")  # Main CSV file with item/offer information (DB schema compatible)
    
    # Organization/store information database
    org_info_path: str = os.getenv("ORG_INFO_PATH", "./data/offer_master_data.csv")  # CSV file with organization/store details (Korean encoding)
    
    # Program classification information
    pgm_info_path: str = os.getenv("PGM_INFO_PATH", "./data/pgm_tag_ext_250516.csv")  # CSV file with program classification tags and clues
    
    # MMS message samples for testing
    mms_msg_path: str = os.getenv("MMS_MSG_PATH", "./data/mms_data_250408.csv")  # CSV file with sample MMS messages for testing

@dataclass
class EmbeddingConfig:
    """Embedding and model file path configuration settings.
    These paths point to pre-computed embeddings and model files.
    """
    
    # Pre-computed embedding cache files (NumPy .npz format)
    item_embeddings_path: str = os.getenv("ITEM_EMBEDDINGS_PATH", "./data/item_embeddings_250527.npz")  # Cached embeddings for item names
    org_all_embeddings_path: str = os.getenv("ORG_ALL_EMBEDDINGS_PATH", "./data/org_all_embeddings_250605.npz")  # Cached embeddings for organization info (name + address)
    org_nm_embeddings_path: str = os.getenv("ORG_NM_EMBEDDINGS_PATH", "./data/org_nm_embeddings_250605.npz")  # Cached embeddings for organization names only
    
    # Local model storage paths
    local_model_base_path: str = os.getenv("LOCAL_MODEL_BASE_PATH", "./models")  # Base directory for storing local models
    ko_sbert_model_path: str = os.getenv("KO_SBERT_MODEL_PATH", "./models/ko-sbert-nli")  # Path to Korean SBERT model for embeddings
    
@dataclass
class ModelConfig:
    """Model configuration settings for various AI models used in the system."""
    
    # Embedding model configuration
    embedding_model: str = "jhgan/ko-sbert-nli"  # Hugging Face model ID for Korean sentence embeddings
    local_embedding_model_path: str = os.getenv("LOCAL_EMBEDDING_MODEL_PATH", "./models/ko-sbert-nli")  # Local path for embedding model
    model_loading_mode: str = os.getenv("MODEL_LOADING_MODE", "auto")  # Model loading strategy: 'auto', 'local', 'remote'
    disable_embedding: bool = os.getenv("DISABLE_EMBEDDING", "false").lower() == "true"  # Disable embedding model for server environments
    
    # LLM model specifications
    gemma_model: str = "skt/gemma3-12b-it"  # Gemma model ID for Korean language processing
    gemini_model: str = "gcp/gemini-2.5-flash" 
    claude_model: str = "amazon/anthropic/claude-sonnet-4-20250514"  # Anthropic Claude model for advanced reasoning
    ax_model: str = "skt/ax4"
    gpt_model: str = "azure/openai/gpt-4o-2024-08-06"  # OpenAI GPT-4o model for high-quality reasoning  
    
    # Active LLM selection
    llm_model: str = os.getenv("LLM_MODEL", "skt/ax4")  # Currently active LLM: 'gemma', 'ax', or 'claude'
    
    # LLM generation parameters
    llm_max_tokens: int = 4000  # Maximum tokens for LLM responses
    temperature: float = 0.0  # Temperature for LLM generation (0.0 = deterministic, 1.0 = creative)
    llm_seed: int = 42  # Seed for LLM generation
    
    def __post_init__(self):
        """Validate model loading mode after initialization."""
        valid_modes = ['auto', 'local', 'remote']
        if self.model_loading_mode not in valid_modes:
            raise ValueError(f"model_loading_mode must be one of {valid_modes}, got: {self.model_loading_mode}")
    
    def get_loading_mode_description(self) -> str:
        """Get human-readable description of current loading mode."""
        descriptions = {
            'auto': 'Automatically use local model if available, otherwise download from internet',
            'local': 'Only use local models, fail if not found (offline mode)',
            'remote': 'Always download from internet, ignore local models'
        }
        return descriptions.get(self.model_loading_mode, 'Unknown mode')

@dataclass
class StorageConfig:
    """Storage configuration for DAG images and other files."""
    
    # DAG image storage mode: 'local' or 'nas'
    # This controls URL generation, not file storage location
    dag_storage_mode: str = "local"  # Will be overridden in __post_init__
    
    # Storage path (single directory for all modes)
    dag_images_dir: str = "dag_images"  # DAG images directory (can be symlink to NAS)
    
    # Server URL configuration
    local_base_url: str = ""  # Will be overridden in __post_init__
    local_port: int = 8000  # Will be overridden in __post_init__
    nas_base_url: str = "http://172.27.7.58"  # Will be overridden in __post_init__
    nas_url_path: str = "/dag_images"  # Will be overridden in __post_init__
    
    def __post_init__(self):
        """Validate storage mode and auto-detect server IP if needed."""
        # 환경변수에서 값을 읽어서 덮어쓰기 (런타임 결정)
        self.dag_storage_mode = os.getenv("DAG_STORAGE_MODE", "local")
        self.local_base_url = os.getenv("LOCAL_BASE_URL", "")
        self.local_port = int(os.getenv("LOCAL_PORT", "8000"))
        self.nas_base_url = os.getenv("NAS_BASE_URL", "http://172.27.7.58")
        self.nas_url_path = os.getenv("NAS_URL_PATH", "/dag_images")
        
        # Validate storage mode
        valid_modes = ['local', 'nas']
        if self.dag_storage_mode not in valid_modes:
            raise ValueError(f"dag_storage_mode must be one of {valid_modes}, got: {self.dag_storage_mode}")
        
        # Auto-detect local server IP if LOCAL_BASE_URL not set
        if not self.local_base_url:
            server_ip = get_server_ip()
            self.local_base_url = f"http://{server_ip}:{self.local_port}"
    
    def get_dag_images_dir(self) -> str:
        """Get the DAG images directory (same for all storage modes)."""
        return self.dag_images_dir
    
    def get_storage_description(self) -> str:
        """Get human-readable description of current storage mode."""
        descriptions = {
            'local': 'API server provides images (URL: API server IP)',
            'nas': 'NAS server provides images (URL: NAS server IP)'
        }
        return descriptions.get(self.dag_storage_mode, 'Unknown mode')
    
    def get_dag_image_url(self, filename: str) -> str:
        """Get the DAG image URL based on storage mode.
        
        Args:
            filename: DAG image filename (e.g., 'dag_xxx.png')
        
        Returns:
            str: Full URL to access the DAG image
        """
        if self.dag_storage_mode == 'nas':
            # Use NAS server absolute URL (NAS IP address)
            return f"{self.nas_base_url.rstrip('/')}{self.nas_url_path.rstrip('/')}/{filename}"
        else:
            # Use API server absolute URL (fixed server address)
            return f"{self.local_base_url.rstrip('/')}/dag_images/{filename}"

@dataclass
class ProcessingConfig:
    """Processing configuration settings that control the behavior of entity extraction and matching."""
    
    # Similarity thresholds for matching
    similarity_threshold: float = 0.7  # Minimum similarity score for entity matching (0.0-1.0)
    similarity_threshold_for_store: float = 0.6  # Minimum similarity score for entity matching (0.0-1.0)
    similarity_threshold_for_store_secondary: float = 0.3  # Minimum similarity score for entity matching (0.0-1.0)
    fuzzy_threshold: float = 0.4  # Minimum fuzzy matching score for initial filtering (0.0-1.0)
    combined_similarity_threshold: float = 0.4  # Minimum threshold for combined similarity scores (s1, s2)
    high_similarity_threshold: float = 1.1  # Minimum high similarity score for final entity filtering (0.0-2.0)
    
    # Processing parameters
    num_candidate_programs: int = 5  # Number of candidate programs to consider for classification
    batch_size: int = 100  # Batch size for parallel processing operations
    n_jobs: int = 6  # Number of parallel jobs for similarity calculations
    
    excluded_domain_codes_for_items: List[str] = None # Domain codes to exclude from item processing (e.g., ['R'] for agency domains)

    # User-defined entities that should always be recognized
    user_defined_entities: List[str] = None  # Custom entities to add to the recognition vocabulary
    
    # Processing mode configurations
    product_info_extraction_mode: str = 'llm'  # Product extraction strategy: 'rag', 'llm', 'nlp'
    entity_extraction_mode: str = 'llm'  # Entity matching strategy: 'llm', 'logic'
    
    # === Threshold Settings (임계값 설정) ===
    # Entity Recognition Thresholds (엔티티 인식 임계값)
    entity_fuzzy_threshold: float = 0.5  # Fuzzy matching threshold for entity recognition
    entity_similarity_threshold: float = 0.2  # Sequence similarity threshold
    entity_combined_similarity_threshold: float = 0.2  # Combined similarity threshold
    entity_high_similarity_threshold: float = 1.0  # High similarity threshold for filtering
    entity_llm_fuzzy_threshold: float = 0.6  # Fuzzy threshold for LLM-based entity extraction
    
    # Store Matching Thresholds (매장 매칭 임계값)
    store_matching_threshold: float = 0.5  # Threshold for store name matching
    
    # Parallel Processing Thresholds (병렬 처리 임계값)
    parallel_fuzzy_threshold: float = 0.5  # Default threshold for parallel fuzzy similarity

    # 엔티티 추출 프롬프트는 이제 prompts 디렉토리에서 관리됩니다.
    # prompts.DETAILED_ENTITY_EXTRACTION_PROMPT 를 사용하세요.
    entity_extraction_prompt: str = None  # Deprecated: Use prompts.DETAILED_ENTITY_EXTRACTION_PROMPT instead
    
    def __post_init__(self):
        """Initialize default values and validate configuration after creation."""
        # Set default excluded domain codes if none provided
        if self.excluded_domain_codes_for_items is None:
            self.excluded_domain_codes_for_items = ['R']  # 'R' represents agency/dealer domain codes
        
        # Set default user-defined entities if none provided
        if self.user_defined_entities is None:
            self.user_defined_entities = [
                'AIA Vitality',  # Insurance/health program
                '부스트 파크 건대입구',  # Specific location/venue
                'Boost Park 건대입구'  # English variant of the above
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
        """Get processing chain of thought based on extraction mode.
        
        Returns:
            str: Step-by-step processing instructions for the chosen mode
        """
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
        """Get extraction guidelines based on current mode and available candidate items.
        
        Args:
            candidate_items: List of candidate item names to guide extraction
            
        Returns:
            str: Extraction guidelines for the current configuration
        """
        base_guide = "* 상품 추출시 정확도(precision) 보다는 재현율(recall)에 중심을 두어라."
        
        if self.product_info_extraction_mode == 'rag' and candidate_items:
            return f"""{base_guide}
* 후보 상품 이름 목록에 포함된 상품 이름은 참고하여 Product 정보를 추출하라."""
        elif self.product_info_extraction_mode == 'nlp':
            return "* Product 정보에서 position, action 필드의 정보를 추출하라."
        else:
            return base_guide

# Global configuration instances
# These are singleton instances that can be imported and used throughout the application
API_CONFIG = APIConfig()  # API keys and endpoints
MODEL_CONFIG = ModelConfig()  # AI model configurations
PROCESSING_CONFIG = ProcessingConfig()  # Processing behavior settings
METADATA_CONFIG = METADATAConfig()  # Data file paths
EMBEDDING_CONFIG = EmbeddingConfig()  # Embedding and model file paths
STORAGE_CONFIG = StorageConfig()  # Storage configuration for DAG images
DATABASE_CONFIG = DatabaseConfig()  # Database table names and queries