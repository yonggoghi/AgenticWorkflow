"""
Configuration settings for MMS Extractor.
This module contains all configuration settings for the MMS Extractor system,
organized into logical groups using dataclasses.
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