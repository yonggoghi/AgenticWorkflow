"""
Configuration settings for MMS Extractor.
This module contains all configuration settings for the MMS Extractor system,
organized into logical groups using dataclasses.
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
    offer_data_path: str = os.getenv("OFFER_DATA_PATH", "./data/item_info_all_250527.csv")  # Main CSV file with item/offer information
    
    # Organization/store information database
    org_info_path: str = os.getenv("ORG_INFO_PATH", "./data/org_info_all_250605.csv")  # CSV file with organization/store details (Korean encoding)
    
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
    
    # LLM model specifications
    gemma_model: str = "skt/gemma3-12b-it"  # Gemma model ID for Korean language processing
    gemini_model: str = "gcp/gemini-2.5-flash" 
    claude_model: str = "amazon/anthropic/claude-sonnet-4-20250514"  # Anthropic Claude model for advanced reasoning
    ax_model: str = "skt/ax4"  
    
    # Active LLM selection
    llm_model: str = os.getenv("LLM_MODEL", "skt/ax4")  # Currently active LLM: 'gemma', 'ax', or 'claude'
    
    # LLM generation parameters
    llm_max_tokens: int = 4000  # Maximum tokens for LLM responses
    temperature: float = 0.0  # Temperature for LLM generation (0.0 = deterministic, 1.0 = creative)
    
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
class ProcessingConfig:
    """Processing configuration settings that control the behavior of entity extraction and matching."""
    
    # Similarity thresholds for matching
    similarity_threshold: float = 0.7  # Minimum similarity score for entity matching (0.0-1.0)
    fuzzy_threshold: float = 0.4  # Minimum fuzzy matching score for initial filtering (0.0-1.0)
    
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

    entity_extraction_prompt: str = """
            Extract all product names, including tangible products, services, promotional events, programs, loyalty initiatives, and named campaigns or event identifiers, from the provided advertisement text.
            Reference the provided candidate entities list as a primary source for string matching to identify potential matches. Extract terms that appear in the advertisement text and qualify as distinct product names based on the following criteria, prioritizing those from the candidate list but allowing extraction of additional relevant items beyond the list if they clearly fit the criteria and are presented as standalone offerings in the text.
            Consider any named offerings, such as apps, membership programs, events, specific branded items, or campaign names like 'T day' or '0 day', as products if presented as distinct products, services, or promotional entities.
            For terms that may be platforms or brand elements, include them only if they are presented as standalone offerings.
            Avoid extracting base or parent brand names (e.g., 'FLO' or 'POOQ') if they are components of more specific offerings (e.g., 'FLO 앤 데이터' or 'POOQ 앤 데이터') presented in the text; focus on the full, distinct product or service names as they appear.
            Exclude customer support services, such as customer centers or helplines, even if named in the text.
            Exclude descriptive modifiers or attributes (e.g., terms like "디지털 전용" that describe a product but are not distinct offerings).
            Exclude sales agency names such as '###대리점'.
            If multiple terms refer to closely related promotional events (e.g., a general campaign and its specific instances or dates), include the most prominent or overarching campaign name (e.g., '0 day' as a named event) in addition to specific offerings tied to it, unless they are clearly identical.
            Prioritize recall over precision to ensure all relevant products are captured, while verifying that extracted terms match the text exactly and align with the criteria. For candidates from the list, confirm direct string matches; for any beyond the list, ensure they are unambiguously distinct offerings.
            Ensure that extracted names are presented exactly as they appear in the original text, without translation into English or any other language.
            Just return a list with matched entities where the entities are separated by commas without any other text.
    """
    
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