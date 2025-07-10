"""
MMS Extractor Package - Professional information extraction from MMS messages.

This package provides tools for extracting structured information from Korean MMS 
advertising messages using NLP techniques, entity recognition, and similarity matching.
"""

__version__ = "1.0.0"
__author__ = "MMS Extractor Team"

# Import main classes
from .core.mms_extractor import MMSExtractor
from .core.data_manager import DataManager
from .core.entity_extractor import KiwiEntityExtractor
from .models.language_models import LLMManager, EmbeddingManager
from .config.settings import (
    API_CONFIG, MODEL_CONFIG, DATA_CONFIG, 
    PROCESSING_CONFIG, EXTRACTION_SCHEMA
)

# Import utility functions
from .utils.text_processing import (
    clean_text, preprocess_text, extract_json_objects,
    convert_df_to_json_list, dataframe_to_markdown_prompt
)
from .utils.similarity import (
    combined_sequence_similarity, parallel_fuzzy_similarity,
    parallel_seq_similarity
)

__all__ = [
    # Main classes
    'MMSExtractor',
    'DataManager', 
    'KiwiEntityExtractor',
    'LLMManager',
    'EmbeddingManager',
    
    # Configuration
    'API_CONFIG',
    'MODEL_CONFIG',
    'DATA_CONFIG',
    'PROCESSING_CONFIG',
    'EXTRACTION_SCHEMA',
    
    # Utility functions
    'clean_text',
    'preprocess_text',
    'extract_json_objects',
    'convert_df_to_json_list',
    'dataframe_to_markdown_prompt',
    'combined_sequence_similarity',
    'parallel_fuzzy_similarity',
    'parallel_seq_similarity',
] 