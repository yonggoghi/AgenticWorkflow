"""
프롬프트 모듈 초기화
모든 프롬프트 관련 함수들을 임포트하여 쉽게 사용할 수 있도록 함
"""

from .main_extraction_prompt import (
    build_extraction_prompt,
    JSON_SCHEMA,
    CHAIN_OF_THOUGHT_LLM_MODE,
    CHAIN_OF_THOUGHT_DEFAULT_MODE,
    CHAIN_OF_THOUGHT_NLP_MODE
)

from .retry_enhancement_prompt import (
    enhance_prompt_for_retry,
    get_fallback_result,
    SCHEMA_PREVENTION_INSTRUCTION
)

from .dag_extraction_prompt import (
    build_dag_extraction_prompt,
    DAG_EXTRACTION_PROMPT_TEMPLATE
)

from .entity_extraction_prompt import (
    build_entity_extraction_prompt,
    DEFAULT_ENTITY_EXTRACTION_PROMPT
)

__all__ = [
    # Main extraction
    'build_extraction_prompt',
    'JSON_SCHEMA',
    'CHAIN_OF_THOUGHT_LLM_MODE',
    'CHAIN_OF_THOUGHT_DEFAULT_MODE',
    'CHAIN_OF_THOUGHT_NLP_MODE',
    
    # Retry enhancement
    'enhance_prompt_for_retry',
    'get_fallback_result',
    'SCHEMA_PREVENTION_INSTRUCTION',
    
    # DAG extraction
    'build_dag_extraction_prompt',
    'DAG_EXTRACTION_PROMPT_TEMPLATE',
    
    # Entity extraction
    'build_entity_extraction_prompt',
    'DEFAULT_ENTITY_EXTRACTION_PROMPT',
]
