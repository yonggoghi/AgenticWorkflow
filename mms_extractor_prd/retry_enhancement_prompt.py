"""
재시도 및 프롬프트 강화 관련 프롬프트 템플릿
LLM 호출 실패 시 사용되는 강화된 프롬프트와 fallback 로직
"""

# 스키마 응답 방지를 위한 강화 지시사항
SCHEMA_PREVENTION_INSTRUCTION = """
🚨 CRITICAL INSTRUCTION 🚨
You MUST return actual extracted data, NOT the schema definition.

DO NOT return:
- Schema structures like {"type": "array", "items": {...}}
- Template definitions
- Example formats

DO return:
- Real extracted values from the advertisement
- Actual product names, purposes, channels found in the text
- Concrete data only

For example:
WRONG: {"purpose": {"type": "array", "items": {"type": "string"}}}
CORRECT: {"purpose": ["상품 가입 유도", "혜택 안내"]}

WRONG: {"product": {"type": "array", "items": {"type": "object"}}}
CORRECT: {"product": [{"name": "ZEM폰", "action": "가입"}]}

"""

# Fallback 결과 템플릿
FALLBACK_RESULT_TEMPLATE = {
    "title": "광고 메시지",
    "purpose": ["정보 제공"],
    "product": [],
    "channel": [],
    "pgm": []
}


def enhance_prompt_for_retry(original_prompt: str) -> str:
    """
    스키마 응답 방지를 위한 프롬프트 강화
    
    Args:
        original_prompt: 원본 프롬프트
        
    Returns:
        강화된 프롬프트
    """
    return SCHEMA_PREVENTION_INSTRUCTION + "\n" + original_prompt


def get_fallback_result() -> dict:
    """
    LLM 실패 시 사용할 fallback 결과 반환
    
    Returns:
        기본 fallback 결과 딕셔너리
    """
    return FALLBACK_RESULT_TEMPLATE.copy()
