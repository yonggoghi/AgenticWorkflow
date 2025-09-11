"""
엔티티 추출 관련 프롬프트 템플릿
NLP 기반 엔티티 추출에 사용되는 프롬프트들
"""

# 기본 엔티티 추출 프롬프트
DEFAULT_ENTITY_EXTRACTION_PROMPT = "다음 메시지에서 상품명을 추출하세요."

# 상세한 엔티티 추출 프롬프트 (settings.py에서 이동)
DETAILED_ENTITY_EXTRACTION_PROMPT = """
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

# LLM 기반 엔티티 추출 프롬프트 템플릿
LLM_ENTITY_EXTRACTION_PROMPT_TEMPLATE = """
{base_prompt}

## message:                
{message}

상품명을 정확히 추출해주세요. 원문의 표현을 그대로 사용하세요.
"""


def build_entity_extraction_prompt(message: str, base_prompt: str = None) -> str:
    """
    엔티티 추출용 프롬프트를 구성합니다.
    
    Args:
        message: 분석할 메시지
        base_prompt: 기본 프롬프트 (없으면 기본값 사용)
        
    Returns:
        구성된 엔티티 추출 프롬프트
    """
    if base_prompt is None:
        base_prompt = DEFAULT_ENTITY_EXTRACTION_PROMPT
    
    return LLM_ENTITY_EXTRACTION_PROMPT_TEMPLATE.format(
        base_prompt=base_prompt,
        message=message
    )
