"""
엔티티 추출 관련 프롬프트 템플릿
NLP 기반 엔티티 추출에 사용되는 프롬프트들
"""

# 기본 엔티티 추출 프롬프트
DEFAULT_ENTITY_EXTRACTION_PROMPT = "다음 메시지에서 상품명을 추출하세요."

# 상세한 엔티티 추출 프롬프트 (settings.py에서 이동)
DETAILED_ENTITY_EXTRACTION_PROMPT = """
Extract all product names from the advertisement text, including tangible products, services, promotional events, programs, loyalty initiatives, and named campaigns.

Guidelines:
1. Prioritize string matching with the provided candidate entities list, but include additional relevant items if they clearly fit the criteria
2. Include named offerings such as apps, membership programs, events, branded items, or campaign names (e.g., 'T day', '0 day') when presented as distinct products or services
3. Extract full, specific product names (e.g., 'FLO 앤 데이터') rather than base brand names (e.g., 'FLO') when they appear as components of longer names
4. Exclude:
   - Customer support services (customer centers, helplines)
   - Descriptive modifiers that are not standalone offerings (e.g., "디지털 전용")
   - Sales agency names (e.g., '###대리점')
   - Platform or brand names unless presented as standalone offerings
5. For related promotional events, include the overarching campaign name plus specific offerings tied to it, unless identical
6. Preserve exact text as it appears in the original (no translation)
7. Prioritize recall over precision - capture all relevant products

Return only a comma-separated list of matched entities.
"""

SIMPLE_ENTITY_EXTRACTION_PROMPT = """
Select product/service names from 'candidate entities' that are directly mentioned and promoted in the message.

Guidelines:
1. Only include entities explicitly offered/promoted in the message
2. Exclude general concepts not tied to specific offerings  
3. Consider message context and product categories (plan, service, device, app, event, coupon, etc.)
4. Multiple entities from 'entities in message' may combine into one composite entity - this is also a valid selection

Return format:
REASON: Brief explanation (max 50 chars Korean) - why the candidate entity directly corresponds to the message's core offer
ENTITY: comma-separated list from candidates, or empty if none match
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
