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
아래 광고 메시지에서 수신자에게 offer하거나, 사용 유도, 구매, 가입, 응모, 접속하기를 원하는 개체명을 추출해라. 
'entities in messages'는 메시지 내에서 있는 후보 개체명들이다.
'candidate entities in vocabulairy'는 'entities in messages'를 바탕으로 추출한 사전에 있는 후보 개체명들이다.
메시지와 'entities in messages'를 참고해서 'candidate entities in vocabulary' 중에서 유력한 것들을 선택해라.
메시지 맥락을 파악해서 개체명들의 분류도 고려해라. (요금제, 부가서비스, 단말기, 앱, 이벤트, 쿠폰 등)
"아이폰", "갤럭시" 같은 일반적인 개체명인 경우, 'candidate entities in vocabulary'에서 가장 최신의 것들을 선택해라.

다음과 같은 포맷으로 결과를 반환해라.

REASON: 선택 이유
ENTITY: 제공하는 결과는 candidate entities in vocabulary 값들을 ,(콤마)로 연결해라. 없다고 판단하면, 공백으로 결과를 반환해라.
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
