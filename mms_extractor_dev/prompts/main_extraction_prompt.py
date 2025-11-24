"""
메인 정보 추출 프롬프트 모듈 (Main Extraction Prompts)
================================================================

🎯 목적
-------
MMS 광고 메시지에서 구조화된 정보를 추출하기 위한 핵심 프롬프트를 정의합니다.
LLM이 일관된 품질로 정확한 정보를 추출할 수 있도록 세심하게 설계된 프롬프트입니다.

📊 추출 대상 정보
--------------
- **제목 (title)**: 광고의 메인 제목
- **목적 (purpose)**: 광고의 주요 목적 (가입 유도, 혜택 안내 등)
- **상품 (product)**: 광고된 상품/서비스와 기대 액션
- **채널 (channel)**: 고객 접점 채널 (URL, 전화번호, 앱 링크 등)
- **프로그램 (pgm)**: 관련 프로그램 카테고리

🧠 사고 과정 (Chain of Thought)
--------------------------
모드별로 최적화된 사고 과정을 제공하여 LLM이 체계적으로 사고할 수 있도록 안내:
- **LLM_MODE**: LLM 기반 엔티티 추출 시 사용
- **DEFAULT_MODE**: 일반적인 추출 상황에서 사용
- **NLP_MODE**: NLP 기반 전처리와 결합 시 사용

📝 JSON 스키마
-----------
출력 데이터의 일관성과 유효성을 보장하기 위한 엄격한 JSON 스키마 정의

작성자: MMS 분석팀
최종 수정: 2024-09
버전: 2.0.0
"""

# =============================================================================
# 사고 과정 정의 (Chain of Thought Templates)
# =============================================================================

# LLM 모드: LLM 기반 엔티티 추출과 결합 시 사용
CHAIN_OF_THOUGHT_LLM_MODE = """
1. Identify the advertisement's purpose first, using expressions as they appear in the original text.
2. Extract ONLY explicitly mentioned product/service names from the text, using exact original expressions.
3. For each product, assign a standardized action from: [구매, 가입, 사용, 방문, 참여, 코드입력, 쿠폰다운로드, 기타].
4. Avoid inferring or adding products not directly mentioned in the text.
5. Provide channel information considering the extracted product information, preserving original text expressions.
"""

CHAIN_OF_THOUGHT_DEFAULT_MODE = """
1. Identify the advertisement's purpose first, using expressions as they appear in the original text.
2. Extract product names based on the identified purpose, ensuring only distinct offerings are included and using original text expressions.
3. Provide channel information considering the extracted product information, preserving original text expressions.
"""

CHAIN_OF_THOUGHT_NLP_MODE = """
1. Identify the advertisement's purpose first, using expressions as they appear in the original text.
2. Extract product information based on the identified purpose, ensuring only distinct offerings are included.
3. Extract the action field for each product based on the provided name information.
4. Provide channel information considering the extracted product information.
"""

# =============================================================================
# JSON 스키마 정의 (Structured Output Schema)
# =============================================================================

# 출력 데이터의 구조와 제약 조건을 엄격하게 정의
JSON_SCHEMA = {
    "title": "Extract a concise title summarizing the key content of the advertisement in one sentence (max 50 characters). Guidelines: (1) Clearly capture the core content (benefits, products, events, etc.), (2) Write concisely in one sentence, (3) Exclude labels like '(광고)', '[SKT]', (4) Remove special characters (__, etc.) and create a natural sentence, (5) Prioritize the most important information, (6) Use a headline style (개조식).",
    "purpose": {
        "type": "array",
        "items": {
            "type": "string",
            "enum": ["상품 가입 유도", "대리점/매장 방문 유도", "웹/앱 접속 유도", "이벤트 응모 유도", 
                   "혜택 안내", "쿠폰 제공 안내", "경품 제공 안내", "수신 거부 안내", "기타 정보 제공"]
        },
        "description": "Primary purpose(s) of the advertisement."
    },
    "sales_script": "Generate an extremely concise sales prompt, intended for display on a call center agent's monitor, to be used for a rapid cross-sell immediately after resolving a customer's issue. The resulting script/message should be highly condensed, containing only the absolute essential facts and a clear action cue, and must be easily readable at a glance.",
    "product": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the advertised product or service."},
                "action": {
                    "type": "string",
                    "enum": ["구매", "가입", "사용", "방문", "참여", "코드입력", "쿠폰다운로드", "기타"],
                    "description": "Expected customer action for the product."
                }
            }
        },
        "description": "Extract all product names from the advertisement."
    },
    "channel": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": ["URL", "전화번호", "앱", "대리점"],
                    "description": "Channel type."
                },
                "value": {"type": "string", "description": "Specific information for the channel. 대리점인 경우 ***점으로 표시될 가능성이 높음"},
                "action": {
                    "type": "string",
                    "enum": ["가입", "추가 정보", "문의", "수신", "수신 거부"],
                    "description": "Purpose of the channel."
                }
            }
        },
        "description": "Channels provided in the advertisement."
    },
    "pgm": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Select the two most relevant pgm_nm from the advertising classification criteria."
    }
}

# 추출 가이드라인
EXTRACTION_GUIDELINES_BASE = """
* For title: Create a concise headline (max 50 characters) in headline style (개조식) that captures the core content, excluding labels like '(광고)', '[SKT]' and special characters like __, and prioritizing the most important information (benefits, products, events).
* Prioritize recall over precision to ensure all relevant products are captured, but verify that each extracted term is a distinct offering.
* Extract all information (purpose, product, channel, pgm) using the exact expressions as they appear in the original text without translation, as specified in the schema.
* If the advertisement purpose includes encouraging agency/store visits, provide agency channel information.
"""

EXTRACTION_GUIDELINES_NLP_MODE = """
* Extract the action field for each product based on the identified product names, using the original text context.
"""

EXTRACTION_GUIDELINES_RAG_MODE = """
* Use the provided candidate product names as a reference to guide product extraction, ensuring alignment with the advertisement content and using exact expressions from the original text.
"""

EXTRACTION_GUIDELINES_LLM_MODE = """
* Refer to the candidate product names list as guidance, but extract products based on your understanding of the advertisement content.
* Maintain consistency by using standardized product naming conventions.
* If multiple similar products exist, choose the most specific and relevant one to reduce variability.
"""

# 일관성 유지 지침
CONSISTENCY_GUIDELINES = """

### 일관성 유지 지침 ###
* 동일한 광고 메시지에 대해서는 항상 동일한 결과를 생성해야 합니다.
* 애매한 표현이 있을 때는 가장 명확하고 구체적인 해석을 선택하세요.
* 상품명은 원문에서 정확히 언급된 표현만 사용하세요.
"""

# 스키마 프롬프트 템플릿
SCHEMA_PROMPT_TEMPLATE = """
Return your response as a JSON object that follows this exact structure:

{schema}

IMPORTANT: 
- Do NOT return the schema definition itself
- Return actual extracted data in the specified format
- For "purpose": return an array of strings from the enum values
- For "sales_script": return a concise string for call center agents
- For "product": return an array of objects with "name" and "action" fields
- For "channel": return an array of objects with "type", "value", and "action" fields
- For "pgm": return an array of strings

Example response format:
{{
    "title": "아이폰 신제품 출시 기념 최대 70% 할인 혜택",
    "purpose": ["상품 가입 유도", "혜택 안내"],
    "sales_script": "아이폰 신제품 출시 기념! 티원대리점 화순점 방문 시 최대 70% 할인. 지금 바로 안내드릴까요?",
    "product": [
        {{"name": "실제 상품명", "action": "가입"}},
        {{"name": "다른 상품명", "action": "구매"}}
    ],
    "channel": [
        {{"type": "URL", "value": "실제 URL", "action": "가입"}}
    ],
    "pgm": ["실제 프로그램명"]
}}
"""

# 메인 프롬프트 템플릿
MAIN_PROMPT_TEMPLATE = """
Extract the advertisement purpose and product names from the provided advertisement text.

### Advertisement Message ###
{message}

### Extraction Steps ###
{chain_of_thought}

### Extraction Guidelines ###
{extraction_guidelines}{consistency_note}

{schema_prompt}

### OUTPUT FORMAT REQUIREMENT ###
You MUST respond with a valid JSON object containing actual extracted data.
Do NOT include schema definitions, type specifications, or template structures.
Return only the concrete extracted information in the specified JSON format.

{rag_context}

### FINAL REMINDER ###
Return a JSON object with actual data, not schema definitions!
"""


def build_extraction_prompt(message: str, 
                           rag_context: str,
                           product_element=None,
                           product_info_extraction_mode: str = 'nlp') -> str:
    """
    추출용 프롬프트를 구성합니다.
    
    Args:
        message: 광고 메시지
        rag_context: RAG 컨텍스트
        product_element: 제품 요소 (NLP 모드에서 사용)
        product_info_extraction_mode: 제품 정보 추출 모드 ('nlp', 'llm', 'rag')
        
    Returns:
        구성된 프롬프트 문자열
    """
    import json
    
    # 모드별 사고 과정 선택
    if product_info_extraction_mode == 'llm':
        chain_of_thought = CHAIN_OF_THOUGHT_LLM_MODE
    else:
        chain_of_thought = CHAIN_OF_THOUGHT_DEFAULT_MODE
    
    # 스키마 복사 및 조정
    schema = JSON_SCHEMA.copy()
    extraction_guidelines = EXTRACTION_GUIDELINES_BASE
    
    # 모드별 처리
    if product_info_extraction_mode == 'nlp' and product_element:
        schema['product'] = product_element
        chain_of_thought = CHAIN_OF_THOUGHT_NLP_MODE
        extraction_guidelines += EXTRACTION_GUIDELINES_NLP_MODE
    
    # RAG/LLM 모드별 가이드라인 추가
    if "### 후보 상품 이름 목록 ###" in rag_context:
        extraction_guidelines += EXTRACTION_GUIDELINES_RAG_MODE
    elif "### 참고용 후보 상품 이름 목록 ###" in rag_context:
        extraction_guidelines += EXTRACTION_GUIDELINES_LLM_MODE
    
    # 일관성 지침 (LLM 모드에서만)
    consistency_note = ""
    if product_info_extraction_mode == 'llm':
        consistency_note = CONSISTENCY_GUIDELINES
    
    # 스키마 프롬프트 구성
    schema_prompt = SCHEMA_PROMPT_TEMPLATE.format(
        schema=json.dumps(schema, indent=4, ensure_ascii=False)
    )
    
    # 최종 프롬프트 구성
    prompt = MAIN_PROMPT_TEMPLATE.format(
        message=message,
        chain_of_thought=chain_of_thought,
        extraction_guidelines=extraction_guidelines,
        consistency_note=consistency_note,
        schema_prompt=schema_prompt,
        rag_context=rag_context
    )
    
    return prompt
