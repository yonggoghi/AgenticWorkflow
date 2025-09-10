"""
메인 정보 추출 프롬프트 템플릿
광고 메시지에서 제목, 목적, 상품, 채널, 프로그램 정보를 추출하는 프롬프트
"""

# 사고 과정 정의 (모드별)
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

# JSON 스키마 정의
JSON_SCHEMA = {
    "title": "Advertisement title, using the exact expressions as they appear in the original text.",
    "purpose": {
        "type": "array",
        "items": {
            "type": "string",
            "enum": ["상품 가입 유도", "대리점/매장 방문 유도", "웹/앱 접속 유도", "이벤트 응모 유도", 
                   "혜택 안내", "쿠폰 제공 안내", "경품 제공 안내", "수신 거부 안내", "기타 정보 제공"]
        },
        "description": "Primary purpose(s) of the advertisement."
    },
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
                "value": {"type": "string", "description": "Specific information for the channel."},
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
* Prioritize recall over precision to ensure all relevant products are captured, but verify that each extracted term is a distinct offering.
* Extract all information (title, purpose, product, channel, pgm) using the exact expressions as they appear in the original text without translation, as specified in the schema.
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
- For "product": return an array of objects with "name" and "action" fields
- For "channel": return an array of objects with "type", "value", and "action" fields
- For "pgm": return an array of strings

Example response format:
{{
    "title": "실제 광고 제목",
    "purpose": ["상품 가입 유도", "혜택 안내"],
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
