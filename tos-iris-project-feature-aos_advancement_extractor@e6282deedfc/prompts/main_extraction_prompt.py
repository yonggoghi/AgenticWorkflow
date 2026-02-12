"""
메인 정보 추출 프롬프트 모듈 (Main Extraction Prompts)
================================================================

🎯 목적
-------
MMS 광고 메시지에서 구조화된 정보를 추출하기 위한 핵심 프롬프트를 정의합니다.
Few-shot 예제를 통해 LLM이 정확한 JSON 포맷으로 정보를 추출하도록 안내합니다.

📊 추출 대상 정보
--------------
- **제목 (title)**: 광고의 메인 제목
- **목적 (purpose)**: 광고의 주요 목적 (가입 유도, 혜택 안내 등)
- **세일즈 스크립트 (sales_script)**: 콜센터 상담사용 간결한 멘트
- **상품 (product)**: 광고된 상품/서비스와 기대 액션
- **채널 (channel)**: 고객 접점 채널 (URL, 전화번호, 앱 링크 등)
- **프로그램 (pgm)**: 관련 프로그램 카테고리

🧠 사고 과정 (Chain of Thought)
--------------------------
모드별로 최적화된 사고 과정을 제공하여 LLM이 체계적으로 사고할 수 있도록 안내:
- **LLM_MODE**: LLM 기반 엔티티 추출 시 사용
- **DEFAULT_MODE**: 일반적인 추출 상황에서 사용
- **NLP_MODE**: NLP 기반 전처리와 결합 시 사용

📝 Few-Shot Examples
-----------
3개의 구체적 MMS 메시지→JSON 출력 예제로 출력 포맷을 명확히 전달

"""

import json

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
# 출력 스키마 참조용 (프롬프트에는 포함하지 않음, 검증/문서용)
# =============================================================================

OUTPUT_SCHEMA_REFERENCE = {
    "title": "string (max 50 chars, headline style)",
    "purpose": ["enum: 상품 가입 유도 | 대리점/매장 방문 유도 | 웹/앱 접속 유도 | 이벤트 응모 유도 | 혜택 안내 | 쿠폰 제공 안내 | 경품 제공 안내 | 수신 거부 안내 | 기타 정보 제공"],
    "sales_script": "string (concise cross-sell script for call center)",
    "product": [{"name": "string", "action": "enum: 구매 | 가입 | 사용 | 방문 | 참여 | 코드입력 | 쿠폰다운로드 | 기타"}],
    "channel": [{"type": "enum: URL | 전화번호 | 앱 | 대리점 | 온라인스토어", "value": "string", "action": "enum: 가입 | 추가 정보 | 문의 | 수신 | 수신 거부"}],
    "pgm": ["string"],
}

# Backward compatibility alias
JSON_SCHEMA = OUTPUT_SCHEMA_REFERENCE

# =============================================================================
# Few-Shot Examples
# =============================================================================

FEW_SHOT_EXAMPLES = [
    # Example 1: Store + Equipment ad (대리점 방문 유도 + 상품 가입 유도)
    {
        "input": (
            "(광고)[SKT] CD대리점 동탄목동점에서 아이폰 17 Pro 사전예약 시작! "
            "최대 22만 원 캐시백 + 올리브영 3천 원 기프트카드 증정. "
            "매장 방문 또는 skt.sh/abc123 에서 확인하세요. "
            "수신거부 080-1234-5678"
        ),
        "output": {
            "title": "아이폰17Pro 사전예약 최대 22만원 캐시백",
            "purpose": ["상품 가입 유도", "대리점/매장 방문 유도"],
            "sales_script": "아이폰17Pro 사전예약! CD대리점 동탄목동점 방문 시 최대 22만원 캐시백+올리브영 기프트카드. 안내드릴까요?",
            "product": [
                {"name": "아이폰 17 Pro", "action": "구매"},
                {"name": "올리브영 3천 원 기프트카드", "action": "쿠폰다운로드"}
            ],
            "channel": [
                {"type": "대리점", "value": "CD대리점 동탄목동점", "action": "가입"},
                {"type": "URL", "value": "skt.sh/abc123", "action": "추가 정보"}
            ],
            "pgm": []
        }
    },
    # Example 2: Product + Campaign ad (상품 가입 유도)
    {
        "input": (
            "[SKT] 5GX 프라임 요금제 가입하고 T Day 혜택 받으세요! "
            "이번 달 T Day 기간 한정 데이터 2배 제공. "
            "T world 앱에서 바로 가입 가능합니다."
        ),
        "output": {
            "title": "5GX 프라임 요금제 T Day 데이터 2배 혜택",
            "purpose": ["상품 가입 유도", "이벤트 응모 유도"],
            "sales_script": "T Day 기간 5GX프라임 가입 시 데이터 2배! T world 앱에서 즉시 가입 가능. 안내드릴까요?",
            "product": [
                {"name": "5GX 프라임 요금제", "action": "가입"},
                {"name": "T Day", "action": "참여"}
            ],
            "channel": [
                {"type": "앱", "value": "T world 앱", "action": "가입"}
            ],
            "pgm": []
        }
    },
    # Example 3: Subscription + Voucher ad (혜택 안내 + 가입 유도)
    {
        "input": (
            "(광고) T 우주패스 올리브영&스타벅스&이마트24 구독하면 "
            "매월 올리브영 5천 원 할인 + 스타벅스 아메리카노 1잔 무료! "
            "월 9,900원으로 다양한 혜택을 누리세요. "
            "자세히 보기 skt.sh/xyz789"
        ),
        "output": {
            "title": "T 우주패스 구독 올리브영·스타벅스 혜택",
            "purpose": ["상품 가입 유도", "혜택 안내"],
            "sales_script": "T우주패스 월9,900원! 올리브영 5천원할인+스타벅스 아메리카노 무료. 가입 안내드릴까요?",
            "product": [
                {"name": "T 우주패스 올리브영&스타벅스&이마트24", "action": "가입"}
            ],
            "channel": [
                {"type": "URL", "value": "skt.sh/xyz789", "action": "추가 정보"}
            ],
            "pgm": []
        }
    },
]


def _build_few_shot_section() -> str:
    """Build the few-shot examples section for the prompt."""
    lines = ["### Examples ###"]
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
        lines.append(f"\nMessage {i}:\n{ex['input']}")
        lines.append(f"\nOutput {i}:")
        lines.append(json.dumps(ex['output'], indent=2, ensure_ascii=False))
    return "\n".join(lines)


# =============================================================================
# 추출 가이드라인 (Extraction Guidelines)
# =============================================================================

EXTRACTION_GUIDELINES_BASE = """
* For title: Create a concise headline (max 50 characters) in headline style (개조식) that captures the core content, excluding labels like '(광고)', '[SKT]' and special characters like __, and prioritizing the most important information (benefits, products, events).
* Prioritize recall over precision to ensure all relevant products are captured, but verify that each extracted term is a distinct offering.
* Extract all information (purpose, product, channel, pgm) using the exact expressions as they appear in the original text without translation.
* If the advertisement purpose includes encouraging agency/store visits, provide agency channel information.
* For purpose, select from: 상품 가입 유도, 대리점/매장 방문 유도, 웹/앱 접속 유도, 이벤트 응모 유도, 혜택 안내, 쿠폰 제공 안내, 경품 제공 안내, 수신 거부 안내, 기타 정보 제공.
* For product action, select from: 구매, 가입, 사용, 방문, 참여, 코드입력, 쿠폰다운로드, 기타.
* For channel type, select from: URL, 전화번호, 앱, 대리점, 온라인스토어. For channel action, select from: 가입, 추가 정보, 문의, 수신, 수신 거부.
* For pgm: Select up to {num_select_pgms} most relevant pgm_nm from the "광고 분류 기준 정보" section provided below. Pay close attention to the message's opening section (sender name, subject line, first sentence) as it often reveals the core campaign intent (e.g., device upgrade promotion, plan subscription, store visit campaign). Match this intent against each candidate's clue_tag to choose the best-fitting programs. You MUST copy the exact pgm_nm string as-is — do NOT shorten, paraphrase, or generate your own program names. If no "광고 분류 기준 정보" section is provided, return an empty array.
  - pgm selection hints (apply when the pattern is present in the message):
    - A specific store/agency name (e.g., "○○대리점", "○○직영점") with address or visit info → include a store-visit/opening pgm (매장오픈안내 및 방문유도) if available in the candidate list.
    - A new device name with purchase/pre-order keywords (e.g., "사전예약", "출시") → include a device upgrade pgm (기변유도) if available.
    - A rate plan or subscription service with sign-up keywords (e.g., "가입", "구독") → include a subscription pgm (가입유도) if available.
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

# =============================================================================
# 포맷 지시사항 (Format Instruction)
# =============================================================================

FORMAT_INSTRUCTION = """
Return a JSON object in the same format as the examples above.
Extract actual data from the advertisement message. Use the exact same JSON keys: title, purpose, sales_script, product, channel, pgm.
"""

# =============================================================================
# NLP 모드 제품 정보 주입 템플릿
# =============================================================================

NLP_PRODUCT_CONTEXT_TEMPLATE = """
### Pre-extracted Product Information ###
Use the following pre-extracted product names to fill the product field. Assign the appropriate action for each:
{product_element}
"""

# =============================================================================
# 메인 프롬프트 템플릿
# =============================================================================

MAIN_PROMPT_TEMPLATE = """
Extract the advertisement purpose and product names from the provided advertisement text.

{few_shot_examples}

### Advertisement Message ###
{message}

### Extraction Steps ###
{chain_of_thought}

### Extraction Guidelines ###
{extraction_guidelines}{consistency_note}

{nlp_product_context}{rag_context}

{format_instruction}
"""


def build_extraction_prompt(message: str,
                           rag_context: str,
                           product_element=None,
                           product_info_extraction_mode: str = 'nlp',
                           num_select_pgms: int = 2) -> str:
    """
    추출용 프롬프트를 구성합니다.

    Args:
        message: 광고 메시지
        rag_context: RAG 컨텍스트
        product_element: 제품 요소 (NLP 모드에서 사용)
        product_info_extraction_mode: 제품 정보 추출 모드 ('nlp', 'llm', 'rag')
        num_select_pgms: LLM이 최종 선정할 프로그램 수 (기본값: 2)

    Returns:
        구성된 프롬프트 문자열
    """
    # 모드별 사고 과정 선택
    if product_info_extraction_mode == 'llm':
        chain_of_thought = CHAIN_OF_THOUGHT_LLM_MODE
    else:
        chain_of_thought = CHAIN_OF_THOUGHT_DEFAULT_MODE

    extraction_guidelines = EXTRACTION_GUIDELINES_BASE.format(num_select_pgms=num_select_pgms)

    # NLP 모드: 사전 추출된 제품 정보 주입
    nlp_product_context = ""
    if product_info_extraction_mode == 'nlp' and product_element:
        chain_of_thought = CHAIN_OF_THOUGHT_NLP_MODE
        extraction_guidelines += EXTRACTION_GUIDELINES_NLP_MODE
        nlp_product_context = NLP_PRODUCT_CONTEXT_TEMPLATE.format(
            product_element=json.dumps(product_element, indent=2, ensure_ascii=False)
            if isinstance(product_element, (dict, list))
            else str(product_element)
        )

    # RAG/LLM 모드별 가이드라인 추가
    if "### 후보 상품 이름 목록 ###" in rag_context:
        extraction_guidelines += EXTRACTION_GUIDELINES_RAG_MODE
    elif "### 참고용 후보 상품 이름 목록 ###" in rag_context:
        extraction_guidelines += EXTRACTION_GUIDELINES_LLM_MODE

    # 일관성 지침 (LLM 모드에서만)
    consistency_note = ""
    if product_info_extraction_mode == 'llm':
        consistency_note = CONSISTENCY_GUIDELINES

    # Few-shot 예제 섹션 구성
    few_shot_examples = _build_few_shot_section()

    # 최종 프롬프트 구성
    prompt = MAIN_PROMPT_TEMPLATE.format(
        message=message,
        chain_of_thought=chain_of_thought,
        extraction_guidelines=extraction_guidelines,
        consistency_note=consistency_note,
        few_shot_examples=few_shot_examples,
        nlp_product_context=nlp_product_context,
        rag_context=rag_context,
        format_instruction=FORMAT_INSTRUCTION,
    )

    return prompt
