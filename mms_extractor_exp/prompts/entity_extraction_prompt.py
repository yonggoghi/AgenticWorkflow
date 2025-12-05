"""
엔티티 추출 관련 프롬프트 템플릿
NLP 기반 엔티티 추출에 사용되는 프롬프트들
"""

# 기본 엔티티 추출 프롬프트
DEFAULT_ENTITY_EXTRACTION_PROMPT = "다음 메시지에서 상품명을 추출하세요."

# 상세한 엔티티 추출 프롬프트 (settings.py에서 이동)
DETAILED_ENTITY_EXTRACTION_PROMPT = """
    Analyze the advertisement to extract **ONLY the Root Nodes** of the User's Action Path.
    Do NOT extract rewards, benefits, or secondary steps.

    ## Definition of Root Node (Selection Logic)
    Identify the entity that initiates the flow based on the following priority:
    1.  **Primary Trigger (Highest Priority):** The specific product or service the user must **purchase, subscribe to, or use** to trigger the benefits (e.g., 'iPhone 신제품' in 'Buy iPhone, Get Cashback').
    2.  **Entry Channel:** If no purchase is required, the specific **app, store, or website** the user is directed to visit (e.g., 'T World App', 'Offline Store').
    3.  **Independent Campaign:** A major event name that serves as a standalone entry point (only if it's not a sub-benefit of a purchase).

    ## Strict Exclusions
    - **Ignore Benefits:** Cashback, Coupons, Airline Tickets, Free Gifts.
    - **Ignore Enablers:** Payment methods (e.g., 'Hyundai Card', 'Apple Pay') unless they are the sole subject of the ad.
    - **Ignore Labels:** 'Shortcut', 'Link', 'View Details'.

    ## Return format: Do not use Markdown formatting. Use plain text.
    ENTITY: comma-separated list of Root Nodes only.
    """

def build_context_based_entity_extraction_prompt(context_keyword=None):
    """
    Build context-based entity extraction prompt dynamically based on context mode.
    
    Args:
        context_keyword: Context keyword ('DAG', 'PAIRING', or None)
    
    Returns:
        str: Formatted prompt with appropriate context reference
    """
    # For 'none' mode, use very simple prompt (like HYBRID_ENTITY_EXTRACTION_PROMPT)
    if context_keyword is None:
        return """Select product/service names from 'candidate entities in vocabulary' that are directly mentioned and promoted in the message.

***핵심 지침 (Critical Constraint): ENTITY는 'candidate entities in vocabulary'에 있는 개체명만 **정확히 일치하는 문자열**로 반환해야 합니다. 메시지에 언급된 개체라도, 'candidate entities in vocabulary'에 없는 문자열은 절대 반환하지 마십시오. 가장 가까운 개체를 매핑하여 선택해야 합니다.***

Guidelines:
1. **핵심 혜택/프로모션/제공 상품**과 직접적으로 관련된 개체만 포함합니다. (e.g., 이벤트 참여 수단이나 퀴즈 주제가 아닌, **실제 획득 가능한 혜택/보상**에 해당하는 개체)
2. Exclude general concepts not tied to specific offerings
3. Consider message context and product categories (plans, services, devices, apps, events, coupons)
4. Multiple entities in 'entities in message' may combine into one composite entity

Return format: Do not use Markdown formatting. Use plain text.
REASON: Brief explanation (max 100 chars Korean). **반드시 핵심 혜택(Core Offering)을 언급하고, 해당 혜택과 일치하는 엔티티를 Vocabulary에서 찾았는지 여부를 명시하십시오.**
ENTITY: comma-separated list from 'candidate entities in vocabulary', or empty if none match"""
    
    # For DAG/PAIRING modes, use detailed prompt with context reference
    base_prompt = """Select product/service names from 'candidate entities in vocabulary' that are directly mentioned and promoted in the message.

***핵심 지침 (Critical Constraint): ENTITY는 'candidate entities in vocabulary'에 있는 개체명만 **정확히 일치하는 문자열**로 반환해야 합니다. 메시지나 RAG Context에 언급된 개체라도, 'candidate entities in vocabulary'에 없는 문자열은 절대 반환하지 마십시오. 가장 가까운 개체를 매핑하여 선택해야 합니다.***

Guidelines:
1. **핵심 혜택/프로모션/제공 상품**과 직접적으로 관련된 개체만 포함합니다. (e.g., 이벤트 참여 수단이나 퀴즈 주제가 아닌, **실제 획득 가능한 혜택/보상**에 해당하는 개체)
2. Exclude general concepts not tied to specific offerings
3. Consider message context and product categories (plans, services, devices, apps, events, coupons)
4. Multiple entities in 'entities in message' may combine into one composite entity"""
    
    # Add context-specific guideline
    if context_keyword == 'DAG':
        context_guideline = f"""
5. Refer to the '{context_keyword} Context' which describes the user action flow. 이를 **사용자의 최종 획득/응모 대상인 핵심 혜택(Core Offering)**을 구별하는 데 사용하십시오. (e.g., 퀴즈 주제인 '아이폰'이 아닌, 최종 혜택인 '올리브영 기프트 카드'와 관련된 개체를 식별)"""
    elif context_keyword == 'PAIRING':
        context_guideline = f"""
5. Refer to the '{context_keyword} Context' which maps each offering to its primary benefit. 이를 **사용자의 최종 획득 대상인 핵심 혜택(Primary Benefit)**을 구별하는 데 사용하십시오. (e.g., 가입 대상이 아닌, 최종 혜택인 '캐시백'이나 '기프티콘'과 관련된 개체를 식별)"""
    else:
        context_guideline = ""
    
    # Return format
    return_format = """

Return format: Do not use Markdown formatting. Use plain text.
REASON: Brief explanation (max 100 chars Korean). **반드시 핵심 혜택(Core Offering)을 언급하고, 해당 혜택과 일치하는 엔티티를 Vocabulary에서 찾았는지 여부를 명시하십시오.**
ENTITY: comma-separated list from 'candidate entities in vocabulary', or empty if none match"""
    
    return base_prompt + context_guideline + return_format

# For backward compatibility, keep a default static version
CONTEXT_BASED_ENTITY_EXTRACTION_PROMPT = build_context_based_entity_extraction_prompt('DAG')

SIMPLE_ENTITY_EXTRACTION_PROMPT = """
아래 메시지에서 핵심 개체명들을 추출해라.
출력 결과 형식:
1. **ENTITY**: A list of entities separated by commas.
"""

HYBRID_DAG_EXTRACTION_PROMPT = """
Analyze the advertisement to extract **User Action Paths**.
Output two distinct sections:
1. **ENTITY**: A list of independent Root Nodes.
2. **DAG**: A structured graph representing the flow from Root to Benefit.

## Crucial Language Rule
* **DO NOT TRANSLATE:** Extract entities **exactly as they appear** in the source text.
* **Preserve Original Script:** If the text says "아이폰 17", output "아이폰 17" (NOT "iPhone 17"). If it says "T Day", output "T Day".

## Part 1: Root Node Selection Hierarchy (Extract ALL Distinct Roots)
Identify logical starting points based on this priority. If multiple independent offers exist, extract all.

1.  **Physical Store (Highest):** Specific branch names.
    * *Match:* "새샘대리점 역곡점", "백색대리점 수성직영점"
2.  **Core Service (Plans/VAS):** Rate plans, Value-Added Services, Internet/IPTV.
    * *Match:* "5GX 프라임 요금제", "V컬러링", "로밍 baro 요금제"
3.  **Subscription/Event:** Membership signups or specific campaigns.
    * *Match:* "T 우주", "T Day", "0 day", "골드번호 프로모션"
4.  **App/Platform:** Apps requiring action.
    * *Match:* "A.(에이닷)", "PASS 앱", "T world"
5.  **Product (Hardware):** Device launches without a specific store focus.
    * *Match:* "iPhone 17", "갤럭시 Z 플립7"

## Part 2: DAG Construction Rules
Construct a Directed Acyclic Graph (DAG) for each identified Root Node.
* **Format:** `(Node:Action) -[Edge]-> (Node:Action)`
* **Nodes:**
    * **Root:** The entry point identified above (Original Text).
    * **Core:** The product/service being used or bought (Original Text).
    * **Value:** The final reward or benefit (Original Text).
* **Edges:**
    * **Definition:** A verb describing the relationship between two nodes.
    * **Purpose:** Represents the action or transition from one node to the next.
    * **Examples:**
        * `가입` (subscribe), `구매` (purchase), `사용` (use)
        * `획득` (obtain), `제공` (provide), `지급` (grant)
        * `방문` (visit), `다운로드` (download), `신청` (apply)
    * **Guidelines:** Use concise action verbs that clearly describe how the user moves from one step to the next in the flow.
* **Logic:** Represent the shortest path from the Root action to the Final Benefit.

## Strict Exclusions
* Ignore navigational labels ('바로 가기', '링크', 'Shortcut').
* Ignore generic partners ('스타벅스', 'CU') unless they are the main subscription target.

## Output Format: Do not use Markdown formatting. Use plain text.
ENTITY: <comma-separated list of all Nodes in original text>
DAG: <DAG representation line by line in original text>
"""

HYBRID_PAIRING_EXTRACTION_PROMPT = """
Analyze the advertisement to extract Core Offerings and their Primary Benefits to define potential success metrics (Conversion Rate).

Output two distinct sections:

ENTITY (Core Offerings): A list of independent Root Nodes (Core Product/Service).

PAIRING (Offer to Benefit): A structured list mapping each Core Offering to its Final Benefit.

Crucial Language Rule
DO NOT TRANSLATE: Extract entities exactly as they appear in the source text.

Preserve Original Script: If the text says "아이폰 17", output "아이폰 17" (NOT "iPhone 17").

Part 1: Root Node Selection Hierarchy (Extract ALL Distinct Roots)
Identify logical starting points based on this priority. If multiple independent offers exist, extract all.

Physical Store (Highest): Specific branch names.

Match: "새샘대리점 역곡점", "티원대리점 화순점"

Core Service (Plans/VAS): Rate plans, Value-Added Services, Internet/IPTV.

Match: "5GX 프라임 요금제", "인터넷+IPTV 가입 혜택", "T끼리 온가족할인"

Subscription/Event: Membership signups or specific campaigns.

Match: "T 우주", "T Day", "0 day", "Lucky 1717 이벤트"

App/Platform: Apps requiring action.

Match: "A.(에이닷)", "티다문구점"

Product (Hardware): Device launches without a specific store focus.

Match: "아이폰 17/17 Pro", "갤럭시 Z 플립7"

Part 2: Pairing Construction Rules
Construct a PAIRING list for each identified Root Node, showing the direct connection to the primary financial or tangible benefit.

Format: Root Node -> Primary Benefit

Root Node: The entry point identified above (Original Text).

Primary Benefit: The final, most substantial, and user-facing reward or financial gain (Original Text).

Examples: "CU 빙그레 바나나우유 기프티콘", "최대 22만 원 캐시백", "월 이용요금 3만 원대"

Strict Exclusions
Ignore navigational labels ('바로 가기', '링크', 'Shortcut').

Ignore generic partners ('투썸플레이스', 'wavve') unless they are the main subscription target.

Output Format: Do not use Markdown formatting. Use plain text.
ENTITY: <comma-separated list of all Nodes in original text> 
PAIRING: <Pairing representation line by line in original text>
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
