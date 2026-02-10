"""
Entity Extraction Prompt Templates
===================================

📋 개요
-------
엔티티 추출에 사용되는 다양한 LLM 프롬프트 템플릿을 제공합니다.
메시지 복잡도와 컨텍스트 요구사항에 따라 적절한 프롬프트를 선택할 수 있습니다.

🔗 의존성
---------
**사용되는 곳:**
- `services.entity_recognizer`: LLM 기반 엔티티 추출 시 프롬프트 선택
- `core.mms_workflow_steps`: EntityExtractionStep에서 사용

🏗️ 프롬프트 템플릿 종류
-----------------------

### 1. 컨텍스트 모드별 프롬프트

| 모드 | 프롬프트 | 용도 | 컨텍스트 |
|------|---------|------|---------|
| **DAG** | HYBRID_DAG_EXTRACTION_PROMPT | 사용자 행동 경로 분석 | DAG (Directed Acyclic Graph) |
| **PAIRING** | HYBRID_PAIRING_EXTRACTION_PROMPT | 혜택-제공물 매핑 | PAIRING (Offer → Benefit) |
| **NONE** | SIMPLE_ENTITY_EXTRACTION_PROMPT | 단순 엔티티 추출 | 없음 |

### 2. 프롬프트 선택 가이드

```python
# 복잡한 광고 (다단계 행동 경로)
context_mode = 'dag'
prompt = HYBRID_DAG_EXTRACTION_PROMPT
# 예: "T world 앱 접속 → 퀴즈 참여 → 올리브영 기프티콘 획득"

# 혜택 중심 광고 (제공물 → 혜택)
context_mode = 'pairing'
prompt = HYBRID_PAIRING_EXTRACTION_PROMPT
# 예: "아이폰 17 구매 → 최대 22만원 캐시백"

# 단순 광고 (명확한 상품명)
context_mode = 'none'
prompt = SIMPLE_ENTITY_EXTRACTION_PROMPT
# 예: "5GX 프라임 요금제 가입 혜택"
```

### 3. 2단계 엔티티 추출 프로세스

**1단계: 초기 추출 (HYBRID_DAG/PAIRING_EXTRACTION_PROMPT)**
```
입력: 원본 메시지
출력: 
  - ENTITY: 추출된 엔티티 목록
  - DAG/PAIRING: 컨텍스트 정보
```

**2단계: 필터링 (build_context_based_entity_extraction_prompt)**
```
입력:
  - 원본 메시지
  - 1단계 컨텍스트 (DAG/PAIRING)
  - entities in message (1단계 결과)
  - candidate entities in vocabulary (DB 매칭 결과)

출력:
  - REASON: 선택 이유
  - ENTITY: 최종 필터링된 엔티티
```

📊 프롬프트 구조 비교
-------------------

### HYBRID_DAG_EXTRACTION_PROMPT
**목적**: 사용자 행동 경로를 DAG로 구조화
**출력 형식**:
```
ENTITY: 상품A, 상품B, 이벤트C
DAG:
(상품A:구매) -[획득]-> (혜택B:제공)
(이벤트C:참여) -[응모]-> (혜택B:제공)
```

**특징**:
- Root Node 우선순위: 매장 > 서비스 > 이벤트 > 앱 > 제품
- 원문 언어 보존 (번역 금지)
- 독립적인 Root 모두 추출

### HYBRID_PAIRING_EXTRACTION_PROMPT
**목적**: 제공물과 혜택을 직접 매핑
**출력 형식**:
```
ENTITY: 상품A, 상품B
PAIRING:
상품A -> 캐시백 22만원
상품B -> CU 기프티콘
```

**특징**:
- 최종 혜택(Primary Benefit) 중심
- 전환율(Conversion Rate) 측정 가능
- 재무적/실질적 혜택만 포함

### SIMPLE_ENTITY_EXTRACTION_PROMPT
**목적**: 빠른 엔티티 추출
**출력 형식**:
```
ENTITY: 상품A, 상품B, 이벤트C
```

**특징**:
- Chain-of-Thought 없음
- 컨텍스트 추출 없음
- 가장 빠른 처리

💡 사용 예시
-----------
```python
from prompts.entity_extraction_prompt import (
    build_context_based_entity_extraction_prompt,
    HYBRID_DAG_EXTRACTION_PROMPT,
    HYBRID_PAIRING_EXTRACTION_PROMPT,
    SIMPLE_ENTITY_EXTRACTION_PROMPT
)

# 1. 컨텍스트 모드 선택
context_mode = 'dag'  # 'dag', 'pairing', 'none'

# 2. 1단계 프롬프트 선택
if context_mode == 'dag':
    first_stage_prompt = HYBRID_DAG_EXTRACTION_PROMPT
    context_keyword = 'DAG'
elif context_mode == 'pairing':
    first_stage_prompt = HYBRID_PAIRING_EXTRACTION_PROMPT
    context_keyword = 'PAIRING'
else:
    first_stage_prompt = SIMPLE_ENTITY_EXTRACTION_PROMPT
    context_keyword = None

# 3. 1단계 실행
prompt = f"{first_stage_prompt}\n\n## message:\n{message}"
response = llm.invoke(prompt)

# 4. 2단계 프롬프트 생성 (필터링)
second_stage_prompt = build_context_based_entity_extraction_prompt(context_keyword)

# 5. 2단계 실행
prompt = f"
{second_stage_prompt}

## message:
{message}

## DAG Context:
{extracted_dag_context}

## entities in message:
{entities_from_stage1}

## candidate entities in vocabulary:
{candidates_from_db}
"
final_response = llm.invoke(prompt)
```

📝 프롬프트 설계 원칙
-------------------

### 핵심 제약사항
1. **원문 보존**: 엔티티는 메시지 원문 그대로 추출 (번역 금지)
2. **Vocabulary 제한**: 2단계에서는 vocabulary에 있는 엔티티만 반환
3. **핵심 혜택 중심**: 이벤트 참여 수단이 아닌 최종 획득 대상 추출

### 제외 대상
- 네비게이션 라벨: '바로 가기', '링크', 'Shortcut'
- 결제 수단: 'Hyundai Card', 'Apple Pay' (단독 주제가 아닌 경우)
- 일반 파트너: '스타벅스', 'CU' (구독 대상이 아닌 경우)

📝 참고사항
----------
- `build_context_based_entity_extraction_prompt()`는 동적으로 프롬프트 생성
- context_keyword가 None이면 컨텍스트 참조 없는 간단한 프롬프트
- 모든 프롬프트는 plain text 출력 (Markdown 금지)
- REASON 필드는 핵심 혜택(Core Offering) 명시 필수

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
    
    # ONT mode uses a focused prompt for core offering entities
    if context_keyword == 'ONT':
        base_prompt = """Select product/service names from 'candidate entities in vocabulary' that match the core offerings (Product, Subscription, RatePlan, Store, Campaign, PartnerBrand) extracted from the message.

***핵심 지침 (Critical Constraint):
1. ENTITY는 'candidate entities in vocabulary'에 있는 개체명만 **정확히 일치하는 문자열**로 반환해야 합니다.
2. 'entities in message'의 개체명(예: 올리브영)이 'candidate entities in vocabulary'의 개체명(예: 올리브영_올리브영)과 **부분 일치**하면 해당 vocabulary 개체를 선택하세요.
3. 확신이 없으면 제외하세요 (When in doubt, exclude).***

Guidelines:
1. **핵심 오퍼링만 선택**: 메시지가 홍보하는 핵심 상품/서비스/요금제/매장/캠페인과 직접 관련된 vocabulary 개체만 선택
2. **PartnerBrand 매칭**: 'entities in message'에 제휴 브랜드(예: 올리브영, 스타벅스)가 있고, vocabulary에 해당 브랜드를 포함하는 개체가 있으면 선택
3. **Product/RatePlan 매칭**: 상품/요금제 개체가 vocabulary에 있으면 선택
4. **제외 대상**: 할인 금액/비율 단독, 고객센터, URL, 사은품 브랜드명, 일반 기술 용어(5G, LTE)는 선택하지 마세요"""
    else:
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
        context_guideline = """
5. **DAG Context 활용**: 'DAG Context'의 사용자 행동 경로를 분석하여 핵심 오퍼링을 식별하세요.
   - **PROMOTES 대상 = 핵심 오퍼링**: DAG에서 (Root:Action) -[구매/가입/사용]-> (Core:...) 관계의 Core가 핵심 오퍼링
   - **OFFERS 대상 = 제외**: DAG에서 최종 Value 노드(캐시백, 기프티콘, 할인)는 혜택이므로 제외
   - **이벤트 참여 수단 제외**: 퀴즈 주제(예: '아이폰')가 아닌 최종 혜택(예: '올리브영 기프트 카드')과 관련된 개체를 식별
6. **Vocabulary 매칭 가이드**:
   - **복합 V-domain 아이템**: vocabulary의 "올리브영_올리브영", "투썸플레이스 20% 할인" 등은 'entities in message'의 "올리브영", "투썸플레이스"와 **부분 일치**하면 선택
   - **특정 모델만 선택**: 메시지에 언급된 정확한 모델명과 일치하는 vocabulary 개체만 선택. 메시지에 없는 파생 모델/구세대 모델은 선택하지 마세요
   - **"아이폰 신제품" 등 포괄적 표현**: 메시지가 특정 모델을 명시하지 않으면 현행 세대 모델(예: iPhone 17 시리즈)을 선택하되, 구세대 모델은 제외
7. **Anti-noise 규칙**:
   - 일반 카테고리 단어 하나만 일치한다고 vocabulary 개체를 선택하지 마세요 (예: "갤럭시 폴더블" → vocabulary의 모든 갤럭시 폴더블 모델을 선택하지 말 것)
   - 할인 금액/비율 단독(예: "20% 할인", "900원", "최대 2만 원 할인")은 vocabulary에서 선택하지 마세요
   - 행동/설명 구절(예: "매장 방문", "쓰던 아이폰 반납")은 제외"""
    elif context_keyword == 'PAIRING':
        context_guideline = f"""
5. Refer to the '{context_keyword} Context' which maps each offering to its primary benefit. 이를 **사용자의 최종 획득 대상인 핵심 혜택(Primary Benefit)**을 구별하는 데 사용하십시오. (e.g., 가입 대상이 아닌, 최종 혜택인 '캐시백'이나 '기프티콘'과 관련된 개체를 식별)"""
    elif context_keyword == 'ONT':
        context_guideline = """
5. **Ontology Context 활용**: 'ONT Context'에 제공된 Entities, Relationships, DAG를 참고하세요.
   - **Entities**: 각 엔티티의 온톨로지 타입 - EntityName(Type) 형식
   - **Relationships**: 엔티티 간 관계 - Source -[TYPE]-> Target 형식
     - PROMOTES: 캠페인/매장이 상품을 프로모션
     - OFFERS: 캠페인/상품이 혜택을 제공
     - REQUIRES: 참여에 필요한 조건
     - PROVIDES: 제휴사가 혜택의 실제 제공자
   - **DAG**: 사용자 행동 경로 - (Entity:Action) -[Edge]-> (Entity:Action)

   ***ONT 모드 타입별 선택 기준:***
   | 타입 | 포함 여부 | 근거 |
   |------|----------|------|
   | Product, Subscription, RatePlan | **포함** | 핵심 오퍼링 (PROMOTES 타겟) |
   | Store | 제외 | 매장/채널 브랜드 (T다이렉트샵, T월드 등) |
   | Campaign, Event | 제외 | 특정 캠페인/이벤트명 (T멤버십 위크 등) |
   | ContentOffer, WiredService | **포함** | 콘텐츠/유선 서비스 |
   | PartnerBrand | **포함** | 제휴 브랜드 (올리브영, 스타벅스 등) |
   | Benefit | 제외 | 혜택 설명이지 상품명이 아님 (할인, 캐시백 등) |
   | Channel, Segment, Contract, MembershipTier | 제외 | 부가 정보 |

   **중요**: ONT 모드에서는 메시지가 홍보하는 핵심 상품/서비스/요금제/파트너브랜드와 직접 관련된 vocabulary 개체만 선택하세요. 확신이 없으면 제외하세요."""
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

(Chain-of-Thought) - 개체명 추출 과정:
1. 광고/안내 메시지 분류: 첨부된 텍스트는 SK텔레콤의 다양한 광고 및 안내 메시지들을 포함하고 있다.
2. 핵심 개체 정의: 개체명은 광고의 주제가 되거나, 사용자 행동의 중심이 되는 고유 명사들로 정의한다. (예: 특정 App, Device, Event, Store, Plan 등)
3. 추출 및 정제: 메시지 전체를 스캔하며 광고의 핵심 주제에 해당하는 개체명을 원문 그대로 추출하고, 중복을 제거하여 최종 목록을 구성한다.

출력 결과 형식:
1. **ENTITY**: A list of entities separated by commas.
"""

HYBRID_DAG_EXTRACTION_PROMPT = """
Analyze the SKT advertisement to extract **Core Offering Entities** and **User Action Paths**.
Output two distinct sections:
1. **ENTITY**: A list of Core Offering entities (product/service names only).
2. **DAG**: A structured graph representing the flow from Root to Benefit.

## Crucial Language Rule
* **DO NOT TRANSLATE:** Extract entities **exactly as they appear** in the source text.
* **Preserve Original Script:** If the text says "아이폰 17", output "아이폰 17" (NOT "iPhone 17"). If it says "T Day", output "T Day".

## Entity Type Categories
Classify each entity into one of these types to guide extraction:

- **Product (단말기)**: Hardware devices and models.
  예: "아이폰 17 Pro", "갤럭시 Z 플립7", "갤럭시 S25 울트라", "에어팟"
- **RatePlan (요금제)**: Telecom rate plans.
  예: "5GX 프리미엄", "컴팩트 요금제", "다이렉트5G 69", "프라임플러스 요금제"
- **Subscription (구독)**: Monthly subscription services.
  예: "T 우주패스 올리브영&스타벅스&이마트24", "보이스피싱 보험", "V컬러링"
- **Store (매장)**: Physical stores and branches.
  예: "새샘대리점 역곡점", "에스알대리점 지행역점"
- **PartnerBrand (제휴 브랜드)**: Partner brands when they are the subject of promotion.
  예: "올리브영", "투썸플레이스", "스타벅스" (when promotion centers on the brand, not just a gift coupon)
- **WiredService (유선)**: Wired/broadband services.
  예: "기가인터넷(1G)", "B tv", "IPTV"
- **Campaign (캠페인)**: Marketing campaigns and events.
  예: "T Day", "0 day", "special T", "Lucky 1717 이벤트"

## Part 1: Root Node Selection Hierarchy (Extract ALL Distinct Roots)
Identify logical starting points based on this priority. If multiple independent offers exist, extract all.

1.  **Store (Highest):** Specific branch names.
    * *Match:* "새샘대리점 역곡점", "백색대리점 수성직영점"
2.  **RatePlan / WiredService:** Rate plans, Internet/IPTV.
    * *Match:* "5GX 프라임 요금제", "로밍 baro 요금제", "기가인터넷(1G)"
3.  **Subscription / Campaign:** Monthly subscriptions or specific campaigns.
    * *Match:* "T 우주패스 올리브영&스타벅스&이마트24", "T Day", "0 day"
4.  **PartnerBrand:** Partner brands when the promotion centers on them.
    * *Match:* "올리브영" (in "올리브영 20% 할인 쿠폰 제공")
5.  **Product (Hardware):** Device launches.
    * *Match:* "아이폰 17 Pro", "갤럭시 Z 플립7"

## Part 2: DAG Construction Rules
Construct a Directed Acyclic Graph (DAG) for each identified Root Node.
* **Format:** `(Node:Action) -[Edge]-> (Node:Action)`
* **Nodes:**
    * **Root:** The entry point identified above (Original Text).
    * **Core:** The product/service being used or bought (Original Text).
    * **Value:** The final reward or benefit (Original Text).
* **Edges:** Use concise Korean action verbs: 가입, 구매, 사용, 획득, 제공, 지급, 방문, 다운로드, 신청
* **Logic:** Represent the shortest path from the Root action to the Final Benefit.

## Specificity Rule
* Extract **specific model/plan names**, not generic categories: "갤럭시 Z 플립7" ✅, "갤럭시 폴더블" ❌, "5GX 프리미엄" ✅, "5G 요금제" ❌
* When only a generic term exists in the message (e.g., "아이폰 신제품"), extract that generic term as-is — do NOT invent specific model names.

## Strict Exclusions (DO NOT extract as entities)
* **Standalone discount amounts/rates**: "20% 할인", "최대 22만 원 캐시백", "900원", "50% 할인 쿠폰", "최대 2만 원 할인"
* **Generic tech terms alone**: "5G", "LTE", "IPTV" (but "5GX 프리미엄", "B tv" are OK as named services)
* **Gift brand names**: "[사죠영]", "[크레앙]", "[프리디]" (gift manufacturer names in brackets)
* **Customer service / URLs**: "SKT 고객센터(1558)", "고객센터(114)", URLs, "무료 수신거부 1504"
* **Navigational labels**: "바로 가기", "자세히 보기", "링크", "Shortcut"
* **Action/benefit descriptions** (not product names): "매장 방문", "쓰던 아이폰 반납", "제휴 신용카드", "액정 보호 필름 무료 교체", "180개 고화질 TV 채널"
* **Generic partners** unless they are the main promotion subject: "스타벅스 기프티콘" → extract "스타벅스" only if the ad promotes 스타벅스 itself, not if it's just a gift

## Output Format: Do not use Markdown formatting. Use plain text.
ENTITY: <comma-separated list of Core Offering entities only, in original text>
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
