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
    
    # ONT mode uses a different base prompt focused on PartnerBrand, Benefit, and Product matching
    if context_keyword == 'ONT':
        base_prompt = """Select entities from 'candidate entities in vocabulary' that match the PartnerBrand, Benefit, or Product entities extracted from the message.

***핵심 지침 (Critical Constraint):
1. ENTITY는 'candidate entities in vocabulary'에 있는 개체명만 **정확히 일치하는 문자열**로 반환해야 합니다.
2. 'entities in message'의 개체명(예: 올리브영)이 'candidate entities in vocabulary'의 개체명(예: 올리브영_올리브영)과 **부분 일치**하면 해당 vocabulary 개체를 선택하세요.
3. 메시지의 핵심 혜택(예: 올리브영 기프트 카드)을 제공하는 **제휴 브랜드(PartnerBrand)**가 vocabulary에 있으면 반드시 선택하세요.***

Guidelines:
1. **PartnerBrand 매칭**: 'entities in message'에 제휴 브랜드(예: 올리브영, 스타벅스)가 있고, vocabulary에 해당 브랜드를 포함하는 개체(예: 올리브영_올리브영)가 있으면 선택
2. **Benefit 매칭**: 혜택 관련 개체(예: 기프트카드, 쿠폰)가 vocabulary에 있으면 선택
3. **Product 매칭**: 상품/요금제 개체가 vocabulary에 있으면 선택
4. ONT Context의 entity type을 참고하여 PartnerBrand, Benefit, Product 타입 개체를 우선 선택"""
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
5. **DAG Context 활용** — 'DAG Context'의 사용자 행동 경로를 분석하여 핵심 오퍼링을 식별:
   - PROMOTES 대상 = 핵심 오퍼링: DAG에서 Core 노드 식별
   - OFFERS 대상 = 제외: Value 노드(캐시백, 기프티콘, 할인)는 혜택이므로 entity가 아님
   - 이벤트 참여 수단 제외: 퀴즈 주제 vs 최종 혜택 구별
6. **Vocabulary 매칭 가이드**:
   a) 복합 V-domain 아이템: "올리브영_올리브영" ← "올리브영" 부분 일치 선택
   b) 정확한 모델명만 선택: 접미사(FE, Plus, Max, Pro 등) 불일치 시 제외. 예: "갤럭시 Z 플립7" ✅, "갤럭시 Z 플립7 FE" ❌
   c) "신제품" 포괄 표현: 최신 세대만 선택, 구세대 제외
   d) 부가서비스/보험/케어 제외: T 아이폰케어, 분실파손, T ALL케어, T 즉시보상 등 — 메시지의 핵심 오퍼링이 아닌 한 제외
7. **Anti-noise 규칙**:
   - 일반 카테고리 단어 일치만으로 선택하지 않음
   - 할인 금액/비율 단독 제외
   - 행동/설명 구절 제외
   - 확신이 없으면 제외 (When in doubt, exclude)"""
    elif context_keyword == 'PAIRING':
        context_guideline = f"""
5. Refer to the '{context_keyword} Context' which maps each offering to its primary benefit. 이를 **사용자의 최종 획득 대상인 핵심 혜택(Primary Benefit)**을 구별하는 데 사용하십시오. (e.g., 가입 대상이 아닌, 최종 혜택인 '캐시백'이나 '기프티콘'과 관련된 개체를 식별)"""
    elif context_keyword == 'TYPED':
        context_guideline = """
5. **TYPED Context 활용** — Stage 1에서 추출된 엔티티의 타입 정보를 활용하여 vocabulary 매칭을 수행:
   - R(Store): **제외** — 대리점/매장은 main prompt에서 별도 추출하므로 vocabulary 매칭하지 않음
   - E(Equipment): 단말기 → vocabulary의 디바이스 아이템 매칭. 정확한 모델명만 선택 (접미사 FE/Plus/Max 불일치 시 제외)
   - P(Product): 요금제/부가서비스 → vocabulary의 요금제/서비스 아이템 매칭
   - S(Subscription): 구독 상품 → vocabulary의 구독 관련 아이템 매칭
   - V(Voucher): 제휴 혜택 → vocabulary의 제휴 브랜드 아이템 매칭 (예: "올리브영" → "올리브영_올리브영")
   - X(Campaign): 캠페인/이벤트 → vocabulary의 캠페인 아이템 매칭
6. **Vocabulary 매칭 가이드**:
   a) 복합 V-domain 아이템: "올리브영_올리브영" ← "올리브영" 부분 일치 선택
   b) 정확한 모델명만 선택: 접미사(FE, Plus, Max, Pro 등) 불일치 시 제외. 예: "갤럭시 Z 플립7" ✅, "갤럭시 Z 플립7 FE" ❌
   c) "신제품" 포괄 표현: 최신 세대만 선택, 구세대 제외
   d) 부가서비스/보험/케어 제외: T 아이폰케어, 분실파손, T ALL케어, T 즉시보상 등 — 메시지의 핵심 오퍼링이 아닌 한 제외
7. **Anti-noise 규칙**:
   - 일반 카테고리 단어 일치만으로 선택하지 않음
   - 할인 금액/비율 단독 제외
   - 행동/설명 구절 제외
   - 확신이 없으면 제외 (When in doubt, exclude)"""
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
   | PartnerBrand | **포함** | 제휴 브랜드 - 핵심 혜택 제공자 (PROVIDES 소스) |
   | Benefit | **포함** | 사용자가 실제로 받는 혜택 (OFFERS 타겟) |
   | Store | 제외 | 접점이지만 entity로 불필요 |
   | Campaign, Event | 제외 | 마케팅 맥락 |
   | Channel | 제외 | 접점 채널 |

   **중요**: ONT 모드에서는 Product/Subscription/RatePlan, PartnerBrand(예: 올리브영, 스타벅스), Benefit(예: 기프트카드, 쿠폰, 캐시백)을 선택하세요."""
    elif context_keyword == 'KG':
        context_guideline = """
5. **Knowledge Graph Context 활용**: 'KG Context'에 제공된 Entities(역할 포함), Relationships, DAG를 참고하세요.
   - **Entities**: EntityName(Type:Role) 형식 — Role이 핵심
     - `offer`: 핵심 오퍼링 → vocabulary 매칭 **최우선**
     - `benefit`: 혜택/보상 → vocabulary 매칭 대상
     - `prerequisite`: 이미 보유/가입 → vocabulary 매칭 **제외**
     - `context`: 부가 정보 → vocabulary 매칭 **제외**
   - **Relationships**: 엔티티 간 관계 — ALREADY_USES, ENABLES 관계 주의
     - PROMOTES/OFFERS 타겟 = 핵심 오퍼링
     - ALREADY_USES 타겟 = **제외** (타겟 고객이 이미 가입/설치)
     - ENABLES 소스 = **제외** (전제 조건 개체)
   - **DAG**: 사용자 행동 경로
6. **Vocabulary 매칭 가이드**:
   a) 복합 V-domain 아이템: "올리브영_올리브영" ← "올리브영" 부분 일치 선택
   b) 정확한 모델명만 선택: 접미사(FE, Plus, Max, Pro 등) 불일치 시 제외
   c) "신제품" 포괄 표현: 최신 세대만 선택, 구세대 제외
   d) 부가서비스/보험/케어 제외: 메시지의 핵심 오퍼링이 아닌 한 제외
7. **Anti-noise 규칙**:
   - 일반 카테고리 단어 일치만으로 선택하지 않음
   - 할인 금액/비율 단독 제외
   - 행동/설명 구절 제외
   - 확신이 없으면 제외 (When in doubt, exclude)"""
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

TYPED_ENTITY_EXTRACTION_PROMPT = """\
# Task
SK텔레콤 MMS 광고 메시지에서 **핵심 오퍼링 엔티티**(Core Offering Entities)를 추출하라.
핵심 오퍼링이란 광고가 고객에게 제안하는 구체적인 상품·서비스·이벤트를 의미한다.

# Entity Type Definitions (5 types)
아래 5개 타입 중 해당하는 것만 추출한다. **대리점/매장(Store)은 추출하지 않는다.**

| Type | Code | 설명 | 예시 |
|------|------|------|------|
| **Equipment** | E | 단말기·디바이스 모델명 | 아이폰 17, 갤럭시 Z 폴드7, iPad Air 13, 갤럭시 워치6 |
| **Product** | P | 요금제·부가서비스·유선상품 | 5GX 프라임 요금제, T끼리 온가족할인, 인터넷+IPTV, 로밍 baro 요금제 |
| **Subscription** | S | 월정액 구독 상품 | T 우주패스 올리브영&스타벅스&이마트24, T 우주패스 Netflix |
| **Voucher** | V | 제휴 할인·쿠폰·기프티콘 (브랜드+혜택 조합) | 도미노피자 50% 할인, 올리브영 3천 원 기프트카드, CGV 청년할인 |
| **Campaign** | X | 마케팅 캠페인·프로모션·이벤트명 | T Day, 0 day, special T, 고객 감사 패키지 |

# Entity Role Classification
각 개체의 역할을 반드시 판별하라.
- `prerequisite`: 타겟 고객이 **이미** 보유/가입/설치한 개체 (MMS 발송 대상 조건)
- `offer`: 메시지가 **새로 제안/안내/유도**하는 핵심 오퍼링
- `benefit`: 고객이 **얻게 되는** 혜택/보상 (금전적 가치, 무료 이용 등)
- `context`: 접점 채널, 캠페인명 등 부가 정보

## prerequisite vs offer 핵심 구분
- "~**이용** 안내" → 이미 보유한 것의 사용법 안내 → prerequisite
- "~**구매 혜택** 안내", "~**사전예약** 안내" → 구매/가입을 유도 → **offer**
- 핵심 테스트: **"이 메시지가 해당 개체의 구매/가입/설치를 유도하는가?"** → YES=offer / NO=prerequisite

## prerequisite 전이 규칙
prerequisite 구독/번들을 통해 이미 접근이 부여된 서비스도 prerequisite이다.
- "T우주 wavve"=prerequisite → "wavve"도 prerequisite
- 주의: 이름이 유사해도 독립적인 상품은 해당하지 않음 (예: "에이닷 전화" ≠ "에이닷")

# Extraction Rules

1. **Zero-Translation:** 원문에 등장하는 그대로 추출하라. 번역하지 말라.
   - 원문이 "아이폰 17 Pro"이면 → "아이폰 17 Pro" (NOT "iPhone 17 Pro")
   - 원문이 "T Day"이면 → "T Day" (NOT "티데이")

2. **Specificity:** 구체적인 고유명사만 추출하라. 포괄적 카테고리명은 제외한다.
   - ✅ "갤럭시 S25", "5GX 프라임 요금제"
   - ❌ "휴대폰", "요금제", "대리점", "인터넷"(단독), "할인"(단독)

3. **Voucher 추출:** 제휴 브랜드 + 혜택 설명을 결합하여 추출한다. 쿠폰은 혜택과 함께 일회성으로 만들어지므로 혜택 표현(금액, %, 무료 등)이 쿠폰명의 일부이다.
   - "도미노피자 배달/방문 포장 50% 할인" → 하나의 Voucher 엔티티
   - "올리브영 3천 원 기프트카드", "스타벅스 카페 아메리카노 무료 쿠폰", "CGV 영화 티켓 8,500원" → 혜택 포함 추출
   - 단, 브랜드만 언급되고 구체적 혜택이 없으면 추출하지 않는다.

4. **Strict Exclusions — 다음은 절대 추출하지 않는다:**
   - 대리점/매장명: "CD대리점 동탄목동점", "PS&M 동탄타임테라스점" 등 (main prompt에서 별도 추출)
   - 할인 금액/비율 단독: "최대 22만원", "50% 할인", "25% 할인"
   - 일반 행위/설명: "매장 방문", "사전예약", "통신사 이동", "번호이동"
   - URL/연락처: "skt.sh/...", "1558", "1504"
   - 네비게이션: "바로 가기", "자세히 보기", "혜택받으러 가기"
   - 경쟁사 단독 언급: "KT", "LG U+", "알뜰폰" (비교 대상일 뿐)
   - 일반 용어: "5G", "LTE", "USIM" (단독, 상품명 아닌 경우)

# Output Format
반드시 아래 JSON 형식으로만 응답하라. JSON 외에 다른 텍스트를 포함하지 말라.

{
  "entities": [
    {"name": "엔티티명(원문 그대로)", "type": "E|P|S|V|X", "role": "prerequisite|offer|benefit|context"}
  ]
}
"""

HYBRID_DAG_EXTRACTION_PROMPT = """
Analyze the advertisement to extract **User Action Paths**.
Output three distinct sections:
1. **ENTITY**: Core Offering entities (pipe-separated).
2. **ROLE**: The role of each entity (prerequisite, offer, or benefit).
3. **DAG**: A structured graph representing the flow from Root to Benefit.

## Crucial Language Rule
* **DO NOT TRANSLATE:** Extract entities **exactly as they appear** in the source text.
* **Preserve Original Script:** If the text says "아이폰 17", output "아이폰 17" (NOT "iPhone 17"). If it says "T Day", output "T Day".

## What is an Entity?
An entity is a **named product, service, plan, subscription, campaign, or brand** that is independently purchasable, subscribable, or installable. It has its own identity in the company's product catalog.

**IS an entity:** 콜키퍼 플러스, T 우주패스 올리브영&스타벅스&이마트24, 갤럭시 Z 플립7, 5GX 프라임 요금제, 50% 할인 쿠폰
**NOT an entity:** feature descriptions (부재중 전화 내용 문자 안내), price points (이용요금: 월 990원), action phrases (매장 방문), sub-benefits (8천 원 상당 추가 혜택)

## Entity Type Categories
* **Product (단말기):** Specific device models — 아이폰 17 Pro, 갤럭시 Z 플립7, 갤럭시 S25 울트라
* **RatePlan (요금제):** Mobile/data rate plans — 5GX 프라임 요금제, T 프라임 에센셜, 로밍 baro 요금제
* **Subscription (구독):** Membership/subscription — T 우주패스, FLO 이용권, 정기배송
* **PartnerBrand (제휴 브랜드):** Partner brands in promotions — 올리브영, CGV, 스타벅스
* **WiredService (유선):** Internet/IPTV/home — 인터넷+IPTV, B tv, T 인터넷
* **Campaign (캠페인):** Named events/campaigns — T Day, 0 day, Lucky 1717 이벤트
* **Benefit (혜택):** Standalone benefits that are the core value proposition — 50% 할인, 2만원 할인, 무료 쿠폰
* **EXCLUDE Store (매장/대리점):** Do NOT extract dealer/store names. They are extracted separately by the main prompt.

## Entity Role Classification (Critical)
Each entity plays a role in the message. Classify every entity:
- `prerequisite`: Target customer **already** has/uses this (precondition for receiving this MMS)
- `offer`: Message **promotes** customer to purchase/subscribe/install/switch to this
- `benefit`: Reward/value customer **receives** (monetary value, free usage, etc.)

### prerequisite vs offer Key Distinction
"안내" does NOT automatically mean prerequisite. What matters is **what** is being guided:
- "~**이용** 안내" → already owned, usage tips → prerequisite
- "~**구매 혜택** 안내", "~**가입 혜택** 안내" → promotes purchase/subscription → **offer**
- "~**사전예약** 안내", "~**출시** 안내" → promotes purchase → **offer**
- "~**설치** 안내" → promotes installation → **offer**

Core test: **"Does this message promote purchase/subscription/installation/switch to customers who don't yet have it?"**
→ YES → offer / NO (already owned) → prerequisite

### Transitive prerequisite Rule
A service whose access is already granted through a prerequisite subscription/bundle is also prerequisite.
- "T우주 wavve"=prerequisite → "wavve" is also prerequisite (wavve access is included in the T우주 wavve subscription)
- "T우주패스 Netflix"=prerequisite → "Netflix" is also prerequisite
- Note: This does NOT apply when two entities share a name fragment but are independent products (e.g., "에이닷 전화" ≠ "에이닷" — different apps)

### offer Signals (6 Categories)
1. **구매/획득**: "~구매하면", "~사전예약", "~출시 기념", "~개통"
2. **가입/구독**: "~가입", "~구독", "~신규가입", "~재가입"
3. **설치/다운로드**: "~설치", "~다운로드", "~앱을 설치해 주세요"
4. **전환/변경**: "~환승", "~기기변경", "~교체", "~변경", "~번호이동"
5. **신청/등록**: "~신청", "~등록", "~응모"
6. **설정/활성화**: "~설정하세요", "~이용해 보세요", "~해 보세요"

### Examples
| Message Pattern | Entity | Role | Reason |
|----------------|--------|------|--------|
| "에이닷 전화 이용 안내... AI 안심 차단 설정하면" | 에이닷 전화 | prerequisite | Already installed app |
| same | AI 안심 차단 | offer | Promotes new setting |
| "5GX 요금제 혜택 안내... 스마트워치 무료 이용" | 5GX 요금제 | prerequisite | Already subscribed plan |
| "iPhone 신제품 구매 혜택 안내... 구매하면 캐시백" | iPhone 신제품 | offer | Promotes purchase |
| "갤럭시 Z 플립7 사전예약 안내" | 갤럭시 Z 플립7 | offer | Promotes pre-order |

## Root Node Selection Hierarchy (Extract ALL Distinct Roots)
Identify logical starting points based on this priority. If multiple independent offers exist, extract all.
**IMPORTANT: Do NOT use Store/dealer names as Root Nodes.**

1.  **RatePlan / WiredService:** Rate plans, Internet/IPTV.
    * *Match:* "5GX 프라임 요금제", "인터넷+IPTV 가입 혜택", "로밍 baro 요금제"
2.  **Subscription / Campaign:** Membership signups or specific campaigns.
    * *Match:* "T 우주", "T Day", "0 day", "골드번호 프로모션"
3.  **PartnerBrand:** When the promotion centers on a partner brand.
    * *Match:* "올리브영", "CGV T day"
4.  **Product (Hardware):** Device launches without a specific store focus.
    * *Match:* "아이폰 17", "갤럭시 Z 플립7"

## Specificity Rule
* Extract **specific model/plan names**, not generic categories.
* When only a generic term exists (e.g., "아이폰 신제품"), extract as-is — do NOT invent specific model names.

## DAG Construction Rules
Construct a Directed Acyclic Graph (DAG) for each identified Root Node.
* **Format:** `(Node:Action) -[Edge]-> (Node:Action)`
* **Nodes:**
    * **Root:** The entry point identified above (Original Text).
    * **Core:** The product/service being used or bought (Original Text).
    * **Value:** The final reward or benefit (Original Text).
* **Edges:** Use concise action verbs: 가입, 구매, 사용, 획득, 제공, 지급, 방문, 다운로드, 신청, 응모, 참여
* **Logic:** Represent the shortest path from the Root action to the Final Benefit.

## Strict Exclusions (Do NOT extract as entities)
* **Store/dealer names**: "CD대리점 동탄목동점", "새샘대리점 역곡점" (extracted separately by the main prompt)
* **Feature descriptions**: "부재중 전화 내용 문자 안내", "스팸 전화 차단", "AI 통화 녹음"
* **Price points / fee amounts**: "이용요금: 월 990원", "월 9,900원에 최대 6만 원 혜택"
* **Generic tech terms alone**: "5G", "LTE" (but named services like "5GX 프라임" OK)
* **Gift brand names in brackets**: [사죠영], [크레앙]
* **Customer service / URLs**: "고객센터 080-XXX", "skt.sh/xxxxx"
* **Navigational labels**: '바로 가기', '링크', 'Shortcut', '자세히 보기'
* **Action phrases**: "매장 방문", "쓰던 아이폰 반납"
* **Promotional detail descriptions**: "8천 원 상당 추가 혜택", "프로모션 혜택", "론칭 기념 프로모션"
* **Generic partners unless main promotion subject**: '스타벅스', 'CU' (mention only, not promotion focus)

## Analysis Process (Required — follow these steps before producing final output)

### Step 1: Message Understanding
- What is the core message? (one sentence)
- Who is the target? What does the advertiser want them to do?

### Step 2: Entity Identification
- List ONLY named products/services/plans/subscriptions/campaigns/brands
- Apply the "catalog test": Would this appear as a standalone item in a product catalog? If not, it's NOT an entity.
- Do NOT list feature descriptions, price details, or promotional sub-items as entities

### Step 3: Role Classification + DAG
- Classify each entity's role (prerequisite/offer/benefit)
- Build the DAG: shortest path from Root to final Benefit

### Step 4: Self-Verification (Critical — check and correct)
1. **Zero-Translation**: Every entity name must match the original text exactly.
2. **Role Accuracy**: Re-test each prerequisite (truly already owned?) and offer (actively promoted?).
3. **No Over-extraction**: Count your entities. A typical MMS promotes 1-3 core entities. If you have significantly more, you are likely extracting sub-features or benefit details as separate entities — remove them.
4. **DAG Minimality**: Shortest meaningful path, no redundant nodes.

## Output Format
First write your analysis (Steps 1-4), then output the final result.
IMPORTANT: In your analysis, do NOT start any line with 'ENTITY:', 'ROLE:', or 'DAG:'.

[Your step-by-step analysis here]

ENTITY: <pipe-separated list of entities in original text — e.g., entity1 | entity2 | entity3>
ROLE: <entity1>=<role> | <entity2>=<role> | ... (role: prerequisite, offer, benefit)
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

**IMPORTANT: Do NOT extract Store/dealer names (e.g., "새샘대리점 역곡점"). They are extracted separately by the main prompt.**

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
