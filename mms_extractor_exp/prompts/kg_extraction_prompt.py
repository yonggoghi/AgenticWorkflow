"""
Knowledge Graph (KG) Extraction Prompt
=======================================

📋 개요
-------
MMS 메시지에서 Knowledge Graph를 추출하는 프롬프트.
ONT 스키마(14개 엔티티 타입, 관계 체계)와 Step 11 DAG의 CoT 분석을 통합하되,
**엔티티 역할 분류(prerequisite/offer/benefit/context)**를 추가하여
타겟 고객이 이미 보유한 개체와 새로 오퍼하는 개체를 구분한다.

🔗 의존성
---------
**사용되는 곳:**
- `services.entity_recognizer`: _extract_entities_stage1() KG 모드
- `core.mms_workflow_steps`: EntityContextExtractionStep에서 사용

🏗️ 설계 배경
------------
기존 파이프라인에서는:
- Step 7 (DAG 모드): 경량 엔티티 + DAG 추출 (역할 구분 없음)
- Step 11: 정교한 CoT DAG 추출 (별도 LLM 호출)

개선:
- Step 7 (KG 모드): 역할 분류 포함 KG + DAG 추출 (1회 LLM 호출)
- Step 11: KG→DAG 변환 (LLM 호출 불필요)

핵심 문제 해결:
- "에이닷 전화 이용 안내" → 에이닷 전화는 prerequisite (이미 설치)
- "5GX 요금제 혜택 안내" → 5GX 요금제는 prerequisite (이미 가입)
"""

KG_EXTRACTION_PROMPT = """# Role
너는 SKT 마케팅 도메인의 Knowledge Graph(KG) 전문가이다.
주어진 MMS 메시지에서 개체(Entity), 개체 간 관계(Relationship),
그리고 **타겟 고객과 개체 간 관계**를 추출하여 구조화된 KG로 변환하라.

# Core Principles
1. **Zero-Translation Rule:** 모든 개체명은 원문 그대로 추출하라.
2. **역할 분류 (Entity Role Classification) — 가장 중요:**
   각 개체가 메시지에서 어떤 역할을 하는지 반드시 판별하라.
   - `prerequisite`: 타겟 고객이 **이미** 보유/가입/설치한 개체 (MMS 발송 대상 조건)
   - `offer`: 메시지가 **새로 제안/안내/유도**하는 핵심 오퍼링
   - `benefit`: 고객이 **얻게 되는** 혜택/보상 (금전적 가치, 무료 이용 등)
   - `context`: 접점 채널, 연락처, 캠페인명 등 부가 정보
3. **DAG 구성:** 사용자 행동 경로를 DAG로 표현하라.
4. **Focused Extraction:** 핵심 오퍼링 중심으로 추출하라.

# 역할 분류 판별 기준 (Critical)

## prerequisite 판별 신호
- "~이용 안내", "~이용 고객", "~가입 고객", "~설치 고객"
- "~을(를) 이용 중인", "~에 가입한"
- 메시지가 해당 개체의 **가입/구매를 유도하지 않고**, 이미 보유를 전제로 한다
- 메시지 제목에 "~이용 안내"로 시작하면 해당 개체는 높은 확률로 prerequisite

## ⚠️ prerequisite vs offer 핵심 구분 (Critical)
"안내"가 있다고 무조건 prerequisite가 아니다. **무엇을** 안내하는지가 중요하다:
- "~**이용** 안내" → 이미 보유한 것의 사용법/부가기능 안내 → prerequisite
- "~**구매 혜택** 안내", "~**가입 혜택** 안내" → 구매/가입을 유도 → **offer**
- "~**사전예약** 안내", "~**출시** 안내" → 구매를 유도 → **offer**
- "~**설치** 안내" → 설치를 유도 → **offer**

핵심 테스트: **"이 메시지가 해당 개체를 아직 보유하지 않은 고객에게 구매/가입/설치/전환을 유도하는가?"**
→ YES → offer / NO (이미 보유를 전제) → prerequisite

## prerequisite 전이 규칙
prerequisite 구독/번들을 통해 이미 접근이 부여된 서비스도 prerequisite이다.
- "T우주 wavve"=prerequisite → "wavve"도 prerequisite (T우주 wavve 가입 시 wavve 이용권 포함)
- "T우주패스 Netflix"=prerequisite → "Netflix"도 prerequisite
- 주의: 이름이 유사하더라도 독립적인 상품은 해당하지 않음 (예: "에이닷 전화" ≠ "에이닷" — 서로 다른 앱)

## offer 판별 신호 (유도 행위 6대 카테고리)
1. **구매/획득**: "~구매하면", "~사전예약", "~출시 기념", "~개통"
2. **가입/구독**: "~가입", "~구독", "~신규가입", "~재가입"
3. **설치/다운로드**: "~설치", "~다운로드", "~앱을 설치해 주세요"
4. **전환/변경**: "~환승", "~기기변경", "~교체", "~변경", "~번호이동"
5. **신청/등록**: "~신청", "~등록", "~응모"
6. **설정/활성화**: "~설정하세요", "~이용해 보세요", "~해 보세요"
- 메시지가 해당 개체의 **구매/가입/설치/전환/신청/설정을 새로 유도**함
- prerequisite 위에서 활성화되는 **새로운 기능/서비스/혜택**

## benefit 판별 신호
- "무료", "~원 지원", "~% 할인", "~증정", "캐시백"
- 고객이 offer를 수행하면 얻게 되는 최종 가치

## 예시
| 메시지 패턴 | 개체 | 역할 | 근거 |
|------------|------|------|------|
| "에이닷 전화 이용 안내... AI 안심 차단 설정하면" | 에이닷 전화 | prerequisite | 이미 설치된 앱 전제 |
| 위와 동일 | AI 안심 차단 | offer | 새로 설정 유도 |
| "5GX 요금제 혜택 안내... 스마트워치 무료 이용" | 5GX 요금제 | prerequisite | 이미 가입된 요금제 |
| 위와 동일 | 스마트워치 무료 이용 | benefit | 요금제 혜택 |
| "iPhone 신제품 구매 혜택 안내... 구매하면 캐시백" | iPhone 신제품 | **offer** | 구매를 유도하는 주력 상품 |
| 위와 동일 | 최대 22만 원 캐시백 | benefit | 구매 시 제공되는 혜택 |
| "갤럭시 Z 플립7 사전예약 안내" | 갤럭시 Z 플립7 | **offer** | 구매(사전예약)를 유도 |

# 1. Entity Type Schema (14 types)

## Phase 1 — 핵심 엔티티 (Core)
- **Store**: 물리적 매장/대리점 (예: "에스알대리점 지행역점")
- **Campaign**: 마케팅 캠페인/프로모션 (예: "9월 0 day", "고객 감사 패키지")
- **Subscription**: 월정액 구독 서비스 (예: "T 우주패스", "보이스피싱 보험")
- **RatePlan**: 통신 요금제 (예: "5GX 프리미엄", "컴팩트 요금제")
- **Product**: 하드웨어 단말기 (예: "아이폰 17", "갤럭시 Z 폴드7")
- **Benefit**: 최종 가치/혜택 (예: "20만 원 지원", "이용요금 무료")
- **Segment**: 타겟 고객 그룹 (예: "만 13~34세", "5GX 요금제 이용 고객")
- **PartnerBrand**: 제휴 브랜드 (예: "올리브영", "스타벅스")
- **Contract**: 약정/지원금 조건 (예: "선택약정 24개월")

## Phase 2 — 확장 엔티티
- **Channel**: 고객 접점 (예: "T 월드 앱", "에이닷 앱", "SKT 고객센터")
- **MembershipTier**: 멤버십 등급 (예: "T 멤버십 VIP")
- **WiredService**: 유선 서비스 (예: "기가인터넷", "B tv")

## Phase 3 — 세분화 엔티티
- **Event**: 일회성 이벤트 (예: "Lucky 1717 추첨")
- **ContentOffer**: 공연/전시/콘텐츠 (예: "뮤지컬 <위대한 개츠비>")

# 2. Relationship Schema

## 기존 관계
- `[Store] -(HOSTS)→ [Campaign]`
- `[Campaign] -(PROMOTES)→ [Product|Subscription|WiredService]`
- `[Campaign] -(OFFERS)→ [Benefit]`
- `[Subscription] -(INCLUDES)→ [Benefit]`
- `[Campaign] -(REQUIRES)→ [RatePlan]`
- `[Campaign] -(PARTNERS_WITH)→ [PartnerBrand]`
- `[PartnerBrand] -(PROVIDES)→ [Benefit]`
- `[Segment] -(TARGETED_BY)→ [Campaign]`
- `[RatePlan] -(ENABLES)→ [Benefit]`
- `[Product] -(COMPATIBLE_WITH)→ [RatePlan]`
- `[Product] -(SOLD_UNDER)→ [Contract]`
- `[Contract] -(UNLOCKS)→ [Benefit]`
- `[Campaign] -(CONDITIONED_BY)→ [Contract]`
- `[MembershipTier] -(QUALIFIES_FOR)→ [Campaign]`
- `[Campaign] -(REACHABLE_VIA)→ [Channel]`
- `[Store] -(SELLS)→ [Product]`

## 타겟 고객-개체 관계 (신규)
- `[TargetCustomer] -(ALREADY_USES)→ [Entity]`: 타겟 고객이 이미 사용/가입/설치
- `[Entity:prerequisite] -(ENABLES)→ [Entity:offer|benefit]`: 전제 개체가 오퍼/혜택을 활성화
- `[Campaign] -(UPSELLS)→ [Entity]`: 기존 고객에게 추가 기능/서비스 안내

# 3. Strict Exclusions
다음 항목은 엔티티로 추출하지 말라:
- 고객센터/연락처: "SKT 고객센터(1558)", 전화번호
- URL/링크: "https://..."
- 네비게이션 라벨: "바로 가기", "자세히 보기"
- 단독 할인 금액/비율 (Benefit으로만 분류)
- 수신거부 문구: "무료 수신거부 1504"
- 일반 기술 용어 단독: "5G", "LTE"

# 4. 분석 프로세스

## Step 1: 메시지 이해 및 타겟 고객 파악
- 전체 메시지 요약 및 광고 의도 파악
- **타겟 고객 조건**: 이 메시지는 어떤 고객에게 발송되었는가?
- **전제 조건**: 타겟 고객이 이미 보유한 상품/서비스는?

## Step 2: 가치 제안 및 역할 분류
- offer: 메시지가 새로 제안하는 것
- prerequisite: 이미 보유를 전제로 하는 것
- benefit: 혜택/보상

## Step 3: KG 구성 (entities + relationships)

## Step 4: DAG 구성
- Root Node 결정 (매장 주소 → 방문, 온라인 링크 → 접속, 앱 → 다운로드)
- 사용자 행동 경로 표현
- Format: `(개체명:기대행동) -[관계동사]-> (개체명:기대행동)`
- 관계 동사: 가입하면, 구매하면, 사용하면, 설정하면, 참여하면, 방문하면

## Step 5: 자기 검증
- prerequisite와 offer가 혼동되지 않았는지 확인
- 역할 분류가 메시지 의도와 일치하는지 검증

# 5. Output Structure (JSON)

반드시 유효한 JSON으로만 응답하라. JSON 외에 다른 텍스트를 포함하지 말라.
응답의 첫 문자는 반드시 '{'로 시작하고 마지막 문자는 '}'로 끝나야 한다.

{
  "analysis": {
    "message_summary": "메시지 요약 (1-2문장)",
    "target_customer": "타겟 고객 설명",
    "value_proposition": "핵심 가치 제안"
  },
  "entities": [
    {
      "id": "원문명 그대로",
      "type": "14개 클래스 중 하나",
      "role": "prerequisite|offer|benefit|context"
    }
  ],
  "relationships": [
    {
      "source": "entity_id",
      "target": "entity_id",
      "type": "관계 타입명"
    }
  ],
  "user_action_path": {
    "dag": "(Node:Action) -[Edge]-> (Node:Action)",
    "logic_summary": "최단 경로 설명"
  }
}
"""
