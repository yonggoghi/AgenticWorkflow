"""
KG 프롬프트 응답 시간 테스트

기존 DAG/ONT 프롬프트와 새 KG 프롬프트의 응답 시간을 비교합니다.
"""
import sys
import os
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.llm_factory import LLMFactory

# ─── 테스트 메시지 ───
MESSAGES = [
    {
        "id": "msg_1_에이닷_AI안심차단",
        "text": """(광고)[SKT] 에이닷 전화 이용 안내
고객님, 안녕하세요. 최근 외화 거래를 노린 보이스피싱 범죄가 늘어나고 있어요! 지금 에이닷 전화 앱에서 <AI 안심 차단> 설정하면 AI가 알아서 스팸/피싱 자동 차단해 드려요.
가족, 지인에게도 에이닷 전화 앱 추천하고 보이스피싱 예방해 보세요.
▶  지금 설정하기: https://t-mms.kr/a8k/#74
■ 문의 : 에이닷 고객센터(1670-0075)
나만의 AI 개인비서, 에이닷
무료 수신 거부 1504""",
        "expected_prereq": "에이닷 전화",
        "expected_offer": "AI 안심 차단"
    },
    {
        "id": "msg_2_에이닷_AI안심차단_v2",
        "text": """(광고)[SKT] 에이닷 전화 "AI 안심차단" 안내
고객님, 안녕하세요. 에이닷 앱의 "AI 안심차단"이 보이스피싱/스팸 전화를 AI로 자동 판별하고 차단해 드립니다.
지금 "AI 안심차단"이 설정되어 있는지 확인해 보세요.
▶ "AI 안심차단" 설정 확인하기: https://t-mms.kr/qPf/#74
■ 유의 사항 - 에이닷 앱을 최신 버전으로 업데이트해 주세요.
■ 문의: 에이닷 고객센터(1670-0075)
나만의 AI 개인비서, 에이닷
무료 수신거부 1504""",
        "expected_prereq": "에이닷 전화/에이닷 앱",
        "expected_offer": "AI 안심차단"
    },
    {
        "id": "msg_3_5GX_스마트워치",
        "text": """(광고)[SKT] 스마트워치 무료 이용 안내   고객님, 안녕하세요. 스마트워치를 이제 무료로 이용해 보세요!
휴대폰에 스마트워치를 연결하면 이용요금 무료 혜택을 받을 수 있습니다.   ■ 5GX 요금제 혜택 안내 - 요금제에 따라 스마트워치 1~2회선 이용요금 무료 * 자세한 혜택은 T 월드 매장 또는 SKT 고객센터 문의   ■ 유의 사항 - 휴대폰과 스마트워치 회선의 명의가 같아야 무료 이용이 가능합니다.   ■ 문의: SKT 고객센터(1558, 무료)   SKT와 함께해 주셔서 감사합니다.
무료 수신거부 1504""",
        "expected_prereq": "5GX 요금제",
        "expected_offer": "스마트워치 무료 이용"
    }
]

# ─── 프롬프트 정의 ───

# 기존 DAG 프롬프트 (Step 7)
from prompts.entity_extraction_prompt import HYBRID_DAG_EXTRACTION_PROMPT

# 기존 ONT 프롬프트 (Step 7)
from prompts.ontology_prompt import ONTOLOGY_PROMPT

# 기존 Step 11 DAG 프롬프트
from prompts.dag_extraction_prompt import DAG_EXTRACTION_PROMPT_TEMPLATE

# 새 KG 프롬프트
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
- "~을(를) 이용 중인", "~에 가입한", "~혜택 안내"
- 메시지가 해당 개체의 **가입/구매를 유도하지 않고**, 이미 보유를 전제로 한다
- 메시지 제목에 "~이용 안내"로 시작하면 해당 개체는 높은 확률로 prerequisite

## offer 판별 신호
- "~설정하세요", "~이용해 보세요", "~해 보세요", "~확인해 보세요"
- 메시지가 해당 개체의 **사용/설정/활성화를 새로 유도**함
- prerequisite 위에서 활성화되는 **새로운 기능/서비스/혜택**

## benefit 판별 신호
- "무료", "~원 지원", "~% 할인", "~증정"
- 고객이 offer를 수행하면 얻게 되는 최종 가치

## 예시
| 메시지 패턴 | 개체 | 역할 | 근거 |
|------------|------|------|------|
| "에이닷 전화 이용 안내... AI 안심 차단 설정하면" | 에이닷 전화 | prerequisite | 이미 설치된 앱 전제 |
| 위와 동일 | AI 안심 차단 | offer | 새로 설정 유도 |
| "5GX 요금제 혜택 안내... 스마트워치 무료 이용" | 5GX 요금제 | prerequisite | 이미 가입된 요금제 |
| 위와 동일 | 스마트워치 무료 이용 | benefit | 요금제 혜택 |

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

## 타겟 고객-개체 관계 (신규)
- `[TargetCustomer] -(ALREADY_USES)→ [Entity]`: 타겟 고객이 이미 사용/가입/설치
- `[Entity:prerequisite] -(ENABLES)→ [Entity:offer]`: 전제 개체가 오퍼 개체를 활성화
- `[Campaign] -(UPSELLS)→ [Entity]`: 기존 고객에게 추가 기능/서비스 안내

# 3. Strict Exclusions
- 고객센터/연락처, URL/링크, 네비게이션 라벨
- 단독 할인 금액/비율 (Benefit으로만 분류)
- 수신거부 문구, 일반 기술 용어 단독

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
- Root Node 결정
- 사용자 행동 경로 표현
- Format: `(개체명:기대행동) -[관계동사]-> (개체명:기대행동)`

## Step 5: 자기 검증
- prerequisite와 offer가 혼동되지 않았는지 확인
- 역할 분류가 메시지 의도와 일치하는지 검증

# 5. Output Structure (JSON)

반드시 유효한 JSON으로만 응답하라. JSON 외에 다른 텍스트를 포함하지 말라.

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


def test_prompt(llm_model, prompt_text: str, msg: str, label: str) -> dict:
    """프롬프트를 실행하고 시간 측정"""
    full_prompt = f"{prompt_text}\n\n## message:\n{msg}"

    start = time.time()
    try:
        response = llm_model.invoke(full_prompt).content
        elapsed = time.time() - start
        return {
            "label": label,
            "elapsed": elapsed,
            "response_len": len(response),
            "response": response,
            "error": None
        }
    except Exception as e:
        elapsed = time.time() - start
        return {
            "label": label,
            "elapsed": elapsed,
            "response_len": 0,
            "response": "",
            "error": str(e)
        }


def extract_roles_from_kg(response: str) -> dict:
    """KG JSON 응답에서 역할 분류 추출"""
    try:
        json_str = response.strip()
        if json_str.startswith('```'):
            import re
            json_str = re.sub(r'^```(?:json)?\n?', '', json_str)
            json_str = re.sub(r'\n?```$', '', json_str)
        data = json.loads(json_str)

        roles = {}
        for e in data.get('entities', []):
            eid = e.get('id', '')
            role = e.get('role', 'unknown')
            etype = e.get('type', 'Unknown')
            roles[eid] = f"{role} ({etype})"

        analysis = data.get('analysis', {})
        dag = data.get('user_action_path', {}).get('dag', '')
        relationships = data.get('relationships', [])

        return {
            "roles": roles,
            "analysis": analysis,
            "dag": dag,
            "relationships": relationships
        }
    except Exception as e:
        return {"error": str(e)}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='ax', help='LLM model (ax, gpt, gen, etc.)')
    parser.add_argument('--modes', default='dag,ont,kg', help='Comma-separated modes to test')
    args = parser.parse_args()

    model_name = args.model
    modes = [m.strip() for m in args.modes.split(',')]

    print(f"{'='*80}")
    print(f"KG 프롬프트 응답 시간 테스트")
    print(f"모델: {model_name}")
    print(f"테스트 모드: {modes}")
    print(f"메시지 수: {len(MESSAGES)}")
    print(f"{'='*80}\n")

    # LLM 모델 생성
    factory = LLMFactory()
    llm = factory.create_model(model_name)

    prompt_map = {
        'dag': ("Step 7 DAG", HYBRID_DAG_EXTRACTION_PROMPT),
        'ont': ("Step 7 ONT", ONTOLOGY_PROMPT),
        'dag11': ("Step 11 DAG (CoT)", DAG_EXTRACTION_PROMPT_TEMPLATE),
        'kg': ("Step 7 KG (신규)", KG_EXTRACTION_PROMPT),
    }

    results = []

    for msg_info in MESSAGES:
        msg_id = msg_info["id"]
        msg_text = msg_info["text"]

        print(f"\n{'─'*80}")
        print(f"📨 {msg_id}")
        print(f"   expected prerequisite: {msg_info['expected_prereq']}")
        print(f"   expected offer: {msg_info['expected_offer']}")
        print(f"{'─'*80}")

        for mode in modes:
            if mode not in prompt_map:
                print(f"  ⚠️ 알 수 없는 모드: {mode}")
                continue

            label, prompt_text = prompt_map[mode]

            # Step 11 DAG 프롬프트는 {message} 플레이스홀더 사용
            if mode == 'dag11':
                full_prompt = prompt_text.format(message=msg_text)
                result = test_prompt(llm, "", full_prompt, label)
            else:
                result = test_prompt(llm, prompt_text, msg_text, label)

            result["msg_id"] = msg_id
            results.append(result)

            status = "✅" if not result["error"] else "❌"
            print(f"\n  {status} {label}")
            print(f"     시간: {result['elapsed']:.2f}s | 응답 길이: {result['response_len']} chars")

            if result["error"]:
                print(f"     에러: {result['error']}")

            # KG 모드: 역할 분류 결과 표시
            if mode == 'kg' and not result["error"]:
                kg_parsed = extract_roles_from_kg(result["response"])
                if "error" not in kg_parsed:
                    print(f"     분석: {kg_parsed.get('analysis', {}).get('target_customer', 'N/A')}")
                    print(f"     역할 분류:")
                    for eid, role in kg_parsed.get("roles", {}).items():
                        print(f"       - {eid}: {role}")
                    if kg_parsed.get("dag"):
                        dag_lines = kg_parsed["dag"].split('\n') if '\n' in kg_parsed["dag"] else [kg_parsed["dag"]]
                        print(f"     DAG:")
                        for line in dag_lines[:5]:
                            print(f"       {line}")
                else:
                    print(f"     ⚠️ JSON 파싱 실패: {kg_parsed['error']}")
                    # 원본 응답 일부 표시
                    print(f"     응답 (처음 300자):")
                    print(f"       {result['response'][:300]}")

            # DAG/ONT 모드: 응답 일부 표시
            elif mode in ('dag', 'ont') and not result["error"]:
                resp_preview = result["response"][:200].replace('\n', ' ')
                print(f"     응답 미리보기: {resp_preview}...")

    # ─── 요약 ───
    print(f"\n\n{'='*80}")
    print(f"📊 응답 시간 요약 (모델: {model_name})")
    print(f"{'='*80}")
    print(f"{'모드':<25} {'메시지':<30} {'시간(s)':<10} {'응답길이':<10}")
    print(f"{'─'*75}")

    for r in results:
        print(f"{r['label']:<25} {r['msg_id']:<30} {r['elapsed']:<10.2f} {r['response_len']:<10}")

    # 모드별 평균
    print(f"\n{'─'*75}")
    print(f"{'모드별 평균':}")
    mode_times = {}
    for r in results:
        mode_times.setdefault(r['label'], []).append(r['elapsed'])

    for label, times in mode_times.items():
        avg = sum(times) / len(times)
        print(f"  {label:<25} 평균: {avg:.2f}s (min: {min(times):.2f}s, max: {max(times):.2f}s)")

    # Step 7 + Step 11 합산 vs KG 단독 비교
    if 'dag' in modes and 'dag11' in modes and 'kg' in modes:
        dag7_avg = sum(mode_times.get("Step 7 DAG", [0])) / max(len(mode_times.get("Step 7 DAG", [1])), 1)
        dag11_avg = sum(mode_times.get("Step 11 DAG (CoT)", [0])) / max(len(mode_times.get("Step 11 DAG (CoT)", [1])), 1)
        kg_avg = sum(mode_times.get("Step 7 KG (신규)", [0])) / max(len(mode_times.get("Step 7 KG (신규)", [1])), 1)

        print(f"\n{'─'*75}")
        print(f"📈 현재 vs 개선 비교:")
        print(f"  현재: Step 7 DAG ({dag7_avg:.2f}s) + Step 11 DAG ({dag11_avg:.2f}s) = {dag7_avg + dag11_avg:.2f}s")
        print(f"  개선: Step 7 KG ({kg_avg:.2f}s) + Step 11 변환 (~0.01s) = {kg_avg + 0.01:.2f}s")
        print(f"  절감: {dag7_avg + dag11_avg - kg_avg:.2f}s ({(1 - kg_avg/(dag7_avg + dag11_avg))*100:.0f}%)")


if __name__ == '__main__':
    main()
