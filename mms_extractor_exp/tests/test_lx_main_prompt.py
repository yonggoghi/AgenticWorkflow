"""
Test: Can LangExtract generate main-prompt-style JSON output?

Instead of entity-span extraction, we configure LangExtract with few-shot
examples whose "extractions" mimic the main prompt's JSON fields:
  title, purpose, sales_script, product (name+action), channel (type+value+action), pgm
"""

import sys
import os
import json
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.WARNING, format="%(name)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

from langextract.core.data import ExampleData, Extraction

# ── Few-shot examples: main-prompt-style ────────────────────────────────
# We flatten the nested JSON into extraction classes.
# For arrays (product, channel, purpose), we emit one Extraction per item.
# Attributes carry the sub-fields (action, type, etc.).

MAIN_PROMPT_EXAMPLES = [
    # Example 1: Store + Equipment + Voucher ad
    ExampleData(
        text=(
            "(광고)[SKT] CD대리점 동탄목동점에서 아이폰 17 Pro 사전예약 시작! "
            "최대 22만 원 캐시백 + 올리브영 3천 원 기프트카드 증정. "
            "매장 방문 또는 skt.sh/abc123 에서 확인하세요. "
            "수신거부 080-1234-5678"
        ),
        extractions=[
            Extraction(extraction_class="title", extraction_text="아이폰 17 Pro 사전예약 최대 22만원 캐시백"),
            Extraction(extraction_class="purpose", extraction_text="상품 가입 유도"),
            Extraction(extraction_class="purpose", extraction_text="대리점/매장 방문 유도"),
            Extraction(extraction_class="sales_script", extraction_text="아이폰17Pro 사전예약 시작! CD대리점 동탄목동점 방문 시 최대 22만원 캐시백+올리브영 기프트카드. 안내드릴까요?"),
            Extraction(extraction_class="product", extraction_text="아이폰 17 Pro",
                       attributes={"action": "구매"}),
            Extraction(extraction_class="product", extraction_text="올리브영 3천 원 기프트카드",
                       attributes={"action": "쿠폰다운로드"}),
            Extraction(extraction_class="channel", extraction_text="CD대리점 동탄목동점",
                       attributes={"type": "대리점", "action": "가입"}),
            Extraction(extraction_class="channel", extraction_text="skt.sh/abc123",
                       attributes={"type": "URL", "action": "추가 정보"}),
        ],
    ),
    # Example 2: Product + Campaign + Channel
    ExampleData(
        text=(
            "[SKT] 5GX 프라임 요금제 가입하고 T Day 혜택 받으세요! "
            "이번 달 T Day 기간 한정 데이터 2배 제공. "
            "T world 앱에서 바로 가입 가능합니다."
        ),
        extractions=[
            Extraction(extraction_class="title", extraction_text="5GX 프라임 요금제 T Day 혜택"),
            Extraction(extraction_class="purpose", extraction_text="상품 가입 유도"),
            Extraction(extraction_class="purpose", extraction_text="이벤트 응모 유도"),
            Extraction(extraction_class="sales_script", extraction_text="T Day 기간 5GX프라임 가입 시 데이터 2배! T world 앱에서 바로 가입 가능. 안내드릴까요?"),
            Extraction(extraction_class="product", extraction_text="5GX 프라임 요금제",
                       attributes={"action": "가입"}),
            Extraction(extraction_class="product", extraction_text="T Day",
                       attributes={"action": "참여"}),
            Extraction(extraction_class="channel", extraction_text="T world 앱",
                       attributes={"type": "앱", "action": "가입"}),
        ],
    ),
    # Example 3: Subscription + Voucher
    ExampleData(
        text=(
            "(광고) T 우주패스 올리브영&스타벅스&이마트24 구독하면 "
            "매월 올리브영 5천 원 할인 + 스타벅스 아메리카노 1잔 무료! "
            "월 9,900원으로 다양한 혜택을 누리세요. "
            "자세히 보기 skt.sh/xyz789"
        ),
        extractions=[
            Extraction(extraction_class="title", extraction_text="T 우주패스 구독 올리브영·스타벅스 혜택"),
            Extraction(extraction_class="purpose", extraction_text="상품 가입 유도"),
            Extraction(extraction_class="purpose", extraction_text="혜택 안내"),
            Extraction(extraction_class="sales_script", extraction_text="T우주패스 월9,900원! 올리브영 5천원할인+스타벅스 아메리카노 무료. 가입 안내드릴까요?"),
            Extraction(extraction_class="product", extraction_text="T 우주패스 올리브영&스타벅스&이마트24",
                       attributes={"action": "가입"}),
            Extraction(extraction_class="channel", extraction_text="skt.sh/xyz789",
                       attributes={"type": "URL", "action": "추가 정보"}),
        ],
    ),
]

# ── Prompt description ──────────────────────────────────────────────────

MAIN_PROMPT_DESCRIPTION = """\
SK텔레콤 MMS 광고 메시지에서 다음 구조화된 정보를 추출하라.

## 추출 대상
- **title**: 광고의 핵심 내용을 요약한 간결한 제목 (최대 50자, 개조식)
- **purpose**: 광고의 주요 목적. 다음 중 선택:
  상품 가입 유도, 대리점/매장 방문 유도, 웹/앱 접속 유도, 이벤트 응모 유도,
  혜택 안내, 쿠폰 제공 안내, 경품 제공 안내, 수신 거부 안내, 기타 정보 제공
- **sales_script**: 콜센터 상담사 화면에 표시할 극히 간결한 크로스셀 멘트
- **product**: 광고된 상품/서비스명. action 속성: 구매, 가입, 사용, 방문, 참여, 코드입력, 쿠폰다운로드, 기타
- **channel**: 고객 접점 채널. type 속성: URL, 전화번호, 앱, 대리점, 온라인스토어. action 속성: 가입, 추가 정보, 문의, 수신, 수신 거부

## 추출 규칙
1. 상품명은 원문에 등장하는 그대로 추출하라. 번역하지 말라.
2. title과 sales_script는 원문을 기반으로 새로 생성하라.
3. purpose는 반드시 위 enum 목록에서만 선택하라.
4. product의 action은 반드시 위 enum에서 선택하라.
5. 리콜(recall)을 우선하여 모든 관련 상품을 빠짐없이 추출하라.
"""


def run_test(model_id: str = "ax"):
    """Run LangExtract with main-prompt-style examples on a test message."""
    import langextract as lx
    # Trigger provider registration
    from services import lx_provider as _  # noqa: F401

    test_message = (
        "[SKT] T 우주패스 쇼핑 출시! "
        "지금 링크를 눌러 가입하면 첫 달 1,000원에 이용 가능합니다. "
        "가입 고객 전원에게 11번가 포인트 3,000P와 아마존 무료배송 쿠폰을 드립니다. "
        "문의: 114"
    )

    print("=" * 70)
    print(f"MODEL: {model_id}")
    print(f"MESSAGE: {test_message}")
    print("=" * 70)

    # ── Run LangExtract with main-prompt examples ───────────────────────
    print("\n▶ Running LangExtract (main-prompt-style examples)...")
    t0 = time.time()
    result = lx.extract(
        text_or_documents=test_message,
        prompt_description=MAIN_PROMPT_DESCRIPTION,
        examples=MAIN_PROMPT_EXAMPLES,
        model_id=model_id,
        max_char_buffer=5000,
        extraction_passes=1,
        temperature=0.0,
        fetch_urls=False,
        show_progress=False,
        resolver_params={
            "enable_fuzzy_alignment": True,
            "fuzzy_alignment_threshold": 0.5,  # lower threshold for generated text
            "accept_match_lesser": True,
            "suppress_parse_errors": True,
        },
    )
    elapsed = time.time() - t0
    print(f"  ⏱  {elapsed:.1f}s")

    # ── Display raw extractions ─────────────────────────────────────────
    print(f"\n  Extractions ({len(result.extractions or [])}):")
    for ext in (result.extractions or []):
        attrs = ""
        if ext.attributes:
            attrs = f"  attrs={ext.attributes}"
        align = ""
        if ext.alignment_status:
            align = f"  [{ext.alignment_status.value}]"
        print(f"    {ext.extraction_class:15s} │ {ext.extraction_text}{attrs}{align}")

    # ── Convert to main-prompt JSON ─────────────────────────────────────
    json_result = extractions_to_main_json(result.extractions or [])
    print(f"\n  Reconstructed JSON:")
    print(json.dumps(json_result, indent=2, ensure_ascii=False))

    # ── Also run current entity-only extraction for comparison ──────────
    print("\n" + "─" * 70)
    print("▶ Running LangExtract (entity-only, current setup)...")
    from core.lx_extractor import extract_mms_entities, lx_result_to_dict
    t0 = time.time()
    ent_result = extract_mms_entities(test_message, model_id=model_id, temperature=0.0)
    elapsed = time.time() - t0
    print(f"  ⏱  {elapsed:.1f}s")
    print(f"\n  Entity extractions ({len(ent_result.extractions or [])}):")
    for ext in (ent_result.extractions or []):
        align = f"  [{ext.alignment_status.value}]" if ext.alignment_status else ""
        print(f"    {ext.extraction_class:15s} │ {ext.extraction_text}{align}")
    ent_json = lx_result_to_dict(ent_result)
    print(f"\n  Converted JSON (via lx_result_to_dict):")
    print(json.dumps(ent_json, indent=2, ensure_ascii=False))


def extractions_to_main_json(extractions: list) -> dict:
    """Convert list of Extraction objects to main-prompt JSON structure."""
    result = {
        "title": "",
        "purpose": [],
        "sales_script": "",
        "product": [],
        "channel": [],
        "pgm": [],
    }
    for ext in extractions:
        cls = ext.extraction_class
        text = ext.extraction_text
        attrs = ext.attributes or {}

        if cls == "title":
            result["title"] = text
        elif cls == "purpose":
            result["purpose"].append(text)
        elif cls == "sales_script":
            result["sales_script"] = text
        elif cls == "product":
            result["product"].append({
                "name": text,
                "action": attrs.get("action", "기타"),
            })
        elif cls == "channel":
            result["channel"].append({
                "type": attrs.get("type", "기타"),
                "value": text,
                "action": attrs.get("action", "추가 정보"),
            })
        elif cls == "pgm":
            result["pgm"].append(text)

    return result


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "ax"
    run_test(model)
