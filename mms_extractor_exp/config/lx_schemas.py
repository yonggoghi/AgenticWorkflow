"""
MMS Entity Extraction Class Definitions for langextract.

Defines the extraction classes (entity types) used to categorize entities
extracted from Korean MMS advertisement messages. These mirror the 6-type
system from TYPED_ENTITY_EXTRACTION_PROMPT plus channel and purpose.
"""

# Extraction class names and descriptions for langextract.
# These are passed as extraction_class strings in ExampleData.extractions.

# 6 core entity types (matching TYPED_ENTITY_EXTRACTION_PROMPT codes)
EXTRACTION_CLASSES = {
    "Store": {
        "code": "R",
        "description": "물리적 대리점/매장 (지점명 포함)",
        "examples": ["CD대리점 동탄목동점", "유엔대리점 배곧사거리직영점", "PS&M 동탄타임테라스점"],
        "attributes": ["branch_name"],
    },
    "Equipment": {
        "code": "E",
        "description": "단말기/디바이스 모델명",
        "examples": ["아이폰 17", "갤럭시 Z 폴드7", "iPad Air 13", "갤럭시 워치6"],
        "attributes": ["brand", "category"],
    },
    "Product": {
        "code": "P",
        "description": "요금제/부가서비스/유선상품",
        "examples": ["5GX 프라임 요금제", "T끼리 온가족할인", "인터넷+IPTV", "로밍 baro 요금제"],
        "attributes": ["plan_type", "action"],
    },
    "Subscription": {
        "code": "S",
        "description": "월정액 구독 상품",
        "examples": ["T 우주패스 올리브영&스타벅스&이마트24", "T 우주패스 Netflix"],
        "attributes": ["price", "benefits"],
    },
    "Voucher": {
        "code": "V",
        "description": "제휴 할인/쿠폰/기프티콘 (브랜드+혜택 조합)",
        "examples": ["도미노피자 50% 할인", "올리브영 3천 원 기프트카드", "CGV 청년할인"],
        "attributes": ["partner_brand", "discount_detail"],
    },
    "Campaign": {
        "code": "X",
        "description": "마케팅 캠페인/프로모션/이벤트명",
        "examples": ["T Day", "0 day", "special T", "고객 감사 패키지"],
        "attributes": ["period", "participation"],
    },
    "Channel": {
        "code": "C",
        "description": "고객 접점 채널 (URL, 전화번호, 앱, 대리점, 온라인스토어)",
        "examples": ["T world 앱", "skt.sh/xxxxx", "080-XXX-XXXX"],
        "attributes": ["channel_type", "value"],
    },
    "Purpose": {
        "code": "U",
        "description": "광고의 주요 목적",
        "examples": ["상품 가입 유도", "혜택 안내", "이벤트 응모 유도", "대리점/매장 방문 유도"],
        "attributes": [],
    },
}

# Flat list of class names for convenience
EXTRACTION_CLASS_NAMES = list(EXTRACTION_CLASSES.keys())

# Code → class name reverse mapping
CODE_TO_CLASS = {v["code"]: k for k, v in EXTRACTION_CLASSES.items()}


def get_class_description_text() -> str:
    """Build a human-readable description of all extraction classes for prompt injection."""
    lines = []
    for name, info in EXTRACTION_CLASSES.items():
        examples_str = ", ".join(info["examples"][:3])
        lines.append(f"- **{name}** ({info['code']}): {info['description']}  예: {examples_str}")
    return "\n".join(lines)
