"""
Few-shot ExampleData for Korean MMS entity extraction via langextract.

Provides representative Korean MMS advertisement examples covering text-aligned entity
types (Store, Equipment, Product, Subscription, Voucher, Campaign, Channel).
Purpose is excluded since it's a classification label, not a text span that langextract
can align to source text.
"""

from langextract.core.data import ExampleData, Extraction


def build_mms_examples() -> list[ExampleData]:
    """Build few-shot ExampleData instances for MMS entity extraction.

    Returns:
        List of 5 ExampleData covering all entity types.
    """
    return [
        # Example 1: Store + Equipment + Voucher + Channel
        ExampleData(
            text=(
                "(광고)[SKT] CD대리점 동탄목동점에서 아이폰 17 Pro 사전예약 시작! "
                "최대 22만 원 캐시백 + 올리브영 3천 원 기프트카드 증정. "
                "매장 방문 또는 skt.sh/abc123 에서 확인하세요. "
                "수신거부 080-1234-5678"
            ),
            extractions=[
                Extraction(extraction_class="Store", extraction_text="CD대리점 동탄목동점"),
                Extraction(extraction_class="Equipment", extraction_text="아이폰 17 Pro"),
                Extraction(extraction_class="Voucher", extraction_text="올리브영 3천 원 기프트카드"),
                Extraction(extraction_class="Channel", extraction_text="skt.sh/abc123"),
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
                Extraction(extraction_class="Product", extraction_text="5GX 프라임 요금제"),
                Extraction(extraction_class="Campaign", extraction_text="T Day"),
                Extraction(extraction_class="Channel", extraction_text="T world 앱"),
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
                Extraction(extraction_class="Subscription", extraction_text="T 우주패스 올리브영&스타벅스&이마트24"),
                Extraction(extraction_class="Voucher", extraction_text="올리브영 5천 원 할인"),
                Extraction(extraction_class="Voucher", extraction_text="스타벅스 아메리카노 1잔 무료"),
                Extraction(extraction_class="Channel", extraction_text="skt.sh/xyz789"),
            ],
        ),
        # Example 4: Equipment + Product + Store
        ExampleData(
            text=(
                "[SKT] 갤럭시 Z 플립7 출시 기념! 유엔대리점 배곧사거리직영점 방문 시 "
                "T끼리 온가족할인 동시 가입 혜택. "
                "기기 구매 고객 대상 도미노피자 50% 할인 쿠폰 증정. "
                "문의 1588-0010"
            ),
            extractions=[
                Extraction(extraction_class="Equipment", extraction_text="갤럭시 Z 플립7"),
                Extraction(extraction_class="Store", extraction_text="유엔대리점 배곧사거리직영점"),
                Extraction(extraction_class="Product", extraction_text="T끼리 온가족할인"),
                Extraction(extraction_class="Voucher", extraction_text="도미노피자 50% 할인 쿠폰"),
            ],
        ),
        # Example 5: Campaign + Voucher (event-focused)
        ExampleData(
            text=(
                "(광고)[SKT] 이번 주 0 day 이벤트! "
                "퀴즈 참여하고 CGV 영화 티켓 2매 받자! "
                "T world 앱 > 이벤트 탭에서 참여하세요. "
                "수신거부 080-0000-0000"
            ),
            extractions=[
                Extraction(extraction_class="Campaign", extraction_text="0 day"),
                Extraction(extraction_class="Voucher", extraction_text="CGV 영화 티켓 2매"),
                Extraction(extraction_class="Channel", extraction_text="T world 앱"),
            ],
        ),
    ]
