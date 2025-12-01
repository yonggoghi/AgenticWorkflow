"""
Test script for Non-LLM tools
Quick validation that tools are working
"""

import sys
import os

# Adjust path for proper module import
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Use absolute import
from mms_agent.tools import (
    search_entities_kiwi,
    search_entities_fuzzy,
    classify_program,
    match_store_info,
    validate_entities
)

# Real MMS advertisement for comprehensive testing
TEST_MESSAGE = """(광고)[SKT] 새서울대리점 대치직영점 10월 혜택 안내__고객님, 안녕하세요._대치역 8번 출구 인근 새서울대리점 대치직영점에서 10월 혜택을 안내드립니다._특별 이벤트와 다양한 혜택을 경험해 보세요.__■ 갤럭시 Z 플립7/폴드7 구매 혜택_- 최대 할인 제공_- 갤럭시 워치 무료 증정(5GX 프라임 요금제 이용 시)__■ 아이폰 신제품 구매 혜택_- 최대 할인 및 쓰던 폰 반납 시 최대 보상 제공_- 아이폰 에어 구매 시 에어팟 증정(5GX 프라임 요금제 이용 시)__■ 공신폰/부모님폰 한정 수량 특별 할인_- 매일 선착순 3명 휴대폰 최대 할인__■ 새서울대리점 대치직영점_- 주소: 서울특별시 강남구 삼성로 151_- 연락처: 02-539-9965_- 찾아오시는 길: 3호선 대치역 8번 출구에서 직진 50m, 선경아파트 상가 bbq 건물 1층_- 영업 시간: 평일 오전 10시 30분~오후 7시, 토요일 오전 11시~오후 6시__▶ 매장 홈페이지 예약/상담 : https://t-mms.kr/t.do?m=#61&s=34192&a=&u=https://tworldfriends.co.kr/D138580279__■ 문의: SKT 고객센터(1558, 무료)__SKT와 함께해 주셔서 감사합니다.__무료 수신거부 1504"""

def test_search_entities_kiwi():
    print("=" * 60)
    print("Testing search_entities_kiwi...")
    print("=" * 60)
    
    result = search_entities_kiwi.invoke({"message": TEST_MESSAGE})
    
    print(f"Message (first 100 chars): {TEST_MESSAGE[:100]}...")
    print(f"Entities found: {len(result.get('entities', []))}")
    print(f"Entities: {result.get('entities', [])[:10]}")  # Show first 10
    print(f"Candidate items: {len(result.get('candidate_items', []))}")
    print(f"Top candidates: {result.get('candidate_items', [])[:5]}")
    print()

def test_search_entities_fuzzy():
    print("=" * 60)
    print("Testing search_entities_fuzzy...")
    print("=" * 60)
    
    import json
    entities = "아이폰,넷플릭스"
    result_str = search_entities_fuzzy.invoke({
        "entities": entities,
        "threshold": 0.5
    })
    
    result = json.loads(result_str)
    
    print(f"Input Entities: {entities}")
    print(f"Matches: {len(result)}")
    for match in result[:3]:
        print(f"  - {match.get('item_nm')} (Score: {match.get('similarity', 0):.3f})")
    print()

def test_classify_program():
    print("=" * 60)
    print("Testing classify_program...")
    print("=" * 60)
    
    result = classify_program.invoke({
        "message": TEST_MESSAGE,
        "top_k": 5
    })
    
    print(f"Message (first 100 chars): {TEST_MESSAGE[:100]}...")
    print(f"Programs found: {len(result.get('programs', []))}")
    for prog in result.get('programs', [])[:5]:
        print(f"  - {prog.get('pgm_nm')} (Score: {prog.get('similarity', 0):.3f})")
    print()

def test_match_store_info():
    print("=" * 60)
    print("Testing match_store_info...")
    print("=" * 60)
    
    import json
    store_name = "새서울대리점 대치직영점"  # From TEST_MESSAGE
    result_str = match_store_info.invoke({"store_name": store_name})
    
    result = json.loads(result_str)
    
    print(f"Store Name: {store_name}")
    print(f"Matches: {len(result) if isinstance(result, list) else 0}")
    if isinstance(result, list):
        for match in result[:5]:
            print(f"  - {match.get('org_nm')} ({len(match.get('org_cd', []))} codes)")
    print()

def test_validate_entities():
    print("=" * 60)
    print("Testing validate_entities...")
    print("=" * 60)
    
    import json
    entities = [
        {"item_nm": "아이폰 15", "item_id": "IPHONE15"},
        {"item_nm": "넷플릭스", "item_id": "NETFLIX"}
    ]
    message = "아이폰 구매하고 넷플릭스 받으세요"
    
    result_str = validate_entities.invoke({
        "entities_json": json.dumps(entities),
        "message": message
    })
    
    result = json.loads(result_str)
    
    print(f"Input: {len(entities)} entities")
    print(f"Output: {len(result) if isinstance(result, list) else 0} entities")
    print()

if __name__ == "__main__":
    print("\n🧪 Testing Non-LLM Tools\n")
    
    try:
        test_search_entities_kiwi()
    except Exception as e:
        print(f"❌ search_entities_kiwi failed: {e}\n")
    
    try:
        test_search_entities_fuzzy()
    except Exception as e:
        print(f"❌ search_entities_fuzzy failed: {e}\n")
    
    try:
        test_classify_program()
    except Exception as e:
        print(f"❌ classify_program failed: {e}\n")
    
    try:
        test_match_store_info()
    except Exception as e:
        print(f"❌ match_store_info failed: {e}\n")
    
    try:
        test_validate_entities()
    except Exception as e:
        print(f"❌ validate_entities failed: {e}\n")
    
    print("✅ All tests completed!\n")
