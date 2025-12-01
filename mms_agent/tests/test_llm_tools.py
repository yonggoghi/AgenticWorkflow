"""
Test script for LLM tools
Testing LLM-based extraction capabilities
"""

import sys
import os
import json

# Add path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from mms_agent.tools import (
    extract_entities_llm,
    extract_main_info,
    extract_entity_dag
)

# Real MMS advertisement for testing
TEST_MESSAGE = """(광고)[SKT] 새서울대리점 대치직영점 10월 혜택 안내__고객님, 안녕하세요._대치역 8번 출구 인근 새서울대리점 대치직영점에서 10월 혜택을 안내드립니다._특별 이벤트와 다양한 혜택을 경험해 보세요.__■ 갤럭시 Z 플립7/폴드7 구매 혜택_- 최대 할인 제공_- 갤럭시 워치 무료 증정(5GX 프라임 요금제 이용 시)__■ 아이폰 신제품 구매 혜택_- 최대 할인 및 쓰던 폰 반납 시 최대 보상 제공_- 아이폰 에어 구매 시 에어팟 증정(5GX 프라임 요금제 이용 시)__■ 공신폰/부모님폰 한정 수량 특별 할인_- 매일 선착순 3명 휴대폰 최대 할인__■ 새서울대리점 대치직영점_- 주소: 서울특별시 강남구 삼성로 151_- 연락처: 02-539-9965_- 찾아오시는 길: 3호선 대치역 8번 출구에서 직진 50m, 선경아파트 상가 bbq 건물 1층_- 영업 시간: 평일 오전 10시 30분~오후 7시, 토요일 오전 11시~오후 6시__▶ 매장 홈페이지 예약/상담 : https://t-mms.kr/t.do?m=#61&s=34192&a=&u=https://tworldfriends.co.kr/D138580279__■ 문의: SKT 고객센터(1558, 무료)__SKT와 함께해 주셔서 감사합니다.__무료 수신거부 1504"""

def test_extract_entities_llm():
    print("=" * 60)
    print("Testing extract_entities_llm...")
    print("=" * 60)
    
    result_str = extract_entities_llm.invoke({"message": TEST_MESSAGE})
    result = json.loads(result_str)
    
    print(f"Message (first 100 chars): {TEST_MESSAGE[:100]}...")
    print(f"Extracted entities: {len(result) if isinstance(result, list) else  0}")
    
    if isinstance(result, list):
        for item in result[:5]:
            print(f"  - {item.get('item_name_in_msg')} → {item.get('item_nm')} (Score: {item.get('score', 0):.2f})")
    elif 'error' in result:
        print(f"Error: {result['error']}")
    
    print()

def test_extract_main_info():
    print("=" * 60)
    print("Testing extract_main_info...")
    print("=" * 60)
    
    result_str = extract_main_info.invoke({
        "message": TEST_MESSAGE,
        "mode": "llm"
    })
    result = json.loads(result_str)
    
    print(f"Message (first 100 chars): {TEST_MESSAGE[:100]}...")
    
    if 'error' not in result:
        print(f"Title: {result.get('title', 'N/A')[:80]}...")
        print(f"Purpose: {result.get('purpose', [])}")
        print(f"Products: {len(result.get('product', []))}")
        for prod in result.get('product', [])[:3]:
            print(f"  - {prod.get('name')} ({prod.get('action')})")
        print(f"Channels: {len(result.get('channel', []))}")
        for ch in result.get('channel', [])[:3]:
            print(f"  - {ch.get('type')}: {ch.get('value', '')[:30]}...")
    else:
        print(f"Error: {result['error']}")
    
    print()

def test_extract_entity_dag():
    print("=" * 60)
    print("Testing extract_entity_dag...")
    print("=" * 60)
    
    result_str = extract_entity_dag.invoke({"message": TEST_MESSAGE})
    result = json.loads(result_str)
    
    print(f"Message (first 100 chars): {TEST_MESSAGE[:100]}...")
    
    if 'error' not in result:
        print(f"Entities: {result.get('entities', [])}")
        print(f"\nDAG:")
        print(result.get('dag', '')[:300] + "...")
    else:
        print(f"Error: {result['error']}")
    
    print()

if __name__ == "__main__":
    print("\n🧪 Testing LLM Tools\n")
    print("Note: This requires LLM API access and may take some time.\n")
    
    try:
        test_extract_entities_llm()
    except Exception as e:
        print(f"❌ extract_entities_llm failed: {e}\n")
    
    try:
        test_extract_main_info()
    except Exception as e:
        print(f"❌ extract_main_info failed: {e}\n")
    
    try:
        test_extract_entity_dag()
    except Exception as e:
        print(f"❌ extract_entity_dag failed: {e}\n")
    
    print("✅ All LLM tool tests completed!\n")
