#!/usr/bin/env python3
"""
Quick Extractor API 테스트 스크립트
API 서버의 /quick/extract 및 /quick/extract/batch 엔드포인트를 테스트합니다.
"""

import requests
import json
import sys

# API 서버 URL
API_BASE_URL = "http://localhost:8000"

def test_quick_extract_single():
    """단일 메시지 추출 테스트"""
    print("\n" + "="*60)
    print("📝 테스트 1: 단일 메시지 제목/수신거부 번호 추출")
    print("="*60)
    
    url = f"{API_BASE_URL}/quick/extract"
    
    # 테스트 메시지
    test_message = """
    [SKT] 5G 요금제 변경 시 3개월간 50% 할인!
    
    고객님, 안녕하세요.
    지금 5G 프리미엄 요금제로 변경하시면
    - 3개월간 50% 요금 할인
    - 데이터 2배 제공
    - 최신 스마트폰 할인
    
    자세한 내용은 T월드에서 확인하세요.
    무료 수신거부 1504
    """
    
    # TextRank 방법 테스트
    print("\n[TextRank 방법]")
    payload = {
        "message": test_message,
        "method": "textrank"
    }
    
    try:
        response = requests.post(url, json=payload)
        result = response.json()
        
        if result.get('success'):
            print(f"✅ 성공!")
            print(f"   제목: {result['data']['title'][:100]}...")
            print(f"   수신거부: {result['data']['unsubscribe_phone']}")
            print(f"   처리시간: {result['metadata']['processing_time_seconds']}초")
        else:
            print(f"❌ 실패: {result.get('error')}")
    except Exception as e:
        print(f"❌ 오류: {e}")
    
    # LLM 방법 테스트
    print("\n[LLM 방법 (AX)]")
    payload = {
        "message": test_message,
        "method": "llm",
        "llm_model": "ax"
    }
    
    try:
        response = requests.post(url, json=payload)
        result = response.json()
        
        if result.get('success'):
            print(f"✅ 성공!")
            print(f"   제목: {result['data']['title']}")
            print(f"   수신거부: {result['data']['unsubscribe_phone']}")
            print(f"   처리시간: {result['metadata']['processing_time_seconds']}초")
        else:
            print(f"❌ 실패: {result.get('error')}")
    except Exception as e:
        print(f"❌ 오류: {e}")


def test_quick_extract_batch():
    """배치 메시지 추출 테스트"""
    print("\n" + "="*60)
    print("📦 테스트 2: 배치 메시지 제목/수신거부 번호 추출")
    print("="*60)
    
    url = f"{API_BASE_URL}/quick/extract/batch"
    
    # 테스트 메시지들
    test_messages = [
        "[광고]\nSK텔레콤\n개인고객센터/변경해지",
        "[SKT] 2월 T Day 이벤트 안내",
        "5G 프리미엄 요금제 가입 시 특별 혜택!\n무료 수신거부 1504"
    ]
    
    # TextRank 방법으로 배치 처리
    print("\n[TextRank 방법]")
    payload = {
        "messages": test_messages,
        "method": "textrank"
    }
    
    try:
        response = requests.post(url, json=payload)
        result = response.json()
        
        if result.get('success'):
            print(f"✅ 성공!")
            print(f"   총 메시지: {result['data']['statistics']['total_messages']}개")
            print(f"   수신거부 추출: {result['data']['statistics']['with_unsubscribe_phone']}개")
            print(f"   추출률: {result['data']['statistics']['extraction_rate']}%")
            print(f"   총 처리시간: {result['metadata']['processing_time_seconds']}초")
            print(f"   평균 처리시간: {result['metadata']['avg_time_per_message']}초/메시지")
            
            print("\n   결과 샘플:")
            for msg_result in result['data']['results'][:3]:
                print(f"     [{msg_result['msg_id']}] 제목: {msg_result['title'][:50]}...")
                print(f"         수신거부: {msg_result['unsubscribe_phone']}")
        else:
            print(f"❌ 실패: {result.get('error')}")
    except Exception as e:
        print(f"❌ 오류: {e}")


def test_invalid_request():
    """잘못된 요청 테스트"""
    print("\n" + "="*60)
    print("⚠️  테스트 3: 에러 처리")
    print("="*60)
    
    # 메시지 없는 요청
    print("\n[빈 메시지 요청]")
    url = f"{API_BASE_URL}/quick/extract"
    payload = {}
    
    try:
        response = requests.post(url, json=payload)
        result = response.json()
        
        if not result.get('success'):
            print(f"✅ 예상대로 에러 반환: {result.get('error')}")
        else:
            print(f"❌ 예상과 다름: 에러가 발생해야 함")
    except Exception as e:
        print(f"❌ 오류: {e}")
    
    # 잘못된 method
    print("\n[잘못된 method 요청]")
    payload = {
        "message": "테스트 메시지",
        "method": "invalid_method"
    }
    
    try:
        response = requests.post(url, json=payload)
        result = response.json()
        
        if not result.get('success'):
            print(f"✅ 예상대로 에러 반환: {result.get('error')}")
        else:
            print(f"❌ 예상과 다름: 에러가 발생해야 함")
    except Exception as e:
        print(f"❌ 오류: {e}")


def test_server_health():
    """서버 상태 확인"""
    print("\n" + "="*60)
    print("🏥 서버 상태 확인")
    print("="*60)
    
    url = f"{API_BASE_URL}/health"
    
    try:
        response = requests.get(url)
        result = response.json()
        
        if result.get('status') == 'healthy':
            print(f"✅ 서버 정상 작동 중")
            print(f"   버전: {result.get('version', 'N/A')}")
            print(f"   타임스탬프: {result.get('timestamp', 'N/A')}")
        else:
            print(f"⚠️  서버 상태 이상: {result}")
    except Exception as e:
        print(f"❌ 서버 연결 실패: {e}")
        print(f"   API 서버가 실행 중인지 확인하세요: python api.py")
        sys.exit(1)


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🧪 Quick Extractor API 테스트 시작")
    print("="*60)
    print(f"API 서버: {API_BASE_URL}")
    
    # 서버 상태 먼저 확인
    test_server_health()
    
    # 각 테스트 실행
    test_quick_extract_single()
    test_quick_extract_batch()
    test_invalid_request()
    
    print("\n" + "="*60)
    print("✅ 모든 테스트 완료")
    print("="*60 + "\n")

