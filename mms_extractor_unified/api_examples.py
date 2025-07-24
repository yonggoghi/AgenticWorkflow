#!/usr/bin/env python3
"""
Example usage of MMS Extractor API.
"""
import requests
import json
import time

# API 서버 설정
API_BASE_URL = "http://localhost:8000"

def test_health_check():
    """Health check 테스트"""
    print("=== Health Check ===")
    response = requests.get(f"{API_BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    print()

def test_models_info():
    """모델 정보 조회 테스트"""
    print("=== Models Info ===")
    response = requests.get(f"{API_BASE_URL}/models")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    print()

def test_single_extraction():
    """단일 메시지 추출 테스트"""
    print("=== Single Message Extraction ===")
    
    test_message = """
    [SK텔레콤] ZEM폰 포켓몬에디션3 안내
    (광고)[SKT] 우리 아이 첫 번째 스마트폰, ZEM 키즈폰__#04 고객님, 안녕하세요!
    우리 아이 스마트폰 고민 중이셨다면, 자녀 스마트폰 관리 앱 ZEM이 설치된 SKT만의 안전한 키즈폰,
    ZEM폰 포켓몬에디션3으로 우리 아이 취향을 저격해 보세요!
    신학기를 맞이하여 SK텔레콤 공식 인증 대리점에서 풍성한 혜택을 제공해 드리고 있습니다!
    """
    
    payload = {
        "message": test_message,
        "offer_info_data_src": "local"  # 또는 "db"
    }
    
    response = requests.post(
        f"{API_BASE_URL}/extract",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Success: {result['success']}")
        print(f"Processing Time: {result['metadata']['processing_time_seconds']}s")
        print("Extracted Information:")
        print(json.dumps(result['result'], indent=2, ensure_ascii=False))
    else:
        print(f"Error: {response.json()}")
    print()

def test_batch_extraction():
    """배치 메시지 추출 테스트"""
    print("=== Batch Message Extraction ===")
    
    test_messages = [
        "[SK텔레콤] ZEM폰 포켓몬에디션3 안내 - 우리 아이 첫 스마트폰!",
        "[SK텔레콤] 갤럭시 S25 출시 안내 - 최신 스마트폰을 만나보세요!",
        "[SK텔레콤] T멤버십 혜택 안내 - 다양한 할인과 쿠폰을 받아보세요!"
    ]
    
    payload = {
        "messages": test_messages,
        "offer_info_data_src": "local"
    }
    
    response = requests.post(
        f"{API_BASE_URL}/extract/batch",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Success: {result['success']}")
        print(f"Processing Time: {result['metadata']['processing_time_seconds']}s")
        print(f"Summary: {result['summary']}")
        
        for i, res in enumerate(result['results']):
            print(f"\nMessage {i+1} Result:")
            if res['success']:
                print(f"  Title: {res['result']['title']}")
                print(f"  Products: {len(res['result']['product'])} items")
            else:
                print(f"  Error: {res['error']}")
    else:
        print(f"Error: {response.json()}")
    print()

def test_status():
    """서버 상태 확인 테스트"""
    print("=== Server Status ===")
    response = requests.get(f"{API_BASE_URL}/status")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    print()

def main():
    """모든 테스트 실행"""
    print("MMS Extractor API 테스트 시작\n")
    print("주의: API 서버가 실행 중이어야 합니다.")
    print("서버 시작: python api.py --host localhost --port 8000\n")
    
    try:
        # 기본 테스트들
        test_health_check()
        test_models_info()
        test_status()
        
        # 추출 테스트들
        test_single_extraction()
        test_batch_extraction()
        
        print("모든 테스트 완료!")
        
    except requests.exceptions.ConnectionError:
        print("❌ API 서버에 연결할 수 없습니다.")
        print("다음 명령으로 서버를 시작하세요:")
        print("python api.py --host localhost --port 8000")
    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {e}")

if __name__ == "__main__":
    main() 