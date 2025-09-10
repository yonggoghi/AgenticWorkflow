#!/usr/bin/env python3
"""
API 성능 테스트 스크립트
======================

이 스크립트는 MMS 추출기 API가 매번 호출 시마다 무거운 작업(임베딩 생성 등)을 
반복하는지 확인하기 위한 테스트입니다.

테스트 방법:
1. API 서버를 별도 프로세스로 시작
2. 동일한 요청을 여러 번 보내면서 응답 시간 측정
3. 임베딩 생성 로그 모니터링
"""

import requests
import json
import time
import threading
import subprocess
import sys
import os
from pathlib import Path

# 테스트 설정
API_BASE_URL = "http://127.0.0.1:8080"
TEST_MESSAGE = """
[SK텔레콤] 2월 0 day 혜택 안내
(광고)[SKT] 2월 0 day 혜택 안내__[2월 10일(토) 혜택]_만 13~34세 고객이라면_베어유 모든 강의 14일 무료 수강 쿠폰 드립니다!_(선착순 3만 명 증정)_▶ 자세히 보기: http://t-mms.kr/t.do?m=#61&s=24589&a=&u=https://bit.ly/3SfBjjc__■ 에이닷 X T 멤버십 시크릿코드 이벤트_에이닷 T 멤버십 쿠폰함에 '에이닷이빵쏜닷'을 입력해보세요!_뚜레쥬르 데일리우유식빵 무료 쿠폰을 드립니다._▶ 시크릿코드 입력하러 가기: https://bit.ly/3HCUhLM__■ 문의: SKT 고객센터(1558, 무료)_무료 수신거부 1504
"""

def wait_for_server(url, timeout=60):
    """서버가 시작될 때까지 대기"""
    print(f"서버 시작 대기 중... ({url})")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                print("✅ 서버가 시작되었습니다!")
                return True
        except requests.exceptions.RequestException:
            time.sleep(2)
            print(".", end="", flush=True)
    
    print(f"\n❌ {timeout}초 내에 서버가 시작되지 않았습니다.")
    return False

def test_single_request(request_num, test_config):
    """단일 API 요청 테스트"""
    url = f"{API_BASE_URL}/extract"
    payload = {
        "message": TEST_MESSAGE,
        "llm_model": test_config.get("llm_model", "gemma"),
        "product_info_extraction_mode": test_config.get("product_info_extraction_mode", "nlp"),
        "entity_matching_mode": test_config.get("entity_matching_mode", "logic")
    }
    
    start_time = time.time()
    
    try:
        response = requests.post(url, json=payload, timeout=120)
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            processing_time = result.get("metadata", {}).get("processing_time_seconds", 0)
            total_time = end_time - start_time
            
            return {
                "request_num": request_num,
                "success": True,
                "total_time": round(total_time, 3),
                "server_processing_time": processing_time,
                "network_overhead": round(total_time - processing_time, 3),
                "status_code": response.status_code
            }
        else:
            return {
                "request_num": request_num,
                "success": False,
                "total_time": round(time.time() - start_time, 3),
                "error": f"HTTP {response.status_code}: {response.text}",
                "status_code": response.status_code
            }
            
    except Exception as e:
        return {
            "request_num": request_num,
            "success": False,
            "total_time": round(time.time() - start_time, 3),
            "error": str(e),
            "status_code": None
        }

def test_api_performance():
    """API 성능 테스트 실행"""
    print("="*60)
    print("MMS 추출기 API 성능 테스트")
    print("="*60)
    
    # 1. 서버 상태 확인
    print("\n1. 서버 상태 확인 중...")
    if not wait_for_server(API_BASE_URL):
        print("❌ 서버에 연결할 수 없습니다. 먼저 API 서버를 시작하세요:")
        print("   python api.py --host 0.0.0.0 --port 8080")
        return False
    
    # 2. 헬스체크
    try:
        health_response = requests.get(f"{API_BASE_URL}/health")
        print(f"   서버 상태: {health_response.json()}")
    except Exception as e:
        print(f"   헬스체크 오류: {e}")
    
    # 3. 동일한 설정으로 여러 번 요청 (캐싱 효과 확인)
    print("\n2. 동일한 설정으로 연속 요청 테스트 (캐싱 효과 확인)")
    print("   - 같은 LLM 모델, 같은 설정으로 5번 요청")
    print("   - 두 번째 요청부터는 빨라져야 함 (임베딩 재생성 안 함)")
    
    same_config_results = []
    test_config = {
        "llm_model": "gemma",
        "product_info_extraction_mode": "nlp", 
        "entity_matching_mode": "logic"
    }
    
    for i in range(5):
        print(f"   요청 {i+1}/5 진행 중...", end=" ", flush=True)
        result = test_single_request(i+1, test_config)
        same_config_results.append(result)
        
        if result["success"]:
            print(f"✅ {result['total_time']}초 (서버: {result['server_processing_time']}초)")
        else:
            print(f"❌ 실패: {result['error']}")
        
        time.sleep(1)  # 요청 간 간격
    
    # 4. 다른 LLM 모델로 요청 (LLM 재초기화 확인)
    print("\n3. 다른 LLM 모델로 요청 테스트 (LLM 재초기화 확인)")
    print("   - LLM 모델을 변경하면 재초기화로 인해 느려질 수 있음")
    
    different_llm_results = []
    llm_models = ["gemma", "gpt", "claude"]
    
    for i, llm_model in enumerate(llm_models):
        test_config_diff = {
            "llm_model": llm_model,
            "product_info_extraction_mode": "nlp",
            "entity_matching_mode": "logic"
        }
        
        print(f"   {llm_model} 모델로 요청 중...", end=" ", flush=True)
        result = test_single_request(f"LLM-{llm_model}", test_config_diff)
        different_llm_results.append(result)
        
        if result["success"]:
            print(f"✅ {result['total_time']}초 (서버: {result['server_processing_time']}초)")
        else:
            print(f"❌ 실패: {result['error']}")
        
        time.sleep(1)
    
    # 5. 결과 분석
    print("\n4. 결과 분석")
    print("="*60)
    
    # 동일 설정 결과 분석
    successful_same = [r for r in same_config_results if r["success"]]
    if len(successful_same) >= 2:
        first_request_time = successful_same[0]["total_time"]
        subsequent_times = [r["total_time"] for r in successful_same[1:]]
        avg_subsequent_time = sum(subsequent_times) / len(subsequent_times)
        
        print(f"첫 번째 요청 시간: {first_request_time}초")
        print(f"후속 요청 평균 시간: {avg_subsequent_time:.3f}초")
        
        if avg_subsequent_time < first_request_time * 0.8:
            print("✅ 좋음: 후속 요청이 빨라짐 (캐싱 효과 있음)")
        elif avg_subsequent_time > first_request_time * 1.2:
            print("⚠️  경고: 후속 요청이 느려짐 (매번 재초기화 의심)")
        else:
            print("ℹ️  정보: 요청 시간이 비슷함 (정상 범위)")
        
        # 시간 차이가 큰 경우 경고
        time_variance = max(subsequent_times) - min(subsequent_times)
        if time_variance > 2.0:
            print(f"⚠️  경고: 요청 시간 편차가 큼 ({time_variance:.3f}초)")
            print("   매번 무거운 작업이 실행되고 있을 가능성")
    
    # LLM 변경 결과 분석
    successful_llm = [r for r in different_llm_results if r["success"]]
    if successful_llm:
        llm_times = [r["total_time"] for r in successful_llm]
        print(f"\nLLM 모델별 요청 시간: {llm_times}")
        
        if any(t > 10 for t in llm_times):
            print("⚠️  경고: LLM 모델 변경 시 매우 느림 (재초기화 확인 필요)")
    
    # 6. 상세 결과 출력
    print("\n5. 상세 결과")
    print("-" * 60)
    print("동일 설정 연속 요청:")
    for result in same_config_results:
        status = "✅" if result["success"] else "❌"
        print(f"  {status} 요청 {result['request_num']}: {result['total_time']}초")
        if not result["success"]:
            print(f"     오류: {result['error']}")
    
    print("\nLLM 모델 변경 요청:")
    for result in different_llm_results:
        status = "✅" if result["success"] else "❌"
        print(f"  {status} {result['request_num']}: {result['total_time']}초")
        if not result["success"]:
            print(f"     오류: {result['error']}")
    
    return True

def main():
    """메인 함수"""
    print("MMS 추출기 API 성능 테스트를 시작합니다...")
    print("주의: API 서버가 먼저 실행되어 있어야 합니다.")
    print("서버 시작 명령: python api.py --host 0.0.0.0 --port 8080")
    print()
    
    input("API 서버가 실행 중이면 Enter를 눌러 테스트를 시작하세요...")
    
    try:
        success = test_api_performance()
        if success:
            print("\n" + "="*60)
            print("✅ 테스트가 완료되었습니다!")
            print("="*60)
        else:
            print("\n❌ 테스트 실행 중 오류가 발생했습니다.")
            return 1
            
    except KeyboardInterrupt:
        print("\n\n테스트가 사용자에 의해 중단되었습니다.")
        return 1
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류 발생: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 