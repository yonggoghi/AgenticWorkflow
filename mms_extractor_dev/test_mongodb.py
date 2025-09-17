#!/usr/bin/env python3
"""
MongoDB 통합 테스트 스크립트
MMS Extractor와 MongoDB 연동 기능 테스트
"""

import json
from datetime import datetime
from mongodb_utils import MongoDBManager, save_to_mongodb, test_mongodb_connection

def test_connection():
    """MongoDB 연결 테스트"""
    print("🔌 MongoDB 연결 테스트...")
    success = test_mongodb_connection()
    if success:
        print("✅ MongoDB 연결 성공!")
        return True
    else:
        print("❌ MongoDB 연결 실패!")
        return False

def test_save_sample_data():
    """샘플 데이터 저장 테스트"""
    print("\n📄 샘플 데이터 저장 테스트...")
    
    # 샘플 메시지
    message = "안녕하세요! SKT 5G 요금제 할인 이벤트입니다. 월 39,000원에 데이터 무제한 + 통화 무제한 혜택을 드립니다. 지금 가입하시면 첫 3개월 50% 할인! 자세한 내용은 114로 문의하세요."
    
    # 샘플 추출 결과
    extraction_result = {
        "success": True,
        "result": {
            "title": "SKT 5G 요금제 할인 이벤트",
            "purpose": "요금제 가입 유도",
            "product": "5G 요금제",
            "channel": "SMS",
            "program": "할인 이벤트",
            "offer_info": {
                "price": "월 39,000원",
                "discount": "첫 3개월 50% 할인",
                "benefits": ["데이터 무제한", "통화 무제한"]
            }
        },
        "metadata": {
            "processing_time_seconds": 2.5,
            "model_used": "claude"
        }
    }
    
    # 샘플 프롬프트 정보
    extraction_prompts = {
        "success": True,
        "prompts": {
            "main_extraction_prompt": {
                "title": "메인 정보 추출 프롬프트",
                "description": "MMS 메시지에서 기본 정보 추출",
                "content": "다음 MMS 메시지에서 제목, 목적, 상품 정보를 추출하세요...",
                "length": 500
            },
            "entity_extraction_prompt": {
                "title": "엔티티 추출 프롬프트", 
                "description": "개체명 인식 및 분류",
                "content": "메시지에서 인물, 장소, 조직 등의 개체명을 추출하세요...",
                "length": 300
            },
            "dag_extraction_prompt": {
                "title": "DAG 관계 추출 프롬프트",
                "description": "오퍼 관계 그래프 생성",
                "content": "추출된 정보들 간의 관계를 DAG 형태로 구성하세요...",
                "length": 400
            }
        },
        "settings": {
            "llm_model": "claude",
            "data_source": "local",
            "entity_matching_mode": "logic",
            "extract_entity_dag": True
        }
    }
    
    # MongoDB에 저장
    saved_id = save_to_mongodb(message, extraction_result, extraction_prompts)
    
    if saved_id:
        print(f"✅ 샘플 데이터 저장 성공! ID: {saved_id}")
        return saved_id
    else:
        print("❌ 샘플 데이터 저장 실패!")
        return None

def test_query_data():
    """저장된 데이터 조회 테스트"""
    print("\n🔍 저장된 데이터 조회 테스트...")
    
    manager = MongoDBManager()
    if not manager.connect():
        print("❌ MongoDB 연결 실패!")
        return
    
    # 최근 데이터 조회
    recent_data = manager.get_recent_extractions(limit=3)
    print(f"📊 최근 저장된 데이터 {len(recent_data)}건:")
    
    for i, doc in enumerate(recent_data, 1):
        print(f"\n{i}. ID: {doc['_id']}")
        print(f"   메시지 길이: {len(doc.get('message', ''))} 문자")
        print(f"   저장 시간: {doc.get('metadata', {}).get('timestamp')}")
        print(f"   성공 여부: {doc.get('metadata', {}).get('success')}")
        
        # 프롬프트 정보 확인
        main_prompt = doc.get('main_prompt')
        if main_prompt:
            print(f"   메인 프롬프트: {main_prompt.get('title', 'N/A')}")
    
    # 통계 정보 조회
    stats = manager.get_extraction_stats()
    if stats:
        print(f"\n📈 통계 정보:")
        print(f"   총 저장 건수: {stats.get('total_extractions', 0):,}")
        print(f"   성공 건수: {stats.get('successful_extractions', 0):,}")
        print(f"   성공률: {stats.get('success_rate', 0):.1f}%")
        print(f"   최근 24시간: {stats.get('recent_24h', 0):,}")
    
    manager.disconnect()

def main():
    """메인 테스트 함수"""
    print("🧪 MMS Extractor MongoDB 통합 테스트")
    print("=" * 50)
    
    # 1. 연결 테스트
    if not test_connection():
        print("\n❌ MongoDB 연결 실패로 테스트를 중단합니다.")
        print("MongoDB 서버가 실행 중인지 확인하세요.")
        return
    
    # 2. 샘플 데이터 저장 테스트
    saved_id = test_save_sample_data()
    
    # 3. 데이터 조회 테스트
    test_query_data()
    
    print("\n" + "=" * 50)
    print("✅ 모든 테스트 완료!")
    
    if saved_id:
        print(f"💡 샘플 데이터가 저장되었습니다. Streamlit 앱에서 확인해보세요.")
        print(f"   저장된 ID: {saved_id}")

if __name__ == "__main__":
    main()
