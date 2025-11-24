#!/usr/bin/env python3
"""
Quick Extractor API 테스트 스크립트
API로 사용할 때의 예시를 보여줍니다.
"""

import json
from quick_extractor import MessageInfoExtractor

def test_single_message_api():
    """단일 메시지 처리 API 테스트"""
    print("\n" + "="*60)
    print("📝 테스트 1: 단일 메시지 처리 (API 방식)")
    print("="*60)
    
    # 추출기 초기화 (LLM 사용)
    extractor = MessageInfoExtractor(csv_path=None, use_llm=True, llm_model='ax')
    
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
    
    # TextRank 방법으로 처리
    result_textrank = extractor.process_single_message(test_message, method='textrank')
    print("\n[TextRank 방법]")
    print(json.dumps(result_textrank, indent=2, ensure_ascii=False))
    
    # LLM 방법으로 처리
    result_llm = extractor.process_single_message(test_message, method='llm')
    print("\n[LLM 방법]")
    print(json.dumps(result_llm, indent=2, ensure_ascii=False))


def test_batch_file_api():
    """배치 파일 처리 API 테스트"""
    print("\n" + "="*60)
    print("📁 테스트 2: 배치 파일 처리 (API 방식)")
    print("="*60)
    
    # 추출기 초기화
    extractor = MessageInfoExtractor(csv_path='./data/reg_test.txt', use_llm=False)
    
    # 배치 파일 처리
    result = extractor.process_batch_file('./data/reg_test.txt', method='textrank')
    
    if result['success']:
        print(f"\n✅ 성공!")
        print(f"총 메시지: {result['data']['statistics']['total_messages']}개")
        print(f"수신거부 번호 추출: {result['data']['statistics']['with_unsubscribe_phone']}개")
        print(f"추출률: {result['data']['statistics']['extraction_rate']}%")
        print(f"\n처음 3개 메시지 결과:")
        for i, msg in enumerate(result['data']['messages'][:3], 1):
            print(f"\n  [{i}] 제목: {msg['title']}")
            print(f"      수신거부: {msg.get('unsubscribe_phone', 'N/A')}")
    else:
        print(f"\n❌ 실패: {result['error']}")


def test_api_error_handling():
    """에러 처리 테스트"""
    print("\n" + "="*60)
    print("⚠️  테스트 3: 에러 처리")
    print("="*60)
    
    # 존재하지 않는 파일 처리
    extractor = MessageInfoExtractor(csv_path='./nonexistent.csv', use_llm=False)
    result = extractor.process_batch_file('./nonexistent.csv', method='textrank')
    
    print(f"\n존재하지 않는 파일 처리 결과:")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🧪 Quick Extractor API 테스트")
    print("="*60)
    
    # 테스트 1: 단일 메시지
    test_single_message_api()
    
    # 테스트 2: 배치 파일
    test_batch_file_api()
    
    # 테스트 3: 에러 처리
    test_api_error_handling()
    
    print("\n" + "="*60)
    print("✅ 모든 테스트 완료")
    print("="*60 + "\n")

