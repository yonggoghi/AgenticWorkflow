
import sys
import os
import json
import logging

# Add dev directory to path
sys.path.append(os.path.abspath("/Users/yongwook/workspace/AgenticWorkflow/mms_extractor_dev"))

from mms_extractor import MMSExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)

def main():
    sample_msg = """
    (광고)[SKT] iPhone 신제품 구매 혜택 안내 __#04 고객님, 안녕하세요._SK텔레콤에서 iPhone 신제품 구매하면, 최대 22만 원 캐시백 이벤트에 참여하실 수 있습니다.__현대카드로 애플 페이도 더 편리하게 이용해 보세요.__▶ 현대카드 바로 가기: https://t-mms.kr/ais/#74_ _애플 페이 티머니 충전 쿠폰 96만 원, 샌프란시스코 왕복 항공권, 애플 액세서리 팩까지!_Lucky 1717 이벤트 응모하고 경품 당첨의 행운을 누려 보세요.__▶ 이벤트 자세히 보기: https://t-mms.kr/aiN/#74_ _■ 문의: SKT 고객센터(1558, 무료)__SKT와 함께해 주셔서 감사합니다.__무료 수신거부 1504',
    """
    
    try:
        print("Initializing MMSExtractor (Dev)...")
        extractor = MMSExtractor(extract_entity_dag=True)
        
        print("Processing message...")
        result = extractor.process_message(sample_msg)
        
        print("\n" + "="*60)
        print("Extraction Result (Dev):")
        print("="*60)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print("="*60)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
