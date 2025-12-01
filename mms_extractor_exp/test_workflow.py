"""
Workflow 테스트 스크립트
======================

Workflow 기반으로 리팩토링된 MMSExtractor를 테스트합니다.
"""

import sys
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_workflow():
    """Workflow 테스트"""
    try:
        logger.info("=" * 60)
        logger.info("Workflow 테스트 시작")
        logger.info("=" * 60)
        
        # MMSExtractor 임포트 및 초기화
        from mms_extractor import MMSExtractor
        
        logger.info("MMSExtractor 초기화 중...")
        extractor = MMSExtractor(
            llm_model='gen',  # Gemini Pro
            entity_extraction_mode='llm',
            extract_entity_dag=False
        )
        
        # 테스트 메시지
        test_message = """
[SKT] T 우주패스 쇼핑 출시! 
지금 링크를 눌러 가입하면 첫 달 1,000원에 이용 가능합니다. 
가입 고객 전원에게 11번가 포인트 3,000P와 아마존 무료배송 쿠폰을 드립니다.
문의: 114
        """
        
        logger.info(f"테스트 메시지: {test_message[:100]}...")
        
        # 메시지 처리
        logger.info("\n메시지 처리 시작...")
        result = extractor.process_message(test_message)
        
        # 결과 확인
        logger.info("\n" + "=" * 60)
        logger.info("테스트 결과")
        logger.info("=" * 60)
        
        extracted = result.get('ext_result', {})
        
        logger.info(f"✅ 제목: {extracted.get('title', 'N/A')}")
        logger.info(f"✅ 목적: {extracted.get('purpose', [])}")
        logger.info(f"✅ 상품 수: {len(extracted.get('product', []))}개")
        
        products = extracted.get('product', [])
        if products:
            logger.info(f"✅ 상품 목록:")
            for i, product in enumerate(products[:3], 1):
                logger.info(f"   {i}. {product.get('item_nm', 'N/A')}")
        
        logger.info(f"✅ 채널 수: {len(extracted.get('channel', []))}개")
        logger.info(f"✅ 프로그램 수: {len(extracted.get('pgm', []))}개")
        
        # 성공 여부 판단
        if extracted and extracted.get('title'):
            logger.info("\n" + "=" * 60)
            logger.info("✅ Workflow 테스트 성공!")
            logger.info("=" * 60)
            return True
        else:
            logger.error("\n" + "=" * 60)
            logger.error("❌ Workflow 테스트 실패: 결과가 비어있습니다")
            logger.error("=" * 60)
            return False
            
    except Exception as e:
        logger.error(f"\n❌ Workflow 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_workflow()
    sys.exit(0 if success else 1)
