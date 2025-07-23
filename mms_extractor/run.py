#!/usr/bin/env python3
"""
Standalone runner for MMS Extractor package.

Usage Examples:
    # Basic usage with default NLP mode
    python run.py --text "광고 메시지 텍스트"
    
    # Using different extraction modes
    python run.py --text "광고 메시지 텍스트" --extraction_mode rag
    python run.py --text "광고 메시지 텍스트" --extraction_mode llm --model claude_37
    
    # Using mock data for testing
    python run.py --mock-data --extraction_mode nlp
    
    # Verbose logging
    python run.py --verbose --extraction_mode rag
"""
import sys
import os
import json
import logging
import argparse
from pathlib import Path

# Add current directory to Python path for imports
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))
# sys.path.insert(0, str(current_dir.parent))  # Avoid importing from agentic files

def main():
    """Main function for standalone execution."""
    parser = argparse.ArgumentParser(description="MMS Extractor - Korean Marketing Message Analysis")
    parser.add_argument("--text", "-t", help="Text message to process")
    parser.add_argument("--mock-data", action="store_true", help="Use mock data for testing")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--model", "-m", default="gemma_3", 
                        choices=["gemma_3", "claude_37", "gpt_4", "claude_sonnet_4"],
                        help="LLM model to use (default: gemma_3)")
    parser.add_argument("--extraction_mode", "-e", default="nlp",
                        choices=["rag", "llm", "nlp"],
                        help="Extraction mode: 'nlp' (NLP entities + schema), 'rag' (RAG context), 'llm' (pure LLM) (default: nlp)")
    parser.add_argument("--model-loading-mode", default="auto",
                        choices=["auto", "local", "remote"],
                        help="Model loading mode: 'auto' (try local first, fallback to remote), 'local' (offline only), 'remote' (always download) (default: auto)")
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    try:
        # Import the MMS extractor and DataManager
        try:
            from mms_extractor.core.mms_extractor import MMSExtractor
            from mms_extractor.core.data_manager import DataManager
            from mms_extractor.config.settings import PROCESSING_CONFIG
        except ImportError:
            from core.mms_extractor import MMSExtractor
            from core.data_manager import DataManager
            from config.settings import PROCESSING_CONFIG
        
        logger.info("Initializing MMS Extractor...")
        logger.info(f"Selected LLM model: {args.model}")
        logger.info(f"Selected extraction mode: {args.extraction_mode}")
        logger.info(f"Selected model loading mode: {args.model_loading_mode}")
        
        # Set the extraction mode in the configuration
        PROCESSING_CONFIG.product_info_extraction_mode = args.extraction_mode
        
        # Set the model loading mode in the configuration
        try:
            from mms_extractor.config.settings import MODEL_CONFIG
        except ImportError:
            from config.settings import MODEL_CONFIG
        
        MODEL_CONFIG.model_loading_mode = args.model_loading_mode
        logger.info(f"Model loading mode set to: {MODEL_CONFIG.model_loading_mode}")
        logger.info(f"Description: {MODEL_CONFIG.get_loading_mode_description()}")
        
        # Initialize DataManager with mock data setting
        data_manager = DataManager(use_mock_data=args.mock_data)
        
        # Initialize MMS Extractor with DataManager and selected model
        extractor = MMSExtractor(data_manager=data_manager, model_name=args.model)
        
        # Load data
        logger.info("Loading data...")
        extractor.load_data()
        logger.info("MMS Extractor ready!")
        
        # Process message
        if args.text:
            message = args.text
        else:
            # Default test message
            message = """[SK텔레콤]추석맞이 추가할인 쿠폰 증정
(광고)[SKT]공식인증매장 고촌점 추석맞이 행사__안녕하세요 고객님!_고촌역 1번 출구 고촌파출소 방향 100m SK텔레콤 대리점 입니다._스마트폰 개통, 인터넷/TV 설치 시 조건 없이 추가 할인 행사를 진행합니다.__■삼성 갤럭시 Z플립5/Z폴드5는_  9월 내내 즉시개통 가능!!_1.갤럭시 워치6 개통 시 추가 할인_2.삼성케어+ 파손보장 1년권_3.삼성 정품 악세사리 30% 할인 쿠폰_4.정품 보호필름 1회 무료 부착__■새로운 아이폰15 출시 전_  아이폰14 재고 대방출!!_1.투명 범퍼 케이스 증정_2.방탄 유리 필름 부착_3.25W C타입 PD 충전기__여기에 5만원 추가 할인 적용!!__■기가인터넷+IPTV 가입 시_1.최대 36만원 상당 상품권 지급_2.스마트폰 개통 시 10만원 할인_3.매장 특별 사은품 지급_(특별 사은품은 매장 상황에 따라 변경될 수 있습니다)__■SKT 공식인증매장 고촌점_- 주소: 경기 김포시 고촌읍 장차로 3, SK텔레콤_- 연락처: 0507-1480-7833_- 네이버 예약하기: http://t-mms.kr/bSo/#74_- 매장 홈페이지: http://t-mms.kr/bSt/#74__■ 문의 : SKT 고객센터(1558, 무료)_무료 수신거부 1504_
"""
            logger.info("No text provided, using sample message...")
        
        logger.info(f"Processing message with {args.model} model in {args.extraction_mode} mode...")
        result = extractor.extract(message)
        
        print("\n" + "="*60)
        print("EXTRACTION RESULTS:")
        print(f"Model: {args.model} | Mode: {args.extraction_mode}")
        print("="*60)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("="*60)
        
        logger.info(f"Processing completed successfully using {args.model} model in {args.extraction_mode} mode!")
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Please install dependencies: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        if not args.mock_data:
            logger.info("💡 Try using --mock-data flag for testing without data files")
        sys.exit(1)

if __name__ == "__main__":
    main()
