#!/usr/bin/env python3
"""
MMS 추출기 Flask API 서비스
==========================

이 모듈은 MMS 광고 텍스트 추출기를 REST API 서비스로 제공합니다.

주요 기능:
- 단일 메시지 처리 API (/extract)
- 배치 메시지 처리 API (/extract/batch)
- 서비스 상태 확인 API (/health, /status)
- 사용 가능한 모델 목록 API (/models)
- 다양한 LLM 모델 지원 (Gemma, GPT, Claude)
- 실시간 설정 변경 지원

사용법:
    python api.py --host 0.0.0.0 --port 8000
    python api.py --test --message "테스트 메시지"
"""
import sys
import os
import json
import logging
import time
import argparse
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from config import settings

# 현재 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

# MMSExtractor 및 설정 모듈 임포트
try:
    from mms_extractor import MMSExtractor
    from config.settings import API_CONFIG, MODEL_CONFIG, PROCESSING_CONFIG
except ImportError as e:
    print(f"MMSExtractor 임포트 오류: {e}")
    print("mms_extractor.py가 같은 디렉토리에 있는지 확인하세요")
    sys.exit(1)

# Flask 앱 초기화
app = Flask(__name__)
CORS(app)  # CORS 활성화 (크로스 오리진 요청 허용)

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 전역 추출기 인스턴스 - 서버 시작 시 한 번만 로드
global_extractor = None

# CLI에서 설정된 데이터 소스 (전역 변수)
CLI_DATA_SOURCE = 'local'

def initialize_global_extractor(offer_info_data_src='local'):
    """
    전역 추출기 인스턴스를 서버 시작 시 한 번만 초기화
    
    이 함수는 무거운 데이터 로딩 작업(상품 정보, 임베딩 모델 등)을 
    서버 시작 시 미리 수행하여 API 요청 처리 시간을 단축합니다.
    
    Args:
        offer_info_data_src: 상품 정보 데이터 소스 ('local' 또는 'db')
    
    Returns:
        MMSExtractor: 초기화된 추출기 인스턴스
    """
    global global_extractor
    
    if global_extractor is None:
        logger.info(f"데이터 소스로 전역 추출기 초기화 중: {offer_info_data_src}")
        
        # 기본 설정으로 추출기 초기화 (요청 시 동적으로 변경 가능)
        global_extractor = MMSExtractor(
            model_path='./models/ko-sbert-nli',      # 임베딩 모델 경로
            data_dir='./data',                       # 데이터 디렉토리
            offer_info_data_src=offer_info_data_src, # 상품 정보 소스
            llm_model='gemma',                       # 기본 LLM (요청별 변경 가능)
            product_info_extraction_mode='nlp',     # 기본 상품 추출 모드
            entity_extraction_mode='logic'          # 기본 엔티티 매칭 모드
        )
        
        logger.info("전역 추출기 초기화 완료")
    
    return global_extractor

def get_configured_extractor(llm_model='gemma', product_info_extraction_mode='nlp', entity_matching_mode='logic'):
    """
    런타임 설정으로 전역 추출기 구성
    
    데이터 재로딩 없이 LLM 모델과 처리 모드만 변경하여 
    API 요청별로 다른 설정을 사용할 수 있습니다.
    
    Args:
        llm_model: 사용할 LLM 모델 ('gemma', 'gpt', 'claude')
        product_info_extraction_mode: 상품 정보 추출 모드 ('nlp', 'llm', 'rag')
        entity_matching_mode: 엔티티 매칭 모드 ('logic', 'llm')
    
    Returns:
        MMSExtractor: 구성된 추출기 인스턴스
    
    Raises:
        RuntimeError: 전역 추출기가 초기화되지 않은 경우
    """
    if global_extractor is None:
        raise RuntimeError("전역 추출기가 초기화되지 않았습니다. initialize_global_extractor()를 먼저 호출하세요.")
    
    # 데이터 재로딩 없이 런타임 설정만 업데이트
    global_extractor.llm_model_name = llm_model
    global_extractor.product_info_extraction_mode = product_info_extraction_mode
    global_extractor.entity_extraction_mode = entity_matching_mode
    
    # 모델이 변경된 경우 LLM 재초기화
    global_extractor._initialize_llm()
    
    return global_extractor

@app.route('/health', methods=['GET'])
def health_check():
    """
    서비스 상태 확인 엔드포인트
    
    서비스가 정상 작동하는지 확인하는 간단한 헬스체크 API입니다.
    로드밸런서나 모니터링 시스템에서 사용됩니다.
    
    Returns:
        JSON: 서비스 상태 정보
            - status: 서비스 상태 ("healthy")
            - service: 서비스 이름
            - version: 버전 정보
            - model: 사용 중인 기본 모델
            - timestamp: 응답 시간
    """
    return jsonify({
        "status": "healthy",
        "service": "MMS Extractor API",
        "version": "2.0.0",
        "model": "skt/gemma3-12b-it",
        "timestamp": time.time()
    })

@app.route('/models', methods=['GET'])
def list_models():
    """
    사용 가능한 모델 및 설정 옵션 목록 조회
    
    클라이언트가 API에서 지원하는 모든 옵션을 확인할 수 있도록 
    사용 가능한 모델과 설정값들을 반환합니다.
    
    Returns:
        JSON: 사용 가능한 설정 옵션들
            - available_llm_models: 지원하는 LLM 모델 목록
            - available_data_sources: 지원하는 데이터 소스
            - available_product_info_extraction_modes: 상품 추출 모드
            - available_entity_matching_modes: 엔티티 매칭 모드
            - features: 주요 기능 목록
    """
    return jsonify({
        "available_llm_models": ["gemma", "gpt", "claude"],
        "default_llm_model": "gemma",
        "available_data_sources": ["local", "db"],
        "default_data_source": "local",
        "available_product_info_extraction_modes": ["nlp", "llm", "rag"],
        "default_product_info_extraction_mode": "nlp",
        "available_entity_matching_modes": ["logic", "llm"],
        "default_entity_matching_mode": "logic",
        "features": [
            "Korean morphological analysis (Kiwi)",      # 한국어 형태소 분석
            "Embedding-based similarity search",         # 임베딩 기반 유사도 검색
            "Entity extraction and matching",            # 엔티티 추출 및 매칭
            "Program classification",                     # 프로그램 분류
            "Multiple LLM support (Gemma, GPT, Claude)" # 다중 LLM 지원
        ]
    })

@app.route('/extract', methods=['POST'])
def extract_message():
    """
    단일 MMS 메시지 정보 추출 API
    
    하나의 MMS 메시지에서 상품명, 채널 정보, 광고 목적 등을 추출합니다.
    
    Request Body (JSON):
        - message (required): 추출할 MMS 메시지 텍스트
        - llm_model (optional): 사용할 LLM 모델 (기본값: 'gemma')
        - offer_info_data_src (optional): 데이터 소스 (기본값: CLI 설정값)
        - product_info_extraction_mode (optional): 상품 추출 모드 (기본값: 'nlp')
        - entity_matching_mode (optional): 엔티티 매칭 모드 (기본값: 'logic')
    
    Returns:
        JSON: 추출 결과
            - success: 처리 성공 여부
            - result: 추출된 정보 (title, purpose, product, channel, pgm)
            - metadata: 처리 메타데이터 (처리 시간, 사용된 설정 등)
    
    HTTP Status Codes:
        - 200: 성공
        - 400: 잘못된 요청 (필수 필드 누락, 잘못된 파라미터 등)
        - 500: 서버 내부 오류
    """
    try:
        # 전역 추출기 초기화 상태 확인
        if global_extractor is None:
            return jsonify({"error": "추출기가 초기화되지 않았습니다. 서버 시작 중 오류가 발생했을 수 있습니다."}), 500
        
        # 요청 데이터 검증
        if not request.is_json:
            return jsonify({"error": "Content-Type은 application/json이어야 합니다"}), 400
        
        data = request.get_json()
        
        # 필수 필드 검증
        if 'message' not in data:
            return jsonify({"error": "필수 필드가 누락되었습니다: 'message'"}), 400
        
        message = data['message']
        if not message or not message.strip():
            return jsonify({"error": "메시지는 비어있을 수 없습니다"}), 400
        
        # 선택적 파라미터 추출 (기본값 사용)
        data_source = data.get('data_source', CLI_DATA_SOURCE)
        offer_info_data_src = data.get('offer_info_data_src', CLI_DATA_SOURCE)
        llm_model = data.get('llm_model', settings.ModelConfig.llm_model)
        product_info_extraction_mode = data.get('product_info_extraction_mode', settings.ProcessingConfig.product_info_extraction_mode)
        entity_matching_mode = data.get('entity_matching_mode', settings.ProcessingConfig.entity_extraction_mode)
        
        # 파라미터 유효성 검증
        valid_sources = ['local', 'db']
        if offer_info_data_src not in valid_sources:
            return jsonify({"error": f"잘못된 offer_info_data_src입니다. 사용 가능: {valid_sources}"}), 400
            
        valid_llm_models = ['gemma', 'gpt', 'claude']
        if llm_model not in valid_llm_models:
            return jsonify({"error": f"잘못된 llm_model입니다. 사용 가능: {valid_llm_models}"}), 400
            
        valid_product_modes = ['nlp', 'llm', 'rag']
        if product_info_extraction_mode not in valid_product_modes:
            return jsonify({"error": f"잘못된 product_info_extraction_mode입니다. 사용 가능: {valid_product_modes}"}), 400
            
        valid_entity_modes = ['logic', 'llm']
        if entity_matching_mode not in valid_entity_modes:
            return jsonify({"error": f"잘못된 entity_matching_mode입니다. 사용 가능: {valid_entity_modes}"}), 400
        
        # 구성된 추출기로 메시지 처리
        start_time = time.time()
        extractor = get_configured_extractor(llm_model, product_info_extraction_mode, entity_matching_mode)
        
        logger.info(f"데이터 소스로 메시지 처리 중: {offer_info_data_src}")
        result = extractor.process_message(message)
        processing_time = time.time() - start_time
        
        # 성공 응답 반환
        response = {
            "success": True,
            "result": result,
            "metadata": {
                "llm_model": llm_model,
                "offer_info_data_src": offer_info_data_src,
                "product_info_extraction_mode": product_info_extraction_mode,
                "entity_matching_mode": entity_matching_mode,
                "processing_time_seconds": round(processing_time, 3),
                "timestamp": time.time(),
                "message_length": len(message)
            }
        }
        
        logger.info(f"추출 완료: {processing_time:.3f}초")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"추출 중 오류 발생: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": time.time()
        }), 500

@app.route('/extract/batch', methods=['POST'])
def extract_batch():
    """
    다중 MMS 메시지 배치 처리 API
    
    여러 개의 MMS 메시지를 한 번에 처리하여 효율성을 높입니다.
    대량 데이터 처리나 배치 작업에 유용합니다.
    
    Request Body (JSON):
        - messages (required): 처리할 메시지 배열 (최대 100개)
        - llm_model (optional): 사용할 LLM 모델
        - offer_info_data_src (optional): 데이터 소스
        - product_info_extraction_mode (optional): 상품 추출 모드
        - entity_matching_mode (optional): 엔티티 매칭 모드
    
    Returns:
        JSON: 배치 처리 결과
            - success: 전체 배치 처리 성공 여부
            - results: 각 메시지별 처리 결과 배열
            - summary: 처리 요약 (총 개수, 성공/실패 개수)
            - metadata: 배치 처리 메타데이터
    
    HTTP Status Codes:
        - 200: 성공 (개별 메시지 실패가 있어도 배치 자체는 성공)
        - 400: 잘못된 요청
        - 500: 서버 내부 오류
    """
    try:
        # 전역 추출기 초기화 상태 확인
        if global_extractor is None:
            return jsonify({"error": "추출기가 초기화되지 않았습니다. 서버 시작 중 오류가 발생했을 수 있습니다."}), 500
        
        if not request.is_json:
            return jsonify({"error": "Content-Type은 application/json이어야 합니다"}), 400
        
        data = request.get_json()
        
        # 필수 필드 검증
        if 'messages' not in data:
            return jsonify({"error": "필수 필드가 누락되었습니다: 'messages'"}), 400
        
        messages = data['messages']
        if not isinstance(messages, list):
            return jsonify({"error": "'messages' 필드는 배열이어야 합니다"}), 400
        
        if len(messages) > 100:  # 배치 크기 제한
            return jsonify({"error": "배치당 최대 100개 메시지까지 처리 가능합니다"}), 400
        
        # 선택적 파라미터 추출
        offer_info_data_src = data.get('offer_info_data_src', CLI_DATA_SOURCE)
        llm_model = data.get('llm_model', settings.ModelConfig.llm_model)
        product_info_extraction_mode = data.get('product_info_extraction_mode', settings.ProcessingConfig.product_info_extraction_mode)
        entity_matching_mode = data.get('entity_matching_mode', settings.ProcessingConfig.entity_extraction_mode)
        
        # 파라미터 유효성 검증
        valid_sources = ['local', 'db']
        if offer_info_data_src not in valid_sources:
            return jsonify({"error": f"잘못된 offer_info_data_src입니다. 사용 가능: {valid_sources}"}), 400
            
        valid_llm_models = ['gemma', 'gpt', 'claude']
        if llm_model not in valid_llm_models:
            return jsonify({"error": f"잘못된 llm_model입니다. 사용 가능: {valid_llm_models}"}), 400
            
        valid_product_modes = ['nlp', 'llm', 'rag']
        if product_info_extraction_mode not in valid_product_modes:
            return jsonify({"error": f"잘못된 product_info_extraction_mode입니다. 사용 가능: {valid_product_modes}"}), 400
            
        valid_entity_modes = ['logic', 'llm']
        if entity_matching_mode not in valid_entity_modes:
            return jsonify({"error": f"잘못된 entity_matching_mode입니다. 사용 가능: {valid_entity_modes}"}), 400
        
        # 구성된 추출기 가져오기
        extractor = get_configured_extractor(llm_model, product_info_extraction_mode, entity_matching_mode)
        
        # 모든 메시지 순차 처리
        start_time = time.time()
        results = []
        
        for i, message in enumerate(messages):
            if not message or not message.strip():
                results.append({
                    "index": i,
                    "success": False,
                    "error": "빈 메시지입니다"
                })
                continue
            
            try:
                result = extractor.process_message(message)
                results.append({
                    "index": i,
                    "success": True,
                    "result": result
                })
            except Exception as e:
                logger.error(f"메시지 {i} 처리 중 오류: {e}")
                results.append({
                    "index": i,
                    "success": False,
                    "error": str(e)
                })
        
        processing_time = time.time() - start_time
        
        # 성공/실패 개수 집계
        successful = sum(1 for r in results if r["success"])
        failed = len(results) - successful
        
        response = {
            "success": True,
            "results": results,
            "summary": {
                "total_messages": len(messages),
                "successful": successful,
                "failed": failed
            },
            "metadata": {
                "llm_model": llm_model,
                "offer_info_data_src": offer_info_data_src,
                "product_info_extraction_mode": product_info_extraction_mode,
                "entity_matching_mode": entity_matching_mode,
                "processing_time_seconds": round(processing_time, 3),
                "timestamp": time.time()
            }
        }
        
        logger.info(f"배치 추출 완료: {successful}/{len(messages)}개 성공, {processing_time:.3f}초 소요")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"배치 추출 중 오류 발생: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": time.time()
        }), 500

@app.route('/status', methods=['GET'])
def get_status():
    """
    API 상태 및 추출기 정보 조회
    
    현재 서버의 상태와 추출기의 설정 정보를 제공합니다.
    디버깅이나 모니터링 목적으로 사용됩니다.
    
    Returns:
        JSON: 서버 및 추출기 상태 정보
            - status: 서버 실행 상태
            - extractor: 추출기 상태 정보
                - initialized: 초기화 여부
                - data_source: 현재 데이터 소스
                - current_llm_model: 현재 LLM 모델
                - current_product_mode: 현재 상품 추출 모드
                - current_entity_mode: 현재 엔티티 매칭 모드
            - timestamp: 응답 시간
    """
    global global_extractor
    
    # 추출기 상태 정보 수집
    extractor_status = {
        "initialized": global_extractor is not None,
        "data_source": CLI_DATA_SOURCE if global_extractor else None,
        "current_llm_model": global_extractor.llm_model_name if global_extractor else None,
        "current_product_mode": global_extractor.product_info_extraction_mode if global_extractor else None,
        "current_entity_mode": global_extractor.entity_extraction_mode if global_extractor else None
    }
    
    return jsonify({
        "status": "running",
        "extractor": extractor_status,
        "timestamp": time.time()
    })

@app.errorhandler(404)
def not_found(error):
    """404 에러 핸들러 - 존재하지 않는 엔드포인트 접근 시"""
    return jsonify({"error": "엔드포인트를 찾을 수 없습니다"}), 404

@app.errorhandler(500)
def internal_error(error):
    """500 에러 핸들러 - 서버 내부 오류 발생 시"""
    return jsonify({"error": "서버 내부 오류가 발생했습니다"}), 500

def main():
    """
    메인 함수 - CLI 사용을 위한 진입점
    
    커맨드라인 인자를 파싱하고 서버를 시작하거나 테스트 모드를 실행합니다.
    
    CLI 옵션:
        --host: 바인딩할 호스트 (기본값: 0.0.0.0)
        --port: 바인딩할 포트 (기본값: 8000)
        --debug: 디버그 모드 활성화
        --test: 테스트 모드 실행
        --message: 테스트할 메시지 (테스트 모드에서 사용)
        --offer-data-source: 데이터 소스 선택
        --product-info-extraction-mode: 상품 추출 모드 선택
        --entity-matching-mode: 엔티티 매칭 모드 선택
        --llm-model: 사용할 LLM 모델 선택
    
    사용 예시:
        # 서버 모드
        python api.py --host 0.0.0.0 --port 8000
        
        # 테스트 모드
        python api.py --test --message "테스트 메시지" --llm-model gpt
        
        # 데이터베이스 사용
        python api.py --offer-data-source db
    """
    global CLI_DATA_SOURCE
    
    # 커맨드라인 인자 파서 설정
    parser = argparse.ArgumentParser(description='MMS 추출기 API 서버')
    parser.add_argument('--host', default='0.0.0.0', help='바인딩할 호스트 주소')
    parser.add_argument('--port', type=int, default=8000, help='바인딩할 포트 번호')
    parser.add_argument('--debug', action='store_true', help='디버그 모드 활성화')
    parser.add_argument('--test', action='store_true', help='테스트 추출 실행')
    parser.add_argument('--message', type=str, help='테스트할 메시지')
    parser.add_argument('--offer-data-source', choices=['local', 'db'], default='local',
                       help='데이터 소스 선택 (local: CSV 파일, db: 데이터베이스)')
    parser.add_argument('--product-info-extraction-mode', choices=['nlp', 'llm' ,'rag'], default='nlp',
                       help='상품 정보 추출 모드 (nlp: 형태소분석, llm: LLM 기반, rag: 검색증강생성)')
    parser.add_argument('--entity-matching-mode', choices=['logic', 'llm'], default='llm',
                       help='엔티티 매칭 모드 (logic: 로직 기반, llm: LLM 기반)')
    parser.add_argument('--llm-model', choices=['gemma', 'gpt', 'claude'], default='gemma',
                       help='사용할 LLM 모델 (gemma: Gemma, gpt: GPT, claude: Claude)')
    
    args = parser.parse_args()
    
    # 전역 CLI 데이터 소스 설정
    CLI_DATA_SOURCE = args.offer_data_source
    logger.info(f"CLI 데이터 소스 설정: {CLI_DATA_SOURCE}")
    
    # 지정된 데이터 소스로 전역 추출기 초기화
    logger.info("전역 추출기 초기화 중...")
    initialize_global_extractor(CLI_DATA_SOURCE)
    
    if args.test:
        # 테스트 모드 실행
        logger.info("테스트 모드에서 실행 중...")
        
        # 제공된 메시지 또는 기본 샘플 메시지 사용
        message = args.message or """
        [SK텔레콤] ZEM폰 포켓몬에디션3 안내
        (광고)[SKT] 우리 아이 첫 번째 스마트폰, ZEM 키즈폰__#04 고객님, 안녕하세요!
        우리 아이 스마트폰 고민 중이셨다면, 자녀 스마트폰 관리 앱 ZEM이 설치된 SKT만의 안전한 키즈폰,
        ZEM폰 포켓몬에디션3으로 우리 아이 취향을 저격해 보세요!
        """
        
        try:
            logger.info(f"추출기 설정: llm_model={args.llm_model}, product_mode={args.product_info_extraction_mode}, entity_mode={args.entity_matching_mode}")
            extractor = get_configured_extractor(args.llm_model, args.product_info_extraction_mode, args.entity_matching_mode)
            
            if not message.strip():
                logger.info("텍스트가 제공되지 않아 샘플 메시지를 사용합니다...")
            
            logger.info("메시지 처리 중...")
            result = extractor.process_message(message)
            
            print("\n" + "="*60)
            print("추출 결과:")
            print("="*60)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            print("="*60)
            
            logger.info("처리가 성공적으로 완료되었습니다!")
            
        except Exception as e:
            logger.error(f"❌ 오류: {e}")
            sys.exit(1)
    else:
        # 서버 모드 실행
        logger.info(f"파싱된 인자: host={args.host}, port={args.port}, debug={args.debug}")
        logger.info("✅ 전역 추출기 초기화 완료, 요청 처리 준비됨")
        logger.info(f"MMS 추출기 API 서버를 {args.host}:{args.port}에서 시작합니다")
        logger.info("사용 가능한 엔드포인트:")
        logger.info("  GET  /health - 헬스체크")
        logger.info("  GET  /models - 사용 가능한 모델 목록")
        logger.info("  GET  /status - 서버 상태 조회")
        logger.info("  POST /extract - 단일 메시지 추출")
        logger.info("  POST /extract/batch - 다중 메시지 배치 추출")
        
        # Flask 설정 적용
        app.config['DEBUG'] = args.debug
        
        try:
            # 서버 시작 (리로더 비활성화, 스레딩 활성화)
            app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False, threaded=True)
        except Exception as e:
            logger.error(f"서버 시작 실패: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()
