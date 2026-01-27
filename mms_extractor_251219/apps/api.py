#!/usr/bin/env python3
"""
MMS 추출기 REST API 서비스 (MMS Extractor API Service)
================================================================

🎯 개요
-------
이 모듈은 MMS 광고 텍스트 분석 시스템을 RESTful API 서비스로 제공하는
엔터프라이즈급 웹 서비스입니다. Flask 기반으로 구축되어 고성능과 확장성을 보장합니다.

🔗 의존성
---------
**사용하는 모듈:**
- `core.mms_extractor`: MMSExtractor 메인 엔진
- `config.settings`: API, 모델, 처리 설정
- `flask`: 웹 프레임워크
- `flask_cors`: CORS 지원

**아키텍처:**
```
Client Request
    ↓
Flask API Server (api.py)
    ↓
global_extractor (MMSExtractor)
    ↓
WorkflowEngine → 9 Steps
    ↓
JSON Response
```

🚀 핵심 기능
-----------
• **단일 메시지 처리**: `POST /extract` - 실시간 메시지 분석
• **배치 처리**: `POST /extract/batch` - 대량 메시지 일괄 처리
• **DAG 추출**: `POST /dag` - 엔티티 관계 그래프 생성
• **서비스 모니터링**: `GET /health`, `GET /status` - 서비스 상태 및 성능 지표
• **모델 관리**: `GET /models` - 사용 가능한 LLM 모델 목록
• **다중 LLM 지원**: OpenAI GPT, Anthropic Claude, Gemini, AX 등
• **실시간 설정**: 런타임 중 설정 변경 지원

📊 성능 특징
-----------
• **고성능**: 전역 추출기 재사용으로 초기화 오버헤드 제거
• **확장성**: 마이크로서비스 아키텍처 지원
• **안정성**: 포괄적인 에러 처리 및 로깅
• **보안**: CORS 설정 및 입력 검증

🚀 사용법
---------
```bash
# 기본 서비스 시작
python api.py --host 0.0.0.0 --port 8000

# 특정 LLM 모델로 시작
python api.py --llm-model ax --port 8080

# 엔티티 매칭 모드 설정
python api.py --entity-matching-mode llm

# 데이터 소스 지정
python api.py --data-source db

# 테스트 모드
python api.py --test --message "샘플 MMS 텍스트"
```

🏗️ API 엔드포인트
--------------

### 메인 추출 API
- **POST /extract**: 단일 메시지 분석
  - Request: `{"message": "...", "llm_model": "ax", ...}`
  - Response: `{"success": true, "result": {...}, "metadata": {...}}`

- **POST /extract/batch**: 배치 메시지 분석
  - Request: `{"messages": ["...", "..."], ...}`
  - Response: `{"success": true, "results": [...], "summary": {...}}`

### DAG 추출 API
- **POST /dag**: Entity DAG 추출
  - Request: `{"message": "...", "llm_models": ["ax", "gpt"]}`
  - Response: `{"dag_section": "...", "entities": [...], ...}`

- **GET /dag_images/<filename>**: DAG 이미지 파일 제공

### Quick Extractor API
- **POST /quick/extract**: 제목/수신거부 번호 추출 (단일)
- **POST /quick/extract/batch**: 제목/수신거부 번호 추출 (배치)

### 모니터링 API
- **GET /health**: 서비스 상태 확인
  - Response: `{"status": "healthy", "service": "MMS Extractor API", ...}`

- **GET /status**: 상세 성능 지표
- **GET /models**: 사용 가능한 모델 목록
  - Response: `{"available_llm_models": ["ax", "gpt", ...], ...}`

📈 모니터링
-----------
- **로깅**: 회전 로그 파일 (api_server.log, 5MB x 10개)
- **성능 메트릭스**: 처리 시간, 성공/실패율
- **에러 추적**: 상세 스택 트레이스
- **자원 사용량**: 메모리, CPU 모니터링

💡 사용 예시
-----------
```python
import requests

# 1. 단일 메시지 추출
response = requests.post('http://localhost:8000/extract', json={
    "message": "아이폰 17 구매 시 최대 22만원 캐시백",
    "llm_model": "ax",
    "entity_matching_mode": "llm"
})
result = response.json()

# 2. 배치 처리
response = requests.post('http://localhost:8000/extract/batch', json={
    "messages": ["메시지1", "메시지2", "메시지3"],
    "llm_model": "ax"
})
results = response.json()

# 3. DAG 추출
response = requests.post('http://localhost:8000/dag', json={
    "message": "T world 앱 접속 후 퀴즈 참여하면 올리브영 기프티콘 획득",
    "llm_models": ["ax", "gpt"]
})
dag_result = response.json()

# 4. 헬스체크
response = requests.get('http://localhost:8000/health')
health = response.json()
```

📝 참고사항
----------
- 전역 추출기는 서버 시작 시 한 번만 초기화됨
- 런타임 설정 변경은 데이터 재로딩 없이 가능
- MongoDB 저장은 선택적 (save_to_mongodb 파라미터)
- DAG 이미지는 ./dag_images/ 디렉토리에 저장
- 프롬프트는 스레드 로컬 저장소에 캐시됨

"""
# =============================================================================
# 필수 라이브러리 임포트
# =============================================================================
import sys
import os
# Add parent directory to path to allow imports from core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import logging
import time
import argparse
import warnings
import atexit
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from config import settings

# =============================================================================
# 경고 메시지 억제 (로그 노이즈 감소)
# =============================================================================
# joblib과 multiprocessing 관련 경고 억제
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")
warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing") 
warnings.filterwarnings("ignore", message=".*resource_tracker.*")
warnings.filterwarnings("ignore", message=".*leaked.*")

# =============================================================================
# 경로 설정 및 모듈 임포트 준비
# =============================================================================
# 현재 디렉토리를 Python 경로에 추가 (로컬 모듈 임포트를 위해)
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

# =============================================================================
# 핵심 모듈 임포트 (오류 처리 포함)
# =============================================================================
# MMS 추출기 및 설정 모듈 임포트
try:
    from core.mms_extractor import MMSExtractor, process_message_worker, process_messages_batch, save_result_to_mongodb_if_enabled
    from config.settings import API_CONFIG, MODEL_CONFIG, PROCESSING_CONFIG
    # Lazy import for DAG extractor
    from core.entity_dag_extractor import DAGParser, extract_dag, llm_ax, llm_gem, llm_cld, llm_gen, llm_gpt
    from quick_extractor import MessageInfoExtractor  # Quick Extractor 임포트
except ImportError as e:
    print(f"❌ 모듈 임포트 오류: {e}")
    print("📝 mms_extractor.py가 같은 디렉토리에 있는지 확인하세요")
    print("📝 config/ 디렉토리와 설정 파일들을 확인하세요")
    print("📝 quick_extractor.py가 같은 디렉토리에 있는지 확인하세요")
    sys.exit(1)

# Flask 앱 초기화
app = Flask(__name__)
CORS(app)  # CORS 활성화 (크로스 오리진 요청 허용)

def cleanup_resources():
    """리소스 정리 함수 - 프로그램 종료 시 호출"""
    try:
        import gc
        import multiprocessing
        
        # 가비지 컬렉션 실행
        gc.collect()
        
        # 멀티프로세싱 리소스 정리
        if hasattr(multiprocessing, 'active_children'):
            for child in multiprocessing.active_children():
                try:
                    child.terminate()
                    child.join(timeout=1)
                except:
                    pass
                    
        print("리소스 정리 완료")
    except Exception as e:
        print(f"리소스 정리 중 오류: {e}")

# 프로그램 종료 시 리소스 정리
atexit.register(cleanup_resources)

# 로깅 설정 - 콘솔과 파일 모두에 출력
import logging.handlers

# 로그 디렉토리 생성
log_dir = Path(__file__).parent.parent / 'logs'
log_dir.mkdir(exist_ok=True)

# API 전용 로그 파일 경로 - 실시간 API 요청/응답 로그
log_file = log_dir / 'api_server.log'

# 로거 설정
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 포맷터 설정 - 모듈명 포함하여 로그 출처 명확화
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 콘솔 핸들러
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# API 전용 파일 핸들러 (회전 로그 - 5MB씩 최대 10개 파일, 짧은 보존기간)
file_handler = logging.handlers.RotatingFileHandler(
    log_file, 
    maxBytes=5*1024*1024,   # 5MB (API 로그는 상대적으로 작음)
    backupCount=10,         # 더 많은 파일 보존 (실시간 모니터링용)
    encoding='utf-8'
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

# 루트 로거에만 핸들러 추가하여 모든 하위 로거의 로그를 처리
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(console_handler)
root_logger.addHandler(file_handler)

# 기존 핸들러 제거 (중복 방지)
root_logger.handlers = [console_handler, file_handler]

# 개별 로거들은 루트 로거로 전파하도록 설정 (핸들러 중복 등록 방지)
logger.setLevel(logging.INFO)
mms_logger = logging.getLogger('mms_extractor')
mms_logger.setLevel(logging.INFO)

# 전파 설정 확인 (기본값이 True이므로 명시적으로 설정)
logger.propagate = True
mms_logger.propagate = True

# 전역 추출기 인스턴스 - 서버 시작 시 한 번만 로드
global_extractor = None

# 전역 Quick Extractor 인스턴스 (제목/수신거부 번호 추출용)
global_quick_extractor = None

# CLI에서 설정된 데이터 소스 (전역 변수)
CLI_DATA_SOURCE = 'local'

def initialize_global_extractor(offer_info_data_src='db'):
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
        
        # 기본 설정으로 추출기 초기화 (CLI와 동일한 기본값 사용)
        global_extractor = MMSExtractor(
            model_path='./models/ko-sbert-nli',      # 임베딩 모델 경로
            data_dir='./data',                       # 데이터 디렉토리
            offer_info_data_src=offer_info_data_src, # 상품 정보 소스
            llm_model='gemini',                      # 기본 LLM: Gemini (CLI와 동일)
            product_info_extraction_mode='llm',      # 기본 상품 추출 모드: LLM (CLI와 동일)
            entity_extraction_mode='llm',            # 기본 엔티티 매칭 모드: LLM (CLI와 동일)
            extract_entity_dag=True,
            entity_extraction_context_mode='dag'     # 기본 컨텍스트 모드: DAG
        )
        
        logger.info("전역 추출기 초기화 완료")
    
    return global_extractor

def initialize_quick_extractor(use_llm=False, llm_model='ax'):
    """
    전역 Quick Extractor 인스턴스를 초기화
    
    Args:
        use_llm: LLM 사용 여부
        llm_model: 사용할 LLM 모델 ('ax', 'gpt', 'claude', 'gemini' 등)
    
    Returns:
        MessageInfoExtractor: 초기화된 Quick Extractor 인스턴스
    """
    global global_quick_extractor
    
    if global_quick_extractor is None:
        logger.info(f"Quick Extractor 초기화 중... (LLM: {use_llm}, 모델: {llm_model})")
        
        # Quick Extractor 초기화 (csv_path는 API에서 필요 없음)
        global_quick_extractor = MessageInfoExtractor(
            csv_path=None,
            use_llm=use_llm,
            llm_model=llm_model
        )
        
        logger.info("Quick Extractor 초기화 완료")
    
    return global_quick_extractor

def get_configured_quick_extractor(use_llm=False, llm_model='ax'):
    """
    런타임 설정으로 Quick Extractor 구성
    
    Args:
        use_llm: LLM 사용 여부
        llm_model: 사용할 LLM 모델
    
    Returns:
        MessageInfoExtractor: 구성된 Quick Extractor 인스턴스
    """
    if global_quick_extractor is None:
        return initialize_quick_extractor(use_llm, llm_model)
    
    # LLM 설정이 변경된 경우 재초기화
    if use_llm != global_quick_extractor.use_llm or llm_model != global_quick_extractor.llm_model_name:
        logger.info(f"Quick Extractor 재설정 중... (LLM: {use_llm}, 모델: {llm_model})")
        return initialize_quick_extractor(use_llm, llm_model)
    
    return global_quick_extractor

def get_configured_extractor(llm_model='gemini', product_info_extraction_mode='llm', entity_matching_mode='llm', entity_llm_model='ax', extract_entity_dag=True, entity_extraction_context_mode='dag'):
    """
    런타임 설정으로 전역 추출기 구성
    
    데이터 재로딩 없이 LLM 모델과 처리 모드만 변경하여 
    API 요청별로 다른 설정을 사용할 수 있습니다.
    
    Args:
        llm_model: 메인 프롬프트에 사용할 LLM 모델 ('gemma', 'ax', 'claude', 'gpt', 'gemini')
        product_info_extraction_mode: 상품 정보 추출 모드 ('nlp', 'llm', 'rag')
        entity_matching_mode: 엔티티 매칭 모드 ('logic', 'llm')
        entity_llm_model: 엔티티 추출에 사용할 LLM 모델 ('gemma', 'ax', 'claude', 'gpt', 'gemini')
        entity_extraction_context_mode: 엔티티 추출 컨텍스트 모드 ('dag', 'pairing', 'none')
    
    Returns:
        MMSExtractor: 구성된 추출기 인스턴스
    
    Raises:
        RuntimeError: 전역 추출기가 초기화되지 않은 경우
    """
    if global_extractor is None:
        raise RuntimeError("전역 추출기가 초기화되지 않았습니다. initialize_global_extractor()를 먼저 호출하세요.")
    
    # 현재 설정과 비교하여 변경된 경우만 업데이트
    current_llm_model = getattr(global_extractor, 'llm_model_name', None)
    llm_model_changed = current_llm_model != llm_model
    
    # 데이터 재로딩 없이 런타임 설정만 업데이트
    global_extractor.llm_model_name = llm_model
    global_extractor.entity_llm_model_name = entity_llm_model
    global_extractor.product_info_extraction_mode = product_info_extraction_mode
    global_extractor.entity_extraction_mode = entity_matching_mode
    global_extractor.extract_entity_dag = extract_entity_dag
    global_extractor.entity_extraction_context_mode = entity_extraction_context_mode
    
    # ResultBuilder의 llm_model도 업데이트
    if hasattr(global_extractor, 'result_builder'):
        global_extractor.result_builder.llm_model = entity_llm_model
    
    # LLM 모델이 실제로 변경된 경우에만 재초기화
    if llm_model_changed:
        logger.info(f"LLM 모델이 {current_llm_model} -> {llm_model}로 변경됨. 재초기화 중...")
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
        "available_llm_models": ["gemma", "ax", "claude", "gemini"],
        "default_llm_model": "ax",
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
        - llm_model (optional): 사용할 LLM 모델 (기본값: 'ax')
        - offer_info_data_src (optional): 데이터 소스 (기본값: CLI 설정값)
        - product_info_extraction_mode (optional): 상품 추출 모드 (기본값: 'nlp')
        - entity_matching_mode (optional): 엔티티 매칭 모드 (기본값: 'logic')
        - extract_entity_dag (optional): 엔티티 DAG 추출 여부 (기본값: True)
                                         True일 경우 메시지에서 엔티티 간 관계를 DAG 형태로 추출하고
                                         시각적 다이어그램도 함께 생성합니다.
        - result_type (optional): 추출 결과 타입 (기본값: 'ext')
        - save_to_mongodb (optional): MongoDB 저장 여부 (기본값: True)
    
    Returns:
        JSON: 추출 결과
            - success: 처리 성공 여부
            - result: 추출된 정보 (title, purpose, product, channel, pgm)
                     extract_entity_dag=True인 경우 entity_dag 필드도 포함
            - metadata: 처리 메타데이터 (처리 시간, 사용된 설정, DAG 추출 여부 등)
    
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
        entity_llm_model = data.get('entity_llm_model', 'ax')
        product_info_extraction_mode = data.get('product_info_extraction_mode', settings.ProcessingConfig.product_info_extraction_mode)
        entity_matching_mode = data.get('entity_matching_mode', settings.ProcessingConfig.entity_extraction_mode)
        extract_entity_dag = data.get('extract_entity_dag', True)
        entity_extraction_context_mode = data.get('entity_extraction_context_mode', 'dag')
        save_to_mongodb = data.get('save_to_mongodb', True)
        result_type = data.get('result_type', 'ext')
        message_id = data.get('message_id', '#')  # 메시지 ID (기본값: '#')

        data['save_to_mongodb'] = save_to_mongodb
        data['result_type'] = result_type
        data['processing_mode'] = 'single'
        
        # DAG 추출 요청 로깅
        if extract_entity_dag:
            logger.info(f"🎯 DAG 추출 요청됨 - LLM: {llm_model}, 메시지 길이: {len(message)}자")
        
        # 메시지 ID 로깅
        if message_id != '#':
            logger.info(f"📋 메시지 ID: {message_id}")
        
        # 파라미터 유효성 검증
        valid_sources = ['local', 'db']
        if offer_info_data_src not in valid_sources:
            return jsonify({"error": f"잘못된 offer_info_data_src입니다. 사용 가능: {valid_sources}"}), 400
            
        valid_llm_models = ['gemma', 'ax', 'claude', 'gemini']
        if llm_model not in valid_llm_models:
            return jsonify({"error": f"잘못된 llm_model입니다. 사용 가능: {valid_llm_models}"}), 400
            
        valid_product_modes = ['nlp', 'llm', 'rag']
        if product_info_extraction_mode not in valid_product_modes:
            return jsonify({"error": f"잘못된 product_info_extraction_mode입니다. 사용 가능: {valid_product_modes}"}), 400
            
        valid_entity_modes = ['logic', 'llm']
        if entity_matching_mode not in valid_entity_modes:
            return jsonify({"error": f"잘못된 entity_matching_mode입니다. 사용 가능: {valid_entity_modes}"}), 400
        
        # DAG 추출 기능 활성화
        # extract_entity_dag=True인 경우:
        # 1. 메시지에서 엔티티 간 관계를 DAG(Directed Acyclic Graph) 형태로 추출
        # 2. NetworkX를 사용하여 그래프 구조 생성
        # 3. Graphviz를 통해 시각적 다이어그램 생성 (./dag_images/ 디렉토리에 저장)
        # 4. 결과의 entity_dag 필드에 DAG 텍스트 표현 포함
        
        # 구성된 추출기로 메시지 처리 (프롬프트 캡처 포함)
        start_time = time.time()
        extractor = get_configured_extractor(llm_model, product_info_extraction_mode, entity_matching_mode, entity_llm_model, extract_entity_dag, entity_extraction_context_mode)
        
        logger.info(f"데이터 소스로 메시지 처리 중: {offer_info_data_src}")
        
        # 프롬프트 캡처를 위한 스레드 로컬 저장소 초기화
        import threading
        current_thread = threading.current_thread()
        current_thread.stored_prompts = {}
        
        # DAG 추출 여부에 따라 병렬 처리 또는 단일 처리
        if extract_entity_dag:
            logger.info("DAG 추출과 함께 순차 처리 시작")
            result = process_message_worker(extractor, message, extract_dag=True, message_id=message_id)
        else:
            result = extractor.process_message(message, message_id=message_id)
            result['ext_result']['entity_dag'] = []
            result['raw_result']['entity_dag'] = []  # DAG 추출하지 않은 경우 빈 배열

        if save_to_mongodb:
            logger.info("MongoDB 저장 중...")
            saved_id = save_result_to_mongodb_if_enabled(message, result, data, extractor)
            if saved_id:
                logger.info("MongoDB 저장 완료!")

        if result_type == 'raw':
            result = result.get('raw_result', {})
        else:
            result = result.get('ext_result', {})
            
        processing_time = time.time() - start_time
        
        # 캡처된 프롬프트들 가져오기
        captured_prompts = getattr(current_thread, 'stored_prompts', {})
        logger.info(f"추출 과정에서 캡처된 프롬프트: {len(captured_prompts)}개")
        
        # DAG 추출 결과 검증 및 로깅
        # entity_dag 필드는 추출된 엔티티 간의 관계를 텍스트로 표현한 것
        # 예: "(고객:가입) -[하면]-> (혜택:수령)"
        if extract_entity_dag and 'entity_dag' in result:
            dag_length = len(result['entity_dag']) if result['entity_dag'] else 0
            if dag_length > 0:
                logger.info(f"✅ DAG 추출 성공 - 길이: {dag_length}자")
                logger.info(f"DAG 내용 미리보기: {result['entity_dag'][:100]}...")
            else:
                logger.warning("⚠️ DAG 추출 요청되었으나 결과가 비어있음")
        
        # 성공 응답 반환 (프롬프트 포함)
        response = {
            "success": True,
            "result": result,
            "metadata": {
                "llm_model": llm_model,
                "offer_info_data_src": offer_info_data_src,
                "product_info_extraction_mode": product_info_extraction_mode,
                "entity_matching_mode": entity_matching_mode,
                "extract_entity_dag": extract_entity_dag,
                "processing_time_seconds": round(processing_time, 3),
                "timestamp": time.time(),
                "message_length": len(message)
            },
            "prompts": {
                "success": True,
                "prompts": captured_prompts,
                "settings": {
                    "llm_model": llm_model,
                    "offer_info_data_src": offer_info_data_src,
                    "product_info_extraction_mode": product_info_extraction_mode,
                    "entity_matching_mode": entity_matching_mode,
                    "extract_entity_dag": extract_entity_dag
                },
                "message_info": {
                    "length": len(message),
                    "preview": message[:200] + "..." if len(message) > 200 else message
                },
                "timestamp": time.time()
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
        - extract_entity_dag (optional): 엔티티 DAG 추출 여부 (기본값: True)
        - max_workers (optional): 병렬 처리 워커 수 (기본값: CPU 코어 수)
    
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
        entity_llm_model = data.get('entity_llm_model', 'ax')
        product_info_extraction_mode = data.get('product_info_extraction_mode', settings.ProcessingConfig.product_info_extraction_mode)
        entity_matching_mode = data.get('entity_matching_mode', settings.ProcessingConfig.entity_extraction_mode)
        extract_entity_dag = data.get('extract_entity_dag', True)
        entity_extraction_context_mode = data.get('entity_extraction_context_mode', 'dag')
        max_workers = data.get('max_workers', None)
        save_to_mongodb = data.get('save_to_mongodb', True)
        result_type = data.get('result_type', 'ext')

        data['save_to_mongodb'] = save_to_mongodb
        data['result_type'] = result_type
        data['processing_mode'] = 'batch'
        
        # 파라미터 유효성 검증
        valid_sources = ['local', 'db']
        if offer_info_data_src not in valid_sources:
            return jsonify({"error": f"잘못된 offer_info_data_src입니다. 사용 가능: {valid_sources}"}), 400
            
        valid_llm_models = ['gemma', 'ax', 'claude', 'gemini']
        if llm_model not in valid_llm_models:
            return jsonify({"error": f"잘못된 llm_model입니다. 사용 가능: {valid_llm_models}"}), 400
            
        valid_product_modes = ['nlp', 'llm', 'rag']
        if product_info_extraction_mode not in valid_product_modes:
            return jsonify({"error": f"잘못된 product_info_extraction_mode입니다. 사용 가능: {valid_product_modes}"}), 400
            
        valid_entity_modes = ['logic', 'llm']
        if entity_matching_mode not in valid_entity_modes:
            return jsonify({"error": f"잘못된 entity_matching_mode입니다. 사용 가능: {valid_entity_modes}"}), 400
        
        # 구성된 추출기 가져오기
        extractor = get_configured_extractor(llm_model, product_info_extraction_mode, entity_matching_mode, entity_llm_model, extract_entity_dag, entity_extraction_context_mode)
        
        # DAG 추출 요청 로깅
        if extract_entity_dag:
            logger.info(f"🎯 배치 DAG 추출 요청됨 - {len(messages)}개 메시지, 워커: {max_workers}")
        
        # 멀티프로세스 배치 처리
        start_time = time.time()
        
        # 프롬프트 캡처를 위한 스레드 로컬 저장소 초기화
        import threading
        current_thread = threading.current_thread()
        current_thread.stored_prompts = {}
        
        # 빈 메시지 필터링 및 message_id 추출
        valid_messages = []
        message_ids = []
        message_indices = []
        for i, msg_item in enumerate(messages):
            # 메시지가 문자열이거나 딕셔너리일 수 있음
            if isinstance(msg_item, dict):
                message = msg_item.get('message', '')
                message_id = msg_item.get('message_id', '#')
            else:
                message = msg_item
                message_id = '#'
            
            if message and message.strip():
                valid_messages.append(message)
                message_ids.append(message_id)
                message_indices.append(i)
        
        logger.info(f"배치 처리 시작: {len(valid_messages)}/{len(messages)}개 유효한 메시지")
        
        # MongoDB 저장 카운터 초기화
        saved_count = 0
        
        try:
            # 각 메시지를 message_id와 함께 처리
            batch_results = []
            for message, message_id in zip(valid_messages, message_ids):
                if extract_entity_dag:
                    result = process_message_worker(
                        extractor, 
                        message, 
                        extract_dag=True,
                        message_id=message_id
                    )
                else:
                    result = extractor.process_message(message, message_id=message_id)
                    result['ext_result']['entity_dag'] = []
                    result['raw_result']['entity_dag'] = []
                
                batch_results.append(result)
            
            # 결과를 원래 인덱스와 매핑 및 MongoDB 저장
            results = []
            valid_result_idx = 0
            
            for i, msg_item in enumerate(messages):
                # 메시지가 문자열이거나 딕셔너리일 수 있음
                if isinstance(msg_item, dict):
                    message_text = msg_item.get('message', '')
                else:
                    message_text = msg_item
                
                if not message_text or not message_text.strip():
                    results.append({
                        "index": i,
                        "success": False,
                        "error": "빈 메시지입니다"
                    })
                else:
                    if valid_result_idx < len(batch_results):
                        batch_result = batch_results[valid_result_idx]

                        # result_type에 따라 결과 선택
                        if result_type == 'raw':
                            result_data = batch_result.get('raw_result', {})
                        else:
                            result_data = batch_result.get('ext_result', {})

                        # print("=" * 50 + " batch_result " + "=" * 50)
                        # print(batch_result)
                        # print("=" * 50 + " batch_result " + "=" * 50)
                        
                        if result_data.get('error'):
                            results.append({
                                "index": i,
                                "success": False,
                                "error": result_data['error']
                            })
                        else:
                            # MongoDB 저장 (배치 처리에서는 각 메시지별로 저장)
                            if save_to_mongodb:
                                try:
                                    saved_id = save_result_to_mongodb_if_enabled(message_text, batch_result, data, extractor)
                                    if saved_id:
                                        saved_count += 1
                                        logger.debug(f"메시지 {i} MongoDB 저장 완료 (ID: {saved_id[:8]}...)")
                                except Exception as e:
                                    logger.warning(f"메시지 {i} MongoDB 저장 실패: {str(e)}")
                                
                            results.append({
                                "index": i,
                                "success": True,
                                "result": result_data
                            })
                        valid_result_idx += 1
                    else:
                        results.append({
                            "index": i,
                            "success": False,
                            "error": "배치 처리 결과 부족"
                        })
            
            if save_to_mongodb and saved_count > 0:
                logger.info(f"MongoDB 저장 완료: {saved_count}/{len(valid_messages)}개 메시지")
        
        except Exception as e:
            logger.error(f"배치 처리 중 오류: {e}")
            # 배치 처리 실패 시 모든 메시지를 실패로 처리
            results = []
            for i, message in enumerate(messages):
                results.append({
                    "index": i,
                    "success": False,
                    "error": f"배치 처리 실패: {str(e)}"
                })
        
        processing_time = time.time() - start_time
        
        # 캡처된 프롬프트들 가져오기
        captured_prompts = getattr(current_thread, 'stored_prompts', {})
        logger.info(f"배치 추출 과정에서 캡처된 프롬프트: {len(captured_prompts)}개")
        
        # 성공/실패 개수 집계
        successful = sum(1 for r in results if r["success"])
        failed = len(results) - successful
        
        response = {
            "success": True,
            "results": results,
            "summary": {
                "total_messages": len(messages),
                "successful": successful,
                "failed": failed,
                "saved_to_mongodb": saved_count if save_to_mongodb else 0
            },
            "metadata": {
                "llm_model": llm_model,
                "offer_info_data_src": offer_info_data_src,
                "product_info_extraction_mode": product_info_extraction_mode,
                "entity_matching_mode": entity_matching_mode,
                "extract_entity_dag": extract_entity_dag,
                "max_workers": max_workers,
                "processing_time_seconds": round(processing_time, 3),
                "timestamp": time.time()
            },
            "prompts": {
                "success": True,
                "prompts": captured_prompts,
                "settings": {
                    "llm_model": llm_model,
                    "offer_info_data_src": offer_info_data_src,
                    "product_info_extraction_mode": product_info_extraction_mode,
                    "entity_matching_mode": entity_matching_mode,
                    "extract_entity_dag": extract_entity_dag
                },
                "batch_info": {
                    "total_messages": len(messages),
                    "successful": successful,
                    "failed": failed
                },
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

@app.route('/prompts', methods=['POST'])
def get_prompts():
    """
    실제 추출 과정에서 사용된 프롬프트들을 반환하는 엔드포인트
    
    실제 추출을 수행하고 그 과정에서 LLM에 전송된 프롬프트들을 캡처하여 반환합니다.
    """
    try:
        if not global_extractor:
            return jsonify({
                "success": False,
                "error": "추출기가 초기화되지 않았습니다"
            }), 500
        
        # 요청 데이터 파싱
        data = request.get_json()
        if not data:
            return jsonify({
                "success": False,
                "error": "JSON 데이터가 필요합니다"
            }), 400
        
        message = data.get('message', '')
        if not message:
            return jsonify({
                "success": False,
                "error": "메시지가 필요합니다"
            }), 400
        
        # 설정 파라미터 추출
        llm_model = data.get('llm_model', 'ax')
        offer_info_data_src = data.get('offer_info_data_src', 'db')
        product_info_extraction_mode = data.get('product_info_extraction_mode', 'llm')
        entity_matching_mode = data.get('entity_matching_mode', 'logic')
        extract_entity_dag = data.get('extract_entity_dag', True)
        
        # 추출기 설정 업데이트
        extractor = get_configured_extractor(llm_model, product_info_extraction_mode, entity_matching_mode, extract_entity_dag)
        
        # 실제 추출 수행 (프롬프트 캡처를 위해)
        import threading
        current_thread = threading.current_thread()
        current_thread.stored_prompts = {}  # 프롬프트 저장소 초기화
        
        logger.info(f"프롬프트 캡처 시작 - 스레드 ID: {current_thread.ident}")
        
        # 추출 수행
        if extract_entity_dag:
            result = process_message_worker(extractor, message, extract_dag=True)['extracted_result']
        else:
            result = extractor.process_message(message)['extracted_result']
        
        # 저장된 프롬프트들 가져오기
        stored_prompts = getattr(current_thread, 'stored_prompts', {})
        
        logger.info(f"프롬프트 캡처 완료 - 스레드 ID: {current_thread.ident}")
        logger.info(f"실제 stored_prompts 내용: {stored_prompts}")
        
        logger.info(f"프롬프트 캡처 상태: {len(stored_prompts)}개 프롬프트")
        logger.info(f"프롬프트 키들: {list(stored_prompts.keys())}")
        
        # 프롬프트가 없어도 성공으로 처리 (일부 모드에서는 특정 프롬프트만 생성됨)
        # if not stored_prompts:
        #     return jsonify({
        #         "success": False,
        #         "error": "프롬프트가 캡처되지 않았습니다",
        #         "prompts": {},
        #         "settings": {...}
        #     }), 200
        
        # 응답 구성
        response = {
            "success": True,
            "prompts": stored_prompts,
            "settings": {
                "llm_model": llm_model,
                "offer_info_data_src": offer_info_data_src,
                "product_info_extraction_mode": product_info_extraction_mode,
                "entity_matching_mode": entity_matching_mode,
                "extract_entity_dag": extract_entity_dag
            },
            "message_info": {
                "length": len(message),
                "preview": message[:200] + "..." if len(message) > 200 else message
            },
            "timestamp": time.time(),
            "extraction_result": result  # 추출 결과도 함께 반환 (참고용)
        }
        
        logger.info(f"실제 프롬프트 캡처 완료: {len(stored_prompts)}개 프롬프트")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"프롬프트 캡처 중 오류 발생: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": time.time()
        }), 500

@app.route('/dag', methods=['POST'])
def extract_dag_endpoint():
    """
    Entity DAG 추출 API
    
    MMS 메시지에서 엔티티 간의 관계를 분석하여 DAG(Directed Acyclic Graph) 형태로 추출합니다.
    
    Request Body (JSON):
        - message (required): 분석할 MMS 메시지 텍스트
        - llm_model (optional): 사용할 LLM 모델 (기본값: 'ax')
                                선택 가능: 'ax', 'gem', 'cld', 'gen', 'gpt'
        - save_dag_image (optional): DAG 이미지 저장 여부 (기본값: False)
    
    Returns:
        JSON: DAG 추출 결과
            - success: 처리 성공 여부
            - result: DAG 추출 결과
                - dag_section: 파싱된 DAG 텍스트
                - dag_raw: LLM 원본 응답
                - dag_json: NetworkX 그래프를 JSON으로 변환
                - analysis: 그래프 분석 정보 (노드 수, 엣지 수, root/leaf 노드 등)
            - metadata: 처리 메타데이터 (처리 시간, 사용된 설정 등)
    
    HTTP Status Codes:
        - 200: 성공
        - 400: 잘못된 요청 (필수 필드 누락, 잘못된 파라미터 등)
        - 500: 서버 내부 오류
    
    Example Request:
        ```json
        {
            "message": "SK텔레콤 가입하시면 ZEM폰을 드립니다",
            "llm_model": "ax",
            "save_dag_image": true
        }
        ```
    """
    try:
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
        
        # 선택적 파라미터 추출
        llm_model_name = data.get('llm_model', 'ax')
        save_dag_image = data.get('save_dag_image', False)
        message_id = data.get('message_id', '#')  # 메시지 ID (기본값: '#')
        
        # 파라미터 유효성 검증
        valid_llm_models = ['ax', 'gem', 'cld', 'gen', 'gpt']
        if llm_model_name not in valid_llm_models:
            return jsonify({"error": f"잘못된 llm_model입니다. 사용 가능: {valid_llm_models}"}), 400
        
        # LLM 모델 매핑
        llm_model_map = {
            'ax': llm_ax,
            'gem': llm_gem,
            'cld': llm_cld,
            'gen': llm_gen,
            'gpt': llm_gpt
        }
        llm_model = llm_model_map[llm_model_name]
        
        logger.info(f"🎯 DAG 추출 요청 - LLM: {llm_model_name}, 메시지 길이: {len(message)}자")
        
        # 메시지 ID 로깅
        if message_id != '#':
            logger.info(f"📋 메시지 ID: {message_id}")
        
        # DAG 파서 초기화
        parser = DAGParser()
        
        # DAG 추출 실행
        start_time = time.time()
        result = extract_dag(parser, message, llm_model)
        processing_time = time.time() - start_time
        
        # NetworkX 그래프를 JSON으로 변환
        dag = result['dag']
        dag_json = parser.to_json(dag)
        analysis = parser.analyze_graph(dag)
        
        # 이미지 저장 (선택 사항)
        dag_image_url = None
        dag_image_path = None
        if save_dag_image:
            try:
                from utils import create_dag_diagram, sha256_hash
                from config import settings
                
                dag_hash = sha256_hash(message)
                dag_image_filename = f'dag_{message_id}_{dag_hash}.png'
                
                # 설정에 따라 저장 위치 결정 (재생성된 STORAGE_CONFIG 사용)
                dag_dir = settings.STORAGE_CONFIG.get_dag_images_dir()
                output_dir = f'./{dag_dir}'
                
                # DAG 다이어그램 생성 및 저장 (output_dir 명시적으로 전달)
                create_dag_diagram(dag, filename=f'dag_{message_id}_{dag_hash}', output_dir=output_dir)
                
                # HTTP URL 생성 (스토리지 모드에 따라 URL 결정)
                # - local 모드: API 서버 고정 주소 사용 (http://skt-tosaipoc01:8000)
                # - nas 모드: NAS 서버 절대 IP 주소 사용 (http://172.27.7.58)
                dag_image_url = settings.STORAGE_CONFIG.get_dag_image_url(dag_image_filename)
                
                # 실제 로컬 경로 (저장된 실제 경로)
                dag_image_path = str(Path(__file__).parent / dag_dir / dag_image_filename)
                
                logger.info(f"📊 DAG 이미지 저장됨: {dag_image_path} ({settings.STORAGE_CONFIG.dag_storage_mode} 모드)")
                logger.info(f"🌐 DAG 이미지 URL: {dag_image_url}")
            except Exception as e:
                logger.warning(f"⚠️ DAG 이미지 저장 실패: {e}")
        
        # MongoDB 저장 (선택 사항)
        save_to_mongodb = data.get('save_to_mongodb', False)
        if save_to_mongodb:
            try:
                # save_result_to_mongodb_if_enabled 함수가 기대하는 형식으로 결과 구성
                # ext_result와 raw_result에 DAG 정보 포함
                dag_list = sorted([d for d in result['dag_section'].split('\n') if d!=''])
                
                mock_result = {
                    'ext_result': {
                        'message_id': message_id,
                        'entity_dag': dag_list,
                        'dag_json': json.loads(dag_json),
                        'dag_analysis': analysis
                    },
                    'raw_result': {
                        'message_id': message_id,
                        'dag_raw': result['dag_raw']
                    },
                    'processing_time': processing_time
                }
                
                # 가짜 args 객체 생성 (함수 시그니처 맞추기 위함)
                mock_args = {
                    'save_to_mongodb': True,
                    'llm_model': llm_model_name,
                    'processing_mode': 'api_dag',
                    'user_id': 'API_USER'
                }
                
                logger.info("MongoDB 저장 중...")
                saved_id = save_result_to_mongodb_if_enabled(message, mock_result, mock_args)
                if saved_id:
                    logger.info(f"MongoDB 저장 완료! ID: {saved_id}")
            except Exception as e:
                logger.error(f"MongoDB 저장 실패: {e}")
        
        # 응답 구성
        response = {
            "success": True,
            "result": {
                "message_id": message_id,  # message_id 추가
                "dag_section": result['dag_section'],
                "dag_raw": result['dag_raw'],
                "dag_json": json.loads(dag_json),
                "analysis": analysis,
                "dag_image_url": dag_image_url,  # HTTP URL (외부 시스템용)
                "dag_image_path": dag_image_path  # 로컬 경로 (내부 참조용)
            },
            "metadata": {
                "llm_model": llm_model_name,
                "processing_time_seconds": round(processing_time, 3),
                "timestamp": time.time(),
                "message_length": len(message),
                "save_dag_image": save_dag_image
            }
        }
        
        logger.info(f"✅ DAG 추출 완료: {processing_time:.3f}초, 노드: {analysis['num_nodes']}, 엣지: {analysis['num_edges']}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"❌ DAG 추출 중 오류 발생: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": time.time()
        }), 500

# =============================================================================
# Quick Extractor API 엔드포인트 (제목 및 수신거부 번호 추출)
# =============================================================================

@app.route('/quick/extract', methods=['POST'])
def quick_extract():
    """
    단일 메시지에서 제목과 수신거부 전화번호를 추출하는 API
    
    Request Body (JSON):
    {
        "message": "메시지 텍스트",
        "method": "textrank|tfidf|first_bracket|llm",  // 선택사항, 기본값: textrank
        "llm_model": "ax|gpt|claude|gemini",            // LLM 방법 사용 시 선택사항, 기본값: ax
        "use_llm": false                                 // LLM 사용 여부, 기본값: false
    }
    
    Response (JSON):
    {
        "success": true,
        "data": {
            "title": "추출된 제목",
            "unsubscribe_phone": "1504",
            "message": "전체 메시지 내용..."
        },
        "metadata": {
            "method": "textrank",
            "message_length": 188,
            "processing_time_seconds": 0.123
        }
    }
    """
    try:
        # 요청 시작 시간
        start_time = time.time()
        
        # 요청 데이터 파싱
        data = request.get_json()
        if not data:
            return jsonify({
                "success": False,
                "error": "요청 본문이 비어있습니다. JSON 형식으로 데이터를 전송하세요."
            }), 400
        
        # 필수 파라미터 검증
        message = data.get('message')
        if not message:
            return jsonify({
                "success": False,
                "error": "'message' 필드는 필수입니다."
            }), 400
        
        # 선택적 파라미터 (기본값 설정)
        method = data.get('method', 'textrank')
        use_llm = data.get('use_llm', method == 'llm')
        llm_model = data.get('llm_model', 'ax')
        message_id = data.get('message_id', '#')  # 메시지 ID (기본값: '#')
        
        # 메서드 검증
        valid_methods = ['textrank', 'tfidf', 'first_bracket', 'llm']
        if method not in valid_methods:
            return jsonify({
                "success": False,
                "error": f"유효하지 않은 method: {method}. 사용 가능: {', '.join(valid_methods)}"
            }), 400
        
        # Quick Extractor 구성 및 가져오기
        extractor = get_configured_quick_extractor(use_llm=use_llm, llm_model=llm_model)
        
        # 메시지 처리
        logger.info(f"📝 Quick Extract 시작: method={method}, use_llm={use_llm}, llm_model={llm_model}")
        result = extractor.process_single_message(message, method=method)
        
        # 처리 시간 계산
        processing_time = time.time() - start_time
        
        # 메타데이터에 처리 시간 추가
        result['metadata']['processing_time_seconds'] = round(processing_time, 3)
        result['metadata']['timestamp'] = time.time()
        
        # message_id 추가
        result['data']['message_id'] = message_id
        
        logger.info(f"✅ Quick Extract 완료: {processing_time:.3f}초, 제목={result['data']['title'][:50]}...")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ Quick Extract 오류: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": time.time()
        }), 500

@app.route('/quick/extract/batch', methods=['POST'])
def quick_extract_batch():
    """
    여러 메시지에서 제목과 수신거부 전화번호를 일괄 추출하는 API
    
    Request Body (JSON):
    {
        "messages": ["메시지1", "메시지2", ...],
        "method": "textrank|tfidf|first_bracket|llm",  // 선택사항, 기본값: textrank
        "llm_model": "ax|gpt|claude|gemini",            // LLM 방법 사용 시 선택사항, 기본값: ax
        "use_llm": false                                 // LLM 사용 여부, 기본값: false
    }
    
    Response (JSON):
    {
        "success": true,
        "data": {
            "results": [
                {
                    "msg_id": 0,
                    "title": "추출된 제목",
                    "unsubscribe_phone": "1504",
                    "message": "전체 메시지 내용..."
                },
                ...
            ],
            "statistics": {
                "total_messages": 10,
                "with_unsubscribe_phone": 8,
                "extraction_rate": 80.0
            }
        },
        "metadata": {
            "method": "textrank",
            "processing_time_seconds": 1.234,
            "avg_time_per_message": 0.123
        }
    }
    """
    try:
        # 요청 시작 시간
        start_time = time.time()
        
        # 요청 데이터 파싱
        data = request.get_json()
        if not data:
            return jsonify({
                "success": False,
                "error": "요청 본문이 비어있습니다. JSON 형식으로 데이터를 전송하세요."
            }), 400
        
        # 필수 파라미터 검증
        messages = data.get('messages')
        if not messages:
            return jsonify({
                "success": False,
                "error": "'messages' 필드는 필수입니다."
            }), 400
        
        if not isinstance(messages, list):
            return jsonify({
                "success": False,
                "error": "'messages'는 리스트 형식이어야 합니다."
            }), 400
        
        if len(messages) == 0:
            return jsonify({
                "success": False,
                "error": "최소 1개 이상의 메시지가 필요합니다."
            }), 400
        
        # 선택적 파라미터 (기본값 설정)
        method = data.get('method', 'textrank')
        use_llm = data.get('use_llm', method == 'llm')
        llm_model = data.get('llm_model', 'ax')
        
        # 메서드 검증
        valid_methods = ['textrank', 'tfidf', 'first_bracket', 'llm']
        if method not in valid_methods:
            return jsonify({
                "success": False,
                "error": f"유효하지 않은 method: {method}. 사용 가능: {', '.join(valid_methods)}"
            }), 400
        
        # Quick Extractor 구성 및 가져오기
        extractor = get_configured_quick_extractor(use_llm=use_llm, llm_model=llm_model)
        
        # 배치 메시지 처리
        logger.info(f"📝 Quick Extract Batch 시작: {len(messages)}개 메시지, method={method}, use_llm={use_llm}")
        
        results = []
        msg_processing_times = []
        
        for idx, msg_item in enumerate(messages):
            # 메시지가 문자열이거나 딕셔너리일 수 있음
            if isinstance(msg_item, dict):
                message = msg_item.get('message', '')
                message_id = msg_item.get('message_id', '#')
            else:
                message = msg_item
                message_id = '#'
            
            msg_start_time = time.time()
            result = extractor.process_single_message(message, method=method)
            msg_processing_time = time.time() - msg_start_time
            
            # 결과에 메시지 ID와 처리 시간 추가
            message_result = {
                'msg_id': idx,
                'message_id': message_id,  # message_id 추가
                'title': result['data']['title'],
                'unsubscribe_phone': result['data']['unsubscribe_phone'],
                'message': result['data']['message'],
                'processing_time_seconds': round(msg_processing_time, 3)
            }
            results.append(message_result)
            msg_processing_times.append(msg_processing_time)
        
        # 통계 계산
        total = len(results)
        with_phone = sum(1 for r in results if r.get('unsubscribe_phone'))
        
        # 처리 시간 계산
        processing_time = time.time() - start_time
        avg_time = sum(msg_processing_times) / total if total > 0 else 0
        min_time = min(msg_processing_times) if msg_processing_times else 0
        max_time = max(msg_processing_times) if msg_processing_times else 0
        
        # 응답 구성
        response = {
            'success': True,
            'data': {
                'results': results,
                'statistics': {
                    'total_messages': total,
                    'with_unsubscribe_phone': with_phone,
                    'extraction_rate': round(with_phone / total * 100, 2) if total > 0 else 0,
                    'total_processing_time_seconds': round(sum(msg_processing_times), 3),
                    'avg_processing_time_seconds': round(avg_time, 3),
                    'min_processing_time_seconds': round(min_time, 3),
                    'max_processing_time_seconds': round(max_time, 3)
                }
            },
            'metadata': {
                'method': method,
                'total_time_seconds': round(processing_time, 3),
                'timestamp': time.time()
            }
        }
        
        logger.info(f"✅ Quick Extract Batch 완료: {processing_time:.3f}초, {total}개 메시지 처리")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"❌ Quick Extract Batch 오류: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": time.time()
        }), 500

@app.route('/dag_images/<path:filename>', methods=['GET'])
def serve_dag_image(filename):
    """
    DAG 이미지 파일 제공 엔드포인트
    
    외부 시스템에서 HTTP를 통해 DAG 이미지에 접근할 수 있도록 합니다.
    설정에 따라 로컬 또는 NAS 디렉토리에서 파일을 제공합니다.
    
    Parameters:
    -----------
    filename : str
        이미지 파일명 (예: dag_abc123.png)
    
    Returns:
    --------
    file : 이미지 파일
    """
    try:
        from config import settings
        
        # DAG 이미지 디렉토리 (스토리지 모드와 관계없이 동일)
        dag_dir = settings.STORAGE_CONFIG.get_dag_images_dir()
        dag_images_dir = Path(__file__).parent / dag_dir
        
        logger.info(f"📊 DAG 이미지 요청: {filename} (from {dag_dir})")
        
        return send_from_directory(dag_images_dir, filename)
    except FileNotFoundError:
        logger.warning(f"⚠️ DAG 이미지 없음: {filename}")
        return jsonify({
            "success": False,
            "error": "Image not found"
        }), 404
    except Exception as e:
        logger.error(f"❌ DAG 이미지 제공 오류: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

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
    parser.add_argument('--offer-data-source', choices=['local', 'db'], default='db',
                       help='데이터 소스 (local: CSV 파일, db: 데이터베이스)')
    parser.add_argument('--product-info-extraction-mode', choices=['nlp', 'llm' ,'rag'], default='nlp',
                       help='상품 정보 추출 모드 (nlp: 형태소분석, llm: LLM 기반, rag: 검색증강생성)')
    parser.add_argument('--entity-matching-mode', choices=['logic', 'llm'], default='llm',
                       help='엔티티 매칭 모드 (logic: 로직 기반, llm: LLM 기반)')
    parser.add_argument('--llm-model', choices=['gem', 'ax', 'cld', 'gen', 'gpt'], default='ax',
                       help='사용할 LLM 모델 (gem: Gemma, ax: ax, cld: Claude, gen: Gemini, gpt: GPT)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO',
                       help='로그 레벨 설정 (DEBUG: 상세, INFO: 일반, WARNING: 경고, ERROR: 오류만)')
    parser.add_argument('--extract-entity-dag', action='store_true', default=False, help='Entity DAG extraction (default: False)')
    parser.add_argument('--storage', choices=['local', 'nas'], default='local',
                       help='DAG 이미지 저장 위치 (local: 로컬 디스크, nas: NAS 서버)')
    
    args = parser.parse_args()
    
    # 로그 레벨 설정 - 루트 로거와 모든 핸들러에 적용
    log_level = getattr(logging, args.log_level)
    root_logger.setLevel(log_level)
    for handler in root_logger.handlers:
        handler.setLevel(log_level)
    logger.setLevel(log_level)
    mms_logger.setLevel(log_level)
    
    logger.info(f"로그 레벨 설정: {args.log_level}")
    
    # DAG 저장 모드 설정
    logger.info(f"🔧 --storage 옵션: {args.storage}")
    os.environ['DAG_STORAGE_MODE'] = args.storage
    logger.info(f"🔧 환경변수 DAG_STORAGE_MODE 설정: {os.environ.get('DAG_STORAGE_MODE')}")
    
    # STORAGE_CONFIG 재생성 (환경변수 적용)
    from config.settings import StorageConfig
    from config import settings
    settings.STORAGE_CONFIG = StorageConfig()
    STORAGE_CONFIG = settings.STORAGE_CONFIG
    
    logger.info(f"📁 DAG 저장 모드: {STORAGE_CONFIG.dag_storage_mode} - {STORAGE_CONFIG.get_storage_description()}")
    logger.info(f"📂 DAG 저장 경로: {STORAGE_CONFIG.get_dag_images_dir()}")
    if STORAGE_CONFIG.dag_storage_mode == 'local':
        logger.info(f"🌐 로컬 서버 URL: {STORAGE_CONFIG.local_base_url}")
    else:
        logger.info(f"🌐 NAS 서버 URL: {STORAGE_CONFIG.nas_base_url}")
    
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
            logger.info(f"추출기 설정: llm_model={args.llm_model}, product_mode={args.product_info_extraction_mode}, entity_mode={args.entity_matching_mode}, dag_extract={args.extract_entity_dag}")
            extractor = get_configured_extractor(args.llm_model, args.product_info_extraction_mode, args.entity_matching_mode, args.extract_entity_dag)
            
            if not message.strip():
                logger.info("텍스트가 제공되지 않아 샘플 메시지를 사용합니다...")
            
            logger.info("메시지 처리 중...")
            
            # DAG 추출 여부에 따라 병렬 처리 또는 단일 처리
            if args.extract_entity_dag:
                logger.info("DAG 추출과 함께 병렬 처리 시작")
                result = process_message_worker(extractor, message, extract_dag=True)
            else:
                result = extractor.process_message(message)
                result['entity_dag'] = []
            
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
        logger.info("  POST /dag - Entity DAG 추출")
        
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
