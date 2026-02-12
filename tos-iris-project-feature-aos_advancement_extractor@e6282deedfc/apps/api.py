#!/usr/bin/env python3
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
from pprint import pprint

from config import settings
# for [FastAPI]
from fastapi import FastAPI, Request            # for FastAPI definition and request
from fastapi.responses import JSONResponse, FileResponse      # for JSON Response, File Dowload(DAG)
from starlette.exceptions import HTTPException as APIHTTPException # for error handling
import uvicorn # for FastAPI running
from pydantic import BaseModel # for POST
from fastapi import BackgroundTasks # for async
# for [global application variable(ex: global_extractor)
from store import ExtractorStore
from contextlib import asynccontextmanager
from utils.db_utils import get_message_from_database, insert_extract_result_to_database, get_message_list_from_database
from typing import Dict, Any, List
import inspect

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

def initialize_global_extractor(offer_info_data_src='db', num_cand_pgms=None, num_select_pgms=None):
    """
    전역 추출기 인스턴스를 서버 시작 시 한 번만 초기화

    이 함수는 무거운 데이터 로딩 작업(상품 정보, 임베딩 모델 등)을
    서버 시작 시 미리 수행하여 API 요청 처리 시간을 단축합니다.

    Args:
        offer_info_data_src: 상품 정보 데이터 소스 ('local' 또는 'db')
        num_cand_pgms: 프로그램 후보 개수 (None이면 config 기본값 사용)
        num_select_pgms: LLM이 최종 선정할 프로그램 수 (None이면 config 기본값 사용)

    Returns:
        MMSExtractor: 초기화된 추출기 인스턴스
    """
    global global_extractor
    if global_extractor is None:
        # 기본 설정으로 추출기 초기화 (CLI와 동일한 기본값 사용)
        global_extractor = MMSExtractor(
            model_path='./models/ko-sbert-nli',      # 임베딩 모델 경로
            data_dir='./data',                       # 데이터 디렉토리
            offer_info_data_src=offer_info_data_src, # 상품 정보 소스
            llm_model='ax',                          # 기본 LLM: AX (CLI와 동일)
            product_info_extraction_mode='llm',      # 기본 상품 추출 모드: LLM (CLI와 동일)
            entity_extraction_mode='llm',            # 기본 엔티티 매칭 모드: LLM (CLI와 동일)
            extract_entity_dag=True,
            entity_extraction_context_mode='dag',    # 기본 컨텍스트 모드: DAG
            num_cand_pgms=num_cand_pgms,
            num_select_pgms=num_select_pgms,
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

def get_configured_extractor(llm_model='ax', product_info_extraction_mode='llm', entity_matching_mode='llm', entity_llm_model='ax', extract_entity_dag=True, entity_extraction_context_mode='dag'):
    #global global_extractor
    #global_extractor = app.state.store.global_extractor

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

# 전역 추출기 변수 정보를 application 전역에서 관리
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.store = ExtractorStore()
    # 지정된 데이터 소스로 전역 추출기 초기화
    logger.info("전역 추출기 초기화 중...")
    app.state.store.init('db')
    yield
    # shutdown 시 처리 기능 #

app = FastAPI(lifespan=lifespan)

#Added by P099870, 2026.01.14
class MessageRequest(BaseModel):
    message_id: str

class SaveAnswerSheet(BaseModel):
    message_id: str
    data: Dict[str, Any]

class BatchRequest(BaseModel):
    message_ids: List[str]

@app.post('/ai/mms/v1/extract/batch')
async def extract_batch(request: BatchRequest, background_tasks: BackgroundTasks):
    logger.info(f">>>>>>>>>>REQUEST_API::{inspect.currentframe().f_code.co_name}<<<<<<<<<<")
    global global_extractor
    global_extractor = app.state.store.global_extractor
    # 전역 추출기 초기화 상태 확인
    if global_extractor is None:
        logger.error(f"추출기가 초기화되지 않았습니다. 서버 시작 중 오류가 발생했을 수 있습니다..")
        return {
             "success": False,
             "error": "서버 내부 에러 발생(EM01)",
             "timestamp": time.time()
        }, 500
    
    try:
        message_ids = request.message_ids
        if not message_ids or len(message_ids) == 0:
            return {"status": "400", "success": False, "error": "메시지ID List가 없습니다","timestamp": time.time()}, 400
        if len(message_ids) > 100:
            return {"status":"400", "success": False, "error": f"배치당 최대 100개 메시지까지 처리 가능합니다({len(message_ids)})","timestamp": time.time()}, 400
        if not isinstance(message_ids, list):
            return {"status":"400", "success": False, "error": "'message_ids' 필드는 배열이어야 합니다","timestamp": time.time()}, 400

        # 메시지 IDs 로깅
        logger.info(f"📋 메시지 ID Count: {len(message_ids)}")
        logger.info(f"📋 메시지 ID List: {message_ids}")
        pprint(message_ids);
        message_list = get_message_list_from_database(message_ids)
        if not message_list :
            logger.warning(f"DB에서 message list 데이터 없음")
            return {"status":"400", "success": False, "error": "'message_ids'에 해당하는 message list가 DB에 한건도 없습니다","timestamp": time.time()}, 400
        else:
            logger.info(f"DB에서 조회된 message_list 건수[{len(message_list)}/{len(message_ids)}]")
        data = {}
        # 선택적 파라미터 추출
        data["offer_info_data_src"] = "db"
        data["llm_model"] = "ax"
        data["entity_llm_model"] =  'ax'
        data["product_info_extraction_mode"] = settings.ProcessingConfig.product_info_extraction_mode
        data["entity_matching_mode"] = settings.ProcessingConfig.entity_extraction_mode
        data["extract_entity_dag"] = True
        data["entity_extraction_context_mode"] = 'dag'
        data["max_workers"] = None
        data["save_to_mongodb"] = True
        data["result_type"] = 'ext'
        data['processing_mode'] = 'batch'
        data['message_list'] = message_list

        background_tasks.add_task(extract_batch_background, data)

        # 요청 파리미터 유효성 검증 결과 리턴
        return {
             "status": "200",
             "error" : None,
             "success": True,
             "timestamp": time.time()
        }, 200

    except Exception as e:
        logger.error(f"요청 검증 중 오류 발생: {e}")
        return {
            "status": "500",
             "success": False,
             #"error": str(e),
             "error": "예상치 못한 서버 에러 발생",
             "timestamp": time.time()
        }, 500

# 분석 요청 배치 비동기 처리
def extract_batch_background(data):        
    try:
        message_list = data['message_list']
        offer_info_data_src = data.get('offer_info_data_src')
        llm_model = data.get('llm_model')
        entity_llm_model = data.get('entity_llm_model')
        product_info_extraction_mode = data.get('product_info_extraction_mode')
        entity_matching_mode = data.get('entity_matching_mode')
        extract_entity_dag = data.get('extract_entity_dag')
        entity_extraction_context_mode = data.get('entity_extraction_context_mode')
        max_workers = data.get('max_workers')
        save_to_mongodb = data.get('save_to_mongodb')
        result_type = data.get('result_type')

        # 구성된 추출기 가져오기
        extractor = get_configured_extractor(llm_model, product_info_extraction_mode, entity_matching_mode, entity_llm_model, extract_entity_dag, entity_extraction_context_mode)
        
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
        for i, msg_item in enumerate(message_list):
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
        
        logger.info(f"배치 처리 시작: {len(valid_messages)}/{len(message_list)}개 유효한 메시지")
        
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
            
            for i, msg_item in enumerate(message_list):
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
                                    logger.info(f"SavedId = {saved_id}")
                                    if saved_id:
                                        saved_count += 1
                                        logger.debug(f"메시지 {i} MongoDB 저장 완료 (ID: {saved_id[:8]}...)")
                                        insert_extract_result_to_database(result, message_id, saved_id)
                                        logger.info("DB 분석 결과 저장 완료!(API-BATCH)")
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
            for i, message in enumerate(message_list):
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
                "total_messages": len(message_list),
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
                    "total_messages": len(message_list),
                    "successful": successful,
                    "failed": failed
                },
                "timestamp": time.time()
            }
        }
        
        logger.info(f"배치 추출 완료: {successful}/{len(message_list)}개 성공, {processing_time:.3f}초 소요")
        logger.info(f"추출 완료: {processing_time:.3f}초")
    except Exception as e:
        logger.error(f"배치 추출 중 오류 발생: {e}")

@app.put('/ai/mms/v1/answer_sheet/{message_id}')
def save_answer_sheet(request: SaveAnswerSheet):
    message_id = None
    try:
        logger.info(f">>>>>>>>>>REQUEST_API::{inspect.currentframe().f_code.co_name}<<<<<<<<<<")
        message_id = request.message_id
        if not message_id or not message_id.strip():
            return {"success": False, "error": "메시지는 비어있을 수 없습니다","timestamp": time.time()}, 400

        # 메시지 ID 로깅
        if message_id != '#':
            logger.info(f"📋 메시지 ID: {message_id}")

        data = request.data;
        if not data:
            return {"success": False, "error": "정답지 데이터가 비어있습니다","timestamp": time.time()}, 400
        
        pprint(data)
        from utils.mongodb_utils import MongoDBManager
        mongoMgr = MongoDBManager()
        saved_id = mongoMgr.save_answer_sheet(message_id, data)
        logger.info(f"saved_id: {saved_id}")

        if saved_id:
            return {
                "success": True,
                "status": 200,
                "data": {"message_id": message_id},
                "timestamp": time.time()
            }, 200
        else:
            return {
                "success": False,
                "status": 400,
                "data": {"message_id": message_id},
                "error": "처리시 예상치 못한 에러 발생",
                "timestamp": time.time()
            }, 200

    except Exception as e:
        logger.error(f"요청 처리 중 오류 발생: {e}")
        return {
             "success": False,
             "status": 500,
             #"error": str(e),
             "data": {"message_id": message_id},
             "error": "예상치 못한 서버 에러 발생",
             "timestamp": time.time()
        }, 500

@app.post('/ai/mms/v1/extract_result')
def get_extract_result(request: MessageRequest):
    try:
        logger.info(f">>>>>>>>>>REQUEST_API::{inspect.currentframe().f_code.co_name}<<<<<<<<<<")
        logger.info("[get_extract_result::Requested Options]");
        message_id = request.message_id
        #logger.info(json.dumps(data, indent=2, ensure_ascii=False))
        #pprint(req)

        # 필수 필드 검증
        #if 'message_id' not in req:
        #    return {"success": False, "error": "필수 필드가 누락되었습니다: 'message_id'","timestamp": time.time()}, 400

        #message_id = req.get('message_id', '#')
        if not message_id or not message_id.strip():
            return {"success": False, "error": "메시지는 비어있을 수 없습니다","timestamp": time.time()}, 400

        # 메시지 ID 로깅
        if message_id != '#':
            logger.info(f"📋 메시지 ID: {message_id}")

        from utils.mongodb_utils import MongoDBManager
        mongoMgr = MongoDBManager()
        result = mongoMgr.get_extract_result(message_id)
        pprint(result)

        if not result:
            return {
                "error": "No Data",
                "status": "400",
                "success": False,
                "data": {},
                "timestamp": time.time()
            }, 400
        else:
            return {
                "status": "200",
                "success": True,
                "data": result.get('data'),
                "timestamp": time.time()
            }, 200

    except Exception as e:
        logger.error(f"요청 처리 중 오류 발생: {e}")
        return {
             "success": False,
             "status": "500",
             "data": {},
             #"error": str(e),
             "error": "예상치 못한 서버 에러 발생",
             "timestamp": time.time()
        }, 500

@app.post('/ai/mms/v1/answer_sheet')
def get_answer_sheet(request: MessageRequest):
    try:
        logger.info(f">>>>>>>>>>REQUEST_API::{inspect.currentframe().f_code.co_name}<<<<<<<<<<")
        message_id = request.message_id
        errorMsg = None
        if not message_id or not message_id.strip():
            errorMsg = "메시지ID는 비어있을 수 없습니다"
            logger.warn(errorMsg)
            return {"success": False, "error": {errorMsg},"timestamp": time.time()}, 400

        # 메시지 ID 로깅
        if message_id != '#':
            logger.info(f"📋 메시지 ID: {message_id}")

        from utils.mongodb_utils import MongoDBManager
        mongoMgr = MongoDBManager()
        result = mongoMgr.get_answer_sheet(message_id)
        pprint(result)

        if not result:
            errorMsg = f"message_id: [{message_id}]의 정답지가 없습니다"
            logger.warn(errorMsg)
            return {
                "success": False,
                "status": "400",
                "error": errorMsg,
                "data": {},
                "timestamp": time.time()
            }, 400
        else:
            logger.info(f"message_id: [{message_id}] 정답지 조회결과 정상리턴")
            return {
                "success": True,
                "status": "200",
                "data": result.get('data'),
                "timestamp": time.time()
            }, 200

    except Exception as e:
        errorMsg = f"[정답지 조회]요청 처리 중 오류 발생: {e}"
        logger.error(errorMsg)
        return {
             "success": False,
             "status": "500",
             "data": {},
             #"error": str(e),
             "error": "예상치 못한 서버 에러 발생",
             "timestamp": time.time()
        }, 500

# 이전(초기) 버전 분석추출 API for CLI : message, message_id, options을 요청받아 처리
@app.post('/extract_cli')
async def extract_message_cli(request: Request, background_tasks: BackgroundTasks):
    logger.info(f">>>>>>>>>>REQUEST_API::{inspect.currentframe().f_code.co_name}<<<<<<<<<<")
    global global_extractor
    global_extractor = app.state.store.global_extractor
    # 전역 추출기 초기화 상태 확인
    if global_extractor is None:
        logger.error(f"추출기가 초기화되지 않았습니다. 서버 시작 중 오류가 발생했을 수 있습니다..")
        return {
             "success": False,
             "status": "500",
             "error": "서버 내부 에러 발생(EM01)",
             "timestamp": time.time()
        }, 500
    # print(global_extractor.__dict__)
    try:
        if request.headers.get("content-type") != "application/json":
            return {"success": False, "error": "Content-Type은 application/json이어야 합니다","timestamp": time.time()}, 400
        data = await request.json()
        logger.info("[Requested Extract Options]");
        logger.info(json.dumps(data, indent=2, ensure_ascii=False))

        # 필수 필드 검증
        if 'message' not in data:
            return {"success": False, "error": "필수 필드가 누락되었습니다: 'message'","timestamp": time.time()}, 400
        
        message = data['message']
        if not message or not message.strip():
            return {"success": False, "error": "메시지는 비어있을 수 없습니다","timestamp": time.time()}, 400
        
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
        
        # 요청 파라미터 유효성 검증
        valid_sources = ['local', 'db']
        if offer_info_data_src not in valid_sources:
            return {"success": False, "error": f"잘못된 offer_info_data_src입니다. 사용 가능: {valid_sources}", "timestamp": time.time()}, 400
            
        valid_llm_models = ['gemma', 'ax', 'claude', 'gemini']
        if llm_model not in valid_llm_models:
            return {"success": False, "error": f"잘못된 llm_model입니다. 사용 가능: {valid_llm_models}", "timestamp": time.time()}, 400
            
        valid_product_modes = ['nlp', 'llm', 'rag']
        if product_info_extraction_mode not in valid_product_modes:
            return {"success": False, "error": f"잘못된 product_info_extraction_mode입니다. 사용 가능: {valid_product_modes}", "timestamp": time.time()}, 400
            
        valid_entity_modes = ['logic', 'llm']
        if entity_matching_mode not in valid_entity_modes:
            return {"success": False, "error": f"잘못된 entity_matching_mode입니다. 사용 가능: {valid_entity_modes}", "timestamp": time.time()}, 400
        
        # from pprint import pprint
        # logger.info("[Arranged Extract Options]");
        # pprint(data)

        background_tasks.add_task(extract_background, data)
        # 요청 파리미터 유효성 검증 결과 리턴
        return {
             "error" : None,
             "success": True,
             "status": "200",
             "timestamp": time.time()
        }, 200

    except Exception as e:
        logger.error(f"요청 검증 중 오류 발생: {e}")
        return {
             "success": False,
             "status": "500",
             #"error": str(e),
             "error": "예상치 못한 서버 에러 발생",
             "timestamp": time.time()
        }, 500

# Web 분석 추출 요청 API - message_id만 받아 처리
# (1) message_id로 TMSG_MMS_SCRPT 테이블에서 message 조회
#     SELECT MMS_PHRS FROM TMSG_MMS_SCRPT WHERE MSG_ID = 'M23ALO262368'
# (2) message 분석 추출 처리
# (3) 처리결과 db에 저장 - TCAM_MSG_ANALS_RSLT
# (4) 처리결과 mongodb에 저장 - aos.mmsext
@app.post('/ai/mms/v1/extract')
async def extract_message(request: Request, background_tasks: BackgroundTasks):
    logger.info(f">>>>>>>>>>REQUEST_API::{inspect.currentframe().f_code.co_name}<<<<<<<<<<")
    global global_extractor
    global_extractor = app.state.store.global_extractor
    # 전역 추출기 초기화 상태 확인
    if global_extractor is None:
        logger.error(f"추출기가 초기화되지 않았습니다. 서버 시작 중 오류가 발생했을 수 있습니다..")
        return {
             "success": False,
             "status": "500",
             "error": "서버 내부 에러 발생(EM01)",
             "timestamp": time.time()
        }, 500
    # print(global_extractor.__dict__)
    try:
        if request.headers.get("content-type") != "application/json":
            return {"status": 400, "success": False, "error": "Content-Type은 application/json이어야 합니다","timestamp": time.time()}, 400
        data = await request.json()
        logger.info("[Requested Extract Options]");
        logger.info(json.dumps(data, indent=2, ensure_ascii=False))

        # 필수 필드 검증
        if 'message_id' not in data:
            return {"status": 400, "success": False, "error": "필수 필드가 누락되었습니다: 'message_id'","timestamp": time.time()}, 400
        
        message_id = data.get('message_id', '#')  # 메시지 ID (기본값: '#')
        if not message_id or not message_id.strip():
            return {"status": 400, "success": False, "error": "메시지는 비어있을 수 없습니다","timestamp": time.time()}, 400

        # 메시지 ID 로깅
        if message_id != '#':
            logger.info(f"📋 메시지 ID: {message_id}")

        # (1) message_id로 TMSG_MMS_SCRPT 테이블에서 message 조회
        #     SELECT MMS_PHRS FROM TMSG_MMS_SCRPT WHERE MSG_ID = 'M23ALO262368'
        message = get_message_from_database(message_id)
        if message is not None:
            data["message"] = message
        else:
            return {"status": 400, "success": False, "error": f"메시지 정보가 없습니다. message_id={message_id}","timestamp": time.time()}, 400

        logger.info(f"📋 메시지: {message} by message_id={message_id}")

        # 분석 파라미터 추출 (기본값 사용)
        data_source = data.get('data_source', CLI_DATA_SOURCE)
        offer_info_data_src = data.get('offer_info_data_src', CLI_DATA_SOURCE)
        #llm_model = data.get('llm_model', settings.ModelConfig.llm_model)
        llm_model = data.get('llm_model', "ax")
        entity_llm_model = data.get('entity_llm_model', 'ax')
        product_info_extraction_mode = data.get('product_info_extraction_mode', settings.ProcessingConfig.product_info_extraction_mode)
        entity_matching_mode = data.get('entity_matching_mode', settings.ProcessingConfig.entity_extraction_mode)
        extract_entity_dag = data.get('extract_entity_dag', True)
        entity_extraction_context_mode = data.get('entity_extraction_context_mode', 'dag')
        save_to_mongodb = data.get('save_to_mongodb', True)
        result_type = data.get('result_type', 'ext')

        data['save_to_mongodb'] = save_to_mongodb
        data['result_type'] = result_type
        data['processing_mode'] = 'single'
        # DAG 추출 요청 로깅
        if extract_entity_dag:
            logger.info(f"🎯 DAG 추출 요청됨 - LLM: {llm_model}, 메시지ID 길이: {len(message_id)}자")
        
        # 요청 파라미터 유효성 검증
        valid_sources = ['local', 'db']
        if offer_info_data_src not in valid_sources:
            raise Exception(f"잘못된 offer_info_data_src입니다. 사용 가능: {valid_sources}")
            
        valid_llm_models = ['gemma', 'ax', 'claude', 'gemini']
        if llm_model not in valid_llm_models:
            raise Exception(f"잘못된 llm_model입니다. 사용 가능: {valid_llm_models}")
            
        valid_product_modes = ['nlp', 'llm', 'rag']
        if product_info_extraction_mode not in valid_product_modes:
            raise Exception(f"잘못된 product_info_extraction_mode입니다. 사용 가능: {valid_product_modes}")

        valid_entity_modes = ['logic', 'llm']
        if entity_matching_mode not in valid_entity_modes:
            raise Exception(f"잘못된 entity_matching_mode입니다. 사용 가능: {valid_entity_modes}")
        
        # from pprint import pprint
        # logger.info("[Arranged Extract Options]");
        # pprint(data)

        background_tasks.add_task(extract_background, data)
        # 요청 파리미터 유효성 검증 결과 리턴
        return {
             "error" : None,
             "status": "200",
             "success": True,
             "timestamp": time.time()
        }, 200

    except Exception as e:
        logger.error(f"요청 검증 중 오류 발생: {e}")
        return {
             "success": False,
             "status": "500",
             #"error": str(e),
             "error": "예상치 못한 서버 에러 발생",
             "timestamp": time.time()
        }, 500

def extract_background(data):        
    try:
        message = data['message']
        data_source = data.get('data_source')
        offer_info_data_src = data.get('offer_info_data_src')
        llm_model = data.get('llm_model')
        entity_llm_model = data.get('entity_llm_model')
        product_info_extraction_mode = data.get('product_info_extraction_mode')
        entity_matching_mode = data.get('entity_matching_mode')
        extract_entity_dag = data.get('extract_entity_dag')
        entity_extraction_context_mode = data.get('entity_extraction_context_mode')
        save_to_mongodb = data.get('save_to_mongodb')
        result_type = data.get('result_type')
        message_id = data.get('message_id')

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

            logger.info(f"MessageId={message_id}")
            logger.info("#######extractor_result######"*4)
            logger.info(result)

            logger.info("========db_result======="*4)
            #db_result = result.get('ext_result')
            db_result = result['ext_result']
            logger.info(f"db_result-------->{db_result}")
            pprint(db_result)

            saved_id = save_result_to_mongodb_if_enabled(message, result, data, extractor)
            logger.info(f"SavedId = {saved_id}")
            if saved_id:
                logger.info("MongoDB 저장 완료!")
                insert_extract_result_to_database(result, message_id, saved_id)
                logger.info("DB 분석 결과 저장 완료!(API)")
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
        
        # 성공 결과 반환 (프롬프트 포함)
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
        logger.info(json.dumps(response, indent=2, ensure_ascii=False))

    except Exception as e:
        logger.error(f"추출 중 오류 발생: {e}")


@app.get("/health")
def health_check():
    logger.info(f">>>>>>>>>>REQUEST_API::{inspect.currentframe().f_code.co_name}<<<<<<<<<<")
    return {
        "status": "healthy",
        "service": "MMS Extractor API",
        "version": "2.0.0",
        "model": "skt/gemma3-12b-it",
        "timestamp": time.time()
    }

@app.get('/models')
def list_models():
    logger.info(f">>>>>>>>>>REQUEST_API::{inspect.currentframe().f_code.co_name}<<<<<<<<<<")
    return {
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
    }

@app.get('/status')
def get_status():
    logger.info(f">>>>>>>>>>REQUEST_API::{inspect.currentframe().f_code.co_name}<<<<<<<<<<")
    global global_extractor
    global_extractor = app.state.store.global_extractor
    # 추출기 상태 정보 수집
    extractor_status = {
        "initialized": global_extractor is not None,
        "data_source": CLI_DATA_SOURCE if global_extractor else None,
        "current_llm_model": global_extractor.llm_model_name if global_extractor else None,
        "current_product_mode": global_extractor.product_info_extraction_mode if global_extractor else None,
        "current_entity_mode": global_extractor.entity_extraction_mode if global_extractor else None
    }
    
    return {
        "status": "running",
        "extractor": extractor_status,
        "timestamp": time.time()
    }

@app.post('/prompts')
async def get_prompts(request: Request):
    logger.info(f">>>>>>>>>>REQUEST_API::{inspect.currentframe().f_code.co_name}<<<<<<<<<<")
    content_type = request.headers.get("content-type")
    logger.info(f"content-type: {content_type}" )
    if content_type != "application/json":
        return {"success": False, "error": "Content-Type is not application/json","timestamp": time.time()}, 400
    global global_extractor
    global_extractor = app.state.store.global_extractor
    try:
        if not global_extractor:
            return {
                "success": False,
                "error": "추출기가 초기화되지 않았습니다"
            }, 500
        # 요청 데이터 파싱
        data = await request.json()
        if not data:
            return {
                "success": False,
                "error": "JSON 데이터가 필요합니다"
            }, 400
        # logger.info(json.dumps(data, indent=4, ensure_ascii=False))
        message = data.get('message', '')
        if not message:
            return {
                "success": False,
                "error": "메시지가 필요합니다"
            }, 400
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
        return response
        
    except Exception as e:
        logger.error(f"프롬프트 캡처 중 오류 발생: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": time.time()
        }, 500

@app.post('/dag')
async def extract_dag_endpoint(request: Request):
    logger.info(f">>>>>>>>>>REQUEST_API::{inspect.currentframe().f_code.co_name}<<<<<<<<<<")
    try:
        # 요청 데이터 검증
        content_type = request.headers.get("content-type")
        logger.info(f"content-type: {content_type}" )
        if content_type != "application/json":
            return {"success": False, "error": "Content-Type is not application/json","timestamp": time.time()}, 400

        data = await request.json()
        
        # 필수 필드 검증
        if 'message' not in data:
            return {"error": "필수 필드가 누락되었습니다: 'message'"}, 400
        
        message = data['message']
        if not message or not message.strip():
            return {"error": "메시지는 비어있을 수 없습니다"}, 400
        
        # 선택적 파라미터 추출
        llm_model_name = data.get('llm_model', 'ax')
        save_dag_image = data.get('save_dag_image', False)
        message_id = data.get('message_id', '#')  # 메시지 ID (기본값: '#')
        
        # 파라미터 유효성 검증
        valid_llm_models = ['ax', 'gem', 'cld', 'gen', 'gpt']
        if llm_model_name not in valid_llm_models:
            return {"error": f"잘못된 llm_model입니다. 사용 가능: {valid_llm_models}"}, 400
        
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
        return response
        
    except Exception as e:
        logger.error(f"❌ DAG 추출 중 오류 발생: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e),
            "timestamp": time.time()
        }, 500

# =============================================================================
# Quick Extractor API 엔드포인트 (제목 및 수신거부 번호 추출)
# =============================================================================

@app.post('/quick/extract')
async def quick_extract(request: Request):
    logger.info(f">>>>>>>>>>REQUEST_API::{inspect.currentframe().f_code.co_name}<<<<<<<<<<")
    try:
        content_type = request.headers.get("content-type")
        logger.info(f"content-type: {content_type}" )
        if content_type != "application/json":
            return {"success": False, "error": "Content-Type is not application/json","timestamp": time.time()}, 400

        # 요청 시작 시간
        start_time = time.time()
        
        # 요청 데이터 파싱
        data = await request.json()
        if not data:
            return {
                "success": False,
                "error": "요청 본문이 비어있습니다. JSON 형식으로 데이터를 전송하세요."
            }, 400
        
        # 필수 파라미터 검증
        message = data.get('message')
        if not message:
            return {
                "success": False,
                "error": "'message' 필드는 필수입니다."
            }, 400
        
        # 선택적 파라미터 (기본값 설정)
        method = data.get('method', 'textrank')
        use_llm = data.get('use_llm', method == 'llm')
        llm_model = data.get('llm_model', 'ax')
        message_id = data.get('message_id', '#')  # 메시지 ID (기본값: '#')
        
        # 메서드 검증
        valid_methods = ['textrank', 'tfidf', 'first_bracket', 'llm']
        if method not in valid_methods:
            return {
                "success": False,
                "error": f"유효하지 않은 method: {method}. 사용 가능: {', '.join(valid_methods)}"
            }, 400
        
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
        return result
        
    except Exception as e:
        logger.error(f"❌ Quick Extract 오류: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e),
            "timestamp": time.time()
        }, 500

@app.post('/quick/extract/batch')
async def quick_extract_batch(request: Request):
    logger.info(f">>>>>>>>>>REQUEST_API::{inspect.currentframe().f_code.co_name}<<<<<<<<<<")
    try:
        content_type = request.headers.get("content-type")
        logger.info(f"content-type: {content_type}" )
        if content_type != "application/json":
            return {"success": False, "error": "Content-Type is not application/json","timestamp": time.time()}, 400

        # 요청 시작 시간
        start_time = time.time()
        
        # 요청 데이터 파싱
        data = await request.json()
        if not data:
            return {
                "success": False,
                "error": "요청 본문이 비어있습니다. JSON 형식으로 데이터를 전송하세요."
            }, 400
        
        # 필수 파라미터 검증
        messages = data.get('messages')
        if not messages:
            return {
                "success": False,
                "error": "'messages' 필드는 필수입니다."
            }, 400
        
        if not isinstance(messages, list):
            return {
                "success": False,
                "error": "'messages'는 리스트 형식이어야 합니다."
            }, 400
        
        if len(messages) == 0:
            return {
                "success": False,
                "error": "최소 1개 이상의 메시지가 필요합니다."
            }, 400
        
        # 선택적 파라미터 (기본값 설정)
        method = data.get('method', 'textrank')
        use_llm = data.get('use_llm', method == 'llm')
        llm_model = data.get('llm_model', 'ax')
        
        # 메서드 검증
        valid_methods = ['textrank', 'tfidf', 'first_bracket', 'llm']
        if method not in valid_methods:
            return {
                "success": False,
                "error": f"유효하지 않은 method: {method}. 사용 가능: {', '.join(valid_methods)}"
            }, 400
        
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
        return response
        
    except Exception as e:
        logger.error(f"❌ Quick Extract Batch 오류: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e),
            "timestamp": time.time()
        }, 500

@app.get('/dag_images/{filename}')
def serve_dag_image(filename: str):
    logger.info(f">>>>>>>>>>REQUEST_API::{inspect.currentframe().f_code.co_name}<<<<<<<<<<")
    try:
        from config import settings
        
        # DAG 이미지 디렉토리 (스토리지 모드와 관계없이 동일)
        dag_dir = settings.STORAGE_CONFIG.get_dag_images_dir()
        logger.info(f"dag_dir:{dag_dir}")
        dag_images_dir = Path(__file__).parent / dag_dir
        logger.info(f"📊 DAG 이미지 요청: {filename} (from {dag_dir})")
    
        file_path = dag_images_dir / filename
        logger.info(f"DAG file_path: {file_path}")
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type="application/octet-stream"
        )
        
    except FileNotFoundError:
        logger.warning(f"⚠️ DAG 이미지 없음: {filename}")
        return {
            "success": False,
            "error": "Image not found"
        }, 404
    except Exception as e:
        logger.error(f"❌ DAG 이미지 제공 오류: {e}")
        return {
            "success": False,
            "error": str(e)
        }, 500


@app.exception_handler(APIHTTPException)
async def not_found(request: Request, exc: APIHTTPException):
    if exc.status_code == 404:
        return JSONResponse(
            status_code=404,content={"error": "엔드포인트를 찾을 수 없습니다"}
        )
        """404 에러 핸들러 - 존재하지 않는 엔드포인트 접근 시"""
    return {"error": "엔드포인트를 찾을 수 없습니다"}, 404

@app.exception_handler(APIHTTPException)
async def internal_error(request: Request, exc: APIHTTPException):
    if exc.status_code == 500:
        return JSONResponse(
            status_code=500,content={"error": "서버 내부 오류가 발생했습니다"}
        )
        """500 에러 핸들러 - 서버 내부 오류 발생 시"""
    return {"error": "서버 내부 오류가 발생했습니다"}, 500

def main():

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
    parser.add_argument('--product-info-extraction-mode', choices=['nlp', 'llm' ,'rag'], default='llm',
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
    parser.add_argument('--num-cand-pgms', type=int, default=None,
                       help='프로그램 후보 개수 (기본값: config의 num_candidate_programs=15)')
    parser.add_argument('--num-select-pgms', type=int, default=None,
                       help='LLM이 최종 선정할 프로그램 수 (기본값: config의 num_select_programs=1)')

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
    #logger.info("전역 추출기 초기화 중...")

    #deprecated    
    #initialize_global_extractor(CLI_DATA_SOURCE)

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
    
    log_level = 'debug'

    try:
        # 서버 시작 (리로더 비활성화, 스레딩 활성화)
        uvicorn.run("api:app", host=args.host, port=args.port, reload=False, log_level="debug" if args.debug else "info")
    except Exception as e:
        logger.error(f"서버 시작 실패: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()


