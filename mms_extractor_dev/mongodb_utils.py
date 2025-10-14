"""
MongoDB 유틸리티 함수들
MMS Extractor 결과를 MongoDB에 저장하기 위한 기능 제공
"""

import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
import logging
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MongoDBManager:
    """MongoDB 연결 및 작업 관리자"""
    
    def __init__(self, connection_string: str = "mongodb://localhost:27017/", 
                 db_name: str = "aos", collection_name: str = "mmsext"):
        """
        MongoDB 매니저 초기화
        
        Args:
            connection_string: MongoDB 연결 문자열
            db_name: 데이터베이스 이름
            collection_name: 컬렉션 이름
        """
        self.connection_string = connection_string
        self.db_name = db_name
        self.collection_name = collection_name
        self.client = None
        self.db = None
        self.collection = None
    
    def connect(self) -> bool:
        """MongoDB에 연결"""
        try:
            self.client = MongoClient(self.connection_string, serverSelectionTimeoutMS=5000)
            # 연결 테스트
            self.client.admin.command('ping')
            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]
            logger.info(f"MongoDB 연결 성공: {self.db_name}.{self.collection_name}")
            return True
        except ConnectionFailure as e:
            logger.error(f"MongoDB 연결 실패: {e}")
            return False
        except Exception as e:
            logger.error(f"MongoDB 연결 중 예상치 못한 오류: {e}")
            return False
    
    def disconnect(self):
        """MongoDB 연결 해제"""
        if self.client:
            self.client.close()
            logger.info("MongoDB 연결 해제")
    
    def save_extraction_result(self, message: str, extraction_result: Dict[str, Any], raw_result: Dict[str, Any], 
                               extraction_prompts: Dict[str, Any], user_id: str = "SKT1110566", 
                               message_id: str = None) -> Optional[str]:
        """
        MMS 추출 결과를 MongoDB에 저장
        
        Args:
            message: 원본 메시지
            extraction_result: API에서 반환된 전체 추출 결과
            raw_result: API에서 반환된 원본 JSON 결과
            extraction_prompts: 프롬프트 정보
            user_id: 분석 작업 요청자 ID (기본값: SKT1110566)
            message_id: 메시지 ID (None이면 자동 생성)
            
        Returns:
            저장된 문서의 ObjectId (문자열) 또는 None
        """
        try:
            if self.collection is None:
                if not self.connect():
                    return None
            
            # 프롬프트 데이터 파싱
            prompts_data = extraction_prompts.get('prompts', {}) if extraction_prompts else {}
            
            # 각 프롬프트 추출 (내용만)
            main_prompt = self._extract_prompt_content(prompts_data, 'main_extraction_prompt')
            ent_prompt = self._extract_prompt_content(prompts_data, 'entity_extraction_prompt')
            dag_prompt = self._extract_prompt_content(prompts_data, 'dag_extraction_prompt')
            
            # 추출 결과 파싱 (result 키에서 실제 추출 데이터 가져오기)
            ext_result = extraction_result.get('result', extraction_result.get('extracted_data', {}))

            raw_result = raw_result.get('result', raw_result.get('raw_result', {}))
            
            # 메타데이터 구성
            metadata = {
                'timestamp': datetime.utcnow(),
                'processing_time': extraction_result.get('metadata', {}).get('processing_time_seconds'),
                'processing_mode': extraction_result.get('metadata', {}).get('processing_mode', 'unknown'),
                'success': extraction_result.get('success', False),
                'settings': extraction_prompts.get('settings', {}) if extraction_prompts else {},
                'api_response_keys': list(extraction_result.keys()),
                'prompts_available': list(prompts_data.keys()) if prompts_data else []
            }
            
            # message_id가 None이면 UUID로 자동 생성
            if message_id is None:
                message_id = str(uuid.uuid4())
            
            # 저장할 문서 구성
            document = {
                'message': message,
                'main_prompt': main_prompt,
                'ent_prompt': ent_prompt,
                'dag_prompt': dag_prompt,
                'raw_result': raw_result,
                'ext_result': ext_result,
                'metadata': metadata,
                'ext_timestamp': datetime.utcnow(),  # 저장 시간
                'user_id': user_id,                  # 분석 작업 요청자 ID
                'message_id': message_id             # 메시지 ID
            }
            
            # MongoDB에 저장
            result = self.collection.insert_one(document)
            logger.info(f"문서 저장 성공: {result.inserted_id}")
            
            return str(result.inserted_id)
            
        except OperationFailure as e:
            logger.error(f"MongoDB 저장 실패: {e}")
            return None
        except Exception as e:
            logger.error(f"문서 저장 중 예상치 못한 오류: {e}")
            return None
    
    def _extract_prompt_content(self, prompts_data: Dict[str, Any], prompt_key: str) -> Optional[Dict[str, Any]]:
        """
        프롬프트 데이터에서 특정 프롬프트 내용 추출
        
        Args:
            prompts_data: 전체 프롬프트 데이터
            prompt_key: 추출할 프롬프트 키
            
        Returns:
            프롬프트 정보 딕셔너리 또는 None
        """
        if not prompts_data or prompt_key not in prompts_data:
            return None
            
        prompt_info = prompts_data[prompt_key]
        return {
            'title': prompt_info.get('title', prompt_key),
            'description': prompt_info.get('description', ''),
            'content': prompt_info.get('content', ''),
            'length': prompt_info.get('length', 0)
        }
    
    def get_recent_extractions(self, limit: int = 10) -> list:
        """최근 추출 결과 조회"""
        try:
            if self.collection is None:
                if not self.connect():
                    return []
            
            cursor = self.collection.find().sort('metadata.timestamp', -1).limit(limit)
            results = []
            
            for doc in cursor:
                doc['_id'] = str(doc['_id'])  # ObjectId를 문자열로 변환
                results.append(doc)
            
            return results
            
        except Exception as e:
            logger.error(f"최근 추출 결과 조회 실패: {e}")
            return []
    
    def get_extraction_stats(self) -> Dict[str, Any]:
        """추출 통계 정보 조회"""
        try:
            if self.collection is None:
                if not self.connect():
                    return {}
            
            total_count = self.collection.count_documents({})
            success_count = self.collection.count_documents({'metadata.success': True})
            
            # 최근 24시간 통계
            from datetime import timedelta
            yesterday = datetime.utcnow() - timedelta(days=1)
            recent_count = self.collection.count_documents({
                'metadata.timestamp': {'$gte': yesterday}
            })
            
            return {
                'total_extractions': total_count,
                'successful_extractions': success_count,
                'success_rate': (success_count / total_count * 100) if total_count > 0 else 0,
                'recent_24h': recent_count
            }
            
        except Exception as e:
            logger.error(f"통계 조회 실패: {e}")
            return {}

# 전역 MongoDB 매니저 인스턴스
_mongodb_manager = None

def get_mongodb_manager() -> MongoDBManager:
    """MongoDB 매니저 싱글톤 인스턴스 반환"""
    global _mongodb_manager
    if _mongodb_manager is None:
        _mongodb_manager = MongoDBManager()
    return _mongodb_manager

def save_to_mongodb(message: str, extraction_result: Dict[str, Any], raw_result: Dict[str, Any], 
                   extraction_prompts: Dict[str, Any], user_id: str = "SKT1110566", 
                   message_id: str = None) -> Optional[str]:
    """
    편의 함수: MMS 추출 결과를 MongoDB에 저장
    
    Args:
        message: 원본 메시지
        extraction_result: 추출 결과
        raw_result: 원본 JSON 결과
        extraction_prompts: 프롬프트 정보
        user_id: 분석 작업 요청자 ID (기본값: SKT1110566)
        message_id: 메시지 ID (None이면 자동 생성)
        
    Returns:
        저장된 문서의 ObjectId (문자열) 또는 None
    """
    # 매번 새로운 매니저 인스턴스 생성 (연결 문제 해결)
    manager = MongoDBManager()
    result = manager.save_extraction_result(message, extraction_result, raw_result, extraction_prompts, user_id, message_id)
    manager.disconnect()
    return result

def test_mongodb_connection() -> bool:
    """MongoDB 연결 테스트"""
    manager = MongoDBManager()
    success = manager.connect()
    if success:
        manager.disconnect()
    return success
