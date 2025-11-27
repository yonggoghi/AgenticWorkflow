"""
Validation Helpers - 입력/출력 검증
==================================

입력 데이터 및 추출 결과의 유효성을 검증합니다.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def validate_extraction_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    추출 결과의 유효성을 검증하고 필요시 수정
    
    Args:
        result: 추출 결과 딕셔너리
        
    Returns:
        검증된 결과 딕셔너리
    """
    try:
        # 필수 필드 확인
        required_fields = ['title', 'purpose', 'product', 'channel', 'pgm', 'offer']
        
        for field in required_fields:
            if field not in result:
                logger.warning(f"필수 필드 누락: {field}")
                if field in ['product', 'channel', 'pgm', 'purpose']:
                    result[field] = []
                elif field == 'offer':
                    result[field] = {'type': 'product', 'value': []}
                else:
                    result[field] = ''
        
        # 리스트 필드 검증
        list_fields = ['purpose', 'product', 'channel', 'pgm']
        for field in list_fields:
            if not isinstance(result.get(field), list):
                logger.warning(f"{field}가 리스트가 아닙니다. 변환합니다.")
                result[field] = []
        
        # offer 필드 검증
        if not isinstance(result.get('offer'), dict):
            logger.warning("offer가 딕셔너리가 아닙니다. 기본값으로 설정합니다.")
            result['offer'] = {'type': 'product', 'value': []}
        
        # offer.value가 리스트인지 확인
        if 'value' not in result['offer'] or not isinstance(result['offer']['value'], list):
            result['offer']['value'] = []
        
        # offer.type 기본값 설정
        if 'type' not in result['offer']:
            result['offer']['type'] = 'product'
        
        return result
        
    except Exception as e:
        logger.error(f"결과 검증 실패: {e}")
        return result


def detect_schema_response(json_objects: Dict[str, Any]) -> bool:
    """
    LLM 응답이 스키마 정의인지 실제 데이터인지 감지
    
    Args:
        json_objects: 파싱된 JSON 객체
        
    Returns:
        True if schema response, False if actual data
    """
    try:
        # 스키마 응답 패턴 감지
        schema_indicators = [
            'type',
            'properties',
            'items',
            'description',
            'required'
        ]
        
        # product 필드 확인
        if 'product' in json_objects:
            product = json_objects['product']
            
            # product가 딕셔너리이고 스키마 키워드를 포함하는 경우
            if isinstance(product, dict):
                schema_keys = [k for k in schema_indicators if k in product]
                if len(schema_keys) >= 2:
                    logger.warning(f"스키마 응답 감지: {schema_keys}")
                    return True
                
                # "type": "array" 패턴
                if product.get('type') == 'array':
                    logger.warning("스키마 응답 감지: type=array")
                    return True
        
        # 전체 응답이 스키마 형식인 경우
        if isinstance(json_objects, dict):
            schema_keys = [k for k in schema_indicators if k in json_objects]
            if len(schema_keys) >= 3:
                logger.warning(f"전체 스키마 응답 감지: {schema_keys}")
                return True
        
        return False
        
    except Exception as e:
        logger.error(f"스키마 감지 실패: {e}")
        return False
