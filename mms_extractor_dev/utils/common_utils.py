import time
import logging
from functools import wraps

logger = logging.getLogger(__name__)

def log_performance(func):
    """함수 실행 시간을 로깅하는 데코레이터"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.info(f"{func.__name__} 실행완료: {elapsed:.2f}초")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"{func.__name__} 실행실패 ({elapsed:.2f}초): {e}")
            raise
    return wrapper

def safe_execute(func, *args, default_return=None, max_retries=2, **kwargs):
    """
    안전한 함수 실행을 위한 유틸리티 함수
    
    이 함수는 네트워크 오류, API 호출 실패 등의 일시적 오류에 대해
    지수 백오프(exponential backoff)를 사용하여 재시도합니다.
    
    Args:
        func: 실행할 함수
        *args: 함수에 전달할 위치 인수
        default_return: 모든 재시도 실패 시 반환할 기본값
        max_retries: 최대 재시도 횟수 (default: 2)
        **kwargs: 함수에 전달할 키워드 인수
        
    Returns:
        함수 실행 결과 또는 default_return
        
    Example:
        result = safe_execute(api_call, data, default_return={}, max_retries=3)
    """
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries:
                # 모든 재시도 실패 시 에러 로깅 및 기본값 반환
                logger.error(f"{func.__name__} 최종 실패: {e}")
                return default_return
            else:
                # 재시도 전 대기 시간: 1초, 2초, 4초, 8초... (지수 백오프)
                logger.warning(f"{func.__name__} 재시도 {attempt + 1}/{max_retries}: {e}")
                time.sleep(2 ** attempt)
    return default_return

def safe_check_empty(obj) -> bool:
    """다양한 타입의 객체가 비어있는지 안전하게 확인"""
    try:
        if hasattr(obj, '__len__'):
            return len(obj) == 0
        elif hasattr(obj, 'size'):  # numpy 배열
            return obj.size == 0
        elif hasattr(obj, 'empty'):  # pandas DataFrame/Series
            return obj.empty
        else:
            return not bool(obj)
    except (ValueError, TypeError):
        # numpy 배열의 truth value 에러 등을 처리
        try:
            return not bool(obj)
        except:
            return True
