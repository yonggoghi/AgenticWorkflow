from typing import List, Dict, Any
import re
import logging

logger = logging.getLogger(__name__)

def validate_text_input(text: str) -> str:
    """
    텍스트 입력 검증 및 정리 함수
    
    MMS 텍스트 처리 전에 입력된 텍스트의 유효성을 검증하고
    처리에 적합한 형태로 정리합니다.
    
    Args:
        text (str): 검증할 입력 텍스트
        
    Returns:
        str: 정리된 텍스트
        
    Raises:
        ValueError: 비어있거나 잘못된 형식의 입력
        
    Example:
        clean_text = validate_text_input("  [SK텔레콤] 혜택 안내  ")
    """
    # 타입 검증: 문자열이 아닌 경우 에러 발생
    if not isinstance(text, str):
        raise ValueError(f"텍스트 입력이 문자열이 아닙니다: {type(text)}")
    
    # 앞뒤 공백 제거
    text = text.strip()
    
    # 빈 문자열 검사
    if not text:
        raise ValueError("빈 텍스트는 처리할 수 없습니다")
    
    # 최대 길이 제한 (LLM 토큰 제한 및 성능 고려)
    if len(text) > 10000:
        logger.warning(f"텍스트가 너무 깁니다 ({len(text)} 문자). 처음 10000자만 사용합니다.")
        text = text[:10000]
    
    return text


def select_most_comprehensive(strings: List[str]) -> List[str]:
    """
    Select the most comprehensive string from a list of overlapping strings.
    Returns the longest string that contains other strings as substrings.
    
    Args:
        strings: List of strings to filter
        
    Returns:
        List of most comprehensive strings (usually one, but could be multiple if no containment)
    """
    if not strings:
        return []
    
    # Remove duplicates and sort by length (longest first)
    unique_strings = list(set(strings))
    unique_strings.sort(key=len, reverse=True)
    
    result = []
    
    for current in unique_strings:
        # Check if current string contains any of the strings already in result
        is_contained = any(current in existing for existing in result)
        
        # Check if current string contains other strings not yet in result
        # contains_others = any(other in current for other in unique_strings if other != current and other not in result)
        
        # If current is not contained by existing results
        if not is_contained:
            # Remove any strings from result that are contained in current
            result = [r for r in result if r not in current]
            result.append(current)
    
    return result

def preprocess_text(text):
    """텍스트 전처리 (특수문자 제거, 공백 정규화)"""
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def replace_special_chars_with_space(text):
    """
    문자열에서 특수 문자를 공백으로 변환하는 함수
    
    Args:
        text (str): 변환할 문자열
        
    Returns:
        str: 특수 문자가 공백으로 변환된 문자열
    """
    # 영문자, 숫자, 한글을 제외한 모든 문자를 공백으로 변환
    return re.sub(r'[^a-zA-Z0-9가-힣\s]', ' ', text)

def filter_specific_terms(strings: List[str]) -> List[str]:
    """중복되거나 포함 관계에 있는 용어들 필터링"""
    unique_strings = list(set(strings))
    unique_strings.sort(key=len, reverse=True)
    
    filtered = []
    for s in unique_strings:
        if not any(s in other for other in filtered):
            filtered.append(s)
    
    return filtered

def extract_ngram_candidates(text, min_n=1, max_n=5):
    """
    상품명 매칭을 위한 n-gram 후보 추출
    
    Args:
        text: 입력 텍스트
        min_n: 최소 단어 수
        max_n: 최대 단어 수
        
    Returns:
        list of dict: [{'ngram': [...], 'text': '...', 'start': 0, 'end': 2}, ...]
    """
    words = text.split()
    candidates = []
    
    for n in range(min_n, min(max_n + 1, len(words) + 1)):
        for i in range(len(words) - n + 1):
            window = words[i:i + n]
            candidates.append({
                'ngram': window,
                'text': ' '.join(window),
                'start_idx': i,
                'end_idx': i + n - 1,
                'n': n
            })
    
    return candidates
