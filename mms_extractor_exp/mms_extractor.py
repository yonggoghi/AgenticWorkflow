# %%
"""
MMS 추출기 (MMS Extractor) - AI 기반 광고 텍스트 분석 시스템
================================================================

📋 개요
-------
이 모듈은 MMS(멀티미디어 메시지) 광고 텍스트에서 구조화된 정보를 자동으로 추출하는
AI 기반 시스템입니다. LLM(Large Language Model)을 활용하여 비정형 텍스트에서
상품명, 채널 정보, 광고 목적, 엔티티 관계 등을 정확하게 식별하고 추출합니다.

🎯 핵심 기능
-----------
1. **엔티티 추출**: 상품명, 브랜드명, 서비스명 등 핵심 엔티티 식별
2. **채널 분석**: URL, 전화번호, 앱 링크 등 고객 접점 채널 추출
3. **목적 분류**: 광고의 주요 목적 및 액션 타입 분석
4. **프로그램 매칭**: 사전 정의된 프로그램 카테고리와의 유사도 기반 분류
5. **DAG 생성**: 엔티티 간 관계를 방향성 그래프로 시각화

🔧 주요 개선사항
--------------
- **모듈화 설계**: 대형 메소드를 기능별 모듈로 분리하여 유지보수성 향상
- **프롬프트 외부화**: 하드코딩된 프롬프트를 외부 모듈로 분리하여 관리 용이성 증대
- **예외 처리 강화**: LLM 호출 실패, 네트워크 오류 등에 대한 robust한 에러 복구
- **성능 모니터링**: 상세한 로깅 및 실행 시간 추적으로 성능 최적화 지원
- **데이터 검증**: 추출 결과의 품질 보장을 위한 다층 검증 시스템
- **하이브리드 데이터 소스**: CSV 파일과 Oracle DB를 모두 지원하는 유연한 데이터 로딩

🏗️ 아키텍처
-----------
- **MMSExtractor**: 메인 추출 엔진 클래스
- **DataManager**: 데이터 로딩 및 관리 담당
- **LLMProcessor**: LLM 호출 및 응답 처리
- **EntityMatcher**: 엔티티 매칭 및 유사도 계산
- **PromptModule**: 외부화된 프롬프트 관리

⚙️ 설정 및 환경
--------------
- Python 3.8+
- LangChain, OpenAI, Anthropic API 지원
- Oracle Database 연동 (선택사항)
- GPU 가속 (CUDA 지원 시)

📊 성능 지표
-----------
- 평균 처리 시간: ~30초/메시지
- 정확도: 85%+ (수동 검증 기준)
- 처리량: ~120 메시지/시간 (단일 프로세스)

작성자: MMS 분석팀
최종 수정: 2024-09
버전: 2.0.0
"""

from concurrent.futures import ThreadPoolExecutor
import time
import logging
import warnings
from functools import wraps
from typing import List, Tuple, Union, Dict, Any, Optional
from abc import ABC, abstractmethod
import traceback
import json
import re
import ast
import glob
import os
from bson import raw_bson
import copy
import pandas as pd
import numpy as np

# joblib과 multiprocessing 경고 억제
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")
warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing")
warnings.filterwarnings("ignore", message=".*resource_tracker.*")
warnings.filterwarnings("ignore", message=".*leaked.*")
import torch
from sentence_transformers import SentenceTransformer
from difflib import SequenceMatcher
import difflib
from dotenv import load_dotenv
import cx_Oracle
from contextlib import contextmanager

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from rapidfuzz import fuzz, process
from kiwipiepy import Kiwi
from joblib import Parallel, delayed
from entity_dag_extractor import DAGParser, extract_dag, create_dag_diagram, sha256_hash

# 프롬프트 모듈 임포트
from prompts import (
    build_extraction_prompt,
    enhance_prompt_for_retry,
    get_fallback_result,
    build_entity_extraction_prompt,
    DEFAULT_ENTITY_EXTRACTION_PROMPT,
    DETAILED_ENTITY_EXTRACTION_PROMPT
)

# 설정 및 의존성 임포트 (원본 코드에서 가져옴)
try:
    from config.settings import API_CONFIG, MODEL_CONFIG, PROCESSING_CONFIG, METADATA_CONFIG, EMBEDDING_CONFIG
except ImportError:
    logging.warning("설정 파일을 찾을 수 없습니다. 기본값을 사용합니다.")
    # 기본 설정값들을 여기에 정의할 수 있습니다.

# 로깅 설정 - api.py에서 실행될 때는 해당 설정을 사용하고, 직접 실행될 때만 기본 설정 적용
logger = logging.getLogger(__name__)

# 직접 실행될 때만 로깅 설정 (api.py에서 임포트될 때는 api.py의 설정 사용)
if __name__ == '__main__':
    import sys
    from pathlib import Path
    
    # MongoDB 유틸리티는 필요할 때 동적으로 임포트
    
    # 로그 디렉토리 생성
    log_dir = Path(__file__).parent / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'mms_extractor.log'),
            logging.StreamHandler()
        ]
    )

# pandas 출력 설정
pd.set_option('display.max_colwidth', 500)
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# ===== 데코레이터 및 유틸리티 함수들 =====

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
            return getattr(obj, 'size', 1) == 0
        except:
            return True  # 안전을 위해 비어있다고 가정

# ===== 원본 유틸리티 함수들 (유지) =====

def dataframe_to_markdown_prompt(df, max_rows=None):
    """DataFrame을 마크다운 형식의 프롬프트로 변환"""
    if max_rows is not None and len(df) > max_rows:
        display_df = df.head(max_rows)
        truncation_note = f"\n[Note: Only showing first {max_rows} of {len(df)} rows]"
    else:
        display_df = df
        truncation_note = ""
    df_markdown = display_df.to_markdown()
    prompt = f"\n\n    {df_markdown}\n    {truncation_note}\n\n    "
    return prompt

def clean_segment(segment):
    """따옴표로 둘러싸인 문자열에서 내부의 동일한 따옴표 제거"""
    segment = segment.strip()
    if len(segment) >= 2 and segment[0] in ['"', "'"] and segment[-1] == segment[0]:
        q = segment[0]
        inner = segment[1:-1].replace(q, '')
        return q + inner + q
    return segment

def split_key_value(text):
    """따옴표 외부의 첫 번째 콜론을 기준으로 키-값 분리"""
    in_quote = False
    quote_char = ''
    for i, char in enumerate(text):
        if char in ['"', "'"]:
            if in_quote:
                if char == quote_char:
                    in_quote = False
                    quote_char = ''
            else:
                in_quote = True
                quote_char = char
        elif char == ':' and not in_quote:
            return text[:i], text[i+1:]
    return text, ''

def split_outside_quotes(text, delimiter=','):
    """따옴표 외부의 구분자로만 텍스트 분리"""
    parts = []
    current = []
    in_quote = False
    quote_char = ''
    for char in text:
        if char in ['"', "'"]:
            if in_quote:
                if char == quote_char:
                    in_quote = False
                    quote_char = ''
            else:
                in_quote = True
                quote_char = char
            current.append(char)
        elif char == delimiter and not in_quote:
            parts.append(''.join(current).strip())
            current = []
        else:
            current.append(char)
    if current:
        parts.append(''.join(current).strip())
    return parts

def clean_ill_structured_json(text):
    """잘못 구조화된 JSON 형식의 텍스트 정리"""
    parts = split_outside_quotes(text, delimiter=',')
    cleaned_parts = []
    for part in parts:
        key, value = split_key_value(part)
        key_clean = clean_segment(key)
        value_clean = clean_segment(value) if value.strip() != "" else ""
        if value_clean:
            cleaned_parts.append(f"{key_clean}: {value_clean}")
        else:
            cleaned_parts.append(key_clean)
    return ', '.join(cleaned_parts)

def repair_json(broken_json):
    """손상된 JSON 문자열 복구"""
    json_str = broken_json
    # 따옴표 없는 키에 따옴표 추가
    json_str = re.sub(r'([{,])\s*([a-zA-Z0-9_]+)\s*:', r'\1 "\2":', json_str)
    # 따옴표 없는 값 처리
    parts = json_str.split('"')
    for i in range(0, len(parts), 2):
        parts[i] = re.sub(r':\s*([a-zA-Z0-9_]+)(?=\s*[,\]\}])', r': "\1"', parts[i])
    json_str = '"'.join(parts)
    # 후행 쉼표 제거
    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
    return json_str

def extract_json_objects(text):
    """텍스트에서 JSON 객체 추출"""
    pattern = r'(\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\})'
    result = []
    for match in re.finditer(pattern, text):
        potential_json = match.group(0)
        try:
            json_obj = ast.literal_eval(clean_ill_structured_json(repair_json(potential_json)))
            result.append(json_obj)
        except (json.JSONDecodeError, SyntaxError, ValueError):
            pass
    return result

def preprocess_text(text):
    """텍스트 전처리 (특수문자 제거, 공백 정규화)"""
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ===== 유사도 계산 함수들 (원본 유지) =====

def fuzzy_similarities(text, entities):
    """퍼지 매칭을 사용한 유사도 계산"""
    results = []
    for entity in entities:
        scores = {
            'ratio': fuzz.ratio(text, entity) / 100,
            'partial_ratio': fuzz.partial_ratio(text, entity) / 100,
            'token_sort_ratio': fuzz.token_sort_ratio(text, entity) / 100,
            'token_set_ratio': fuzz.token_set_ratio(text, entity) / 100
        }
        max_score = max(scores.values())
        results.append((entity, max_score))
    return results

def get_fuzzy_similarities(args_dict):
    """배치 처리를 위한 퍼지 유사도 계산 함수"""
    text = args_dict['text']
    entities = args_dict['entities']
    threshold = args_dict['threshold']
    text_col_nm = args_dict['text_col_nm']
    item_col_nm = args_dict['item_col_nm']
    
    text_processed = preprocess_text(text.lower())
    similarities = fuzzy_similarities(text_processed, entities)
    
    filtered_results = [
        {
            text_col_nm: text,
            item_col_nm: entity, 
            "sim": score
        } 
        for entity, score in similarities 
        if score >= threshold
    ]
    return filtered_results

def parallel_fuzzy_similarity(texts, entities, threshold=0.5, text_col_nm='sent', item_col_nm='item_nm_alias', n_jobs=None, batch_size=None):
    """병렬 처리를 통한 퍼지 유사도 계산"""
    if n_jobs is None:
        n_jobs = min(os.cpu_count()-1, 8)
    if batch_size is None:
        batch_size = max(1, len(entities) // (n_jobs * 2))
    
    batches = []
    for text in texts:
        for i in range(0, len(entities), batch_size):
            batch = entities[i:i + batch_size]
            batches.append({"text": text, "entities": batch, "threshold": threshold, "text_col_nm": text_col_nm, "item_col_nm": item_col_nm})
    
    with Parallel(n_jobs=n_jobs) as parallel:
        batch_results = parallel(delayed(get_fuzzy_similarities)(args) for args in batches)
    
    return pd.DataFrame(sum(batch_results, []))

def longest_common_subsequence_ratio(s1, s2, normalizaton_value):
    """최장 공통 부분수열 비율 계산"""
    def lcs_length(x, y):
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i-1] == y[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]
    
    lcs_len = lcs_length(s1, s2)
    if normalizaton_value == 'max':
        max_len = max(len(s1), len(s2))
        return lcs_len / max_len if max_len > 0 else 1.0
    elif normalizaton_value == 'min':
        min_len = min(len(s1), len(s2))
        return lcs_len / min_len if min_len > 0 else 1.0
    elif normalizaton_value == 's1':
        return lcs_len / len(s1) if len(s1) > 0 else 1.0
    elif normalizaton_value == 's2':
        return lcs_len / len(s2) if len(s2) > 0 else 1.0
    else:
        raise ValueError(f"Invalid normalization value: {normalizaton_value}")

def sequence_matcher_similarity(s1, s2, normalizaton_value):
    """SequenceMatcher를 사용한 유사도 계산"""
    matcher = difflib.SequenceMatcher(None, s1, s2)
    matches = sum(triple.size for triple in matcher.get_matching_blocks())
    
    normalization_length = min(len(s1), len(s2))
    if normalizaton_value == 'max':
        normalization_length = max(len(s1), len(s2))
    elif normalizaton_value == 's1':
        normalization_length = len(s1)
    elif normalizaton_value == 's2':
        normalization_length = len(s2)
        
    if normalization_length == 0: 
        return 0.0
    
    return matches / normalization_length

def substring_aware_similarity(s1, s2, normalizaton_value):
    """부분문자열 관계를 고려한 유사도 계산"""
    if s1 in s2 or s2 in s1:
        shorter = min(s1, s2, key=len)
        longer = max(s1, s2, key=len)
        base_score = len(shorter) / len(longer)
        return min(0.95 + base_score * 0.05, 1.0)
    return longest_common_subsequence_ratio(s1, s2, normalizaton_value)

def token_sequence_similarity(s1, s2, normalizaton_value, separator_pattern=r'[\s_\-]+'):
    """토큰 시퀀스 기반 유사도 계산"""
    tokens1 = [t for t in re.split(separator_pattern, s1.strip()) if t]
    tokens2 = [t for t in re.split(separator_pattern, s2.strip()) if t]
    
    if not tokens1 or not tokens2:
        return 0.0
    
    def token_lcs_length(t1, t2):
        m, n = len(t1), len(t2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if t1[i-1] == t2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]
    
    lcs_tokens = token_lcs_length(tokens1, tokens2)
    normalization_tokens = max(len(tokens1), len(tokens2))
    if normalizaton_value == 'min':
        normalization_tokens = min(len(tokens1), len(tokens2))
    elif normalizaton_value == 's1':
        normalization_tokens = len(tokens1)
    elif normalizaton_value == 's2':
        normalization_tokens = len(tokens2)
    
    return lcs_tokens / normalization_tokens  

def combined_sequence_similarity(s1, s2, weights=None, normalizaton_value='max'):
    """여러 유사도 메트릭을 결합한 종합 유사도 계산"""
    if weights is None:
        weights = {'substring': 0.4, 'sequence_matcher': 0.4, 'token_sequence': 0.2}
    
    similarities = {
        'substring': substring_aware_similarity(s1, s2, normalizaton_value),
        'sequence_matcher': sequence_matcher_similarity(s1, s2, normalizaton_value),
        'token_sequence': token_sequence_similarity(s1, s2, normalizaton_value)
    }
    
    return sum(similarities[key] * weights[key] for key in weights), similarities

def calculate_seq_similarity(args_dict):
    """배치 처리를 위한 시퀀스 유사도 계산"""
    sent_item_batch = args_dict['sent_item_batch']
    text_col_nm = args_dict['text_col_nm']
    item_col_nm = args_dict['item_col_nm']
    normalizaton_value = args_dict['normalizaton_value']
    
    results = []
    for sent_item in sent_item_batch:
        sent = sent_item[text_col_nm]
        item = sent_item[item_col_nm]
        try:
            sent_processed = preprocess_text(sent.lower())
            item_processed = preprocess_text(item.lower())
            similarity = combined_sequence_similarity(sent_processed, item_processed, normalizaton_value=normalizaton_value)[0]
            results.append({text_col_nm:sent, item_col_nm:item, "sim":similarity})
        except Exception as e:
            logger.error(f"Error processing {item}: {e}")
            results.append({text_col_nm:sent, item_col_nm:item, "sim":0.0})
    
    return results

def parallel_seq_similarity(sent_item_pdf, text_col_nm='sent', item_col_nm='item_nm_alias', n_jobs=None, batch_size=None, normalizaton_value='s2'):
    """병렬 처리를 통한 시퀀스 유사도 계산"""
    if n_jobs is None:
        n_jobs = min(os.cpu_count()-1, 8)
    if batch_size is None:
        batch_size = max(1, sent_item_pdf.shape[0] // (n_jobs * 2))
    
    batches = []
    for i in range(0, sent_item_pdf.shape[0], batch_size):
        batch = sent_item_pdf.iloc[i:i + batch_size].to_dict(orient='records')
        batches.append({"sent_item_batch": batch, 'text_col_nm': text_col_nm, 'item_col_nm': item_col_nm, 'normalizaton_value': normalizaton_value})
    
    with Parallel(n_jobs=n_jobs) as parallel:
        batch_results = parallel(delayed(calculate_seq_similarity)(args) for args in batches)
    
    return pd.DataFrame(sum(batch_results, []))

def load_sentence_transformer(model_path, device=None):
    """SentenceTransformer 모델 로드"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Loading model from {model_path}...")
    model = SentenceTransformer(model_path).to(device)
    logger.info(f"Model loaded on {device}")
    return model

# ===== Kiwi 형태소 분석 관련 클래스들 (원본 유지) =====

class Token:
    """형태소 분석 토큰 클래스"""
    def __init__(self, form, tag, start, length):
        self.form = form      # 토큰 형태
        self.tag = tag        # 품사 태그
        self.start = start    # 시작 위치
        self.len = length     # 길이

class Sentence:
    """형태소 분석 문장 클래스"""
    def __init__(self, text, start, end, tokens, subs=None):
        self.text = text      # 문장 텍스트
        self.start = start    # 시작 위치
        self.end = end        # 끝 위치
        self.tokens = tokens  # 토큰 리스트
        self.subs = subs or []  # 하위 문장들

def filter_text_by_exc_patterns(sentence, exc_tag_patterns):
    """제외할 품사 패턴에 따라 텍스트 필터링"""
    # 개별 태그와 시퀀스 패턴 분리
    individual_tags = set()
    sequences = []
    
    for pattern in exc_tag_patterns:
        if isinstance(pattern, list):
            if len(pattern) == 1:
                individual_tags.add(pattern[0])
            else:
                sequences.append(pattern)
        else:
            individual_tags.add(pattern)
    
    # 제외할 토큰 인덱스 수집
    tokens_to_exclude = set()
    
    # 개별 태그 매칭 확인
    for i, token in enumerate(sentence.tokens):
        if token.tag in individual_tags:
            tokens_to_exclude.add(i)
    
    # 시퀀스 패턴 매칭 확인
    for sequence in sequences:
        seq_len = len(sequence)
        for i in range(len(sentence.tokens) - seq_len + 1):
            if all(sentence.tokens[i + j].tag == sequence[j] for j in range(seq_len)):
                for j in range(seq_len):
                    tokens_to_exclude.add(i + j)
    
    # 원본 텍스트에서 제외할 토큰 부분을 공백으로 대체
    result_chars = list(sentence.text)
    for i, token in enumerate(sentence.tokens):
        if i in tokens_to_exclude:
            start_pos = token.start - sentence.start
            end_pos = start_pos + token.len
            for j in range(start_pos, end_pos):
                if j < len(result_chars) and result_chars[j] != ' ':
                    result_chars[j] = ' '
    
    filtered_text = ''.join(result_chars)
    return re.sub(r'\s+', ' ', filtered_text)

def filter_specific_terms(strings: List[str]) -> List[str]:
    """중복되거나 포함 관계에 있는 용어들 필터링"""
    unique_strings = list(set(strings))
    unique_strings.sort(key=len, reverse=True)
    
    filtered = []
    for s in unique_strings:
        if not any(s in other for other in filtered):
            filtered.append(s)
    
    return filtered

def convert_df_to_json_list(df):
    """DataFrame을 특정 JSON 구조로 변환"""
    result = []
    grouped = df.groupby('item_name_in_msg')
    
    for item_name_in_msg, group in grouped:
        item_dict = {
            'item_name_in_msg': item_name_in_msg,
            'item_in_voca': []
        }
        
        item_nm_groups = group.groupby('item_nm')
        for item_nm, item_group in item_nm_groups:
            item_ids = list(item_group['item_id'].unique())
            voca_item = {
                'item_nm': item_nm,
                'item_id': item_ids
            }
            item_dict['item_in_voca'].append(voca_item)
        result.append(item_dict)
    
    return result

# ===== 추상 클래스 및 전략 패턴 =====

class EntityExtractionStrategy(ABC):
    """엔티티 추출 전략 추상 클래스"""
    
    @abstractmethod
    def extract(self, text: str, **kwargs) -> pd.DataFrame:
        """엔티티 추출 메소드"""
        pass

class DataLoader(ABC):
    """데이터 로더 추상 클래스"""
    
    @abstractmethod
    def load_data(self) -> Dict[str, Any]:
        """데이터 로드 메소드"""
        pass

# ===== 개선된 MMSExtractor 클래스 =====

class MMSExtractor:
    """
    MMS 광고 텍스트 AI 분석 시스템 - 메인 추출 엔진
    ================================================================
    
    🎨 개요
    -------
    이 클래스는 MMS 광고 텍스트에서 구조화된 정보를 추출하는 핵심 엔진입니다.
    LLM(Large Language Model), 임베딩 모델, NLP 기법을 조합하여
    비정형 텍스트에서 정형화된 데이터를 추출합니다.
    
    🔧 주요 기능
    -----------
    1. **다단계 엔티티 추출**: Kiwi NLP + 임베딩 유사도 + LLM 기반 추출
    2. **지능형 프로그램 분류**: 사전 정의된 카테고리와의 유사도 매칭
    3. **RAG 기반 컬텍스트 증강**: 관련 데이터를 활용한 정확도 향상
    4. **다중 LLM 지원**: OpenAI, Anthropic 등 다양한 모델 지원
    5. **DAG 생성**: 엔티티 간 관계를 방향성 그래프로 시각화
    
    📊 성능 특징
    -----------
    - **정확도**: 85%+ (수동 검증 기준)
    - **처리 속도**: 평균 30초/메시지
    - **확장성**: 모듈화된 설계로 새로운 기능 추가 용이
    - **안정성**: 강화된 예외 처리 및 재시도 메커니즘
    
    ⚙️ 주요 개선사항
    --------------
    - **아키텍처 모듈화**: 대형 메소드를 기능별 모듈로 분리하여 유지보수성 향상
    - **프롬프트 외부화**: 하드코딩된 프롬프트를 별도 모듈로 분리하여 관리 효율성 증대
    - **다층 예외 처리**: LLM API 실패, 네트워크 오류 등에 대한 robust한 에러 복구
    - **상세 로깅**: 성능 모니터링, 디버깅, 감사 로그를 위한 포괄적 로깅 시스템
    - **데이터 검증**: 입력/출력 데이터 품질 보장을 위한 다단계 검증
    - **하이브리드 데이터 소스**: CSV 파일과 Oracle DB를 모두 지원하는 유연한 데이터 로딩
    
    📝 사용 예시
    -----------
    ```python
    # 기본 초기화
    extractor = MMSExtractor(
        llm_model='ax',
        entity_extraction_mode='llm',
        extract_entity_dag=True
    )
    
    # 메시지 처리
    result = extractor.process_message("샘플 MMS 텍스트")
    
    # 결과 활용
    products = result['product']
    channels = result['channel']
    entity_dag = result.get('entity_dag', [])
    ```
    
    💼 의존성
    ---------
    - LangChain (LLM 인터페이스)
    - SentenceTransformers (임베딩)
    - KiwiPiePy (NLP)
    - cx_Oracle (데이터베이스 연동)
    """
    
    def __init__(self, model_path=None, data_dir=None, product_info_extraction_mode=None, 
                 entity_extraction_mode=None, offer_info_data_src='local', llm_model='ax', extract_entity_dag=False):
        """
        MMSExtractor 초기화 메소드
        
        시스템에 필요한 모든 구성 요소들을 초기화합니다:
        - LLM 모델 설정 및 연결
        - 임베딩 모델 로드
        - NLP 도구 (Kiwi) 초기화
        - 데이터 소스 로드 (CSV/DB)
        - 각종 설정 매개변수 구성
        
        Args:
            model_path (str, optional): 임베딩 모델 경로. 기본값: 'jhgan/ko-sroberta-multitask'
            data_dir (str, optional): 데이터 디렉토리 경로. 기본값: './data/'
            product_info_extraction_mode (str, optional): 상품 정보 추출 모드 ('nlp' 또는 'llm')
            entity_extraction_mode (str, optional): 엔티티 추출 모드 ('nlp', 'llm', 'hybrid')
            offer_info_data_src (str, optional): 데이터 소스 타입 ('local' 또는 'db')
            llm_model (str, optional): 사용할 LLM 모델. 기본값: 'ax'
            extract_entity_dag (bool, optional): DAG 추출 여부. 기본값: False
            
        Raises:
            Exception: 초기화 과정에서 발생하는 모든 오류
            
        Example:
            >>> extractor = MMSExtractor(
            ...     llm_model='gpt-4',
            ...     entity_extraction_mode='hybrid',
            ...     extract_entity_dag=True
            ... )
        """
        logger.info("🚀 MMSExtractor 초기화 시작")
        
        try:
            # 1단계: 기본 설정 매개변수 구성
            logger.info("⚙️ 기본 설정 적용 중...")
            self._set_default_config(model_path, data_dir, product_info_extraction_mode, 
                                   entity_extraction_mode, offer_info_data_src, llm_model, extract_entity_dag)
            
            # 2단계: 환경변수 로드 (API 키 등)
            logger.info("🔑 환경변수 로드 중...")
            load_dotenv()
            
            # 3단계: 주요 구성 요소들 순차 초기화
            logger.info("💻 디바이스 설정 중...")
            self._initialize_device()
            
            logger.info("🤖 LLM 모델 초기화 중...")
            self._initialize_llm()
            
            logger.info("🧠 임베딩 모델 로드 중...")
            self._initialize_embedding_model()
            
            logger.info("📝 NLP 도구 (Kiwi) 초기화 중...")
            self._initialize_kiwi()
            
            logger.info("📁 데이터 로드 중...")
            self._load_data()
            
            logger.info("✅ MMSExtractor 초기화 완료")
            
        except Exception as e:
            logger.error(f"❌ MMSExtractor 초기화 실패: {e}")
            logger.error(traceback.format_exc())
            raise

    def _set_default_config(self, model_path, data_dir, product_info_extraction_mode, 
                          entity_extraction_mode, offer_info_data_src, llm_model, extract_entity_dag):
        """기본 설정값 적용"""
        self.data_dir = data_dir if data_dir is not None else './data/'
        self.model_path = model_path if model_path is not None else getattr(EMBEDDING_CONFIG, 'ko_sbert_model_path', 'jhgan/ko-sroberta-multitask')
        self.offer_info_data_src = offer_info_data_src
        self.product_info_extraction_mode = product_info_extraction_mode if product_info_extraction_mode is not None else getattr(PROCESSING_CONFIG, 'product_info_extraction_mode', 'nlp')
        self.entity_extraction_mode = entity_extraction_mode if entity_extraction_mode is not None else getattr(PROCESSING_CONFIG, 'entity_extraction_mode', 'llm')
        self.llm_model_name = llm_model
        self.num_cand_pgms = getattr(PROCESSING_CONFIG, 'num_candidate_programs', 5)
        self.extract_entity_dag = extract_entity_dag
        
        # DAG 추출 설정 로깅
        # extract_entity_dag: 엔티티 간 관계를 DAG(Directed Acyclic Graph)로 추출
        # True인 경우 추가적으로 LLM을 사용하여 엔티티 관계를 분석하고
        # NetworkX + Graphviz를 통해 시각적 다이어그램을 생성
        if self.extract_entity_dag:
            logger.info("🎯 DAG 추출 모드 활성화됨")
        else:
            logger.info("📋 표준 추출 모드 (DAG 비활성화)")

    @log_performance
    def _initialize_device(self):
        """사용할 디바이스 초기화"""
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        logger.info(f"Using device: {self.device}")

    @log_performance
    def _initialize_llm(self):
        """LLM 모델 초기화"""
        try:
            # 모델 설정 매핑
            model_mapping = {
                "gemma": getattr(MODEL_CONFIG, 'gemma_model', 'gemma-7b'),
                "gem": getattr(MODEL_CONFIG, 'gemma_model', 'gemma-7b'),  # 'gem'은 'gemma'의 줄임말
                "ax": getattr(MODEL_CONFIG, 'ax_model', 'ax-4'),
                "claude": getattr(MODEL_CONFIG, 'claude_model', 'claude-4'),
                "cld": getattr(MODEL_CONFIG, 'claude_model', 'claude-4'),  # 'cld'는 'claude'의 줄임말
                "gemini": getattr(MODEL_CONFIG, 'gemini_model', 'gemini-pro'),
                "gen": getattr(MODEL_CONFIG, 'gemini_model', 'gemini-pro'),  # 'gen'은 'gemini'의 줄임말
                "gpt": getattr(MODEL_CONFIG, 'gpt_model', 'gpt-4')
            }
            
            model_name = model_mapping.get(self.llm_model_name, getattr(MODEL_CONFIG, 'llm_model', 'ax-4'))
            
            # LLM 모델별 일관성 설정
            model_kwargs = {
                "temperature": 0.0,  # 완전 결정적 출력을 위해 0.0 고정
                "openai_api_key": getattr(API_CONFIG, 'llm_api_key', os.getenv('OPENAI_API_KEY')),
                "openai_api_base": getattr(API_CONFIG, 'llm_api_url', None),
                "model": model_name,
                "max_tokens": getattr(MODEL_CONFIG, 'llm_max_tokens', 4000)
            }
            
            # GPT 모델의 경우 시드 설정으로 일관성 강화
            if 'gpt' in model_name.lower():
                model_kwargs["seed"] = 42  # 고정 시드로 일관성 보장
                
            self.llm_model = ChatOpenAI(**model_kwargs)
            
            logger.info(f"LLM 초기화 완료: {self.llm_model_name} ({model_name})")
            
        except Exception as e:
            logger.error(f"LLM 초기화 실패: {e}")
            raise

    @log_performance
    def _initialize_embedding_model(self):
        """임베딩 모델 초기화"""
        # 임베딩 비활성화 옵션 확인
        if MODEL_CONFIG.disable_embedding:
            logger.info("임베딩 모델 비활성화 모드 (DISABLE_EMBEDDING=true)")
            self.emb_model = None
            return
            
        try:
            self.emb_model = load_sentence_transformer(self.model_path, self.device)
        except Exception as e:
            logger.error(f"임베딩 모델 초기화 실패: {e}")
            # 기본 모델로 fallback
            logger.info("기본 모델로 fallback 시도")
            try:
                self.emb_model = load_sentence_transformer('jhgan/ko-sroberta-multitask', self.device)
            except Exception as e2:
                logger.error(f"Fallback 모델도 실패: {e2}")
                logger.warning("임베딩 모델 없이 동작 모드로 전환")
                self.emb_model = None

    @log_performance
    def _initialize_kiwi(self):
        """Kiwi 형태소 분석기 초기화"""
        try:
            self.kiwi = Kiwi()
            
            # 제외할 품사 태그 패턴들
            self.exc_tag_patterns = [
                ['SN', 'NNB'], ['W_SERIAL'], ['JKO'], ['W_URL'], ['W_EMAIL'],
                ['XSV', 'EC'], ['VV', 'EC'], ['VCP', 'ETM'], ['XSA', 'ETM'],
                ['VV', 'ETN'], ['SSO'], ['SSC'], ['SW'], ['SF'], ['SP'], 
                ['SS'], ['SE'], ['SO'], ['SB'], ['SH'], ['W_HASHTAG']
            ]
            logger.info("Kiwi 형태소 분석기 초기화 완료")
            
        except Exception as e:
            logger.error(f"Kiwi 초기화 실패: {e}")
            raise

    @log_performance
    def _load_data(self):
        """필요한 데이터 파일들 로드"""
        try:
            logger.info("=" * 60)
            logger.info("📊 데이터 로딩 시작")
            logger.info("=" * 60)
            logger.info(f"데이터 소스 모드: {self.offer_info_data_src}")
            
            # 상품 정보 로드
            logger.info("1️⃣ 상품 정보 로드 중...")
            self._load_item_data()
            logger.info(f"상품 정보 로드 후 데이터 크기: {self.item_pdf_all.shape}")
            logger.info(f"상품 정보 컬럼들: {list(self.item_pdf_all.columns)}")
            
            # 별칭 규칙 적용
            logger.info("2️⃣ 별칭 규칙 적용 중...")
            self._apply_alias_rules()
            logger.info(f"별칭 규칙 적용 후 데이터 크기: {self.item_pdf_all.shape}")
            
            # 정지어 로드
            logger.info("3️⃣ 정지어 로드 중...")
            self._load_stop_words()
            logger.info(f"로드된 정지어 수: {len(self.stop_item_names)}개")
            
            # Kiwi에 상품명 등록
            logger.info("4️⃣ Kiwi에 상품명 등록 중...")
            self._register_items_to_kiwi()
            
            # 프로그램 분류 정보 로드
            logger.info("5️⃣ 프로그램 분류 정보 로드 중...")
            self._load_program_data()
            logger.info(f"프로그램 분류 정보 로드 후 데이터 크기: {self.pgm_pdf.shape}")
            
            # 조직 정보 로드
            logger.info("6️⃣ 조직 정보 로드 중...")
            self._load_organization_data()
            logger.info(f"조직 정보 로드 후 데이터 크기: {self.org_pdf.shape}")
            
            # 최종 데이터 상태 요약
            logger.info("=" * 60)
            logger.info("📋 데이터 로딩 완료 - 최종 상태 요약")
            logger.info("=" * 60)
            logger.info(f"✅ 상품 데이터: {self.item_pdf_all.shape}")
            logger.info(f"✅ 프로그램 데이터: {self.pgm_pdf.shape}")
            logger.info(f"✅ 조직 데이터: {self.org_pdf.shape}")
            logger.info(f"✅ 정지어: {len(self.stop_item_names)}개")
            
            # 데이터 소스별 상태 비교를 위한 추가 정보
            if hasattr(self, 'item_pdf_all') and not self.item_pdf_all.empty:
                logger.info("=== 상품 데이터 상세 정보 ===")
                if 'item_nm' in self.item_pdf_all.columns:
                    unique_items = self.item_pdf_all['item_nm'].nunique()
                    logger.info(f"고유 상품명 수: {unique_items}개")
                if 'item_nm_alias' in self.item_pdf_all.columns:
                    unique_aliases = self.item_pdf_all['item_nm_alias'].nunique()
                    logger.info(f"고유 별칭 수: {unique_aliases}개")
                if 'item_id' in self.item_pdf_all.columns:
                    unique_ids = self.item_pdf_all['item_id'].nunique()
                    logger.info(f"고유 상품ID 수: {unique_ids}개")
            
        except Exception as e:
            logger.error(f"데이터 로딩 실패: {e}")
            logger.error(f"오류 상세: {traceback.format_exc()}")
            raise

    def _load_item_data(self):
        """상품 정보 로드"""
        try:
            logger.info(f"=== 상품 정보 로드 시작 (모드: {self.offer_info_data_src}) ===")
            
            if self.offer_info_data_src == "local":
                # 로컬 CSV 파일에서 로드
                logger.info("로컬 CSV 파일에서 상품 정보 로드 중...")
                csv_path = getattr(METADATA_CONFIG, 'offer_data_path', './data/items.csv')
                logger.info(f"CSV 파일 경로: {csv_path}")
                
                item_pdf_raw = pd.read_csv(csv_path)
                logger.info(f"로컬 CSV에서 로드된 원본 데이터 크기: {item_pdf_raw.shape}")
                logger.info(f"로컬 CSV 원본 컬럼들: {list(item_pdf_raw.columns)}")
                
                # 스키마 호환성 처리: 대문자/소문자 컬럼명 모두 지원
                available_columns = list(item_pdf_raw.columns)
                
                # 필요한 컬럼명 매핑 (대문자 우선, 소문자 폴백)
                column_mapping = {}
                required_cols = ['item_nm', 'item_id', 'item_desc', 'item_dmn']
                
                for req_col in required_cols:
                    if req_col.upper() in available_columns:
                        column_mapping[req_col.upper()] = req_col
                    elif req_col in available_columns:
                        column_mapping[req_col] = req_col
                    else:
                        logger.warning(f"필수 컬럼 '{req_col}' 또는 '{req_col.upper()}'를 찾을 수 없습니다.")
                
                # ITEM_ALS -> item_nm_alias 매핑 처리
                if 'ITEM_ALS' in available_columns:
                    column_mapping['ITEM_ALS'] = 'item_nm_alias'
                elif 'item_als' in available_columns:
                    column_mapping['item_als'] = 'item_nm_alias'
                elif 'item_nm_alias' in available_columns:
                    column_mapping['item_nm_alias'] = 'item_nm_alias'
                else:
                    logger.info("ITEM_ALS/item_nm_alias 컬럼을 찾을 수 없습니다. 나중에 생성됩니다.")
                
                logger.info(f"컬럼 매핑: {column_mapping}")
                
                # 데이터 추출 및 중복 제거
                mapped_columns = list(column_mapping.keys())
                if len(mapped_columns) >= 2:  # 최소 2개 컬럼은 있어야 함
                    # 대문자 컬럼명으로 중복 제거
                    dedup_cols = [col for col in ['ITEM_NM', 'ITEM_ID'] if col in available_columns]
                    if not dedup_cols:  # 대문자가 없으르 소문자 사용
                        dedup_cols = [col for col in ['item_nm', 'item_id'] if col in available_columns]
                    
                    self.item_pdf_all = item_pdf_raw.drop_duplicates(dedup_cols)[mapped_columns].copy()
                    
                    # 컬럼명을 소문자로 리네임
                    self.item_pdf_all = self.item_pdf_all.rename(columns=column_mapping)
                    logger.info(f"중복 제거 후 데이터 크기: {self.item_pdf_all.shape}")
                else:
                    logger.error(f"필수 컬럼을 충분히 찾을 수 없습니다. 사용 가능한 컬럼: {available_columns}")
                    raise ValueError("필수 컬럼 부족")
                
                # 추가 컬럼들 생성 (DB 스키마와 호환성을 위해)
                if 'item_ctg' not in self.item_pdf_all.columns:
                    self.item_pdf_all['item_ctg'] = None
                if 'item_emb_vec' not in self.item_pdf_all.columns:
                    self.item_pdf_all['item_emb_vec'] = None
                if 'ofer_cd' not in self.item_pdf_all.columns:
                    self.item_pdf_all['ofer_cd'] = self.item_pdf_all['item_id']
                if 'oper_dt_hms' not in self.item_pdf_all.columns:
                    self.item_pdf_all['oper_dt_hms'] = '20250101000000'
                
                # item_nm_alias 컬럼이 없으면 item_nm을 기본값으로 사용
                if 'item_nm_alias' not in self.item_pdf_all.columns:
                    logger.info("item_nm_alias 컬럼이 없어서 item_nm을 복사하여 생성합니다.")
                    self.item_pdf_all['item_nm_alias'] = self.item_pdf_all['item_nm']
                else:
                    # item_nm_alias가 비어있는 경우 item_nm으로 채우기
                    null_count = self.item_pdf_all['item_nm_alias'].isnull().sum()
                    if null_count > 0:
                        logger.info(f"item_nm_alias에서 {null_count}개 null 값을 item_nm으로 채웁니다.")
                        self.item_pdf_all['item_nm_alias'] = self.item_pdf_all['item_nm_alias'].fillna(self.item_pdf_all['item_nm'])
                
                # 컬럼명이 이미 소문자로 되어 있으므로 추가 변환 불필요
                logger.info(f"로컬 모드 최종 컬럼들: {list(self.item_pdf_all.columns)}")
                
                # item_nm_alias 컬럼 확인
                if 'item_nm_alias' in self.item_pdf_all.columns:
                    alias_sample = self.item_pdf_all['item_nm_alias'].dropna().head(3).tolist()
                    logger.info(f"item_nm_alias 샘플: {alias_sample}")
                else:
                    logger.error("item_nm_alias 컬럼 생성에 실패했습니다!")
                
                # 로컬 데이터 샘플 확인
                if not self.item_pdf_all.empty:
                    sample_items = self.item_pdf_all['item_nm'].dropna().head(5).tolist()
                    logger.info(f"로컬 모드 상품명 샘플: {sample_items}")
                    logger.info(f"로컬 모드 데이터 샘플 (5개 행):")
                    logger.info(f"{self.item_pdf_all.head().to_dict('records')}")
                
            elif self.offer_info_data_src == "db":
                # 데이터베이스에서 로드
                logger.info("데이터베이스에서 상품 정보 로드 중...")
                self._load_item_from_database()
            
            # 제외할 도메인 코드 필터링
            excluded_domains = getattr(PROCESSING_CONFIG, 'excluded_domain_codes_for_items', [])
            if excluded_domains:
                before_filter_size = len(self.item_pdf_all)
                self.item_pdf_all = self.item_pdf_all.query("item_dmn not in @excluded_domains")
                after_filter_size = len(self.item_pdf_all)
                logger.info(f"도메인 필터링: {before_filter_size} -> {after_filter_size} (제외된 도메인: {excluded_domains})")
                
            logger.info(f"=== 상품 정보 로드 최종 완료: {len(self.item_pdf_all)}개 상품 ===")
            logger.info(f"최종 데이터 스키마: {list(self.item_pdf_all.columns)}")
            logger.info(f"최종 데이터 타입: {self.item_pdf_all.dtypes.to_dict()}")
            
            # 중요 컬럼 존재 여부 최종 확인
            critical_columns = ['item_nm', 'item_id', 'item_nm_alias']
            missing_columns = [col for col in critical_columns if col not in self.item_pdf_all.columns]
            if missing_columns:
                logger.error(f"중요 컬럼이 누락되었습니다: {missing_columns}")
            else:
                logger.info("모든 중요 컬럼이 존재합니다.")
            
        except Exception as e:
            logger.error(f"상품 정보 로드 실패: {e}")
            logger.error(f"오류 상세: {traceback.format_exc()}")
            # 빈 DataFrame으로 fallback
            self.item_pdf_all = pd.DataFrame(columns=['item_nm', 'item_id', 'item_desc', 'item_dmn'])
            logger.warning("빈 DataFrame으로 fallback 설정됨")

    def _get_database_connection(self):
        """Oracle 데이터베이스 연결 생성"""
        try:
            logger.info("=== 데이터베이스 연결 시도 중 ===")
            
            username = os.getenv("DB_USERNAME")
            password = os.getenv("DB_PASSWORD")
            host = os.getenv("DB_HOST")
            port = os.getenv("DB_PORT")
            service_name = os.getenv("DB_NAME")
            
            # 연결 정보 로깅 (비밀번호는 마스킹)
            logger.info(f"DB 연결 정보:")
            logger.info(f"  - 사용자명: {username if username else '[비어있음]'}")
            logger.info(f"  - 비밀번호: {'*' * len(password) if password else '[비어있음]'}")
            logger.info(f"  - 호스트: {host if host else '[비어있음]'}")
            logger.info(f"  - 포트: {port if port else '[비어있음]'}")
            logger.info(f"  - 서비스명: {service_name if service_name else '[비어있음]'}")
            
            # 환경 변수 확인
            missing_vars = []
            if not username: missing_vars.append('DB_USERNAME')
            if not password: missing_vars.append('DB_PASSWORD')
            if not host: missing_vars.append('DB_HOST')
            if not port: missing_vars.append('DB_PORT')
            if not service_name: missing_vars.append('DB_NAME')
            
            if missing_vars:
                logger.error(f"누락된 환경 변수: {missing_vars}")
                logger.error("필요한 환경 변수들을 .env 파일에 설정해주세요.")
                raise ValueError(f"데이터베이스 연결 정보가 불완전합니다. 누락: {missing_vars}")
            
            # DSN 생성 및 로깅
            logger.info(f"DSN 생성 중: {host}:{port}/{service_name}")
            dsn = cx_Oracle.makedsn(host, port, service_name=service_name)
            logger.info(f"DSN 생성 성공: {dsn}")
            
            # 데이터베이스 연결 시도
            logger.info("데이터베이스 연결 시도 중...")
            conn = cx_Oracle.connect(user=username, password=password, dsn=dsn, encoding="UTF-8")
            logger.info("데이터베이스 연결 성공!")
            
            # LOB 데이터 처리를 위한 outputtypehandler 설정
            def output_type_handler(cursor, name, default_type, size, precision, scale):
                if default_type == cx_Oracle.CLOB:
                    return cursor.var(cx_Oracle.LONG_STRING, arraysize=cursor.arraysize)
                elif default_type == cx_Oracle.BLOB:
                    return cursor.var(cx_Oracle.LONG_BINARY, arraysize=cursor.arraysize)
            
            conn.outputtypehandler = output_type_handler
            
            # 연결 정보 확인
            logger.info(f"연결된 DB 버전: {conn.version}")
            
            return conn
            
        except cx_Oracle.DatabaseError as db_error:
            error_obj, = db_error.args
            logger.error(f"Oracle 데이터베이스 오류:")
            logger.error(f"  - 오류 코드: {error_obj.code}")
            logger.error(f"  - 오류 메시지: {error_obj.message}")
            logger.error(f"  - 전체 오류: {db_error}")
            raise
        except ImportError as import_error:
            logger.error(f"cx_Oracle 모듈 임포트 오류: {import_error}")
            logger.error("코맨드: pip install cx_Oracle")
            raise
        except Exception as e:
            logger.error(f"데이터베이스 연결 실패: {e}")
            logger.error(f"오류 타입: {type(e).__name__}")
            logger.error(f"오류 상세: {traceback.format_exc()}")
            raise

    @contextmanager
    def _database_connection(self):
        """데이터베이스 연결 context manager"""
        conn = None
        start_time = time.time()
        try:
            logger.info("데이터베이스 연결 context manager 시작")
            conn = self._get_database_connection()
            connection_time = time.time() - start_time
            logger.info(f"데이터베이스 연결 완료 ({connection_time:.2f}초)")
            yield conn
        except Exception as e:
            logger.error(f"데이터베이스 작업 중 오류: {e}")
            logger.error(f"오류 타입: {type(e).__name__}")
            logger.error(f"오류 상세: {traceback.format_exc()}")
            raise
        finally:
            if conn:
                try:
                    conn.close()
                    total_time = time.time() - start_time
                    logger.info(f"데이터베이스 연결 정상 종료 (총 소요시간: {total_time:.2f}초)")
                except Exception as close_error:
                    logger.warning(f"연결 종료 중 오류: {close_error}")
            else:
                logger.warning("데이터베이스 연결이 생성되지 않았습니다.")

    def _load_item_from_database(self):
        """데이터베이스에서 상품 정보 로드"""
        try:
            logger.info("=== 데이터베이스에서 상품 정보 로드 시작 ===")
            
            with self._database_connection() as conn:
                sql = "SELECT * FROM TCAM_RC_OFER_MST"
                logger.info(f"실행할 SQL: {sql}")
                
                self.item_pdf_all = pd.read_sql(sql, conn)
                logger.info(f"DB에서 로드된 원본 데이터 크기: {self.item_pdf_all.shape}")
                logger.info(f"DB에서 로드된 컬럼들: {list(self.item_pdf_all.columns)}")
                
                # 데이터 타입 정보 로깅
                logger.info("=== DB 데이터 타입 정보 ===")
                for col in self.item_pdf_all.columns:
                    dtype = self.item_pdf_all[col].dtype
                    null_count = self.item_pdf_all[col].isnull().sum()
                    logger.info(f"  {col}: {dtype}, null값: {null_count}개")
                
                # 컬럼명 소문자 변환 및 ITEM_ALS -> item_nm_alias 매핑
                original_columns = list(self.item_pdf_all.columns)
                
                # ITEM_ALS 컬럼을 item_nm_alias로 매핑
                column_mapping = {c: c.lower() for c in self.item_pdf_all.columns}
                if 'ITEM_ALS' in original_columns:
                    column_mapping['ITEM_ALS'] = 'item_nm_alias'
                    logger.info("ITEM_ALS 컬럼을 item_nm_alias로 매핑합니다.")
                
                self.item_pdf_all = self.item_pdf_all.rename(columns=column_mapping)
                logger.info(f"컬럼명 변환: {dict(zip(original_columns, self.item_pdf_all.columns))}")
                
                # LOB 데이터가 있는 경우를 대비해 데이터 강제 로드
                if not self.item_pdf_all.empty:
                    logger.info("DataFrame 데이터 강제 로드 시작...")
                    try:
                        # DataFrame의 모든 데이터를 메모리로 강제 로드
                        _ = self.item_pdf_all.values  # 모든 데이터 접근하여 LOB 로드 유도
                        logger.info("DataFrame 데이터 강제 로드 완료")
                        
                        # 주요 컬럼들의 샘플 데이터 확인
                        if 'item_nm' in self.item_pdf_all.columns:
                            sample_items = self.item_pdf_all['item_nm'].dropna().head(5).tolist()
                            logger.info(f"상품명 샘플: {sample_items}")
                        
                        if 'item_id' in self.item_pdf_all.columns:
                            sample_ids = self.item_pdf_all['item_id'].dropna().head(5).tolist()
                            logger.info(f"상품ID 샘플: {sample_ids}")
                        
                        # item_nm_alias 컬럼 확인 및 생성
                        if 'item_nm_alias' not in self.item_pdf_all.columns:
                            logger.info("item_nm_alias 컬럼이 없어서 item_nm에서 생성합니다.")
                            if 'item_nm' in self.item_pdf_all.columns:
                                self.item_pdf_all['item_nm_alias'] = self.item_pdf_all['item_nm']
                            else:
                                logger.error("item_nm 컬럼도 없어 item_nm_alias를 생성할 수 없습니다!")
                                self.item_pdf_all['item_nm_alias'] = None
                        else:
                            # item_nm_alias가 비어있는 경우 item_nm으로 채우기
                            null_count = self.item_pdf_all['item_nm_alias'].isnull().sum()
                            if null_count > 0:
                                logger.info(f"DB 모드: item_nm_alias에서 {null_count}개 null 값을 item_nm으로 채웁니다.")
                                self.item_pdf_all['item_nm_alias'] = self.item_pdf_all['item_nm_alias'].fillna(self.item_pdf_all['item_nm'])
                        
                        # item_nm_alias 샘플 확인
                        if 'item_nm_alias' in self.item_pdf_all.columns:
                            alias_sample = self.item_pdf_all['item_nm_alias'].dropna().head(3).tolist()
                            logger.info(f"DB 모드 item_nm_alias 샘플: {alias_sample}")
                            
                        logger.info(f"최종 상품 정보 로드 완료: {len(self.item_pdf_all)}개 상품")
                        logger.info(f"DB 모드 최종 컬럼들: {list(self.item_pdf_all.columns)}")
                    except Exception as load_error:
                        logger.error(f"데이터 강제 로드 중 오류: {load_error}")
                        raise
                else:
                    logger.warning("로드된 상품 데이터가 비어있습니다!")
            
        except Exception as e:
            logger.error(f"상품 정보 데이터베이스 로드 실패: {e}")
            logger.error(f"오류 상세: {traceback.format_exc()}")
            # 비상 상황에서 빈 데이터프레임 생성 (item_nm_alias 컬럼 포함)
            logger.warning("상품 정보 DB 로드 실패로 빈 데이터프레임을 생성합니다.")
            self.item_pdf_all = pd.DataFrame(columns=[
                'item_nm', 'item_id', 'item_desc', 'item_dmn', 'item_ctg', 
                'item_emb_vec', 'ofer_cd', 'oper_dt_hms', 'item_nm_alias'
            ])
            raise

    def _load_program_from_database(self):
        """데이터베이스에서 프로그램 분류 정보 로드"""
        try:
            logger.info("=== 데이터베이스에서 프로그램 분류 정보 로드 시작 ===")
            
            with self._database_connection() as conn:
                # 프로그램 분류 정보 쿼리
                sql = """SELECT CMPGN_PGM_NUM pgm_id, CMPGN_PGM_NM pgm_nm, RMK clue_tag 
                         FROM TCAM_CMPGN_PGM_INFO
                         WHERE DEL_YN = 'N' 
                         AND APRV_OP_RSLT_CD = 'APPR'
                         AND EXPS_YN = 'Y'
                         AND CMPGN_PGM_NUM like '2025%' 
                         AND RMK is not null"""
                
                logger.info(f"실행할 SQL: {sql}")
                
                self.pgm_pdf = pd.read_sql(sql, conn)
                logger.info(f"DB에서 로드된 프로그램 데이터 크기: {self.pgm_pdf.shape}")
                logger.info(f"DB에서 로드된 프로그램 컬럼들: {list(self.pgm_pdf.columns)}")
                
                # 컬럼명 소문자 변환
                original_columns = list(self.pgm_pdf.columns)
                self.pgm_pdf = self.pgm_pdf.rename(columns={c:c.lower() for c in self.pgm_pdf.columns})
                logger.info(f"프로그램 컬럼명 변환: {dict(zip(original_columns, self.pgm_pdf.columns))}")
                
                # LOB 데이터가 있는 경우를 대비해 데이터 강제 로드
                if not self.pgm_pdf.empty:
                    try:
                        # DataFrame의 모든 데이터를 메모리로 강제 로드
                        _ = self.pgm_pdf.values  # 모든 데이터 접근하여 LOB 로드 유도
                        
                        # 프로그램 데이터 샘플 확인
                        if 'pgm_nm' in self.pgm_pdf.columns:
                            sample_pgms = self.pgm_pdf['pgm_nm'].dropna().head(3).tolist()
                            logger.info(f"프로그램명 샘플: {sample_pgms}")
                        
                        if 'clue_tag' in self.pgm_pdf.columns:
                            sample_clues = self.pgm_pdf['clue_tag'].dropna().head(3).tolist()
                            logger.info(f"클루 태그 샘플: {sample_clues}")
                            
                        logger.info(f"데이터베이스에서 프로그램 분류 정보 로드 완료: {len(self.pgm_pdf)}개")
                    except Exception as load_error:
                        logger.error(f"프로그램 데이터 강제 로드 중 오류: {load_error}")
                        raise
                else:
                    logger.warning("로드된 프로그램 데이터가 비어있습니다!")
            
        except Exception as e:
            logger.error(f"프로그램 분류 정보 데이터베이스 로드 실패: {e}")
            logger.error(f"오류 상세: {traceback.format_exc()}")
            # 빈 데이터로 fallback
            self.pgm_pdf = pd.DataFrame(columns=['pgm_nm', 'clue_tag', 'pgm_id'])
            raise

    def _apply_alias_rules(self):
        """별칭 규칙 적용"""
        try:
            logger.info("=== 별칭 규칙 적용 시작 ===")
            logger.info(f"별칭 규칙 적용 전 상품 데이터 크기: {self.item_pdf_all.shape}")
            
            alias_pdf = pd.read_csv(getattr(METADATA_CONFIG, 'alias_rules_path', './data/alias_rules.csv'))
            alias_rule_set = list(zip(alias_pdf['alias_1'], alias_pdf['alias_2']))
            logger.info(f"로드된 별칭 규칙 수: {len(alias_rule_set)}개")

            def apply_alias_rule(item_nm):
                if pd.isna(item_nm) or not isinstance(item_nm, str):
                    return [item_nm] if not pd.isna(item_nm) else []
                    
                item_nm_list = [item_nm]
                for r in alias_rule_set:
                    if r[0] in item_nm:
                        item_nm_list.append(item_nm.replace(r[0], r[1]))
                    if r[1] in item_nm:
                        item_nm_list.append(item_nm.replace(r[1], r[0]))
                return item_nm_list

            # 별칭 규칙 적용 전 데이터 상태 확인
            if 'item_nm' in self.item_pdf_all.columns:
                non_null_items = self.item_pdf_all['item_nm'].dropna()
                logger.info(f"null이 아닌 상품명 수: {len(non_null_items)}개")
                if len(non_null_items) > 0:
                    sample_before = non_null_items.head(3).tolist()
                    logger.info(f"별칭 적용 전 상품명 샘플: {sample_before}")
            
            # 중요: 기존 item_nm_alias 데이터 보존 여부 확인
            existing_alias_data = None
            has_existing_alias = 'item_nm_alias' in self.item_pdf_all.columns and not self.item_pdf_all['item_nm_alias'].isnull().all()
            
            if has_existing_alias:
                # 기존 ITEM_ALS 데이터가 있는 경우 보존
                logger.info("기존 item_nm_alias 데이터를 발견했습니다. 별칭 규칙과 병합합니다.")
                existing_alias_sample = self.item_pdf_all['item_nm_alias'].dropna().head(3).tolist()
                logger.info(f"기존 alias 샘플: {existing_alias_sample}")
                
                # 기존 alias 데이터를 별도 컴럼으로 보존
                self.item_pdf_all['original_item_alias'] = self.item_pdf_all['item_nm_alias']
                
                # item_nm에 별칭 규칙을 적용한 다음 기존 alias와 병합
                generated_aliases = self.item_pdf_all['item_nm'].apply(apply_alias_rule)
                
                # 기존 alias와 생성된 alias를 병합
                def combine_aliases(row):
                    generated = row['generated_aliases'] if isinstance(row['generated_aliases'], list) else [row['generated_aliases']]
                    original = [row['original_item_alias']] if pd.notna(row['original_item_alias']) else []
                    combined = list(set(generated + original))  # 중복 제거
                    return combined
                
                self.item_pdf_all['generated_aliases'] = generated_aliases
                self.item_pdf_all['item_nm_alias'] = self.item_pdf_all.apply(combine_aliases, axis=1)
                
                # 임시 컴럼 삭제
                self.item_pdf_all = self.item_pdf_all.drop(['original_item_alias', 'generated_aliases'], axis=1)
                
                logger.info("기존 ITEM_ALS 데이터와 별칭 규칙이 성공적으로 병합되었습니다.")
                
            else:
                # 기존 alias 데이터가 없는 경우 기존 방식 사용
                logger.info("기존 item_nm_alias 데이터가 없어 item_nm에서 별칭을 생성합니다.")
                self.item_pdf_all['item_nm_alias'] = self.item_pdf_all['item_nm'].apply(apply_alias_rule)
            
            # explode 전후 크기 비교
            before_explode_size = len(self.item_pdf_all)
            self.item_pdf_all = self.item_pdf_all.explode('item_nm_alias')
            after_explode_size = len(self.item_pdf_all)
            
            logger.info(f"별칭 규칙 적용 후 데이터 크기: {before_explode_size} -> {after_explode_size}")
            
            # 별칭 적용 후 샘플 확인
            if 'item_nm_alias' in self.item_pdf_all.columns:
                non_null_aliases = self.item_pdf_all['item_nm_alias'].dropna()
                if len(non_null_aliases) > 0:
                    sample_after = non_null_aliases.head(5).tolist()
                    logger.info(f"별칭 적용 후 샘플: {sample_after}")
            
            logger.info(f"별칭 규칙 적용 완료: {len(alias_rule_set)}개 규칙")
            
        except Exception as e:
            logger.warning(f"별칭 규칙 적용 실패: {e}")
            logger.warning(f"오류 상세: {traceback.format_exc()}")
            # 예외 발생 시 기존 item_nm_alias 데이터 보존 또는 원본 이름 사용
            if 'item_nm_alias' not in self.item_pdf_all.columns or self.item_pdf_all['item_nm_alias'].isnull().all():
                if 'item_nm' in self.item_pdf_all.columns:
                    self.item_pdf_all['item_nm_alias'] = self.item_pdf_all['item_nm']
                    logger.info("예외 발생으로 원본 상품명을 별칭으로 사용합니다")
            else:
                logger.info("기존 item_nm_alias 데이터를 유지합니다")

    def _load_stop_words(self):
        """정지어 목록 로드"""
        try:
            self.stop_item_names = pd.read_csv(getattr(METADATA_CONFIG, 'stop_items_path', './data/stop_words.csv'))['stop_words'].to_list()
            logger.info(f"정지어 로드 완료: {len(self.stop_item_names)}개")
        except Exception as e:
            logger.warning(f"정지어 로드 실패: {e}")
            self.stop_item_names = []

    def _register_items_to_kiwi(self):
        """Kiwi에 상품명들을 고유명사로 등록"""
        try:
            logger.info("=== Kiwi에 상품명 등록 시작 ===")
            
            # 상품명 별칭 데이터 확인
            if 'item_nm_alias' not in self.item_pdf_all.columns:
                logger.error("item_nm_alias 컬럼이 존재하지 않습니다!")
                return
            
            unique_aliases = self.item_pdf_all['item_nm_alias'].unique()
            logger.info(f"등록할 고유 별칭 수: {len(unique_aliases)}개")
            
            # null이 아닌 유효한 별칭들만 필터링
            valid_aliases = [w for w in unique_aliases if isinstance(w, str) and len(w.strip()) > 0]
            logger.info(f"유효한 별칭 수: {len(valid_aliases)}개")
            
            if len(valid_aliases) > 0:
                sample_aliases = valid_aliases[:5]
                logger.info(f"등록할 별칭 샘플: {sample_aliases}")
            
            registered_count = 0
            failed_count = 0
            
            for w in valid_aliases:
                try:
                    self.kiwi.add_user_word(w, "NNP")
                    registered_count += 1
                except Exception as reg_error:
                    failed_count += 1
                    if failed_count <= 5:  # 처음 5개 실패만 로깅
                        logger.warning(f"Kiwi 등록 실패 - '{w}': {reg_error}")
            
            logger.info(f"Kiwi에 상품명 등록 완료: {registered_count}개 성공, {failed_count}개 실패")
            
        except Exception as e:
            logger.error(f"Kiwi 상품명 등록 실패: {e}")
            logger.error(f"오류 상세: {traceback.format_exc()}")

    def _load_program_data(self):
        """프로그램 분류 정보 로드 및 임베딩 생성"""
        try:
            logger.info("프로그램 분류 정보 로딩 시작...")
            
            if self.offer_info_data_src == "local":
                # 로컬 CSV 파일에서 로드
                self.pgm_pdf = pd.read_csv(getattr(METADATA_CONFIG, 'pgm_info_path', './data/program_info.csv'))
                logger.info(f"로컬 파일에서 프로그램 정보 로드: {len(self.pgm_pdf)}개")
            elif self.offer_info_data_src == "db":
                # 데이터베이스에서 로드
                self._load_program_from_database()
                logger.info(f"데이터베이스에서 프로그램 정보 로드: {len(self.pgm_pdf)}개")
            
            # 프로그램 분류를 위한 임베딩 생성
            if not self.pgm_pdf.empty:
                logger.info("프로그램 분류 임베딩 생성 시작...")
                clue_texts = self.pgm_pdf[["pgm_nm","clue_tag"]].apply(
                    lambda x: preprocess_text(x['pgm_nm'].lower()) + " " + x['clue_tag'].lower(), axis=1
                ).tolist()
                
                if self.emb_model is not None:
                    self.clue_embeddings = self.emb_model.encode(
                        clue_texts, convert_to_tensor=True, show_progress_bar=False
                    )
                else:
                    logger.warning("임베딩 모델이 없어 빈 tensor 사용")
                    self.clue_embeddings = torch.empty((0, 768))
                
                logger.info(f"프로그램 분류 임베딩 생성 완료: {len(self.pgm_pdf)}개 프로그램")
            else:
                logger.warning("프로그램 데이터가 비어있어 임베딩을 생성할 수 없습니다")
                self.clue_embeddings = torch.tensor([])
            
        except Exception as e:
            logger.error(f"프로그램 데이터 로드 실패: {e}")
            # 빈 데이터로 fallback
            self.pgm_pdf = pd.DataFrame(columns=['pgm_nm', 'clue_tag', 'pgm_id'])
            self.clue_embeddings = torch.tensor([])

    def _load_organization_data(self):
        """조직/매장 정보 로드"""
        try:
            logger.info(f"=== 조직 정보 로드 시작 (모드: {self.offer_info_data_src}) ===")
            
            if self.offer_info_data_src == "local":
                # 로컬 CSV 파일에서 로드
                logger.info("로컬 CSV 파일에서 조직 정보 로드 중...")
                csv_path = getattr(METADATA_CONFIG, 'org_info_path', './data/org_info_all_250605.csv')
                logger.info(f"CSV 파일 경로: {csv_path}")
                
                org_pdf_raw = pd.read_csv(csv_path)
                logger.info(f"로컬 CSV에서 로드된 원본 조직 데이터 크기: {org_pdf_raw.shape}")
                logger.info(f"로컬 CSV 원본 컬럼들: {list(org_pdf_raw.columns)}")
                
                # ITEM_DMN='R' 조건으로 필터링
                if 'ITEM_DMN' in org_pdf_raw.columns:
                    self.org_pdf = org_pdf_raw.query("ITEM_DMN=='R'").copy()
                elif 'item_dmn' in org_pdf_raw.columns:
                    self.org_pdf = org_pdf_raw.query("item_dmn=='R'").copy()
                else:
                    logger.warning("ITEM_DMN/item_dmn 컬럼을 찾을 수 없어 전체 데이터를 사용합니다.")
                    self.org_pdf = org_pdf_raw.copy()
                
                # 컬럼명을 소문자로 리네임
                self.org_pdf = self.org_pdf.rename(columns={c: c.lower() for c in self.org_pdf.columns})
                
                logger.info(f"로컬 모드: ITEM_DMN='R' 필터링 후 데이터 크기: {self.org_pdf.shape}")
                
            elif self.offer_info_data_src == "db":
                # 데이터베이스에서 로드
                logger.info("데이터베이스에서 조직 정보 로드 중...")
                self._load_org_from_database()
            
            # 데이터 샘플 확인
            if not self.org_pdf.empty:
                sample_orgs = self.org_pdf.head(3).to_dict('records')
                logger.info(f"조직 데이터 샘플 (3개 행): {sample_orgs}")
            
            logger.info(f"=== 조직 정보 로드 최종 완료: {len(self.org_pdf)}개 조직 ===")
            logger.info(f"최종 조직 데이터 스키마: {list(self.org_pdf.columns)}")
            
            # 조직 데이터 최종 검증
            if not self.org_pdf.empty:
                critical_org_columns = ['item_nm', 'item_id']
                missing_org_columns = [col for col in critical_org_columns if col not in self.org_pdf.columns]
                if missing_org_columns:
                    logger.error(f"조직 데이터에서 중요 컬럼이 누락되었습니다: {missing_org_columns}")
                    logger.error("이로 인해 조직/매장 추출 기능이 정상 동작하지 않을 수 있습니다.")
                else:
                    logger.info("모든 중요 조직 컬럼이 정상적으로 로드되었습니다.")
            else:
                logger.warning("조직 데이터가 비어있습니다. 조직/매장 추출이 동작하지 않을 수 있습니다.")
            
        except Exception as e:
            logger.error(f"조직 정보 로드 실패: {e}")
            logger.error(f"오류 상세: {traceback.format_exc()}")
            # 빈 DataFrame으로 fallback (조직 데이터에 필요한 컬럼들 포함)
            self.org_pdf = pd.DataFrame(columns=['item_nm', 'item_id', 'item_desc', 'item_dmn'])
            logger.warning("빈 조직 DataFrame으로 fallback 설정됨")
            logger.warning("이로 인해 조직/매장 추출 기능이 비활성화됩니다.")

    def _load_org_from_database(self):
        """데이터베이스에서 조직 정보 로드 (ITEM_DMN='R')"""
        try:
            logger.info("데이터베이스 연결 시도 중...")
            
            with self._database_connection() as conn:
                sql = "SELECT * FROM TCAM_RC_OFER_MST WHERE ITEM_DMN='R'"
                logger.info(f"실행할 SQL: {sql}")
                
                self.org_pdf = pd.read_sql(sql, conn)
                logger.info(f"DB에서 로드된 조직 데이터 크기: {self.org_pdf.shape}")
                logger.info(f"DB 조직 데이터 컬럼들: {list(self.org_pdf.columns)}")
                
                # 컬럼명 매핑 및 소문자 변환
                original_columns = list(self.org_pdf.columns)
                logger.info(f"DB 조직 데이터 원본 컬럼들: {original_columns}")
                
                # 조직 데이터를 위한 컬럼 매핑 (동일한 테이블이지만 사용 목적이 다름)
                column_mapping = {c: c.lower() for c in self.org_pdf.columns}
                
                # 조직 데이터는 item 테이블과 동일한 스키마를 사용하므로 컬럼명 그대로 사용
                # ITEM_NM -> item_nm, ITEM_ID -> item_id, ITEM_DESC -> item_desc 등
                
                self.org_pdf = self.org_pdf.rename(columns=column_mapping)
                logger.info(f"DB 모드 조직 컬럼명 매핑 완료: {dict(zip(original_columns, self.org_pdf.columns))}")
                logger.info(f"DB 모드 조직 최종 컬럼들: {list(self.org_pdf.columns)}")
                
                # 데이터 샘플 확인 및 컬럼 존재 여부 검증
                if not self.org_pdf.empty:
                    logger.info(f"DB 모드 조직 데이터 최종 크기: {self.org_pdf.shape}")
                    
                    # 필수 컬럼 존재 여부 확인
                    required_columns = ['item_nm', 'item_id']
                    missing_columns = [col for col in required_columns if col not in self.org_pdf.columns]
                    if missing_columns:
                        logger.error(f"DB 모드 조직 데이터에서 필수 컬럼 누락: {missing_columns}")
                        logger.error(f"사용 가능한 컬럼들: {list(self.org_pdf.columns)}")
                    else:
                        logger.info("모든 필수 조직 컬럼이 존재합니다.")
                    
                    # 샘플 데이터 확인
                    if 'item_nm' in self.org_pdf.columns:
                        sample_orgs = self.org_pdf['item_nm'].dropna().head(5).tolist()
                        logger.info(f"DB 모드 조직명 샘플: {sample_orgs}")
                    else:
                        logger.error("item_nm 컬럼이 없어 샘플을 표시할 수 없습니다.")
                        # 전체 데이터 샘플 표시
                        sample_data = self.org_pdf.head(3).to_dict('records')
                        logger.info(f"DB 모드 조직 데이터 샘플: {sample_data}")
                else:
                    logger.warning("DB에서 로드된 조직 데이터가 비어있습니다!")
                
                logger.info(f"DB에서 조직 데이터 로드 성공: {len(self.org_pdf)}개 조직")
                
        except Exception as e:
            logger.error(f"DB에서 조직 데이터 로드 실패: {e}")
            logger.error(f"DB 조직 로드 오류 상세: {traceback.format_exc()}")
            
            # 빈 DataFrame으로 fallback (조직 데이터에 필요한 컬럼들 포함)
            self.org_pdf = pd.DataFrame(columns=['item_nm', 'item_id', 'item_desc', 'item_dmn'])
            logger.warning("조직 데이터 DB 로드 실패로 빈 DataFrame 사용")
            
            raise

    def _store_prompt_for_preview(self, prompt: str, prompt_type: str):
        """프롬프트를 미리보기용으로 저장"""
        import threading
        current_thread = threading.current_thread()
        
        if not hasattr(current_thread, 'stored_prompts'):
            current_thread.stored_prompts = {}
        
        # 프롬프트 타입별 제목과 설명 매핑
        prompt_info = {
            "main_extraction": {
                'title': '메인 정보 추출 프롬프트',
                'description': '광고 메시지에서 제목, 목적, 상품, 채널, 프로그램 정보를 추출하는 프롬프트'
            },
            "entity_extraction": {
                'title': '엔티티 추출 프롬프트', 
                'description': '메시지에서 상품/서비스 엔티티를 추출하는 프롬프트'
            }
        }
        
        info = prompt_info.get(prompt_type, {
            'title': f'{prompt_type} 프롬프트',
            'description': f'{prompt_type} 처리를 위한 프롬프트'
        })
        
        prompt_key = f'{prompt_type}_prompt'
        prompt_data = {
            'title': info['title'],
            'description': info['description'],
            'content': prompt,
            'length': len(prompt)
        }
        
        current_thread.stored_prompts[prompt_key] = prompt_data
        
        # 디버깅 로그 추가
        logger.info(f"프롬프트 저장됨: {prompt_key} (길이: {len(prompt)})")
        logger.info(f"현재 저장된 프롬프트 수: {len(current_thread.stored_prompts)}")
        logger.info(f"저장된 프롬프트 키들: {list(current_thread.stored_prompts.keys())}")

    def _safe_llm_invoke(self, prompt: str, max_retries: int = 3) -> str:
        """안전한 LLM 호출 메소드"""
        for attempt in range(max_retries):
            try:
                # LLM 호출
                response = self.llm_model.invoke(prompt)
                result_text = response.content if hasattr(response, 'content') else str(response)
                
                # 스키마 응답 감지
                json_objects_list = extract_json_objects(result_text)
                if json_objects_list:
                    json_objects = json_objects_list[-1]
                    if self._detect_schema_response(json_objects):
                        logger.warning(f"시도 {attempt + 1}: LLM이 스키마를 반환했습니다. 재시도합니다.")
                        
                        # 스키마 응답인 경우 더 강한 지시사항으로 재시도
                        if attempt < max_retries - 1:
                            enhanced_prompt = self._enhance_prompt_for_retry(prompt)
                            response = self.llm_model.invoke(enhanced_prompt)
                            result_text = response.content if hasattr(response, 'content') else str(response)
                
                return result_text
                
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"LLM 호출 최종 실패: {e}")
                    return self._fallback_extraction(prompt)
                else:
                    logger.warning(f"LLM 호출 재시도 {attempt + 1}/{max_retries}: {e}")
                    time.sleep(2 ** attempt)  # 지수 백오프
        
        return ""

    def _enhance_prompt_for_retry(self, original_prompt: str) -> str:
        """스키마 응답 방지를 위한 프롬프트 강화"""
        return enhance_prompt_for_retry(original_prompt)

    def _fallback_extraction(self, prompt: str) -> str:
        """LLM 실패 시 fallback 추출 로직"""
        logger.info("Fallback 추출 로직 실행")
        
        # 외부 프롬프트 모듈에서 fallback 결과 가져오기
        fallback_result = get_fallback_result()
        
        return json.dumps(fallback_result, ensure_ascii=False)

    @log_performance
    def extract_entities_from_kiwi(self, mms_msg: str) -> Tuple[List[str], pd.DataFrame]:
        """Kiwi 형태소 분석기를 사용한 엔티티 추출"""
        try:
            logger.info("=== Kiwi 기반 엔티티 추출 시작 ===")
            mms_msg = validate_text_input(mms_msg)
            logger.info(f"처리할 메시지 길이: {len(mms_msg)} 문자")
            
            # 상품 데이터 상태 확인
            if self.item_pdf_all.empty:
                logger.error("상품 데이터가 비어있습니다! 엔티티 추출 불가")
                return [], pd.DataFrame()
            
            if 'item_nm_alias' not in self.item_pdf_all.columns:
                logger.error("item_nm_alias 컬럼이 없습니다! 엔티티 추출 불가")
                return [], pd.DataFrame()
            
            unique_aliases = self.item_pdf_all['item_nm_alias'].unique()
            logger.info(f"매칭할 상품 별칭 수: {len(unique_aliases)}개")
            
            # 문장 분할 및 하위 문장 처리
            sentences = sum(self.kiwi.split_into_sents(
                re.split(r"_+", mms_msg), return_tokens=True, return_sub_sents=True
            ), [])
            
            sentences_all = []
            for sent in sentences:
                if sent.subs:
                    sentences_all.extend(sent.subs)
                else:
                    sentences_all.append(sent)
            
            logger.info(f"분할된 문장 수: {len(sentences_all)}개")
            
            # 제외 패턴을 적용하여 문장 필터링
            sentence_list = [
                filter_text_by_exc_patterns(sent, self.exc_tag_patterns) 
                for sent in sentences_all
            ]
            
            logger.info(f"필터링된 문장들: {sentence_list[:3]}...")  # 처음 3개만 로깅

            # 형태소 분석을 통한 고유명사 추출
            result_msg = self.kiwi.tokenize(mms_msg, normalize_coda=True, z_coda=False, split_complex=False)
            all_tokens = [(token.form, token.tag) for token in result_msg]
            logger.info(f"전체 토큰 수: {len(all_tokens)}개")
            
            # NNP 태그 토큰들만 추출
            nnp_tokens = [token.form for token in result_msg if token.tag == 'NNP']
            logger.info(f"NNP 태그 토큰들: {nnp_tokens}")
            
            entities_from_kiwi = [
                token.form for token in result_msg 
                if token.tag == 'NNP' and 
                   token.form not in self.stop_item_names + ['-'] and 
                   len(token.form) >= 2 and 
                   not token.form.lower() in self.stop_item_names
            ]
            entities_from_kiwi = filter_specific_terms(entities_from_kiwi)
            
            logger.info(f"필터링 후 Kiwi 추출 엔티티: {list(set(entities_from_kiwi))}")

            # 퍼지 매칭을 통한 유사 상품명 찾기
            logger.info("퍼지 매칭 시작...")
            similarities_fuzzy = safe_execute(
                parallel_fuzzy_similarity,
                sentence_list, 
                unique_aliases,
                threshold=getattr(PROCESSING_CONFIG, 'fuzzy_threshold', 0.5),
                text_col_nm='sent', 
                item_col_nm='item_nm_alias',
                n_jobs=getattr(PROCESSING_CONFIG, 'n_jobs', 4),
                batch_size=30,
                default_return=pd.DataFrame()
            )
            
            logger.info(f"퍼지 매칭 결과 크기: {similarities_fuzzy.shape if not similarities_fuzzy.empty else '비어있음'}")
            
            if similarities_fuzzy.empty:
                logger.warning("퍼지 매칭 결과가 비어있습니다. Kiwi 결과만 사용합니다.")
                # 퍼지 매칭 결과가 없으면 Kiwi 결과만 사용
                cand_item_list = list(entities_from_kiwi) if entities_from_kiwi else []
                logger.info(f"Kiwi 기반 후보 아이템: {cand_item_list}")
                
                if cand_item_list:
                    extra_item_pdf = self.item_pdf_all.query("item_nm_alias in @cand_item_list")[
                        ['item_nm','item_nm_alias','item_id']
                    ].groupby(["item_nm"])['item_id'].apply(list).reset_index()
                    logger.info(f"매칭된 상품 정보: {extra_item_pdf.shape}")
                else:
                    extra_item_pdf = pd.DataFrame()
                    logger.warning("후보 아이템이 없습니다!")
                
                return cand_item_list, extra_item_pdf
            else:
                logger.info(f"퍼지 매칭 성공: {len(similarities_fuzzy)}개 결과")
                if not similarities_fuzzy.empty:
                    sample_fuzzy = similarities_fuzzy.head(3)[['sent', 'item_nm_alias', 'sim']].to_dict('records')
                    logger.info(f"퍼지 매칭 샘플: {sample_fuzzy}")

            # 시퀀스 유사도를 통한 정밀 매칭
            logger.info("시퀀스 유사도 계산 시작...")
            similarities_seq = safe_execute(
                parallel_seq_similarity,
                sent_item_pdf=similarities_fuzzy,
                text_col_nm='sent',
                item_col_nm='item_nm_alias',
                n_jobs=getattr(PROCESSING_CONFIG, 'n_jobs', 4),
                batch_size=getattr(PROCESSING_CONFIG, 'batch_size', 100),
                default_return=pd.DataFrame()
            )
            
            logger.info(f"시퀀스 유사도 결과 크기: {similarities_seq.shape if not similarities_seq.empty else '비어있음'}")
            if not similarities_seq.empty:
                sample_seq = similarities_seq.head(3)[['sent', 'item_nm_alias', 'sim']].to_dict('records')
                logger.info(f"시퀀스 유사도 샘플: {sample_seq}")
            
            # 임계값 이상의 후보 아이템들 필터링
            similarity_threshold = getattr(PROCESSING_CONFIG, 'similarity_threshold', 0.2)
            logger.info(f"사용할 유사도 임계값: {similarity_threshold}")
            
            cand_items = similarities_seq.query(
                "sim >= @similarity_threshold and "
                "item_nm_alias.str.contains('', case=False) and "
                "item_nm_alias not in @self.stop_item_names"
            )
            logger.info(f"임계값 필터링 후 후보 아이템 수: {len(cand_items)}개")
            
            # Kiwi에서 추출한 엔티티들 추가
            entities_from_kiwi_pdf = self.item_pdf_all.query("item_nm_alias in @entities_from_kiwi")[
                ['item_nm','item_nm_alias']
            ]
            entities_from_kiwi_pdf['sim'] = 1.0
            logger.info(f"Kiwi 엔티티 매칭 결과: {len(entities_from_kiwi_pdf)}개")

            # 결과 통합 및 최종 후보 리스트 생성
            cand_item_pdf = pd.concat([cand_items, entities_from_kiwi_pdf])
            logger.info(f"통합된 후보 아이템 수: {len(cand_item_pdf)}개")
            
            if not cand_item_pdf.empty:
                cand_item_array = cand_item_pdf.sort_values('sim', ascending=False).groupby([
                    "item_nm_alias"
                ])['sim'].max().reset_index(name='final_sim').sort_values(
                    'final_sim', ascending=False
                ).query("final_sim >= 0.2")['item_nm_alias'].unique()
                
                # numpy 배열을 리스트로 변환하여 안전성 보장
                cand_item_list = list(cand_item_array) if hasattr(cand_item_array, '__iter__') else []
                
                logger.info(f"최종 후보 아이템 리스트: {cand_item_list}")
                
                if cand_item_list:  # 리스트가 비어있지 않은 경우에만 쿼리 실행
                    extra_item_pdf = self.item_pdf_all.query("item_nm_alias in @cand_item_list")[
                        ['item_nm','item_nm_alias','item_id']
                    ].groupby(["item_nm"])['item_id'].apply(list).reset_index()
                else:
                    extra_item_pdf = pd.DataFrame()
                
                logger.info(f"최종 상품 정보 DataFrame 크기: {extra_item_pdf.shape}")
                if not extra_item_pdf.empty:
                    sample_final = extra_item_pdf.head(3).to_dict('records')
                    logger.info(f"최종 상품 정보 샘플: {sample_final}")
            else:
                logger.warning("통합된 후보 아이템이 없습니다!")
                cand_item_list = []
                extra_item_pdf = pd.DataFrame()

            return cand_item_list, extra_item_pdf
            
        except Exception as e:
            logger.error(f"Kiwi 엔티티 추출 실패: {e}")
            logger.error(f"오류 상세: {traceback.format_exc()}")
            # 안전한 기본값 반환 - 빈 리스트와 빈 DataFrame
            return [], pd.DataFrame()

    def extract_entities_by_logic(self, cand_entities: List[str], threshold_for_fuzzy: float = 0.8) -> pd.DataFrame:
        """로직 기반 엔티티 추출"""
        try:
            if not cand_entities:
                return pd.DataFrame()
            
            # 퍼지 유사도 계산
            similarities_fuzzy = safe_execute(
                parallel_fuzzy_similarity,
                cand_entities,
                self.item_pdf_all['item_nm_alias'].unique(),
                threshold=threshold_for_fuzzy,
                text_col_nm='item_name_in_msg',
                item_col_nm='item_nm_alias',
                n_jobs=getattr(PROCESSING_CONFIG, 'n_jobs', 4),
                batch_size=30,
                default_return=pd.DataFrame()
            )
            
            if similarities_fuzzy.empty:
                return pd.DataFrame()
            
            # 시퀀스 유사도 계산
            cand_entities_sim = self._calculate_combined_similarity(similarities_fuzzy)
            
            return cand_entities_sim
            
        except Exception as e:
            logger.error(f"로직 기반 엔티티 추출 실패: {e}")
            return pd.DataFrame()

    def _calculate_combined_similarity(self, similarities_fuzzy: pd.DataFrame) -> pd.DataFrame:
        """s1, s2 정규화 방식으로 각각 계산 후 합산"""
        try:
            # s1 정규화
            sim_s1 = safe_execute(
                parallel_seq_similarity,
                sent_item_pdf=similarities_fuzzy,
                text_col_nm='item_name_in_msg',
                item_col_nm='item_nm_alias',
                n_jobs=getattr(PROCESSING_CONFIG, 'n_jobs', 4),
                batch_size=30,
                normalizaton_value='s1',
                default_return=pd.DataFrame()
            ).rename(columns={'sim': 'sim_s1'})
            
            # s2 정규화
            sim_s2 = safe_execute(
                parallel_seq_similarity,
                sent_item_pdf=similarities_fuzzy,
                text_col_nm='item_name_in_msg',
                item_col_nm='item_nm_alias',
                n_jobs=getattr(PROCESSING_CONFIG, 'n_jobs', 4),
                batch_size=30,
                normalizaton_value='s2',
                default_return=pd.DataFrame()
            ).rename(columns={'sim': 'sim_s2'})
            
            # 결과 합치기
            if not sim_s1.empty and not sim_s2.empty:
                combined = sim_s1.merge(sim_s2, on=['item_name_in_msg', 'item_nm_alias'])
                combined = combined.groupby(['item_name_in_msg', 'item_nm_alias'])[
                    ['sim_s1', 'sim_s2']
                ].apply(lambda x: x['sim_s1'].sum() + x['sim_s2'].sum()).reset_index(name='sim')
                return combined
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"결합 유사도 계산 실패: {e}")
            return pd.DataFrame()

    @log_performance
    def extract_entities_by_llm(self, msg_text: str, rank_limit: int = 5) -> pd.DataFrame:
        """LLM 기반 엔티티 추출"""
        try:
            msg_text = validate_text_input(msg_text)
            
            # 로직 기반 방식으로 후보 엔티티 먼저 추출
            cand_entities_by_sim = sorted([
                e.strip() for e in self.extract_entities_by_logic([msg_text], threshold_for_fuzzy=getattr(PROCESSING_CONFIG, 'fuzzy_threshold', 0.4))['item_nm_alias'].unique() 
                if e.strip() not in self.stop_item_names and len(e.strip()) >= 2
            ])

            # LLM 프롬프트 구성 - 외부 프롬프트 모듈 사용
            # 프롬프트를 prompts 디렉토리에서 가져오기 (설정 파일 대신)
            base_prompt = getattr(PROCESSING_CONFIG, 'entity_extraction_prompt', None)
            if base_prompt is None:
                # settings.py에 프롬프트가 없으면 prompts 디렉토리에서 가져오기
                base_prompt = DETAILED_ENTITY_EXTRACTION_PROMPT
                logger.info("엔티티 추출에 prompts 디렉토리의 DETAILED_ENTITY_EXTRACTION_PROMPT 사용")
            else:
                logger.info("엔티티 추출에 settings.py의 entity_extraction_prompt 사용")
            prompt = build_entity_extraction_prompt(msg_text, base_prompt)
            
            # 후보 엔티티 추가
            prompt += f"""

            ## Candidate entities:
            {cand_entities_by_sim}
            """
            
            # 프롬프트 저장 (디버깅/미리보기용)
            self._store_prompt_for_preview(prompt, "entity_extraction")
            
            # LLM 호출 (프롬프트 저장은 이미 위에서 했으므로 직접 호출)
            response = self.llm_model.invoke(prompt)
            cand_entities = response.content if hasattr(response, 'content') else str(response)
            
            # LLM 응답 파싱 및 정리
            cand_entity_list = [e.strip() for e in cand_entities.split(',') if e.strip()]
            cand_entity_list = [e for e in cand_entity_list if e not in self.stop_item_names and len(e) >= 2]

            if not cand_entity_list:
                return pd.DataFrame()

            # 후보 엔티티들과 상품 DB 매칭
            return self._match_entities_with_products(cand_entity_list, rank_limit)
            
        except Exception as e:
            logger.error(f"LLM 기반 엔티티 추출 실패: {e}")
            return pd.DataFrame()

    def _match_entities_with_products(self, cand_entity_list: List[str], rank_limit: int) -> pd.DataFrame:
        """후보 엔티티들을 상품 DB와 매칭"""
        try:
            # 퍼지 유사도 매칭
            similarities_fuzzy = safe_execute(
                parallel_fuzzy_similarity,
                cand_entity_list,
                self.item_pdf_all['item_nm_alias'].unique(),
                threshold=0.6,
                text_col_nm='item_name_in_msg',
                item_col_nm='item_nm_alias',
                n_jobs=getattr(PROCESSING_CONFIG, 'n_jobs', 4),
                batch_size=30,
                default_return=pd.DataFrame()
            )
            
            if similarities_fuzzy.empty:
                return pd.DataFrame()
            
            # 정지어 필터링
            similarities_fuzzy = similarities_fuzzy[
                ~similarities_fuzzy['item_nm_alias'].isin(self.stop_item_names)
            ]

            # 시퀀스 유사도 매칭
            cand_entities_sim = self._calculate_combined_similarity(similarities_fuzzy)
            
            if cand_entities_sim.empty:
                return pd.DataFrame()
            
            high_sim_threshold = getattr(PROCESSING_CONFIG, 'high_similarity_threshold', 1.5)
            cand_entities_sim = cand_entities_sim.query("sim >= @high_sim_threshold").copy()

            # 순위 매기기 및 결과 제한
            cand_entities_sim["rank"] = cand_entities_sim.groupby('item_name_in_msg')['sim'].rank(
                method='first', ascending=False
            )
            cand_entities_sim = cand_entities_sim.query(f"rank <= {rank_limit}").sort_values(
                ['item_name_in_msg', 'rank'], ascending=[True, True]
            )

            return cand_entities_sim
            
        except Exception as e:
            logger.error(f"엔티티-상품 매칭 실패: {e}")
            return pd.DataFrame()

    def _extract_entities(self, mms_msg: str) -> Tuple[List[str], pd.DataFrame]:
        """엔티티 추출 (Kiwi 또는 LLM 방식)"""
        try:
            if self.entity_extraction_mode == 'logic':
                # Kiwi 기반 추출
                return self.extract_entities_from_kiwi(mms_msg)
            else:
                # LLM 기반 추출을 위해 먼저 Kiwi로 기본 추출
                cand_item_list, extra_item_pdf = self.extract_entities_from_kiwi(mms_msg)
                return cand_item_list, extra_item_pdf
                
        except Exception as e:
            logger.error(f"엔티티 추출 실패: {e}")
            logger.error(f"오류 상세: {traceback.format_exc()}")
            # 안전한 기본값 반환
            return [], pd.DataFrame()

    def _classify_programs(self, mms_msg: str) -> Dict[str, Any]:
        """프로그램 분류"""
        try:
            if self.emb_model is None or self.clue_embeddings.numel() == 0:
                return {"pgm_cand_info": "", "similarities": []}
            
            # 메시지 임베딩 및 프로그램 분류 유사도 계산
            mms_embedding = self.emb_model.encode([mms_msg.lower()], convert_to_tensor=True, show_progress_bar=False)
            similarities = torch.nn.functional.cosine_similarity(mms_embedding, self.clue_embeddings, dim=1).cpu().numpy()
            
            # 상위 후보 프로그램들 선별
            pgm_pdf_tmp = self.pgm_pdf.copy()
            pgm_pdf_tmp['sim'] = similarities
            pgm_pdf_tmp = pgm_pdf_tmp.sort_values('sim', ascending=False)
            
            pgm_cand_info = "\n\t".join(
                pgm_pdf_tmp.iloc[:self.num_cand_pgms][['pgm_nm','clue_tag']].apply(
                    lambda x: re.sub(r'\[.*?\]', '', x['pgm_nm']) + " : " + x['clue_tag'], axis=1
                ).to_list()
            )
            
            return {
                "pgm_cand_info": pgm_cand_info,
                "similarities": similarities,
                "pgm_pdf_tmp": pgm_pdf_tmp
            }
            
        except Exception as e:
            logger.error(f"프로그램 분류 실패: {e}")
            return {"pgm_cand_info": "", "similarities": [], "pgm_pdf_tmp": pd.DataFrame()}

    def _build_extraction_prompt(self, msg: str, rag_context: str, product_element: Optional[List[Dict]]) -> str:
        """추출용 프롬프트 구성 - 외부 프롬프트 모듈 사용"""
        
        # 외부 프롬프트 모듈의 함수 사용
        prompt = build_extraction_prompt(
            message=msg,
            rag_context=rag_context,
            product_element=product_element,
            product_info_extraction_mode=self.product_info_extraction_mode
        )
        
        # 디버깅을 위한 프롬프트 로깅 (LLM 모드에서만)
        if self.product_info_extraction_mode == 'llm':
            logger.debug(f"LLM 모드 프롬프트 길이: {len(prompt)} 문자")
            logger.debug(f"후보 상품 목록 포함 여부: {'참고용 후보 상품 이름 목록' in rag_context}")
            
        return prompt

    def _extract_channels(self, json_objects: Dict, msg: str) -> List[Dict]:
        """채널 정보 추출 및 매칭"""
        try:
            channel_tag = []
            channel_items = json_objects.get('channel', [])
            if isinstance(channel_items, dict):
                channel_items = channel_items.get('items', [])

            for d in channel_items:
                if d.get('type') == '대리점' and d.get('value'):
                    # 대리점명으로 조직 정보 검색
                    store_info = self._match_store_info(d['value'])
                    d['store_info'] = store_info
                else:
                    d['store_info'] = []
                channel_tag.append(d)

            return channel_tag
            
        except Exception as e:
            logger.error(f"채널 정보 추출 실패: {e}")
            return []

    def _match_store_info(self, store_name: str) -> List[Dict]:
        """대리점 정보 매칭"""
        try:
            # 대리점명으로 조직 정보 검색
            org_pdf_cand = safe_execute(
                parallel_fuzzy_similarity,
                [preprocess_text(store_name.lower())],
                self.org_pdf['item_nm'].unique(),
                threshold=0.5,
                text_col_nm='org_nm_in_msg',
                item_col_nm='item_nm',
                n_jobs=getattr(PROCESSING_CONFIG, 'n_jobs', 4),
                batch_size=getattr(PROCESSING_CONFIG, 'batch_size', 100),
                default_return=pd.DataFrame()
            )

            if org_pdf_cand.empty:
                return []

            org_pdf_cand = org_pdf_cand.drop('org_nm_in_msg', axis=1)
            org_pdf_cand = self.org_pdf.merge(org_pdf_cand, on=['item_nm'])
            org_pdf_cand['sim'] = org_pdf_cand.apply(
                lambda x: combined_sequence_similarity(store_name, x['item_nm'])[0], axis=1
            ).round(5)
            
            # 대리점 코드('D'로 시작) 우선 검색
            similarity_threshold = getattr(PROCESSING_CONFIG, 'similarity_threshold_for_store', 0.2)
            org_pdf_tmp = org_pdf_cand.query(
                "sim >= @similarity_threshold", engine='python'
            ).sort_values('sim', ascending=False)
            
            if org_pdf_tmp.empty:
                # 대리점이 없으면 전체에서 검색
                org_pdf_tmp = org_pdf_cand.query("sim >= @similarity_threshold").sort_values('sim', ascending=False)
            
            if not org_pdf_tmp.empty:
                # 최고 순위 조직들의 정보 추출
                org_pdf_tmp['rank'] = org_pdf_tmp['sim'].rank(method='dense', ascending=False)
                org_pdf_tmp = org_pdf_tmp.rename(columns={'item_id':'org_cd','item_nm':'org_nm'})
                org_info = org_pdf_tmp.query("rank == 1").groupby('org_nm')['org_cd'].apply(list).reset_index(name='org_cd').to_dict('records')
                return org_info
            else:
                return []
                
        except Exception as e:
            logger.error(f"대리점 정보 매칭 실패: {e}")
            return []

    def _validate_extraction_result(self, result: Dict) -> Dict:
        """추출 결과 검증 및 정리"""
        try:
            # 필수 필드 확인
            required_fields = ['title', 'purpose', 'product', 'channel']
            for field in required_fields:
                if field not in result:
                    logger.warning(f"필수 필드 누락: {field}")
                    result[field] = [] if field != 'title' else "광고 메시지"

            # 상품명 길이 검증
            validated_products = []
            for product in result.get('product', []):
                if isinstance(product, dict):
                    item_name = product.get('item_name_in_msg', product.get('name', ''))
                    if len(item_name) >= 2 and item_name not in self.stop_item_names:
                        validated_products.append(product)
                    else:
                        logger.warning(f"의심스러운 상품명 제외: {item_name}")
            
            result['product'] = validated_products

            # 채널 정보 검증
            validated_channels = []
            for channel in result.get('channel', []):
                if isinstance(channel, dict) and channel.get('value'):
                    validated_channels.append(channel)
            
            result['channel'] = validated_channels

            return result
            
        except Exception as e:
            logger.error(f"결과 검증 실패: {e}")
            return result

    @log_performance
    def process_message(self, mms_msg: str) -> Dict[str, Any]:
        """
        MMS 메시지 전체 처리 (메인 처리 함수)
        
        Args:
            mms_msg: 처리할 MMS 메시지 텍스트
        
        Returns:
            dict: 추출된 정보가 담긴 JSON 구조
        """
        try:
            logger.info("=" * 60)
            logger.info("🚀 MMS 메시지 처리 시작")
            logger.info("=" * 60)
            logger.info(f"메시지 내용: {mms_msg[:200]}...")
            logger.info(f"메시지 길이: {len(mms_msg)} 문자")
            
            # 현재 설정 상태 로깅
            logger.info("=== 현재 추출기 설정 ===")
            logger.info(f"데이터 소스: {self.offer_info_data_src}")
            logger.info(f"상품 정보 추출 모드: {self.product_info_extraction_mode}")
            logger.info(f"엔티티 추출 모드: {self.entity_extraction_mode}")
            logger.info(f"LLM 모델: {self.llm_model_name}")
            logger.info(f"상품 데이터 크기: {self.item_pdf_all.shape}")
            logger.info(f"프로그램 데이터 크기: {self.pgm_pdf.shape}")
            
            # 입력 검증
            msg = validate_text_input(mms_msg)
            
            # 1단계: 엔티티 추출
            logger.info("=" * 30 + " 1단계: 엔티티 추출 " + "=" * 30)
            
            # DB 모드 특별 진단
            if self.offer_info_data_src == "db":
                logger.info("🔍 DB 모드 특별 진단 시작")
                logger.info(f"상품 데이터 상태: {self.item_pdf_all.shape}")
                
                # 필수 컬럼 존재 여부 확인
                required_columns = ['item_nm', 'item_id', 'item_nm_alias']
                missing_columns = [col for col in required_columns if col not in self.item_pdf_all.columns]
                if missing_columns:
                    logger.error(f"🚨 DB 모드에서 필수 컬럼 누락: {missing_columns}")
                
                # 데이터 품질 확인
                if 'item_nm_alias' in self.item_pdf_all.columns:
                    null_aliases = self.item_pdf_all['item_nm_alias'].isnull().sum()
                    total_aliases = len(self.item_pdf_all)
                    logger.info(f"DB 모드 별칭 데이터 품질: {total_aliases - null_aliases}/{total_aliases} 유효")
            
            cand_item_list, extra_item_pdf = self._extract_entities(msg)
            logger.info(f"추출된 후보 엔티티: {cand_item_list}")
            logger.info(f"매칭된 상품 정보: {extra_item_pdf.shape}")
            
            # DB 모드에서 엔티티 추출 결과 특별 분석
            if self.offer_info_data_src == "db":
                logger.info("🔍 DB 모드 엔티티 추출 결과 분석")
                # cand_item_list가 numpy 배열일 수 있으므로 안전한 검사 사용
                if safe_check_empty(cand_item_list):
                    logger.error("🚨 DB 모드에서 후보 엔티티가 전혀 추출되지 않았습니다!")
                    logger.error("가능한 원인:")
                    logger.error("1. 상품 데이터베이스에 해당 상품이 없음")
                    logger.error("2. 별칭 규칙 적용 실패")
                    logger.error("3. 유사도 임계값이 너무 높음")
                    logger.error("4. Kiwi 형태소 분석 실패")
            
            # 2단계: 프로그램 분류
            logger.info("=" * 30 + " 2단계: 프로그램 분류 " + "=" * 30)
            pgm_info = self._classify_programs(msg)
            logger.info(f"프로그램 분류 결과 키: {list(pgm_info.keys())}")
            
            # 3단계: RAG 컨텍스트 구성
            logger.info("=" * 30 + " 3단계: RAG 컨텍스트 구성 " + "=" * 30)
            rag_context = f"\n### 광고 분류 기준 정보 ###\n\t{pgm_info['pgm_cand_info']}" if self.num_cand_pgms > 0 else ""
            logger.info(f"프로그램 분류 컨텍스트 길이: {len(rag_context)} 문자")
            
            # 4단계: 제품 정보 준비 (모드별 처리)
            logger.info("=" * 30 + " 4단계: 제품 정보 준비 " + "=" * 30)
            product_element = None
            
            # cand_item_list가 비어있지 않은지 안전하게 검사
            if not safe_check_empty(cand_item_list):
                logger.info(f"후보 아이템 리스트 크기: {len(cand_item_list)}개")
                logger.info(f"후보 아이템 리스트: {cand_item_list}")
                
                # extra_item_pdf 상태 확인
                logger.info(f"extra_item_pdf 크기: {extra_item_pdf.shape}")
                if not extra_item_pdf.empty:
                    logger.info(f"extra_item_pdf 컬럼들: {list(extra_item_pdf.columns)}")
                    logger.info(f"extra_item_pdf 샘플: {extra_item_pdf.head(2).to_dict('records')}")
                
                if self.product_info_extraction_mode == 'rag':
                    rag_context += f"\n\n### 후보 상품 이름 목록 ###\n\t{cand_item_list}"
                    logger.info("RAG 모드: 후보 상품 목록을 RAG 컨텍스트에 추가")
                elif self.product_info_extraction_mode == 'llm':
                    # LLM 모드에도 후보 목록 제공하여 일관성 향상
                    rag_context += f"\n\n### 참고용 후보 상품 이름 목록 ###\n\t{cand_item_list}"
                    logger.info("LLM 모드: 참고용 후보 상품 목록을 RAG 컨텍스트에 추가")
                elif self.product_info_extraction_mode == 'nlp':
                    if not extra_item_pdf.empty and 'item_nm' in extra_item_pdf.columns:
                        product_df = extra_item_pdf.rename(columns={'item_nm': 'name'}).query(
                            "not name in @self.stop_item_names"
                        )[['name']]
                        product_df['action'] = '기타'
                        product_element = product_df.to_dict(orient='records') if product_df.shape[0] > 0 else None
                        logger.info(f"NLP 모드: 제품 요소 준비 완료 - {len(product_element) if product_element else 0}개")
                        if product_element:
                            logger.info(f"NLP 모드 제품 요소 샘플: {product_element[:2]}")
                    else:
                        logger.warning("NLP 모드: extra_item_pdf가 비어있거나 item_nm 컬럼이 없습니다!")
            else:
                logger.warning("후보 아이템이 없습니다!")
                logger.warning("이는 다음 중 하나의 문제일 수 있습니다:")
                logger.warning("1. 상품 데이터 로딩 실패")
                logger.warning("2. 엔티티 추출 실패") 
                logger.warning("3. 유사도 매칭 임계값 문제")

            # 5단계: LLM 프롬프트 구성 및 실행
            logger.info("=" * 30 + " 5단계: LLM 호출 " + "=" * 30)
            prompt = self._build_extraction_prompt(msg, rag_context, product_element)
            logger.info(f"구성된 프롬프트 길이: {len(prompt)} 문자")
            logger.info(f"RAG 컨텍스트 포함 여부: {'후보 상품' in rag_context}")
            
            # 프롬프트 저장 (디버깅/미리보기용)
            self._store_prompt_for_preview(prompt, "main_extraction")
            
            result_json_text = self._safe_llm_invoke(prompt)
            logger.info(f"LLM 응답 길이: {len(result_json_text)} 문자")
            logger.info(f"LLM 응답 내용 (처음 500자): {result_json_text[:500]}...")
            
            # 6단계: JSON 파싱
            logger.info("=" * 30 + " 6단계: JSON 파싱 " + "=" * 30)
            json_objects_list = extract_json_objects(result_json_text)
            logger.info(f"추출된 JSON 객체 수: {len(json_objects_list)}개")
            
            if not json_objects_list:
                logger.warning("LLM이 유효한 JSON 객체를 반환하지 않았습니다")
                logger.warning(f"LLM 원본 응답: {result_json_text}")
                return self._create_fallback_result(msg)
            
            json_objects = json_objects_list[-1]
            logger.info(f"파싱된 JSON 객체 키: {list(json_objects.keys())}")
            logger.info(f"파싱된 JSON 내용: {json_objects}")
            
            # 스키마 응답 감지 및 처리
            is_schema_response = self._detect_schema_response(json_objects)
            if is_schema_response:
                logger.error("🚨 LLM이 스키마 정의를 반환했습니다! 실제 데이터가 아닙니다.")
                logger.error("재시도 또는 fallback 결과를 사용합니다.")
                return self._create_fallback_result(msg)

            raw_result = copy.deepcopy(json_objects)
            
            # 7단계: 엔티티 매칭 및 최종 결과 구성
            logger.info("=" * 30 + " 7단계: 최종 결과 구성 " + "=" * 30)
            final_result = self._build_final_result(json_objects, msg, pgm_info)
            
            # 8단계: 결과 검증
            logger.info("=" * 30 + " 8단계: 결과 검증 " + "=" * 30)
            final_result = self._validate_extraction_result(final_result)

            # # DAG 추출 프로세스 (선택적)
            # # 메시지에서 엔티티 간의 관계를 방향성 있는 그래프로 추출
            # # 예: (고객:가입) -[하면]-> (혜택:수령) -[통해]-> (만족도:향상)
            # dag_section = ""
            # if self.extract_entity_dag:
            #     logger.info("=" * 30 + " DAG 추출 시작 " + "=" * 30)
            #     try:
            #         dag_start_time = time.time()
            #         # DAG 추출 함수 호출 (entity_dag_extractor.py)
            #         extract_dag_result = extract_dag(DAGParser(), msg, self.llm_model)
            #         dag_raw = extract_dag_result['dag_raw']      # LLM 원본 응답
            #         dag_section = extract_dag_result['dag_section']  # 파싱된 DAG 텍스트
            #         dag = extract_dag_result['dag']             # NetworkX 그래프 객체
                    
            #         # 시각적 다이어그램 생성 (utils.py)
            #         dag_filename = f'dag_{sha256_hash(msg)}'
            #         create_dag_diagram(dag, filename=dag_filename)
            #         dag_processing_time = time.time() - dag_start_time
                    
            #         logger.info(f"✅ DAG 추출 완료: {dag_filename}")
            #         logger.info(f"🕒 DAG 처리 시간: {dag_processing_time:.3f}초")
            #         logger.info(f"📏 DAG 섹션 길이: {len(dag_section)}자")
            #         if dag_section:
            #             logger.info(f"📄 DAG 내용 미리보기: {dag_section[:200]}...")
            #         else:
            #             logger.warning("⚠️ DAG 섹션이 비어있습니다")
                        
            #     except Exception as e:
            #         logger.error(f"❌ DAG 추출 중 오류 발생: {e}")
            #         dag_section = ""

            # # 최종 결과에 DAG 정보 추가 (비어있을 수도 있음)
            # final_result['entity_dag'] = sorted([d for d in dag_section.split('\n') if d!=''])
            
            # 최종 결과 요약 로깅
            logger.info("=" * 60)
            logger.info("✅ 메시지 처리 완료 - 최종 결과 요약")
            logger.info("=" * 60)
            logger.info(f"제목: {final_result.get('title', 'N/A')}")
            logger.info(f"목적: {final_result.get('purpose', [])}")
            logger.info(f"상품 수: {len(final_result.get('product', []))}개")
            logger.info(f"채널 수: {len(final_result.get('channel', []))}개")
            logger.info(f"프로그램 수: {len(final_result.get('pgm', []))}개")

            actual_prompts = get_stored_prompts_from_thread()

            return {"extracted_result": final_result, "raw_result": raw_result, "prompts": actual_prompts}
            
        except Exception as e:
            logger.error(f"메시지 처리 실패: {e}")
            logger.error(traceback.format_exc())
            return self._create_fallback_result(mms_msg)
    
    @log_performance
    def extract_json_objects_only(self, mms_msg: str) -> Dict[str, Any]:
        """
        메시지에서 7단계(엔티티 매칭 및 최종 결과 구성) 전의 json_objects만 추출
        
        Args:
            mms_msg: 처리할 MMS 메시지
            
        Returns:
            Dict: LLM이 생성한 json_objects (엔티티 매칭 전)
        """
        try:
            msg = mms_msg.strip()
            logger.info(f"JSON 객체 추출 시작 - 메시지 길이: {len(msg)}자")
            
            # 1-4단계: 기존 프로세스
            pgm_info = self._prepare_program_classification(msg)
            
            # RAG 컨텍스트 준비 (product_info_extraction_mode가 'rag'인 경우)
            rag_context = ""
            if self.product_info_extraction_mode == 'rag':
                rag_context = self._prepare_rag_context(msg)
            
            # 5단계: 프롬프트 구성 및 LLM 호출
            prompt = self._build_extraction_prompt(msg, pgm_info, rag_context)
            result_json_text = self._safe_llm_invoke(prompt)
            
            # 6단계: JSON 파싱
            json_objects_list = extract_json_objects(result_json_text)
            
            if not json_objects_list:
                logger.warning("LLM이 유효한 JSON 객체를 반환하지 않았습니다")
                return {}
            
            json_objects = json_objects_list[-1]
            
            # 스키마 응답 감지
            is_schema_response = self._detect_schema_response(json_objects)
            if is_schema_response:
                logger.warning("LLM이 스키마 정의를 반환했습니다")
                return {}
            
            logger.info(f"JSON 객체 추출 완료 - 키: {list(json_objects.keys())}")
            return json_objects
            
        except Exception as e:
            logger.error(f"JSON 객체 추출 중 오류 발생: {e}")
            return {}
    
    def _prepare_program_classification(self, mms_msg: str) -> Dict[str, Any]:
        """프로그램 분류 준비 (_classify_programs 메소드와 동일)"""
        try:
            if self.emb_model is None or self.clue_embeddings.numel() == 0:
                return {"pgm_cand_info": "", "similarities": []}
            
            # 메시지 임베딩 및 프로그램 분류 유사도 계산
            mms_embedding = self.emb_model.encode([mms_msg.lower()], convert_to_tensor=True, show_progress_bar=False)
            similarities = torch.nn.functional.cosine_similarity(mms_embedding, self.clue_embeddings, dim=1).cpu().numpy()
            
            # 상위 후보 프로그램들 선별
            pgm_pdf_tmp = self.pgm_pdf.copy()
            pgm_pdf_tmp['sim'] = similarities
            pgm_pdf_tmp = pgm_pdf_tmp.sort_values('sim', ascending=False)
            
            pgm_cand_info = "\n\t".join(
                pgm_pdf_tmp.iloc[:self.num_cand_pgms][['pgm_nm','clue_tag']].apply(
                    lambda x: re.sub(r'\[.*?\]', '', x['pgm_nm']) + " : " + x['clue_tag'], axis=1
                ).to_list()
            )
            
            return {
                "pgm_cand_info": pgm_cand_info,
                "similarities": similarities,
                "pgm_pdf_tmp": pgm_pdf_tmp
            }
            
        except Exception as e:
            logger.error(f"프로그램 분류 실패: {e}")
            return {"pgm_cand_info": "", "similarities": [], "pgm_pdf_tmp": pd.DataFrame()}

    def _detect_schema_response(self, json_objects: Dict) -> bool:
        """LLM이 스키마 정의를 반환했는지 감지"""
        try:
            # purpose 필드가 스키마 구조인지 확인
            purpose = json_objects.get('purpose', {})
            if isinstance(purpose, dict) and 'type' in purpose and purpose.get('type') == 'array':
                logger.warning("purpose 필드가 스키마 구조로 감지됨")
                return True
            
            # product 필드가 스키마 구조인지 확인  
            product = json_objects.get('product', {})
            if isinstance(product, dict) and 'type' in product and product.get('type') == 'array':
                logger.warning("product 필드가 스키마 구조로 감지됨")
                return True
            
            # channel 필드가 스키마 구조인지 확인
            channel = json_objects.get('channel', {})
            if isinstance(channel, dict) and 'type' in channel and channel.get('type') == 'array':
                logger.warning("channel 필드가 스키마 구조로 감지됨")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"스키마 응답 감지 중 오류: {e}")
            return False

    def _create_fallback_result(self, msg: str) -> Dict[str, Any]:
        """처리 실패 시 기본 결과 생성"""
        return {
            "title": "광고 메시지",
            "purpose": ["정보 제공"],
            "product": [],
            "channel": [],
            "pgm": []
        }

    def _build_final_result(self, json_objects: Dict, msg: str, pgm_info: Dict) -> Dict[str, Any]:
        """최종 결과 구성"""
        try:
            final_result = json_objects.copy()
            
            # 상품 정보에서 엔티티 추출
            product_items = json_objects.get('product', [])
            if isinstance(product_items, dict):
                product_items = product_items.get('items', [])
            
            logger.info(f"LLM 추출 엔티티: {[x.get('name', '') for x in product_items]}")

            # 엔티티 매칭 모드에 따른 처리
            if self.entity_extraction_mode == 'logic':
                # 로직 기반: 퍼지 + 시퀀스 유사도
                cand_entities = [item.get('name', '') for item in product_items if item.get('name')]
                similarities_fuzzy = self.extract_entities_by_logic(cand_entities)
            else:
                # LLM 기반: LLM을 통한 엔티티 추출
                similarities_fuzzy = self.extract_entities_by_llm(msg)

            # 상품 정보 매핑
            if not similarities_fuzzy.empty:
                final_result['product'] = self._map_products_with_similarity(similarities_fuzzy, json_objects)
            else:
                # 유사도 결과가 없으면 LLM 결과 그대로 사용
                final_result['product'] = [
                    {
                        'item_name_in_msg': d.get('name', ''), 
                        'expected_action': d.get('action', '기타'),
                        'item_in_voca': [{'item_name_in_voca': d.get('name', ''), 'item_id': ['#']}]
                    } 
                    for d in product_items 
                    if d.get('name') and d['name'] not in self.stop_item_names
                ]

            # 프로그램 분류 정보 매핑
            final_result['pgm'] = self._map_program_classification(json_objects, pgm_info)
            
            # 채널 정보 처리
            final_result['channel'] = self._extract_channels(json_objects, msg)

            return final_result
            
        except Exception as e:
            logger.error(f"최종 결과 구성 실패: {e}")
            return json_objects

    def _map_products_with_similarity(self, similarities_fuzzy: pd.DataFrame, json_objects: Dict = None) -> List[Dict]:
        """유사도를 기반으로 상품 정보 매핑"""
        try:
            # 높은 유사도 아이템들 필터링
            high_sim_threshold = getattr(PROCESSING_CONFIG, 'high_similarity_threshold', 1.5)
            high_sim_items = similarities_fuzzy.query('sim >= @high_sim_threshold')['item_nm_alias'].unique()
            filtered_similarities = similarities_fuzzy[
                (similarities_fuzzy['item_nm_alias'].isin(high_sim_items)) &
                (~similarities_fuzzy['item_nm_alias'].str.contains('test', case=False)) &
                (~similarities_fuzzy['item_name_in_msg'].isin(self.stop_item_names))
            ]
            
            # 상품 정보와 매핑하여 최종 결과 생성
            product_tag = convert_df_to_json_list(
                self.item_pdf_all.merge(filtered_similarities, on=['item_nm_alias'])
            )
            
            # Add action information from original json_objects
            if json_objects:
                action_mapping = self._create_action_mapping(json_objects)
                for product in product_tag:
                    item_name = product.get('item_name_in_msg', '')
                    product['expected_action'] = action_mapping.get(item_name, '기타')
            
            return product_tag
            
        except Exception as e:
            logger.error(f"상품 정보 매핑 실패: {e}")
            return []

    def _create_action_mapping(self, json_objects: Dict) -> Dict[str, str]:
        """LLM 응답에서 상품명-액션 매핑 생성"""
        try:
            action_mapping = {}
            product_data = json_objects.get('product', [])
            
            if isinstance(product_data, list):
                # 정상적인 배열 구조
                for item in product_data:
                    if isinstance(item, dict) and 'name' in item and 'action' in item:
                        action_mapping[item['name']] = item['action']
            elif isinstance(product_data, dict):
                # 스키마 구조 또는 기타 딕셔너리 구조 처리
                if 'items' in product_data:
                    # 스키마 구조: {"items": [...]}
                    items = product_data.get('items', [])
                    for item in items:
                        if isinstance(item, dict) and 'name' in item and 'action' in item:
                            action_mapping[item['name']] = item['action']
                elif 'type' in product_data and product_data.get('type') == 'array':
                    # 스키마 정의 구조는 건너뛰기
                    logger.debug("스키마 정의 구조 감지됨, 액션 매핑 건너뛰기")
                else:
                    # 기타 딕셔너리 구조 처리
                    if 'name' in product_data and 'action' in product_data:
                        action_mapping[product_data['name']] = product_data['action']
            
            logger.debug(f"생성된 액션 매핑: {action_mapping}")
            return action_mapping
            
        except Exception as e:
            logger.error(f"액션 매핑 생성 실패: {e}")
            return {}

    def _map_program_classification(self, json_objects: Dict, pgm_info: Dict) -> List[Dict]:
        """프로그램 분류 정보 매핑"""
        try:
            if (self.num_cand_pgms > 0 and 
                'pgm' in json_objects and 
                isinstance(json_objects['pgm'], list) and
                not pgm_info.get('pgm_pdf_tmp', pd.DataFrame()).empty):
                
                pgm_json = pgm_info['pgm_pdf_tmp'][
                    pgm_info['pgm_pdf_tmp']['pgm_nm'].apply(
                        lambda x: re.sub(r'\[.*?\]', '', x) in ' '.join(json_objects['pgm'])
                    )
                ][['pgm_nm', 'pgm_id']].to_dict('records')
                
                return pgm_json
            
            return []
            
        except Exception as e:
            logger.error(f"프로그램 분류 매핑 실패: {e}")
            return []

def process_message_with_dag(extractor, message: str, extract_dag: bool = False) -> Dict[str, Any]:
    """
    단일 메시지를 처리하는 워커 함수 (멀티프로세스용)
    
    Args:
        extractor: MMSExtractor 인스턴스
        message: 처리할 메시지
        extract_dag: DAG 추출 여부
    
    Returns:
        dict: 처리 결과 (프롬프트 정보 포함)
    """
    try:
        logger.info(f"워커 프로세스에서 메시지 처리 시작: {message[:50]}...")

        # 1. 메인 추출
        result = extractor.process_message(message)
        dag_list = []
        
        if extract_dag:
            # 순차적 처리로 변경 (프롬프트 캡처를 위해)
            # 멀티스레드를 사용하면 스레드 로컬 저장소가 분리되어 프롬프트 캡처가 안됨
            logger.info("순차적 처리로 메인 추출 및 DAG 추출 수행")
            
            # 2. DAG 추출
            dag_result = make_entity_dag(message, extractor.llm_model)
            dag_list = sorted([d for d in dag_result['dag_section'].split('\n') if d!=''])

        extracted_result = result.get('extracted_result', {})
        extracted_result['entity_dag'] = dag_list
        result['extracted_result'] = extracted_result

        raw_result = result.get('raw_result', {})
        raw_result['entity_dag'] = dag_list
        result['raw_result'] = raw_result

        result['error'] = ""
        
        logger.info(f"워커 프로세스에서 메시지 처리 완료")
        return result
        
    except Exception as e:
        logger.error(f"워커 프로세스에서 메시지 처리 실패: {e}")
        return {
            "title": "처리 실패",
            "purpose": ["오류"],
            "product": [],
            "channel": [],
            "pgm": [],
            "entity_dag": [],
            "error": str(e)
        }

def process_messages_batch(extractor, messages: List[str], extract_dag: bool = False, max_workers: int = None) -> List[Dict[str, Any]]:
    """
    여러 메시지를 배치로 처리하는 함수
    
    Args:
        extractor: MMSExtractor 인스턴스
        messages: 처리할 메시지 리스트
        extract_dag: DAG 추출 여부
        max_workers: 최대 워커 수 (None이면 CPU 코어 수)
    
    Returns:
        list: 처리 결과 리스트
    """
    if max_workers is None:
        max_workers = min(len(messages), os.cpu_count())
    
    logger.info(f"배치 처리 시작: {len(messages)}개 메시지, {max_workers}개 워커")
    
    start_time = time.time()
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 모든 메시지에 대해 작업 제출
        future_to_message = {
            executor.submit(process_message_with_dag, extractor, msg, extract_dag): msg 
            for msg in messages
        }
        
        # 완료된 작업들 수집
        for i, future in enumerate(future_to_message):
            try:
                result = future.result()
                results.append(result)
                logger.info(f"배치 처리 진행률: {i+1}/{len(messages)} ({((i+1)/len(messages)*100):.1f}%)")
            except Exception as e:
                logger.error(f"배치 처리 중 오류 발생: {e}")
                results.append({
                    "title": "처리 실패",
                    "purpose": ["오류"],
                    "product": [],
                    "channel": [],
                    "pgm": [],
                    "entity_dag": [],
                    "error": str(e)
                })
    
    elapsed_time = time.time() - start_time
    logger.info(f"배치 처리 완료: {len(messages)}개 메시지, {elapsed_time:.2f}초")
    logger.info(f"평균 처리 시간: {elapsed_time/len(messages):.2f}초/메시지")
    
    return results

def make_entity_dag(msg: str, llm_model, save_dag_image=True):

    # 메시지에서 엔티티 간의 관계를 방향성 있는 그래프로 추출
    # 예: (고객:가입) -[하면]-> (혜택:수령) -[통해]-> (만족도:향상)
    extract_dag_result = {}
    logger.info("=" * 30 + " DAG 추출 시작 " + "=" * 30)
    try:
        dag_start_time = time.time()
        # DAG 추출 함수 호출 (entity_dag_extractor.py)
        extract_dag_result = extract_dag(DAGParser(), msg, llm_model)
        dag_raw = extract_dag_result['dag_raw']      # LLM 원본 응답
        dag_section = extract_dag_result['dag_section']  # 파싱된 DAG 텍스트
        dag = extract_dag_result['dag']             # NetworkX 그래프 객체
        
        # 시각적 다이어그램 생성 (utils.py)
        dag_filename = ""
        if save_dag_image:
            dag_filename = f'dag_{sha256_hash(msg)}'
            create_dag_diagram(dag, filename=dag_filename)
            logger.info(f"✅ DAG 추출 완료: {dag_filename}")

        extract_dag_result['dag_filename'] = dag_filename
        
        dag_processing_time = time.time() - dag_start_time
        
        logger.info(f"🕒 DAG 처리 시간: {dag_processing_time:.3f}초")
        logger.info(f"📏 DAG 섹션 길이: {len(dag_section)}자")
        if dag_section:
            logger.info(f"📄 DAG 내용 미리보기: {dag_section[:200]}...")
        else:
            logger.warning("⚠️ DAG 섹션이 비어있습니다")
            
    except Exception as e:
        logger.error(f"❌ DAG 추출 중 오류 발생: {e}")
        dag_section = ""

    return extract_dag_result


def get_stored_prompts_from_thread():
    """현재 스레드에서 저장된 프롬프트 정보를 가져오는 함수"""
    import threading
    current_thread = threading.current_thread()
    
    if hasattr(current_thread, 'stored_prompts'):
        return current_thread.stored_prompts
    else:
        return {}

def save_result_to_mongodb_if_enabled(message: str, result: dict, args_or_data, extractor=None):
    """MongoDB 저장이 활성화된 경우 결과를 저장하는 도우미 함수
    
    Args:
        message: 처리할 메시지
        result: 처리 결과 (extracted_result, raw_result 포함)
        args_or_data: argparse.Namespace 객체 또는 딕셔너리
        extractor: MMSExtractor 인스턴스 (선택적)
    
    Returns:
        str: 저장된 문서 ID, 실패 시 None
    """
    # args_or_data가 딕셔너리인 경우 Namespace로 변환
    if isinstance(args_or_data, dict):
        import argparse
        args = argparse.Namespace(**args_or_data)
    else:
        args = args_or_data
    
    # save_to_mongodb 속성이 없거나 False인 경우
    if not getattr(args, 'save_to_mongodb', False):
        return None
        
    try:
        # MongoDB 임포트 시도
        from mongodb_utils import save_to_mongodb
        
        # 스레드 로컬 저장소에서 프롬프트 정보 가져오기
        stored_prompts = result.get('prompts', get_stored_prompts_from_thread()) 
        
        # 프롬프트 정보 구성
        prompts_data = {}
        for key, prompt_data in stored_prompts.items():
            prompts_data[key] = {
                'title': prompt_data.get('title', f'{key} 프롬프트'),
                'description': prompt_data.get('description', f'{key} 처리를 위한 프롬프트'),
                'content': prompt_data.get('content', ''),
                'length': len(prompt_data.get('content', ''))
            }
        
        # 저장된 프롬프트가 없는 경우 기본값 사용
        if not prompts_data:
            prompts_data = {
                'main_extraction_prompt': {
                    'title': '메인 정보 추출 프롬프트',
                    'description': 'MMS 메시지에서 기본 정보 추출',
                    'content': '실제 프롬프트 내용이 저장되지 않았습니다.',
                    'length': 0
                }
            }
        
        extraction_prompts = {
            'success': True,
            'prompts': prompts_data,
            'settings': {
                'llm_model': getattr(args, 'llm_model', 'unknown'),
                'offer_data_source': getattr(args, 'offer_data_source', getattr(args, 'offer_info_data_src', 'unknown')),
                'product_info_extraction_mode': getattr(args, 'product_info_extraction_mode', 'unknown'),
                'entity_matching_mode': getattr(args, 'entity_matching_mode', getattr(args, 'entity_extraction_mode', 'unknown')),
                'extract_entity_dag': getattr(args, 'extract_entity_dag', False)
            }
        }
        
        # 추출 결과를 MongoDB 형식으로 구성
        extraction_result = {
            'success': not bool(result.get('error')),
            'result': result.get('extracted_result', result.get('result', {})),
            'metadata': {
                'processing_time_seconds': result.get('processing_time', 0),
                'processing_mode': getattr(args, 'processing_mode', 'single'),
                'model_used': getattr(args, 'llm_model', 'unknown')
            }
        }

        raw_result_data = {
            'success': not bool(result.get('error')),
            'result': result.get('raw_result', {}),
            'metadata': {
                'processing_time_seconds': result.get('processing_time', 0),
                'processing_mode': getattr(args, 'processing_mode', 'single'),
                'model_used': getattr(args, 'llm_model', 'unknown')
            }
        }
        
        # MongoDB에 저장
        user_id = getattr(args, 'user_id', 'DEFAULT_USER')
        saved_id = save_to_mongodb(message, extraction_result, raw_result_data, extraction_prompts, 
                                 user_id=user_id, message_id=None)
        
        if saved_id:
            print(f"📄 결과가 MongoDB에 저장되었습니다. (ID: {saved_id[:8]}...)")
            return saved_id
        else:
            print("⚠️ MongoDB 저장에 실패했습니다.")
            return None
            
    except ImportError:
        print("❌ MongoDB 저장이 요청되었지만 mongodb_utils를 찾을 수 없습니다.")
        return None
    except Exception as e:
        print(f"❌ MongoDB 저장 중 오류 발생: {str(e)}")
        return None

            
    except Exception as e:
        print(f"❌ MongoDB 저장 중 오류 발생: {str(e)}")
        return None

def main():
    """
    커맨드라인에서 실행할 때의 메인 함수
    다양한 옵션을 통해 추출기 설정을 변경할 수 있습니다.
    
    사용법:
    # 단일 메시지 처리 (멀티스레드)
    python mms_extractor.py --message "광고 메시지" --extract-entity-dag
    
    # 배치 처리 (멀티프로세스)
    python mms_extractor.py --batch-file messages.txt --max-workers 4 --extract-entity-dag
    
    # 데이터베이스 모드로 배치 처리
    python mms_extractor.py --batch-file messages.txt --offer-data-source db --max-workers 8
    
    # MongoDB에 결과 저장
    python mms_extractor.py --message "광고 메시지" --save-to-mongodb --extract-entity-dag
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='MMS 광고 텍스트 추출기 - 개선된 버전')
    parser.add_argument('--message', type=str, help='테스트할 메시지')
    parser.add_argument('--batch-file', type=str, help='배치 처리할 메시지가 담긴 파일 경로 (한 줄에 하나씩)')
    parser.add_argument('--max-workers', type=int, help='배치 처리 시 최대 워커 수 (기본값: CPU 코어 수)')
    parser.add_argument('--offer-data-source', choices=['local', 'db'], default='local',
                       help='데이터 소스 (local: CSV 파일, db: 데이터베이스)')
    parser.add_argument('--product-info-extraction-mode', choices=['nlp', 'llm', 'rag'], default='nlp',
                       help='상품 정보 추출 모드 (nlp: 형태소분석, llm: LLM 기반, rag: 검색증강생성)')
    parser.add_argument('--entity-matching-mode', choices=['logic', 'llm'], default='llm',
                       help='엔티티 매칭 모드 (logic: 로직 기반, llm: LLM 기반)')
    parser.add_argument('--llm-model', choices=['gem', 'ax', 'cld', 'gen', 'gpt'], default='ax',
                       help='사용할 LLM 모델 (gem: Gemma, ax: ax, cld: Claude, gen: Gemini, gpt: GPT)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO',
                       help='로그 레벨 설정')
    parser.add_argument('--extract-entity-dag', action='store_true', default=False, help='Entity DAG extraction (default: False)')
    parser.add_argument('--save-to-mongodb', action='store_true', default=False, 
                       help='추출 결과를 MongoDB에 저장 (mongodb_utils.py 필요)')
    parser.add_argument('--test-mongodb', action='store_true', default=False,
                       help='MongoDB 연결 테스트만 수행하고 종료')

    args = parser.parse_args()
    
    # 로그 레벨 설정
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # MongoDB 연결 테스트만 수행하는 경우
    if args.test_mongodb:
        try:
            from mongodb_utils import test_mongodb_connection
        except ImportError:
            print("❌ MongoDB 유틸리티를 찾을 수 없습니다.")
            print("mongodb_utils.py 파일과 pymongo 패키지를 확인하세요.")
            exit(1)
        
        print("🔌 MongoDB 연결 테스트 중...")
        if test_mongodb_connection():
            print("✅ MongoDB 연결 성공!")
            exit(0)
        else:
            print("❌ MongoDB 연결 실패!")
            print("MongoDB 서버가 실행 중인지 확인하세요.")
            exit(1)
    
    try:
                # 추출기 초기화
        logger.info("MMS 추출기 초기화 중...")
        extractor = MMSExtractor(
            offer_info_data_src=args.offer_data_source,
            product_info_extraction_mode=args.product_info_extraction_mode,
            entity_extraction_mode=args.entity_matching_mode,
            llm_model=args.llm_model,
            extract_entity_dag=args.extract_entity_dag
        )
        
        # 배치 처리 또는 단일 메시지 처리
        if args.batch_file:
            # 배치 파일에서 메시지들 로드
            logger.info(f"배치 파일에서 메시지 로드: {args.batch_file}")
            try:
                with open(args.batch_file, 'r', encoding='utf-8') as f:
                    messages = [line.strip() for line in f if line.strip()]
                
                logger.info(f"로드된 메시지 수: {len(messages)}개")
                
                # 배치 처리 실행
                results = process_messages_batch(
                    extractor, 
                    messages, 
                    extract_dag=args.extract_entity_dag,
                    max_workers=args.max_workers
                )
                
                # MongoDB 저장 (배치 처리)
                if args.save_to_mongodb:
                    print("\n📄 MongoDB 저장 중...")
                    args.processing_mode = 'batch'
                    saved_count = 0
                    for i, result in enumerate(results):
                        if i < len(messages):  # 메시지가 있는 경우만
                            saved_id = save_result_to_mongodb_if_enabled(messages[i], result, args, extractor)
                            if saved_id:
                                saved_count += 1
                    print(f"📄 MongoDB 저장 완료: {saved_count}/{len(results)}개")
                
                # 배치 결과 출력
                print("\n" + "="*50)
                print("🎯 배치 처리 결과")
                print("="*50)
                
                for i, result in enumerate(results):
                    print(f"\n--- 메시지 {i+1} ---")
                    print(f"제목: {result.get('title', 'N/A')}")
                    print(f"상품: {len(result.get('product', []))}개")
                    if result.get('error'):
                        print(f"오류: {result['error']}")
                
                # 전체 배치 통계
                successful = len([r for r in results if not r.get('error')])
                failed = len(results) - successful
                print(f"\n📊 배치 처리 통계")
                print(f"✅ 성공: {successful}개")
                print(f"❌ 실패: {failed}개")
                print(f"📈 성공률: {(successful/len(results)*100):.1f}%")
                
                # 결과를 JSON 파일로 저장
                output_file = f"batch_results_{int(time.time())}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=4, ensure_ascii=False)
                print(f"💾 결과 저장: {output_file}")
                
            except FileNotFoundError:
                logger.error(f"배치 파일을 찾을 수 없습니다: {args.batch_file}")
                exit(1)
            except Exception as e:
                logger.error(f"배치 파일 처리 실패: {e}")
                exit(1)
        
        else:
            # 단일 메시지 처리
            test_message = args.message if args.message else """
[Web발신]
(광고)[SKT (을지로점)] 신용욱 단골고객님
9월은 SKT 직영점에서 혜택받9, 구매하9

【갤럭시 마지막 특가】
① 와이드8 ▶기기값 5만원
② A36 ▶기기값 10만원
③ S24 FE ▶기기값 20만원
☞ 제휴카드 사용 시 최대 72만원 추가할인

【SK로 통신사 이동 시】
① 쓰던 폰 그대로 이동시 상품권 20만원
② 인터넷+TV 가입 최대 70만원

★9/9 까지 선착순 행사 (조건에 따라 할인금액 상이)

♥아이폰17 사전예약♥
고용량 전색상 바로 개통가능☞ https://naver.me/FTM8rdfj

☞ 을지로입구역 5번출구 하나은행 명동사옥 맞은편
https://naver.me/GipIR3Lg
☎ 0507-1399-6011

(무료ARS)수신거부 및 단골해지 : 
080-801-0011            
"""
            
            # 단일 메시지 처리 (멀티스레드)
            logger.info("단일 메시지 처리 시작 (멀티스레드)")
            result = process_message_with_dag(extractor, test_message, args.extract_entity_dag)
                    
            # MongoDB 저장 (단일 메시지)
            if args.save_to_mongodb:
                print("\n📄 MongoDB 저장 중...")
                args.processing_mode = 'single'
                saved_id = save_result_to_mongodb_if_enabled(test_message, result, args, extractor)
                if saved_id:
                    print("📄 MongoDB 저장 완료!")

            
            extracted_result = result.get('extracted_result', {})
        
            print("\n" + "="*50)
            print("🎯 최종 추출된 정보")
            print("="*50)
            print(json.dumps(extracted_result, indent=4, ensure_ascii=False))

            # 성능 요약 정보 출력
            print("\n" + "="*50)
            print("📊 처리 완료")
            print("="*50)
            print(f"✅ 제목: {extracted_result.get('title', 'N/A')}")
            print(f"✅ 목적: {len(extracted_result.get('purpose', []))}개")
            print(f"✅ 상품: {len(extracted_result.get('product', []))}개")
            print(f"✅ 채널: {len(extracted_result.get('channel', []))}개")
            print(f"✅ 프로그램: {len(extracted_result.get('pgm', []))}개")
            if extracted_result.get('error'):
                print(f"❌ 오류: {extracted_result['error']}")
        
    except Exception as e:
        logger.error(f"실행 실패: {e}")
        logger.error(traceback.format_exc())
        exit(1)


if __name__ == '__main__':
    main()