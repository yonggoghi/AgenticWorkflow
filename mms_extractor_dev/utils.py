"""
MMS Extractor 유틸리티 함수 모듈
================================

이 모듈은 MMS 추출기에서 사용되는 다양한 유틸리티 함수들을 포함합니다:
- 데코레이터 및 안전 실행 함수들
- 텍스트 처리 및 JSON 복구 함수들
- 유사도 계산 함수들
- 형태소 분석 관련 클래스들

작성자: MMS 분석팀
버전: 2.0.0
"""

import time
import logging
import re
import ast
import json
import os
import hashlib
from functools import wraps
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import difflib
from rapidfuzz import fuzz
from joblib import Parallel, delayed
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rc

# 로깅 설정
logger = logging.getLogger(__name__)

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

def escape_quotes_in_value(value):
    """값 내부의 따옴표를 이스케이프하거나 제거"""
    value = value.strip()
    if not value:
        return value
    
    # 값이 따옴표로 시작하는 경우
    if value[0] in ['"', "'"]:
        quote_char = value[0]
        # 닫는 따옴표 찾기
        end_idx = -1
        for i in range(1, len(value)):
            if value[i] == quote_char:
                end_idx = i
                break
        
        if end_idx > 0:
            # 따옴표로 감싸진 부분과 나머지 분리
            quoted_part = value[1:end_idx]
            remaining = value[end_idx+1:].strip()
            
            # 나머지가 있으면 합치기 (따옴표 내부의 내용으로 통합)
            if remaining:
                combined = quoted_part + ' ' + remaining
                return quote_char + combined + quote_char
            else:
                return quote_char + quoted_part + quote_char
        else:
            # 닫는 따옴표가 없으면 전체를 따옴표로 감싸기
            return quote_char + value[1:] + quote_char
    
    # 따옴표로 시작하지 않으면 전체를 따옴표로 감싸기
    return '"' + value + '"'

def split_key_value(text):
    """따옴표 외부의 첫 번째 콜론을 기준으로 키-값 분리 (개선된 버전)"""
    in_quote = False
    quote_char = ''
    escape_next = False
    
    for i, char in enumerate(text):
        if escape_next:
            escape_next = False
            continue
            
        if char == '\\':
            escape_next = True
            continue
            
        if char in ['"', "'"]:
            if in_quote:
                if char == quote_char:
                    in_quote = False
                    quote_char = ''
            else:
                in_quote = True
                quote_char = char
        elif char == ':' and not in_quote:
            return text[:i].strip(), text[i+1:].strip()
    
    return text.strip(), ''

def split_outside_quotes(text, delimiter=','):
    """따옴표 외부의 구분자로만 텍스트 분리 (개선된 버전)"""
    parts = []
    current = []
    in_quote = False
    quote_char = ''
    escape_next = False
    
    for char in text:
        if escape_next:
            current.append(char)
            escape_next = False
            continue
            
        if char == '\\':
            current.append(char)
            escape_next = True
            continue
            
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
            if current:
                parts.append(''.join(current).strip())
                current = []
        else:
            current.append(char)
    
    if current:
        parts.append(''.join(current).strip())
    
    return parts

def clean_ill_structured_json(text):
    """잘못 구조화된 JSON 형식의 텍스트 정리 (개선된 버전)"""
    # 중괄호 제거
    text = text.strip()
    if text.startswith('{'):
        text = text[1:]
    if text.endswith('}'):
        text = text[:-1]
    
    parts = split_outside_quotes(text.strip(), delimiter=',')
    cleaned_parts = []
    
    for part in parts:
        if not part.strip():
            continue
            
        key, value = split_key_value(part)
        
        if not value:
            continue
        
        # 키 정리 (따옴표 추가)
        key_clean = key.strip()
        if not (key_clean.startswith('"') and key_clean.endswith('"')):
            if key_clean.startswith('"') or key_clean.startswith("'"):
                key_clean = key_clean[1:]
            if key_clean.endswith('"') or key_clean.endswith("'"):
                key_clean = key_clean[:-1]
            key_clean = '"' + key_clean + '"'
        
        # 값 정리 (따옴표 처리)
        value_clean = escape_quotes_in_value(value)
        
        cleaned_parts.append(f"{key_clean}: {value_clean}")
    
    return '{' + ', '.join(cleaned_parts) + '}'

def repair_json(broken_json):
    """손상된 JSON 문자열 복구 (개선된 버전)"""
    json_str = broken_json.strip()
    
    # 후행 쉼표 제거
    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
    
    # 연속된 쉼표 제거
    json_str = re.sub(r',\s*,', ',', json_str)
    
    return json_str

def extract_json_objects(text):
    """텍스트에서 JSON 객체 추출 (개선된 버전)"""
    pattern = r'(\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\})'
    result = []
    
    for match in re.finditer(pattern, text):
        potential_json = match.group(0)
        
        # 여러 방법으로 시도
        attempts = [
            lambda: json.loads(potential_json),
            lambda: json.loads(repair_json(potential_json)),
            lambda: json.loads(clean_ill_structured_json(repair_json(potential_json))),
            lambda: ast.literal_eval(clean_ill_structured_json(repair_json(potential_json))),
        ]
        
        for attempt in attempts:
            try:
                json_obj = attempt()
                if isinstance(json_obj, dict):
                    result.append(json_obj)
                    break
            except (json.JSONDecodeError, SyntaxError, ValueError, TypeError):
                continue
    
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
    
    text_processed = preprocess_text(text)
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
 
def combined_sequence_similarity(s1, s2, weights=None, normalizaton_value='max'):
    """여러 유사도 메트릭을 결합한 종합 유사도 계산"""

    s1 = replace_special_chars_with_space(s1)
    s2 = replace_special_chars_with_space(s2)
    
    if weights is None:
        weights = {'substring': 0.1, 'sequence_matcher': 0.7, 'token_sequence': 0.2}
    
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
    weights = args_dict.get('weights', None)
    results = []
    for sent_item in sent_item_batch:
        sent = sent_item[text_col_nm]
        item = sent_item[item_col_nm]
        try:
            sent_processed = preprocess_text(sent.lower())
            item_processed = preprocess_text(item.lower())
            similarity = combined_sequence_similarity(sent_processed, item_processed, weights=weights, normalizaton_value=normalizaton_value)[0]
            results.append({text_col_nm:sent, item_col_nm:item, "sim":similarity})
        except Exception as e:
            logger.error(f"Error processing {item}: {e}")
            results.append({text_col_nm:sent, item_col_nm:item, "sim":0.0})
    
    return results

def parallel_seq_similarity(sent_item_pdf, text_col_nm='sent', item_col_nm='item_nm_alias', n_jobs=None, batch_size=None, weights=None, normalizaton_value='s2'):
    """병렬 처리를 통한 시퀀스 유사도 계산"""
    if n_jobs is None:
        n_jobs = min(os.cpu_count()-1, 8)
    if batch_size is None:
        batch_size = max(1, sent_item_pdf.shape[0] // (n_jobs * 2))
    
    batches = []
    for i in range(0, sent_item_pdf.shape[0], batch_size):
        batch = sent_item_pdf.iloc[i:i + batch_size].to_dict(orient='records')
        batches.append({"sent_item_batch": batch, 'text_col_nm': text_col_nm, 'item_col_nm': item_col_nm, 'weights': weights, 'normalizaton_value': normalizaton_value})
    
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

# ===== DAG 관련 유틸리티 함수들 =====

def sha256_hash(text: str) -> str:
    """텍스트의 SHA256 해시값을 반환"""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:8]

def create_dag_diagram(dag: nx.DiGraph, filename: str = "dag", save_dir: str = "dag_images"):
    """
    DAG를 시각화하여 이미지 파일로 저장
    
    Args:
        dag: NetworkX DiGraph 객체
        filename: 저장할 파일명 (확장자 제외)
        save_dir: 저장할 디렉토리
    """
    try:
        # 저장 디렉토리 생성
        os.makedirs(save_dir, exist_ok=True)
        
        # 한글 폰트 설정
        try:
            # macOS에서 사용 가능한 한글 폰트 찾기
            font_list = [f.name for f in fm.fontManager.ttflist]
            korean_fonts = [f for f in font_list if any(k in f.lower() for k in ['apple', 'malgun', 'nanum', 'dotum', 'gulim'])]
            
            if korean_fonts:
                plt.rcParams['font.family'] = korean_fonts[0]
            else:
                plt.rcParams['font.family'] = 'DejaVu Sans'
        except:
            plt.rcParams['font.family'] = 'DejaVu Sans'
        
        # 그래프 크기 설정
        plt.figure(figsize=(12, 8))
        
        # 레이아웃 계산
        pos = nx.spring_layout(dag, k=3, iterations=50)
        
        # 노드 그리기
        nx.draw_networkx_nodes(dag, pos, 
                             node_color='lightblue', 
                             node_size=1000,
                             alpha=0.7)
        
        # 엣지 그리기
        nx.draw_networkx_edges(dag, pos, 
                              edge_color='gray',
                              arrows=True,
                              arrowsize=20,
                              alpha=0.6)
        
        # 라벨 그리기
        labels = {node: node for node in dag.nodes()}
        nx.draw_networkx_labels(dag, pos, labels, font_size=8)
        
        # 제목 설정
        plt.title(f"DAG Diagram: {filename}", fontsize=14, pad=20)
        
        # 축 제거
        plt.axis('off')
        
        # 레이아웃 조정
        plt.tight_layout()
        
        # 파일 저장
        filepath = os.path.join(save_dir, f"{filename}.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"DAG 다이어그램이 저장되었습니다: {filepath}")
        
    except Exception as e:
        logger.error(f"DAG 다이어그램 생성 실패: {e}")
        raise

def select_most_comprehensive(strings):
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
        contains_others = any(other in current for other in unique_strings if other != current and other not in result)
        
        # If current is not contained by existing results and either:
        # 1. It contains other strings, or 
        # 2. No strings contain each other (keep all unique)
        if not is_contained:
            # Remove any strings from result that are contained in current
            result = [r for r in result if r not in current]
            result.append(current)
    
    return result

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

