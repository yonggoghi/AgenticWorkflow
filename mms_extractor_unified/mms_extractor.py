
# %%
"""
MMS 추출기 (MMS Extractor)
=========================

이 모듈은 MMS(멀티미디어 메시지) 광고 텍스트에서 상품명, 채널 정보, 광고 목적 등을 
자동으로 추출하는 시스템입니다.

주요 기능:
- 형태소 분석을 통한 개체명 추출
- 퍼지 매칭 및 시퀀스 유사도를 이용한 상품명 매칭
- LLM을 활용한 광고 정보 구조화 추출
- 대리점/매장 정보 매칭
"""

from concurrent.futures import ThreadPoolExecutor
import time
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import json
import re
# from pygments import highlight
# from pygments.lexers import JsonLexer
# from pygments.formatters import HtmlFormatter
# from IPython.display import HTML
import pandas as pd
# from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from openai import OpenAI
from typing import List, Tuple, Union, Dict, Any
import ast
from rapidfuzz import fuzz, process
import re
import json
import glob
import os
from config.settings import API_CONFIG, MODEL_CONFIG, PROCESSING_CONFIG, METADATA_CONFIG, EMBEDDING_CONFIG
from kiwipiepy import Kiwi
from joblib import Parallel, delayed
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from difflib import SequenceMatcher
import difflib
from dotenv import load_dotenv
import cx_Oracle

# pandas 출력 설정
pd.set_option('display.max_colwidth', 500)
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# ===== 유틸리티 함수들 =====

def dataframe_to_markdown_prompt(df, max_rows=None):
    """
    DataFrame을 마크다운 형식의 프롬프트로 변환
    
    Args:
        df: 변환할 DataFrame
        max_rows: 최대 행 수 제한 (None이면 모든 행 포함)
    
    Returns:
        str: 마크다운 형식의 테이블 문자열
    """
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
    """
    따옴표로 둘러싸인 문자열에서 내부의 동일한 따옴표 제거
    
    Args:
        segment: 정리할 문자열 세그먼트
    
    Returns:
        str: 정리된 문자열
    """
    segment = segment.strip()
    if len(segment) >= 2 and segment[0] in ['"', "'"] and segment[-1] == segment[0]:
        q = segment[0]
        inner = segment[1:-1].replace(q, '')
        return q + inner + q
    return segment

def split_key_value(text):
    """
    따옴표 외부의 첫 번째 콜론을 기준으로 키-값 분리
    
    Args:
        text: 분리할 텍스트
    
    Returns:
        tuple: (키, 값) 튜플
    """
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
    """
    따옴표 외부의 구분자로만 텍스트 분리
    
    Args:
        text: 분리할 텍스트
        delimiter: 구분자 (기본값: 쉼표)
    
    Returns:
        list: 분리된 문자열 리스트
    """
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
    """
    잘못 구조화된 JSON 형식의 텍스트 정리
    
    Args:
        text: 정리할 JSON 형식 텍스트
    
    Returns:
        str: 정리된 텍스트
    """
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
    """
    손상된 JSON 문자열 복구
    
    Args:
        broken_json: 복구할 JSON 문자열
    
    Returns:
        str: 복구된 JSON 문자열
    """
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
    """
    텍스트에서 JSON 객체 추출
    
    Args:
        text: JSON이 포함된 텍스트
    
    Returns:
        list: 추출된 JSON 객체 리스트
    """
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
    """
    텍스트 전처리 (특수문자 제거, 공백 정규화)
    
    Args:
        text: 전처리할 텍스트
    
    Returns:
        str: 전처리된 텍스트
    """
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ===== 유사도 계산 함수들 =====

def fuzzy_similarities(text, entities):
    """
    퍼지 매칭을 사용한 유사도 계산
    
    Args:
        text: 비교할 텍스트
        entities: 비교 대상 엔티티 리스트
    
    Returns:
        list: (엔티티, 최대유사도점수) 튜플 리스트
    """
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
    """
    배치 처리를 위한 퍼지 유사도 계산 함수
    
    Args:
        args_dict: 처리 인자 딕셔너리
            - text: 비교할 텍스트
            - entities: 엔티티 리스트
            - threshold: 임계값
            - text_col_nm: 텍스트 컬럼명
            - item_col_nm: 아이템 컬럼명
    
    Returns:
        list: 필터링된 결과 리스트
    """
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
    """
    병렬 처리를 통한 퍼지 유사도 계산
    
    Args:
        texts: 비교할 텍스트 리스트
        entities: 엔티티 리스트
        threshold: 유사도 임계값
        text_col_nm: 텍스트 컬럼명
        item_col_nm: 아이템 컬럼명
        n_jobs: 병렬 작업 수
        batch_size: 배치 크기
    
    Returns:
        DataFrame: 유사도 결과 데이터프레임
    """
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
    """
    최장 공통 부분수열 비율 계산
    
    Args:
        s1, s2: 비교할 두 문자열
        normalizaton_value: 정규화 방식 ('max', 'min', 's1', 's2')
    
    Returns:
        float: LCS 비율 (0.0-1.0)
    """
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
    """
    SequenceMatcher를 사용한 유사도 계산
    
    Args:
        s1, s2: 비교할 두 문자열
        normalizaton_value: 정규화 방식
    
    Returns:
        float: 유사도 점수 (0.0-1.0)
    """
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
    """
    부분문자열 관계를 고려한 유사도 계산
    
    Args:
        s1, s2: 비교할 두 문자열
        normalizaton_value: 정규화 방식
    
    Returns:
        float: 유사도 점수 (0.0-1.0)
    """
    # 한 문자열이 다른 문자열의 부분문자열인지 확인
    if s1 in s2 or s2 in s1:
        shorter = min(s1, s2, key=len)
        longer = max(s1, s2, key=len)
        base_score = len(shorter) / len(longer)
        return min(0.95 + base_score * 0.05, 1.0)
    return longest_common_subsequence_ratio(s1, s2, normalizaton_value)

def token_sequence_similarity(s1, s2, normalizaton_value, separator_pattern=r'[\s_\-]+'):
    """
    토큰 시퀀스 기반 유사도 계산
    
    Args:
        s1, s2: 비교할 두 문자열
        normalizaton_value: 정규화 방식
        separator_pattern: 토큰 분리 패턴
    
    Returns:
        float: 토큰 시퀀스 유사도 (0.0-1.0)
    """
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
    """
    여러 유사도 메트릭을 결합한 종합 유사도 계산
    
    Args:
        s1, s2: 비교할 두 문자열
        weights: 각 메트릭의 가중치 딕셔너리
        normalizaton_value: 정규화 방식
    
    Returns:
        tuple: (종합유사도, 개별유사도딕셔너리)
    """
    if weights is None:
        weights = {'substring': 0.4, 'sequence_matcher': 0.4, 'token_sequence': 0.2}
    
    similarities = {
        'substring': substring_aware_similarity(s1, s2, normalizaton_value),
        'sequence_matcher': sequence_matcher_similarity(s1, s2, normalizaton_value),
        'token_sequence': token_sequence_similarity(s1, s2, normalizaton_value)
    }
    
    return sum(similarities[key] * weights[key] for key in weights), similarities

def calculate_seq_similarity(args_dict):
    """
    배치 처리를 위한 시퀀스 유사도 계산
    
    Args:
        args_dict: 처리 인자 딕셔너리
    
    Returns:
        list: 유사도 결과 리스트
    """
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
            print(f"Error processing {item}: {e}")
            results.append({text_col_nm:sent, item_col_nm:item, "sim":0.0})
    
    return results

def parallel_seq_similarity(sent_item_pdf, text_col_nm='sent', item_col_nm='item_nm_alias', n_jobs=None, batch_size=None, normalizaton_value='s2'):
    """
    병렬 처리를 통한 시퀀스 유사도 계산
    
    Args:
        sent_item_pdf: 처리할 데이터프레임
        text_col_nm: 텍스트 컬럼명
        item_col_nm: 아이템 컬럼명
        n_jobs: 병렬 작업 수
        batch_size: 배치 크기
        normalizaton_value: 정규화 방식
    
    Returns:
        DataFrame: 유사도 결과 데이터프레임
    """
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
    """
    SentenceTransformer 모델 로드
    
    Args:
        model_path: 모델 경로
        device: 사용할 디바이스 (None이면 자동 선택)
    
    Returns:
        SentenceTransformer: 로드된 모델
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model from {model_path}...")
    model = SentenceTransformer(model_path).to(device)
    print(f"Model loaded on {device}")
    return model

# ===== Kiwi 형태소 분석 관련 클래스들 =====

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
    """
    제외할 품사 패턴에 따라 텍스트 필터링
    
    Args:
        sentence: 필터링할 문장 객체
        exc_tag_patterns: 제외할 품사 패턴 리스트
    
    Returns:
        str: 필터링된 텍스트
    """
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
    """
    중복되거나 포함 관계에 있는 용어들 필터링
    
    Args:
        strings: 필터링할 문자열 리스트
    
    Returns:
        list: 필터링된 문자열 리스트
    """
    unique_strings = list(set(strings))
    unique_strings.sort(key=len, reverse=True)
    
    filtered = []
    for s in unique_strings:
        if not any(s in other for other in filtered):
            filtered.append(s)
    
    return filtered

def convert_df_to_json_list(df):
    """
    DataFrame을 특정 JSON 구조로 변환
    
    Args:
        df: 변환할 DataFrame
    
    Returns:
        list: 변환된 JSON 구조 리스트
    """
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


class MMSExtractor:
    """
    MMS 광고 텍스트 추출기 메인 클래스
    
    이 클래스는 MMS 광고 메시지에서 상품명, 채널 정보, 광고 목적 등을 
    자동으로 추출하는 기능을 제공합니다.
    """
    
    def __init__(self, model_path=None, data_dir=None, product_info_extraction_mode=None, 
                 entity_extraction_mode=None, offer_info_data_src='local', llm_model='gemma'):
        """
        MMSExtractor 초기화
        
        Args:
            model_path: 임베딩 모델 경로 (None이면 설정에서 가져옴)
            data_dir: 데이터 디렉토리 경로 (None이면 기본값 사용)
            product_info_extraction_mode: 상품 정보 추출 모드 ('rag', 'llm', 'nlp')
            entity_extraction_mode: 엔티티 추출 모드 ('llm', 'logic')
            offer_info_data_src: 상품 정보 데이터 소스 ('local', 'db')
            llm_model: 사용할 LLM 모델 ('gemma', 'gpt', 'claude')
        """
        # 설정에서 기본값 가져오기
        self.data_dir = data_dir if data_dir is not None else './data/'
        self.model_path = model_path if model_path is not None else EMBEDDING_CONFIG.ko_sbert_model_path
        self.offer_info_data_src = offer_info_data_src  # 'local' 또는 'db'
        self.product_info_extraction_mode = product_info_extraction_mode if product_info_extraction_mode is not None else PROCESSING_CONFIG.product_info_extraction_mode
        self.entity_extraction_mode = entity_extraction_mode if entity_extraction_mode is not None else PROCESSING_CONFIG.entity_extraction_mode
        self.llm_model_name = llm_model
        self.num_cand_pgms = PROCESSING_CONFIG.num_candidate_programs
        
        # 환경 변수 로드
        load_dotenv()
        
        # 초기화 단계별 실행
        self._initialize_device()
        self._initialize_llm()
        self._initialize_embedding_model(self.model_path)
        self._initialize_kiwi()
        self._load_data()

    def _initialize_device(self):
        """
        사용할 디바이스 초기화 (MPS > CUDA > CPU 순서로 선택)
        """
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = "mps"  # Apple Silicon Mac
        elif torch.cuda.is_available():
            self.device = "cuda"  # NVIDIA GPU
        else:
            self.device = "cpu"   # CPU
        print(f"Using device: {self.device}")

    def _initialize_llm(self):
        """
        선택된 LLM 모델 초기화
        """
        if self.llm_model_name == "gemma":
            self.llm_model = ChatOpenAI(
                temperature=MODEL_CONFIG.temperature,
                openai_api_key=API_CONFIG.llm_api_key,
                openai_api_base=API_CONFIG.llm_api_url,
                model=MODEL_CONFIG.gemma_model,
                max_tokens=MODEL_CONFIG.llm_max_tokens
            )
        elif self.llm_model_name == "gpt":
            self.llm_model = ChatOpenAI(
                temperature=MODEL_CONFIG.temperature,
                openai_api_key=API_CONFIG.openai_api_key,
                model=MODEL_CONFIG.gpt_model,
                max_tokens=MODEL_CONFIG.llm_max_tokens
            )
        elif self.llm_model_name == "claude":
            self.llm_model = ChatAnthropic(
                temperature=MODEL_CONFIG.temperature,
                api_key=API_CONFIG.anthropic_api_key,
                model=MODEL_CONFIG.claude_model,
                max_tokens=MODEL_CONFIG.llm_max_tokens
            )

        print(f"Initialized LLM: {self.llm_model_name}")

    def _initialize_embedding_model(self, model_path):
        """
        임베딩 모델 초기화
        
        Args:
            model_path: 모델 경로
        """
        self.emb_model = load_sentence_transformer(model_path, self.device)

    def _initialize_kiwi(self):
        """
        Kiwi 형태소 분석기 초기화 및 제외 패턴 설정
        """
        self.kiwi = Kiwi()
        
        # 제외할 품사 태그 패턴들
        self.exc_tag_patterns = [
            ['SN', 'NNB'],    # 숫자 + 의존명사
            ['W_SERIAL'],     # 일련번호
            ['JKO'],          # 목적격 조사
            ['W_URL'],        # URL
            ['W_EMAIL'],      # 이메일
            ['XSV', 'EC'],    # 동사 파생 접미사 + 연결어미
            ['VV', 'EC'],     # 동사 + 연결어미
            ['VCP', 'ETM'],   # 긍정 지정사 + 관형형 전성어미
            ['XSA', 'ETM'],   # 형용사 파생 접미사 + 관형형 전성어미
            ['VV', 'ETN'],    # 동사 + 명사형 전성어미
            ['SSO'], ['SSC'], ['SW'], ['SF'], ['SP'], ['SS'], ['SE'], ['SO'], ['SB'], ['SH'],  # 각종 기호
            ['W_HASHTAG']     # 해시태그
        ]
        print("Initialized Kiwi morphological analyzer.")

    def _load_data(self):
        """
        필요한 데이터 파일들 로드 (상품 정보, 별칭 규칙, 정지어, 프로그램 분류, 조직 정보)
        """
        print("Loading data...")
        
        # 상품 정보 로드 (로컬 파일 또는 데이터베이스)
        if self.offer_info_data_src == "local":
            item_pdf_raw = pd.read_csv(METADATA_CONFIG.offer_data_path)
            self.item_pdf_all = item_pdf_raw.drop_duplicates(['item_nm','item_id'])[['item_nm','item_id','item_desc','domain']].copy()
            self.item_pdf_all['item_ctg'] = None
            self.item_pdf_all['item_emb_vec'] = None
            self.item_pdf_all['ofer_cd'] = self.item_pdf_all['item_id']
            self.item_pdf_all['oper_dt_hms'] = '20250101000000'
            self.item_pdf_all = self.item_pdf_all.rename(columns={c:c.lower() for c in self.item_pdf_all.columns}).query("domain!='R'")
            
        elif self.offer_info_data_src == "db":
            # 데이터베이스 연결 정보
            username = os.getenv("DB_USERNAME")
            password = os.getenv("DB_PASSWORD")
            host = os.getenv("DB_HOST")
            port = os.getenv("DB_PORT")
            service_name = os.getenv("DB_NAME")
            
            # Oracle 데이터베이스 연결
            dsn = cx_Oracle.makedsn(host, port, service_name=service_name)
            conn = cx_Oracle.connect(user=username, password=password, dsn=dsn, encoding="UTF-8")
            
            # 상품 정보 조회 (최대 100만 건)
            sql = "SELECT * FROM TCAM_RC_OFER_MST WHERE ROWNUM <= 1000000 and ITEM_DMN!='R'"
            self.item_pdf_all = pd.read_sql(sql, conn)
            conn.close()
            
            # 컬럼명 소문자 변환
            self.item_pdf_all = self.item_pdf_all.rename(columns={c:c.lower() for c in self.item_pdf_all.columns})

        # 별칭 규칙 로드 및 적용
        alias_pdf = pd.read_csv(METADATA_CONFIG.alias_rules_path)
        alia_rule_set = list(zip(alias_pdf['alias_1'], alias_pdf['alias_2']))

        def apply_alias_rule(item_nm):
            """상품명에 별칭 규칙 적용"""
            item_nm_list = [item_nm]
            for r in alia_rule_set:
                if r[0] in item_nm:
                    item_nm_list.append(item_nm.replace(r[0], r[1]))
                if r[1] in item_nm:
                    item_nm_list.append(item_nm.replace(r[1], r[0]))
            return item_nm_list

        self.item_pdf_all['item_nm_alias'] = self.item_pdf_all['item_nm'].apply(apply_alias_rule)
        self.item_pdf_all = self.item_pdf_all.explode('item_nm_alias')
        
        # 사용자 정의 엔티티 추가
        user_defined_entity = PROCESSING_CONFIG.user_defined_entities
        item_pdf_ext = pd.DataFrame([
            {'item_nm':e,'item_id':e,'item_desc':e, 'domain':'user_defined', 
             'start_dt':20250101, 'end_dt':99991231, 'rank':1, 'item_nm_alias':e} 
            for e in user_defined_entity
        ])
        # 주석: 컬럼 불일치 문제로 인해 현재는 추가하지 않음
        # self.item_pdf_all = pd.concat([self.item_pdf_all,item_pdf_ext])
        
        # 정지어 목록 로드
        self.stop_item_names = pd.read_csv(METADATA_CONFIG.stop_items_path)['stop_words'].to_list()

        # Kiwi에 상품명들을 고유명사로 등록
        for w in self.item_pdf_all['item_nm_alias'].unique():
            self.kiwi.add_user_word(w, "NNP")

        # 프로그램 분류 정보 로드 및 임베딩 생성
        self.pgm_pdf = pd.read_csv(METADATA_CONFIG.pgm_info_path)
        self.clue_embeddings = self.emb_model.encode(
            self.pgm_pdf[["pgm_nm","clue_tag"]].apply(
                lambda x: preprocess_text(x['pgm_nm'].lower())+" "+x['clue_tag'].lower(), axis=1
            ).tolist(),
            convert_to_tensor=True, show_progress_bar=True
        )

        # 조직/매장 정보 로드
        self.org_pdf = pd.read_csv(METADATA_CONFIG.org_info_path, encoding='cp949')
        self.org_pdf['sub_org_cd'] = self.org_pdf['sub_org_cd'].apply(lambda x: str(x).zfill(4))
        print("Data loading complete.")

    def extract_entities_from_kiwi(self, mms_msg):
        """
        Kiwi 형태소 분석기를 사용한 엔티티 추출
        
        Args:
            mms_msg: 분석할 MMS 메시지 텍스트
        
        Returns:
            tuple: (후보_아이템_리스트, 추가_아이템_데이터프레임)
        """
        # 문장 분할 및 하위 문장 처리
        sentences = sum(self.kiwi.split_into_sents(re.split(r"_+", mms_msg), return_tokens=True, return_sub_sents=True), [])
        sentences_all = []
        
        for sent in sentences:
            if sent.subs:
                sentences_all.extend(sent.subs)
            else:
                sentences_all.append(sent)
        
        # 제외 패턴을 적용하여 문장 필터링
        sentence_list = [filter_text_by_exc_patterns(sent, self.exc_tag_patterns) for sent in sentences_all]

        # 형태소 분석을 통한 고유명사 추출
        result_msg = self.kiwi.tokenize(mms_msg, normalize_coda=True, z_coda=False, split_complex=False)
        entities_from_kiwi = [
            token.form for token in result_msg 
            if token.tag == 'NNP' and 
               token.form not in self.stop_item_names+['-'] and 
               len(token.form)>=2 and 
               not token.form.lower() in self.stop_item_names
        ]
        entities_from_kiwi = filter_specific_terms(entities_from_kiwi)
        print("추출된 개체명 (Kiwi):", list(set(entities_from_kiwi)))

        # 퍼지 매칭을 통한 유사 상품명 찾기
        similarities_fuzzy = parallel_fuzzy_similarity(
            sentence_list, self.item_pdf_all['item_nm_alias'].unique(), 
            threshold=PROCESSING_CONFIG.fuzzy_threshold,
            text_col_nm='sent', item_col_nm='item_nm_alias', 
            n_jobs=PROCESSING_CONFIG.n_jobs, batch_size=30
        )
        
        if similarities_fuzzy.empty:
            # 퍼지 매칭 결과가 없으면 Kiwi 결과만 사용
            cand_item_list = entities_from_kiwi
            extra_item_pdf = self.item_pdf_all.query("item_nm_alias in @cand_item_list")[['item_nm','item_nm_alias','item_id']].groupby(["item_nm"])['item_id'].apply(list).reset_index()
            return cand_item_list, extra_item_pdf

        # 시퀀스 유사도를 통한 정밀 매칭
        similarities_seq = parallel_seq_similarity(
            sent_item_pdf=similarities_fuzzy, text_col_nm='sent', item_col_nm='item_nm_alias',
            n_jobs=PROCESSING_CONFIG.n_jobs, batch_size=PROCESSING_CONFIG.batch_size
        )
        
        # 임계값 이상의 후보 아이템들 필터링
        cand_items = similarities_seq.query("sim>=@PROCESSING_CONFIG.similarity_threshold and item_nm_alias.str.contains('', case=False) and item_nm_alias not in @self.stop_item_names")
        
        # Kiwi에서 추출한 엔티티들 추가 (높은 신뢰도로 설정)
        entities_from_kiwi_pdf = self.item_pdf_all.query("item_nm_alias in @entities_from_kiwi")[['item_nm','item_nm_alias']]
        entities_from_kiwi_pdf['sim'] = 1.0

        # 결과 통합 및 최종 후보 리스트 생성
        cand_item_pdf = pd.concat([cand_items, entities_from_kiwi_pdf])
        cand_item_list = cand_item_pdf.sort_values('sim', ascending=False).groupby(["item_nm_alias"])['sim'].max().reset_index(name='final_sim').sort_values('final_sim', ascending=False).query("final_sim>=0.2")['item_nm_alias'].unique()
        extra_item_pdf = self.item_pdf_all.query("item_nm_alias in @cand_item_list")[['item_nm','item_nm_alias','item_id']].groupby(["item_nm"])['item_id'].apply(list).reset_index()

        return cand_item_list, extra_item_pdf

    def extract_entities_by_logic(self, cand_entities, threshold_for_fuzzy=0.8):
        """
        로직 기반 엔티티 추출 (퍼지 + 시퀀스 유사도)
        
        Args:
            cand_entities: 후보 엔티티 리스트
            threshold_for_fuzzy: 퍼지 매칭 임계값
        
        Returns:
            DataFrame: 추출된 엔티티와 유사도 정보
        """
        # 퍼지 유사도 계산
        similarities_fuzzy = parallel_fuzzy_similarity(
            cand_entities, self.item_pdf_all['item_nm_alias'].unique(), 
            threshold=threshold_for_fuzzy,
            text_col_nm='item_name_in_msg', item_col_nm='item_nm_alias', 
            n_jobs=PROCESSING_CONFIG.n_jobs, batch_size=30
        )
        
        if similarities_fuzzy.empty:
            return pd.DataFrame()
        
        # 시퀀스 유사도 계산 (s1, s2 정규화 방식으로 각각 계산 후 합산)
        cand_entities_sim = parallel_seq_similarity(
            sent_item_pdf=similarities_fuzzy, text_col_nm='item_name_in_msg', item_col_nm='item_nm_alias',
            n_jobs=PROCESSING_CONFIG.n_jobs, batch_size=30, normalizaton_value='s1'
        ).rename(columns={'sim':'sim_s1'}).merge(parallel_seq_similarity(
            sent_item_pdf=similarities_fuzzy, text_col_nm='item_name_in_msg', item_col_nm='item_nm_alias',
            n_jobs=PROCESSING_CONFIG.n_jobs, batch_size=30, normalizaton_value='s2'
        ).rename(columns={'sim':'sim_s2'}), on=['item_name_in_msg','item_nm_alias']).groupby(['item_name_in_msg','item_nm_alias'])[['sim_s1','sim_s2']].apply(lambda x: x['sim_s1'].sum() + x['sim_s2'].sum()).reset_index(name='sim')
        
        return cand_entities_sim

    def extract_entities_by_llm(self, msg_text, rank_limit=5):
        """
        LLM 기반 엔티티 추출
        
        Args:
            msg_text: 분석할 메시지 텍스트
            rank_limit: 반환할 최대 엔티티 수
        
        Returns:
            DataFrame: LLM이 추출한 엔티티와 유사도 정보
        """
        from langchain.prompts import PromptTemplate
        
        # 로직 기반 방식으로 후보 엔티티 먼저 추출
        cand_entities_by_sim = self.extract_entities_by_logic([msg_text], threshold_for_fuzzy=PROCESSING_CONFIG.similarity_threshold)['item_nm_alias'].unique()
        
        # LLM 프롬프트 템플릿 정의
        zero_shot_prompt = PromptTemplate(
            input_variables=["msg","cand_entities"],
            template="""
            Extract all product names, including tangible products, services, promotional events, programs, loyalty initiatives, and named campaigns or event identifiers, from the provided advertisement text.
            Reference the provided candidate entities list as a guide for potential matches. Extract only those terms from the candidate list that appear in the advertisement text and qualify as distinct product names based on the following criteria.
            Consider any named offerings, such as apps, membership programs, events, specific branded items, or campaign names like 'T day' or '0 day', as products if presented as distinct products, services, or promotional entities.
            For terms that may be platforms or brand elements, include them only if they are presented as standalone offerings.
            Avoid extracting base or parent brand names (e.g., 'FLO' or 'POOQ') if they are components of more specific offerings (e.g., 'FLO 앤 데이터' or 'POOQ 앤 데이터') presented in the text; focus on the full, distinct product or service names as they appear.
            Exclude customer support services, such as customer centers or helplines, even if named in the text.
            Exclude descriptive modifiers or attributes (e.g., terms like "디지털 전용" that describe a product but are not distinct offerings).
            Exclude sales agency names such as '###대리점'.
            If multiple terms refer to closely related promotional events (e.g., a general campaign and its specific instances or dates), include the most prominent or overarching campaign name (e.g., '0 day' as a named event) in addition to specific offerings tied to it, unless they are clearly identical.
            Prioritize recall over precision to ensure all relevant products are captured, but verify that each extracted term is a distinct offering from the candidate list that matches the text.
            Ensure that extracted names are presented exactly as they appear in the original text, without translation into English or any other language.
            Just return a list with matched entities where the entities are separated by commas without any other text.

            ## message:                
            {msg}

            ## Candidate entities:
            {cand_entities}
            """
        )
        
        # LLM 체인 실행
        chain = zero_shot_prompt | self.llm_model
        cand_entities = chain.invoke({"msg": msg_text, "cand_entities": cand_entities_by_sim}).content

        # LLM 응답 파싱 및 정리
        cand_entity_list = [e.strip() for e in cand_entities.split(',') if e.strip()]
        cand_entity_list = [e for e in cand_entity_list if e not in self.stop_item_names and len(e)>=2]

        if not cand_entity_list:
            return pd.DataFrame()

        # 퍼지 유사도 매칭
        similarities_fuzzy = parallel_fuzzy_similarity(
            cand_entity_list, 
            self.item_pdf_all['item_nm_alias'].unique(), 
            threshold=0.6,
            text_col_nm='item_name_in_msg',
            item_col_nm='item_nm_alias',
            n_jobs=PROCESSING_CONFIG.n_jobs,
            batch_size=30
        )
        
        if similarities_fuzzy.empty:
            return pd.DataFrame()
        
        # 정지어 필터링
        similarities_fuzzy = similarities_fuzzy[~similarities_fuzzy['item_nm_alias'].isin(self.stop_item_names)]

        # 시퀀스 유사도 매칭
        cand_entities_sim = parallel_seq_similarity(
            sent_item_pdf=similarities_fuzzy,
            text_col_nm='item_name_in_msg',
            item_col_nm='item_nm_alias',
            n_jobs=PROCESSING_CONFIG.n_jobs,
            batch_size=30,
            normalizaton_value='s1'
        ).rename(columns={'sim':'sim_s1'}).merge(parallel_seq_similarity(
            sent_item_pdf=similarities_fuzzy,
            text_col_nm='item_name_in_msg',
            item_col_nm='item_nm_alias',
            n_jobs=PROCESSING_CONFIG.n_jobs,
            batch_size=30,
            normalizaton_value='s2'
        ).rename(columns={'sim':'sim_s2'}), on=['item_name_in_msg','item_nm_alias'])
        
        # 유사도 점수 합산
        cand_entities_sim = cand_entities_sim.groupby(['item_name_in_msg','item_nm_alias'])[['sim_s1','sim_s2']].apply(lambda x: x['sim_s1'].sum() + x['sim_s2'].sum()).reset_index(name='sim')
        cand_entities_sim = cand_entities_sim.query("sim>=1.5")

        # 순위 매기기 및 결과 제한
        cand_entities_sim["rank"] = cand_entities_sim.groupby('item_name_in_msg')['sim'].rank(method='first',ascending=False)
        cand_entities_sim = cand_entities_sim.query(f"rank<={rank_limit}").sort_values(['item_name_in_msg','rank'], ascending=[True,True])

        return cand_entities_sim

    def process_message(self, mms_msg):
        """
        MMS 메시지 전체 처리 (메인 처리 함수)
        
        Args:
            mms_msg: 처리할 MMS 메시지 텍스트
        
        Returns:
            dict: 추출된 정보가 담긴 JSON 구조
                - title: 광고 제목
                - purpose: 광고 목적
                - product: 상품 정보 리스트
                - channel: 채널 정보 리스트
                - pgm: 프로그램 분류 정보
        """
        print(f"Processing message: {mms_msg[:100]}...")
        msg = mms_msg.strip()
        
        # Kiwi를 통한 초기 엔티티 추출
        cand_item_list, extra_item_pdf = self.extract_entities_from_kiwi(msg)
        
        # NLP 모드용 상품 요소 준비
        product_df = extra_item_pdf.rename(columns={'item_nm':'name'}).query("not name in @self.stop_item_names")[['name']]
        product_df['action'] = '고객에게 기대하는 행동: [구매, 가입, 사용, 방문, 참여, 코드입력, 쿠폰다운로드, 기타] 중에서 선택'
        product_element = product_df.to_dict(orient='records') if product_df.shape[0] > 0 else None
        
        # 메시지 임베딩 및 프로그램 분류 유사도 계산
        mms_embedding = self.emb_model.encode([msg.lower()], convert_to_tensor=True)
        similarities = torch.nn.functional.cosine_similarity(mms_embedding, self.clue_embeddings, dim=1).cpu().numpy()
        
        # 상위 후보 프로그램들 선별
        pgm_pdf_tmp = self.pgm_pdf.copy()
        pgm_pdf_tmp['sim'] = similarities
        pgm_pdf_tmp = pgm_pdf_tmp.sort_values('sim', ascending=False)
        pgm_cand_info = "\n\t".join(pgm_pdf_tmp.iloc[:self.num_cand_pgms][['pgm_nm','clue_tag']].apply(lambda x: re.sub(r'\[.*?\]', '', x['pgm_nm'])+" : "+x['clue_tag'], axis=1).to_list())
        rag_context = f"\n### 광고 분류 기준 정보 ###\n\t{pgm_cand_info}" if self.num_cand_pgms > 0 else ""

        # LLM 처리를 위한 사고 과정 정의
        chain_of_thought = """
1. Identify the advertisement's purpose first, using expressions as they appear in the original text.
2. Extract product names based on the identified purpose, ensuring only distinct offerings are included and using original text expressions.
3. Provide channel information considering the extracted product information, preserving original text expressions.
"""

        # JSON 스키마 정의 (LLM 응답 구조화용)
        schema_prd = {
            "title": {
                "type": "string",
                "description": "Advertisement title, using the exact expressions as they appear in the original text. Clearly describe the core theme and value proposition of the advertisement."
            },
            "purpose": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["상품 가입 유도", "대리점/매장 방문 유도", "웹/앱 접속 유도", "이벤트 응모 유도", "혜택 안내", "쿠폰 제공 안내", "경품 제공 안내", "수신 거부 안내", "기타 정보 제공"]
                },
                "description": "Primary purpose(s) of the advertisement, expressed using the exact terms from the original text where applicable."
            },
            "product": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name of the advertised product or service, as it appears in the original text without translation."
                        },
                        "action": {
                            "type": "string",
                            "enum": ["구매", "가입", "사용", "방문", "참여", "코드입력", "쿠폰다운로드", "기타"],
                            "description": "Expected customer action for the product, derived from the original text context."
                        }
                    }
                },
            "description": "Extract all product names, including tangible products, services, promotional events, programs, loyalty initiatives, and named campaigns or event identifiers, using the exact expressions as they appear in the original text without translation. Consider only named offerings (e.g., apps, membership programs, events, specific branded items, or campaign names like 'T day' or '0 day') presented as distinct products, services, or promotional entities. Include platform or brand elements only if explicitly presented as standalone offerings. Avoid extracting base or parent brand names (e.g., 'FLO' or 'POOQ') if they are components of more specific offerings (e.g., 'FLO 앤 데이터' or 'POOQ 앤 데이터') presented in the text; focus on the full, distinct product or service names as they appear. Exclude customer support services (e.g., customer centers, helplines). Exclude descriptive modifiers, attributes, or qualifiers (e.g., '디지털 전용'). Exclude sales agency names such as '###대리점'. If multiple terms refer to closely related promotional events (e.g., a general campaign and its specific instances or dates), include the most prominent or overarching campaign name (e.g., '0 day' as a named event) in addition to specific offerings tied to it, unless they are clearly identical. Prioritize recall over precision, but verify each term is a distinct offering."            },
            "channel": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["URL", "전화번호", "앱", "대리점"],
                            "description": "Channel type, as derived from the original text."
                        },
                        "value": {
                            "type": "string",
                            "description": "Specific information for the channel (e.g., URL, phone number, app name, agency name), as it appears in the original text."
                        },
                        "action": {
                            "type": "string",
                            "enum": ["가입", "추가 정보", "문의", "수신", "수신 거부"],
                            "description": "Purpose of the channel, derived from the original text context."
                        }
                    }
                },
                "description": "Channels provided in the advertisement, including URLs, phone numbers, apps, or agencies, using the exact expressions from the original text where applicable, based on the purpose and products."
            },
            "pgm": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "Select the two most relevant pgm_nm from the advertising classification criteria, using the exact expressions from the criteria, ordered by relevance, based on the message content."
            }
        }

        # 추출 가이드라인 설정
        prd_ext_guide = """
* Prioritize recall over precision to ensure all relevant products are captured, but verify that each extracted term is a distinct offering.
* Extract all information (title, purpose, product, channel, pgm) using the exact expressions as they appear in the original text without translation, as specified in the schema.
* If the advertisement purpose includes encouraging agency/store visits, provide agency channel information.
"""

        # 추출 모드별 처리
        if len(cand_item_list) > 0:
            if self.product_info_extraction_mode == 'rag':
                # RAG 모드: 후보 상품 목록을 컨텍스트로 제공
                rag_context += f"\n\n### 후보 상품 이름 목록 ###\n\t{cand_item_list}"
                prd_ext_guide += """
* Use the provided candidate product names as a reference to guide product extraction, ensuring alignment with the advertisement content and using exact expressions from the original text.
"""
            elif self.product_info_extraction_mode == 'nlp' and product_element:
                # NLP 모드: 미리 추출된 상품 정보 사용
                schema_prd['product'] = product_element
                chain_of_thought = """
1. Identify the advertisement's purpose first, using expressions as they appear in the original text.
2. Extract product information based on the identified purpose, ensuring only distinct offerings are included and using original text expressions.
3. Extract the action field for each product based on the provided name information, derived from the original text context.
4. Provide channel information considering the extracted product information, preserving original text expressions.
"""
                prd_ext_guide += """
* Extract the action field for each product based on the identified product names, using the original text context.
"""

        # LLM 프롬프트 구성
        schema_prompt = f"""
Provide the results in the following schema:

{json.dumps(schema_prd, indent=4, ensure_ascii=False)}
"""

        prompt = f"""
Extract the advertisement purpose and product names from the provided advertisement text.

### Advertisement Message ###
{msg}

### Extraction Steps ###
{chain_of_thought}

### Extraction Guidelines ###
{prd_ext_guide}

{schema_prompt}

{rag_context}
"""

        # LLM 실행 및 JSON 파싱
        result_json_text = self.llm_model.invoke(prompt).content
        json_objects_list = extract_json_objects(result_json_text)
        if not json_objects_list:
            print("LLM did not return a valid JSON object.")
            return {}
        
        json_objects = json_objects_list[0]
        
        # LLM 응답에서 상품 정보 추출
        product_items = json_objects.get('product', [])
        if isinstance(product_items, dict):
            product_items = product_items.get('items', [])
        
        # 엔티티 매칭 모드에 따른 처리
        if self.entity_extraction_mode == 'logic':
            # 로직 기반: 퍼지 + 시퀀스 유사도
            cand_entities = [item['name'] for item in product_items]
            similarities_fuzzy = self.extract_entities_by_logic(cand_entities)
        else:
            # LLM 기반: LLM을 통한 엔티티 추출
            similarities_fuzzy = self.extract_entities_by_llm(msg)

        final_result = json_objects.copy()
        
        print("Entity from LLM:", [x['name'] for x in product_items])

        # 유사도 결과가 있는 경우 상품 정보 매핑
        if not similarities_fuzzy.empty:
            # 높은 유사도 아이템들 필터링
            high_sim_items = similarities_fuzzy.query('sim >= 1.5')['item_nm_alias'].unique()
            filtered_similarities = similarities_fuzzy[
                (similarities_fuzzy['item_nm_alias'].isin(high_sim_items)) &
                (~similarities_fuzzy['item_nm_alias'].str.contains('test', case=False)) &
                (~similarities_fuzzy['item_name_in_msg'].isin(self.stop_item_names))
            ]
            # 상품 정보와 매핑하여 최종 결과 생성
            product_tag = convert_df_to_json_list(self.item_pdf_all.merge(filtered_similarities, on=['item_nm_alias'])) # 대리점 제외
            final_result['product'] = product_tag
        else:
            # 유사도 결과가 없으면 LLM 결과 그대로 사용
            product_items = json_objects.get('product', [])
            if isinstance(product_items, dict):
                product_items = product_items.get('items', [])
            final_result['product'] = [
                {'item_name_in_msg':d['name'], 'item_in_voca':[{'item_name_in_voca':d['name'], 'item_id': ['#']}]} 
                for d in product_items 
                if d.get('name') and d['name'] not in self.stop_item_names
            ]

        # 프로그램 분류 정보 매핑
        if self.num_cand_pgms > 0 and 'pgm' in json_objects and isinstance(json_objects['pgm'], list):
            pgm_json = self.pgm_pdf[self.pgm_pdf['pgm_nm'].apply(lambda x: re.sub(r'\[.*?\]', '', x) in ' '.join(json_objects['pgm']))][['pgm_nm','pgm_id']].to_dict('records')
            final_result['pgm'] = pgm_json

        # 채널 정보 처리 (특히 대리점 정보 매칭)
        channel_tag = []
        channel_items = json_objects.get('channel', [])
        if isinstance(channel_items, dict):
            channel_items = channel_items.get('items', [])

        for d in channel_items:
            if d.get('type') == '대리점' and d.get('value'):
                # 대리점명으로 조직 정보 검색
                org_pdf_cand = parallel_fuzzy_similarity(
                    [preprocess_text(d['value'].lower())], self.org_pdf['org_abbr_nm'].unique(), 
                    threshold=0.5, text_col_nm='org_nm_in_msg', item_col_nm='org_abbr_nm', 
                    n_jobs=PROCESSING_CONFIG.n_jobs, batch_size=PROCESSING_CONFIG.batch_size
                ).drop('org_nm_in_msg', axis=1)

                if not org_pdf_cand.empty:
                    # 조직 정보와 매칭
                    org_pdf_cand = self.org_pdf.merge(org_pdf_cand, on=['org_abbr_nm'])
                    org_pdf_cand['sim'] = org_pdf_cand.apply(lambda x: combined_sequence_similarity(d['value'], x['org_nm'])[0], axis=1).round(5)
                    
                    # 대리점 코드('D'로 시작) 우선 검색
                    org_pdf_tmp = org_pdf_cand.query("org_cd.str.startswith('D') & sim >= @PROCESSING_CONFIG.similarity_threshold", engine='python').sort_values('sim', ascending=False)
                    if org_pdf_tmp.empty:
                        # 대리점이 없으면 전체에서 검색
                        org_pdf_tmp = org_pdf_cand.query("sim>=@PROCESSING_CONFIG.similarity_threshold").sort_values('sim', ascending=False)
                    
                    if not org_pdf_tmp.empty:
                        # 최고 순위 조직들의 정보 추출
                        org_pdf_tmp['rank'] = org_pdf_tmp['sim'].rank(method='dense',ascending=False)
                        org_pdf_tmp['org_cd_full'] = org_pdf_tmp.apply(lambda x: x['org_cd']+x['sub_org_cd'], axis=1)
                        org_info = org_pdf_tmp.query("rank==1").groupby('org_nm')['org_cd_full'].apply(list).reset_index(name='org_cd').to_dict('records')
                        d['store_info'] = org_info
                    else:
                        d['store_info'] = []
                else:
                    d['store_info'] = []
            else:
                d['store_info'] = []
            channel_tag.append(d)

        final_result['channel'] = channel_tag
        return final_result

if __name__ == '__main__':
    """
    커맨드라인에서 실행할 때의 메인 함수
    다양한 옵션을 통해 추출기 설정을 변경할 수 있습니다.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='MMS 광고 텍스트 추출기')
    parser.add_argument('--message', type=str, help='테스트할 메시지')
    parser.add_argument('--offer-data-source', choices=['local', 'db'], default='local',
                       help='데이터 소스 (local: CSV 파일, db: 데이터베이스)')
    parser.add_argument('--product-info-extraction-mode', choices=['nlp', 'llm' ,'rag'], default='nlp',
                       help='상품 정보 추출 모드 (nlp: 형태소분석, llm: LLM 기반, rag: 검색증강생성)')
    parser.add_argument('--entity-matching-mode', choices=['logic', 'llm'], default='llm',
                       help='엔티티 매칭 모드 (logic: 로직 기반, llm: LLM 기반)')
    parser.add_argument('--llm-model', choices=['gemma', 'gpt', 'claude'], default='gemma',
                       help='사용할 LLM 모델 (gemma: Gemma, gpt: GPT, claude: Claude)')
    
    args = parser.parse_args()
    
    # 파싱된 인자들 사용
    offer_info_data_src = args.offer_data_source
    product_info_extraction_mode = args.product_info_extraction_mode
    entity_extraction_mode = args.entity_matching_mode
    llm_model = args.llm_model
    
    # 추출기 초기화
    extractor = MMSExtractor(
        offer_info_data_src=offer_info_data_src, 
        product_info_extraction_mode=product_info_extraction_mode, 
        entity_extraction_mode=entity_extraction_mode, 
        llm_model=llm_model
    )
    
    # 테스트 메시지
    test_text = """
    [SK텔레콤] ZEM폰 포켓몬에디션3 안내
    (광고)[SKT] 우리 아이 첫 번째 스마트폰, ZEM 키즈폰__#04 고객님, 안녕하세요!
    우리 아이 스마트폰 고민 중이셨다면, 자녀 스마트폰 관리 앱 ZEM이 설치된 SKT만의 안전한 키즈폰,
    ZEM폰 포켓몬에디션3으로 우리 아이 취향을 저격해 보세요!
    신학기를 맞이하여 SK텔레콤 공식 인증 대리점에서 풍성한 혜택을 제공해 드리고 있습니다!
    ■ 주요 기능
    1. 실시간 위치 조회
    2. 모르는 회선 자동 차단
    3. 스마트폰 사용 시간 제한
    4. IP68 방수 방진
    5. 수업 시간 자동 무음모드
    6. 유해 콘텐츠 차단
    ■ 가까운 SK텔레콤 공식 인증 대리점 찾기
    http://t-mms.kr/t.do?m=#61&s=30684&a=&u=https://bit.ly/3yQF2hx
    ■ 문의 : SKT 고객센터(1558, 무료)
    무료 수신거부 1504
    """

    # 메시지 처리 및 결과 출력
    result = extractor.process_message(test_text)
    
    print("\n" + "="*40)
    print("최종 추출된 정보")
    print("="*40)
    print(json.dumps(result, indent=4, ensure_ascii=False)) 