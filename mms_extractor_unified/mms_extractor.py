# %%
"""
MMS 추출기 (MMS Extractor) - 개선된 버전
==============================================

이 모듈은 MMS(멀티미디어 메시지) 광고 텍스트에서 상품명, 채널 정보, 광고 목적 등을 
자동으로 추출하는 시스템입니다.

주요 개선사항:
- 코드 모듈화: 대형 메소드들을 기능별로 분리
- 예외 처리 강화: 안전한 LLM 호출 및 에러 복구
- 로깅 시스템 추가: 성능 모니터링 및 디버깅 지원
- 데이터 검증 추가: 추출 결과 품질 보장
- 데이터베이스 지원: 상품 정보와 프로그램 분류 정보 모두 DB에서 로드 가능
"""

from concurrent.futures import ThreadPoolExecutor
import time
import logging
from functools import wraps
from typing import List, Tuple, Union, Dict, Any, Optional
from abc import ABC, abstractmethod
import traceback
import json
import re
import ast
import glob
import os
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from difflib import SequenceMatcher
import difflib
from dotenv import load_dotenv
import cx_Oracle

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from rapidfuzz import fuzz, process
from kiwipiepy import Kiwi
from joblib import Parallel, delayed

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
    """안전한 함수 실행을 위한 유틸리티"""
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries:
                logger.error(f"{func.__name__} 최종 실패: {e}")
                return default_return
            else:
                logger.warning(f"{func.__name__} 재시도 {attempt + 1}/{max_retries}: {e}")
                time.sleep(2 ** attempt)  # 지수 백오프
    return default_return

def validate_text_input(text: str) -> str:
    """텍스트 입력 검증 및 정리"""
    if not isinstance(text, str):
        raise ValueError(f"텍스트 입력이 문자열이 아닙니다: {type(text)}")
    
    text = text.strip()
    if not text:
        raise ValueError("빈 텍스트는 처리할 수 없습니다")
    
    if len(text) > 10000:  # 너무 긴 텍스트 제한
        logger.warning(f"텍스트가 너무 깁니다 ({len(text)} 문자). 처음 10000자만 사용합니다.")
        text = text[:10000]
    
    return text

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
    MMS 광고 텍스트 추출기 메인 클래스 - 개선된 버전
    
    주요 개선사항:
    - 메소드 모듈화: 큰 메소드들을 기능별로 분리
    - 예외 처리 강화: 안전한 LLM 호출 및 에러 복구
    - 로깅 시스템: 성능 모니터링 및 디버깅
    - 데이터 검증: 입출력 데이터 품질 보장
    """
    
    def __init__(self, model_path=None, data_dir=None, product_info_extraction_mode=None, 
                 entity_extraction_mode=None, offer_info_data_src='local', llm_model='gemma'):
        """MMSExtractor 초기화"""
        logger.info("MMSExtractor 초기화 시작")
        
        try:
            # 기본 설정 적용
            self._set_default_config(model_path, data_dir, product_info_extraction_mode, 
                                   entity_extraction_mode, offer_info_data_src, llm_model)
            
            # 환경변수 로드
            load_dotenv()
            
            # 단계별 초기화
            self._initialize_device()
            self._initialize_llm()
            self._initialize_embedding_model()
            self._initialize_kiwi()
            self._load_data()
            
            logger.info("MMSExtractor 초기화 완료")
            
        except Exception as e:
            logger.error(f"MMSExtractor 초기화 실패: {e}")
            logger.error(traceback.format_exc())
            raise

    def _set_default_config(self, model_path, data_dir, product_info_extraction_mode, 
                          entity_extraction_mode, offer_info_data_src, llm_model):
        """기본 설정값 적용"""
        self.data_dir = data_dir if data_dir is not None else './data/'
        self.model_path = model_path if model_path is not None else getattr(EMBEDDING_CONFIG, 'ko_sbert_model_path', 'jhgan/ko-sroberta-multitask')
        self.offer_info_data_src = offer_info_data_src
        self.product_info_extraction_mode = product_info_extraction_mode if product_info_extraction_mode is not None else getattr(PROCESSING_CONFIG, 'product_info_extraction_mode', 'nlp')
        self.entity_extraction_mode = entity_extraction_mode if entity_extraction_mode is not None else getattr(PROCESSING_CONFIG, 'entity_extraction_mode', 'llm')
        self.llm_model_name = llm_model
        self.num_cand_pgms = getattr(PROCESSING_CONFIG, 'num_candidate_programs', 5)

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
                "ax": getattr(MODEL_CONFIG, 'ax_model', 'gpt-4'),
                "claude": getattr(MODEL_CONFIG, 'claude_model', 'claude-3'),
                "gemini": getattr(MODEL_CONFIG, 'gemini_model', 'gemini-pro'),
                "gpt": getattr(MODEL_CONFIG, 'gpt_model', 'gpt-4')
            }
            
            model_name = model_mapping.get(self.llm_model_name, getattr(MODEL_CONFIG, 'llm_model', 'gpt-4'))
            
            self.llm_model = ChatOpenAI(
                temperature=getattr(MODEL_CONFIG, 'temperature', 0.1),
                openai_api_key=getattr(API_CONFIG, 'llm_api_key', os.getenv('OPENAI_API_KEY')),
                openai_api_base=getattr(API_CONFIG, 'llm_api_url', None),
                model=model_name,
                max_tokens=getattr(MODEL_CONFIG, 'llm_max_tokens', 4000)
            )
            
            logger.info(f"LLM 초기화 완료: {self.llm_model_name} ({model_name})")
            
        except Exception as e:
            logger.error(f"LLM 초기화 실패: {e}")
            raise

    @log_performance
    def _initialize_embedding_model(self):
        """임베딩 모델 초기화"""
        try:
            self.emb_model = load_sentence_transformer(self.model_path, self.device)
        except Exception as e:
            logger.error(f"임베딩 모델 초기화 실패: {e}")
            # 기본 모델로 fallback
            logger.info("기본 모델로 fallback 시도")
            self.emb_model = load_sentence_transformer('jhgan/ko-sroberta-multitask', self.device)

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
            logger.info("데이터 로딩 시작")
            
            # 상품 정보 로드
            self._load_item_data()
            
            # 별칭 규칙 적용
            self._apply_alias_rules()
            
            # 정지어 로드
            self._load_stop_words()
            
            # Kiwi에 상품명 등록
            self._register_items_to_kiwi()
            
            # 프로그램 분류 정보 로드
            self._load_program_data()
            
            # 조직 정보 로드
            self._load_organization_data()
            
            logger.info("데이터 로딩 완료")
            
        except Exception as e:
            logger.error(f"데이터 로딩 실패: {e}")
            raise

    def _load_item_data(self):
        """상품 정보 로드"""
        try:
            if self.offer_info_data_src == "local":
                # 로컬 CSV 파일에서 로드
                item_pdf_raw = pd.read_csv(getattr(METADATA_CONFIG, 'offer_data_path', './data/items.csv'))
                self.item_pdf_all = item_pdf_raw.drop_duplicates(['item_nm','item_id'])[['item_nm','item_id','item_desc','item_dmn']].copy()
                self.item_pdf_all['item_ctg'] = None
                self.item_pdf_all['item_emb_vec'] = None
                self.item_pdf_all['ofer_cd'] = self.item_pdf_all['item_id']
                self.item_pdf_all['oper_dt_hms'] = '20250101000000'
                self.item_pdf_all = self.item_pdf_all.rename(columns={c:c.lower() for c in self.item_pdf_all.columns})
                
            elif self.offer_info_data_src == "db":
                # 데이터베이스에서 로드
                self._load_item_from_database()
            
            # 제외할 도메인 코드 필터링
            excluded_domains = getattr(PROCESSING_CONFIG, 'excluded_domain_codes_for_items', [])
            if excluded_domains:
                self.item_pdf_all = self.item_pdf_all.query("item_dmn not in @excluded_domains")
                
            logger.info(f"상품 정보 로드 완료: {len(self.item_pdf_all)}개 상품")
            
        except Exception as e:
            logger.error(f"상품 정보 로드 실패: {e}")
            # 빈 DataFrame으로 fallback
            self.item_pdf_all = pd.DataFrame(columns=['item_nm', 'item_id', 'item_desc', 'item_dmn'])

    def _get_database_connection(self):
        """Oracle 데이터베이스 연결 생성"""
        try:
            username = os.getenv("DB_USERNAME")
            password = os.getenv("DB_PASSWORD")
            host = os.getenv("DB_HOST")
            port = os.getenv("DB_PORT")
            service_name = os.getenv("DB_NAME")
            
            if not all([username, password, host, port, service_name]):
                raise ValueError("데이터베이스 연결 정보가 불완전합니다")
            
            dsn = cx_Oracle.makedsn(host, port, service_name=service_name)
            conn = cx_Oracle.connect(user=username, password=password, dsn=dsn, encoding="UTF-8")
            
            return conn
            
        except Exception as e:
            logger.error(f"데이터베이스 연결 실패: {e}")
            raise

    def _load_item_from_database(self):
        """데이터베이스에서 상품 정보 로드"""
        try:
            conn = self._get_database_connection()
            
            sql = "SELECT * FROM TCAM_RC_OFER_MST WHERE ROWNUM <= 1000000"
            self.item_pdf_all = pd.read_sql(sql, conn)
            conn.close()
            
            self.item_pdf_all = self.item_pdf_all.rename(columns={c:c.lower() for c in self.item_pdf_all.columns})
            
        except Exception as e:
            logger.error(f"상품 정보 데이터베이스 로드 실패: {e}")
            raise

    def _load_program_from_database(self):
        """데이터베이스에서 프로그램 분류 정보 로드"""
        try:
            conn = self._get_database_connection()
            
            # 프로그램 분류 정보 쿼리
            sql = """SELECT CMPGN_PGM_NUM pgm_id, CMPGN_PGM_NM pgm_nm, RMK clue_tag 
                     FROM TCAM_CMPGN_PGM_INFO
                     WHERE DEL_YN = 'N' 
                     AND APRV_OP_RSLT_CD = 'APPR'
                     AND EXPS_YN = 'Y'
                     AND CMPGN_PGM_NUM like '2025%' 
                     AND RMK is not null"""
            
            self.pgm_pdf = pd.read_sql(sql, conn)
            self.pgm_pdf = self.pgm_pdf.rename(columns={c:c.lower() for c in self.pgm_pdf.columns})
            conn.close()
            
            logger.info(f"데이터베이스에서 프로그램 분류 정보 로드 완료: {len(self.pgm_pdf)}개")
            
        except Exception as e:
            logger.error(f"프로그램 분류 정보 데이터베이스 로드 실패: {e}")
            # 빈 데이터로 fallback
            self.pgm_pdf = pd.DataFrame(columns=['pgm_nm', 'clue_tag', 'pgm_id'])
            raise

    def _apply_alias_rules(self):
        """별칭 규칙 적용"""
        try:
            alias_pdf = pd.read_csv(getattr(METADATA_CONFIG, 'alias_rules_path', './data/alias_rules.csv'))
            alias_rule_set = list(zip(alias_pdf['alias_1'], alias_pdf['alias_2']))

            def apply_alias_rule(item_nm):
                item_nm_list = [item_nm]
                for r in alias_rule_set:
                    if r[0] in item_nm:
                        item_nm_list.append(item_nm.replace(r[0], r[1]))
                    if r[1] in item_nm:
                        item_nm_list.append(item_nm.replace(r[1], r[0]))
                return item_nm_list

            self.item_pdf_all['item_nm_alias'] = self.item_pdf_all['item_nm'].apply(apply_alias_rule)
            self.item_pdf_all = self.item_pdf_all.explode('item_nm_alias')
            
            logger.info(f"별칭 규칙 적용 완료: {len(alias_rule_set)}개 규칙")
            
        except Exception as e:
            logger.warning(f"별칭 규칙 적용 실패: {e}")
            # 원본 이름을 별칭으로 사용
            self.item_pdf_all['item_nm_alias'] = self.item_pdf_all['item_nm']

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
            registered_count = 0
            for w in self.item_pdf_all['item_nm_alias'].unique():
                if isinstance(w, str) and len(w.strip()) > 0:
                    self.kiwi.add_user_word(w, "NNP")
                    registered_count += 1
            
            logger.info(f"Kiwi에 상품명 등록 완료: {registered_count}개")
            
        except Exception as e:
            logger.error(f"Kiwi 상품명 등록 실패: {e}")

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
                
                self.clue_embeddings = self.emb_model.encode(
                    clue_texts, convert_to_tensor=True, show_progress_bar=False
                )
                
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
            self.org_pdf = pd.read_csv(getattr(METADATA_CONFIG, 'org_info_path', './data/org_info.csv'), encoding='cp949')
            self.org_pdf['sub_org_cd'] = self.org_pdf['sub_org_cd'].apply(lambda x: str(x).zfill(4))
            logger.info(f"조직 정보 로드 완료: {len(self.org_pdf)}개 조직")
        except Exception as e:
            logger.warning(f"조직 정보 로드 실패: {e}")
            self.org_pdf = pd.DataFrame(columns=['org_nm', 'org_cd', 'sub_org_cd', 'org_abbr_nm'])

    def _safe_llm_invoke(self, prompt: str, max_retries: int = 3) -> str:
        """안전한 LLM 호출 메소드"""
        for attempt in range(max_retries):
            try:
                response = self.llm_model.invoke(prompt)
                return response.content if hasattr(response, 'content') else str(response)
                
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"LLM 호출 최종 실패: {e}")
                    return self._fallback_extraction(prompt)
                else:
                    logger.warning(f"LLM 호출 재시도 {attempt + 1}/{max_retries}: {e}")
                    time.sleep(2 ** attempt)  # 지수 백오프
        
        return ""

    def _fallback_extraction(self, prompt: str) -> str:
        """LLM 실패 시 fallback 추출 로직"""
        logger.info("Fallback 추출 로직 실행")
        
        # 기본적인 패턴 매칭으로 정보 추출 시도
        fallback_result = {
            "title": "광고 메시지",
            "purpose": ["정보 제공"],
            "product": [],
            "channel": [],
            "pgm": []
        }
        
        return json.dumps(fallback_result, ensure_ascii=False)

    @log_performance
    def extract_entities_from_kiwi(self, mms_msg: str) -> Tuple[List[str], pd.DataFrame]:
        """Kiwi 형태소 분석기를 사용한 엔티티 추출"""
        try:
            mms_msg = validate_text_input(mms_msg)
            
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
            
            # 제외 패턴을 적용하여 문장 필터링
            sentence_list = [
                filter_text_by_exc_patterns(sent, self.exc_tag_patterns) 
                for sent in sentences_all
            ]

            # 형태소 분석을 통한 고유명사 추출
            result_msg = self.kiwi.tokenize(mms_msg, normalize_coda=True, z_coda=False, split_complex=False)
            entities_from_kiwi = [
                token.form for token in result_msg 
                if token.tag == 'NNP' and 
                   token.form not in self.stop_item_names + ['-'] and 
                   len(token.form) >= 2 and 
                   not token.form.lower() in self.stop_item_names
            ]
            entities_from_kiwi = filter_specific_terms(entities_from_kiwi)
            
            logger.info(f"Kiwi 추출 엔티티: {list(set(entities_from_kiwi))}")

            # 퍼지 매칭을 통한 유사 상품명 찾기
            similarities_fuzzy = safe_execute(
                parallel_fuzzy_similarity,
                sentence_list, 
                self.item_pdf_all['item_nm_alias'].unique(),
                threshold=getattr(PROCESSING_CONFIG, 'fuzzy_threshold', 0.5),
                text_col_nm='sent', 
                item_col_nm='item_nm_alias',
                n_jobs=getattr(PROCESSING_CONFIG, 'n_jobs', 4),
                batch_size=30,
                default_return=pd.DataFrame()
            )
            
            if similarities_fuzzy.empty:
                # 퍼지 매칭 결과가 없으면 Kiwi 결과만 사용
                cand_item_list = entities_from_kiwi
                extra_item_pdf = self.item_pdf_all.query("item_nm_alias in @cand_item_list")[
                    ['item_nm','item_nm_alias','item_id']
                ].groupby(["item_nm"])['item_id'].apply(list).reset_index()
                return cand_item_list, extra_item_pdf

            # 시퀀스 유사도를 통한 정밀 매칭
            similarities_seq = safe_execute(
                parallel_seq_similarity,
                sent_item_pdf=similarities_fuzzy,
                text_col_nm='sent',
                item_col_nm='item_nm_alias',
                n_jobs=getattr(PROCESSING_CONFIG, 'n_jobs', 4),
                batch_size=getattr(PROCESSING_CONFIG, 'batch_size', 100),
                default_return=pd.DataFrame()
            )
            
            # 임계값 이상의 후보 아이템들 필터링
            similarity_threshold = getattr(PROCESSING_CONFIG, 'similarity_threshold', 0.2)
            cand_items = similarities_seq.query(
                "sim >= @similarity_threshold and "
                "item_nm_alias.str.contains('', case=False) and "
                "item_nm_alias not in @self.stop_item_names"
            )
            
            # Kiwi에서 추출한 엔티티들 추가
            entities_from_kiwi_pdf = self.item_pdf_all.query("item_nm_alias in @entities_from_kiwi")[
                ['item_nm','item_nm_alias']
            ]
            entities_from_kiwi_pdf['sim'] = 1.0

            # 결과 통합 및 최종 후보 리스트 생성
            cand_item_pdf = pd.concat([cand_items, entities_from_kiwi_pdf])
            cand_item_list = cand_item_pdf.sort_values('sim', ascending=False).groupby([
                "item_nm_alias"
            ])['sim'].max().reset_index(name='final_sim').sort_values(
                'final_sim', ascending=False
            ).query("final_sim >= 0.2")['item_nm_alias'].unique()
            
            extra_item_pdf = self.item_pdf_all.query("item_nm_alias in @cand_item_list")[
                ['item_nm','item_nm_alias','item_id']
            ].groupby(["item_nm"])['item_id'].apply(list).reset_index()

            return cand_item_list, extra_item_pdf
            
        except Exception as e:
            logger.error(f"Kiwi 엔티티 추출 실패: {e}")
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
                e.strip() for e in self.extract_entities_by_logic([msg_text], threshold_for_fuzzy=0.7)['item_nm_alias'].unique() 
                if e.strip() not in self.stop_item_names and len(e.strip()) >= 2
            ])

            # LLM 프롬프트 구성
            prompt = f"""
            {getattr(PROCESSING_CONFIG, 'entity_extraction_prompt', '다음 메시지에서 상품명을 추출하세요.')}

            ## message:                
            {msg_text}

            ## Candidate entities:
            {cand_entities_by_sim}
            """
            
            # LLM 호출
            cand_entities = self._safe_llm_invoke(prompt)
            
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
            
            cand_entities_sim = cand_entities_sim.query("sim >= 1.5").copy()

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
            return [], pd.DataFrame()

    def _classify_programs(self, mms_msg: str) -> Dict[str, Any]:
        """프로그램 분류"""
        try:
            if self.clue_embeddings.numel() == 0:
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
        """추출용 프롬프트 구성"""
        
        # 사고 과정 정의
        chain_of_thought = """
1. Identify the advertisement's purpose first, using expressions as they appear in the original text.
2. Extract product names based on the identified purpose, ensuring only distinct offerings are included and using original text expressions.
3. Provide channel information considering the extracted product information, preserving original text expressions.
"""

        # JSON 스키마 정의
        schema_prd = {
            "title": {
                "type": "string",
                "description": "Advertisement title, using the exact expressions as they appear in the original text."
            },
            "purpose": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["상품 가입 유도", "대리점/매장 방문 유도", "웹/앱 접속 유도", "이벤트 응모 유도", 
                           "혜택 안내", "쿠폰 제공 안내", "경품 제공 안내", "수신 거부 안내", "기타 정보 제공"]
                },
                "description": "Primary purpose(s) of the advertisement."
            },
            "product": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Name of the advertised product or service."},
                        "action": {
                            "type": "string",
                            "enum": ["구매", "가입", "사용", "방문", "참여", "코드입력", "쿠폰다운로드", "기타"],
                            "description": "Expected customer action for the product."
                        }
                    }
                },
                "description": "Extract all product names from the advertisement."
            },
            "channel": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["URL", "전화번호", "앱", "대리점"],
                            "description": "Channel type."
                        },
                        "value": {"type": "string", "description": "Specific information for the channel."},
                        "action": {
                            "type": "string",
                            "enum": ["가입", "추가 정보", "문의", "수신", "수신 거부"],
                            "description": "Purpose of the channel."
                        }
                    }
                },
                "description": "Channels provided in the advertisement."
            },
            "pgm": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Select the two most relevant pgm_nm from the advertising classification criteria."
            }
        }

        # 추출 가이드라인 설정
        prd_ext_guide = """
* Prioritize recall over precision to ensure all relevant products are captured.
* Extract all information using the exact expressions as they appear in the original text.
* If the advertisement purpose includes encouraging agency/store visits, provide agency channel information.
"""

        # 제품 정보 모드에 따른 스키마 조정
        if self.product_info_extraction_mode == 'nlp' and product_element:
            schema_prd['product'] = product_element
            chain_of_thought = """
1. Identify the advertisement's purpose first, using expressions as they appear in the original text.
2. Extract product information based on the identified purpose, ensuring only distinct offerings are included.
3. Extract the action field for each product based on the provided name information.
4. Provide channel information considering the extracted product information.
"""
            prd_ext_guide += """
* Extract the action field for each product based on the identified product names.
"""

        # 프롬프트 구성
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
                self.org_pdf['org_abbr_nm'].unique(),
                threshold=0.5,
                text_col_nm='org_nm_in_msg',
                item_col_nm='org_abbr_nm',
                n_jobs=getattr(PROCESSING_CONFIG, 'n_jobs', 4),
                batch_size=getattr(PROCESSING_CONFIG, 'batch_size', 100),
                default_return=pd.DataFrame()
            )

            if org_pdf_cand.empty:
                return []

            org_pdf_cand = org_pdf_cand.drop('org_nm_in_msg', axis=1)
            org_pdf_cand = self.org_pdf.merge(org_pdf_cand, on=['org_abbr_nm'])
            org_pdf_cand['sim'] = org_pdf_cand.apply(
                lambda x: combined_sequence_similarity(store_name, x['org_nm'])[0], axis=1
            ).round(5)
            
            # 대리점 코드('D'로 시작) 우선 검색
            similarity_threshold = getattr(PROCESSING_CONFIG, 'similarity_threshold', 0.2)
            org_pdf_tmp = org_pdf_cand.query(
                "org_cd.str.startswith('D') & sim >= @similarity_threshold", engine='python'
            ).sort_values('sim', ascending=False)
            
            if org_pdf_tmp.empty:
                # 대리점이 없으면 전체에서 검색
                org_pdf_tmp = org_pdf_cand.query("sim >= @similarity_threshold").sort_values('sim', ascending=False)
            
            if not org_pdf_tmp.empty:
                # 최고 순위 조직들의 정보 추출
                org_pdf_tmp['rank'] = org_pdf_tmp['sim'].rank(method='dense', ascending=False)
                org_pdf_tmp['org_cd_full'] = org_pdf_tmp.apply(lambda x: x['org_cd'] + x['sub_org_cd'], axis=1)
                org_info = org_pdf_tmp.query("rank == 1").groupby('org_nm')['org_cd_full'].apply(list).reset_index(name='org_cd').to_dict('records')
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
            logger.info(f"메시지 처리 시작: {mms_msg[:100]}...")
            
            # 입력 검증
            msg = validate_text_input(mms_msg)
            
            # 1단계: 엔티티 추출
            cand_item_list, extra_item_pdf = self._extract_entities(msg)
            
            # 2단계: 프로그램 분류
            pgm_info = self._classify_programs(msg)
            
            # 3단계: RAG 컨텍스트 구성
            rag_context = f"\n### 광고 분류 기준 정보 ###\n\t{pgm_info['pgm_cand_info']}" if self.num_cand_pgms > 0 else ""
            
            # 4단계: 제품 정보 준비 (NLP 모드용)
            product_element = None
            if len(cand_item_list) > 0:
                if self.product_info_extraction_mode == 'rag':
                    rag_context += f"\n\n### 후보 상품 이름 목록 ###\n\t{cand_item_list}"
                elif self.product_info_extraction_mode == 'nlp':
                    product_df = extra_item_pdf.rename(columns={'item_nm': 'name'}).query(
                        "not name in @self.stop_item_names"
                    )[['name']]
                    product_df['action'] = '고객에게 기대하는 행동: [구매, 가입, 사용, 방문, 참여, 코드입력, 쿠폰다운로드, 기타] 중에서 선택'
                    product_element = product_df.to_dict(orient='records') if product_df.shape[0] > 0 else None

            # 5단계: LLM 프롬프트 구성 및 실행
            prompt = self._build_extraction_prompt(msg, rag_context, product_element)
            result_json_text = self._safe_llm_invoke(prompt)
            
            # 6단계: JSON 파싱
            json_objects_list = extract_json_objects(result_json_text)
            if not json_objects_list:
                logger.warning("LLM이 유효한 JSON 객체를 반환하지 않았습니다")
                return self._create_fallback_result(msg)
            
            json_objects = json_objects_list[0]
            
            # 7단계: 엔티티 매칭 및 최종 결과 구성
            final_result = self._build_final_result(json_objects, msg, pgm_info)
            
            # 8단계: 결과 검증
            final_result = self._validate_extraction_result(final_result)
            
            logger.info("메시지 처리 완료")
            return final_result
            
        except Exception as e:
            logger.error(f"메시지 처리 실패: {e}")
            logger.error(traceback.format_exc())
            return self._create_fallback_result(mms_msg)

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
                final_result['product'] = self._map_products_with_similarity(similarities_fuzzy)
            else:
                # 유사도 결과가 없으면 LLM 결과 그대로 사용
                final_result['product'] = [
                    {
                        'item_name_in_msg': d.get('name', ''), 
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

    def _map_products_with_similarity(self, similarities_fuzzy: pd.DataFrame) -> List[Dict]:
        """유사도를 기반으로 상품 정보 매핑"""
        try:
            # 높은 유사도 아이템들 필터링
            high_sim_items = similarities_fuzzy.query('sim >= 1.5')['item_nm_alias'].unique()
            filtered_similarities = similarities_fuzzy[
                (similarities_fuzzy['item_nm_alias'].isin(high_sim_items)) &
                (~similarities_fuzzy['item_nm_alias'].str.contains('test', case=False)) &
                (~similarities_fuzzy['item_name_in_msg'].isin(self.stop_item_names))
            ]
            
            # 상품 정보와 매핑하여 최종 결과 생성
            product_tag = convert_df_to_json_list(
                self.item_pdf_all.merge(filtered_similarities, on=['item_nm_alias'])
            )
            
            return product_tag
            
        except Exception as e:
            logger.error(f"상품 정보 매핑 실패: {e}")
            return []

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


def main():
    """
    커맨드라인에서 실행할 때의 메인 함수
    다양한 옵션을 통해 추출기 설정을 변경할 수 있습니다.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='MMS 광고 텍스트 추출기 - 개선된 버전')
    parser.add_argument('--message', type=str, help='테스트할 메시지')
    parser.add_argument('--offer-data-source', choices=['local', 'db'], default='local',
                       help='데이터 소스 (local: CSV 파일, db: 데이터베이스)')
    parser.add_argument('--product-info-extraction-mode', choices=['nlp', 'llm', 'rag'], default='nlp',
                       help='상품 정보 추출 모드 (nlp: 형태소분석, llm: LLM 기반, rag: 검색증강생성)')
    parser.add_argument('--entity-matching-mode', choices=['logic', 'llm'], default='llm',
                       help='엔티티 매칭 모드 (logic: 로직 기반, llm: LLM 기반)')
    parser.add_argument('--llm-model', choices=['gemma', 'ax', 'claude', 'gemini', 'gpt'], default='ax',
                       help='사용할 LLM 모델 (gemma: Gemma, ax: ax, claude: Claude, gpt: GPT-4o)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO',
                       help='로그 레벨 설정')
    
    args = parser.parse_args()
    
    # 로그 레벨 설정
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    try:
        # 추출기 초기화
        logger.info("MMS 추출기 초기화 중...")
        extractor = MMSExtractor(
            offer_info_data_src=args.offer_data_source,
            product_info_extraction_mode=args.product_info_extraction_mode,
            entity_extraction_mode=args.entity_matching_mode,
            llm_model=args.llm_model
        )
        
        # 테스트 메시지 설정
        test_message = args.message if args.message else """
        [SKT] ZEM폰 포켓몬에디션3 안내
        (광고)[SKT] 우리 아이 첫 번째 스마트폰, ZEM 키즈폰__#04 고객님, 안녕하세요!
        우리 아이 스마트폰 고민 중이셨다면, 자녀 스마트폰 관리 앱 ZEM이 설치된 SKT만의 안전한 키즈폰,
        ZEM폰 포켓몬에디션3으로 우리 아이 취향을 저격해 보세요!
        신학기를 맞이하여 SK텔레콤 공식 인증 대리점에서 풍성한 혜택을 제공해 드리고 있습니다!
        ▶ 주요 기능
        1. 실시간 위치 조회
        2. 모르는 회선 자동 차단
        3. 스마트폰 사용 시간 제한
        4. IP68 방수 방진
        5. 수업 시간 자동 무음모드
        6. 유해 콘텐츠 차단
        ▶ 가까운 SK텔레콤 공식 인증 대리점 찾기
        http://t-mms.kr/t.do?m=#61&s=30684&a=&u=https://bit.ly/3yQF2hx
        ▶ 문의 : SKT 고객센터(1558, 무료)
        무료 수신거부 1504
        """
        
        # 메시지 처리 및 결과 출력
        logger.info("메시지 처리 시작")
        result = extractor.process_message(test_message)
        
        print("\n" + "="*50)
        print("🎯 최종 추출된 정보")
        print("="*50)
        print(json.dumps(result, indent=4, ensure_ascii=False))
        
        # 성능 요약 정보 출력
        print("\n" + "="*50)
        print("📊 처리 완료")
        print("="*50)
        print(f"✅ 제목: {result.get('title', 'N/A')}")
        print(f"✅ 목적: {len(result.get('purpose', []))}개")
        print(f"✅ 상품: {len(result.get('product', []))}개")
        print(f"✅ 채널: {len(result.get('channel', []))}개")
        print(f"✅ 프로그램: {len(result.get('pgm', []))}개")
        
    except Exception as e:
        logger.error(f"실행 실패: {e}")
        logger.error(traceback.format_exc())
        exit(1)


if __name__ == '__main__':
    main()