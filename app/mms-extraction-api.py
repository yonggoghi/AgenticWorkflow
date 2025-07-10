"""
MMS Extraction API
-----------------
이 API 서비스는 다양한 LLM 모델을 사용하여 MMS 메시지에서 구조화된 정보를 추출합니다.
Claude 3.5, Claude 3.7, ChatGPT, A.X 등 여러 모델을 지원합니다.

주요 기능:
- MMS 메시지에서 정보 추출
- 다양한 LLM 모델 지원
- Few-shot 학습 기능
- 제품 매칭 및 개체 인식
- 헬스 체크 및 추출을 위한 REST API 엔드포인트

의존성:
- Flask: 웹 프레임워크
- Pandas: 데이터 조작
- OpenAI/Anthropic: LLM 클라이언트
- RapidFuzz: 문자열 매칭
- LangChain: LLM 통합
- scikit-learn: TF-IDF 벡터화
"""

import os
import re
import json
import time
from typing import List, Dict, Any, Optional

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from openai import OpenAI
from rapidfuzz import fuzz, process
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
import pickle

# CORS 지원과 함께 Flask 앱 초기화
app = Flask(__name__)
CORS(app)  # 모든 라우트에 대해 CORS 활성화

# 텍스트 정리 및 전처리 유틸리티
def clean_text(text):
    """
    특수 문자를 제거하면서 중요한 구조적 요소를 보존하여 텍스트를 정리합니다.
    특히 한글 텍스트(한글)를 적절하게 처리합니다.
    
    Args:
        text (str): 정리할 입력 텍스트
        
    Returns:
        str: 구조가 보존된 정리된 텍스트
    """
    import re
    
    # 중요한 문자를 임시 토큰으로 대체하여 기본 구조를 보존합니다
    # 이 토큰들은 정리 과정에서 영향을 받지 않습니다
    
    # 1단계: JSON 구조 요소를 임시로 대체
    placeholders = {
        '"': "DQUOTE_TOKEN",
        "'": "SQUOTE_TOKEN",
        "{": "OCURLY_TOKEN",
        "}": "CCURLY_TOKEN",
        "[": "OSQUARE_TOKEN",
        "]": "CSQUARE_TOKEN",
        ":": "COLON_TOKEN",
        ",": "COMMA_TOKEN"
    }
    
    for char, placeholder in placeholders.items():
        text = text.replace(char, placeholder)
    
    # 2단계: 문제가 되는 문자 제거
    
    # 제어 문자 제거(의미 있는 뉴라인, 캐리지 리턴, 탭 제외)
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    # 모든 유형의 뉴라인을 \n으로 정규화
    text = re.sub(r'\r\n|\r', '\n', text)
    
    # 제로 너비 문자 및 기타 보이지 않는 유니코드 제거
    text = re.sub(r'[\u200B-\u200D\uFEFF\u00A0]', '', text)
    
    # 수정: 다른 유용한 문자 집합과 함께 한글 문자(한글) 유지
    # 유지할 문자 정의
    allowed_chars_pattern = r'[^\x00-\x7F\u0080-\u00FF\u0100-\u024F\u0370-\u03FF\u0400-\u04FF' + \
                           r'\u1100-\u11FF\u3130-\u318F\uA960-\uA97F\u3000-\u303F' + \
                           r'\uAC00-\uD7A3\uFF00-\uFFEF\u4E00-\u9FFF\n\r\t ]'
    text = re.sub(allowed_chars_pattern, '', text)
    
    # 3단계: 공백 정규화(의도적인 줄바꿈 보존)
    text = re.sub(r'[ \t]+', ' ', text)  # 여러 공백/탭을 단일 공백으로 변환
    
    # 먼저 모든 뉴라인이 표준화되었는지 확인
    text = re.sub(r'\r\n|\r', '\n', text)  # 모든 뉴라인 변형을 \n으로 변환
    
    # 그런 다음 여러 빈 줄을 최대 두 개로 정규화
    text = re.sub(r'\n\s*\n+', '\n\n', text)  # 여러 개의 뉴라인을 최대 두 개로 변환
    
    # 4단계: 원래 JSON 구조 요소 복원
    for char, placeholder in placeholders.items():
        text = text.replace(placeholder, char)
    
    # 5단계: 남아있을 수 있는 일반적인 JSON 구문 문제 수정
    # JSON에서 따옴표와 콜론 사이의 공백 수정
    text = re.sub(r'"\s+:', r'":', text)
    
    # 배열에서 후행 쉼표 수정
    text = re.sub(r',\s*]', r']', text)
    
    # 객체에서 후행 쉼표 수정
    text = re.sub(r',\s*}', r'}', text)
    
    return text

def remove_control_characters(text):
    """
    일반적인 공백을 유지하면서 텍스트에서 제어 문자를 제거합니다.
    
    Args:
        text (str): 입력 텍스트
        
    Returns:
        str: 제어 문자가 제거된 텍스트
    """
    if isinstance(text, str):
        # 일반적으로 사용되는 공백을 제외한 제어 문자 제거
        return re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)
    return text

def repair_json(broken_json):
    """
    잘못된 JSON 문자열에서 일반적인 JSON 구문 문제를 수정합니다.
    
    Args:
        broken_json (str): 잘못된 형식의 JSON 문자열
        
    Returns:
        str: 수정된 JSON 문자열
    """
    # 따옴표 없는 값 수정(예: NI00001863)
    json_str = re.sub(r':\s*([a-zA-Z0-9_]+)(\s*[,}])', r': "\1"\2', broken_json)
    
    # 따옴표 없는 키 수정
    json_str = re.sub(r'([{,])\s*([a-zA-Z0-9_]+):', r'\1 "\2":', json_str)
    
    # 후행 쉼표 수정
    json_str = re.sub(r',\s*}', '}', json_str)
    
    return json_str

def extract_store_code(text):
    """
    텍스트의 URL에서 매장 코드를 추출합니다.
    여러 URL 패턴 및 형식을 지원합니다.
    
    Args:
        text (str): 매장 URL이 포함된 텍스트
        
    Returns:
        str: 추출된 매장 코드 또는 찾지 못한 경우 None
    """
    # tworldfriends.co.kr URL에서 매장 코드를 찾는 패턴
    pattern1 = r'tworldfriends\.co\.kr/([D][0-9]{9})'
    # detail?code= 매개변수에서 매장 코드를 찾는 패턴
    pattern2 = r'code=([D][0-9\-]+)'
    
    # 첫 번째 패턴 시도
    match = re.search(pattern1, text)
    if match:
        return match.group(1)
    
    # 두 번째 패턴 시도
    match = re.search(pattern2, text)
    if match:
        return match.group(1)
    
    return None

def extract_phone_numbers(text):
    """
    다양한 한국 전화번호 형식을 사용하여 텍스트에서 전화번호를 추출합니다.
    
    Args:
        text (str): 전화번호가 포함된 텍스트
        
    Returns:
        list: 추출된 전화번호 목록
    """
    # 한국의 다양한 전화번호 형식에 대한 패턴
    patterns = [
        r'(\d{3,4}-\d{3,4}-\d{4})',  # 형식: XXX-XXXX-XXXX 또는 XXXX-XXXX-XXXX
        r'(0\d{1,2}-\d{3,4}-\d{4})',  # 형식: 0X-XXXX-XXXX 또는 0XX-XXXX-XXXX
        r'(0\d{3}-\d{3,4}-\d{4})',    # 형식: 0XXX-XXX-XXXX 또는 0XXX-XXXX-XXXX
        r'(\d{4}-\d{4})',             # 형식: XXXX-XXXX
        r'(1\d{3})',                  # 1588과 같은 단축 번호
        r'(0507-\d{4}-\d{4})'         # 형식: 0507-XXXX-XXXX (가상 번호)
    ]
    
    phone_numbers = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        phone_numbers.extend(matches)
    
    return phone_numbers

def replace_strings(text, replacements):
    """
    대체 사전에 따라 텍스트에서 지정된 문자열을 대체합니다.
    
    Args:
        text (str): 처리할 입력 텍스트
        replacements (dict): 이전 문자열을 새 문자열에 매핑하는 사전
        
    Returns:
        str: 대체가 적용된 텍스트
    """
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def extract_json_objects(text):
    """
    패턴 매칭 및 검증을 사용하여 텍스트에서 JSON 객체를 추출합니다.
    다양한 JSON 형식을 처리하기 위해 여러 파싱 방법을 시도합니다.
    
    Args:
        text (str): 잠재적 JSON 객체가 포함된 텍스트
        
    Returns:
        list: 추출 및 검증된 JSON 객체 목록
    """
    # JSON 구문을 적절하게 일치시키려는 보다 정교한 패턴
    pattern = r'(\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\})'
    
    result = []
    for match in re.finditer(pattern, text):
        potential_json = match.group(0)
        try:
            # 파싱 및 검증 시도
            json_obj = json.loads(repair_json(potential_json))
            result.append(json_obj)
        except json.JSONDecodeError:
            try:
                # 대체 방법 시도
                from ast import literal_eval
                json_obj = literal_eval(clean_ill_structured_json(repair_json(potential_json)))
                result.append(json_obj)
            except:
                # 유효한 JSON이 아님, 건너뛰기
                pass
    
    return result

def clean_ill_structured_json(text):
    """
    각 키-값 쌍을 처리하여 구조가 잘못된 JSON과 유사한 텍스트를 정리합니다.
    중첩된 따옴표 및 기타 일반적인 JSON 구문 문제를 처리합니다.
    
    Args:
        text (str): 구조가 잘못된 JSON과 유사한 텍스트
        
    Returns:
        str: 정리된 JSON과 유사한 텍스트
    """
    # 먼저 따옴표 외부의 쉼표로 텍스트를 분할합니다.
    parts = split_outside_quotes(text, delimiter=',')
    
    cleaned_parts = []
    for part in parts:
        # 따옴표 내부에 없는 첫 번째 콜론에서 키와 값으로 분할해 봅니다.
        key, value = split_key_value(part)
        key_clean = clean_segment(key)
        value_clean = clean_segment(value) if value.strip() != "" else ""
        if value_clean:
            cleaned_parts.append(f"{key_clean}: {value_clean}")
        else:
            cleaned_parts.append(key_clean)
    
    # 정리된 부분을 쉼표로 다시 결합
    return ', '.join(cleaned_parts)

def clean_segment(segment):
    """
    따옴표로 묶일 것으로 예상되는 텍스트 세그먼트를 정리합니다.
    따옴표 문자의 내부 발생을 제거합니다.
    
    Args:
        segment (str): 정리할 텍스트 세그먼트
        
    Returns:
        str: 정리된 세그먼트
    """
    segment = segment.strip()
    if len(segment) >= 2 and segment[0] in ['"', "'"] and segment[-1] == segment[0]:
        q = segment[0]
        # 따옴표 문자의 내부 발생을 제거합니다.
        inner = segment[1:-1].replace(q, '')
        return q + inner + q
    return segment

def split_key_value(text):
    """
    따옴표 외부의 첫 번째 콜론을 기준으로 텍스트를 키와 값으로 분할합니다.
    
    Args:
        text (str): 분할할 텍스트
        
    Returns:
        tuple: (key, value) 쌍
    """
    in_quote = False
    quote_char = ''
    for i, char in enumerate(text):
        if char in ['"', "'"]:
            # 따옴표 상태 전환(각 토큰에 대해 잘 형성된 시작/종료 따옴표를 가정)
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
    구분자가 따옴표 외부에 있을 때만 텍스트를 구분자로 분할합니다.
    
    Args:
        text (str): 분할할 텍스트
        delimiter (str): 구분자 문자(기본값: ',')
        
    Returns:
        list: 분할된 부분 목록
    """
    parts = []
    current = []
    in_quote = False
    quote_char = ''
    for char in text:
        if char in ['"', "'"]:
            # 따옴표를 만나면 상태를 전환합니다
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

def extract_unsubscribe_info(text):
    # 수신 거부 텍스트와 관련 번호 찾기
    unsubscribe_pattern = r'수신\s*거부\s*(\d+)'
    match = re.search(unsubscribe_pattern, text)
    if match:
        return {
            "type": "전화번호",
            "value": match.group(1),
            "action": "수신 거부"
        }
    return None

# 한국어 엔티티 매칭을 위한 클래스
class KoreanEntityMatcher:
    """
    텍스트에서 한국어 엔티티의 퍼지 매칭을 처리합니다.
    효율적인 매칭을 위해 n-그램 인덱싱 및 코사인 유사도를 사용합니다.
    
    기능:
    - N-그램 기반 인덱싱
    - 한국어 텍스트 세분화
    - 유사도 점수가 있는 퍼지 매칭
    - 중복 매치 해결
    """
    def __init__(self, min_similarity=75, ngram_size=2, min_entity_length=2):
        self.min_similarity = min_similarity
        self.ngram_size = ngram_size
        self.min_entity_length = min_entity_length
        self.entities = []
        self.entity_data = {}
        
    def build_from_list(self, entities):
        """엔티티 목록에서 엔티티 인덱스 구축"""
        self.entities = []
        self.entity_data = {}
        
        for i, entity in enumerate(entities):
            if isinstance(entity, tuple) and len(entity) == 2:
                entity_name, data = entity
                self.entities.append(entity_name)
                self.entity_data[entity_name] = data
            else:
                self.entities.append(entity)
                self.entity_data[entity] = {'id': i, 'entity': entity}
                
        # 후보 선택을 빠르게 하기 위한 n-그램 인덱스 생성
        self._build_ngram_index(n=self.ngram_size)
    
    def _build_ngram_index(self, n=2):
        """한국어 문자에 최적화된 n-그램 인덱스 구축"""
        self.ngram_index = {}
        
        for entity in self.entities:
            # min_entity_length보다 짧은 엔티티 건너뛰기
            if len(entity) < self.min_entity_length:
                continue
                
            # 엔티티에 대한 n-그램 생성
            entity_chars = list(entity)  # 한국어를 적절히 처리하기 위해 문자로 분할
            ngrams = []
            
            # 문자 수준 n-그램 생성(한국어에 더 적합)
            for i in range(len(entity_chars) - n + 1):
                ngram = ''.join(entity_chars[i:i+n])
                ngrams.append(ngram)
            
            # 각 n-그램에 대한 인덱스에 엔티티 추가
            for ngram in ngrams:
                if ngram not in self.ngram_index:
                    self.ngram_index[ngram] = set()
                self.ngram_index[ngram].add(entity)
    
    def _get_candidates(self, text, n=None):
        """n-그램 중복을 기반으로 후보 엔티티 가져오기(한국어에 최적화됨)"""
        if n is None:
            n = self.ngram_size
            
        text_chars = list(text)  # 한국어를 적절히 처리하기 위해 문자로 분할
        text_ngrams = set()
        
        # 문자 수준 n-그램 생성
        for i in range(len(text_chars) - n + 1):
            ngram = ''.join(text_chars[i:i+n])
            text_ngrams.add(ngram)
        
        candidates = set()
        for ngram in text_ngrams:
            if ngram in self.ngram_index:
                candidates.update(self.ngram_index[ngram])
        
        # 여러 n-그램 일치가 있는 후보 우선 순위 지정
        candidate_scores = {}
        for candidate in candidates:
            candidate_chars = list(candidate)
            candidate_ngrams = set()
            for i in range(len(candidate_chars) - n + 1):
                ngram = ''.join(candidate_chars[i:i+n])
                candidate_ngrams.add(ngram)
            
            overlap = len(candidate_ngrams.intersection(text_ngrams))
            candidate_scores[candidate] = overlap
        
        # n-그램 중복 점수로 정렬된 후보 반환
        return sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
    
    def find_entities(self, text, max_candidates_per_span=10):
        """퍼지 매칭을 사용하여 한국어 텍스트에서 엔티티 일치 찾기"""
        # 엔티티를 포함할 수 있는 범위 추출
        potential_spans = self._extract_korean_spans(text)
        matches = []
        
        for span_text, start, end in potential_spans:
            if len(span_text.strip()) < self.min_entity_length:  # min_entity_length보다 짧은 범위 건너뛰기
                continue
                
            # n-그램 중복을 기반으로 후보 엔티티 가져오기
            candidates = self._get_candidates(span_text)
            
            # n-그램 필터링을 통해 후보를 찾지 못한 경우 건너뛰기
            if not candidates:
                continue
            
            # 확인할 후보 수 제한
            top_candidates = [c[0] for c in candidates[:max_candidates_per_span]]
            
            # 최상의 퍼지 매치 찾기
            best_matches = process.extract(
                span_text, 
                top_candidates, 
                scorer=fuzz.ratio,  # token_sort_ratio보다 한국어에 더 적합
                score_cutoff=self.min_similarity,
                limit=3
            )
            
            for entity, score, _ in best_matches:
                matches.append({
                    'text': span_text,
                    'matched_entity': entity,
                    'score': score,
                    'start': start,
                    'end': end,
                    'data': self.entity_data.get(entity, {})
                })
        
        # 텍스트 위치별로 정렬
        matches.sort(key=lambda x: (x['start'], -x['score']))
        
        # 최상의 매치를 유지하여 중복 매치 처리
        final_matches = self._resolve_overlapping_matches(matches)
        
        return final_matches
    
    def _extract_korean_spans(self, text):
        """한국어 텍스트에서 엔티티일 수 있는 잠재적 텍스트 범위 추출"""
        spans = []
        min_len = self.min_entity_length
        
        # 1. 숫자와 영문 결합 패턴 (숫자 space 영문 패턴, e.g. "0 day")을 명시적으로 추출
        for match in re.finditer(r'\d+\s+[a-zA-Z]+', text):
            if len(match.group(0)) >= min_len:
                spans.append((match.group(0), match.start(), match.end()))
        
        # 2. 일반적인 구분자로 분리된 텍스트 조각 추출
        for span in re.split(r'[,\.!?;:"\'…\(\)\[\]\{\}\s_/]+', text):
            if span and len(span) >= min_len:
                span_pos = text.find(span)
                if span_pos != -1:
                    spans.append((span, span_pos, span_pos + len(span)))

        # 3. 더 명확한 구분자로 범위 추출 - 한국어에 적합한 구분자 세트 사용
        for span in re.split(r'[,\.!?;:"\'…\(\)\[\]\{\}\s_/]+', text):
            if span and len(span) >= min_len:
                span_pos = text.find(span)
                if span_pos != -1:
                    spans.append((span, span_pos, span_pos + len(span)))
        
        # 4. 명사구 추출 - 일반적인 한국어 구분자로 깨끗하게 분리
        for match in re.finditer(r'[가-힣a-zA-Z0-9]+(?:[^\s.,!?;:_/]\s*[가-힣a-zA-Z0-9]+)*', text):
            if len(match.group(0)) >= min_len:
                spans.append((match.group(0), match.start(), match.end()))
        
        # 5. 순수 한글만 추출 (특수문자 없음)
        for match in re.finditer(r'[가-힣]+', text):
            if len(match.group(0)) >= min_len:
                spans.append((match.group(0), match.start(), match.end()))
        
        # 6. 순수 영숫자만 추출
        for match in re.finditer(r'[a-zA-Z0-9]+', text):
            if len(match.group(0)) >= min_len:
                spans.append((match.group(0), match.start(), match.end()))
        
        return spans
    
    def _remove_duplicate_entities(self, matches):
        """Keep only one instance of each unique entity"""
        if not matches:
            return []
        
        # Dictionary to track highest-scoring match for each entity
        best_matches = {}
        
        for match in matches:
            entity_key = match['matched_entity']
            
            # If we haven't seen this entity before, or if this match has a higher score
            # than the previously saved match for this entity, save this one
            if (entity_key not in best_matches or 
                match['score'] > best_matches[entity_key]['score']):
                best_matches[entity_key] = match
        
        # Return the best matches sorted by start position
        return sorted(best_matches.values(), key=lambda x: x['start'])
    
    def _resolve_overlapping_matches(self, matches):
        """가장 높은 점수의 매치를 유지하여 중복 매치 해결"""
        if not matches:
            return []
        
        # 시작 위치별로 정렬한 다음 점수별로 정렬(내림차순)
        sorted_matches = sorted(matches, key=lambda x: (x['start'], -x['score']))
        
        final_matches = []
        
        for i, current_match in enumerate(sorted_matches):
            should_add = True
            current_start, current_end = current_match['start'], current_match['end']
            current_range = set(range(current_start, current_end))
            
            # 이미 추가된 매치와 비교하여 중복/포함 관계 확인
            for existing_match in final_matches:
                existing_start, existing_end = existing_match['start'], existing_match['end']
                existing_range = set(range(existing_start, existing_end))
                
                # 현재 매치가 기존 매치에 완전히 포함되는 경우 (하위 문자열)
                if current_range.issubset(existing_range):
                    should_add = False
                    break
                    
                # 현재 매치가 기존 매치를 완전히 포함하는 경우 (상위 문자열)
                if existing_range.issubset(current_range):
                    # 기존 매치보다 현재 매치가 더 넓은 범위를 커버하고, 
                    # 현재 매치의 점수가 기존 매치의 점수의 90% 이상인 경우
                    # 기존 매치를 제거하고 현재 매치를 추가
                    if current_match['score'] >= existing_match['score'] * 0.9:
                        final_matches.remove(existing_match)
                    else:
                        # 현재 매치의 점수가 낮으면 추가하지 않음
                        should_add = False
                        break
                
                # 부분 겹침이 있는 경우
                elif current_range.intersection(existing_range):
                    # 겹치는 부분의 비율 계산
                    overlap_ratio = len(current_range.intersection(existing_range)) / len(current_range)
                    
                    # 30% 이상 겹치면 현재 매치를 추가하지 않음
                    if overlap_ratio > 0.3:
                        should_add = False
                        break
            
            if should_add:
                final_matches.append(current_match)
        
        # 시작 위치별로 정렬
        final_matches.sort(key=lambda x: x['start'])
        
        return self._remove_duplicate_entities(final_matches)

def find_entities_in_text(text, entity_list, min_similarity=75, ngram_size=2, min_entity_length=2):
    """
    퍼지 매칭을 사용하여 텍스트에서 엔티티 일치 찾기.
    
    매개변수:
    -----------
    text : str
        엔티티를 검색할 텍스트
    entity_list : list
        일치시킬 엔티티 목록
    min_similarity : int, default=75
        퍼지 매칭을 위한 최소 유사도 점수(0-100)
    ngram_size : int, default=2
        인덱싱에 사용할 문자 n-그램 크기(한국어의 경우 2 또는 3 권장)
    min_entity_length : int, default=2
        고려할 엔티티의 최소 길이(문자)
        
    반환:
    --------
    list
        위치 및 메타데이터가 포함된 일치하는 엔티티 목록
    """
    matcher = KoreanEntityMatcher(
        min_similarity=min_similarity,
        ngram_size=ngram_size,
        min_entity_length=min_entity_length
    )
    matcher.build_from_list(entity_list)
    
    matches = matcher.find_entities(text)
    return matches

# 스키마 정의
SCHEMA_CLD = {
    "properties": {
        "message_info": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "광고 제목 - 원본 그대로 추출"},
                "main_theme": {"type": "string", "description": "광고의 핵심 주제와 가치 제안을 명확하게 설명"},
                "period": {"type": "string", "description": "이벤트/프로모션 기간 - 명시적으로 언급된 경우에만 추출, 없으면 \"상시\"로 설정"}
            }
        },
        "purpose": {"type": "array", "description": "광고의 주요 목적을 다음 중에서 선택(복수 가능): [상품 가입 유도, 대리점 방문 유도, 웹/앱 접속 유도, 이벤트 응모 유도, 할인 혜택 안내, 쿠폰 제공 안내, 경품 제공 안내, 기타 정보 제공]"},
        "target": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "segment": {"type": "string", "description": "타겟 고객층 - 광고에서 명시적으로 언급하거나 암시한 대상 고객"},
                    "characteristics": {"type": "string", "description": "해당 타겟의 특성과 니즈"},
                    "priority": {"type": "integer", "description": "광고 내용에서의 중요도에 따른 타겟팅 우선순위 (1이 최우선)"}
                }
            }
        },
        "product": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "광고하는 제품이나 서비스 이름 - \"상품 후보 정보\" 목록에서 일치하는 항목 모두 선택"},
                    "id": {"type": "string", "description": "제품/서비스 ID - \"상품 후보 정보\"에서 name에 일치하는 항목의 ID 선택"},
                    "category": {"type": "string", "description": "제품/서비스 카테고리 - \"상품 후보 정보\"에서 확인"},
                    "benefit": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "제공되는 구체적인 혜택 내용 상세 설명"},
                            "type": {"type": "string", "description": "혜택 유형: [할인, 쿠폰, 경품, 기타] 중에서 선택"}
                        }
                    },
                    "conditions": {"type": "string", "description": "혜택/구매를 받기 위한 구체적인 조건"},
                    "action": {"type": "string", "description": "고객에게 기대하는 행동: [구매, 가입, 사용, 방문, 참여, 작성, 기타] 중에서 선택"}
                }
            }
        },
        "channel": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "description": "채널 종류: [URL, 전화번호, 앱, 대리점] 중에서 선택"},
                    "value": {"type": "string", "description": "실제 URL, 전화번호, 앱 이름, 대리점 이름 등 구체적 정보"},
                    "action": {"type": "string", "description": "채널 목적: [가입, 추가 정보, 문의, 수신, 수신 거부] 중에서 선택"},
                    "primary": {"type": "boolean", "description": "주요 채널 여부 - 광고의 핵심 연락 채널이면 true"},
                    "availability": {"type": "string", "description": "채널 이용 가능 시간/조건 - 언급이 없으면 \"상시\"로 설정"},
                    "benefit": {"type": "string", "description": "해당 채널 이용 시 특별 혜택"},
                    "store_code": {"type": "string", "description": "매장 코드 - tworldfriends.co.kr URL에서 D+숫자 9자리(D[0-9]{9}) 패턴의 코드 추출하여 대리점 채널에 설정"}
                }
            }
        },
        "metadata": {
            "type": "object",
            "properties": {
                "message_type": {"type": "string", "description": "메시지 유형 - 항상 \"광고\"로 설정"},
                "target_response": {"type": "string", "description": "광고주가 고객으로부터 기대하는 반응 요약"},
                "success_metrics": {"type": "array", "description": "이 캠페인의 성공을 측정하기 위한 지표들 추출"}
            }
        }
    },
    "required": ["purpose", "target", "product", "channel", "metadata"],
    "objectType": "object"
}

SCHEMA_AX = {
    'properties': {
        'message_info': {
            'type': 'object', 
            'properties': {
                'title': {'type': 'string', 'description': '광고 제목 - 원본 그대로 추출'},
                'main_theme': {'type': 'string', 'description': '광고의 핵심 주제와 가치 제안을 명확하게 설명'},
                'period': {'type': 'string', 'description': '이벤트/프로모션 기간 - 명시적으로 언급된 경우에만 추출, 없으면 "상시"로 설정'}
            }
        },
        'purpose': {
            'type': 'array', 
            'description': '광고의 주요 목적을 다음 중에서 선택(복수 가능): [상품 가입 유도, 대리점 방문 유도, 웹/앱 접속 유도, 이벤트 응모 유도, 할인 혜택 안내, 쿠폰 제공 안내, 경품 제공 안내, 기타 정보 제공]'
        },
        'target': {
            'type': 'array', 
            'items': {
                'type': 'object', 
                'properties': {
                    'segment': {'type': 'string', 'description': '타겟 고객층 - 광고에서 명시적으로 언급하거나 암시한 대상 고객'},
                    'characteristics': {'type': 'string', 'description': '해당 타겟의 특성과 니즈'},
                    'priority': {'type': 'integer', 'description': '광고 내용에서의 중요도에 따른 타겟팅 우선순위 (1이 최우선)'}
                }
            }
        },
        'product': {
            'type': 'array', 
            'items': {
                'type': 'object', 
                'properties': {
                    'name': {'type': 'string', 'description': '광고하는 제품이나 서비스 이름 - "상품 후보 정보" 목록에서 일치하는 항목 모두 선택'},
                    'id': {'type': 'string', 'description': '제품/서비스 ID - "상품 후보 정보"에서 name에 일치하는 항목의 ID 선택'},
                    'category': {'type': 'string', 'description': '제품/서비스 카테고리 - "상품 후보 정보"에서 확인'},
                    'benefit': {
                        'type': 'object', 
                        'properties': {
                            'name': {'type': 'string', 'description': '제공되는 구체적인 혜택 내용 상세 설명'},
                            'type': {'type': 'string', 'description': '혜택 유형: [할인, 쿠폰, 경품, 기타] 중에서 선택'}
                        }
                    },
                    'conditions': {'type': 'string', 'description': '혜택/구매를 받기 위한 구체적인 조건'},
                    'action': {'type': 'string', 'description': '고객에게 기대하는 행동: [구매, 가입, 사용, 방문, 참여, 작성, 기타] 중에서 선택'}
                }
            }
        },
        'channel': {
            'type': 'array', 
            'items': {
                'type': 'object', 
                'properties': {
                    'type': {'type': 'string', 'description': '채널 종류: [URL, 전화번호, 앱, 대리점] 중에서 선택'},
                    'value': {'type': 'string', 'description': '실제 URL, 전화번호, 앱 이름, 대리점 이름 등 구체적 정보'},
                    'action': {'type': 'string', 'description': '채널 목적: [가입, 추가 정보, 문의, 수신, 수신 거부] 중에서 선택'},
                    'primary': {'type': 'boolean', 'description': '주요 채널 여부 - 광고의 핵심 연락 채널이면 true'},
                    'availability': {'type': 'string', 'description': '채널 이용 가능 시간/조건 - 언급이 없으면 "상시"로 설정'},
                    'benefit': {'type': 'string', 'description': '해당 채널 이용 시 특별 혜택'},
                    'store_code': {'type': 'string', 'description': "매장 코드 - tworldfriends.co.kr URL에서 D+숫자 9자리(D[0-9]{9}) 패턴의 코드 추출하여 대리점 채널에 설정"}
                }
            }
        },
        'metadata': {
            'type': 'object', 
            'properties': {
                'message_type': {'type': 'string', 'description': '메시지 유형 - 항상 "광고"로 설정'},
                'target_response': {'type': 'string', 'description': '광고주가 고객으로부터 기대하는 반응 요약'},
                'success_metrics': {'type': 'array', 'description': '이 캠페인의 성공을 측정하기 위한 지표들 추출'}
            }
        }
    }, 
    'required': ['purpose', 'target', 'product', 'channel', 'metadata'], 
    'objectType': 'object'
}

EXTRACTION_GUIDE = """
### 분석 과정 ###
• 먼저 광고 메시지를 전체적으로 읽고 주요 내용, 목적, 혜택, 타겟을 파악하세요.
• "상품 후보 정보" 목록을 확인하여 광고에 언급된 제품/서비스와 일치하는 항목을 모두 선택하세요.
• 각 제품별 혜택과 조건을 명확하게 구분하여 추출하세요.
• 모든 채널 정보(URL, 전화번호, 대리점 등)를 빠짐없이 추출하고, tworldfriends.co.kr URL에서 매장 코드를 반드시 추출하세요.
• 각 필드에 적합한 값을 선택할 때 광고 내용에 명시적으로 언급된 정보를 우선하고, 없는 경우에만 적절히 추론하세요.
• 결과 JSON은 완전하고 정확해야 하며, schema에 정의된 모든 필수 필드를 포함해야 합니다.
• 학습용 예시를 참고 정보로 사용하지 마세요.
"""

def extract_json_objects(text):
    """
    패턴 매칭 및 검증을 사용하여 텍스트에서 JSON 객체를 추출합니다.
    다양한 JSON 형식을 처리하기 위해 여러 파싱 방법을 시도합니다.
    
    Args:
        text (str): 잠재적 JSON 객체가 포함된 텍스트
        
    Returns:
        list: 추출 및 검증된 JSON 객체 목록
    """
    # 먼저 정규식을 사용하여 JSON 추출 시도
    pattern = r'(\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\})'
    
    result = []
    for match in re.finditer(pattern, text):
        potential_json = match.group(0)
        try:
            # 파싱 및 검증 시도
            json_obj = json.loads(repair_json(potential_json))
            result.append(json_obj)
        except json.JSONDecodeError:
            try:
                # 대체 방법 시도
                from ast import literal_eval
                json_obj = literal_eval(clean_ill_structured_json(repair_json(potential_json)))
                result.append(json_obj)
            except:
                # 유효한 JSON이 아님, 건너뛰기
                pass
    
    # 정규식을 사용하여 JSON 객체를 찾지 못한 경우 더 포괄적인 접근 방식 시도
    if not result:
        # 텍스트에서 JSON과 유사한 구조를 찾으려고 시도
        try:
            # 중괄호 사이의 텍스트 찾기
            json_match = re.search(r'({[\s\S]*})', text)
            if json_match:
                json_str = json_match.group(1)
                try:
                    result.append(json.loads(repair_json(json_str)))
                except:
                    pass
        except:
            pass
    
    return result


# 다양한 모델에 대한 LLM 클라이언트 클래스
class LLMClient:
    """
    LLM 클라이언트를 위한 기본 클래스.
    정보 추출을 위한 공통 인터페이스를 정의합니다.
    """
    def __init__(self, api_key=None, model=None):
        self.api_key = api_key
        self.model = model
        
    def extract_info(self, text, schema, rag_context=None):
        """LLM을 사용하여 텍스트에서 정보 추출"""
        raise NotImplementedError("하위 클래스는 이 메서드를 구현해야 합니다")

class Claude35Client(LLMClient):
    """
    A15T 플랫폼의 Claude 3.5 Sonnet 클라이언트.
    A15T 플랫폼용 사용자 지정 기본 URL이 있는 OpenAI 클라이언트를 사용합니다.
    """
    def __init__(self, api_key=None, model=None, api_base_url=None):
        # 기본값이 있는 환경 변수에서 구성 가져오기
        self.model = model or os.environ.get('MODEL_NAME_CLAUDE35', 'skt/claude-3-5-sonnet-20241022')
        self.api_base_url = api_base_url or os.environ.get('API_BASE_URL_CLAUDE35', 'https://api.platform.a15t.com/v1')
        
        super().__init__(api_key, self.model)

        self.llm = ChatOpenAI(
            temperature=0,  
            openai_api_key=api_key, 
            openai_api_base=self.api_base_url, 
            model=self.model,
            max_tokens=3000
        )
        
    def extract_info(self, text, schema, rag_context=None, few_shot_examples=None):
        """개선된 프롬프팅으로 Claude 3.5를 사용하여 정보 추출"""
        system_message = f"""당신은 SKT 캠페인 메시지에서 정확한 정보를 추출하는 전문가입니다. 다음과 같은 schema에 따라 광고 메시지를 분석하여 완전하고 정확한 JSON 객체를 생성해 주세요:

{schema}

{EXTRACTION_GUIDE}

응답은 설명 없이 순수한 JSON 형식으로만 제공하세요. 응답의 시작과 끝은 {{}}여야 합니다. 어떠한 추가 텍스트나 설명도 포함하지 마세요."""
        
        user_content = text
        if rag_context:
            user_content += f"\n\n=====관련 참고 정보=====\n{rag_context}"
        
        if few_shot_examples:
            user_content += f"\n\n=====학습용 예시=====\n{few_shot_examples}"
        
        try:
            response = self.llm.invoke([
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_content}
            ])
            
            content = response.content
            result = extract_json_objects(content)
            
            if result:
                return result[0]
            else:
                return {"error": "No valid JSON found in response", "raw_response": content}
                
        except Exception as e:
            print(f"Error in Claude 3.5 extraction: {str(e)}")
            return {"error": str(e)}

class Claude37Client(LLMClient):
    """
    Anthropic API를 사용하는 Claude 3.7 Sonnet 클라이언트.
    Anthropic API와 직접 통합됩니다.
    """
    def __init__(self, api_key=None, model=None, api_base_url=None):
        # 기본값이 있는 환경 변수에서 구성 가져오기
        self.model = model or os.environ.get('MODEL_NAME_CLAUDE37', 'claude-3-7-sonnet-20250219')
        
        super().__init__(api_key, self.model)

        self.llm = ChatAnthropic(
            temperature=0,  
            anthropic_api_key=api_key, 
            model=self.model,
            max_tokens=3000
        )
        
    def extract_info(self, text, schema, rag_context=None):
        """개선된 프롬프팅으로 Claude 3.7을 사용하여 정보 추출"""
        system_message = f"""당신은 SKT 캠페인 메시지에서 정확한 정보를 추출하는 전문가입니다. 다음과 같은 schema에 따라 광고 메시지를 분석하여 완전하고 정확한 JSON 객체를 생성해 주세요:

{schema}

{EXTRACTION_GUIDE}

응답은 설명 없이 순수한 JSON 형식으로만 제공하세요. 응답의 시작과 끝은 {{}}여야 합니다. 어떠한 추가 텍스트나 설명도 포함하지 마세요."""
        
        user_content = text
        if rag_context:
            user_content += f"\n\n=====관련 참고 정보=====\n{rag_context}"
        
        try:
            response = self.llm.invoke([
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_content}
            ])
            
            content = response.content
            result = extract_json_objects(content)
            
            if result:
                return result[0]
            else:
                return {"error": "No valid JSON found in response", "raw_response": content}
                
        except Exception as e:
            print(f"Error in Claude 3.7 extraction: {str(e)}")
            return {"error": str(e)}

class ChatGPTClient(LLMClient):
    """
    ChatGPT 모델용 클라이언트.
    정보 추출을 위해 OpenAI의 API를 사용합니다.
    """
    def __init__(self, api_key=None, model=None, api_base_url=None):
        # 기본값이 있는 환경 변수에서 구성 가져오기
        self.model = model or os.environ.get('MODEL_NAME_CHATGPT', 'gpt-4o')
        
        super().__init__(api_key, self.model)
        
        self.llm = ChatOpenAI(
            temperature=0,
            openai_api_key=api_key,
            model=self.model,
            max_tokens=3000
        )
        
    def extract_info(self, text, schema, rag_context=None):
        """개선된 프롬프팅으로 ChatGPT를 사용하여 정보 추출"""
        system_message = f"""당신은 SKT 캠페인 메시지에서 정확한 정보를 추출하는 전문가입니다. 다음과 같은 schema에 따라 광고 메시지를 분석하여 완전하고 정확한 JSON 객체를 생성해 주세요:

{schema}

{EXTRACTION_GUIDE}

응답은 설명 없이 순수한 JSON 형식으로만 제공하세요. 응답의 시작과 끝은 {{}}여야 합니다. 어떠한 추가 텍스트나 설명도 포함하지 마세요."""
        
        user_content = text
        if rag_context:
            user_content += f"\n\n=====관련 참고 정보=====\n{rag_context}"
        
        try:
            response = self.llm.invoke([
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_content}
            ])
            
            content = response.content
            result = extract_json_objects(content)
            
            if result:
                return result[0]
            else:
                return {"error": "No valid JSON found in response", "raw_response": content}
                
        except Exception as e:
            print(f"Error in ChatGPT extraction: {str(e)}")
            return {"error": str(e)}
        
class AXClient(LLMClient):
    """
    SKT의 A.X 모델용 클라이언트.
    A.X 모델 API에 대한 개선된 구조로 사용자 정의 구현.
    """
    def __init__(self, api_key=None, model=None, api_base_url=None):
        # 기본값이 있는 환경 변수에서 구성 가져오기
        self.model = model or os.environ.get('MODEL_NAME_AX', 'skt/a.x-3-lg')
        self.api_base_url = api_base_url or os.environ.get('API_BASE_URL_AX', 'https://api.platform.a15t.com/v1')
        
        super().__init__(api_key, self.model)
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.api_base_url
        )
        
    def extract_info(self, text, schema, rag_context=None, few_shot_examples=None):
        """개선된 프롬프팅 및 구조를 사용하여 A.X 모델로 정보 추출"""
        # 명확한 JSON 출력 요구 사항이 있는 시스템 메시지 생성
        system_message = f"""당신은 SKT 캠페인 메시지에서 정확한 정보를 추출하는 전문가입니다. 아래 schema에 따라 광고 메시지를 분석하여 완전하고 정확한 JSON 객체를 생성해 주세요:

{json.dumps(schema, indent=2, ensure_ascii=False)}

{EXTRACTION_GUIDE}

중요: 응답은 설명이나 추가 텍스트 없이 순수한 JSON 형식으로만 제공하세요. 응답은 '{{'로 시작하고 '}}'로 끝나야 합니다.
"""

        # 구조화된 메시지 내용 생성
        user_message = f"""=====분석 대상 광고 메세지=====
{text}"""

        if rag_context:
            user_message += f"\n\n=====관련 참고 정보=====\n{rag_context}"
            
        if few_shot_examples:
            user_message += f"\n\n=====학습용 예시=====\n{few_shot_examples}"
        
        try:
            print(f"Using A.X model: {self.model}")  # 디버그 로그
            
            # 명시적 JSON 응답 형식을 사용한 OpenAI의 ChatCompletion
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}  # 명시적으로 JSON 형식 요청
            )
            
            content = response.choices[0].message.content
            result = extract_json_objects(content)
            
            if result:
                return result[0]
            else:
                return {"error": "No valid JSON found in response", "raw_response": content}
                
        except Exception as e:
            print(f"Error in A.X extraction: {str(e)}")
            print(f"Model configuration: {self.model}")  # 디버그 로그
            print(f"API Base URL: {self.api_base_url}")  # 디버그 로그
            return {"error": str(e), "model": self.model, "api_base_url": self.api_base_url}

# 데이터 관리를 위한 데이터베이스 클래스
class ItemDatabase:
    """
    매칭을 위한 제품/항목 데이터베이스를 관리합니다.
    제품 정보의 로딩 및 쿼리를 처리합니다.
    
    기능:
    - CSV/Excel 데이터 로딩
    - 유사도 점수가 있는 제품 매칭
    - 엔티티 목록 관리
    - 텍스트 정리 및 전처리
    """
    def __init__(self, item_data_path=None):
        self.item_data_path = item_data_path
        self.item_df = None
        self.entity_list = []
        self.load_data()
        
    def load_data(self):
        """파일에서 항목 데이터 로드"""
        if self.item_data_path:
            try:
                if self.item_data_path.endswith('.csv'):
                    self.item_df = pd.read_csv(self.item_data_path)
                elif self.item_data_path.endswith('.xlsx'):
                    self.item_df = pd.read_excel(self.item_data_path, engine="openpyxl")
                else:
                    raise ValueError("지원되지 않는 파일 형식입니다. CSV 또는 XLSX를 사용하세요.")
                
                # 데이터프레임 처리
                self.process_data()
            except Exception as e:
                print(f"항목 데이터 로딩 중 오류: {str(e)}")
                # 빈 데이터프레임으로 초기화
                self.item_df = pd.DataFrame(columns=['item_id', 'item_nm', 'item_desc', 'item_cate_ax', 'create_dt'])
        else:
            # 빈 데이터프레임으로 초기화
            self.item_df = pd.DataFrame(columns=['item_id', 'item_nm', 'item_desc', 'item_cate_ax', 'create_dt'])
    
    def process_data(self):
        """로드된 데이터프레임 처리"""
        # 중복 제거
        self.item_df = self.item_df.drop_duplicates(['item_nm', 'item_desc']).copy()
        
        # 결합된 텍스트 필드 생성
        self.item_df['item_item'] = self.item_df['item_nm'] + "\n" + self.item_df['item_desc']
        
        # 필드 정리
        self.item_df['item_nm_cl'] = self.item_df['item_nm'].apply(clean_text)
        self.item_df['item_desc_cl'] = self.item_df['item_desc'].fillna('').astype(str).apply(clean_text)
        self.item_df['item_item_cl'] = self.item_df['item_nm_cl'] + "\n" + self.item_df['item_desc_cl']
        
        # 매칭을 위한 엔티티 목록 생성
        self.entity_list = []
        for row in self.item_df.to_dict('records'):
            self.entity_list.append((row['item_nm'], {
                'item_id': row['item_id'],
                'category': row['item_cate_ax'], 
                'description': row['item_desc'], 
                'create_dt': row['create_dt']
            }))
    
    def find_matching_entities(self, text, min_similarity=70, ngram_size=3, min_entity_length=3):
        """데이터베이스의 항목과 일치하는 텍스트에서 엔티티 찾기"""
        matches = find_entities_in_text(
            text, 
            self.entity_list, 
            min_similarity=min_similarity,
            ngram_size=ngram_size,
            min_entity_length=min_entity_length
        )
        return matches
    
    def get_matching_products_context(self, text):
        """일치하는 엔티티에서 제품 컨텍스트 정보 가져오기"""
        matches = self.find_matching_entities(text)
        
        if not matches:
            return "일치하는 제품을 찾을 수 없습니다."
            
        mdf = pd.DataFrame(matches)
        mdf['item_id'] = mdf['data'].apply(lambda x: x['item_id'])
        mdf['category'] = mdf['data'].apply(lambda x: x['category'])
        
        product_dict = mdf.rename(columns={
        'text': 'item_name_in_message',
        'matched_entity': 'item_name_in_voca'
        })[['item_name_in_message', 'item_name_in_voca', 'item_id', 'category']].drop_duplicates().to_dict(orient='records')
        
        # ensure_ascii=False를 사용한 json.dumps
        import json
        product_info = json.dumps(product_dict, ensure_ascii=False)
    
        return product_info

class FewShotDatabase:
    """
    LLM 학습을 위한 few-shot 예제를 관리합니다.
    예제 유사성 매칭을 위해 TF-IDF 벡터화를 사용합니다.
    
    기능:
    - 다양한 파일 형식에서 예제 로딩
    - TF-IDF 기반 유사성 검색
    - 동적 예제 선택
    - 구성 가능한 결과 열
    """
    def __init__(self, few_shot_data_path=None):
        self.few_shot_data_path = few_shot_data_path
        self.examples_df = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.result_column = os.environ.get('FEW_SHOT_RESULT_COLUMN', 'res_cld37')
        self.load_data()
        
    def load_data(self):
        """파일에서 few-shot 예제 로드"""
        if self.few_shot_data_path:
            try:
                print(f"few-shot 데이터 로드 위치: {self.few_shot_data_path}")
                
                # 상대 경로를 절대 경로로 변환
                if not os.path.isabs(self.few_shot_data_path):
                    self.few_shot_data_path = os.path.abspath(self.few_shot_data_path)
                
                # 파일이 존재하는지 확인
                if not os.path.exists(self.few_shot_data_path):
                    print(f"오류: Few-shot 데이터 파일을 {self.few_shot_data_path}에서 찾을 수 없습니다")
                    self.examples_df = pd.DataFrame()
                    return
                
                # 파일 확장자에 따라 데이터 로드
                if self.few_shot_data_path.endswith('.csv'):
                    self.examples_df = pd.read_csv(self.few_shot_data_path)
                elif self.few_shot_data_path.endswith('.xlsx'):
                    self.examples_df = pd.read_excel(self.few_shot_data_path, engine="openpyxl")
                elif self.few_shot_data_path.endswith('.pkl'):
                    with open(self.few_shot_data_path, 'rb') as f:
                        self.examples_df = pickle.load(f)
                else:
                    raise ValueError("지원되지 않는 파일 형식입니다. CSV, XLSX 또는 PKL을 사용하세요.")
                
                print(f"{len(self.examples_df)}개의 few-shot 예제 로드됨")
                print(f"데이터프레임의 열: {self.examples_df.columns.tolist()}")
                
                # 데이터가 있는 경우 TF-IDF 벡터라이저 초기화
                if not self.examples_df.empty:
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    self.tfidf_vectorizer = TfidfVectorizer()
                    # NaN 값을 빈 문자열로 채우기
                    self.examples_df['msg_head'] = self.examples_df['msg_head'].fillna('')
                    self.examples_df['msg_body'] = self.examples_df['msg_body'].fillna('')
                    combined_text = self.examples_df['msg_head'] + " " + self.examples_df['msg_body']
                    print(f"{len(combined_text)}개 예제에 대한 TF-IDF 행렬 생성")
                    self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(combined_text)
                    print("TF-IDF 행렬이 성공적으로 생성되었습니다")
                else:
                    print("경고: Few-shot 예제 데이터프레임이 비어 있습니다")
            except Exception as e:
                print(f"Few-shot 데이터 로드 중 오류: {str(e)}")
                import traceback
                traceback.print_exc()
                self.examples_df = pd.DataFrame()
        else:
            print("경고: Few-shot 데이터 경로가 제공되지 않았습니다")
            self.examples_df = pd.DataFrame()
            
    def get_similar_examples(self, text, n=3, result_column=None):
        """텍스트 유사성에 기반한 n개의 유사한 few-shot 예제를 가져옵니다. 개선된 형식으로 제공됩니다."""
        if result_column is None:
            result_column = self.result_column
            
        if self.examples_df.empty or self.tfidf_vectorizer is None:
            print("경고: Few-shot 예제를 사용할 수 없거나 TF-IDF 벡터라이저가 초기화되지 않았습니다")
            return []
            
        try:
            # 입력 텍스트 변환
            text_tfidf = self.tfidf_vectorizer.transform([text])
            
            # 유사성 계산
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(text_tfidf, self.tfidf_matrix)[0]
            
            # 상위 n개 예제 가져오기
            few_shot_temp = self.examples_df.copy()
            few_shot_temp['sim'] = similarities
            few_shot_temp['rank'] = few_shot_temp["sim"].rank(method='min', ascending=False)
            
            # 1위부터 n위까지의 예제 가져오기(첫 번째 포함)
            top_examples = few_shot_temp[
                (few_shot_temp['rank'] >= 1) & 
                (few_shot_temp['rank'] <= n)
            ].sort_values("rank")
            
            print(f"{len(top_examples)}개의 유사한 예제를 찾았습니다")
            
            # 더 나은 JSON 예제를 위한 개선된 구조로 예제 형식 지정
            few_shot_exm = []
            for idx, r in top_examples.iterrows():
                # 예제 JSON을 가져와 적절히 형식 지정
                example_json = r[result_column]
                
                # 예제에 설명 텍스트가 포함된 경우 JSON만 추출 시도
                if not (isinstance(example_json, str) and example_json.startswith('{') and example_json.endswith('}')):
                    if isinstance(example_json, str):
                        json_match = re.search(r'({[\s\S]*})', example_json)
                        if json_match:
                            example_json = json_match.group(1)
                        try:
                            # JSON 검증 및 형식 지정
                            example_json = json.dumps(json.loads(example_json), ensure_ascii=False)
                        except:
                            # 파싱할 수 없는 경우 있는 그대로 사용
                            pass
                            
                few_shot_exm.append(f"""
    [학습용 광고 메세지_{idx}]
    광고 제목:{r['msg_head']}
    광고 내용:{r['msg_body']}
    [학습용 정답 결과_{idx}]
    {example_json}""")
            
            return "\n".join(few_shot_exm)
        except Exception as e:
            print(f"유사한 예제를 가져오는 중 오류: {str(e)}")
            import traceback
            traceback.print_exc()
            return []


# 메인 서비스 클래스
class MMSExtractionService:
    """
    MMS 정보 추출을 위한 핵심 서비스.
    LLM 클라이언트와 데이터베이스 사이를 조정합니다.
    
    기능:
    - 다중 모델 지원
    - RAG 컨텍스트 생성
    - Few-shot 학습 통합
    - 제품 매칭
    - 구조화된 정보 추출
    """
    def __init__(self, anthropic_api_key=None, ax_api_key=None, openai_api_key=None, item_data_path=None, few_shot_data_path=None):
        # 상대 경로를 절대 경로로 변환
        if item_data_path and not os.path.isabs(item_data_path):
            item_data_path = os.path.abspath(item_data_path)
        if few_shot_data_path and not os.path.isabs(few_shot_data_path):
            few_shot_data_path = os.path.abspath(few_shot_data_path)
            
        print(f"서비스 초기화:")
        print(f"항목 데이터 경로: {item_data_path}")
        print(f"Few-shot 데이터 경로: {few_shot_data_path}")
        
        # 데이터베이스 초기화
        self.item_db = ItemDatabase(item_data_path)
        self.few_shot_db = FewShotDatabase(few_shot_data_path)
        
        # 환경에서 구성과 함께 LLM 클라이언트 초기화
        self.claude35_client = Claude35Client(
            api_key=anthropic_api_key,  # Claude 3.5용 A15T 플랫폼 키
            model=os.environ.get('MODEL_NAME_CLAUDE35'),
            api_base_url=os.environ.get('API_BASE_URL_CLAUDE35')
        )
        self.claude37_client = Claude37Client(
            api_key=os.environ.get('ANTHROPIC_API_KEY_CLAUDE37'),  # Claude 3.7용 Anthropic API 키
            model=os.environ.get('MODEL_NAME_CLAUDE37')
        )
        self.chatgpt_client = ChatGPTClient(
            api_key=openai_api_key,
            model=os.environ.get('MODEL_NAME_CHATGPT')
        )
        self.ax_client = AXClient(
            api_key=ax_api_key,
            model=os.environ.get('MODEL_NAME_AX'),
            api_base_url=os.environ.get('API_BASE_URL_AX')
        )
    
    def create_rag_context(self, text):
        """구조화된 보조 정보로 향상된 RAG 컨텍스트 생성"""
        store_code = extract_store_code(text)
        phone_numbers = extract_phone_numbers(text)
        unsubscribe_info = extract_unsubscribe_info(text)
        
        # 제품 정보 가져오기
        product_info = self.item_db.get_matching_products_context(text)
        
        # 구조화된 컨텍스트 섹션 생성
        context_parts = []
        
        # 제품 정보 추가
        if product_info:
            context_parts.append(
                f"\t상품 후보 정보: {product_info}"
            )
        
        # 매장 코드가 발견된 경우 추가
        if store_code:
            context_parts.append(
                f"\t매장 코드 정보: '{store_code}'. 이 코드는 type='대리점'인 channel 객체의 store_code 필드에 설정해주세요."
            )
        
        # 수신 거부 정보가 발견된 경우 추가
        if unsubscribe_info:
            context_parts.append(
                f"\t수신 거부 정보: {unsubscribe_info}. 이 정보를 type='전화번호', action='수신 거부' channel 객체로 추가해주세요."
            )
        
        # 전화번호가 발견되고 이미 처리되지 않은 경우 추가
        if phone_numbers and len(phone_numbers) > 0:
            phone_str = ', '.join(phone_numbers)
            context_parts.append(
                f"\t추출된 전화번호 목록: {phone_str}. 각 번호의 용도에 맞게 적절한 channel 객체로 추가해주세요."
            )
        
        return "\n".join(context_parts)
    
    def extract_info(self, message_head, message_body, model_type="claude35"):
        """지정된 모델을 사용하여 메시지에서 정보 추출"""
        # 전처리
        special_symbols_to_remove = {''':'', ''':'', ':':'', "'": '', '"': '', '_': '\n'}
        head = replace_strings(message_head, special_symbols_to_remove)
        body = replace_strings(message_body, special_symbols_to_remove)
        
        # 메시지 텍스트 생성
        message_text = f"""
        광고 제목: {head}
        광고 내용: {body}
        """
        
        # RAG 컨텍스트 생성
        rag_context = self.create_rag_context(message_text)
        
        # A.X 모델을 사용하는 경우 few-shot 예제 가져오기
        few_shot_examples = None
        if model_type.lower() in ["ax", "a.x", "skt", "cld", "claude35","claude"]:
            few_shot_examples = self.few_shot_db.get_similar_examples(message_text)
        
        # 지정된 모델을 사용하여 정보 추출
        if model_type.lower() in ["cld","claude","cld35", "claude35"]:
            schema = SCHEMA_CLD
            result = self.claude35_client.extract_info(message_text, schema, rag_context, few_shot_examples)
        elif model_type.lower() in ["cld37", "claude37"]:
            schema = SCHEMA_CLD
            result = self.claude37_client.extract_info(message_text, schema, rag_context)
        elif model_type.lower() in ["gpt", "chatgpt"]:
            schema = SCHEMA_CLD
            result = self.chatgpt_client.extract_info(message_text, schema, rag_context)
        elif model_type.lower() in ["ax", "a.x", "skt"]:
            schema = SCHEMA_AX
            result = self.ax_client.extract_info(
                message_text, 
                schema, 
                rag_context, 
                few_shot_examples=few_shot_examples
            )
        else:
            return {"error": f"지원되지 않는 모델 유형: {model_type}"}
        
        return {
            "result": result,
            "processed_text": message_text,
            "rag_context": rag_context,
            "few_shot_examples": few_shot_examples
        }

# 서비스 초기화(API 라우트에서 수행됨)
service = None

# API 라우트
@app.route('/api/health', methods=['GET'])
def health_check():
    """
    헬스 체크 엔드포인트.
    서비스 상태 및 기본 정보를 반환합니다.
    """
    return jsonify({"status": "ok", "service": "MMS Extraction API"})

@app.route('/api/extract', methods=['POST'])
def extract_info():
    """
    메인 추출 엔드포인트.
    MMS 메시지를 처리하고 구조화된 정보를 반환합니다.
    
    예상되는 JSON 페이로드:
    {
        "message_head": str,  # 메시지 제목/헤더
        "message_body": str,  # 메시지 내용
        "model_type": str     # 선택 사항, 기본값은 "claude35"
    }
    """
    global service
    
    try:
        data = request.json
        
        # 필수 매개변수 검증
        if not data or 'message_head' not in data or 'message_body' not in data:
            return jsonify({
                "error": "필수 매개변수가 누락되었습니다: message_head와 message_body가 필요합니다"
            }), 400
        
        # 메시지 정보 가져오기
        message_head = data['message_head']
        message_body = data['message_body']
        model_type = data.get('model_type', 'claude35')  # 기본값을 Claude 3.5로 설정
        
        # 아직 초기화되지 않은 경우 서비스 초기화
        if service is None:
            anthropic_api_key = os.environ.get('ANTHROPIC_API_KEY')
            ax_api_key = os.environ.get('AX_API_KEY')
            openai_api_key = os.environ.get('OPENAI_API_KEY')
            item_data_path = os.environ.get('ITEM_DATA_PATH')
            few_shot_data_path = os.environ.get('FEW_SHOT_DATA_PATH')
            
            print(f"구성으로 서버 시작:")
            print(f"포트: {port}")
            print(f"디버그: {debug}")
            print(f"항목 데이터 경로: {item_data_path}")
            print(f"Few-shot 데이터 경로: {few_shot_data_path}")
            
            service = MMSExtractionService(
                anthropic_api_key=anthropic_api_key,
                ax_api_key=ax_api_key,
                openai_api_key=openai_api_key,
                item_data_path=item_data_path,
                few_shot_data_path=few_shot_data_path
            )
        
        # 정보 추출
        result = service.extract_info(message_head, message_body, model_type)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/models', methods=['GET'])
def get_models():
    """
    사용 가능한 LLM 모델 목록을 반환합니다.
    모델 ID, 이름 및 제공업체를 포함합니다.
    """
    return jsonify({
        "models": [
            {"id": "claude35", "name": "Claude 3.5 Sonnet", "provider": "Anthropic"},
            {"id": "claude37", "name": "Claude 3.7 Sonnet", "provider": "Anthropic"},
            {"id": "chatgpt", "name": "ChatGPT", "provider": "OpenAI"},
            {"id": "ax", "name": "A.X 3 Large", "provider": "SKT"}
        ]
    })

# 메인 진입점
if __name__ == '__main__':
    """
    서버 시작 및 초기화.
    환경 변수에서 구성을 로드하고 Flask 서버를 시작합니다.
    """
    # 환경에서 구성 가져오기
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    # API 키 및 데이터 경로 로드
    anthropic_api_key = os.environ.get('ANTHROPIC_API_KEY')  # Claude 3.5용(A15T 플랫폼)
    anthropic_api_key_claude37 = os.environ.get('ANTHROPIC_API_KEY_CLAUDE37')  # Claude 3.7용(Anthropic API)
    ax_api_key = os.environ.get('AX_API_KEY')
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    item_data_path = os.environ.get('ITEM_DATA_PATH')
    few_shot_data_path = os.environ.get('FEW_SHOT_DATA_PATH')
    
    print(f"구성으로 서버 시작:")
    print(f"포트: {port}")
    print(f"디버그: {debug}")
    print(f"항목 데이터 경로: {item_data_path}")
    print(f"Few-shot 데이터 경로: {few_shot_data_path}")
    
    # 서비스 초기화
    service = MMSExtractionService(
        anthropic_api_key=anthropic_api_key,
        ax_api_key=ax_api_key,
        openai_api_key=openai_api_key,
        item_data_path=item_data_path,
        few_shot_data_path=few_shot_data_path
    )
    
    # 서버 시작
    app.run(host='0.0.0.0', port=port, debug=debug)