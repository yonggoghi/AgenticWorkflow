#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MMS Message Extractor - Cleaned Version
"""

import os
import re
import json
import pandas as pd
import numpy as np
from difflib import SequenceMatcher
from typing import List
from rapidfuzz import fuzz
import kiwipiepy
from kiwipiepy import Kiwi
from sentence_transformers import SentenceTransformer
import torch
from openai import OpenAI

# Set pandas display options
pd.set_option('display.max_colwidth', 500)

# Import configuration
from config import config

# Environment variables setup
API_KEYS = {
    'ANTHROPIC_API_KEY': config.ANTHROPIC_API_KEY,
    'OPENAI_API_KEY': config.OPENAI_API_KEY,
    'LANGSMITH_API_KEY': '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',
    'TAVILY_API_KEY': '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'
}

# Set environment variables
for key, value in API_KEYS.items():
    os.environ[key] = value

os.environ['LANGSMITH_TRACING'] = 'true'
os.environ['LANGCHAIN_API_KEY'] = API_KEYS['LANGSMITH_API_KEY']
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = 'Multi-agent Collaboration'

# Initialize OpenAI client
llm_api_key = config.CUSTOM_API_KEY"https://api.platform.a15t.com/v1"

client = OpenAI(
    api_key=llm_api_key,
    base_url=llm_api_url
)

def extract_json_objects(text):
    """Extract JSON objects from text using regex and parsing."""
    pattern = r'(\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\})'
    
    result = []
    for match in re.finditer(pattern, text):
        potential_json = match.group(0)
        try:
            json_obj = json.loads(potential_json)
            result.append(json_obj)
        except:
            pass
    
    return result

def sliding_window_with_step(data, window_size, step=1):
    """Sliding window with configurable step size."""
    return [data[i:i + window_size] for i in range(0, len(data) - window_size + 1, step)]

def needleman_wunsch_similarity(list1, list2, match_score=1, mismatch_penalty=1, gap_penalty=1):
    """Global sequence alignment with Needleman-Wunsch algorithm."""
    m, n = len(list1), len(list2)
    
    score = np.zeros((m+1, n+1))
    
    for i in range(m+1):
        score[i][0] = -i * gap_penalty
    for j in range(n+1):
        score[0][j] = -j * gap_penalty
    
    for i in range(1, m+1):
        for j in range(1, n+1):
            match = score[i-1][j-1] + (match_score if list1[i-1] == list2[j-1] else -mismatch_penalty)
            delete = score[i-1][j] - gap_penalty
            insert = score[i][j-1] - gap_penalty
            score[i][j] = max(match, delete, insert)
    
    max_possible_score = min(m, n) * match_score
    alignment_score = score[m][n]
    
    min_possible_score = -max(m, n) * max(gap_penalty, mismatch_penalty)
    normalized_score = (alignment_score - min_possible_score) / (max_possible_score - min_possible_score)
    
    return normalized_score

def advanced_sequential_similarity(str1, str2, metrics=None):
    """Calculate multiple character-level similarity metrics between two strings."""
    if metrics is None:
        metrics = ['difflib']
    
    results = {}
    
    if not str1 or not str2:
        return {metric: 0.0 for metric in metrics}
    
    s1, s2 = str1.lower(), str2.lower()
    
    # SequenceMatcher from difflib
    if 'difflib' in metrics:
        sm = SequenceMatcher(None, s1, s2)
        results['difflib'] = sm.ratio()
    
    return results

class KoreanEntityMatcher:
    def __init__(self, min_similarity=70, ngram_size=2, min_entity_length=2, token_similarity=True):
        self.min_similarity = min_similarity
        self.ngram_size = ngram_size
        self.min_entity_length = min_entity_length
        self.token_similarity = token_similarity
        self.entities = []
        self.entity_data = {}
        
    def build_from_list(self, entities):
        """Build entity index from a list of entities"""
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
                
        # Store normalized forms
        self.normalized_entities = {}
        for entity in self.entities:
            normalized = self._normalize_text(entity)
            self.normalized_entities[normalized] = entity
                
        # Create n-gram index
        self._build_ngram_index(n=self.ngram_size)
    
    def _normalize_text(self, text):
        """Normalize text - lowercase, remove spaces"""
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _tokenize(self, text):
        """Tokenize text (Korean, English, numbers)"""
        tokens = re.findall(r'[가-힣]+|[a-z0-9]+', self._normalize_text(text))
        return tokens
    
    def _build_ngram_index(self, n=2):
        """Build n-gram index optimized for Korean characters"""
        self.ngram_index = {}
        
        for entity in self.entities:
            if len(entity) < self.min_entity_length:
                continue
                
            normalized_entity = self._normalize_text(entity)
            
            # Create character-level n-grams
            entity_chars = list(normalized_entity)
            ngrams = []
            
            for i in range(len(entity_chars) - n + 1):
                ngram = ''.join(entity_chars[i:i+n])
                ngrams.append(ngram)
            
            # Add entity to index for each n-gram
            for ngram in ngrams:
                if ngram not in self.ngram_index:
                    self.ngram_index[ngram] = set()
                self.ngram_index[ngram].add(entity)
                
            # Add token-based n-grams
            tokens = self._tokenize(normalized_entity)
            for token in tokens:
                if len(token) >= n:
                    token_key = f"TOKEN:{token}"
                    if token_key not in self.ngram_index:
                        self.ngram_index[token_key] = set()
                    self.ngram_index[token_key].add(entity)
    
    def _get_candidates(self, text, n=None):
        """Get candidate entities based on n-gram overlap"""
        if n is None:
            n = self.ngram_size
            
        normalized_text = self._normalize_text(text)
        
        # Check for exact match
        if normalized_text in self.normalized_entities:
            entity = self.normalized_entities[normalized_text]
            return [(entity, float('inf'))]
        
        text_chars = list(normalized_text)
        text_ngrams = set()
        
        # Create character-level n-grams
        for i in range(len(text_chars) - n + 1):
            ngram = ''.join(text_chars[i:i+n])
            text_ngrams.add(ngram)
        
        # Add token-based n-grams
        tokens = self._tokenize(normalized_text)
        for token in tokens:
            if len(token) >= n:
                text_ngrams.add(f"TOKEN:{token}")
        
        candidates = set()
        for ngram in text_ngrams:
            if ngram in self.ngram_index:
                candidates.update(self.ngram_index[ngram])
        
        # Score candidates
        candidate_scores = {}
        for candidate in candidates:
            candidate_normalized = self._normalize_text(candidate)
            candidate_chars = list(candidate_normalized)
            candidate_ngrams = set()
            
            # Character n-grams
            for i in range(len(candidate_chars) - n + 1):
                ngram = ''.join(candidate_chars[i:i+n])
                candidate_ngrams.add(ngram)
            
            # Token n-grams
            candidate_tokens = self._tokenize(candidate_normalized)
            for token in candidate_tokens:
                if len(token) >= n:
                    candidate_ngrams.add(f"TOKEN:{token}")
            
            # Calculate overlap score
            overlap = len(candidate_ngrams.intersection(text_ngrams))
            
            # Token similarity bonus
            token_bonus = 0
            if self.token_similarity:
                query_tokens = set(tokens)
                cand_tokens = set(candidate_tokens)
                
                if query_tokens and cand_tokens:
                    common = query_tokens.intersection(cand_tokens)
                    token_bonus = len(common) * 2
            
            candidate_scores[candidate] = overlap + token_bonus
        
        return sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
    
    def _calculate_similarity(self, text, entity):
        """Calculate similarity using multiple methods"""
        normalized_text = self._normalize_text(text)
        normalized_entity = self._normalize_text(entity)
        
        if normalized_text == normalized_entity:
            return 100
        
        # Basic string similarity
        ratio_score = fuzz.ratio(normalized_text, normalized_entity)
        
        # Partial string check
        partial_score = 0
        if normalized_text in normalized_entity:
            text_len = len(normalized_text)
            entity_len = len(normalized_entity)
            partial_score = (text_len / entity_len) * 100 if entity_len > 0 else 0
        elif normalized_entity in normalized_text:
            text_len = len(normalized_text)
            entity_len = len(normalized_entity)
            partial_score = (entity_len / text_len) * 100 if text_len > 0 else 0
        
        # Token similarity
        token_score = 0
        if self.token_similarity:
            text_tokens = set(self._tokenize(normalized_text))
            entity_tokens = set(self._tokenize(normalized_entity))
            
            if text_tokens and entity_tokens:
                common_tokens = text_tokens.intersection(entity_tokens)
                all_tokens = text_tokens.union(entity_tokens)
                
                if all_tokens:
                    token_score = (len(common_tokens) / len(all_tokens)) * 100
        
        # Token sort and set ratios
        token_sort_score = fuzz.token_sort_ratio(normalized_text, normalized_entity)
        token_set_score = fuzz.token_set_ratio(normalized_text, normalized_entity)
        
        # Final weighted score
        final_score = (
            ratio_score * 0.3 +
            max(partial_score, 0) * 0.1 +
            token_score * 0.2 +
            token_sort_score * 0.2 +
            token_set_score * 0.2
        )
        
        return final_score
    
    def find_entities(self, text, max_candidates_per_span=10):
        """Find entity matches in Korean text using fuzzy matching"""
        potential_spans = self._extract_korean_spans(text)
        matches = []
        
        for span_text, start, end in potential_spans:
            if len(span_text.strip()) < self.min_entity_length:
                continue

            candidates = self._get_candidates(span_text)

            if not candidates:
                continue
            
            top_candidates = [c[0] for c in candidates[:max_candidates_per_span]]
            
            scored_matches = []
            for entity in top_candidates:
                score = self._calculate_similarity(span_text, entity)
                
                if score >= self.min_similarity:
                    scored_matches.append((entity, score, 0))

            best_matches = scored_matches

            for entity, score, _ in best_matches:
                matches.append({
                    'text': span_text,
                    'matched_entity': entity,
                    'score': score,
                    'start': start,
                    'end': end,
                    'data': self.entity_data.get(entity, {})
                })

        # Sort by position in text
        matches.sort(key=lambda x: (x['start'], -x['score']))
        
        # Handle overlapping matches
        final_matches = self._resolve_overlapping_matches(matches)

        return final_matches
    
    def _extract_korean_spans(self, text):
        """Extract potential text spans that could be entities"""
        spans = []
        min_len = self.min_entity_length
        
        # Various patterns for Korean+English mixed text
        patterns = [
            r'[a-zA-Z]+[가-힣]+(?:\s+[가-힣가-힣a-zA-Z0-9]+)*',  # English+Korean
            r'[a-zA-Z]+\s+[가-힣]+(?:\s+[가-힣가-힣a-zA-Z0-9]+)*',  # English space Korean
            r'[a-zA-Z]+[가-힣]+(?:[0-9]+)?',  # English+Korean+numbers
            r'[a-zA-Z]+[가-힣]+\s+[가-힣]+',  # English+Korean space Korean
            r'[a-zA-Z]+[가-힣]+\s+[가-힣]+\s+[가-힣]+',  # Three words
            r'[a-zA-Z가-힣]+(?:\s+[a-zA-Z가-힣]+){1,3}',  # Brand + product
            r'\d+\s+[a-zA-Z]+',  # Number space English
            r'[a-zA-Z가-힣0-9]+(?:\s+[a-zA-Z가-힣0-9]+)*'  # General mixed
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                if len(match.group(0)) >= min_len:
                    spans.append((match.group(0), match.start(), match.end()))
        
        # General separator-based extraction
        for span in re.split(r'[,\.!?;:\"\'…\(\)\[\]\{\}\s_/]+', text):
            if span and len(span) >= min_len:
                span_pos = text.find(span)
                if span_pos != -1:
                    spans.append((span, span_pos, span_pos + len(span)))
                
        return spans
    
    def _resolve_overlapping_matches(self, matches, high_score_threshold=50, overlap_tolerance=0.5):
        if not matches:
            return []
        
        sorted_matches = sorted(matches, key=lambda x: (-x['score'], x['end'] - x['start']))
        
        final_matches = []
        
        for current_match in sorted_matches:
            current_score = current_match['score']
            current_start, current_end = current_match['start'], current_match['end']
            current_range = set(range(current_start, current_end))
            current_len = len(current_range)
            
            current_match['overlap_ratio'] = 0.0
            
            if current_score >= high_score_threshold:
                is_too_similar = False
                
                for existing_match in final_matches:
                    if existing_match['score'] < high_score_threshold:
                        continue
                        
                    existing_start, existing_end = existing_match['start'], existing_match['end']
                    existing_range = set(range(existing_start, existing_end))
                    
                    intersection = current_range.intersection(existing_range)
                    current_overlap_ratio = len(intersection) / current_len if current_len > 0 else 0
                    
                    current_match['overlap_ratio'] = max(current_match['overlap_ratio'], current_overlap_ratio)
                    
                    if (current_overlap_ratio > overlap_tolerance
                        and current_match['matched_entity'] == existing_match['matched_entity']):
                        is_too_similar = True
                        break
                
                if not is_too_similar:
                    final_matches.append(current_match)
            else:
                should_add = True
                
                for existing_match in final_matches:
                    existing_start, existing_end = existing_match['start'], existing_match['end']
                    existing_range = set(range(existing_start, existing_end))
                    
                    intersection = current_range.intersection(existing_range)
                    current_overlap_ratio = len(intersection) / current_len if current_len > 0 else 0
                    
                    current_match['overlap_ratio'] = max(current_match['overlap_ratio'], current_overlap_ratio)
                    
                    if current_overlap_ratio > (1 - overlap_tolerance):
                        should_add = False
                        break
                
                if should_add:
                    final_matches.append(current_match)
        
        final_matches.sort(key=lambda x: x['start'])
        
        return final_matches

def main():
    """Main function to run the MMS extractor"""
    
    # Load data
    print("Loading MMS data...")
    try:
        mms_pdf = pd.read_csv("./data/mms_data_250408.csv")
        mms_pdf['msg'] = mms_pdf['msg_nm'] + "\n" + mms_pdf['mms_phrs']
        mms_pdf = mms_pdf.groupby(["msg_nm", "mms_phrs", "msg"])['offer_dt'].min().reset_index(name="offer_dt")
        mms_pdf = mms_pdf.reset_index().astype('str')
        print(f"Loaded {len(mms_pdf)} MMS messages")
    except FileNotFoundError:
        print("MMS data file not found. Please ensure ./data/mms_data_250408.csv exists.")
        return
    
    # Load item data
    print("Loading item data...")
    try:
        # All items
        item_pdf_raw = pd.read_csv("./data/item_info_all_250527.csv")
        item_pdf_all = item_pdf_raw.drop_duplicates(['item_nm','item_id'])[['item_nm','item_id','item_desc','domain','start_dt','end_dt','rank']].copy()
        
        print(f"Loaded {len(item_pdf_all)} items")
    except FileNotFoundError:
        print("Item data files not found. Please ensure data files exist.")
        return
    
    # Add user-defined entities
    user_defined_entity = ['AIA Vitality', '부스트 파크 건대입구', 'Boost Park 건대입구']
    item_pdf_ext = pd.DataFrame([{'item_nm':e,'item_id':e,'item_desc':e, 'domain':'user_defined', 'start_dt':20250101, 'end_dt':99991231, 'rank':1} for e in user_defined_entity])
    item_pdf_all = pd.concat([item_pdf_all, item_pdf_ext])
    
    # Load stop words
    stop_item_names_str = """멤버십
무료쿠폰
무료
쿠폰
150
가입
구독상품
우리
할인
휴대폰
114
기본
SK텔레콤
혜택안내
이용요금
이벤트
베이커리
데이터
검색
product
쿠폰
적립
친구
패키지
MMS
SKT
skt
카페
음원
부가서비스
수신거부
고객센터
TBD Product
전화
광고
인터넷
tbd product
모바일
t멤버십
sk텔레콤
매장
선택적 수신거부
T멤버십
채팅
레터링 수신거부
mms
바로가기
test1111
...
1113
추후 변경
테스트1
ㅇㅇ
사무용
미운영
ㄴ
교사
유아
환율
123
대기
종교
SMS
돈
공유기
클린
경찰청
오피스
미팅
ARS
사무실
동부
스포츠
TEST
공통
test
삭제
구.
이메일
미사용
11
백업
본인인증
비 폰
개별통화수신거부
챗봇
.
-
삼성
결합
스마트폰
후
로밍
보안
페이지
스마트
하나
미니
프로
CCTV
찬스
유선
여름
라이브
테스트
케릭터
프라임
건강
Pro
대학생
울트라
패밀리
하나카드
폴더
All
라지
멀티
이벤트
영상통화
1월 이벤트
2월 이벤트
3월 이벤트
4월 이벤트
5월 이벤트
6월 이벤트
7월 이벤트
8월 이벤트
9월 이벤트
10월 이벤트
11월 이벤트"""
    
    stop_item_names = stop_item_names_str.split('\n')
    stop_item_names = list(set(stop_item_names + [x.lower() for x in stop_item_names]))
    
    # Load BERT model
    print("Loading BERT model...")
    model = SentenceTransformer('jhgan/ko-sbert-nli')
    
    # Load PGM data
    try:
        pgm_pdf = pd.read_csv("./data/pgm_tag_ext_250516.csv")
        print(f"Loaded {len(pgm_pdf)} PGM entries")
    except FileNotFoundError:
        print("PGM data file not found.")
        pgm_pdf = pd.DataFrame()
    
    # Initialize Kiwi
    print("Initializing Kiwi...")
    kiwi = Kiwi()
    kiwi_raw = Kiwi()
    kiwi_raw.space_tolerance = 2
    
    # Add user words to Kiwi
    entity_list_for_kiwi = list(item_pdf_all['item_nm'].unique())
    for w in entity_list_for_kiwi:
        kiwi.add_user_word(w, "NNP")
    
    for w in stop_item_names:
        kiwi.add_user_word(w, "NNG")
    
    # Sample messages for testing
    msg_text_list = [
        """
        광고 제목:[SK텔레콤] 2월 0 day 혜택 안내
        광고 내용:(광고)[SKT] 2월 0 day 혜택 안내__[2월 10일(토) 혜택]_만 13~34세 고객이라면_베어유 모든 강의 14일 무료 수강 쿠폰 드립니다!_(선착순 3만 명 증정)_▶ 자세히 보기: http://t-mms.kr/t.do?m=#61&s=24589&a=&u=https://bit.ly/3SfBjjc__■ 에이닷 X T 멤버십 시크릿코드 이벤트_에이닷 T 멤버십 쿠폰함에 '에이닷이빵쏜닷'을 입력해보세요!_뚜레쥬르 데일리우유식빵 무료 쿠폰을 드립니다._▶ 시크릿코드 입력하러 가기: https://bit.ly/3HCUhLM__■ 문의: SKT 고객센터(1558, 무료)_무료 수신거부 1504
        """,
        """
        광고 제목:통화 부가서비스를 패키지로 저렴하게!
        광고 내용:(광고)[SKT] 콜링플러스 이용 안내  #04 고객님, 안녕하세요. <콜링플러스>에 가입하고 콜키퍼, 컬러링, 통화가능통보플러스까지 총 3가지의 부가서비스를 패키지로 저렴하게 이용해보세요.  ■ 콜링플러스 - 이용요금: 월 1,650원, 부가세 포함 - 콜키퍼(550원), 컬러링(990원), 통화가능통보플러스(770원)를 저렴하게 이용할 수 있는 상품  ■ 콜링플러스 가입 방법 - T월드 앱: 오른쪽 위에 있는 돋보기를 눌러 콜링플러스 검색 > 가입  ▶ 콜링플러스 가입하기: http://t-mms.kr/t.do?m=#61&u=https://skt.sh/17tNH  ■ 유의 사항 - 콜링플러스에 가입하면 기존에 이용 중인 콜키퍼, 컬러링, 통화가능통보플러스 서비스는 자동으로 해지됩니다. - 기존에 구매한 컬러링 음원은 콜링플러스 가입 후에도 계속 이용할 수 있습니다.(시간대, 발신자별 설정 정보는 다시 설정해야 합니다.)  * 최근 다운로드한 음원은 보관함에서 무료로 재설정 가능(다운로드한 날로부터 1년 이내)   ■ 문의: SKT 고객센터(114)  SKT와 함께해주셔서 감사합니다.  무료 수신거부 1504\n    
        """,
        """
        (광고)[SKT] 1월 0 day 혜택 안내_ _[1월 20일(토) 혜택]_만 13~34세 고객이라면 _CU에서 핫바 1,000원에 구매 하세요!_(선착순 1만 명 증정)_▶ 자세히 보기 : http://t-mms.kr/t.do?m=#61&s=24264&a=&u=https://bit.ly/3H2OHSs__■ 에이닷 X T 멤버십 구독캘린더 이벤트_0 day 일정을 에이닷 캘린더에 등록하고 혜택 날짜에 알림을 받아보세요! _알림 설정하면 추첨을 통해 [스타벅스 카페 라떼tall 모바일쿠폰]을 드립니다. _▶ 이벤트 참여하기 : https://bit.ly/3RVSojv_ _■ 문의: SKT 고객센터(1558, 무료)_무료 수신거부 1504
        """,
        """
        광고 제목: 부스트 파크 건대입구 제휴처 혜택을 만나 보세요. 
        광고 내용:(광고)[SKT] 부스트 파크 건대입구 제휴 혜택 안내  고객님, 안녕하세요. [소상공인 지원 프로그램] SKT 5GX Boost Park 건대입구 제휴처 할인 혜택 안내드립니다  ■ 제휴 쿠폰 받기  T멤버십App > Boost Park > 건대입구 > 쿠폰 다운로드하기 > 매장 직원에게 제시   ▶T멤버십 바로가기 : https://www.sktmembership.co.kr:443/onepass.do?m1=01&SHARE_SEQ=723825 위 URL에 접속하여 할인 쿠폰 다운로드하기 클릭 후, 매장 직원에게 쿠폰 제시 ※ 쿠폰은 제휴처당 1일 1회 사용 가능(월 1인당 쿠폰 5개 제공)  ■ 9월 부스트 파크 건대입구 T멤버십 혜택 혜택1: RENDEJA-VOUS(랑데자뷰) 건대점  - 부스트 파크 1세트 또는 2세트 주문 시 2,000~3,500원 할인(테이블당 1회)  혜택2: BROSIS(브로시스)  - 남녀 헤어 커트 5,000원 할인(본인에 한해 1회) * SK텔레콤 할인은 현장 결제 시 적용, 방문 전 전화 예약 필수, 0507-1339-7222)  혜택3: 홈워크 - 베이글 세트 구매 시 1,500원 할인(1인당 1회)  혜택4: 릴리베이커리 - 베이커리 2만 원 이상 결제 시, 1L 아메리카노 증정(1인당 1회, 소비자가 5,800원)  ※ 코로나19 확산으로 고객센터에 문의가 증가하고 있습니다. 고객센터와 전화 연결이 원활하지 않을 수 있으니 양해 바랍니다.  SKT와 함께해주셔서 감사합니다.  무료 수신거부 1504
        """,
        """
        광고 제목:[SK텔레콤] T건강습관 X AIA Vitality, 우리 가족의 든든한 보험!
        광고 내용:(광고)[SKT] 가족의 든든한 보험 (무배당)AIA Vitality 베스트핏 보장보험 안내  고객님, 안녕하세요. 4인 가족 표준생계비, 준비하고 계시나요? (무배당)AIA Vitality 베스트핏 보장보험(디지털 전용)으로 최대 20% 보험료 할인과 가족의 든든한 보험 보장까지 누려 보세요.   ▶ 자세히 보기: http://t-mms.kr/t.do?m=#61&u=https://bit.ly/36oWjgX  ■ AIA Vitality  혜택 - 매달 리워드 최대 12,000원 - 등급 업그레이드 시 특별 리워드 - T건강습관 제휴 할인 최대 40% ※ 제휴사별 할인 조건과 주간 미션 달성 혜택 등 자세한 내용은 AIA Vitality 사이트에서 확인하세요. ※ 이 광고는 AIA생명의 광고이며 SK텔레콤은 모집 행위를 하지 않습니다.  - 보험료 납입 기간 중 피보험자가 장해분류표 중 동일한 재해 또는 재해 이외의 동일한 원인으로 여러 신체 부위의 장해지급률을 더하여 50% 이상인 장해 상태가 된 경우 차회 이후의 보험료 납입 면제 - 사망보험금은 계약일(부활일/효력회복일)로부터 2년 안에 자살한 경우 보장하지 않음 - 일부 특약 갱신 시 보험료 인상 가능 - 기존 계약 해지 후 신계약 체결 시 보험인수 거절, 보험료 인상, 보장 내용 변경 가능 - 해약 환급금(또는 만기 시 보험금이나 사고보험금)에 기타 지급금을 합해 5천만 원까지(본 보험 회사 모든 상품 합산) 예금자 보호 - 계약 체결 전 상품 설명서 및 약관 참조 - 월 보험료 5,500원(부가세 포함)  * 생명보험협회 심의필 제2020-03026호(2020-09-22) COM-2020-09-32426  ■문의: 청약 관련(1600-0880)  무료 수신거부 1504    
        """
    ]
    
    # Process a sample message
    message_idx = 0
    mms_msg = msg_text_list[message_idx]
    
    print(f"\nProcessing message {message_idx + 1}...")
    print("="*80)
    print("Message:")
    print(mms_msg[:200] + "..." if len(mms_msg) > 200 else mms_msg)
    print("="*80)
    
    # Process PGM candidates if available
    num_cand_pgms = 5
    if len(pgm_pdf) > 0:
        def preprocess_text(text):
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            return text.strip()

        clue_embeddings = model.encode(
            pgm_pdf[["pgm_nm","clue_tag"]].apply(
                lambda x: preprocess_text(x['pgm_nm'].lower())+" "+x['clue_tag'].lower(), 
                axis=1
            ).tolist(), 
            convert_to_tensor=True
        )

        mms_embedding = model.encode([mms_msg.lower()], convert_to_tensor=True)

        similarities = torch.nn.functional.cosine_similarity(
            mms_embedding,  
            clue_embeddings,  
            dim=1 
        ).cpu().numpy()

        pgm_pdf_tmp = pgm_pdf.copy()
        pgm_pdf_tmp['sim'] = similarities
        pgm_pdf_tmp = pgm_pdf_tmp.sort_values('sim', ascending=False)

        print("Top PGM candidates:")
        print(pgm_pdf_tmp[['pgm_id','pgm_nm','clue_tag','sim']].head(num_cand_pgms))
    else:
        pgm_pdf_tmp = pd.DataFrame()
    
    # Entity extraction using Kiwi
    tags_to_exclude = ['W_SERIAL','W_URL','JKO','SSO','SSC','SW','SF','SP','SS','SE','SO','SB','SH']
    
    # Tokenize message
    result_msg_raw = kiwi_raw.tokenize(mms_msg, normalize_coda=True, z_coda=False, split_complex=False)
    token_list_msg = [d for d in result_msg_raw if d[1] not in tags_to_exclude]

    result_msg = kiwi.tokenize(mms_msg, normalize_coda=True, z_coda=False, split_complex=False)
    entities_from_kiwi = []
    for token in result_msg:
        if token.tag == 'NNP' and token.form not in stop_item_names+['-'] and len(token.form)>=2 and not token.form.lower() in stop_item_names:
            entities_from_kiwi.append(token.form)

    print(f"\nExtracted entities from Kiwi: {list(set(entities_from_kiwi))}")
    
    # Create entity dataframe for processing
    edf = item_pdf_all.copy()
    edf['token_entity'] = edf.apply(
        lambda x: kiwi_raw.tokenize(x['item_nm'], normalize_coda=True, z_coda=False, split_complex=False), 
        axis=1
    )
    edf['token_entity'] = edf.apply(
        lambda x: [d[0] for d in x['token_entity'] if d[1] not in tags_to_exclude], 
        axis=1
    )
    edf['char_entity'] = edf.apply(
        lambda x: list(x['item_nm'].lower().replace(' ', '')), 
        axis=1
    )
    
    # Process message tokens
    ngram_list_msg = []
    for w_size in range(1,4):
        windows = sliding_window_with_step(token_list_msg, w_size, step=1)
        windows_new = []
        for w in windows:
            tag_str = ','.join([t.tag for t in w])
            flag = True
            exc_tag_patterns = [
                ['SN', 'NNB'], ['W_SERIAL'], ['JKO'], ['W_URL'], ['W_EMAIL'],
                ['XSV', 'EC'], ['VV', 'EC'], ['VCP', 'ETM'], ['XSA', 'ETM'], ['VV', 'ETN']
            ] + [[t] for t in tags_to_exclude]
            
            for et in exc_tag_patterns:
                if ','.join(et) in tag_str:
                    flag = False
                    break
        
            if flag:
                windows_new.append([[d.form for d in w], [d.tag for d in w]])

        ngram_list_msg.extend(windows_new)

    print(f"Generated {len(ngram_list_msg)} n-grams from message")
    
    # Filter tokens by patterns
    exc_tag_patterns = [
        ['SN', 'NNB'], ['W_SERIAL'], ['JKO'], ['W_URL'], ['W_EMAIL'],
        ['XSV', 'EC'], ['VV', 'EC'], ['VCP', 'ETM'], ['XSA', 'ETM'], ['VV', 'ETN']
    ] + [[t] for t in tags_to_exclude]

    def find_pattern_indices(tokens, patterns):
        indices_to_exclude = set()
        
        # Single tag patterns
        for i in range(len(tokens)):
            for pattern in patterns:
                if len(pattern) == 1 and tokens[i].tag == pattern[0]:
                    indices_to_exclude.add(i)
        
        # Multi-tag patterns
        i = 0
        while i < len(tokens):
            if i in indices_to_exclude:
                i += 1
                continue
                
            for pattern in patterns:
                if len(pattern) > 1:
                    if i + len(pattern) <= len(tokens):
                        match = True
                        for j in range(len(pattern)):
                            if tokens[i+j].tag != pattern[j]:
                                match = False
                                break
                        
                        if match:
                            for j in range(len(pattern)):
                                indices_to_exclude.add(i+j)
            i += 1
                        
        return indices_to_exclude

    def filter_tokens_by_patterns(tokens, patterns):
        indices_to_exclude = find_pattern_indices(tokens, patterns)
        return [tokens[i] for i in range(len(tokens)) if i not in indices_to_exclude]

    def reconstruct_text_preserved_positions(original_tokens, filtered_tokens):
        token_map = {}
        for i, token in enumerate(original_tokens):
            token_map[(token.start, token.len)] = (i, token.form)
        
        filtered_indices = set()
        for token in filtered_tokens:
            key = (token.start, token.len)
            if key in token_map:
                filtered_indices.add(token_map[key][0])
        
        result = []
        for i, token in enumerate(original_tokens):
            if i in filtered_indices:
                result.append(token.form)
        
        return ' '.join(result)

    # Filter and reconstruct message
    filtered_tokens = filter_tokens_by_patterns(result_msg_raw, exc_tag_patterns)
    msg_text_filtered = reconstruct_text_preserved_positions(result_msg_raw, filtered_tokens)
    
    print(f"Filtered message length: {len(msg_text_filtered)} characters")
    
    # Create joint dataframe for entity matching
    col_for_form_tmp_ent = 'char_entity'
    col_for_form_tmp_msg = 'char_msg'

    edf['form_tmp'] = edf[col_for_form_tmp_ent].apply(
        lambda x: [' '.join(s) for s in sliding_window_with_step(x, 2, step=1)]
    )

    tdf = pd.DataFrame(ngram_list_msg).rename(columns={0:'token_txt', 1:'token_tag'})
    tdf['token_key'] = tdf.apply(lambda x: ''.join(x['token_txt'])+''.join(x['token_tag']), axis=1)
    tdf = tdf.drop_duplicates(['token_key']).drop(['token_key'], axis=1)
    tdf['char_msg'] = tdf.apply(lambda x: list((" ".join(x['token_txt'])).lower().replace(' ', '')), axis=1)

    tdf['form_tmp'] = tdf[col_for_form_tmp_msg].apply(
        lambda x: [' '.join(s) for s in sliding_window_with_step(x, 2, step=1)]
    )
    tdf['token_txt_str'] = tdf['token_txt'].str.join(',')
    tdf['token_tag_str'] = tdf['token_tag'].str.join(',')

    fdf = edf.explode('form_tmp').merge(tdf.explode('form_tmp'), on='form_tmp').drop(['form_tmp'], axis=1)
    fdf = fdf.drop_duplicates(['item_nm','item_id','token_txt_str','token_tag_str'])

    print(f"Created joint dataframe with {len(fdf)} entity-token pairs")
    
    # Calculate similarities
    fdf['sim_score_token'] = fdf.apply(
        lambda row: needleman_wunsch_similarity(row['token_txt'], row['token_entity']), 
        axis=1
    )
    fdf['sim_score_char'] = fdf.apply(
        lambda row: advanced_sequential_similarity(
            (''.join(row['char_msg'])), 
            (''.join(row['char_entity'])), 
            metrics='difflib'
        )['difflib'], 
        axis=1
    )

    # Filter high-similarity matches
    entity_list = list(edf['item_nm'].unique())
    
    # Kiwi-based matches
    kdf = fdf.query(
        "token_txt_str.str.replace(',',' ').str.lower() in @entities_from_kiwi or "
        "token_txt_str.str.replace(',','').str.lower() in @entities_from_kiwi"
    ).copy()
    kdf = kdf.query(
        "(sim_score_token>=0.75 and sim_score_char>=0.75) or sim_score_char>=1"
    ).query(
        "item_nm.str.replace(',',' ').str.lower() in @entity_list or "
        "item_nm.str.replace(',','').str.lower() in @entity_list"
    )
    kdf['rank'] = kdf.groupby(['token_txt_str'])['sim_score_char'].rank(ascending=False, method='dense')
    kdf = kdf.query("rank<=1")[['item_nm','item_id','token_txt_str','domain']].drop_duplicates()

    # Similarity-based matches
    tags_to_exclude_final = ['SN']
    filtering_condition = [
        "not token_tag_str in @tags_to_exclude_final",
        "and token_txt_str.str.len()>=2",
        "and not token_txt_str in @stop_item_names",
        "and not token_txt_str.str.replace(',','').str.lower() in @stop_item_names",
        "and not item_nm in @stop_item_names"
    ]

    sdf = (
        fdf
        .query("item_nm.str.lower() not in @stop_item_names")
        .query("(sim_score_token>=0.7 and sim_score_char>=0.8) or (sim_score_token>=0.1 and sim_score_char>=0.9)")
        .query(' '.join(filtering_condition))
        .sort_values('sim_score_char', ascending=False)
        [['item_nm','item_id','token_txt','token_txt_str','sim_score_token','sim_score_char','domain']]
    ).copy()

    sdf['rank_e'] = sdf.groupby(['item_nm'])['sim_score_char'].rank(ascending=False, method='dense')
    sdf['rank_t'] = sdf.groupby(['token_txt_str'])['sim_score_char'].rank(ascending=False, method='dense')
    sdf = sdf.query("rank_t<=1 and rank_e<=1")[['item_nm','item_id','token_txt_str','domain']].drop_duplicates()

    # Combine results
    product_df = pd.concat([kdf, sdf]).drop_duplicates(['item_id','item_nm','domain']).groupby(
        ["item_nm","item_id","domain"]
    )['token_txt_str'].apply(list).reset_index(name='item_name_in_message').rename(
        columns={'item_nm':'item_name_in_voca'}
    ).sort_values('item_name_in_voca')

    product_df['item_name_in_message'] = product_df['item_name_in_message'].apply(
        lambda x: ",".join(list(set([w.replace(',',' ') for w in x])))
    )

    print(f"\nFound {len(product_df)} product matches:")
    print(product_df[['item_name_in_message','item_name_in_voca','item_id','domain']])
    
    # Prepare for LLM processing
    product_df_clean = product_df[['item_name_in_voca','item_id','domain']].drop_duplicates()
    product_df_clean['action'] = '고객에게 기대하는 행동. [구매, 가입, 사용, 방문, 참여, 코드입력, 쿠폰다운로드, 없음, 기타] 중에서 선택'
    product_element = product_df_clean.to_dict(orient='records') if product_df_clean.shape[0] > 0 else []

    # Prepare PGM context
    if len(pgm_pdf_tmp) > 0:
        pgm_cand_info = "\n\t".join(
            pgm_pdf_tmp.iloc[:num_cand_pgms][['pgm_nm','clue_tag']].apply(
                lambda x: re.sub(r'\[.*?\]', '', x['pgm_nm'])+" : "+x['clue_tag'], 
                axis=1
            ).to_list()
        )
        rag_context = f"\n### 광고 분류 기준 정보 ###\n\t{pgm_cand_info}"
    else:
        rag_context = ""

    # Create schema for LLM
    schema_prd_1 = {
        "title": {
            "type": "string", 
            'description': '광고 제목. 광고의 핵심 주제와 가치 제안을 명확하게 설명할 수 있도록 생성'
        },
        "purpose": {
            "type": "array", 
            'description': '광고의 주요 목적을 다음 중에서 선택(복수 가능): [상품 가입 유도, 대리점 방문 유도, 웹/앱 접속 유도, 이벤트 응모 유도, 혜택 안내, 쿠폰 제공 안내, 경품 제공 안내, 기타 정보 제공]'
        },
        "product": product_element,
        'channel': {
            'type': 'array', 
            'items': {
                'type': 'object', 
                'properties': {
                    'type': {'type': 'string', 'description': '채널 종류: [URL, 전화번호, 앱, 대리점] 중에서 선택'},
                    'value': {'type': 'string', 'description': '실제 URL, 전화번호, 앱 이름, 대리점 이름 등 구체적 정보'},
                    'action': {'type': 'string', 'description': '채널 목적: [방문, 접속, 가입, 추가 정보, 문의, 수신, 수신 거부] 중에서 선택'},
                    'store_code': {'type': 'string', 'description': "매장 코드 - tworldfriends.co.kr URL에서 D+숫자 9자리(D[0-9]{9}) 패턴의 코드 추출하여 대리점 채널에 설정"}
                }
            },
        },
        'pgm': {
             'type': 'array', 
            'description': '아래 광고 분류 기준 정보에서 선택. 메세지 내용과 광고 분류 기준을 참고하여, 광고 메세지에 가장 부합하는 2개의 pgm_nm을 적합도 순서대로 제공'
        }
    }

    # Extraction guidance
    extraction_guide = """
### 분석 목표 ###
* Schema의 Product 태그 내에 action을 추출하세요.
* Schema내 action 항목 외 태그 정보는 원본 그대로 두세요.

### 고려사항 ###
* 상품 정보에 있는 항목을 임의로 변형하거나 누락시키지 마세요.
* 광고 분류 기준 정보는 pgm_nm : clue_tag 로 구성

### JSON 응답 형식 ###
응답은 설명 없이 순수한 JSON 형식으로만 제공하세요. 응답의 시작과 끝은 '{'와 '}'여야 합니다. 어떠한 추가 텍스트나 설명도 포함하지 마세요.
"""

    # Create user message for LLM
    user_message = f"""당신은 SKT 캠페인 메시지에서 정확한 정보를 추출하는 전문가입니다. 아래 schema에 따라 광고 메시지를 분석하여 완전하고 정확한 JSON 객체를 생성해 주세요:

### 분석 대상 광고 메세지 ###
{mms_msg}

### 결과 Schema ###
{json.dumps(schema_prd_1, indent=2, ensure_ascii=False)}

{extraction_guide}

{rag_context}

"""

    print("\n" + "="*80)
    print("SENDING REQUEST TO LLM...")
    print("="*80)

    try:
        # Call LLM
        response = client.chat.completions.create(
            model="skt/a.x-3-lg",
            messages=[
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,
            max_tokens=4000,
            top_p=0.95,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            response_format={"type": "json_object"}
        )
        
        # Extract and parse JSON
        result_json_text = response.choices[0].message.content
        json_objects = extract_json_objects(result_json_text)[0]

        print("\nLLM RESULT:")
        print("="*80)
        print(json.dumps(json_objects, indent=4, ensure_ascii=False))

        # Process PGM results
        if len(pgm_pdf) > 0:
            pgm_json = pgm_pdf[
                pgm_pdf['pgm_nm'].apply(
                    lambda x: re.sub(r'\[.*?\]', '', x) in ' '.join(json_objects['pgm'])
                )
            ][['pgm_nm','pgm_id']].to_dict('records')
            
            final_json = json_objects.copy()
            final_json['pgm'] = pgm_json
        else:
            final_json = json_objects

        print("\nFINAL RESULT:")
        print("="*80)
        print(json.dumps(final_json, indent=4, ensure_ascii=False))
        
        return final_json
                        
    except Exception as e:
        print(f"Error with API call: {e}")
        return None

if __name__ == "__main__":
    result = main()
    if result:
        print(f"\nExtraction completed successfully!")
    else:
        print(f"\nExtraction failed!")