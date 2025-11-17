#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick Extractor - 빠른 메시지 정보 추출기
- 메시지에서 제목과 수신 거부 전화번호를 빠르게 추출합니다.
- NLP 기법 및 LLM을 활용하여 제목을 추출합니다.
"""

import pandas as pd
import json
import re
import os
from typing import Dict, List, Optional
from collections import Counter
import numpy as np

# mms_extractor.py와 동일한 설정 사용
try:
    from config.settings import API_CONFIG, MODEL_CONFIG
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    print("⚠️  config/settings.py를 찾을 수 없습니다. 기본값을 사용합니다.")

# LLM 관련 import (선택적)
try:
    from langchain_openai import ChatOpenAI
    from langchain.prompts import PromptTemplate
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("⚠️  LLM 기능을 사용하려면 'pip install langchain langchain-openai' 실행이 필요합니다.")


class MessageInfoExtractor:
    """메시지에서 제목과 수신 거부 전화번호를 추출하는 클래스"""
    
    def __init__(self, csv_path: str, use_llm: bool = False, llm_model: str = 'gpt'):
        """
        Args:
            csv_path: CSV 파일 경로
            use_llm: LLM 사용 여부
            llm_model: 사용할 LLM 모델 ('gpt', 'claude', 'gemini' 등)
        """
        self.csv_path = csv_path
        self.df = None
        self.use_llm = use_llm and LLM_AVAILABLE
        self.llm_model = None
        self.llm_model_name = llm_model
        
        if self.use_llm:
            self._initialize_llm()
    
    def _initialize_llm(self):
        """LLM 모델 초기화 (mms_extractor.py와 완전히 동일한 방식)"""
        try:
            print("🤖 LLM 모델 초기화 중...")
            
            # mms_extractor.py와 동일한 모델 매핑 (config/settings.py 우선, 환경변수 fallback)
            if CONFIG_AVAILABLE:
                model_mapping = {
                    "gemma": getattr(MODEL_CONFIG, 'gemma_model', 'gemma-7b'),
                    "gem": getattr(MODEL_CONFIG, 'gemma_model', 'gemma-7b'),
                    "ax": getattr(MODEL_CONFIG, 'ax_model', 'skt/ax4'),
                    "claude": getattr(MODEL_CONFIG, 'claude_model', 'amazon/anthropic/claude-sonnet-4-20250514'),
                    "cld": getattr(MODEL_CONFIG, 'claude_model', 'amazon/anthropic/claude-sonnet-4-20250514'),
                    "gemini": getattr(MODEL_CONFIG, 'gemini_model', 'gcp/gemini-2.5-flash'),
                    "gen": getattr(MODEL_CONFIG, 'gemini_model', 'gcp/gemini-2.5-flash'),
                    "gpt": getattr(MODEL_CONFIG, 'gpt_model', 'azure/openai/gpt-4o-2024-08-06')
                }
            else:
                # config/settings.py가 없는 경우 환경변수 직접 사용
                model_mapping = {
                    "gemma": os.getenv('GEMMA_MODEL', 'gemma-7b'),
                    "gem": os.getenv('GEMMA_MODEL', 'gemma-7b'),
                    "ax": os.getenv('AX_MODEL', 'skt/ax4'),
                    "claude": os.getenv('CLAUDE_MODEL', 'amazon/anthropic/claude-sonnet-4-20250514'),
                    "cld": os.getenv('CLAUDE_MODEL', 'amazon/anthropic/claude-sonnet-4-20250514'),
                    "gemini": os.getenv('GEMINI_MODEL', 'gcp/gemini-2.5-flash'),
                    "gen": os.getenv('GEMINI_MODEL', 'gcp/gemini-2.5-flash'),
                    "gpt": os.getenv('GPT_MODEL', 'azure/openai/gpt-4o-2024-08-06')
                }
            
            model_name = model_mapping.get(self.llm_model_name, getattr(MODEL_CONFIG, 'llm_model', 'gcp/gemini-2.5-flash') if CONFIG_AVAILABLE else 'gcp/gemini-2.5-flash')
            
            # mms_extractor.py와 동일한 LLM 초기화 (API_CONFIG 사용)
            if CONFIG_AVAILABLE:
                api_key = getattr(API_CONFIG, 'llm_api_key', os.getenv('OPENAI_API_KEY'))
                api_base = getattr(API_CONFIG, 'llm_api_url', None)
            else:
                api_key = os.getenv('OPENAI_API_KEY')
                api_base = os.getenv('OPENAI_API_BASE')
            
            model_kwargs = {
                "temperature": 0.0,  # mms_extractor.py와 동일하게 완전 결정적 출력
                "openai_api_key": api_key,
                "openai_api_base": api_base,
                "model": model_name,
                "max_tokens": getattr(MODEL_CONFIG, 'llm_max_tokens', 1000) if CONFIG_AVAILABLE else 1000
            }
            
            # GPT 모델의 경우 시드 설정 (mms_extractor.py와 동일)
            if 'gpt' in model_name.lower():
                model_kwargs["seed"] = 42  # 고정 시드로 일관성 보장
            
            self.llm_model = ChatOpenAI(**model_kwargs)
            print(f"✅ LLM 초기화 완료: {self.llm_model_name} ({model_name})")
            if CONFIG_AVAILABLE:
                print(f"   📋 설정 소스: config/settings.py (mms_extractor.py와 동일)")
            else:
                print(f"   📋 설정 소스: 환경변수 직접 사용")
            
        except Exception as e:
            print(f"❌ LLM 초기화 실패: {e}")
            print("기본 NLP 방법만 사용합니다.")
            self.use_llm = False
    
    def load_data(self) -> pd.DataFrame:
        """CSV 또는 텍스트 파일 데이터를 로드합니다."""
        # 파일 확장자 확인
        file_ext = os.path.splitext(self.csv_path)[1].lower()
        
        if file_ext == '.txt':
            # 텍스트 파일: 각 줄이 하나의 메시지
            print(f"📄 텍스트 파일 형식 감지: {self.csv_path}")
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                messages = [line.strip() for line in f if line.strip()]
            
            # \n을 실제 줄바꿈으로 변환
            messages = [msg.replace('\\n', '\n').replace('\\t', '\t') for msg in messages]
            
            # DataFrame으로 변환 (message 컬럼만 사용)
            self.df = pd.DataFrame({
                'message': messages
            })
            print(f"데이터 로드 완료: {len(self.df)}개의 메시지 (텍스트 파일)")
        else:
            # CSV 파일: 기존 방식
            print(f"📊 CSV 파일 형식 감지: {self.csv_path}")
            self.df = pd.read_csv(self.csv_path, encoding='utf-8')
            print(f"데이터 로드 완료: {len(self.df)}개의 메시지 (CSV)")
        
        return self.df
    
    def extract_unsubscribe_phone(self, text: str) -> Optional[str]:
        """
        수신 거부 전화번호를 추출합니다.
        
        Args:
            text: 메시지 본문
            
        Returns:
            수신 거부 전화번호 또는 None
        """
        if pd.isna(text):
            return None
            
        # 패턴 1: "무료 수신거부 [전화번호]"
        pattern1 = r'무료\s*수신\s*거부\s*([0-9\-]+)'
        match = re.search(pattern1, text)
        if match:
            return match.group(1)
        
        # 패턴 2: "수신거부 [전화번호]"
        pattern2 = r'수신\s*거부\s*([0-9\-]+)'
        match = re.search(pattern2, text)
        if match:
            return match.group(1)
            
        return None
    
    def extract_title_by_llm(self, text: str) -> str:
        """
        LLM을 활용하여 제목을 추출합니다.
        
        Args:
            text: 메시지 본문
            
        Returns:
            추출된 제목
        """
        if pd.isna(text):
            return ""
        
        if not self.use_llm or self.llm_model is None:
            print("⚠️  LLM이 초기화되지 않았습니다. TextRank 방법을 사용합니다.")
            return self._extract_by_textrank(text)
        
        try:
            # 프롬프트 템플릿
            prompt_template = """당신은 광고 메시지 분석 전문가입니다.
아래 MMS 광고 메시지에서 핵심 내용을 요약한 제목을 한 문장으로 추출해주세요.

## 지침:
1. 광고의 핵심 내용(혜택, 상품, 이벤트 등)을 명확히 담아야 합니다
2. 한 문장으로 간결하게 작성합니다 (최대 50자)
3. "(광고)", "[SKT]" 같은 라벨은 제외합니다
4. 특수문자(__,  등)는 제거하고 자연스러운 문장으로 만듭니다
5. 가장 중요한 정보를 우선적으로 포함합니다
6. 제목은 개조식으로 생성합니다.

## 출력 형식:
- 제목만 출력하고 다른 설명은 하지 마세요
- JSON이나 마크다운 형식 없이 순수한 텍스트로만 출력하세요

## MMS 메시지:
{message}

## 추출된 제목:"""

            # 프롬프트 실행
            prompt = PromptTemplate(
                input_variables=["message"],
                template=prompt_template
            )
            
            chain = prompt | self.llm_model
            response = chain.invoke({"message": text[:1000]})  # 긴 메시지는 앞부분만
            
            # 응답 추출
            title = response.content.strip() if hasattr(response, 'content') else str(response).strip()
            
            # 후처리: 따옴표 제거, 너무 긴 경우 자르기
            title = title.strip('"\'')
            if len(title) > 150:
                title = title[:150] + '...'
            
            # 빈 응답인 경우 fallback
            if not title or len(title) < 5:
                print("⚠️  LLM이 유효한 제목을 생성하지 못했습니다. TextRank를 사용합니다.")
                return self._extract_by_textrank(text)
            
            return title
            
        except Exception as e:
            print(f"⚠️  LLM 제목 추출 실패: {e}")
            print("TextRank 방법으로 fallback합니다.")
            return self._extract_by_textrank(text)
    
    def extract_title_by_nlp(self, text: str, method: str = 'textrank') -> str:
        """
        NLP 기법을 활용하여 제목을 추출합니다.
        
        Args:
            text: 메시지 본문
            method: 추출 방법 ('textrank', 'tfidf', 'first_bracket', 'llm')
            
        Returns:
            추출된 제목
        """
        if pd.isna(text):
            return ""
        
        # LLM 방법
        if method == 'llm':
            return self.extract_title_by_llm(text)
        
        # 먼저 대괄호 안의 텍스트를 제목 후보로 추출
        bracket_pattern = r'\[([^\]]+)\]'
        bracket_matches = re.findall(bracket_pattern, text)
        
        # 광고 라벨 제거
        if bracket_matches:
            for match in bracket_matches:
                if '광고' not in match and 'SKT' not in match:
                    # 광고/SKT가 아닌 대괄호 내용을 제목으로 사용
                    return match.strip()
        
        if method == 'first_bracket':
            # 첫 번째 대괄호 내용 (광고 제외)
            for match in bracket_matches:
                if match.strip() not in ['광고', 'SK텔레콤', 'SKT']:
                    return match.strip()
        
        elif method == 'textrank':
            return self._extract_by_textrank(text)
        
        elif method == 'tfidf':
            return self._extract_by_tfidf(text)
        
        # 기본: 첫 문장 (언더바 기준으로 분리)
        sentences = text.split('_')
        if len(sentences) > 1:
            # (광고)[SKT] 부분 제거
            first_sentence = sentences[1].strip() if len(sentences) > 1 else sentences[0].strip()
            # 대괄호 제거
            first_sentence = re.sub(r'\[.*?\]', '', first_sentence).strip()
            # (광고) 제거
            first_sentence = re.sub(r'\(광고\)', '', first_sentence).strip()
            return first_sentence
        
        return text[:50]  # 기본값: 처음 50자
    
    def _extract_by_textrank(self, text: str) -> str:
        """
        TextRank 알고리즘을 사용하여 중요 문장을 추출합니다.
        간단한 구현: 단어 빈도와 문장 길이를 고려
        
        Args:
            text: 메시지 본문
            
        Returns:
            추출된 제목
        """
        # 언더바로 문장 분리
        sentences = [s.strip() for s in text.split('_') if s.strip()]
        
        if not sentences:
            return text[:50]
        
        # 광고/SKT 포함 문장 제거
        filtered_sentences = []
        for sent in sentences:
            if '(광고)' not in sent and '[광고]' not in sent:
                # 대괄호 제거
                clean_sent = re.sub(r'\[.*?\]', '', sent).strip()
                if clean_sent and len(clean_sent) > 5:
                    filtered_sentences.append(clean_sent)
        
        if not filtered_sentences:
            return text[:50]
        
        # 각 문장의 중요도 계산 (단순화된 버전)
        scores = []
        for sent in filtered_sentences:
            # 길이와 키워드 포함 여부로 점수 계산
            score = 0
            
            # 적절한 길이의 문장 선호 (10-100자)
            if 10 <= len(sent) <= 100:
                score += 10
            
            # 중요 키워드 포함 시 가중치
            keywords = ['혜택', '안내', '이벤트', '할인', '무료', '특별', '서비스']
            for keyword in keywords:
                if keyword in sent:
                    score += 5
            
            # 너무 긴 문장은 감점
            if len(sent) > 150:
                score -= 10
                
            scores.append(score)
        
        # 가장 높은 점수의 문장 선택
        if scores:
            best_idx = np.argmax(scores)
            title = filtered_sentences[best_idx]
            
            # 길이 제한 (최대 100자)
            if len(title) > 100:
                title = title[:100] + '...'
            
            return title
        
        return filtered_sentences[0][:100] if filtered_sentences else text[:50]
    
    def _extract_by_tfidf(self, text: str) -> str:
        """
        TF-IDF 기반으로 중요 문장을 추출합니다.
        간단한 구현: 문자 빈도 기반
        
        Args:
            text: 메시지 본문
            
        Returns:
            추출된 제목
        """
        # 언더바로 문장 분리
        sentences = [s.strip() for s in text.split('_') if s.strip()]
        
        if not sentences:
            return text[:50]
        
        # 광고 문장 필터링
        filtered_sentences = []
        for sent in sentences:
            if '(광고)' not in sent:
                clean_sent = re.sub(r'\[.*?\]', '', sent).strip()
                if clean_sent and len(clean_sent) > 5:
                    filtered_sentences.append(clean_sent)
        
        if not filtered_sentences:
            return text[:50]
        
        # 간단한 TF 계산 (문자 기반)
        all_chars = ''.join(filtered_sentences)
        char_freq = Counter(all_chars)
        
        # 각 문장의 TF 점수 계산
        scores = []
        for sent in filtered_sentences:
            score = sum(char_freq[c] for c in sent if c.isalnum())
            # 문장 길이로 정규화
            if len(sent) > 0:
                score = score / len(sent)
            scores.append(score)
        
        # 가장 높은 점수의 문장 선택
        if scores:
            best_idx = np.argmax(scores)
            title = filtered_sentences[best_idx]
            
            # 길이 제한
            if len(title) > 100:
                title = title[:100] + '...'
            
            return title
        
        return filtered_sentences[0][:100] if filtered_sentences else text[:50]
    
    def extract_all(self, title_method: str = 'textrank') -> List[Dict]:
        """
        모든 메시지에서 정보를 추출합니다.
        
        Args:
            title_method: 제목 추출 방법
            
        Returns:
            추출된 정보 리스트
        """
        import time
        
        if self.df is None:
            self.load_data()
        
        results = []
        
        # 파일 타입 확인 (CSV vs 텍스트)
        is_text_file = 'message' in self.df.columns and 'mms_phrs' not in self.df.columns
        
        for idx, row in self.df.iterrows():
            # 메시지별 처리 시작 시간
            start_time = time.time()
            
            # 메시지 텍스트 추출 (파일 형식에 따라)
            if is_text_file:
                # 텍스트 파일: 'message' 컬럼 사용
                message_text = row.get('message', '')
            else:
                # CSV 파일: 'mms_phrs' 컬럼 사용
                message_text = row.get('mms_phrs', '')
            
            # 제목 추출
            title = self.extract_title_by_nlp(message_text, method=title_method)
            
            # 수신 거부 전화번호 추출
            unsubscribe_phone = self.extract_unsubscribe_phone(message_text)
            
            # 메시지별 처리 시간 계산
            processing_time = time.time() - start_time
            
            # 결과 구성 (파일 형식에 따라)
            if is_text_file:
                # 텍스트 파일: 메시지 본문만 저장
                result = {
                    'msg_id': int(idx),
                    'title': title,
                    'unsubscribe_phone': unsubscribe_phone,
                    'message': message_text,
                    'processing_time_seconds': round(processing_time, 3)
                }
            else:
                # CSV 파일: 기존 구조 유지
                result = {
                    'msg_id': int(idx),
                    'offer_date': str(row.get('offer_dt', '')),
                    'title': title,
                    'unsubscribe_phone': unsubscribe_phone,
                    'original_message_name': str(row.get('msg_nm', '')),
                    'processing_time_seconds': round(processing_time, 3)
                }
            
            results.append(result)
        
        return results
    
    def save_to_json(self, results: List[Dict], output_path: str):
        """
        결과를 JSON 파일로 저장합니다.
        
        Args:
            results: 추출된 정보 리스트
            output_path: 출력 파일 경로
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"결과 저장 완료: {output_path}")
        print(f"총 {len(results)}개의 메시지 정보 추출")
    
    def process_single_message(self, message: str, method: str = 'textrank') -> Dict:
        """
        단일 메시지를 처리하고 JSON 결과를 반환합니다. (API용)
        
        Args:
            message: 처리할 메시지 텍스트
            method: 제목 추출 방법 ('textrank', 'tfidf', 'first_bracket', 'llm')
            
        Returns:
            JSON 형식의 추출 결과 딕셔너리
        """
        # 제목 추출
        title = self.extract_title_by_nlp(message, method=method)
        
        # 수신 거부 전화번호 추출
        unsubscribe_phone = self.extract_unsubscribe_phone(message)
        
        # JSON 결과 구성
        result = {
            'success': True,
            'data': {
                'title': title,
                'unsubscribe_phone': unsubscribe_phone,
                'message': message
            },
            'metadata': {
                'method': method,
                'message_length': len(message)
            }
        }
        
        return result
    
    def process_batch_file(self, file_path: str, method: str = 'textrank') -> Dict:
        """
        배치 파일을 처리하고 JSON 결과를 반환합니다. (API용)
        
        Args:
            file_path: 처리할 파일 경로 (CSV 또는 텍스트)
            method: 제목 추출 방법 ('textrank', 'tfidf', 'first_bracket', 'llm')
            
        Returns:
            JSON 형식의 추출 결과 딕셔너리
        """
        # 파일 존재 확인
        if not os.path.exists(file_path):
            return {
                'success': False,
                'error': f'파일을 찾을 수 없습니다: {file_path}',
                'data': None
            }
        
        try:
            # 데이터 로드
            self.load_data()
            
            # 정보 추출
            results = self.extract_all(title_method=method)
            
            # 통계 계산
            total = len(results)
            with_phone = sum(1 for r in results if r.get('unsubscribe_phone'))
            
            # 처리 시간 통계
            processing_times = [r.get('processing_time_seconds', 0) for r in results]
            total_time = sum(processing_times)
            avg_time = total_time / total if total > 0 else 0
            min_time = min(processing_times) if processing_times else 0
            max_time = max(processing_times) if processing_times else 0
            
            # JSON 결과 구성
            result = {
                'success': True,
                'data': {
                    'messages': results,
                    'statistics': {
                        'total_messages': total,
                        'with_unsubscribe_phone': with_phone,
                        'extraction_rate': round(with_phone / total * 100, 2) if total > 0 else 0,
                        'total_processing_time_seconds': round(total_time, 3),
                        'avg_processing_time_seconds': round(avg_time, 3),
                        'min_processing_time_seconds': round(min_time, 3),
                        'max_processing_time_seconds': round(max_time, 3)
                    }
                },
                'metadata': {
                    'method': method,
                    'file_path': file_path,
                    'file_type': 'text' if file_path.endswith('.txt') else 'csv'
                }
            }
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'data': None
            }


def main():
    """
    커맨드라인에서 실행할 때의 메인 함수 (mms_extractor.py와 동일한 방식)
    
    사용법:
    # 단일 메시지 처리
    python quick_extractor.py --message "광고 메시지 내용" --method llm --llm-model gpt
    
    # 배치 파일 처리 (CSV 또는 텍스트)
    python quick_extractor.py --batch-file ./data/messages.csv --method textrank --output results.json
    
    # 기본 설정으로 실행 (기본 배치 파일)
    python quick_extractor.py
    """
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        description='Quick Extractor - 빠른 메시지 정보 추출기 (mms_extractor.py와 동일한 입력 방식)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 단일 메시지 처리 (LLM 사용)
  python quick_extractor.py --message "광고 메시지" --method llm --llm-model gpt
  
  # 단일 메시지 처리 (NLP 방법)
  python quick_extractor.py --message "광고 메시지" --method textrank
  
  # 배치 파일 처리 (CSV 또는 텍스트)
  python quick_extractor.py --batch-file ./data/messages.csv --output results.json
  python quick_extractor.py --batch-file ./data/messages.txt --method llm --llm-model ax
  
  # 기본 설정으로 실행
  python quick_extractor.py
        """
    )
    
    # 입력 소스 (mms_extractor.py와 동일)
    parser.add_argument('--message', type=str, help='처리할 단일 메시지 텍스트')
    parser.add_argument('--batch-file', type=str, default='./data/mms_data_251023.csv', 
                       help='배치 처리할 파일 경로 (CSV 또는 텍스트, 기본값: ./data/mms_data_251023.csv)')
    
    # 출력 옵션
    parser.add_argument('--output', type=str, default='./quick_extracted_info.json',
                       help='결과를 저장할 JSON 파일 경로 (배치 파일 모드 전용, 기본값: ./quick_extracted_info.json)')
    
    # 추출 방법 옵션
    parser.add_argument('--method', type=str, default='llm',
                       choices=['textrank', 'tfidf', 'first_bracket', 'llm'],
                       help='제목 추출 방법 (기본값: llm)')
    
    # LLM 옵션 (mms_extractor.py와 동일)
    parser.add_argument('--llm-model', type=str, default='ax',
                       choices=['gpt', 'claude', 'gemini', 'ax', 'gem', 'gen', 'cld'],
                       help='LLM 모델 선택 (llm 방법 사용 시, 기본값: ax)')
    
    args = parser.parse_args()
    
    # 입력 검증
    if args.message and args.batch_file != './data/mms_data_251023.csv':
        print("⚠️  --message와 --batch-file을 동시에 지정할 수 없습니다. --message 우선 처리합니다.")
    
    print(f"\n{'='*60}")
    print(f"Quick Extractor - 메시지 정보 추출기")
    print(f"{'='*60}")
    
    use_llm = (args.method == 'llm')
    
    if use_llm:
        print(f"🤖 LLM 모드 활성화: {args.llm_model}")
    
    # 단일 메시지 처리 (mms_extractor.py와 동일한 방식)
    if args.message:
        print(f"\n📝 단일 메시지 처리 모드")
        print(f"제목 추출 방법: {args.method}")
        if use_llm:
            print(f"LLM 모델: {args.llm_model}")
        
        # CSV 경로가 없으므로 임시로 None 전달
        extractor = MessageInfoExtractor(csv_path=None, use_llm=use_llm, llm_model=args.llm_model)
        
        # 단일 메시지에서 정보 추출
        print(f"\n처리 중...")
        title = extractor.extract_title_by_nlp(args.message, method=args.method)
        unsubscribe_phone = extractor.extract_unsubscribe_phone(args.message)
        
        # 결과 출력
        result = {
            'title': title,
            'unsubscribe_phone': unsubscribe_phone,
            'original_message': args.message[:100] + '...' if len(args.message) > 100 else args.message
        }
        
        print(f"\n{'='*60}")
        print(f"📊 추출 결과")
        print(f"{'='*60}")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"{'='*60}\n")
        
    else:
        # 배치 파일 전체 처리 (기존 방식)
        batch_file = args.batch_file
        output_path = args.output
        
        print(f"\n📁 배치 파일 처리 모드")
        print(f"입력 파일: {batch_file}")
        print(f"출력 파일: {output_path}")
        
        if not os.path.exists(batch_file):
            print(f"\n❌ 파일을 찾을 수 없습니다: {batch_file}")
            sys.exit(1)
        
        extractor = MessageInfoExtractor(batch_file, use_llm=use_llm, llm_model=args.llm_model)
        
        # 데이터 로드
        extractor.load_data()
        
        # 정보 추출
        method_names = {
            'textrank': 'TextRank (문장 중요도 기반)',
            'tfidf': 'TF-IDF (단어 빈도 기반)',
            'first_bracket': '첫 번째 대괄호 내용',
            'llm': 'LLM (Large Language Model)'
        }
        print(f"\n제목 추출 방법: {method_names.get(args.method, args.method)}")
        results = extractor.extract_all(title_method=args.method)
        
        # JSON으로 저장
        extractor.save_to_json(results, output_path)
        
        # 샘플 출력 (파일 형식에 따라 다르게 표시)
        print("\n=== 추출 결과 샘플 (처음 5개) ===")
        is_text_file = 'message' in results[0] if results else False
        
        for i, result in enumerate(results[:5]):
            print(f"\n[메시지 {i+1}] (처리시간: {result.get('processing_time_seconds', 0)}초)")
            if is_text_file:
                # 텍스트 파일 출력
                print(f"  - 추출된 제목: {result['title']}")
                print(f"  - 수신거부 번호: {result['unsubscribe_phone']}")
                print(f"  - 메시지 미리보기: {result['message']}")
            else:
                # CSV 파일 출력
                print(f"  - 날짜: {result.get('offer_date', 'N/A')}")
                print(f"  - 추출된 제목: {result['title']}")
                print(f"  - 수신거부 번호: {result['unsubscribe_phone']}")
                print(f"  - 원본 제목: {result.get('original_message_name', 'N/A')}")
        
        # 통계 출력
        print("\n=== 추출 통계 ===")
        total = len(results)
        with_phone = sum(1 for r in results if r['unsubscribe_phone'])
        
        # 처리 시간 통계
        processing_times = [r.get('processing_time_seconds', 0) for r in results]
        total_time = sum(processing_times)
        avg_time = total_time / total if total > 0 else 0
        min_time = min(processing_times) if processing_times else 0
        max_time = max(processing_times) if processing_times else 0
        
        print(f"전체 메시지: {total}개")
        print(f"수신거부 번호 추출: {with_phone}개 ({with_phone/total*100:.1f}%)")
        print(f"\n처리 시간:")
        print(f"  - 총 처리 시간: {total_time:.3f}초")
        print(f"  - 평균 처리 시간: {avg_time:.3f}초/메시지")
        print(f"  - 최소 처리 시간: {min_time:.3f}초")
        print(f"  - 최대 처리 시간: {max_time:.3f}초")
        
        # 수신거부 번호 분포
        phone_counter = Counter(r['unsubscribe_phone'] for r in results if r['unsubscribe_phone'])
        print("\n수신거부 번호 분포:")
        for phone, count in phone_counter.most_common():
            print(f"  - {phone}: {count}개")
        
        print(f"\n{'='*60}")
        print(f"완료! 결과가 '{output_path}'에 저장되었습니다.")
        print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

