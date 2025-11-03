#!/usr/bin/env python3
"""
상품/서비스 정보 크롤러

웹 페이지에서 상품/서비스 정보를 추출하고 상세 페이지까지 자동으로 크롤링합니다.
LLM을 사용하여 비구조화된 데이터도 처리 가능합니다.
"""

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from urllib.parse import urljoin, urlparse

import pandas as pd
from tqdm import tqdm

try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("Warning: Playwright is not installed.")

try:
    from langchain_openai import ChatOpenAI
    from langchain.schema import HumanMessage, SystemMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("Warning: langchain-openai is not installed.")

try:
    from config import settings
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    print("Warning: config.py not found. Using default settings.")


class ProductCrawler:
    """상품/서비스 정보를 크롤링하고 추출하는 클래스"""
    
    def __init__(self, base_url: str, use_llm: bool = True, model_name: str = "ax"):
        """
        Args:
            base_url: 크롤링할 기본 URL
            use_llm: LLM을 사용하여 정보 추출 여부
            model_name: 사용할 LLM 모델 ("gemma", "ax", "claude", "gemini", "gpt")
        """
        self.base_url = base_url
        self.use_llm = use_llm
        self.model_name = model_name
        self.products = []
        
        # LLM 클라이언트 초기화
        if use_llm and LANGCHAIN_AVAILABLE:
            self._init_llm_client()
        else:
            if use_llm:
                print("Warning: langchain not available. LLM features disabled.")
            self.use_llm = False
            self.llm_client = None
    
    def _init_llm_client(self):
        """LLM 클라이언트를 초기화합니다."""
        if not CONFIG_AVAILABLE:
            print("Warning: config not available. Using environment variables.")
            api_key = os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY"))
            base_url = os.getenv("LLM_API_URL", "https://api.openai.com/v1")
            model = "gpt-4o-mini"
            max_tokens = 8000
            temperature = 0.0
            seed = 42
        else:
            api_key = settings.API_CONFIG.llm_api_key
            base_url = settings.API_CONFIG.llm_api_url
            max_tokens = settings.ModelConfig.llm_max_tokens
            temperature = settings.ModelConfig.temperature
            seed = settings.ModelConfig.seed
            
            # 모델 선택
            model_map = {
                "gemma": settings.ModelConfig.gemma_model,
                "ax": settings.ModelConfig.ax_model,
                "claude": settings.ModelConfig.claude_model,
                "gemini": settings.ModelConfig.gemini_model,
                "gpt": settings.ModelConfig.gpt_model,
            }
            model = model_map.get(self.model_name, settings.ModelConfig.default_model)
        
        if not api_key:
            print("Warning: LLM API key not found. LLM features disabled.")
            self.use_llm = False
            self.llm_client = None
            return
        
        try:
            self.llm_client = ChatOpenAI(
                temperature=temperature,
                openai_api_key=api_key,
                openai_api_base=base_url,
                model=model,
                max_tokens=max_tokens,
                seed=seed
            )
            print(f"LLM 초기화: {model}")
        except Exception as e:
            print(f"LLM 초기화 오류: {e}")
            self.use_llm = False
            self.llm_client = None
    
    def crawl_page(self, url: str, infinite_scroll: bool = False, 
                   scroll_count: int = 10, wait_time: int = 2000) -> Dict:
        """
        페이지를 크롤링합니다.
        
        Args:
            url: 크롤링할 URL
            infinite_scroll: 무한 스크롤 여부
            scroll_count: 스크롤 횟수
            wait_time: 대기 시간 (밀리초)
            
        Returns:
            크롤링 결과 딕셔너리
        """
        if not PLAYWRIGHT_AVAILABLE:
            raise ImportError("Playwright가 설치되어 있지 않습니다.")
        
        result = {
            'url': url,
            'success': False,
            'html_content': '',
            'text_content': '',
            'links': [],
            'error': None
        }
        
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                
                print(f"  페이지 로딩: {url}")
                page.goto(url, wait_until='networkidle', timeout=30000)
                page.wait_for_timeout(wait_time)
                
                # 무한 스크롤 처리
                if infinite_scroll:
                    for i in range(scroll_count):
                        previous_height = page.evaluate('document.body.scrollHeight')
                        page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
                        page.wait_for_timeout(2000)  # 2초 대기 (더 많은 콘텐츠 로드 보장)
                        new_height = page.evaluate('document.body.scrollHeight')
                        
                        if new_height == previous_height:
                            break
                    
                    # 최종 대기 시간 증가 (모든 콘텐츠 완전 로드)
                    print(f"  스크롤 완료, 최종 콘텐츠 로딩 대기...")
                    page.wait_for_timeout(3000)
                
                # 콘텐츠 수집
                result['html_content'] = page.content()
                result['text_content'] = page.inner_text('body')
                
                # 링크 수집
                links = page.evaluate('''() => {
                    return Array.from(document.querySelectorAll('a[href]')).map(a => ({
                        text: a.innerText.trim(),
                        href: a.href
                    })).filter(link => link.href && link.text);
                }''')
                result['links'] = links
                result['success'] = True
                
                browser.close()
                
        except Exception as e:
            result['error'] = str(e)
            print(f"  크롤링 오류: {e}")
        
        return result
    
    def _chunk_text(self, text: str, chunk_size: int = 10000, overlap: int = 500) -> List[str]:
        """
        텍스트를 청크로 나눕니다.
        
        Args:
            text: 나눌 텍스트
            chunk_size: 청크 크기
            overlap: 청크 간 오버랩 크기
            
        Returns:
            청크 리스트
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # 마지막 청크가 아니면 문장 경계에서 자르기
            if end < len(text):
                # 다음 줄바꿈이나 마침표를 찾아서 자르기
                for sep in ['\n\n', '\n', '. ', ' ']:
                    last_sep = text[start:end].rfind(sep)
                    if last_sep != -1:
                        end = start + last_sep + len(sep)
                        break
            
            chunks.append(text[start:end])
            
            # 다음 청크는 오버랩을 고려해서 시작
            start = end - overlap if end < len(text) else end
        
        return chunks
    
    def _extract_product_ids_from_html(self, html_content: str) -> Dict[str, str]:
        """
        HTML에서 상품 ID와 이름 매핑 추출
        
        Args:
            html_content: HTML 내용
            
        Returns:
            {상품명: 상품ID} 딕셔너리
        """
        product_map = {}
        
        try:
            import re
            from bs4 import BeautifulSoup
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # prdid 속성이 있는 상품 컨테이너 찾기
            product_containers = soup.find_all(attrs={'prdid': True})
            
            for container in product_containers:
                prd_id = container.get('prdid')
                if not prd_id:
                    continue
                
                # 상품명 찾기 (여러 패턴 시도)
                product_name = None
                
                # 패턴 1: txt-product 클래스 내 strong 태그
                txt_product = container.find(class_=re.compile(r'txt-product'))
                if txt_product:
                    strong_tag = txt_product.find('strong')
                    if strong_tag:
                        product_name = strong_tag.get_text(strip=True)
                
                # 패턴 2: title 또는 name 클래스
                if not product_name:
                    for class_name in ['title', 'name', 'product-name', 'prod-name']:
                        elem = container.find(class_=re.compile(class_name, re.I))
                        if elem:
                            product_name = elem.get_text(strip=True)
                            break
                
                # 패턴 3: strong, em, b 태그 (가격이 아닌 것)
                if not product_name:
                    for tag_name in ['strong', 'em', 'b', 'h1', 'h2', 'h3', 'h4']:
                        for elem in container.find_all(tag_name):
                            text = elem.get_text(strip=True)
                            # 가격이 아니고, 충분히 긴 텍스트
                            if text and len(text) >= 3 and not re.match(r'^[\d,]+원', text):
                                product_name = text
                                break
                        if product_name:
                            break
                
                # 패턴 4: 컨테이너 전체 텍스트에서 첫 번째 의미있는 텍스트
                if not product_name:
                    all_text = container.get_text(separator='|', strip=True)
                    parts = [p.strip() for p in all_text.split('|') if p.strip()]
                    for part in parts:
                        # 가격, 할인율, 배지 등이 아닌 텍스트
                        if (len(part) >= 3 and 
                            not re.match(r'^[\d,]+원', part) and
                            not re.match(r'^\d+%', part) and
                            part not in ['SKT 할인', '할인', '신규', '인기', 'NEW', 'BEST']):
                            product_name = part
                            break
                
                if product_name and len(product_name) >= 2:
                    # 상품명 정리
                    product_name = re.sub(r'\d+,?\d*원.*', '', product_name).strip()
                    product_name = re.sub(r'\d+%\s*(할인|OFF).*', '', product_name).strip()
                    product_name = product_name[:100]  # 최대 100자
                    
                    if product_name and len(product_name) >= 2:
                        # 중복 방지: 같은 이름이 여러 개면 ID 리스트로 저장
                        if product_name in product_map:
                            # 이미 있으면 더 짧은 ID 또는 리스트로 관리
                            existing_id = product_map[product_name]
                            if isinstance(existing_id, str):
                                # 첫 번째 것 유지 (보통 첫 번째가 대표 상품)
                                pass
                        else:
                            product_map[product_name] = prd_id
            
            print(f"  HTML에서 {len(product_map)}개의 상품 ID 추출")
            
            # 샘플 출력 (디버깅용)
            if product_map and len(product_map) > 0:
                sample_items = list(product_map.items())[:3]
                print(f"  샘플: ", end="")
                for name, pid in sample_items:
                    print(f"[{name[:20]}→{pid}] ", end="")
                print()
            
        except Exception as e:
            print(f"  HTML 파싱 오류 (무시): {str(e)}")
        
        return product_map
    
    def extract_products_with_llm(self, html_content: str, text_content: str) -> List[Dict]:
        """
        LLM을 사용하여 상품/서비스 정보를 추출합니다.
        텍스트가 길면 청크로 나눠서 처리합니다.
        
        Args:
            html_content: HTML 내용
            text_content: 텍스트 내용
            
        Returns:
            상품 정보 리스트
        """
        if not self.use_llm or not self.llm_client:
            return []
        
        # HTML에서 실제 상품 ID 추출
        product_id_map = self._extract_product_ids_from_html(html_content)
        
        # 텍스트를 청크로 나누기 (청크 크기: 4000자, 오버랩: 500자)
        # 청크 크기를 작게 하면 LLM이 완전한 응답을 생성할 확률이 높아집니다
        chunks = self._chunk_text(text_content, chunk_size=4000, overlap=500)
        
        print(f"  텍스트를 {len(chunks)}개 청크로 분할")
        
        all_products = []
        seen_products = set()  # 중복 제거용 (상품명 기준)
        
        # 각 청크마다 LLM 호출
        for idx, chunk in enumerate(chunks):
            print(f"  청크 {idx+1}/{len(chunks)} 처리 중... ({len(chunk)} 문자)")
            
            prompt = f"""다음은 웹 페이지의 일부 내용입니다. 이 페이지에서 상품 또는 서비스 정보를 추출해주세요.

각 상품/서비스에 대해 다음 정보를 JSON 배열 형식으로 반환해주세요:
- id: 상품 식별자 (빈 문자열 "")
- name: 상품/서비스 이름 (정확하게, 최대 50자)
- description: 상품/서비스 설명 (최대 10자, 핵심만)
- price: 가격
- detail_url: 상세 페이지 URL (있으면 사용, 없으면 빈 문자열)

웹 페이지 내용 (일부):
---
{chunk}
---

응답 형식 (JSON만, 설명 없이):
[
  {{"id": "", "name": "상품명", "description": "할인", "price": "가격", "detail_url": ""}},
  {{"id": "", "name": "상품명2", "description": "쿠폰", "price": "가격2", "detail_url": ""}}
]

중요 규칙:
1. 완전한 JSON 배열만 반환 (마지막 ] 필수)
2. id는 빈 문자열로 (나중에 자동으로 매칭됨)
3. name은 정확하게 (ID 매칭에 사용됨)
4. description은 10자 이내 (예: "할인", "쿠폰", "배송")
5. 모든 상품 추출 (빠뜨리지 말 것)
6. JSON 외 다른 텍스트 금지
"""
            
            try:
                # langchain ChatOpenAI 사용
                messages = [HumanMessage(content=prompt)]
                response = self.llm_client.invoke(messages)
                response_text = response.content
                
                # 응답이 잘렸는지 확인 (디버깅용)
                if hasattr(response, 'response_metadata'):
                    finish_reason = response.response_metadata.get('finish_reason', '')
                    if finish_reason == 'length':
                        print(f"    ⚠️ 주의: LLM 응답이 max_tokens 제한으로 잘렸습니다")
                
                # JSON 파싱 전처리
                # 마크다운 코드 블록 제거
                response_text = re.sub(r'^```json\s*', '', response_text, flags=re.MULTILINE)
                response_text = re.sub(r'^```\s*$', '', response_text, flags=re.MULTILINE)
                response_text = response_text.strip()
                
                # 빈 응답 체크
                if not response_text or response_text == "[]":
                    continue
                
                # JSON 배열 찾기 (설명이나 주석 제거)
                json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
                if json_match:
                    response_text = json_match.group(0)
                
                products = json.loads(response_text)
                
                # 중복 제거하면서 추가
                for product in products:
                    product_key = product.get('name', '') + product.get('price', '')
                    if product_key and product_key not in seen_products:
                        seen_products.add(product_key)
                        all_products.append(product)
                
                print(f"    → {len(products)}개 상품 발견 (총 {len(all_products)}개)")
                
            except json.JSONDecodeError as e:
                print(f"    → JSON 파싱 오류: {e}")
                
                # 불완전한 JSON 복구 시도
                try:
                    # 마지막 항목이 잘렸을 가능성 - 마지막 완전한 객체까지만 파싱
                    # 마지막 완전한 }를 찾아서 거기까지만 사용
                    last_complete = response_text.rfind('},')
                    if last_complete != -1:
                        # 마지막 완전한 객체 다음에 ]를 추가
                        recovered_text = response_text[:last_complete+1] + ']'
                        products = json.loads(recovered_text)
                        
                        # 중복 제거하면서 추가
                        for product in products:
                            product_key = product.get('name', '') + product.get('price', '')
                            if product_key and product_key not in seen_products:
                                seen_products.add(product_key)
                                all_products.append(product)
                        
                        print(f"    → 복구 성공: {len(products)}개 상품 발견 (총 {len(all_products)}개)")
                        continue
                except:
                    pass
                
                # 디버깅을 위해 응답 일부 출력
                print(f"    → 응답 미리보기: {response_text[:200]}...")
                print(f"    → 복구 실패, 이 청크 건너뜀")
                continue
            except Exception as e:
                print(f"    → LLM 호출 오류: {e}")
                continue
        
        print(f"  총 {len(all_products)}개의 고유 상품/서비스를 추출했습니다.")
        
        # 실제 상품 ID 매칭
        if product_id_map:
            from rapidfuzz import fuzz, process
            
            matched_count = 0
            unmatched_products = []
            
            for product in all_products:
                product_name = product.get('name', '').strip()
                if not product_name:
                    continue
                
                matched = False
                
                # 방법 1: 정확히 일치
                if product_name in product_id_map:
                    product['id'] = product_id_map[product_name]
                    matched_count += 1
                    matched = True
                
                # 방법 2: 부분 문자열 포함 (양방향)
                if not matched:
                    for html_name, prd_id in product_id_map.items():
                        # LLM 이름이 HTML 이름에 포함되거나, 그 반대
                        if product_name in html_name or html_name in product_name:
                            product['id'] = prd_id
                            matched_count += 1
                            matched = True
                            break
                
                # 방법 3: Fuzzy matching (부분 일치 우선)
                if not matched:
                    # partial_ratio: 부분 문자열 매칭에 강함
                    best_match = process.extractOne(
                        product_name, 
                        product_id_map.keys(), 
                        scorer=fuzz.partial_ratio,
                        score_cutoff=85
                    )
                    if best_match:
                        matched_name, score, _ = best_match
                        product['id'] = product_id_map[matched_name]
                        matched_count += 1
                        matched = True
                
                # 방법 4: Token sort ratio (단어 순서가 다른 경우)
                if not matched:
                    best_match = process.extractOne(
                        product_name, 
                        product_id_map.keys(), 
                        scorer=fuzz.token_sort_ratio,
                        score_cutoff=85
                    )
                    if best_match:
                        matched_name, score, _ = best_match
                        product['id'] = product_id_map[matched_name]
                        matched_count += 1
                        matched = True
                
                if not matched:
                    unmatched_products.append(product_name[:30])
            
            print(f"  {matched_count}/{len(all_products)}개 상품에 실제 ID 매칭 완료")
            
            if unmatched_products and len(unmatched_products) <= 10:
                print(f"  매칭 실패: {', '.join(unmatched_products[:5])}" + 
                      (f" 외 {len(unmatched_products)-5}개" if len(unmatched_products) > 5 else ""))
        
        return all_products
    
    def extract_product_details_with_llm(self, text_content: str) -> Dict:
        """
        상세 페이지에서 상품 정보를 추출합니다.
        
        Args:
            text_content: 텍스트 내용
            
        Returns:
            상세 정보 딕셔너리
        """
        if not self.use_llm or not self.llm_client:
            return {}
        
        # 텍스트가 너무 길면 앞부분만 사용 (상세 페이지는 보통 처음에 중요 정보가 있음)
        max_length = 8000
        content_for_llm = text_content[:max_length] if len(text_content) > max_length else text_content
        
        if len(text_content) > max_length:
            print(f"    텍스트 길이 {len(text_content)} → {max_length}로 축소")
        
        prompt = f"""다음은 상품/서비스 상세 페이지의 내용입니다. 이 페이지에서 상세 정보를 추출해주세요.

웹 페이지 내용:
---
{content_for_llm}
---

다음 정보를 JSON 형식으로 반환해주세요 (다른 설명 없이):
{{
  "name": "상품/서비스 이름",
  "description": "상세 설명 (200자 이내로 요약)",
  "price": "가격",
  "category": "카테고리",
  "features": ["주요 특징1", "주요 특징2", ...],
  "specifications": {{"스펙키": "스펙값", ...}}
}}

정보가 없는 필드는 빈 문자열이나 빈 배열/객체를 사용하세요.
"""
        
        try:
            # langchain ChatOpenAI 사용
            messages = [HumanMessage(content=prompt)]
            response = self.llm_client.invoke(messages)
            response_text = response.content
            
            # JSON 파싱 전처리
            response_text = re.sub(r'^```json\s*', '', response_text, flags=re.MULTILINE)
            response_text = re.sub(r'^```\s*$', '', response_text, flags=re.MULTILINE)
            response_text = response_text.strip()
            
            # JSON 객체 찾기 (설명이나 주석 제거)
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(0)
            
            details = json.loads(response_text)
            return details
            
        except json.JSONDecodeError as e:
            print(f"  상세 정보 JSON 파싱 오류: {e}")
            print(f"  응답 미리보기: {response_text[:200]}...")
            return {}
        except Exception as e:
            print(f"  상세 정보 추출 오류: {e}")
            return {}
    
    def extract_products_simple(self, html_content: str, links: List[Dict]) -> List[Dict]:
        """
        간단한 규칙 기반으로 상품 링크를 추출합니다.
        
        Args:
            html_content: HTML 내용
            links: 링크 리스트
            
        Returns:
            상품 정보 리스트
        """
        products = []
        
        # 상품 관련 키워드
        product_keywords = ['product', 'item', 'detail', 'goods', '상품', '제품', '서비스']
        
        for idx, link in enumerate(links):
            text = link['text'].lower()
            href = link['href'].lower()
            
            # 상품 관련 링크인지 확인
            if any(keyword in text or keyword in href for keyword in product_keywords):
                products.append({
                    'id': str(idx + 1),
                    'name': link['text'][:100],  # 이름은 100자로 제한
                    'description': '',
                    'price': '',
                    'detail_url': link['href']
                })
        
        print(f"  규칙 기반으로 {len(products)}개의 상품 링크를 찾았습니다.")
        return products
    
    def crawl_list_page(self, url: str, infinite_scroll: bool = True, 
                       scroll_count: int = 10) -> List[Dict]:
        """
        목록 페이지를 크롤링하고 상품 정보를 추출합니다.
        
        Args:
            url: 목록 페이지 URL
            infinite_scroll: 무한 스크롤 여부
            scroll_count: 스크롤 횟수
            
        Returns:
            상품 정보 리스트
        """
        print(f"\n[1단계] 목록 페이지 크롤링")
        result = self.crawl_page(url, infinite_scroll=infinite_scroll, scroll_count=scroll_count)
        
        if not result['success']:
            print(f"크롤링 실패: {result['error']}")
            return []
        
        print(f"  텍스트: {len(result['text_content'])} 문자")
        print(f"  링크: {len(result['links'])} 개")
        
        # 상품 정보 추출
        print(f"\n[2단계] 상품/서비스 정보 추출")
        if self.use_llm:
            products = self.extract_products_with_llm(
                result['html_content'], 
                result['text_content']
            )
        else:
            products = self.extract_products_simple(
                result['html_content'], 
                result['links']
            )
        
        # 상대 URL을 절대 URL로 변환
        for product in products:
            if product.get('detail_url'):
                product['detail_url'] = urljoin(url, product['detail_url'])
        
        return products
    
    def crawl_detail_pages(self, products: List[Dict], max_pages: Optional[int] = None) -> List[Dict]:
        """
        상세 페이지들을 크롤링하고 정보를 업데이트합니다.
        
        Args:
            products: 상품 정보 리스트
            max_pages: 최대 크롤링 페이지 수 (None이면 전체)
            
        Returns:
            업데이트된 상품 정보 리스트
        """
        print(f"\n[3단계] 상세 페이지 크롤링")
        
        # 상세 페이지가 있는 상품만 필터링
        products_with_detail = [p for p in products if p.get('detail_url')]
        
        if not products_with_detail:
            print("  상세 페이지가 있는 상품이 없습니다.")
            return products
        
        # 크롤링할 개수 제한
        if max_pages:
            products_with_detail = products_with_detail[:max_pages]
        
        print(f"  총 {len(products_with_detail)}개의 상세 페이지를 크롤링합니다.")
        
        # 각 상세 페이지 크롤링
        for product in tqdm(products_with_detail, desc="  상세 페이지"):
            detail_url = product['detail_url']
            
            # 상세 페이지 크롤링
            result = self.crawl_page(detail_url, infinite_scroll=False)
            
            if not result['success']:
                continue
            
            # 상세 정보 추출
            if self.use_llm:
                details = self.extract_product_details_with_llm(result['text_content'])
                
                # 기존 정보 업데이트
                if details:
                    product['name'] = details.get('name', product['name'])
                    product['description'] = details.get('description', product.get('description', ''))
                    product['price'] = details.get('price', product.get('price', ''))
                    product['category'] = details.get('category', '')
                    product['features'] = details.get('features', [])
                    product['specifications'] = details.get('specifications', {})
            else:
                # LLM 없이는 텍스트 요약만
                product['description'] = result['text_content'][:500]
        
        return products
    
    def save_to_dataframe(self, products: List[Dict], output_path: str = None) -> pd.DataFrame:
        """
        상품 정보를 DataFrame으로 저장합니다.
        
        Args:
            products: 상품 정보 리스트
            output_path: 저장할 경로 (None이면 저장 안 함)
            
        Returns:
            DataFrame
        """
        print(f"\n[4단계] 데이터 정리 및 저장")
        
        if not products:
            print("  저장할 상품 정보가 없습니다.")
            return pd.DataFrame()
        
        # DataFrame 생성
        df = pd.DataFrame(products)
        
        # 기본 컬럼 확인
        base_columns = ['id', 'name', 'description']
        for col in base_columns:
            if col not in df.columns:
                df[col] = ''
        
        print(f"  {len(df)}개의 상품/서비스 정보를 정리했습니다.")
        print(f"\n  컬럼: {list(df.columns)}")
        
        # 저장
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # CSV로 저장
            csv_file = output_file.with_suffix('.csv')
            df.to_csv(csv_file, index=False, encoding='utf-8-sig')
            print(f"  CSV 저장: {csv_file}")
            
            # JSON으로도 저장
            json_file = output_file.with_suffix('.json')
            df.to_json(json_file, orient='records', force_ascii=False, indent=2)
            print(f"  JSON 저장: {json_file}")
            
            # Excel로도 저장 (openpyxl이 있는 경우)
            try:
                excel_file = output_file.with_suffix('.xlsx')
                df.to_excel(excel_file, index=False, engine='openpyxl')
                print(f"  Excel 저장: {excel_file}")
            except ImportError:
                pass
        
        return df
    
    def run(self, url: str, infinite_scroll: bool = True, scroll_count: int = 10,
            crawl_details: bool = True, max_detail_pages: Optional[int] = None,
            output_path: str = None) -> pd.DataFrame:
        """
        전체 크롤링 프로세스를 실행합니다.
        
        Args:
            url: 시작 URL
            infinite_scroll: 무한 스크롤 여부
            scroll_count: 스크롤 횟수
            crawl_details: 상세 페이지 크롤링 여부
            max_detail_pages: 최대 상세 페이지 수
            output_path: 출력 파일 경로
            
        Returns:
            결과 DataFrame
        """
        print("="*80)
        print("상품/서비스 정보 크롤러")
        print("="*80)
        print(f"URL: {url}")
        print(f"LLM: {'활성화 (' + self.model_name + ')' if self.use_llm else '비활성화'}")
        print(f"무한 스크롤: {infinite_scroll}")
        print(f"상세 페이지: {crawl_details}")
        print("="*80)
        
        # 1. 목록 페이지 크롤링
        products = self.crawl_list_page(url, infinite_scroll=infinite_scroll, 
                                       scroll_count=scroll_count)
        
        if not products:
            print("\n상품 정보를 찾을 수 없습니다.")
            return pd.DataFrame()
        
        print(f"\n추출된 상품/서비스: {len(products)}개")
        
        # 2. 상세 페이지 크롤링 (옵션)
        if crawl_details:
            products = self.crawl_detail_pages(products, max_pages=max_detail_pages)
        
        # 3. DataFrame 저장
        df = self.save_to_dataframe(products, output_path=output_path)
        
        print("\n" + "="*80)
        print("크롤링 완료!")
        print("="*80)
        
        return df


def main():
    parser = argparse.ArgumentParser(
        description='상품/서비스 정보 크롤러',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
사용 예시:
  # 기본 사용 (LLM 활성화 - AX)
  python product_crawler.py "https://m.shop.com/products"
  
  # 무한 스크롤 + 상세 페이지 크롤링
  python product_crawler.py "https://m.shop.com/products" --scroll --details
  
  # 상세 페이지는 최대 10개만
  python product_crawler.py "https://m.shop.com/products" --details --max-details 10
  
  # LLM 없이 사용 (규칙 기반)
  python product_crawler.py "https://m.shop.com/products" --no-llm
  
  # 다른 LLM 모델 사용
  python product_crawler.py "https://m.shop.com/products" --model gemini
  python product_crawler.py "https://m.shop.com/products" --model gpt
        '''
    )
    
    parser.add_argument('url', help='크롤링할 페이지 URL')
    parser.add_argument('--scroll', action='store_true',
                       help='무한 스크롤 활성화')
    parser.add_argument('--scroll-count', type=int, default=10,
                       help='최대 스크롤 횟수 (기본값: 10)')
    parser.add_argument('--details', action='store_true',
                       help='상세 페이지 크롤링')
    parser.add_argument('--max-details', type=int, default=None,
                       help='최대 상세 페이지 크롤링 개수')
    parser.add_argument('--no-llm', action='store_true',
                       help='LLM 사용 안 함 (규칙 기반 추출)')
    parser.add_argument('--model', choices=['gemma', 'ax', 'claude', 'gemini', 'gpt'], 
                       default='ax',
                       help='LLM 모델 (기본값: ax)')
    parser.add_argument('--output', '-o', default='product_data',
                       help='출력 파일 경로 (확장자 제외, 기본값: product_data)')
    
    args = parser.parse_args()
    
    # URL 유효성 검사
    if not args.url.startswith(('http://', 'https://')):
        args.url = 'https://' + args.url
    
    # 크롤러 초기화
    crawler = ProductCrawler(
        base_url=args.url,
        use_llm=not args.no_llm,
        model_name=args.model
    )
    
    # 실행
    df = crawler.run(
        url=args.url,
        infinite_scroll=args.scroll,
        scroll_count=args.scroll_count,
        crawl_details=args.details,
        max_detail_pages=args.max_details,
        output_path=args.output
    )
    
    # 결과 미리보기
    if not df.empty:
        print("\n결과 미리보기:")
        print(df.head(10).to_string())
    
    return 0


if __name__ == '__main__':
    exit(main())

