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
    
    def _chunk_html(self, html_content: str, chunk_size: int = 15000) -> List[str]:
        """
        HTML을 청크로 나누기 - 단순하고 확실한 방식
        
        Args:
            html_content: HTML 내용
            chunk_size: 각 청크의 최대 크기 (문자 수)
            
        Returns:
            HTML 청크 리스트
        """
        if len(html_content) <= chunk_size:
            return [html_content]
        
        from bs4 import BeautifulSoup
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # prdid 속성이 있는 요소들 찾기 (상품 컨테이너)
            product_elements = soup.find_all(attrs={'prdid': True})
            
            if not product_elements or len(product_elements) < 5:
                # prdid가 없으면 class에 'product', 'item' 포함된 요소 찾기
                product_elements = soup.find_all(class_=lambda x: x and 
                    any(word in str(x).lower() for word in ['product', 'item', 'pass', 'card']))
            
            if not product_elements or len(product_elements) < 5:
                # 그래도 없으면 단순 분할
                print(f"  ⚠️ 상품 컨테이너를 찾지 못해 단순 분할합니다")
                chunks = []
                for i in range(0, len(html_content), chunk_size):
                    chunks.append(html_content[i:i+chunk_size])
                return chunks
            
            # 각 상품 요소를 HTML 문자열로 변환
            product_htmls = [str(elem) for elem in product_elements]
            
            # 청크 크기에 맞게 그룹화 (각 청크가 chunk_size를 초과하지 않도록)
            chunks = []
            current_chunk = ""
            
            for prod_html in product_htmls:
                # 단일 상품 HTML이 chunk_size보다 크면 강제로 잘라서 별도 청크로
                if len(prod_html) > chunk_size:
                    if current_chunk:
                        chunks.append(current_chunk)
                        current_chunk = ""
                    # 큰 HTML을 여러 청크로 분할
                    for i in range(0, len(prod_html), chunk_size):
                        chunks.append(prod_html[i:i+chunk_size])
                    continue
                
                # 현재 청크에 추가하면 chunk_size 초과하는지 체크
                if len(current_chunk) + len(prod_html) > chunk_size:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = prod_html
                else:
                    current_chunk += "\n" + prod_html
            
            # 마지막 청크 추가
            if current_chunk:
                chunks.append(current_chunk)
            
            # 디버깅: 청크 크기 확인
            max_chunk_size = max(len(c) for c in chunks) if chunks else 0
            if max_chunk_size > chunk_size * 1.5:
                print(f"  ⚠️ 청크 크기 초과 감지: {max_chunk_size:,}자 (목표: {chunk_size:,}자)")
            
            return chunks if chunks else [html_content]
            
        except Exception as e:
            print(f"  ⚠️ HTML 파싱 오류, 단순 분할: {str(e)[:50]}")
            # 파싱 실패 시 단순 분할
            chunks = []
            for i in range(0, len(html_content), chunk_size):
                chunks.append(html_content[i:i+chunk_size])
            return chunks
    
    def extract_products_with_llm(self, html_content: str, text_content: str) -> List[Dict]:
        """
        LLM을 사용하여 상품/서비스 정보를 추출합니다.
        HTML을 직접 LLM에게 전달하여 ID, 이름, 가격 등을 추출합니다.
        
        Args:
            html_content: HTML 내용
            text_content: 텍스트 내용 (사용 안 함)
            
        Returns:
            상품 정보 리스트
        """
        if not self.use_llm or not self.llm_client:
            return []
        
        # HTML을 청크로 나누기 (상품 컨테이너 단위로)
        html_chunks = self._chunk_html(html_content, chunk_size=15000)
        
        print(f"  HTML을 {len(html_chunks)}개 청크로 분할")
        
        # 디버깅: 청크 크기 분포 출력
        chunk_sizes = [len(c) for c in html_chunks]
        if chunk_sizes:
            print(f"  청크 크기: 최소 {min(chunk_sizes):,}자, 최대 {max(chunk_sizes):,}자, 평균 {sum(chunk_sizes)//len(chunk_sizes):,}자")
        
        all_products = []
        seen_products = set()  # 중복 제거용 (상품명+ID 기준)
        
        # 각 HTML 청크마다 LLM 호출
        for idx, html_chunk in enumerate(html_chunks):
            print(f"  청크 {idx+1}/{len(html_chunks)} 처리 중... ({len(html_chunk)} 문자)")
            
            prompt = f"""다음은 웹 페이지의 HTML입니다. 이 HTML에서 상품 또는 서비스 정보를 추출해주세요.

HTML 구조에서 다음 정보를 찾아 JSON 배열로 반환해주세요:
- id: HTML 속성이나 URL에 있는 상품 고유 식별자 (prdid, data-id, product-id 등의 속성에서 찾기)
- name: 상품/서비스 이름
- description: 상세한 설명 (최대 50자)
- detail_url: 상세 페이지 링크
  * <a> 태그의 href 속성에서 실제 URL 찾기
  * 상대 경로도 그대로 반환 (/detail, ./product 등)
  * javascript:void(0)이거나 링크가 없으면 빈 문자열 ""

HTML:
---
{html_chunk}
---

응답 형식 (JSON만):
[
  {{"id": "PR00000123", "name": "상품명", "description": "할인쿠폰", "detail_url": "/detail?id=PR00000123"}},
  {{"id": "PR00000456", "name": "상품명2", "description": "무료배송", "detail_url": ""}}
]

중요:
1. 완전한 JSON 배열 반환 (마지막 ] 필수)
2. id는 HTML 속성에서 찾은 실제 값 사용
3. detail_url은 href에 실제 URL이 있을 때만 사용
4. 모든 상품 추출
5. JSON만 반환, 설명 금지
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
                
                # 중복 제거하면서 추가 (ID + 이름으로 중복 체크)
                for product in products:
                    product_key = (product.get('id', '') + '|||' + 
                                  product.get('name', ''))
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
                            product_key = (product.get('id', '') + '|||' + 
                                          product.get('name', ''))
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
        
        # ID가 있는 상품 개수 확인
        products_with_id = sum(1 for p in all_products if p.get('id'))
        if products_with_id > 0:
            print(f"  {products_with_id}개 상품에 ID가 포함되어 있습니다.")
        
        # detail_url이 있는 상품 개수 확인
        products_with_detail_url = sum(1 for p in all_products if p.get('detail_url'))
        if products_with_detail_url > 0:
            print(f"  {products_with_detail_url}개 상품에 상세 페이지 URL이 있습니다.")
        
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
            
            # 상대 경로를 절대 경로로 변환
            if detail_url and not detail_url.startswith('http'):
                from urllib.parse import urljoin
                detail_url = urljoin(self.base_url, detail_url)
            
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
        
        # price 컬럼 제거 (필요 없음)
        if 'price' in df.columns:
            df = df.drop(columns=['price'])
        
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

