#!/usr/bin/env python3
"""
페이지 타입 자동 감지 모듈

페이지네이션 방식을 자동으로 감지:
- infinite_scroll: 무한 스크롤
- pagination: 페이지 번호 버튼
- load_more: "더보기" 버튼
- static: 정적 페이지 (모든 콘텐츠 로드됨)
"""

from typing import Literal, Dict, List
from playwright.sync_api import Page

PageType = Literal['infinite_scroll', 'pagination', 'load_more', 'static']


class PageTypeDetector:
    """페이지 타입 자동 감지"""
    
    # 페이지네이션 감지용 selector 패턴
    PAGINATION_SELECTORS = [
        'nav[role="navigation"]',
        'ul.pagination',
        'div.pagination',
        '.pager',
        '.page-numbers',
        'a[aria-label*="next"]',
        'a[aria-label*="Next"]',
        'button[aria-label*="next"]',
        'a:has-text("다음")',
        'a:has-text("Next")',
        'a:has-text(">")',
        'button:has-text("다음")',
    ]
    
    # "더보기" 버튼 감지용 selector 패턴
    LOAD_MORE_SELECTORS = [
        'button:has-text("더보기")',
        'button:has-text("더 보기")',
        'button:has-text("Load More")',
        'button:has-text("Show More")',
        'a:has-text("더보기")',
        'a:has-text("Load More")',
        '.load-more',
        '.show-more',
        '#load-more',
    ]
    
    @staticmethod
    def detect(page: Page, verbose: bool = False) -> Dict:
        """
        페이지 타입을 자동으로 감지합니다.
        
        Args:
            page: Playwright Page 객체
            verbose: 상세 로그 출력 여부
            
        Returns:
            {
                'type': 페이지 타입,
                'confidence': 확신도 (0-1),
                'details': 감지 상세 정보
            }
        """
        if verbose:
            print("🔍 페이지 타입 감지 시작...")
        
        results = []
        
        # 1. 페이지네이션 감지
        pagination_result = PageTypeDetector._detect_pagination(page, verbose)
        results.append(pagination_result)
        
        # 2. "더보기" 버튼 감지
        load_more_result = PageTypeDetector._detect_load_more(page, verbose)
        results.append(load_more_result)
        
        # 3. 무한 스크롤 감지
        infinite_scroll_result = PageTypeDetector._detect_infinite_scroll(page, verbose)
        results.append(infinite_scroll_result)
        
        # 4. 정적 페이지 감지 (기본값)
        static_result = {
            'type': 'static',
            'confidence': 0.3,  # 낮은 기본 확신도
            'details': '다른 패턴이 감지되지 않음'
        }
        results.append(static_result)
        
        # 가장 확신도가 높은 결과 선택
        best_result = max(results, key=lambda x: x['confidence'])
        
        if verbose:
            print(f"✅ 감지 결과: {best_result['type']} (확신도: {best_result['confidence']:.2f})")
            print(f"   상세: {best_result['details']}")
        
        return best_result
    
    @staticmethod
    def _detect_pagination(page: Page, verbose: bool = False) -> Dict:
        """페이지네이션 버튼 감지"""
        if verbose:
            print("  📄 페이지네이션 확인 중...")
        
        found_selectors = []
        
        for selector in PageTypeDetector.PAGINATION_SELECTORS:
            try:
                count = page.locator(selector).count()
                if count > 0:
                    found_selectors.append(selector)
                    if verbose:
                        print(f"    ✓ '{selector}': {count}개 발견")
            except:
                pass
        
        if found_selectors:
            return {
                'type': 'pagination',
                'confidence': min(0.95, 0.7 + len(found_selectors) * 0.1),
                'details': f'{len(found_selectors)}개 페이지네이션 요소 발견'
            }
        
        return {
            'type': 'pagination',
            'confidence': 0.0,
            'details': '페이지네이션 요소 없음'
        }
    
    @staticmethod
    def _detect_load_more(page: Page, verbose: bool = False) -> Dict:
        """'더보기' 버튼 감지"""
        if verbose:
            print("  🔘 '더보기' 버튼 확인 중...")
        
        found_selectors = []
        
        for selector in PageTypeDetector.LOAD_MORE_SELECTORS:
            try:
                count = page.locator(selector).count()
                if count > 0:
                    found_selectors.append(selector)
                    if verbose:
                        print(f"    ✓ '{selector}': {count}개 발견")
            except:
                pass
        
        if found_selectors:
            return {
                'type': 'load_more',
                'confidence': min(0.95, 0.7 + len(found_selectors) * 0.1),
                'details': f'{len(found_selectors)}개 "더보기" 버튼 발견'
            }
        
        return {
            'type': 'load_more',
            'confidence': 0.0,
            'details': '"더보기" 버튼 없음'
        }
    
    @staticmethod
    def _detect_infinite_scroll(page: Page, verbose: bool = False) -> Dict:
        """무한 스크롤 감지 (실제 스크롤 테스트)"""
        if verbose:
            print("  ♾️  무한 스크롤 테스트 중...")
        
        try:
            # 초기 높이 측정
            initial_height = page.evaluate('document.body.scrollHeight')
            initial_item_count = page.locator('*[id], *[class*="item"], *[class*="product"], *[class*="card"]').count()
            
            if verbose:
                print(f"    초기 높이: {initial_height}px, 아이템: {initial_item_count}개")
            
            # 스크롤 다운
            page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
            page.wait_for_timeout(2000)  # 콘텐츠 로딩 대기
            
            # 변경 후 측정
            new_height = page.evaluate('document.body.scrollHeight')
            new_item_count = page.locator('*[id], *[class*="item"], *[class*="product"], *[class*="card"]').count()
            
            if verbose:
                print(f"    스크롤 후: {new_height}px, 아이템: {new_item_count}개")
            
            # 높이 또는 아이템 수 증가 확인
            height_increased = new_height > initial_height
            items_increased = new_item_count > initial_item_count
            
            if height_increased or items_increased:
                confidence = 0.9 if (height_increased and items_increased) else 0.7
                changes = []
                if height_increased:
                    changes.append(f"높이 {initial_height}→{new_height}")
                if items_increased:
                    changes.append(f"아이템 {initial_item_count}→{new_item_count}")
                
                if verbose:
                    print(f"    ✓ 무한 스크롤 감지: {', '.join(changes)}")
                
                return {
                    'type': 'infinite_scroll',
                    'confidence': confidence,
                    'details': f'스크롤 시 콘텐츠 증가 ({", ".join(changes)})'
                }
            else:
                if verbose:
                    print(f"    ✗ 스크롤 시 변화 없음")
                
                return {
                    'type': 'infinite_scroll',
                    'confidence': 0.0,
                    'details': '스크롤 시 콘텐츠 변화 없음'
                }
                
        except Exception as e:
            if verbose:
                print(f"    ⚠️  테스트 오류: {str(e)[:50]}")
            
            return {
                'type': 'infinite_scroll',
                'confidence': 0.0,
                'details': f'테스트 실패: {str(e)[:50]}'
            }
    
    @staticmethod
    def get_scroll_strategy(page_type: PageType) -> Dict:
        """페이지 타입에 따른 스크롤 전략 반환"""
        strategies = {
            'infinite_scroll': {
                'should_scroll': True,
                'scroll_count': 10,
                'scroll_delay': 2000,
                'need_rescroll_after_back': True,
                'description': '무한 스크롤: 여러 번 스크롤 필요, 뒤로 가기 후 재스크롤'
            },
            'pagination': {
                'should_scroll': False,
                'scroll_count': 0,
                'scroll_delay': 0,
                'need_rescroll_after_back': False,
                'description': '페이지네이션: 스크롤 불필요'
            },
            'load_more': {
                'should_scroll': False,  # 스크롤 대신 버튼 클릭
                'scroll_count': 0,
                'scroll_delay': 0,
                'need_rescroll_after_back': False,
                'description': '"더보기" 버튼: 버튼 클릭 방식'
            },
            'static': {
                'should_scroll': True,
                'scroll_count': 2,
                'scroll_delay': 1000,
                'need_rescroll_after_back': False,
                'description': '정적 페이지: 가벼운 스크롤만'
            }
        }
        
        return strategies.get(page_type, strategies['static'])


def test_detector():
    """테스트 함수"""
    from playwright.sync_api import sync_playwright
    
    test_urls = [
        ('https://m.sktuniverse.co.kr/category/sub/tab/detail?ctanId=CC00000012&ctgId=CA00000001', '무한 스크롤'),
        # 추가 테스트 URL을 여기에
    ]
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        for url, expected_type in test_urls:
            print(f"\n{'='*80}")
            print(f"테스트 URL: {url}")
            print(f"예상 타입: {expected_type}")
            print('='*80)
            
            page.goto(url, wait_until='networkidle', timeout=30000)
            page.wait_for_timeout(2000)
            
            result = PageTypeDetector.detect(page, verbose=True)
            
            strategy = PageTypeDetector.get_scroll_strategy(result['type'])
            print(f"\n📋 권장 전략:")
            print(f"  {strategy['description']}")
            print(f"  스크롤 필요: {strategy['should_scroll']}")
            print(f"  뒤로 가기 후 재스크롤: {strategy['need_rescroll_after_back']}")
        
        browser.close()


if __name__ == '__main__':
    test_detector()

