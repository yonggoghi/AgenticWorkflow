#!/usr/bin/env python3
"""
상세 페이지 URL 패턴 확인 스크립트
브라우저에서 상품을 클릭했을 때 실제 URL을 캡처합니다.
"""

from playwright.sync_api import sync_playwright
import time

def check_detail_url():
    """상품 클릭시 이동하는 URL 확인"""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # 브라우저 표시
        page = browser.new_page()
        
        # 목록 페이지 접속
        url = "https://m.sktuniverse.co.kr/category/sub/tab/detail?ctanId=CC00000012&ctgId=CA00000001"
        print(f"페이지 접속: {url}")
        page.goto(url, wait_until="networkidle")
        
        # 첫 번째 상품 찾기 (PR00000728)
        print("\n첫 번째 상품을 찾는 중...")
        page.wait_for_timeout(2000)
        
        # 상품 클릭 전 URL
        print(f"클릭 전 URL: {page.url}")
        
        # prdid="PR00000728"인 상품의 링크 클릭
        try:
            # 방법 1: inner-link 클릭
            link = page.locator('a.inner-link[prdid="PR00000728"]').first
            if link.is_visible():
                print("\n'T 우주패스 TVING & Wavve 프리미엄' 상품 클릭...")
                link.click()
                page.wait_for_timeout(3000)  # 페이지 로딩 대기
                
                # 클릭 후 URL
                detail_url = page.url
                print(f"상세 페이지 URL: {detail_url}")
                print(f"\nURL 패턴 분석:")
                print(f"  - 전체 URL: {detail_url}")
                if "PR00000728" in detail_url:
                    print(f"  ✅ URL에 상품 ID 포함됨")
                else:
                    print(f"  ⚠️ URL에 상품 ID가 직접 포함되지 않음")
                
                # 다른 상품도 테스트
                print("\n\n다른 상품도 테스트...")
                page.go_back()
                page.wait_for_timeout(2000)
                
                # 두 번째 상품 (PR00000684)
                link2 = page.locator('a.inner-link[prdid="PR00000684"]').first
                if link2.is_visible():
                    print("\n'스피킹맥스' 상품 클릭...")
                    link2.click()
                    page.wait_for_timeout(3000)
                    detail_url2 = page.url
                    print(f"상세 페이지 URL: {detail_url2}")
                    if "PR00000684" in detail_url2:
                        print(f"  ✅ URL에 상품 ID 포함됨")
                
        except Exception as e:
            print(f"오류: {e}")
        
        print("\n\n브라우저를 10초간 유지합니다. 수동으로 상품을 클릭해보세요...")
        page.wait_for_timeout(10000)
        
        browser.close()

if __name__ == "__main__":
    check_detail_url()

