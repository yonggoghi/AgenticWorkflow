#!/usr/bin/env python3
"""
웹 크롤러 사용 예제

여러 URL을 크롤링하는 예제 코드입니다.
"""

from web_crawler import crawl_with_playwright, save_results


def example_single_url():
    """단일 URL 크롤링 예제"""
    print("\n=== 단일 URL 크롤링 예제 ===\n")
    
    url = "https://m.sktuniverse.co.kr/category/sub/tab/detail?ctanId=CC00000012&ctgId=CA00000001"
    print(f"크롤링 시작: {url}")
    
    result = crawl_with_playwright(url, wait_time=2000, screenshot=False)
    
    if result['success']:
        print(f"\n성공!")
        print(f"제목: {result['title']}")
        print(f"텍스트 길이: {len(result['text_content'])} 문자")
        print(f"링크 수: {len(result['links'])}")
        print(f"이미지 수: {len(result['images'])}")
        
        # 결과 저장
        save_results(result, output_dir='example_output')
    else:
        print(f"\n실패: {result['error']}")


def example_multiple_urls():
    """여러 URL 크롤링 예제"""
    print("\n=== 여러 URL 크롤링 예제 ===\n")
    
    urls = [
        "https://www.python.org",
        "https://github.com",
        "https://stackoverflow.com"
    ]
    
    results = []
    
    for i, url in enumerate(urls, 1):
        print(f"\n[{i}/{len(urls)}] 크롤링: {url}")
        result = crawl_with_playwright(url, wait_time=2000, screenshot=False)
        
        if result['success']:
            print(f"  ✓ 성공: {result['title']}")
            save_results(result, output_dir='example_output')
        else:
            print(f"  ✗ 실패: {result['error']}")
        
        results.append(result)
    
    # 요약
    print("\n=== 크롤링 요약 ===")
    success_count = sum(1 for r in results if r['success'])
    print(f"성공: {success_count}/{len(urls)}")
    print(f"실패: {len(urls) - success_count}/{len(urls)}")


def example_with_link_extraction():
    """링크 추출 예제"""
    print("\n=== 링크 추출 예제 ===\n")
    
    url = "https://news.ycombinator.com"
    print(f"크롤링: {url}")
    
    result = crawl_with_playwright(url, wait_time=2000, screenshot=False)
    
    if result['success']:
        print(f"\n제목: {result['title']}")
        print(f"총 링크 수: {len(result['links'])}\n")
        
        # 상위 10개 링크 출력
        print("상위 10개 링크:")
        for i, link in enumerate(result['links'][:10], 1):
            print(f"{i:2d}. {link['text'][:50]}")
            print(f"    {link['href']}")
    else:
        print(f"실패: {result['error']}")


def example_text_search():
    """텍스트 검색 예제"""
    print("\n=== 텍스트 검색 예제 ===\n")
    
    url = "https://www.python.org"
    search_term = "Python"
    
    print(f"크롤링: {url}")
    print(f"검색어: {search_term}")
    
    result = crawl_with_playwright(url, wait_time=2000, screenshot=False)
    
    if result['success']:
        text = result['text_content']
        count = text.count(search_term)
        
        print(f"\n'{search_term}' 발견 횟수: {count}")
        
        # 검색어가 포함된 첫 번째 문단 찾기
        lines = text.split('\n')
        for line in lines:
            if search_term in line and len(line.strip()) > 10:
                print(f"\n예시 문장:")
                print(f"  {line.strip()[:200]}")
                break
    else:
        print(f"실패: {result['error']}")


def main():
    """메인 함수"""
    print("="*80)
    print("웹 크롤러 예제")
    print("="*80)
    
    # 원하는 예제의 주석을 해제하세요
    
    # 예제 1: 단일 URL 크롤링
    example_single_url()
    
    # 예제 2: 여러 URL 크롤링
    # example_multiple_urls()
    
    # 예제 3: 링크 추출
    # example_with_link_extraction()
    
    # 예제 4: 텍스트 검색
    # example_text_search()
    
    print("\n" + "="*80)
    print("완료!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()

