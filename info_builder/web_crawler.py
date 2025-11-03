#!/usr/bin/env python3
"""
웹 크롤러 - URL을 입력받아 웹 페이지 내용을 크롤링합니다.
Playwright를 사용하여 JavaScript로 렌더링되는 동적 페이지도 크롤링 가능합니다.
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("Warning: Playwright is not installed. Install with: pip install playwright && playwright install")


def sanitize_filename(url: str) -> str:
    """URL을 파일명으로 사용 가능한 형태로 변환합니다."""
    parsed = urlparse(url)
    domain = parsed.netloc.replace('www.', '')
    path = parsed.path.strip('/').replace('/', '_')
    
    if path:
        filename = f"{domain}_{path}"
    else:
        filename = domain
    
    # 파일명에 사용 불가능한 문자 제거
    filename = "".join(c for c in filename if c.isalnum() or c in ('_', '-', '.'))
    return filename[:100]  # 파일명 길이 제한


def crawl_with_playwright(url: str, wait_time: int = 2000, screenshot: bool = False, 
                          infinite_scroll: bool = False, scroll_count: int = 10) -> dict:
    """
    Playwright를 사용하여 웹 페이지를 크롤링합니다.
    
    Args:
        url: 크롤링할 URL
        wait_time: 페이지 로딩 대기 시간 (밀리초)
        screenshot: 스크린샷 저장 여부
        infinite_scroll: 무한 스크롤 페이지 여부 (True시 자동 스크롤)
        scroll_count: 스크롤 횟수 (infinite_scroll=True일 때만 사용)
        
    Returns:
        크롤링 결과를 포함한 딕셔너리
    """
    if not PLAYWRIGHT_AVAILABLE:
        raise ImportError("Playwright가 설치되어 있지 않습니다. 'pip install playwright && playwright install' 명령을 실행하세요.")
    
    result = {
        'url': url,
        'timestamp': datetime.now().isoformat(),
        'success': False,
        'title': '',
        'text_content': '',
        'html_content': '',
        'links': [],
        'images': [],
        'error': None,
        'scroll_performed': False,
        'scroll_iterations': 0
    }
    
    try:
        with sync_playwright() as p:
            # 브라우저 실행 (headless 모드)
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            
            # 페이지 방문
            print(f"페이지 로딩 중: {url}")
            page.goto(url, wait_until='networkidle', timeout=30000)
            
            # 추가 대기 (JavaScript 렌더링 완료 대기)
            page.wait_for_timeout(wait_time)
            
            # 무한 스크롤 처리
            if infinite_scroll:
                print(f"무한 스크롤 시작 (최대 {scroll_count}회)")
                result['scroll_performed'] = True
                
                for i in range(scroll_count):
                    # 현재 페이지 높이 저장
                    previous_height = page.evaluate('document.body.scrollHeight')
                    
                    # 페이지 맨 아래로 스크롤
                    page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
                    
                    # 새 콘텐츠 로딩 대기
                    page.wait_for_timeout(1500)
                    
                    # 새로운 높이 확인
                    new_height = page.evaluate('document.body.scrollHeight')
                    
                    result['scroll_iterations'] += 1
                    print(f"  스크롤 {i+1}/{scroll_count}: {previous_height}px → {new_height}px")
                    
                    # 더 이상 새로운 콘텐츠가 없으면 중단
                    if new_height == previous_height:
                        print(f"  더 이상 새로운 콘텐츠 없음 (총 {i+1}회 스크롤)")
                        break
                
                # 최종 대기 (모든 콘텐츠 렌더링 완료)
                page.wait_for_timeout(2000)
            
            # 페이지 정보 수집
            result['title'] = page.title()
            result['text_content'] = page.inner_text('body')
            result['html_content'] = page.content()
            
            # 링크 수집
            links = page.evaluate('''() => {
                return Array.from(document.querySelectorAll('a[href]')).map(a => ({
                    text: a.innerText.trim(),
                    href: a.href
                })).filter(link => link.href && link.text);
            }''')
            result['links'] = links
            
            # 이미지 수집
            images = page.evaluate('''() => {
                return Array.from(document.querySelectorAll('img[src]')).map(img => ({
                    alt: img.alt,
                    src: img.src
                }));
            }''')
            result['images'] = images
            
            # 스크린샷 저장
            if screenshot:
                screenshot_dir = Path('crawler_output/screenshots')
                screenshot_dir.mkdir(parents=True, exist_ok=True)
                screenshot_path = screenshot_dir / f"{sanitize_filename(url)}.png"
                page.screenshot(path=str(screenshot_path), full_page=True)
                result['screenshot_path'] = str(screenshot_path)
                print(f"스크린샷 저장: {screenshot_path}")
            
            result['success'] = True
            print(f"크롤링 완료: {len(result['text_content'])} 문자, {len(links)} 링크, {len(images)} 이미지")
            
            browser.close()
            
    except Exception as e:
        result['error'] = str(e)
        print(f"크롤링 오류: {e}")
    
    return result


def save_results(result: dict, output_dir: str = 'crawler_output'):
    """크롤링 결과를 파일로 저장합니다."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filename = sanitize_filename(result['url'])
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 텍스트 파일 저장
    text_file = output_path / f"{filename}_{timestamp}.txt"
    with open(text_file, 'w', encoding='utf-8') as f:
        f.write(f"URL: {result['url']}\n")
        f.write(f"Title: {result['title']}\n")
        f.write(f"Timestamp: {result['timestamp']}\n")
        f.write(f"Success: {result['success']}\n")
        if result['error']:
            f.write(f"Error: {result['error']}\n")
        f.write("\n" + "="*80 + "\n")
        f.write("텍스트 내용:\n")
        f.write("="*80 + "\n\n")
        f.write(result['text_content'])
    
    print(f"텍스트 저장: {text_file}")
    
    # HTML 파일 저장
    html_file = output_path / f"{filename}_{timestamp}.html"
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(result['html_content'])
    
    print(f"HTML 저장: {html_file}")
    
    # JSON 메타데이터 저장 (링크, 이미지 등)
    json_file = output_path / f"{filename}_{timestamp}_metadata.json"
    metadata = {
        'url': result['url'],
        'title': result['title'],
        'timestamp': result['timestamp'],
        'success': result['success'],
        'error': result['error'],
        'text_length': len(result['text_content']),
        'links_count': len(result['links']),
        'images_count': len(result['images']),
        'scroll_performed': result.get('scroll_performed', False),
        'scroll_iterations': result.get('scroll_iterations', 0),
        'links': result['links'][:50],  # 처음 50개만 저장
        'images': result['images'][:50]  # 처음 50개만 저장
    }
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"메타데이터 저장: {json_file}")
    
    return text_file, html_file, json_file


def main():
    parser = argparse.ArgumentParser(
        description='웹 페이지 크롤러 - URL의 내용을 크롤링합니다.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
사용 예시:
  python web_crawler.py https://example.com
  python web_crawler.py https://example.com --screenshot
  python web_crawler.py https://example.com --scroll --scroll-count 20
  python web_crawler.py https://example.com --wait 5000 --output my_output
  python web_crawler.py https://example.com --no-save
        '''
    )
    
    parser.add_argument('url', help='크롤링할 웹 페이지 URL')
    parser.add_argument('--wait', type=int, default=2000,
                       help='페이지 로딩 대기 시간 (밀리초, 기본값: 2000)')
    parser.add_argument('--screenshot', action='store_true',
                       help='페이지 스크린샷 저장')
    parser.add_argument('--scroll', action='store_true',
                       help='무한 스크롤 페이지 자동 스크롤 (동적 로딩 콘텐츠)')
    parser.add_argument('--scroll-count', type=int, default=10,
                       help='최대 스크롤 횟수 (기본값: 10)')
    parser.add_argument('--output', '-o', default='crawler_output',
                       help='출력 디렉토리 (기본값: crawler_output)')
    parser.add_argument('--no-save', action='store_true',
                       help='파일로 저장하지 않고 화면에만 출력')
    parser.add_argument('--print-text', action='store_true',
                       help='크롤링한 텍스트를 화면에 출력')
    
    args = parser.parse_args()
    
    # URL 유효성 검사
    if not args.url.startswith(('http://', 'https://')):
        args.url = 'https://' + args.url
    
    print(f"\n{'='*80}")
    print(f"웹 크롤러 시작")
    print(f"{'='*80}")
    print(f"URL: {args.url}")
    print(f"대기 시간: {args.wait}ms")
    if args.scroll:
        print(f"무한 스크롤: 활성화 (최대 {args.scroll_count}회)")
    print(f"{'='*80}\n")
    
    # 크롤링 실행
    result = crawl_with_playwright(
        args.url, 
        wait_time=args.wait, 
        screenshot=args.screenshot,
        infinite_scroll=args.scroll,
        scroll_count=args.scroll_count
    )
    
    if not result['success']:
        print(f"\n크롤링 실패: {result['error']}")
        return 1
    
    # 결과 저장
    if not args.no_save:
        print(f"\n결과 저장 중...")
        text_file, html_file, json_file = save_results(result, args.output)
        print(f"\n{'='*80}")
        print(f"크롤링 완료!")
        print(f"{'='*80}")
        print(f"저장된 파일:")
        print(f"  - 텍스트: {text_file}")
        print(f"  - HTML: {html_file}")
        print(f"  - 메타데이터: {json_file}")
        if result.get('screenshot_path'):
            print(f"  - 스크린샷: {result['screenshot_path']}")
        print(f"{'='*80}\n")
    
    # 텍스트 출력
    if args.print_text or args.no_save:
        print(f"\n{'='*80}")
        print(f"크롤링된 텍스트 내용:")
        print(f"{'='*80}\n")
        print(result['text_content'][:2000])  # 처음 2000자만 출력
        if len(result['text_content']) > 2000:
            print(f"\n... (총 {len(result['text_content'])} 문자)")
        print(f"\n{'='*80}\n")
    
    # 통계 출력
    print(f"통계:")
    print(f"  - 제목: {result['title']}")
    print(f"  - 텍스트 길이: {len(result['text_content'])} 문자")
    print(f"  - 링크 수: {len(result['links'])}")
    print(f"  - 이미지 수: {len(result['images'])}")
    if result.get('scroll_performed'):
        print(f"  - 스크롤 횟수: {result['scroll_iterations']}회")
    
    return 0


if __name__ == '__main__':
    exit(main())

