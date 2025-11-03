#!/usr/bin/env python3
"""
상품/서비스 크롤러 사용 예제
"""

from product_crawler import ProductCrawler
import pandas as pd


def example_basic():
    """기본 사용 예제"""
    print("\n=== 기본 사용 예제 ===\n")
    
    url = "https://m.sktuniverse.co.kr/category/sub/tab/detail?ctanId=CC00000012&ctgId=CA00000001"
    
    # 크롤러 초기화
    crawler = ProductCrawler(
        base_url=url,
        use_llm=True,           # LLM 사용
        model_name="ax"         # AX 모델 사용 (gemma, ax, claude, gemini, gpt 중 선택)
    )
    
    # 실행
    df = crawler.run(
        url=url,
        infinite_scroll=True,    # 무한 스크롤
        scroll_count=30,         # 최대 30회 스크롤 (182개 상품을 모두 로드하기 위해)
        crawl_details=False,     # 상세 페이지는 크롤링 안 함
        output_path="output/products_basic"
    )
    
    print(f"\n추출된 상품: {len(df)}개")
    print("\n상위 5개:")
    print(df.head())


def example_with_details():
    """상세 페이지 포함 예제"""
    print("\n=== 상세 페이지 포함 예제 ===\n")
    
    url = "https://m.sktuniverse.co.kr/category/sub/tab/detail?ctanId=CC00000012&ctgId=CA00000001"
    
    crawler = ProductCrawler(
        base_url=url,
        use_llm=True,
        model_name="ax"
    )
    
    # 실행
    df = crawler.run(
        url=url,
        infinite_scroll=True,
        scroll_count=10,         # 적은 스크롤로 빠르게 테스트
        crawl_details=True,      # 상세 페이지 크롤링 ✅
        max_detail_pages=3,      # 최대 3개만 테스트
        output_path="output/products_with_details"
    )
    
    print(f"\n추출된 상품: {len(df)}개")
    print("\n상위 5개 (상세 정보 포함):")
    print(df.head())


def example_no_llm():
    """LLM 없이 사용하는 예제"""
    print("\n=== LLM 없이 사용 (규칙 기반) ===\n")
    
    url = "https://example.com/products"
    
    crawler = ProductCrawler(
        base_url=url,
        use_llm=False  # LLM 사용 안 함
    )
    
    # 실행
    df = crawler.run(
        url=url,
        infinite_scroll=True,
        scroll_count=10,
        crawl_details=False,
        output_path="output/products_no_llm"
    )
    
    print(f"\n추출된 상품 링크: {len(df)}개")


def example_manual_control():
    """수동 제어 예제"""
    print("\n=== 수동 제어 예제 ===\n")
    
    url = "https://example.com/products"
    
    crawler = ProductCrawler(
        base_url=url,
        use_llm=True,
        model_name="gemini"  # Gemini 모델 사용 예시
    )
    
    # 1단계: 목록 페이지만 크롤링
    products = crawler.crawl_list_page(url, infinite_scroll=True, scroll_count=10)
    print(f"추출된 상품: {len(products)}개")
    
    # 2단계: 특정 상품들만 상세 페이지 크롤링
    selected_products = products[:3]  # 처음 3개만
    products_with_details = crawler.crawl_detail_pages(selected_products)
    
    # 3단계: DataFrame으로 저장
    df = crawler.save_to_dataframe(products_with_details, output_path="output/products_manual")
    
    print(f"\n최종 데이터: {len(df)}개")
    print(df)


def example_data_analysis():
    """데이터 분석 예제"""
    print("\n=== 데이터 분석 예제 ===\n")
    
    # 저장된 데이터 로드
    df = pd.read_csv("output/products_basic.csv")
    
    print(f"총 상품 수: {len(df)}")
    print(f"\n컬럼: {list(df.columns)}")
    
    # 가격이 있는 상품만 필터링
    if 'price' in df.columns:
        df_with_price = df[df['price'].notna() & (df['price'] != '')]
        print(f"가격 정보가 있는 상품: {len(df_with_price)}개")
    
    # 상세 URL이 있는 상품
    if 'detail_url' in df.columns:
        df_with_url = df[df['detail_url'].notna() & (df['detail_url'] != '')]
        print(f"상세 페이지가 있는 상품: {len(df_with_url)}개")
    
    # 이름 길이 분석
    if 'name' in df.columns:
        df['name_length'] = df['name'].str.len()
        print(f"\n평균 상품명 길이: {df['name_length'].mean():.1f}자")


def main():
    """메인 함수"""
    print("="*80)
    print("상품/서비스 크롤러 예제")
    print("="*80)
    
    # 원하는 예제의 주석을 해제하세요
    
    # 예제 1: 기본 사용
    example_basic()
    
    # 예제 2: 상세 페이지 포함
    # example_with_details()
    
    # 예제 3: LLM 없이 사용
    # example_no_llm()
    
    # 예제 4: 수동 제어
    # example_manual_control()
    
    # 예제 5: 데이터 분석
    # example_data_analysis()
    
    print("\n" + "="*80)
    print("완료!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()

