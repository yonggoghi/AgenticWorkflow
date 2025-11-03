#!/usr/bin/env python3
"""
crawl_details 옵션 테스트 스크립트
"""

from product_crawler import ProductCrawler

def test_crawl_details():
    """상세 페이지 크롤링 테스트"""
    print("="*80)
    print("crawl_details 옵션 테스트")
    print("="*80)
    
    url = "https://m.sktuniverse.co.kr/category/sub/tab/detail?ctanId=CC00000012&ctgId=CA00000001"
    
    # 크롤러 초기화
    crawler = ProductCrawler(
        base_url=url,
        use_llm=True,
        model_name="ax"
    )
    
    print("\n[테스트 1] crawl_details=False")
    print("-" * 80)
    df1 = crawler.run(
        url=url,
        infinite_scroll=True,
        scroll_count=5,          # 빠른 테스트를 위해 5회만
        crawl_details=False,     # ❌ 상세 페이지 크롤링 안 함
        output_path="output/test_no_details"
    )
    
    print(f"\n추출된 상품: {len(df1)}개")
    print(f"컬럼: {list(df1.columns)}")
    if not df1.empty:
        print("\n첫 번째 상품:")
        print(df1.iloc[0].to_dict())
        
        # detail_url 통계
        has_detail_url = df1['detail_url'].notna() & (df1['detail_url'] != '')
        print(f"\ndetail_url 통계: {has_detail_url.sum()}/{len(df1)}개 상품에 URL 있음")
    
    print("\n" + "="*80)
    print("\n[테스트 2] crawl_details=True (최대 2개)")
    print("-" * 80)
    df2 = crawler.run(
        url=url,
        infinite_scroll=True,
        scroll_count=5,
        crawl_details=True,      # ✅ 상세 페이지 크롤링
        max_detail_pages=2,      # 2개만 테스트
        output_path="output/test_with_details"
    )
    
    print(f"\n추출된 상품: {len(df2)}개")
    print(f"컬럼: {list(df2.columns)}")
    if not df2.empty:
        print("\n첫 번째 상품:")
        print(df2.iloc[0].to_dict())
        
        # detail_url 통계
        has_detail_url = df2['detail_url'].notna() & (df2['detail_url'] != '')
        print(f"\ndetail_url 통계: {has_detail_url.sum()}/{len(df2)}개 상품에 URL 있음")
    
    print("\n" + "="*80)
    print("테스트 완료!")
    print("="*80)
    
    # 결과 비교
    print("\n[비교 결과]")
    print(f"crawl_details=False: {len(df1)}개 상품")
    print(f"crawl_details=True:  {len(df2)}개 상품")
    
    # detail_url 비교
    if not df1.empty and not df2.empty:
        url_count1 = (df1['detail_url'].notna() & (df1['detail_url'] != '')).sum()
        url_count2 = (df2['detail_url'].notna() & (df2['detail_url'] != '')).sum()
        print(f"\ndetail_url 개수:")
        print(f"  False: {url_count1}개")
        print(f"  True:  {url_count2}개")
    
    if not df2.empty:
        # 상세 정보가 추가되었는지 확인
        detail_fields = ['category', 'features', 'specifications']
        has_details = any(field in df2.columns for field in detail_fields)
        print(f"\n상세 정보 추가 여부: {'✅ 있음' if has_details else '❌ 없음'}")
        
        if has_details:
            for field in detail_fields:
                if field in df2.columns:
                    non_empty = df2[field].notna().sum()
                    print(f"  - {field}: {non_empty}/{len(df2)}개 상품에 데이터 있음")


if __name__ == '__main__':
    test_crawl_details()

