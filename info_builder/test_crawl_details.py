#!/usr/bin/env python3
"""
crawl_details 옵션 테스트 스크립트

🆕 페이지 타입 자동 감지 기능 추가!
"""

from product_crawler import ProductCrawler

def test_crawl_details():
    """상세 페이지 크롤링 테스트 (자동 감지 활용)"""
    print("="*80)
    print("crawl_details 옵션 테스트")
    print("🆕 페이지 타입 자동 감지 사용")
    print("="*80)
    
    url = "https://m.sktuniverse.co.kr/category/sub/tab/detail?ctanId=CC00000012&ctgId=CA00000001"
    
    # 크롤러 초기화
    crawler = ProductCrawler(
        base_url=url,
        use_llm=True,
        model_name="ax"
    )
    
    # print("\n[테스트 1] crawl_details=False")
    # print("-" * 80)
    # df1 = crawler.run(
    #     url=url,
    #     infinite_scroll=True,
    #     scroll_count=5,          # 빠른 테스트를 위해 5회만
    #     crawl_details=False,     # ❌ 상세 페이지 크롤링 안 함
    #     output_path="output/test_no_details"
    # )
    
    # print(f"\n추출된 상품: {len(df1)}개")
    # print(f"컬럼: {list(df1.columns)}")
    # if not df1.empty:
    #     print("\n첫 번째 상품:")
    #     print(df1.iloc[0].to_dict())
        
    #     # detail_url 통계
    #     has_detail_url = df1['detail_url'].notna() & (df1['detail_url'] != '')
    #     print(f"\ndetail_url 통계: {has_detail_url.sum()}/{len(df1)}개 상품에 URL 있음")
    
    print("\n" + "="*80)
    print("\n[테스트] crawl_details=True (최대 2개, 자동 감지)")
    print("-" * 80)
    df2 = crawler.run(
        url=url,
        auto_detect=True,        # 🆕 페이지 타입 자동 감지
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
    
    # 상세 정보 확인
    print("\n[결과 분석]")
    print(f"추출된 상품: {len(df2)}개")
    
    if not df2.empty:
        # 상세 정보가 추가되었는지 확인
        detail_fields = ['category', 'features', 'specifications']
        has_details = any(field in df2.columns for field in detail_fields)
        print(f"\n상세 정보 추가 여부: {'✅ 있음' if has_details else '❌ 없음'}")
        
        if has_details:
            print("\n상세 정보 필드별 데이터:")
            for field in detail_fields:
                if field in df2.columns:
                    non_empty = df2[field].notna().sum()
                    print(f"  - {field}: {non_empty}/{len(df2)}개 상품에 데이터 있음")
        
        print(f"\n💡 자동 감지 기능:")
        print("   - 페이지 타입이 자동으로 감지되었습니다")
        print("   - 최적의 스크롤 전략이 적용되었습니다")
        print("   - 무한 스크롤 페이지: 재스크롤 필요")
        print("   - 일반 페이지: 재스크롤 생략 → 2배 빠름!")


if __name__ == '__main__':
    test_crawl_details()

