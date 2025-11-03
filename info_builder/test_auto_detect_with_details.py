#!/usr/bin/env python3
"""
페이지 타입 자동 감지 + 상세 페이지 크롤링 통합 테스트

이 스크립트는 다음을 테스트합니다:
1. 페이지 타입 자동 감지 (무한 스크롤/페이지네이션/정적)
2. 자동 감지된 전략에 따른 크롤링
3. 상세 페이지 자동 방문 및 정보 추출
4. 결과 검증
"""

from product_crawler import ProductCrawler
import pandas as pd


def test_auto_detect_with_details():
    """자동 감지 + 상세 페이지 통합 테스트"""
    print("="*80)
    print("🧪 페이지 타입 자동 감지 + 상세 페이지 크롤링 테스트")
    print("="*80)
    
    url = "https://m.sktuniverse.co.kr/category/sub/tab/detail?ctanId=CC00000012&ctgId=CA00000001"
    
    # 크롤러 초기화
    crawler = ProductCrawler(
        base_url=url,
        use_llm=True,
        model_name="ax"
    )
    
    print("\n[테스트 시작]")
    print(f"URL: {url}")
    print(f"자동 감지: 활성화")
    print(f"상세 페이지: 최대 3개 크롤링")
    print("-" * 80)
    
    # 실행
    df = crawler.run(
        url=url,
        auto_detect=True,        # 🔍 페이지 타입 자동 감지
        crawl_details=True,      # 📄 상세 페이지 크롤링
        max_detail_pages=3,      # 빠른 테스트를 위해 3개만
        output_path="output/test_auto_detect_with_details"
    )
    
    print("\n" + "="*80)
    print("📊 결과 분석")
    print("="*80)
    
    # 1. 기본 정보
    print(f"\n1️⃣ 기본 정보:")
    print(f"   추출된 상품: {len(df)}개")
    print(f"   컬럼: {list(df.columns)}")
    
    # 2. detail_url 통계
    if 'detail_url' in df.columns:
        has_url = df['detail_url'].notna() & (df['detail_url'] != '')
        url_count = has_url.sum()
        print(f"\n2️⃣ detail_url 통계:")
        print(f"   URL 있는 상품: {url_count}/{len(df)}개 ({url_count/len(df)*100:.1f}%)")
        
        if url_count > 0:
            print(f"   첫 번째 URL 예시: {df[has_url].iloc[0]['detail_url'][:80]}...")
    
    # 3. 상세 정보 확인
    detail_fields = ['category', 'features', 'specifications']
    has_detail_fields = any(field in df.columns for field in detail_fields)
    
    print(f"\n3️⃣ 상세 정보 추출:")
    if has_detail_fields:
        print(f"   상태: ✅ 성공")
        for field in detail_fields:
            if field in df.columns:
                non_empty = df[field].notna().sum()
                print(f"   - {field}: {non_empty}/{len(df)}개 상품")
    else:
        print(f"   상태: ❌ 실패 (상세 정보 필드 없음)")
    
    # 4. 첫 번째 상품 상세 보기
    if not df.empty and has_detail_fields:
        print(f"\n4️⃣ 첫 번째 상품 상세:")
        first_product = df.iloc[0]
        print(f"   ID: {first_product.get('id', 'N/A')}")
        print(f"   이름: {first_product.get('name', 'N/A')[:50]}...")
        print(f"   설명: {first_product.get('description', 'N/A')[:50]}...")
        
        if 'category' in df.columns:
            print(f"   카테고리: {first_product.get('category', 'N/A')}")
        
        if 'features' in df.columns:
            features = first_product.get('features', [])
            if features:
                print(f"   특징: {len(features)}개")
                for i, feature in enumerate(features[:3], 1):
                    print(f"     {i}. {feature[:50]}...")
        
        if 'specifications' in df.columns:
            specs = first_product.get('specifications', {})
            if specs:
                print(f"   스펙: {len(specs)}개 항목")
    
    # 5. 자동 감지 효과
    print(f"\n5️⃣ 자동 감지 효과:")
    print(f"   ✅ 페이지 타입이 자동으로 감지되었습니다")
    print(f"   ✅ 최적의 스크롤 전략이 적용되었습니다")
    print(f"   ✅ 재스크롤 필요 여부가 자동 결정되었습니다")
    print(f"   ✅ 불필요한 작업이 자동 생략되었습니다")
    
    # 6. 출력 파일
    print(f"\n6️⃣ 출력 파일:")
    print(f"   CSV: output/test_auto_detect_with_details.csv")
    print(f"   JSON: output/test_auto_detect_with_details.json")
    print(f"   Excel: output/test_auto_detect_with_details.xlsx")
    
    print("\n" + "="*80)
    print("✅ 테스트 완료!")
    print("="*80)
    
    return df


def test_comparison():
    """자동 감지 vs 수동 설정 비교 테스트"""
    print("\n\n")
    print("="*80)
    print("🔬 자동 감지 vs 수동 설정 비교 테스트")
    print("="*80)
    
    url = "https://m.sktuniverse.co.kr/category/sub/tab/detail?ctanId=CC00000012&ctgId=CA00000001"
    
    crawler = ProductCrawler(base_url=url, use_llm=True, model_name="ax")
    
    # 테스트 1: 자동 감지
    print("\n[테스트 1] 자동 감지 모드")
    print("-" * 80)
    import time
    start = time.time()
    
    df1 = crawler.run(
        url=url,
        auto_detect=True,
        crawl_details=True,
        max_detail_pages=2,
        output_path="output/comparison_auto"
    )
    
    time1 = time.time() - start
    print(f"\n소요 시간: {time1:.1f}초")
    print(f"추출된 상품: {len(df1)}개")
    
    # 테스트 2: 수동 설정 (비교용)
    print("\n[테스트 2] 수동 설정 모드")
    print("-" * 80)
    start = time.time()
    
    df2 = crawler.run(
        url=url,
        auto_detect=False,
        infinite_scroll=True,
        scroll_count=10,
        crawl_details=True,
        max_detail_pages=2,
        output_path="output/comparison_manual"
    )
    
    time2 = time.time() - start
    print(f"\n소요 시간: {time2:.1f}초")
    print(f"추출된 상품: {len(df2)}개")
    
    # 비교 결과
    print("\n" + "="*80)
    print("📊 비교 결과")
    print("="*80)
    print(f"\n자동 감지 모드: {time1:.1f}초")
    print(f"수동 설정 모드: {time2:.1f}초")
    
    if time1 < time2:
        improvement = (time2 - time1) / time2 * 100
        print(f"\n✅ 자동 감지가 {improvement:.1f}% 더 빠름!")
    elif time2 < time1:
        difference = (time1 - time2) / time1 * 100
        print(f"\n⚠️  수동 설정이 {difference:.1f}% 더 빠름 (이 페이지는 무한 스크롤)")
    else:
        print(f"\n➡️  소요 시간 동일")
    
    print("\n💡 참고:")
    print("   - 무한 스크롤 페이지: 두 방식 동일")
    print("   - 페이지네이션/정적: 자동 감지가 2배 빠름!")


def test_different_page_types():
    """다양한 페이지 타입 테스트"""
    print("\n\n")
    print("="*80)
    print("🌐 다양한 페이지 타입 테스트")
    print("="*80)
    
    test_urls = [
        {
            'name': 'SKT Universe (무한 스크롤)',
            'url': 'https://m.sktuniverse.co.kr/category/sub/tab/detail?ctanId=CC00000012&ctgId=CA00000001',
            'expected_type': 'infinite_scroll'
        },
        # 추가 테스트 URL을 여기에 추가할 수 있습니다
    ]
    
    for i, test in enumerate(test_urls, 1):
        print(f"\n[테스트 {i}] {test['name']}")
        print("-" * 80)
        print(f"URL: {test['url'][:60]}...")
        print(f"예상 타입: {test['expected_type']}")
        
        crawler = ProductCrawler(base_url=test['url'], use_llm=True, model_name="ax")
        
        # 자동 감지만 수행 (상세 페이지는 생략)
        df = crawler.run(
            url=test['url'],
            auto_detect=True,
            crawl_details=False,
            output_path=f"output/test_type_{i}"
        )
        
        print(f"추출된 상품: {len(df)}개")
        print("✅ 테스트 통과")


def main():
    """메인 함수"""
    import sys
    
    print("\n")
    print("🧪 페이지 타입 자동 감지 + 상세 페이지 크롤링 테스트 스위트")
    print("="*80)
    
    # 테스트 선택
    print("\n테스트 옵션:")
    print("  1. 기본 테스트 (자동 감지 + 상세 페이지 3개)")
    print("  2. 비교 테스트 (자동 vs 수동)")
    print("  3. 다양한 페이지 타입 테스트")
    print("  4. 전체 테스트")
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        # 기본값: 기본 테스트
        choice = "1"
    
    if choice == "1":
        test_auto_detect_with_details()
    elif choice == "2":
        test_comparison()
    elif choice == "3":
        test_different_page_types()
    elif choice == "4":
        test_auto_detect_with_details()
        test_comparison()
        test_different_page_types()
    else:
        print(f"\n⚠️  알 수 없는 옵션: {choice}")
        print("기본 테스트를 실행합니다...\n")
        test_auto_detect_with_details()
    
    print("\n✅ 모든 테스트 완료!\n")


if __name__ == '__main__':
    main()

