# 일반화 개선 계획

## 🎯 목표
현재 방식을 다양한 웹사이트에 적용 가능하도록 개선

## 📊 현재 상태 분석

### ✅ 잘 작동하는 부분 (유지)
1. Playwright 동적 콘텐츠 처리
2. LLM 기반 정보 추출
3. 다중 selector fallback
4. 클릭 → URL 캡처 메커니즘

### ⚠️ 개선 필요한 부분

#### 1. 무한 스크롤 재실행 (가장 큰 문제)
**현재**: 매번 5회 스크롤 (상품당 2.5초 추가)
**문제**: 
- 일반 페이지에서 불필요
- 180개 상품: 7.5분 낭비

**해결 방법 3가지:**

##### A. 페이지 타입 자동 감지 (권장)
```python
def detect_infinite_scroll(page):
    """페이지가 무한 스크롤인지 감지"""
    # 방법 1: 페이지네이션 버튼 확인
    if page.locator('button.pagination, a.next-page').count() > 0:
        return False  # 일반 페이지
    
    # 방법 2: 스크롤 테스트
    initial_height = page.evaluate('document.body.scrollHeight')
    page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
    page.wait_for_timeout(1000)
    new_height = page.evaluate('document.body.scrollHeight')
    
    return new_height > initial_height  # 무한 스크롤

# 사용
is_infinite = detect_infinite_scroll(list_page)
if is_infinite:
    # 무한 스크롤 재실행
else:
    # 뒤로 가기만
```

##### B. 선택적 재스크롤
```python
def smart_scroll_if_needed(page, target_id):
    """필요할 때만 스크롤"""
    # selector가 존재하는지 확인
    if page.locator(f'[prdid="{target_id}"]').count() == 0:
        # 없으면 스크롤
        for i in range(scroll_count):
            page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
            page.wait_for_timeout(500)
            if page.locator(f'[prdid="{target_id}"]').count() > 0:
                break  # 찾으면 중단
```

##### C. URL 패턴 학습 (가장 빠름)
```python
def learn_url_pattern(captured_urls):
    """성공한 URL에서 패턴 추출"""
    # 예: https://site.com/product/detail?prdId=PR00000538
    #  → https://site.com/product/detail?prdId={ID}
    
    if len(captured_urls) >= 3:
        # 패턴 분석
        common_pattern = extract_pattern(captured_urls)
        return common_pattern
    return None

# 사용
url_pattern = learn_url_pattern(url_mapping.values())
if url_pattern:
    # 클릭 없이 URL 생성
    detail_url = url_pattern.format(ID=prd_id)
else:
    # 클릭 방식
```

#### 2. Selector 일반화

**현재**: 특정 속성명 하드코딩
```python
f'a[prdid="{prd_id}"]'  # prdid만
```

**개선**: 다양한 속성명 시도
```python
def generate_selectors(prd_id):
    """여러 속성명 패턴 생성"""
    attr_names = ['prdid', 'data-id', 'product-id', 'id', 'data-product-id']
    tag_names = ['a', 'div', 'button']
    
    selectors = []
    for attr in attr_names:
        for tag in tag_names:
            selectors.append(f'{tag}[{attr}="{prd_id}"]')
    
    return selectors
```

#### 3. 성능 최적화

**현재**: 순차 처리 (9개 × 5초 = 45초)

**개선 1**: 병렬 처리
```python
from concurrent.futures import ThreadPoolExecutor

def capture_urls_parallel(product_ids, max_workers=3):
    """병렬로 URL 캡처"""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 각 워커가 별도 브라우저 사용
        futures = []
        for prd_id in product_ids:
            future = executor.submit(capture_single_url, prd_id)
            futures.append(future)
        
        results = [f.result() for f in futures]
    return results
```

**개선 2**: 탭 재사용
```python
# 뒤로 가기 대신 탭 닫기
detail_page = browser.new_page()
detail_page.goto(detail_url)
url = detail_page.url
detail_page.close()  # 목록 페이지는 그대로
```

## 📋 구현 우선순위

### Phase 1: 즉시 적용 가능 (1-2시간)
- [ ] **URL 패턴 학습** (가장 효과적)
  - 3개 성공 후 패턴 추출
  - 나머지는 패턴으로 생성
  - 예상 속도: 45초 → **10초**

### Phase 2: 단기 개선 (1일)
- [ ] **선택적 재스크롤**
  - 요소 없을 때만 스크롤
  - 예상 속도: 45초 → 20초

### Phase 3: 중기 개선 (3일)
- [ ] **페이지 타입 자동 감지**
  - 무한 스크롤 vs 일반 페이지
  - 자동 전략 선택

### Phase 4: 장기 개선 (1주)
- [ ] **병렬 처리**
  - 3개 탭 동시 처리
  - 예상 속도: 45초 → 15초

## 🎯 최종 목표

### 현재 (특정 사이트 최적화)
```
180개 상품 처리 시간: ~15분
- LLM 추출: 2분
- URL 캡처: 13분 (상품당 4-5초)
```

### Phase 1 적용 후 (URL 패턴 학습)
```
180개 상품 처리 시간: ~5분
- LLM 추출: 2분
- URL 캡처: 3분 (처음 3개만 클릭, 나머지는 패턴)
```

### Phase 4 완료 후 (모든 최적화)
```
180개 상품 처리 시간: ~3분
- LLM 추출: 2분 (병렬 처리)
- URL 캡처: 1분 (병렬 + 패턴)
```

## 💡 권장 사항

### 1. 즉시 적용: URL 패턴 학습
**가장 큰 효과, 가장 쉬운 구현**

```python
# 3개 성공 후
if len(url_mapping) >= 3:
    pattern = learn_pattern(url_mapping)
    # 나머지는 패턴으로 생성
    for remaining_id in product_ids[3:]:
        url_mapping[remaining_id] = pattern.format(ID=remaining_id)
```

### 2. 설정 옵션 추가
```python
crawler = ProductCrawler(
    url=url,
    mode='auto',  # 'auto', 'infinite_scroll', 'normal', 'pattern_learning'
)
```

### 3. 캐싱
```python
# 한 번 학습한 패턴 저장
cache = {
    'sktuniverse.co.kr': {
        'pattern': 'https://m.sktuniverse.co.kr/product/detail?prdId={ID}',
        'selector': 'a.inner-link[prdid="{ID}"]'
    }
}
```

## 🔍 테스트 계획

### 다양한 사이트 유형
1. **무한 스크롤** (현재)
   - ✅ sktuniverse.co.kr

2. **일반 페이지네이션**
   - [ ] 테스트 필요

3. **정적 링크**
   - [ ] 테스트 필요

4. **AJAX 로딩**
   - [ ] 테스트 필요

## 📚 참고 자료

### 유사 프로젝트
- Scrapy: 범용 크롤링 프레임워크
- Selenium Grid: 병렬 브라우저
- Puppeteer Cluster: 브라우저 풀링

