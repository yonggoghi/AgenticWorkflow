# 페이지 타입 자동 감지 가이드

## 🎯 개요

`info_builder`는 웹 페이지의 구성 방식을 **자동으로 감지**하여 최적의 크롤링 전략을 적용합니다.

## 📊 감지 가능한 페이지 타입

### 1. 무한 스크롤 (Infinite Scroll)
- **감지 방법**: 실제 스크롤 테스트 수행 → 높이/아이템 증가 확인
- **특징**: 스크롤 시 동적으로 콘텐츠 로드
- **전략**: 
  - 여러 번 스크롤 수행 (기본 10회)
  - 뒤로 가기 후 재스크롤 필요 ⚠️
- **예시**: SKT Universe 상품 목록

```python
# 자동 감지 (기본값)
crawler.run(url, auto_detect=True)

# 감지 결과: infinite_scroll (확신도: 0.90)
# 전략: 무한 스크롤 - 뒤로 가기 후 재스크롤 필요
```

### 2. 페이지네이션 (Pagination)
- **감지 방법**: 페이지 번호 버튼, "다음" 링크 탐색
- **특징**: 페이지 번호로 이동
- **전략**:
  - 스크롤 불필요
  - 뒤로 가기 후 재스크롤 불필요 ✅
- **예시**: 전통적인 쇼핑몰, 게시판

```python
# 감지 결과: pagination (확신도: 0.85)
# 전략: 페이지네이션 - 스크롤 불필요
```

### 3. "더보기" 버튼 (Load More)
- **감지 방법**: "더보기", "Load More" 버튼 탐색
- **특징**: 버튼 클릭 시 콘텐츠 추가 로드
- **전략**:
  - 버튼 클릭 방식 (향후 구현 예정)
  - 뒤로 가기 후 재스크롤 불필요 ✅
- **예시**: 모바일 앱 스타일 웹페이지

```python
# 감지 결과: load_more (확신도: 0.90)
# 전략: "더보기" 버튼 클릭 방식
```

### 4. 정적 페이지 (Static)
- **감지 방법**: 다른 패턴이 없을 때 기본값
- **특징**: 모든 콘텐츠가 이미 로드됨
- **전략**:
  - 가벼운 스크롤만 (기본 2회)
  - 뒤로 가기 후 재스크롤 불필요 ✅
- **예시**: 단일 페이지 상품 리스트

```python
# 감지 결과: static (확신도: 0.30)
# 전략: 정적 페이지 - 가벼운 스크롤만
```

## 🚀 사용법

### 1. 기본 사용 (자동 감지)

```python
from product_crawler import ProductCrawler

crawler = ProductCrawler(base_url="https://example.com", model_name="ax")

# 페이지 타입 자동 감지 + 최적 전략 적용
df = crawler.run(
    url="https://example.com/products",
    auto_detect=True,  # 기본값
    crawl_details=True
)
```

**출력 예시:**
```
[1단계] 목록 페이지 크롤링
  🔍 페이지 타입 자동 감지 중...
  ✅ 감지 결과: infinite_scroll (확신도: 0.90)
     무한 스크롤: 여러 번 스크롤 필요, 뒤로 가기 후 재스크롤
```

### 2. 수동 설정 (자동 감지 비활성화)

```python
# 자동 감지 비활성화 + 수동 설정
df = crawler.run(
    url="https://example.com/products",
    auto_detect=False,
    infinite_scroll=True,  # 수동으로 지정
    scroll_count=15
)
```

### 3. CLI 사용

```bash
# 자동 감지 (기본값)
python product_crawler.py "https://example.com/products" --details

# 수동 설정
python product_crawler.py "https://example.com/products" \
    --no-auto-detect \
    --scroll \
    --scroll-count 15 \
    --details
```

## 🎯 장점

### 1. 성능 개선
- **무한 스크롤**: 재스크롤 필요 → 7.5분 추가 소요
- **페이지네이션/정적**: 재스크롤 불필요 → 시간 절약 ✅

```
예시 (180개 상품):
- 무한 스크롤: 15분 소요 (재스크롤 7.5분)
- 정적 페이지: 7.5분 소요 (재스크롤 생략)
→ 2배 속도 향상! 🚀
```

### 2. 일반화
- 다양한 웹사이트에 자동 적용 가능
- 페이지 구조 변경에 자동 대응

### 3. 정확성
- 페이지 타입별 최적 전략 적용
- 불필요한 재스크롤 방지 → 안정성 향상

## 🔧 내부 동작

### 감지 프로세스

```python
# 1. 페이지네이션 확인
nav[role="navigation"], ul.pagination, a:has-text("다음"), ...

# 2. "더보기" 버튼 확인
button:has-text("더보기"), button:has-text("Load More"), ...

# 3. 무한 스크롤 테스트
초기 높이/아이템 수 측정 → 스크롤 → 변화 확인

# 4. 정적 페이지 (기본값)
다른 패턴 없음 → static
```

### 확신도 계산

```python
# 무한 스크롤
- 높이 + 아이템 증가: 0.9
- 높이 또는 아이템 증가: 0.7

# 페이지네이션 / "더보기"
- 1개 요소 발견: 0.7
- 2개 이상 발견: 0.8+
- 최대: 0.95

# 정적 페이지
- 기본 확신도: 0.3
```

## 📊 전략 매핑

| 페이지 타입 | 스크롤 | 스크롤 횟수 | 재스크롤 | 예상 시간 (180개 상품) |
|-----------|-------|-----------|---------|----------------------|
| infinite_scroll | ✅ | 10 | ✅ | 15분 |
| pagination | ❌ | 0 | ❌ | 7.5분 |
| load_more | ❌ | 0 | ❌ | 7.5분 |
| static | ✅ | 2 | ❌ | 7.5분 |

## 🐛 디버깅

### 감지 결과 확인

```python
from page_type_detector import PageTypeDetector
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto("https://example.com/products")
    
    # 상세 로그 포함
    result = PageTypeDetector.detect(page, verbose=True)
    
    print(f"타입: {result['type']}")
    print(f"확신도: {result['confidence']}")
    print(f"상세: {result['details']}")
    
    # 권장 전략 확인
    strategy = PageTypeDetector.get_scroll_strategy(result['type'])
    print(f"전략: {strategy}")
```

### 출력 예시

```
🔍 페이지 타입 감지 시작...
  📄 페이지네이션 확인 중...
  🔘 '더보기' 버튼 확인 중...
  ♾️  무한 스크롤 테스트 중...
    초기 높이: 11726px, 아이템: 326개
    스크롤 후: 22000px, 아이템: 526개
    ✓ 무한 스크롤 감지: 높이 11726→22000, 아이템 326→526
✅ 감지 결과: infinite_scroll (확신도: 0.90)
   상세: 스크롤 시 콘텐츠 증가

타입: infinite_scroll
확신도: 0.90
상세: 스크롤 시 콘텐츠 증가 (높이 11726→22000, 아이템 326→526)

전략: {
  'should_scroll': True,
  'scroll_count': 10,
  'scroll_delay': 2000,
  'need_rescroll_after_back': True,
  'description': '무한 스크롤: 여러 번 스크롤 필요, 뒤로 가기 후 재스크롤'
}
```

## 🔮 향후 개선 (Phase 2-4)

### Phase 2: 선택적 재스크롤 (계획 중)
- 3-5개 URL 캡처 후 패턴 학습
- 패턴 확인 시 나머지 자동 생성
- 180개 상품: 15분 → 5분 (3배 향상)

### Phase 3: "더보기" 버튼 자동 클릭 (계획 중)
- `load_more` 타입 완전 지원
- 버튼 클릭 자동화

### Phase 4: 병렬 처리 (계획 중)
- 다중 브라우저 탭 동시 실행
- 180개 상품: 5분 → 3분 (5배 향상)

## 📝 참고

- 자동 감지는 Playwright가 설치된 경우에만 동작
- 자동 감지 실패 시 수동 설정으로 자동 폴백
- `--no-auto-detect` 플래그로 수동 설정 강제 가능

## 🎓 예제

### 예제 1: SKT Universe (무한 스크롤)

```python
crawler = ProductCrawler(base_url="https://m.sktuniverse.co.kr", model_name="ax")

df = crawler.run(
    url="https://m.sktuniverse.co.kr/category/sub/tab/detail?ctanId=CC00000012&ctgId=CA00000001",
    auto_detect=True,  # 자동으로 infinite_scroll 감지
    crawl_details=True
)

# 자동으로 최적 전략 적용:
# - 무한 스크롤 활성화
# - 뒤로 가기 후 재스크롤 활성화
```

### 예제 2: 일반 쇼핑몰 (페이지네이션)

```python
df = crawler.run(
    url="https://shop.example.com/products",
    auto_detect=True,  # 자동으로 pagination 감지
    crawl_details=True
)

# 자동으로 최적 전략 적용:
# - 스크롤 비활성화
# - 뒤로 가기 후 재스크롤 비활성화
# → 불필요한 작업 생략, 시간 절약!
```

## ✅ 권장 사항

1. **기본적으로 자동 감지 사용** (`auto_detect=True`)
2. **감지 실패 시에만 수동 설정** (`--no-auto-detect`)
3. **디버깅 시 `verbose=True`** 사용

---

**작성일**: 2025-11-03  
**버전**: 1.0.0

