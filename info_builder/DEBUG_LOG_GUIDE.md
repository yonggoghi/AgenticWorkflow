# 디버깅 로그 가이드

## 🔧 디버깅 모드

### 자동 활성화
- `DEBUG_MODE = True` (product_crawler.py 라인 324)
- **첫 번째 청크만 처리** (~9개 상품)
- **처음 3개 상품은 상세 로그**

### 속도
| 모드 | 처리 시간 | 상품 수 |
|------|----------|---------|
| 전체 | ~5분 | 180+ 개 |
| 디버깅 | **~30초** | ~9개 |

## 📊 출력되는 로그

### 1. 청크 정보
```
🔧 [디버깅 모드] 첫 번째 청크만 처리합니다
HTML을 1개 청크로 분할
청크 크기: 최소 14,752자, 최대 14,752자, 평균 14,752자
```

### 2. 첫 번째 상품 전체 Selector 테스트
```
[디버깅] 첫 번째 상품 ID: PR00000538
[디버깅] a.inner-link[prdid="PR00000538"][godetailyn="Y"]: 2개 발견
[디버깅] a.inner-link[prdid="PR00000538"]: 2개 발견
[디버깅] a[prdid="PR00000538"]: 2개 발견
[디버깅] [prdid="PR00000538"]: 4개 발견
```

### 3. 각 상품별 상세 로그 (처음 3개)

#### 성공 케이스
```
[상품 1] ID: PR00000538
  selector 1: 2개 발견
  ✅ selector 1 선택됨
  스크롤 시도...
  ✅ 스크롤 완료
  클릭 전 URL: https://m.sktuniverse.co.kr/category/sub/tab/detail?...
  클릭 시도 (Playwright)...
  ✅ 클릭 성공
  URL 변경 대기...
  ✅ URL 변경 감지
  클릭 후 URL: https://m.sktuniverse.co.kr/netfunnel?path=%2Fproduct%2Fdetail...
  ✅ URL 캡처 성공!
  뒤로 가기...
  ✅ 뒤로 가기 완료
```

#### 실패 케이스 - Selector 못 찾음
```
[상품 2] ID: PR00000123
  selector 1: 0개 발견
  selector 2: 0개 발견
  selector 3: 0개 발견
  ❌ 모든 selector 실패 - not_found
```

#### 실패 케이스 - 클릭 실패
```
[상품 3] ID: PR00000456
  selector 1: 1개 발견
  ✅ selector 1 선택됨
  스크롤 시도...
  ✅ 스크롤 완료
  클릭 전 URL: https://m.sktuniverse.co.kr/...
  클릭 시도 (Playwright)...
  ⚠️ Playwright 클릭 실패: Element is not visible
  JavaScript 클릭 시도...
  ❌ JavaScript 클릭도 실패: Cannot find element
```

#### 실패 케이스 - URL 변경 없음
```
[상품 4] ID: PR00000789
  selector 1: 1개 발견
  ✅ selector 1 선택됨
  스크롤 시도...
  ✅ 스크롤 완료
  클릭 전 URL: https://m.sktuniverse.co.kr/...
  클릭 시도 (Playwright)...
  ✅ 클릭 성공
  URL 변경 대기...
  ⚠️ URL 변경 없음, 1.5초 대기...
  클릭 후 URL: https://m.sktuniverse.co.kr/...
  ❌ URL 변경 없음 - url_not_changed
```

### 4. 최종 통계
```
✅ 1/9개 상품의 detail_url 캡처 완료
❌ 실패 원인 분석:
   - not_found: 8개
```

## 🔍 로그 해석

### not_found
- **의미**: Selector로 요소를 찾지 못함
- **원인**: 
  1. ID가 HTML에 없음
  2. HTML 구조가 다름
  3. JavaScript로 동적 생성되는 요소

### click_failed
- **의미**: Playwright와 JavaScript 클릭 모두 실패
- **원인**:
  1. 요소가 클릭 불가 상태
  2. 다른 요소에 가려짐
  3. disabled 속성

### url_not_changed
- **의미**: 클릭은 성공했으나 URL 변경 없음
- **원인**:
  1. JavaScript 이벤트 핸들러가 페이지 이동 안 함
  2. AJAX로 컨텐츠만 교체
  3. 모달/팝업 열림

## 💡 디버깅 팁

### 1. 모든 Selector가 0개 발견
→ HTML 구조가 완전히 다름. HTML 파일 직접 확인 필요

### 2. Selector는 찾았으나 not_found
→ 스크롤 후 요소가 사라짐. 동적 로딩 문제

### 3. Playwright 클릭 실패 → JavaScript 성공
→ 요소가 viewport 밖. scroll_into_view 문제

### 4. 클릭 성공 → URL 변경 없음
→ 상세 페이지가 없거나 다른 방식으로 표시

## 🚀 테스트 방법

```bash
cd /Users/yongwook/workspace/AgenticWorkflow/info_builder
source ../venv/bin/activate
python test_crawl_details.py
```

**예상 시간**: ~30초 (9개 상품만 처리)

## 📝 디버깅 모드 해제

`product_crawler.py` 라인 324:
```python
DEBUG_MODE = False  # True → False로 변경
```

전체 180+ 상품 처리 (~5분 소요)

