# 무한 스크롤 페이지 뒤로 가기 문제 해결

## 🔍 문제 발견

### 증상
```
[상품 1] ID: PR00000538
  selector 1: 2개 발견
  ✅ URL 캡처 성공!
  뒤로 가기...
  ✅ 뒤로 가기 완료

[상품 2] ID: PR00000684
  selector 1: 0개 발견  ← ❌ 왜?
  selector 2: 0개 발견
  selector 3: 0개 발견
  ❌ 모든 selector 실패 - not_found
```

### 원인 분석

#### 1단계: 초기 로딩
```
페이지 접속 → 무한 스크롤 (5회) → 모든 상품 로드 (9개)
DOM: [상품1, 상품2, 상품3, ..., 상품9] ✅
```

#### 2단계: 상품 1 처리
```
상품1 클릭 → 상세 페이지 → 뒤로 가기
```

#### 3단계: 문제 발생
```
뒤로 가기 후 페이지 상태:
- 페이지가 초기 상태로 리셋됨
- 무한 스크롤이 다시 시작됨
- 처음 1-2개 상품만 DOM에 로드됨
DOM: [상품1] ← 상품2, 3이 없음! ❌
```

#### 4단계: 상품 2 찾기 실패
```
상품2 찾기 → DOM에 없음 → not_found ❌
```

## ✅ 해결 방법

### 핵심 아이디어
**뒤로 가기 후 다시 스크롤해서 모든 상품을 DOM에 로드**

### 구현
```python
# 뒤로 가기
page.go_back()
page.wait_for_timeout(1000)

# 🔧 무한 스크롤 페이지: 뒤로 가기 후 다시 스크롤
if infinite_scroll and idx < len(product_ids) - 1:  # 마지막 상품 아니면
    # 빠르게 스크롤 (모든 상품 다시 로드)
    for i in range(scroll_count):
        page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
        page.wait_for_timeout(500)  # 빠르게 (2000ms → 500ms)
    page.wait_for_timeout(1000)  # 최종 대기
```

### 디버깅 로그
```
[상품 1] ID: PR00000538
  ✅ URL 캡처 성공!
  뒤로 가기...
  무한 스크롤 재실행...        ← 추가!
  ✅ 스크롤 재실행 완료
  ✅ 뒤로 가기 완료

[상품 2] ID: PR00000684
  selector 1: 2개 발견         ← ✅ 이제 찾음!
  ✅ URL 캡처 성공!
```

## 📊 성능 영향

### 추가 시간
- 각 상품당 재스크롤: ~3초 (5회 × 500ms + 1000ms)
- 9개 상품: ~24초 추가
- **허용 가능한 오버헤드**

### 최적화
1. **마지막 상품은 스크롤 안 함** - `idx < len(product_ids) - 1`
2. **빠른 스크롤** - 2000ms → 500ms
3. **DEBUG_MODE에서 충분히 테스트**

## 🎯 예상 결과

| 항목 | 개선 전 | 개선 후 |
|------|---------|---------|
| 성공률 | **1/9 (11%)** | **9/9 (100%)** |
| 상품 1 | ✅ 성공 | ✅ 성공 |
| 상품 2-9 | ❌ not_found | ✅ 성공 |
| 추가 시간 | - | ~24초 (9개 기준) |

## 🚀 테스트 방법

```bash
cd /Users/yongwook/workspace/AgenticWorkflow/info_builder
source ../venv/bin/activate
python test_crawl_details.py
```

**예상 시간**: ~30초 → ~50초 (스크롤 재실행 포함)

## 💡 왜 이런 문제가 발생했나?

### 무한 스크롤의 특성
- 초기 로딩 시 일부만 표시
- 스크롤 이벤트로 추가 로딩
- **뒤로 가기 시 JavaScript 상태 리셋**

### 일반 페이지와의 차이
| 페이지 타입 | 뒤로 가기 후 | 문제 발생 여부 |
|------------|-------------|---------------|
| 일반 페이지 | 모든 요소 존재 | ✅ 문제 없음 |
| 무한 스크롤 | 초기 요소만 존재 | ❌ 문제 발생 |

## 📝 학습 포인트

1. **무한 스크롤 페이지는 특별한 처리 필요**
2. **뒤로 가기 ≠ 이전 상태 복원**
3. **디버깅 로그로 문제 빠르게 발견**
4. **첫 번째만 성공 = 뒤로 가기 문제**

## 🔧 추가 개선 가능성

### 방법 1: 새 탭 사용 (더 빠름)
```python
# 뒤로 가기 대신 새 탭 열기
context = browser.new_context()
detail_page = context.new_page()
detail_page.goto(detail_url)
# ... URL 캡처
detail_page.close()
context.close()
```

### 방법 2: URL 직접 생성 (가장 빠름)
```python
# 클릭하지 않고 ID로 URL 생성
detail_url = f"https://m.sktuniverse.co.kr/netfunnel?path=%2Fproduct%2Fdetail%3FprdId%3D{prd_id}"
```

현재 방법은 **가장 확실하고 안전**한 방법입니다.

