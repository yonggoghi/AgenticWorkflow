# 문제 해결 가이드

## 문제: 실제 상품 개수보다 적게 추출됨

### 증상
웹 페이지에는 182개 상품이 있는데, 90개만 추출됨

### 원인

1. **LLM 청크당 개수 제한**
   - 프롬프트에 "최대 30개 상품만 추출"이라는 제한이 있었음
   - 3개 청크 × 30개 = 90개

2. **무한 스크롤 횟수 부족**
   - 기본 scroll_count=15는 많은 상품을 로드하기에 부족할 수 있음
   - 182개 상품을 모두 로드하려면 더 많이 스크롤 필요

3. **스크롤 후 대기 시간 부족**
   - 각 스크롤 후 1.5초만 대기하면 콘텐츠가 완전히 로드되지 않을 수 있음

### 해결 방법

#### 1. 상품 개수 제한 제거 (자동 적용됨)

프롬프트가 다음과 같이 수정되었습니다:
```
3. 이 청크에 있는 모든 상품을 추출하세요 (빠뜨리지 말 것)
6. 상품이 많아도 모두 포함하세요
```

#### 2. 스크롤 횟수 증가

```bash
# 기본값 (많은 상품)
python product_crawler.py "URL" --scroll --scroll-count 30

# 매우 많은 상품 (200개 이상)
python product_crawler.py "URL" --scroll --scroll-count 50
```

또는 Python 코드에서:
```python
df = crawler.run(
    url=url,
    infinite_scroll=True,
    scroll_count=30,  # 15 → 30으로 증가
    output_path="products"
)
```

#### 3. 스크롤 대기 시간 확인

현재 설정 (자동 적용됨):
- 각 스크롤 후: 2초 대기
- 스크롤 완료 후: 3초 최종 대기

## 문제: JSON 파싱 오류

### 증상
```
→ JSON 파싱 오류: Unterminated string starting at...
```

### 해결 방법

1. **자동 복구 시도** (이미 적용됨)
   - 불완전한 JSON을 자동으로 복구 시도
   - "복구 성공" 메시지가 표시됨

2. **청크 크기 축소**
   ```python
   # product_crawler.py에서
   chunks = self._chunk_text(text_content, chunk_size=3000, overlap=500)
   ```

3. **다른 LLM 모델 시도**
   ```bash
   python product_crawler.py "URL" --model gemini
   python product_crawler.py "URL" --model claude
   ```

## 문제: 중복된 상품 추출

### 증상
같은 상품이 여러 번 나타남

### 원인
- 상품명 + 가격 조합이 다름 (예: 가격이 빈 문자열)
- 청크 오버랩 영역에서 중복 발생

### 해결 방법

중복 제거 로직 강화 (`product_crawler.py`):
```python
# 더 강력한 중복 제거
product_key = (
    product.get('name', '').strip().lower() + 
    product.get('price', '').replace(',', '').strip()
)
```

## 문제: 스크롤이 중간에 멈춤

### 증상
```
스크롤 5/30: 8234px → 8234px
더 이상 새로운 콘텐츠 없음 (총 5회 스크롤)
```

### 원인
- 페이지 높이가 더 이상 증가하지 않음 (정상 동작)
- 모든 콘텐츠가 로드되었거나, 더 이상 로드할 것이 없음

### 해결 방법

1. **대기 시간 증가**
   ```python
   page.wait_for_timeout(3000)  # 2초 → 3초
   ```

2. **스크롤 방식 변경**
   일부 페이지는 조금씩 스크롤해야 함:
   ```python
   # 한 번에 끝까지 대신 조금씩 스크롤
   for i in range(scroll_count):
       page.evaluate('window.scrollBy(0, 500)')  # 500px씩
       page.wait_for_timeout(500)
   ```

## 문제: 크롤링이 너무 느림

### 증상
30회 스크롤 × 2초 = 60초 이상 소요

### 해결 방법

1. **스크롤 횟수 최적화**
   ```python
   # 실제 필요한 만큼만
   scroll_count=20  # 대신 30
   ```

2. **스크롤 대기 시간 축소**
   ```python
   page.wait_for_timeout(1000)  # 빠르지만 불안정
   ```

3. **병렬 처리** (고급)
   여러 페이지를 동시에 크롤링

## 문제: 텍스트는 많은데 상품이 적음

### 증상
```
텍스트: 50000 문자
추출된 상품: 10개
```

### 원인
- 페이지에 상품 정보가 아닌 다른 텍스트가 많음 (광고, 설명 등)
- LLM이 상품을 제대로 인식하지 못함

### 해결 방법

1. **HTML 선택자로 상품 영역만 추출** (고급)
   ```python
   # 특정 CSS 선택자의 내용만 추출
   product_area = page.query_selector('.product-list')
   text_content = product_area.inner_text()
   ```

2. **프롬프트 개선**
   더 명확한 상품 정의 제공

3. **다른 LLM 모델**
   ```bash
   python product_crawler.py "URL" --model claude  # 더 정확
   ```

## 문제: 메모리 부족

### 증상
```
MemoryError: Unable to allocate...
```

### 해결 방법

1. **청크 크기 축소**
   ```python
   chunk_size=3000  # 더 작은 청크
   ```

2. **스크린샷 비활성화**
   ```bash
   # --screenshot 제거
   ```

3. **상세 페이지 크롤링 제한**
   ```bash
   --max-details 5
   ```

## 체크리스트

### 모든 상품을 추출하려면:

- [ ] `scroll_count`를 충분히 크게 설정 (30+)
- [ ] 스크롤 대기 시간 확인 (2초 이상)
- [ ] 최종 대기 시간 추가 (3초)
- [ ] LLM 프롬프트에 개수 제한 없는지 확인
- [ ] 청크 크기가 적절한지 확인 (5000자)
- [ ] 중복 제거 로직 확인
- [ ] LLM 모델이 적절한지 확인

### 빠른 디버깅:

```bash
# 1. 크롤링된 텍스트 길이 확인
python product_crawler.py "URL" --scroll --print-text

# 2. 스크롤 횟수 증가
python product_crawler.py "URL" --scroll --scroll-count 40

# 3. 다른 모델 시도
python product_crawler.py "URL" --scroll --model claude
```

## 로그 해석

### 정상 로그:
```
텍스트를 4개 청크로 분할
청크 1/4 처리 중... (5000 문자)
  → 45개 상품 발견 (총 45개)
청크 2/4 처리 중... (5000 문자)
  → 48개 상품 발견 (총 93개)
...
총 182개의 고유 상품/서비스를 추출했습니다.
```

### 문제 있는 로그:
```
청크 1/2 처리 중... (5000 문자)
  → 30개 상품 발견 (총 30개)  ⚠️ 정확히 30개 = 제한에 걸림
청크 2/2 처리 중... (5000 문자)
  → 30개 상품 발견 (총 60개)  ⚠️ 또 30개 = 문제!
```

해결: 프롬프트에서 개수 제한 제거 (이미 적용됨)

## 추가 도움말

더 많은 정보는 다음 문서를 참고하세요:
- [WEB_CRAWLER_GUIDE.md](WEB_CRAWLER_GUIDE.md) - 웹 크롤러 상세 가이드
- [PRODUCT_CRAWLER_GUIDE.md](PRODUCT_CRAWLER_GUIDE.md) - 상품 크롤러 가이드
- [CHUNKING_GUIDE.md](CHUNKING_GUIDE.md) - 텍스트 청킹 가이드
- [ENV_SETUP.md](ENV_SETUP.md) - 환경 설정 가이드

