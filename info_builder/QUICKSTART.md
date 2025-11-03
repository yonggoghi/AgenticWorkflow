# Info Builder - 빠른 시작 가이드

## 1분 안에 시작하기

### 1단계: 설치 (1분)

```bash
# 디렉토리 이동
cd info_builder

# 패키지 설치
pip install playwright pandas tqdm langchain-openai langchain

# 브라우저 설치
playwright install chromium
```

### 2단계: API 키 설정 (30초)

```bash
# LLM API 키 설정
export LLM_API_KEY="your-api-key-here"
export LLM_API_URL="https://api.openai.com/v1"
```

> 상세 설정은 [ENV_SETUP.md](ENV_SETUP.md)를 참고하세요.

### 3단계: 실행! (10초)

```bash
# SKT 유니버스 상품 크롤링
python product_crawler.py \
  "https://m.sktuniverse.co.kr/category/sub/tab/detail?ctanId=CC00000013&ctgId=CA00000002" \
  --scroll \
  --scroll-count 10 \
  --output my_first_crawl
```

완료! 🎉

결과 파일:
- `my_first_crawl.csv` - Excel에서 열기
- `my_first_crawl.json` - 프로그래밍 사용
- `my_first_crawl.xlsx` - 비즈니스 팀과 공유

## 실행 결과 예시

```
================================================================================
상품/서비스 정보 크롤러
================================================================================
URL: https://m.sktuniverse.co.kr/...
LLM: 활성화 (anthropic)
무한 스크롤: True
상세 페이지: False
================================================================================

[1단계] 목록 페이지 크롤링
  페이지 로딩: https://m.sktuniverse.co.kr/...
무한 스크롤 시작 (최대 10회)
  스크롤 1/10: 8234px → 12456px
  스크롤 2/10: 12456px → 16789px
  ...
  텍스트: 45678 문자
  링크: 234 개

[2단계] 상품/서비스 정보 추출
  LLM이 42개의 상품/서비스를 추출했습니다.

추출된 상품/서비스: 42개

[4단계] 데이터 정리 및 저장
  42개의 상품/서비스 정보를 정리했습니다.
  
  컬럼: ['id', 'name', 'description', 'price', 'detail_url']
  CSV 저장: my_first_crawl.csv
  JSON 저장: my_first_crawl.json
  Excel 저장: my_first_crawl.xlsx

================================================================================
크롤링 완료!
================================================================================
```

## 데이터 확인

### Excel에서 열기

```bash
open my_first_crawl.xlsx  # macOS
# 또는
start my_first_crawl.xlsx  # Windows
# 또는
xdg-open my_first_crawl.xlsx  # Linux
```

### Python에서 분석

```python
import pandas as pd

# 데이터 로드
df = pd.read_csv('my_first_crawl.csv')

# 확인
print(f"총 상품: {len(df)}개")
print(df.head())

# 상품명만 보기
print(df['name'].tolist())
```

## 다음 단계

### 더 많은 옵션

```bash
# 상세 페이지도 크롤링 (더 많은 정보)
python product_crawler.py "URL" --scroll --details --max-details 10

# LLM 없이 (무료, 하지만 정확도 낮음)
python product_crawler.py "URL" --scroll --no-llm

# GPT 사용
python product_crawler.py "URL" --scroll --llm-provider openai
```

### 더 알아보기

- **상세 가이드**: [PRODUCT_CRAWLER_GUIDE.md](PRODUCT_CRAWLER_GUIDE.md)
- **예제 코드**: `product_crawler_example.py`
- **웹 크롤러**: [WEB_CRAWLER_GUIDE.md](WEB_CRAWLER_GUIDE.md)

## 문제 해결

### "LLM API key not found"

```bash
# API 키 설정 확인
echo $LLM_API_KEY
echo $LLM_API_URL

# 없으면 설정
export LLM_API_KEY="your-key"
export LLM_API_URL="https://api.openai.com/v1"
```

상세 설정: [ENV_SETUP.md](ENV_SETUP.md)

### "playwright not installed"

```bash
pip install playwright
playwright install chromium
```

### URL 오류

URL을 따옴표로 감싸주세요:

```bash
# ❌ 잘못됨
python product_crawler.py https://example.com?id=1&type=2

# ✅ 올바름
python product_crawler.py "https://example.com?id=1&type=2"
```

## 추가 예제

### 다른 쇼핑몰 크롤링

```bash
# 예: 11번가 (예시)
python product_crawler.py \
  "https://m.11st.co.kr/products/..." \
  --scroll \
  --scroll-count 20 \
  --output 11st_products

# 예: 쿠팡 (예시)
python product_crawler.py \
  "https://m.coupang.com/..." \
  --scroll \
  --output coupang_products
```

> **주의**: 각 웹사이트의 이용 약관을 확인하세요!

### 여러 페이지 크롤링

```python
from product_crawler import ProductCrawler

# URL 리스트
urls = [
    "https://site.com/category/phones",
    "https://site.com/category/tablets",
    "https://site.com/category/laptops"
]

crawler = ProductCrawler(base_url="https://site.com", use_llm=True)

all_products = []
for url in urls:
    products = crawler.crawl_list_page(url, infinite_scroll=True)
    all_products.extend(products)

# 저장
df = crawler.save_to_dataframe(all_products, output_path="all_products")
print(f"총 {len(df)}개 상품 수집 완료!")
```

## 성공! 🎉

이제 어떤 웹 페이지든 상품 정보를 자동으로 수집할 수 있습니다!

다음 목표:
- [ ] 다른 쇼핑몰 시도해보기
- [ ] 상세 페이지 크롤링 해보기
- [ ] 데이터 분석 해보기
- [ ] 자동화 스크립트 만들기

