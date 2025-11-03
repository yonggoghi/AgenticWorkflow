# 상품/서비스 크롤러 가이드

## 개요

`product_crawler.py`는 웹 페이지에서 상품/서비스 정보를 자동으로 추출하고 정리하는 도구입니다.

### 주요 기능

1. **목록 페이지 크롤링** - 무한 스크롤 지원
2. **상품/서비스 정보 파싱** - LLM 기반 지능형 추출
3. **상세 페이지 자동 방문** - 상세 정보 수집
4. **구조화된 데이터 생성** - pandas DataFrame으로 저장 (CSV, JSON, Excel)

## 설치

### 필수 패키지

```bash
# 기본 패키지
pip install playwright pandas tqdm

# 브라우저 설치
playwright install chromium

# LLM 지원 (선택)
pip install anthropic  # Claude 사용 시
pip install openai     # GPT 사용 시

# Excel 지원 (선택)
pip install openpyxl
```

### 환경 변수 설정

LLM을 사용하려면 API 키가 필요합니다:

```bash
# Claude 사용 시
export ANTHROPIC_API_KEY="your-api-key"

# GPT 사용 시
export OPENAI_API_KEY="your-api-key"
```

또는 `.env` 파일에 저장:

```bash
ANTHROPIC_API_KEY=your-api-key
OPENAI_API_KEY=your-api-key
```

## 사용법

### 기본 사용

```bash
# LLM을 사용한 자동 추출
python product_crawler.py "https://m.shop.com/products"
```

### 무한 스크롤 페이지

```bash
# 무한 스크롤 활성화
python product_crawler.py "https://m.shop.com/products" --scroll

# 스크롤 횟수 지정
python product_crawler.py "https://m.shop.com/products" --scroll --scroll-count 20
```

### 상세 페이지 크롤링

```bash
# 상세 페이지도 크롤링
python product_crawler.py "https://m.shop.com/products" --scroll --details

# 상세 페이지는 최대 10개만
python product_crawler.py "https://m.shop.com/products" --details --max-details 10
```

### LLM 옵션

```bash
# LLM 없이 사용 (규칙 기반)
python product_crawler.py "https://m.shop.com/products" --no-llm

# OpenAI 사용
python product_crawler.py "https://m.shop.com/products" --llm-provider openai

# Claude 사용 (기본값)
python product_crawler.py "https://m.shop.com/products" --llm-provider anthropic
```

### 출력 파일 지정

```bash
# 출력 파일명 지정 (확장자 제외)
python product_crawler.py "https://m.shop.com/products" --output my_products

# 결과: my_products.csv, my_products.json, my_products.xlsx
```

## 실제 사용 예제

### SKT 유니버스 상품 크롤링

```bash
python product_crawler.py \
  "https://m.sktuniverse.co.kr/category/sub/tab/detail?ctanId=CC00000013&ctgId=CA00000002" \
  --scroll \
  --scroll-count 15 \
  --output skt_products
```

### 상세 페이지 포함 크롤링

```bash
python product_crawler.py \
  "https://example.com/products" \
  --scroll \
  --details \
  --max-details 20 \
  --output products_detailed
```

## 작동 방식

### 1단계: 목록 페이지 크롤링

- Playwright로 페이지 방문
- 무한 스크롤 실행 (옵션)
- HTML 및 텍스트 콘텐츠 수집

### 2단계: 상품 정보 추출

#### LLM 사용 시 (권장)

Claude 또는 GPT가 페이지 내용을 분석하여 다음 정보를 자동 추출:
- 상품 ID
- 상품명
- 설명
- 가격
- 상세 페이지 URL

#### 규칙 기반 (--no-llm)

키워드 매칭으로 상품 링크만 추출:
- 'product', 'item', 'detail', '상품' 등의 키워드 포함 링크

### 3단계: 상세 페이지 크롤링 (옵션)

- 각 상품의 상세 페이지 자동 방문
- LLM으로 상세 정보 추출:
  - 상세 설명
  - 카테고리
  - 주요 특징
  - 스펙 정보

### 4단계: 데이터 저장

pandas DataFrame으로 변환 후 저장:
- **CSV**: 엑셀에서 열기 가능 (UTF-8 BOM)
- **JSON**: API 연동 용이
- **Excel**: 비즈니스 사용자 친화적

## 출력 데이터 구조

### 기본 컬럼

| 컬럼 | 설명 | 예시 |
|------|------|------|
| id | 상품 ID | "1", "PROD001" |
| name | 상품/서비스 이름 | "갤럭시 S24" |
| description | 설명 | "최신 플래그십 스마트폰..." |
| price | 가격 | "1,200,000원" |
| detail_url | 상세 페이지 URL | "https://..." |

### 상세 페이지 크롤링 시 추가 컬럼

| 컬럼 | 설명 | 예시 |
|------|------|------|
| category | 카테고리 | "스마트폰" |
| features | 주요 특징 (리스트) | ["5G", "120Hz", ...] |
| specifications | 스펙 (딕셔너리) | {"화면": "6.8인치", ...} |

### 출력 예시

```csv
id,name,description,price,detail_url,category,features
1,갤럭시 S24,최신 플래그십 스마트폰,1200000원,https://...,스마트폰,"['5G', '120Hz']"
2,아이폰 15,애플의 최신 모델,1300000원,https://...,스마트폰,"['A17 Pro', '티타늄']"
```

## Python 코드에서 사용

### 기본 사용

```python
from product_crawler import ProductCrawler

# 크롤러 초기화
crawler = ProductCrawler(
    base_url="https://m.shop.com/products",
    use_llm=True,
    llm_provider="anthropic"
)

# 실행
df = crawler.run(
    url="https://m.shop.com/products",
    infinite_scroll=True,
    scroll_count=15,
    crawl_details=False,
    output_path="output/products"
)

# 결과 확인
print(f"추출된 상품: {len(df)}개")
print(df.head())
```

### 단계별 제어

```python
from product_crawler import ProductCrawler

crawler = ProductCrawler(
    base_url="https://m.shop.com/products",
    use_llm=True
)

# 1. 목록 페이지만 크롤링
products = crawler.crawl_list_page(
    url="https://m.shop.com/products",
    infinite_scroll=True,
    scroll_count=10
)

print(f"추출된 상품: {len(products)}개")

# 2. 일부 상품만 상세 크롤링
selected = products[:5]  # 처음 5개만
products_with_details = crawler.crawl_detail_pages(selected)

# 3. DataFrame으로 저장
df = crawler.save_to_dataframe(
    products_with_details,
    output_path="output/products"
)
```

### 데이터 분석

```python
import pandas as pd

# 저장된 데이터 로드
df = pd.read_csv("output/products.csv")

# 기본 분석
print(f"총 상품 수: {len(df)}")
print(f"컬럼: {list(df.columns)}")

# 가격이 있는 상품
df_with_price = df[df['price'].notna()]
print(f"가격 정보: {len(df_with_price)}개")

# 카테고리별 집계
if 'category' in df.columns:
    category_counts = df['category'].value_counts()
    print("\n카테고리별 상품 수:")
    print(category_counts)
```

## 주요 옵션

| 옵션 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `url` | str | - | 크롤링할 URL (필수) |
| `--scroll` | flag | False | 무한 스크롤 활성화 |
| `--scroll-count` | int | 10 | 최대 스크롤 횟수 |
| `--details` | flag | False | 상세 페이지 크롤링 |
| `--max-details` | int | None | 최대 상세 페이지 수 |
| `--no-llm` | flag | False | LLM 사용 안 함 |
| `--llm-provider` | str | anthropic | LLM 제공자 (anthropic/openai) |
| `--output` | str | product_data | 출력 파일 경로 |

## 성능 및 비용

### 크롤링 속도

- **목록 페이지**: 10-30초 (무한 스크롤 포함)
- **상세 페이지**: 상품당 3-5초
- **LLM 추출**: 요청당 2-5초

### API 비용 (참고)

- **Claude 3.5 Sonnet**: 입력 $3/MTok, 출력 $15/MTok
- **GPT-4o-mini**: 입력 $0.15/MTok, 출력 $0.6/MTok

예상 비용 (상품 100개 기준):
- 목록 추출: $0.05-0.10
- 상세 페이지: $0.50-1.00

## 문제 해결

### LLM API 키 오류

```bash
# 환경 변수 확인
echo $ANTHROPIC_API_KEY

# 설정
export ANTHROPIC_API_KEY="your-key"
```

### 추출된 상품이 너무 적을 때

```bash
# 스크롤 횟수 증가
python product_crawler.py "URL" --scroll --scroll-count 30

# LLM 사용 확인
python product_crawler.py "URL" --llm-provider anthropic
```

### 메모리 부족

```bash
# 상세 페이지 크롤링 수 제한
python product_crawler.py "URL" --details --max-details 10
```

### JSON 파싱 오류

LLM 응답이 잘못된 경우 재시도되지 않습니다. 다음을 확인:
- API 키가 유효한지
- 인터넷 연결 상태
- 페이지 내용이 너무 길지 않은지

## 제한사항

1. **페이지 구조 의존성**: 일부 복잡한 페이지는 추출이 어려울 수 있음
2. **로그인 필요 페이지**: 로그인이 필요한 페이지는 지원 안 됨
3. **CAPTCHA**: CAPTCHA가 있는 페이지는 차단될 수 있음
4. **속도 제한**: 과도한 요청 시 IP 차단 가능

## 베스트 프랙티스

1. **먼저 소량 테스트**: `--max-details 5`로 시작
2. **스크롤 횟수 조절**: 페이지에 맞게 조정
3. **LLM 사용 권장**: 더 정확한 추출
4. **정기적 저장**: 크롤링 중간에 데이터 저장
5. **robots.txt 준수**: 웹사이트 정책 확인

## 주의사항

⚠️ **법적 책임**: 웹사이트의 이용 약관을 확인하고 준수하세요  
⚠️ **속도 제한**: 과도한 크롤링은 차단될 수 있습니다  
⚠️ **개인정보**: 개인정보가 포함된 데이터는 주의해서 다루세요  
⚠️ **저작권**: 크롤링한 데이터의 사용 권한을 확인하세요

## 라이선스

교육 및 연구 목적으로 제공됩니다.

