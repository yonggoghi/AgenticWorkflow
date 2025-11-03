# Info Builder

웹 페이지에서 정보를 자동으로 수집하고 분석 가능한 형태로 저장하는 도구 모음입니다.

## 도구 목록

### 1. 웹 크롤러 (`web_crawler.py`)

일반 웹 페이지 크롤링 도구

**주요 기능:**
- ✅ 동적 페이지 크롤링 (JavaScript 렌더링)
- ✅ 무한 스크롤 지원
- ✅ 다양한 출력 형식 (텍스트, HTML, JSON, 스크린샷)
- ✅ 메타데이터 추출 (링크, 이미지)

**가이드:** [WEB_CRAWLER_GUIDE.md](WEB_CRAWLER_GUIDE.md)

### 2. 상품/서비스 크롤러 (`product_crawler.py`) ⭐️ NEW

상품/서비스 정보를 자동으로 추출하고 정리하는 지능형 크롤러

**주요 기능:**
- ✅ **페이지 타입 자동 감지** 🆕 (무한 스크롤/페이지네이션/정적 페이지)
- ✅ LLM 기반 지능형 정보 추출 (Claude/GPT/Gemini/AX)
- ✅ 긴 텍스트 자동 청킹 처리 (중복 제거)
- ✅ 상품 목록 자동 파싱
- ✅ 상세 페이지 자동 방문 및 크롤링
- ✅ 구조화된 데이터 생성 (CSV, JSON, Excel)
- ✅ pandas DataFrame 지원

**가이드:** 
- [PRODUCT_CRAWLER_GUIDE.md](PRODUCT_CRAWLER_GUIDE.md) - 전체 가이드
- [AUTO_DETECT_GUIDE.md](AUTO_DETECT_GUIDE.md) 🆕 - 페이지 타입 자동 감지

## 빠른 시작

### 1. 설치

```bash
# 기본 패키지
pip install playwright pandas tqdm

# 브라우저 설치
playwright install chromium

# LLM 지원 (상품 크롤러 사용 시)
pip install anthropic  # 또는 openai

# Excel 지원 (선택)
pip install openpyxl
```

### 2. 환경 변수 설정 (상품 크롤러 사용 시)

```bash
# LLM API 설정 (필수)
export LLM_API_KEY="your-api-key"
export LLM_API_URL="https://api.openai.com/v1"  # 또는 커스텀 엔드포인트
```

> 상세 설정은 [ENV_SETUP.md](ENV_SETUP.md)를 참고하세요.

### 3. 기본 사용

#### 웹 크롤러

```bash
# 일반 페이지 크롤링
python web_crawler.py "https://example.com"

# 무한 스크롤 페이지 크롤링
python web_crawler.py "https://m.shop.com/products" --scroll --scroll-count 20
```

#### 상품/서비스 크롤러 ⭐️

```bash
# 기본 사용 (자동 감지 + LLM 추출 - AX 사용) 🆕
python product_crawler.py "https://m.shop.com/products"

# 페이지 타입 자동 감지 + 상세 페이지 크롤링 🆕
python product_crawler.py "https://m.shop.com/products" --details

# 수동 설정 (자동 감지 비활성화)
python product_crawler.py "https://m.shop.com/products" --no-auto-detect --scroll --details

# 상세 페이지는 최대 10개만
python product_crawler.py "https://m.shop.com/products" --details --max-details 10

# 다른 LLM 모델 사용
python product_crawler.py "https://m.shop.com/products" --model claude
python product_crawler.py "https://m.shop.com/products" --model gemini
python product_crawler.py "https://m.shop.com/products" --model gpt
```

**지원 모델:** `gemma`, `ax` (기본), `claude`, `gemini`, `gpt`

> **💡 팁**: URL에 `?`나 `&`가 있는 경우 반드시 따옴표로 감싸주세요!

## 상품/서비스 크롤러 사용 예제

### SKT 유니버스 상품 크롤링

```bash
python product_crawler.py \
  "https://m.sktuniverse.co.kr/category/sub/tab/detail?ctanId=CC00000013&ctgId=CA00000002" \
  --scroll \
  --scroll-count 15 \
  --output skt_products
```

결과:
```
skt_products.csv    # Excel에서 열기
skt_products.json   # API 연동
skt_products.xlsx   # 비즈니스 사용자용
```

### 상세 페이지 포함 크롤링

```bash
python product_crawler.py \
  "https://example.com/products" \
  --scroll \
  --details \
  --max-details 20 \
  --model claude \
  --output products_detailed
```

### Python 코드에서 사용

```python
from product_crawler import ProductCrawler

# 크롤러 초기화
crawler = ProductCrawler(
    base_url="https://m.shop.com/products",
    use_llm=True,
    model_name="ax"  # 또는 "claude", "gemini", "gpt", "gemma"
)

# 실행
df = crawler.run(
    url="https://m.shop.com/products",
    infinite_scroll=True,
    scroll_count=15,
    crawl_details=True,
    max_detail_pages=10,
    output_path="output/products"
)

# 결과 확인
print(f"추출된 상품: {len(df)}개")
print(df.head())

# 데이터 분석
print(df['name'].value_counts())
print(df['category'].value_counts())
```

## 출력 데이터 구조

### 상품/서비스 크롤러 출력

```csv
id,name,description,price,detail_url,category,features
1,갤럭시 S24,최신 플래그십,1200000원,https://...,스마트폰,"['5G', '120Hz']"
2,아이폰 15,애플 최신,1300000원,https://...,스마트폰,"['A17', '티타늄']"
```

**기본 컬럼:**
- `id`: 상품 ID
- `name`: 상품/서비스 이름
- `description`: 설명
- `price`: 가격
- `detail_url`: 상세 페이지 URL

**상세 페이지 크롤링 시 추가:**
- `category`: 카테고리
- `features`: 주요 특징 (리스트)
- `specifications`: 스펙 정보 (딕셔너리)

## 주요 기능 비교

| 기능 | 웹 크롤러 | 상품 크롤러 |
|------|-----------|-------------|
| 페이지 크롤링 | ✅ | ✅ |
| 무한 스크롤 | ✅ | ✅ |
| HTML/텍스트 저장 | ✅ | ✅ |
| LLM 정보 추출 | ❌ | ✅ |
| 상품 목록 파싱 | ❌ | ✅ |
| 상세 페이지 자동 방문 | ❌ | ✅ |
| 구조화된 데이터 (CSV/Excel) | ❌ | ✅ |
| DataFrame 지원 | ❌ | ✅ |

## 사용 시나리오

### 웹 크롤러 사용

- 단순히 웹 페이지 내용을 저장하고 싶을 때
- HTML/텍스트만 필요할 때
- 스크린샷이 필요할 때
- LLM 비용을 절약하고 싶을 때

### 상품 크롤러 사용

- 상품/서비스 정보를 추출하고 싶을 때
- 여러 페이지를 자동으로 방문해야 할 때
- 구조화된 데이터(CSV, Excel)가 필요할 때
- 데이터 분석이나 DB 저장이 필요할 때

## 파일 구조

```
info_builder/
├── web_crawler.py                  # 웹 크롤러
├── web_crawler_example.py          # 웹 크롤러 예제
├── WEB_CRAWLER_GUIDE.md            # 웹 크롤러 가이드
├── product_crawler.py              # 상품 크롤러
├── product_crawler_example.py      # 상품 크롤러 예제
├── PRODUCT_CRAWLER_GUIDE.md        # 상품 크롤러 가이드
├── config.py                       # 설정 파일 (LLM 모델 등)
├── ENV_SETUP.md                    # 환경 변수 설정 가이드
├── requirements.txt                # 필수 패키지
├── README.md                       # 이 파일
└── output/                         # 출력 디렉토리
    ├── *.csv                       # CSV 파일
    ├── *.json                      # JSON 파일
    └── *.xlsx                      # Excel 파일
```

## 문제 해결

### URL에 특수문자가 있을 때

```bash
# ❌ 잘못된 사용
python product_crawler.py https://example.com?id=1&type=2

# ✅ 올바른 사용
python product_crawler.py "https://example.com?id=1&type=2"
```

### LLM API 키 오류

```bash
# 환경 변수 확인
echo $LLM_API_KEY
echo $LLM_API_URL

# 설정
export LLM_API_KEY="your-key"
export LLM_API_URL="https://api.openai.com/v1"
```

상세 설정 방법은 [ENV_SETUP.md](ENV_SETUP.md)를 참고하세요.

### 추출된 상품이 없을 때

```bash
# 스크롤 횟수 증가
python product_crawler.py "URL" --scroll --scroll-count 30

# LLM 사용 확인
python product_crawler.py "URL"  # --no-llm 플래그가 없는지 확인
```

### 메모리 부족

```bash
# 상세 페이지 크롤링 수 제한
python product_crawler.py "URL" --details --max-details 10
```

## 성능 및 비용

### 크롤링 속도

- **목록 페이지**: 10-30초 (무한 스크롤 포함)
- **텍스트 청킹**: 긴 페이지는 자동으로 5,000자 단위로 분할
- **JSON 복구**: 불완전한 응답 자동 복구 시도
- **상세 페이지**: 상품당 3-5초
- **LLM 처리**: 청크당 2-5초

### LLM 비용 (참고)

**Claude 3.5 Sonnet:**
- 입력: $3/MTok
- 출력: $15/MTok

**GPT-4o-mini:**
- 입력: $0.15/MTok
- 출력: $0.6/MTok

**예상 비용 (상품 100개):**
- 목록 추출: $0.05-0.10
- 상세 페이지: $0.50-1.00

## 주의사항

⚠️ **법적 책임**: 웹사이트의 robots.txt와 이용 약관을 확인하세요  
⚠️ **속도 제한**: 과도한 크롤링은 IP 차단의 원인이 됩니다  
⚠️ **저작권**: 크롤링한 데이터의 사용 권한을 확인하세요  
⚠️ **개인정보**: 개인정보 처리에 주의하세요

### 기술적 제한사항

📌 **상세 페이지 URL (detail_url)**:
- LLM이 HTML의 `<a>` 태그 `href` 속성에서 추출합니다
- JavaScript로 처리되는 동적 링크(`javascript:void(0)`)는 추출되지 않습니다
- 이 경우 `detail_url`이 빈 문자열이 되며, `crawl_details` 옵션이 작동하지 않습니다
- **해결 방법**: 실제 href가 있는 사이트를 사용하거나, ID를 기반으로 URL을 수동으로 구성하세요

## 추가 자료

- **웹 크롤러 상세 가이드**: [WEB_CRAWLER_GUIDE.md](WEB_CRAWLER_GUIDE.md)
- **상품 크롤러 상세 가이드**: [PRODUCT_CRAWLER_GUIDE.md](PRODUCT_CRAWLER_GUIDE.md)
- **텍스트 청킹 가이드**: [CHUNKING_GUIDE.md](CHUNKING_GUIDE.md)
- **문제 해결 가이드**: [TROUBLESHOOTING.md](TROUBLESHOOTING.md) ⭐ NEW
- **환경 설정 가이드**: [ENV_SETUP.md](ENV_SETUP.md)
- **예제 코드**: `web_crawler_example.py`, `product_crawler_example.py`

## 라이선스

교육 및 연구 목적으로 제공됩니다.
