# 웹 크롤러 사용 가이드

## 개요

`web_crawler.py`는 Playwright를 사용하여 웹 페이지를 크롤링하는 도구입니다. JavaScript로 렌더링되는 동적 페이지도 크롤링할 수 있습니다.

## 설치

### 1. Python 패키지 설치

```bash
pip install playwright
```

### 2. 브라우저 설치

Playwright가 사용할 브라우저를 설치합니다:

```bash
playwright install chromium
```

또는 모든 브라우저를 설치:

```bash
playwright install
```

## 사용법

### 기본 사용

```bash
python web_crawler.py https://example.com
```

이 명령은:
- 웹 페이지를 크롤링합니다
- `crawler_output/` 디렉토리에 결과를 저장합니다
- 텍스트 (.txt), HTML (.html), 메타데이터 (.json) 파일을 생성합니다

### 옵션

#### 1. 스크린샷 저장

```bash
python web_crawler.py https://example.com --screenshot
```

페이지의 전체 스크린샷을 PNG 파일로 저장합니다.

#### 2. 무한 스크롤 페이지 크롤링 ⭐️ NEW

```bash
python web_crawler.py https://example.com --scroll
```

무한 스크롤이나 레이지 로딩을 사용하는 페이지를 크롤링합니다. 자동으로 페이지를 스크롤하면서 동적으로 로드되는 콘텐츠를 모두 가져옵니다.

```bash
# 최대 20회까지 스크롤
python web_crawler.py https://example.com --scroll --scroll-count 20
```

`--scroll-count`로 최대 스크롤 횟수를 지정할 수 있습니다. (기본값: 10회)

#### 3. 페이지 로딩 대기 시간 조절

```bash
python web_crawler.py https://example.com --wait 5000
```

JavaScript 렌더링 완료를 위해 대기하는 시간을 밀리초 단위로 지정합니다. (기본값: 2000ms)

#### 4. 출력 디렉토리 지정

```bash
python web_crawler.py https://example.com --output my_output
```

결과 파일을 저장할 디렉토리를 지정합니다.

#### 5. 화면에 텍스트 출력

```bash
python web_crawler.py https://example.com --print-text
```

크롤링한 텍스트 내용을 화면에 출력합니다 (처음 2000자).

#### 6. 파일 저장 없이 출력만

```bash
python web_crawler.py https://example.com --no-save
```

파일로 저장하지 않고 화면에만 결과를 출력합니다.

### 조합 예시

```bash
# 무한 스크롤 페이지 크롤링
python web_crawler.py "https://m.shop.com/products?category=1" --scroll --scroll-count 20

# 스크린샷과 함께 크롤링, 5초 대기
python web_crawler.py https://example.com --screenshot --wait 5000

# 무한 스크롤 + 스크린샷 + 사용자 지정 출력 디렉토리
python web_crawler.py https://example.com --scroll --screenshot --output results

# 파일 저장 없이 화면에만 출력
python web_crawler.py https://example.com --no-save --print-text
```

## 출력 파일

크롤링 결과는 다음 파일들로 저장됩니다:

### 1. 텍스트 파일 (`*.txt`)

```
URL: https://example.com
Title: Example Domain
Timestamp: 2025-01-15T10:30:45.123456
Success: True

================================================================================
텍스트 내용:
================================================================================

[페이지의 텍스트 내용]
```

### 2. HTML 파일 (`*.html`)

원본 HTML 소스 코드 전체

### 3. 메타데이터 JSON 파일 (`*_metadata.json`)

```json
{
  "url": "https://example.com",
  "title": "Example Domain",
  "timestamp": "2025-01-15T10:30:45.123456",
  "success": true,
  "error": null,
  "text_length": 1234,
  "links_count": 5,
  "images_count": 3,
  "links": [
    {
      "text": "More information...",
      "href": "https://www.iana.org/domains/example"
    }
  ],
  "images": [
    {
      "alt": "Example image",
      "src": "https://example.com/image.png"
    }
  ]
}
```

### 4. 스크린샷 (`screenshots/*.png`)

`--screenshot` 옵션 사용 시 전체 페이지의 스크린샷

## 기능

### 수집되는 정보

1. **기본 정보**
   - 페이지 제목
   - URL
   - 타임스탬프

2. **콘텐츠**
   - 전체 텍스트 내용 (body 태그 내)
   - 원본 HTML 소스

3. **메타데이터**
   - 모든 링크 (텍스트와 URL)
   - 모든 이미지 (alt 텍스트와 src)
   - 링크 및 이미지 개수

4. **스크린샷** (선택적)
   - 전체 페이지 PNG 이미지

### 특징

- **동적 페이지 지원**: JavaScript로 렌더링되는 페이지도 크롤링 가능
- **무한 스크롤 지원** ⭐️: 스크롤하면서 동적으로 로드되는 콘텐츠 자동 수집
- **자동 대기**: 페이지 로딩 완료 후 추가 대기 시간 설정 가능
- **다양한 출력 형식**: 텍스트, HTML, JSON, 스크린샷
- **안전한 파일명**: URL을 파일명으로 자동 변환
- **에러 처리**: 크롤링 실패 시 에러 정보 저장
- **스마트 스크롤**: 더 이상 새 콘텐츠가 없으면 자동 중단

## 문제 해결

### Playwright 설치 오류

```bash
# Playwright 재설치
pip uninstall playwright
pip install playwright
playwright install chromium
```

### 크롤링 타임아웃

페이지 로딩이 느린 경우 대기 시간을 늘립니다:

```bash
python web_crawler.py https://example.com --wait 10000
```

### 무한 스크롤 페이지에서 콘텐츠가 부족한 경우

스크롤 횟수를 늘립니다:

```bash
python web_crawler.py https://example.com --scroll --scroll-count 30
```

### 메모리 부족

대용량 페이지의 경우 스크린샷 없이 크롤링:

```bash
python web_crawler.py https://example.com  # --screenshot 옵션 제외
```

## Python 코드에서 사용

### 기본 사용

```python
from web_crawler import crawl_with_playwright, save_results

# 크롤링 실행
result = crawl_with_playwright('https://example.com', wait_time=2000, screenshot=True)

if result['success']:
    print(f"제목: {result['title']}")
    print(f"텍스트 길이: {len(result['text_content'])}")
    print(f"링크 수: {len(result['links'])}")
    
    # 결과 저장
    save_results(result, output_dir='my_output')
else:
    print(f"오류: {result['error']}")
```

### 무한 스크롤 페이지 크롤링

```python
from web_crawler import crawl_with_playwright, save_results

# 무한 스크롤 페이지 크롤링
result = crawl_with_playwright(
    'https://m.shop.com/products',
    wait_time=2000,
    screenshot=False,
    infinite_scroll=True,      # 무한 스크롤 활성화
    scroll_count=20            # 최대 20회 스크롤
)

if result['success']:
    print(f"제목: {result['title']}")
    print(f"텍스트 길이: {len(result['text_content'])}")
    print(f"스크롤 횟수: {result['scroll_iterations']}")
    
    # 결과 저장
    save_results(result, output_dir='my_output')
```

## 주의사항

1. **법적 책임**: 웹사이트의 robots.txt 및 이용 약관을 확인하고 준수하세요.
2. **속도 제한**: 과도한 크롤링은 서버에 부담을 줄 수 있습니다.
3. **저작권**: 크롤링한 데이터의 사용에 주의하세요.
4. **개인정보**: 개인정보가 포함된 페이지 크롤링 시 주의하세요.

## 라이선스

이 도구는 교육 및 연구 목적으로 제공됩니다.

