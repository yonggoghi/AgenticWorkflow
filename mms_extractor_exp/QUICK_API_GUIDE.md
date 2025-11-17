# Quick Extractor API 사용 가이드

## 개요

Quick Extractor API는 MMS 메시지에서 **제목**과 **수신거부 전화번호**를 빠르게 추출하는 경량화된 API입니다.

### 주요 특징

- ✅ **빠른 처리**: NLP 기반 방법으로 초당 수백 개 메시지 처리
- ✅ **고품질 LLM 옵션**: 필요 시 LLM 기반 제목 추출 지원
- ✅ **RESTful API**: 표준 HTTP/JSON 프로토콜
- ✅ **배치 처리 지원**: 여러 메시지 일괄 처리

## API 서버 시작

```bash
# 기본 실행 (포트 8000)
cd /Users/yongwook/workspace/AgenticWorkflow/mms_extractor_exp
source ../venv/bin/activate
python api.py

# 특정 포트 지정
python api.py --port 8080

# 외부 접근 허용
python api.py --host 0.0.0.0 --port 8000
```

## API 엔드포인트

### 1. 단일 메시지 처리 (`POST /quick/extract`)

단일 메시지에서 제목과 수신거부 번호를 추출합니다.

#### 요청 예시

```bash
curl -X POST http://localhost:8000/quick/extract \
  -H "Content-Type: application/json" \
  -d '{
    "message": "[SKT] 5G 요금제 변경 시 3개월간 50% 할인!\n고객님, 안녕하세요.\n지금 5G 프리미엄 요금제로 변경하시면\n무료 수신거부 1504",
    "method": "textrank"
  }'
```

#### Python 요청 예시

```python
import requests

url = "http://localhost:8000/quick/extract"
payload = {
    "message": "[SKT] 5G 요금제 변경 시 3개월간 50% 할인!...",
    "method": "textrank"  # 'textrank', 'tfidf', 'first_bracket', 'llm'
}

response = requests.post(url, json=payload)
result = response.json()

print(f"제목: {result['data']['title']}")
print(f"수신거부: {result['data']['unsubscribe_phone']}")
```

#### LLM 사용 예시

```bash
curl -X POST http://localhost:8000/quick/extract \
  -H "Content-Type: application/json" \
  -d '{
    "message": "[SKT] 5G 요금제 변경...",
    "method": "llm",
    "llm_model": "ax"
  }'
```

#### 응답 예시

```json
{
  "success": true,
  "data": {
    "title": "5G 프리미엄 요금제 변경 시 3개월간 50% 할인, 데이터 2배, 최신 스마트폰 할인 제공",
    "unsubscribe_phone": "1504",
    "message_preview": "[SKT] 5G 요금제 변경 시 3개월간 50% 할인!\n고객님, 안녕하세요.\n지금 5G 프리미엄 요금제로 변경하시면\n..."
  },
  "metadata": {
    "method": "llm",
    "message_length": 188,
    "processing_time_seconds": 0.234,
    "timestamp": 1699876543.123
  }
}
```

### 2. 배치 메시지 처리 (`POST /quick/extract/batch`)

여러 메시지를 일괄 처리합니다.

#### 요청 예시

```bash
curl -X POST http://localhost:8000/quick/extract/batch \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      "[광고]\nSK텔레콤\n개인고객센터/변경해지",
      "[SKT] 2월 T Day 이벤트 안내",
      "5G 프리미엄 요금제 가입 시 특별 혜택!\n무료 수신거부 1504"
    ],
    "method": "textrank"
  }'
```

#### Python 요청 예시

```python
import requests

url = "http://localhost:8000/quick/extract/batch"
payload = {
    "messages": [
        "[광고]\nSK텔레콤\n개인고객센터/변경해지",
        "[SKT] 2월 T Day 이벤트 안내",
        "5G 프리미엄 요금제 가입 시 특별 혜택!\n무료 수신거부 1504"
    ],
    "method": "textrank"
}

response = requests.post(url, json=payload)
result = response.json()

print(f"총 메시지: {result['data']['statistics']['total_messages']}")
print(f"수신거부 추출률: {result['data']['statistics']['extraction_rate']}%")

for item in result['data']['results']:
    print(f"[{item['msg_id']}] {item['title']} - {item['unsubscribe_phone']}")
```

#### 응답 예시

```json
{
  "success": true,
  "data": {
    "results": [
      {
        "msg_id": 0,
        "title": "SK텔레콤",
        "unsubscribe_phone": null,
        "message_preview": "[광고]\nSK텔레콤\n개인고객센터/변경해지"
      },
      {
        "msg_id": 1,
        "title": "2월 T Day 이벤트 안내",
        "unsubscribe_phone": null,
        "message_preview": "[SKT] 2월 T Day 이벤트 안내"
      },
      {
        "msg_id": 2,
        "title": "5G 프리미엄 요금제 가입 시 특별 혜택!",
        "unsubscribe_phone": "1504",
        "message_preview": "5G 프리미엄 요금제 가입 시 특별 혜택!\n무료 수신거부 1504"
      }
    ],
    "statistics": {
      "total_messages": 3,
      "with_unsubscribe_phone": 1,
      "extraction_rate": 33.33
    }
  },
  "metadata": {
    "method": "textrank",
    "processing_time_seconds": 0.045,
    "avg_time_per_message": 0.015,
    "timestamp": 1699876543.456
  }
}
```

## 요청 파라미터

### 공통 파라미터

| 파라미터 | 타입 | 필수 | 기본값 | 설명 |
|---------|------|------|--------|------|
| `method` | string | 선택 | `textrank` | 제목 추출 방법 |
| `use_llm` | boolean | 선택 | `false` | LLM 사용 여부 (method가 'llm'이면 자동 true) |
| `llm_model` | string | 선택 | `ax` | LLM 모델 선택 |

### Method 옵션

| Method | 속도 | 품질 | 설명 |
|--------|------|------|------|
| `textrank` | ⚡⚡⚡ | ⭐⭐⭐ | NLP 기반 문장 중요도 분석 (권장) |
| `tfidf` | ⚡⚡⚡ | ⭐⭐ | 단어 빈도 기반 추출 |
| `first_bracket` | ⚡⚡⚡ | ⭐ | 첫 번째 대괄호 내용 추출 |
| `llm` | ⚡ | ⭐⭐⭐⭐⭐ | LLM 기반 지능형 제목 생성 |

### LLM Model 옵션

| 모델 코드 | 실제 모델 | 설명 |
|----------|----------|------|
| `ax` | `skt/ax4` | SKT AX 모델 (기본값) |
| `gpt` | `azure/openai/gpt-4o-2024-08-06` | GPT-4 |
| `claude` / `cld` | `amazon/anthropic/claude-sonnet-4-20250514` | Claude Sonnet |
| `gemini` / `gen` | `gcp/gemini-2.5-flash` | Gemini Flash |

## 에러 응답

### 400 Bad Request

```json
{
  "success": false,
  "error": "'message' 필드는 필수입니다."
}
```

### 500 Internal Server Error

```json
{
  "success": false,
  "error": "Internal server error message",
  "timestamp": 1699876543.789
}
```

## 성능 가이드

### NLP 방법 (textrank, tfidf, first_bracket)

- **처리 속도**: 약 0.01-0.05초/메시지
- **적합한 경우**: 
  - 대량 메시지 빠른 처리
  - 실시간 응답 필요
  - 리소스 제약 환경

### LLM 방법

- **처리 속도**: 약 0.5-2초/메시지 (모델에 따라 다름)
- **적합한 경우**:
  - 고품질 제목 필요
  - 복잡한 메시지 구조
  - 문맥 이해 중요

## 통합 예시

### Flask 애플리케이션 통합

```python
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)
QUICK_API_URL = "http://localhost:8000/quick/extract"

@app.route('/process_message', methods=['POST'])
def process_message():
    data = request.get_json()
    message = data.get('message')
    
    # Quick Extractor API 호출
    response = requests.post(QUICK_API_URL, json={
        'message': message,
        'method': 'textrank'
    })
    
    result = response.json()
    
    if result['success']:
        return jsonify({
            'title': result['data']['title'],
            'unsubscribe': result['data']['unsubscribe_phone']
        })
    else:
        return jsonify({'error': result['error']}), 500

if __name__ == '__main__':
    app.run(port=5000)
```

### FastAPI 애플리케이션 통합

```python
from fastapi import FastAPI, HTTPException
import httpx

app = FastAPI()
QUICK_API_URL = "http://localhost:8000/quick/extract"

@app.post("/analyze")
async def analyze_message(message: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(QUICK_API_URL, json={
            'message': message,
            'method': 'textrank'
        })
        
        result = response.json()
        
        if result['success']:
            return {
                'title': result['data']['title'],
                'unsubscribe': result['data']['unsubscribe_phone']
            }
        else:
            raise HTTPException(status_code=500, detail=result['error'])
```

## 테스트

### 자동 테스트 실행

```bash
# API 서버가 실행 중인 상태에서
python test_quick_api.py
```

### 수동 테스트 (curl)

```bash
# 서버 상태 확인
curl http://localhost:8000/health

# 단일 메시지 테스트
curl -X POST http://localhost:8000/quick/extract \
  -H "Content-Type: application/json" \
  -d '{"message": "테스트 메시지", "method": "textrank"}'

# 배치 메시지 테스트
curl -X POST http://localhost:8000/quick/extract/batch \
  -H "Content-Type: application/json" \
  -d '{"messages": ["메시지1", "메시지2"], "method": "textrank"}'
```

## 참고

- Quick Extractor는 `mms_extractor.py`와 동일한 설정 시스템(`config/settings.py`)을 사용합니다.
- LLM 사용 시 `.env` 파일에 API 키가 설정되어 있어야 합니다.
- 자세한 설정은 `README_QUICK_EXTRACTOR.md`를 참고하세요.

## 관련 파일

- `quick_extractor.py`: 핵심 추출 로직
- `api.py`: Flask API 서버
- `test_quick_api.py`: 자동 테스트 스크립트
- `README_QUICK_EXTRACTOR.md`: Quick Extractor 사용 가이드

