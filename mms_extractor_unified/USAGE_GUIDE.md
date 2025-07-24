# MMS Extractor 사용 가이드

MMS Extractor는 한국어 광고 메시지에서 상품 정보, 채널 정보, 프로그램 분류 등을 자동으로 추출하는 시스템입니다.

## 목차
- [시스템 개요](#시스템-개요)
- [설치 및 설정](#설치-및-설정)
- [CLI 사용법 (mms_extractor.py)](#cli-사용법-mms_extractorpy)
- [API 서버 사용법 (api.py)](#api-서버-사용법-apipy)
- [설정 옵션](#설정-옵션)
- [예제](#예제)
- [문제 해결](#문제-해결)

## 시스템 개요

### 주요 기능
- **상품 정보 추출**: 광고 메시지에서 제품/서비스 이름과 고객 행동 추출
- **채널 정보 추출**: URL, 전화번호, 앱, 대리점 정보 추출
- **프로그램 분류**: 광고 메시지를 사전 정의된 프로그램으로 분류
- **다중 LLM 지원**: Gemma, GPT, Claude 모델 지원
- **유연한 데이터 소스**: 로컬 CSV 파일 또는 Oracle 데이터베이스 지원

### 시스템 아키텍처
```
MMS 메시지 입력
    ↓
형태소 분석 (Kiwi)
    ↓
임베딩 기반 유사도 검색
    ↓
LLM 기반 정보 추출
    ↓
엔티티 매칭 및 검증
    ↓
구조화된 결과 출력
```

## 설치 및 설정

### 1. 필수 패키지 설치
```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정
`.env` 파일을 생성하고 다음 변수들을 설정하세요:

```bash
# LLM API 설정
CUSTOM_API_KEY=your_gemma_api_key_here
CUSTOM_BASE_URL=https://api.platform.a15t.com/v1
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# 데이터베이스 설정 (DB 사용시)
DB_USERNAME=your_db_username
DB_PASSWORD=your_db_password
DB_HOST=your_db_host
DB_PORT=1521
DB_NAME=your_service_name

# 모델 설정
LLM_MODEL=gemma
LOCAL_EMBEDDING_MODEL_PATH=./models/ko-sbert-nli
MODEL_LOADING_MODE=auto
```

### 3. 데이터 파일 준비
`./data/` 디렉토리에 다음 파일들이 필요합니다:
- `item_info_all_250527.csv`: 상품 정보
- `alias_rules.csv`: 별칭 규칙
- `stop_words.csv`: 불용어 목록
- `pgm_tag_ext_250516.csv`: 프로그램 분류 정보
- `org_info_all_250605.csv`: 조직/대리점 정보

### 4. 모델 파일 준비
`./models/` 디렉토리에 임베딩 모델을 다운로드하세요:
```bash
# 모델 다운로드 스크립트 실행 (있는 경우)
python download_model.py
```

## CLI 사용법 (mms_extractor.py)

### 기본 사용법
```bash
python mms_extractor.py --message "광고 메시지 텍스트"
```

### 전체 옵션
```bash
python mms_extractor.py \
  --message "광고 메시지" \
  --offer-data-source local \
  --llm-model gemma \
  --product-info-extraction-mode nlp \
  --entity-matching-mode logic
```

### 옵션 설명
- `--message`: 처리할 MMS 메시지 텍스트
- `--offer-data-source`: 데이터 소스 (`local` 또는 `db`)
- `--llm-model`: 사용할 LLM 모델 (`gemma`, `gpt`, `claude`)
- `--product-info-extraction-mode`: 상품 정보 추출 모드 (`nlp`, `llm`, `rag`)
- `--entity-matching-mode`: 엔티티 매칭 모드 (`logic`, `llm`)

### 예제
```bash
# 기본 설정으로 실행
python mms_extractor.py --message "[SK텔레콤] 5G 요금제 안내 메시지입니다."

# GPT 모델과 RAG 모드 사용
python mms_extractor.py \
  --message "[SK텔레콤] 5G 요금제 안내" \
  --llm-model gpt \
  --product-info-extraction-mode rag

# 데이터베이스 사용
python mms_extractor.py \
  --message "광고 메시지" \
  --offer-data-source db
```

## API 서버 사용법 (api.py)

### 서버 시작

#### 기본 서버 시작
```bash
python api.py
```

#### 고급 옵션으로 서버 시작
```bash
python api.py \
  --host 0.0.0.0 \
  --port 8000 \
  --offer-data-source local \
  --debug
```

#### 테스트 모드
```bash
python api.py --test --message "테스트 메시지"
```

### API 엔드포인트

#### 1. 헬스 체크
```bash
GET /health
```

**응답 예제:**
```json
{
  "status": "healthy",
  "service": "MMS Extractor API",
  "version": "2.0.0",
  "timestamp": 1640995200.0
}
```

#### 2. 모델 정보 조회
```bash
GET /models
```

**응답 예제:**
```json
{
  "available_llm_models": ["gemma", "gpt", "claude"],
  "default_llm_model": "gemma",
  "available_data_sources": ["local", "db"],
  "default_data_source": "local",
  "available_product_info_extraction_modes": ["nlp", "llm", "rag"],
  "default_product_info_extraction_mode": "nlp",
  "available_entity_matching_modes": ["logic", "llm"],
  "default_entity_matching_mode": "logic",
  "features": [
    "Korean morphological analysis (Kiwi)",
    "Embedding-based similarity search",
    "Entity extraction and matching",
    "Program classification",
    "Multiple LLM support (Gemma, GPT, Claude)"
  ]
}
```

#### 3. 서버 상태 조회
```bash
GET /status
```

**응답 예제:**
```json
{
  "status": "running",
  "extractor": {
    "initialized": true,
    "data_source": "local",
    "current_llm_model": "gemma",
    "current_product_mode": "nlp",
    "current_entity_mode": "logic"
  },
  "timestamp": 1640995200.0
}
```

#### 4. 단일 메시지 처리
```bash
POST /extract
Content-Type: application/json
```

**요청 예제:**
```json
{
  "message": "[SK텔레콤] 5G 요금제 가입 안내",
  "llm_model": "gemma",
  "product_info_extraction_mode": "nlp",
  "entity_matching_mode": "logic",
  "offer_info_data_src": "local"
}
```

**응답 예제:**
```json
{
  "success": true,
  "result": {
    "title": "SK텔레콤 5G 요금제 가입 안내",
    "purpose": ["상품 가입 유도"],
    "product": [
      {
        "item_name_in_msg": "5G 요금제",
        "item_in_voca": [
          {
            "item_nm": "5G 프리미엄 요금제",
            "item_id": ["5G_PREMIUM_001"]
          }
        ]
      }
    ],
    "channel": [],
    "pgm": [
      {
        "pgm_nm": "통신서비스",
        "pgm_id": "TELECOM_001"
      }
    ]
  },
  "metadata": {
    "llm_model": "gemma",
    "offer_info_data_src": "local",
    "product_info_extraction_mode": "nlp",
    "entity_matching_mode": "logic",
    "processing_time_seconds": 2.456,
    "timestamp": 1640995200.0,
    "message_length": 25
  }
}
```

#### 5. 배치 처리
```bash
POST /extract/batch
Content-Type: application/json
```

**요청 예제:**
```json
{
  "messages": [
    "[SK텔레콤] 5G 요금제 안내",
    "[LG유플러스] 인터넷 서비스 안내"
  ],
  "llm_model": "gpt",
  "product_info_extraction_mode": "rag",
  "entity_matching_mode": "llm",
  "offer_info_data_src": "local"
}
```

**응답 예제:**
```json
{
  "success": true,
  "results": [
    {
      "index": 0,
      "success": true,
      "result": { /* 추출 결과 */ }
    },
    {
      "index": 1,
      "success": true,
      "result": { /* 추출 결과 */ }
    }
  ],
  "summary": {
    "total_messages": 2,
    "successful": 2,
    "failed": 0
  },
  "metadata": {
    "llm_model": "gpt",
    "offer_info_data_src": "local",
    "product_info_extraction_mode": "rag",
    "entity_matching_mode": "llm",
    "processing_time_seconds": 4.123,
    "timestamp": 1640995200.0
  }
}
```

### cURL 예제

#### 단일 메시지 처리
```bash
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{
    "message": "[SK텔레콤] 5G 요금제 가입하고 혜택 받으세요!",
    "llm_model": "gemma",
    "product_info_extraction_mode": "nlp",
    "entity_matching_mode": "logic"
  }'
```

#### 배치 처리
```bash
curl -X POST http://localhost:8000/extract/batch \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      "[SK텔레콤] 5G 요금제 안내",
      "[KT] 인터넷 서비스 안내"
    ],
    "llm_model": "gpt"
  }'
```

## 설정 옵션

### LLM 모델 선택
- **gemma**: 기본 모델, 빠른 처리 속도
- **gpt**: OpenAI GPT 모델, 높은 정확도
- **claude**: Anthropic Claude 모델, 균형잡힌 성능

### 상품 정보 추출 모드
- **nlp**: 규칙 기반 추출 (기본값)
- **llm**: LLM 기반 추출
- **rag**: RAG 기반 추출 (후보 상품 목록 활용)

### 엔티티 매칭 모드
- **logic**: 논리 기반 유사도 매칭 (기본값)
- **llm**: LLM 기반 매칭

### 데이터 소스
- **local**: 로컬 CSV 파일 사용 (기본값)
- **db**: Oracle 데이터베이스 사용

## 예제

### 완전한 사용 예제

#### 1. 서버 시작
```bash
python api.py --host 0.0.0.0 --port 8000 --offer-data-source local
```

#### 2. Python 클라이언트 예제
```python
import requests
import json

# API 서버 URL
API_URL = "http://localhost:8000"

# 단일 메시지 처리
def extract_single_message(message):
    response = requests.post(
        f"{API_URL}/extract",
        json={
            "message": message,
            "llm_model": "gemma",
            "product_info_extraction_mode": "nlp",
            "entity_matching_mode": "logic"
        }
    )
    return response.json()

# 사용 예제
message = """
[SK텔레콤] 5G 프리미엄 요금제 안내
월 89,000원으로 무제한 데이터 이용하세요!
가입 문의: 1588-0011
"""

result = extract_single_message(message)
print(json.dumps(result, ensure_ascii=False, indent=2))
```

#### 3. JavaScript 클라이언트 예제
```javascript
const API_URL = 'http://localhost:8000';

async function extractMessage(message) {
  const response = await fetch(`${API_URL}/extract`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      message: message,
      llm_model: 'gemma',
      product_info_extraction_mode: 'nlp',
      entity_matching_mode: 'logic'
    })
  });
  
  return await response.json();
}

// 사용 예제
const message = '[SK텔레콤] 5G 요금제 가입 안내';
extractMessage(message)
  .then(result => console.log(result))
  .catch(error => console.error('Error:', error));
```

## 문제 해결

### 일반적인 오류

#### 1. "Extractor not initialized" 오류
**원인**: 서버 시작 시 데이터 로딩 실패
**해결방법**: 
- 데이터 파일들이 `./data/` 디렉토리에 있는지 확인
- 환경 변수가 올바르게 설정되었는지 확인
- 서버 로그를 확인하여 구체적인 오류 메시지 확인

#### 2. "Invalid llm_model" 오류
**원인**: 지원하지 않는 LLM 모델 지정
**해결방법**: `gemma`, `gpt`, `claude` 중 하나를 사용

#### 3. API 키 오류
**원인**: LLM API 키가 설정되지 않음
**해결방법**: `.env` 파일에 해당 모델의 API 키 설정

#### 4. 메모리 부족 오류
**원인**: 임베딩 모델 로딩 시 메모리 부족
**해결방법**: 
- 더 많은 RAM이 있는 환경에서 실행
- `MODEL_LOADING_MODE=local`로 설정하여 로컬 모델 사용

### 성능 최적화

#### 1. 응답 시간 개선
- 서버 시작 시 데이터가 한 번만 로드되므로 첫 요청 후 빨라짐
- GPU 사용 가능한 환경에서 실행 (CUDA 또는 MPS)
- 배치 처리 사용으로 여러 메시지 동시 처리

#### 2. 메모리 사용량 최적화
- 불필요한 데이터 파일 제거
- 배치 크기 조정 (`batch_size` 파라미터)

#### 3. 정확도 개선
- RAG 모드 사용으로 후보 상품 목록 활용
- GPT 또는 Claude 모델 사용
- 도메인별 불용어 목록 업데이트

### 로그 확인
서버 실행 시 로그를 통해 상태를 확인할 수 있습니다:
```bash
python api.py --debug  # 디버그 모드로 실행
```

### 지원 및 문의
- 시스템 관련 문의: 개발팀 연락처
- 버그 리포트: GitHub Issues
- 문서 업데이트: 최신 버전 확인

---

**마지막 업데이트**: 2025년 7월
**버전**: 1.0.0 