# MMS Extractor with Database Support

이 버전의 MMS Extractor는 로컬 CSV 파일뿐만 아니라 Oracle 데이터베이스에서 직접 데이터를 가져올 수 있는 기능을 지원합니다.

## 기능

- **로컬 모드**: CSV 파일에서 데이터 로드
- **데이터베이스 모드**: Oracle 데이터베이스에서 직접 데이터 조회
- **skt/gemma3-12b-it** 모델을 사용한 MMS 메시지 분석
- 한국어 형태소 분석 (Kiwi)
- 임베딩 기반 유사도 검색

## 설치 요구사항

```bash
pip install cx_Oracle pandas sentence-transformers kiwipiepy rapidfuzz langchain-openai python-dotenv
```

## 환경 변수 설정

`.env` 파일을 생성하고 다음 변수들을 설정하세요:

```bash
# API Keys
CUSTOM_API_KEY=your_custom_api_key_here
CUSTOM_BASE_URL=https://api.platform.a15t.com/v1
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Database Configuration (데이터베이스 모드 사용시)
DB_USERNAME=your_db_username
DB_PASSWORD=your_db_password
DB_HOST=your_db_host
DB_PORT=1521
DB_NAME=your_service_name

# Model Configuration
LOCAL_EMBEDDING_MODEL_PATH=./models/ko-sbert-nli
MODEL_LOADING_MODE=auto
```

## 사용법

### 1. 로컬 모드 (CSV 파일 사용)

```python
from mms_extractor import MMSExtractor

# 로컬 CSV 파일에서 데이터 로드
extractor = MMSExtractor(
    data_dir='./data', 
    offer_info_data_src='local'
)

# MMS 메시지 처리
test_message = """
[SK텔레콤] ZEM폰 포켓몬에디션3 안내
(광고)[SKT] 우리 아이 첫 번째 스마트폰...
"""

result = extractor.process_message(test_message)
print(result)
```

### 2. 데이터베이스 모드 (Oracle DB 사용)

```python
from mms_extractor import MMSExtractor

# Oracle 데이터베이스에서 데이터 로드
extractor = MMSExtractor(
    data_dir='./data', 
    offer_info_data_src='db'
)

# MMS 메시지 처리
result = extractor.process_message(test_message)
print(result)
```

### 3. 메인 스크립트 실행

`mms_extractor.py` 파일의 하단에서 데이터 소스를 변경할 수 있습니다:

```python
if __name__ == '__main__':
    # 데이터 소스 설정: "local" 또는 "db"
    offer_info_data_src = "local"  # 또는 "db"
    
    extractor = MMSExtractor(
        data_dir='./data', 
        offer_info_data_src=offer_info_data_src
    )
```

## 데이터베이스 테이블 구조

데이터베이스 모드에서는 다음 테이블에서 데이터를 조회합니다:

- **테이블명**: `TCAM_RC_OFER_MST`
- **조회 조건**: `ROWNUM <= 1000000` (최대 100만 건)

필요에 따라 `_load_data` 메서드의 SQL 쿼리를 수정하여 조회 조건을 변경할 수 있습니다.

## 출력 형식

추출된 정보는 다음과 같은 JSON 형식으로 반환됩니다:

```json
{
    "title": "광고 제목",
    "purpose": ["상품 가입 유도", "혜택 안내"],
    "product": [
        {
            "item_name_in_msg": "ZEM폰 포켓몬에디션3",
            "item_in_voca": [
                {
                    "item_nm": "ZEM폰 포켓몬에디션3",
                    "item_id": ["A5V5"]
                }
            ]
        }
    ],
    "channel": [...],
    "pgm": [...]
}
```

## API 서버 사용법

### 1. API 서버 시작

```bash
# 기본 설정으로 서버 시작 (포트 8000)
python api.py

# 커스텀 설정으로 서버 시작
python api.py --host 0.0.0.0 --port 8080 --debug

# 테스트 모드로 실행
python api.py --test --data-source local
```

### 2. API 엔드포인트

#### Health Check
```bash
GET /health
```

#### 모델 정보 조회
```bash
GET /models
```

#### 단일 메시지 추출
```bash
POST /extract
Content-Type: application/json

{
    "message": "MMS 메시지 내용",
    "offer_info_data_src": "local"  // 또는 "db"
}
```

#### 배치 메시지 추출
```bash
POST /extract/batch
Content-Type: application/json

{
    "messages": ["메시지1", "메시지2", "메시지3"],
    "offer_info_data_src": "local"
}
```

#### 서버 상태 확인
```bash
GET /status
```

### 3. API 테스트

```bash
# API 예제 실행 (서버가 실행 중이어야 함)
python api_examples.py
```

## 주의사항

1. **Oracle 클라이언트**: `cx_Oracle`을 사용하기 위해서는 Oracle Instant Client가 설치되어 있어야 합니다.
2. **환경 변수**: 데이터베이스 모드 사용시 `.env` 파일에 올바른 DB 접속 정보가 설정되어야 합니다.
3. **성능**: 데이터베이스에서 대량의 데이터를 조회할 때는 네트워크 상태와 DB 성능에 따라 초기화 시간이 길어질 수 있습니다.
4. **API 서버**: 첫 번째 요청 시 모델 로딩으로 인해 응답 시간이 길 수 있습니다.

## 문제 해결

### cx_Oracle 설치 오류
```bash
# Oracle Instant Client 설치 후
pip install cx_Oracle
```

### 데이터베이스 연결 오류
- `.env` 파일의 DB 접속 정보를 확인하세요
- 네트워크 연결 및 방화벽 설정을 확인하세요
- Oracle 서비스 상태를 확인하세요

### API 연결 오류
- API 서버가 실행 중인지 확인하세요
- 포트 번호가 올바른지 확인하세요
- 방화벽 설정을 확인하세요 