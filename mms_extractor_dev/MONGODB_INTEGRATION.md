# MMS Extractor MongoDB 통합 가이드

## 개요
MMS Extractor에서 추출한 결과를 MongoDB에 자동으로 저장하는 기능입니다.

## 설치 및 설정

### 1. 필요한 패키지 설치
```bash
pip install pymongo
```

### 2. MongoDB 서버 실행
MongoDB 서버가 로컬에서 실행 중이어야 합니다.
```bash
mongod --dbpath ~/data/db
```

### 3. 파일 구성
- `mongodb_utils.py`: MongoDB 연결 및 저장 유틸리티
- `demo_streamlit.py`: MongoDB 저장 기능이 통합된 Streamlit 앱
- `test_mongodb.py`: MongoDB 통합 테스트 스크립트

## 데이터베이스 구조

### 데이터베이스 정보
- **데이터베이스명**: `aos`
- **컬렉션명**: `mmsext`

### 문서 스키마
```javascript
{
  "_id": ObjectId("..."),
  "message": "원본 MMS 메시지",
  "main_prompt": {
    "title": "메인 정보 추출 프롬프트",
    "description": "프롬프트 설명",
    "content": "실제 프롬프트 내용",
    "length": 500
  },
  "ent_prompt": {
    "title": "엔티티 추출 프롬프트",
    "description": "프롬프트 설명", 
    "content": "실제 프롬프트 내용",
    "length": 300
  },
  "dag_prompt": {
    "title": "DAG 관계 추출 프롬프트",
    "description": "프롬프트 설명",
    "content": "실제 프롬프트 내용", 
    "length": 400
  },
  "ext_result": {
    "title": "추출된 제목",
    "purpose": "메시지 목적",
    "product": "상품 정보",
    "channel": "채널 정보",
    "program": "프로그램 정보",
    // ... 기타 추출된 정보
  },
  "metadata": {
    "timestamp": ISODate("2025-09-17T02:36:08.170Z"),
    "processing_time": 2.5,
    "success": true,
    "settings": {
      "llm_model": "claude",
      "data_source": "local",
      "entity_matching_mode": "logic",
      "extract_entity_dag": true
    },
    "api_response_keys": ["success", "result", "metadata"],
    "prompts_available": ["main_extraction_prompt", "entity_extraction_prompt", "dag_extraction_prompt"]
  }
}
```

## 사용법

### 1. Streamlit 앱에서 자동 저장
```bash
cd /path/to/mms_extractor_exp
streamlit run demo_streamlit.py
```

- 메시지 추출 작업 완료 시 자동으로 MongoDB에 저장됩니다.
- 사이드바에서 MongoDB 상태를 확인할 수 있습니다.
- 저장 성공 시 문서 ID가 표시됩니다.

### 2. 프로그래밍 방식으로 저장
```python
from mongodb_utils import save_to_mongodb

# 추출 결과 저장
message = "원본 MMS 메시지"
extraction_result = {...}  # API 응답
extraction_prompts = {...}  # 프롬프트 정보

saved_id = save_to_mongodb(message, extraction_result, extraction_prompts)
print(f"저장된 문서 ID: {saved_id}")
```

### 3. 저장된 데이터 조회
```python
from mongodb_utils import MongoDBManager

manager = MongoDBManager()
manager.connect()

# 최근 데이터 조회
recent_data = manager.get_recent_extractions(limit=10)

# 통계 정보 조회
stats = manager.get_extraction_stats()

manager.disconnect()
```

## 테스트

### MongoDB 연결 테스트
```bash
python test_mongodb.py
```

테스트 스크립트는 다음을 확인합니다:
1. MongoDB 연결 상태
2. 샘플 데이터 저장 기능
3. 저장된 데이터 조회 기능
4. 통계 정보 조회 기능

### MongoDB Shell에서 직접 확인
```bash
# 데이터베이스 연결
mongosh

# 데이터베이스 선택
use aos

# 모든 문서 조회
db.mmsext.find().pretty()

# 문서 개수 확인
db.mmsext.countDocuments()

# 최근 문서 조회
db.mmsext.find().sort({_id: -1}).limit(5)
```

## 주요 기능

### 1. 자동 저장
- 추출 작업 완료 시 자동으로 MongoDB에 저장
- 실패 시에도 적절한 오류 메시지 표시

### 2. 구조화된 데이터
- 원본 메시지, 프롬프트, 추출 결과를 구조화하여 저장
- 메타데이터를 통한 추적 가능

### 3. 통계 및 모니터링
- 총 저장 건수, 성공률 등 통계 정보 제공
- Streamlit 사이드바에서 실시간 상태 확인

### 4. 오류 처리
- 연결 실패, 저장 실패 등에 대한 적절한 오류 처리
- 로그를 통한 디버깅 지원

## 문제 해결

### MongoDB 연결 실패
1. MongoDB 서버가 실행 중인지 확인
2. 연결 문자열이 올바른지 확인 (기본: `mongodb://localhost:27017/`)
3. 방화벽 설정 확인

### 저장 실패
1. 데이터베이스 권한 확인
2. 디스크 공간 확인
3. 로그 파일에서 상세 오류 메시지 확인

### 성능 최적화
1. 인덱스 생성: `db.mmsext.createIndex({"metadata.timestamp": -1})`
2. 정기적인 데이터 정리 작업 설정
3. 연결 풀링 설정 고려

## 확장 가능성

### 1. 추가 인덱스
```javascript
// 메시지 내용 검색을 위한 텍스트 인덱스
db.mmsext.createIndex({"message": "text"})

// 성공 여부별 조회를 위한 인덱스  
db.mmsext.createIndex({"metadata.success": 1})

// 모델별 조회를 위한 인덱스
db.mmsext.createIndex({"metadata.settings.llm_model": 1})
```

### 2. 집계 파이프라인
```javascript
// 모델별 성공률 통계
db.mmsext.aggregate([
  {$group: {
    _id: "$metadata.settings.llm_model",
    total: {$sum: 1},
    success: {$sum: {$cond: ["$metadata.success", 1, 0]}}
  }},
  {$project: {
    model: "$_id",
    total: 1,
    success: 1,
    success_rate: {$multiply: [{$divide: ["$success", "$total"]}, 100]}
  }}
])
```

### 3. 백업 및 복원
```bash
# 백업
mongodump --db aos --collection mmsext --out backup/

# 복원  
mongorestore --db aos --collection mmsext backup/aos/mmsext.bson
```

이제 MMS Extractor의 모든 추출 결과가 MongoDB에 체계적으로 저장되어 분석 및 모니터링이 가능합니다.
