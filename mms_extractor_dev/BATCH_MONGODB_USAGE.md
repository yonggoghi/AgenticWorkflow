# Batch.py MongoDB 사용 가이드

## 개요
`python batch.py` 명령어로 MMS 메시지를 배치 처리하면서 결과를 MongoDB에 자동으로 저장할 수 있습니다.

## 전제 조건
1. **MongoDB 서버 실행**:
   ```bash
   mongod --dbpath ~/data/db
   ```

2. **필요한 패키지 설치**:
   ```bash
   pip install pymongo
   ```

3. **파일 확인**:
   - `mongodb_utils.py` 파일이 같은 디렉토리에 있어야 함
   - `./data/mms_messages.csv` 파일에 처리할 메시지들이 있어야 함

## 사용법

### 1. MongoDB 연결 테스트
```bash
python batch.py --test-mongodb
```
**출력 예시**:
```
🔌 MongoDB 연결 테스트 중...
✅ MongoDB 연결 성공!
```

### 2. 기본 배치 처리 + MongoDB 저장
```bash
python batch.py --batch-size 10 --save-to-mongodb
```

### 3. 병렬 처리 + MongoDB 저장
```bash
python batch.py --batch-size 50 --max-workers 4 --save-to-mongodb
```

### 4. DAG 추출 포함 + MongoDB 저장
```bash
python batch.py --batch-size 20 --extract-entity-dag --save-to-mongodb --max-workers 2
```

### 5. 순차 처리 + MongoDB 저장
```bash
python batch.py --batch-size 10 --disable-multiprocessing --save-to-mongodb
```

### 6. 모든 옵션 조합
```bash
python batch.py \
  --batch-size 100 \
  --max-workers 8 \
  --extract-entity-dag \
  --save-to-mongodb \
  --llm-model ax \
  --entity-extraction-mode llm \
  --output-file ./data/my_batch_results.csv
```

## 주요 옵션

### 배치 처리 옵션
- `--batch-size, -b`: 처리할 메시지 수 (기본값: 10)
- `--output-file, -o`: 결과 CSV 파일 경로 (기본값: ./data/batch_results.csv)

### 병렬 처리 옵션
- `--max-workers, -w`: 최대 워커 프로세스 수 (기본값: CPU 코어 수)
- `--disable-multiprocessing`: 병렬 처리 비활성화 (순차 처리)

### LLM 설정
- `--llm-model`: 사용할 LLM 모델 (`ax`, `gem`, `cld`, `gen`, `gpt`)
- `--entity-extraction-mode`: 엔티티 매칭 모드 (`logic`, `llm`)
- `--product-info-extraction-mode`: 상품 추출 모드 (`nlp`, `llm`, `rag`)

### 데이터 소스
- `--offer-data-source`: 데이터 소스 (`local`, `db`)

### 고급 기능
- `--extract-entity-dag`: Entity DAG 추출 활성화
- `--save-to-mongodb`: MongoDB 저장 활성화 ⭐
- `--test-mongodb`: MongoDB 연결 테스트만 수행

## 출력 예시

### 성공적인 배치 실행
```bash
$ python batch.py --batch-size 5 --save-to-mongodb --max-workers 2

==================================================
🚀 Starting Batch MMS Processing
==================================================
배치 크기: 5
출력 파일: ./data/batch_results.csv
병렬 처리: ON
최대 워커 수: 2
추출기 설정: {'offer_info_data_src': 'local', 'product_info_extraction_mode': 'llm', ...}
📄 MongoDB 저장 모드 활성화됨
==================================================

INFO:__main__:MMS 추출기 초기화 중...
INFO:__main__:🚀 병렬 처리 모드로 5개 메시지 처리 시작 (워커: 2개)
INFO:mongodb_utils:MongoDB 연결 성공: aos.mmsext
INFO:mongodb_utils:문서 저장 성공: 68ca2230...
INFO:mongodb_utils:MongoDB 연결 해제
...

==================================================
Batch Processing Summary
==================================================
status: completed
processed_count: 5
failed_count: 0
processing_mode: parallel
total_processing_time_seconds: 45.2
throughput_messages_per_second: 0.11
==================================================
```

### MongoDB 저장 성공
```bash
$ python batch.py --batch-size 3 --save-to-mongodb --disable-multiprocessing

INFO:__main__:📄 MongoDB 저장 모드 활성화됨
...
INFO:mongodb_utils:MongoDB 연결 성공: aos.mmsext
INFO:mongodb_utils:문서 저장 성공: 68ca26f1...
INFO:mongodb_utils:MongoDB 연결 해제
...
==================================================
Batch Processing Summary
==================================================
status: completed
processed_count: 3
successful_count: 3
failed_count: 0
processing_mode: 순차 처리
==================================================
```

### MongoDB 연결 실패
```bash
$ python batch.py --save-to-mongodb --batch-size 5

⚠️ MongoDB 유틸리티를 찾을 수 없습니다. --save-to-mongodb 옵션이 비활성화됩니다.
# 또는
WARNING:__main__:MongoDB 저장이 요청되었지만 mongodb_utils를 찾을 수 없습니다.
```

## 데이터 입력 형식

### CSV 파일 구조 (`./data/mms_messages.csv`)
```csv
msg_id,msg
1,"SKT 5G 요금제 할인 이벤트! 월 39,000원에 데이터 무제한"
2,"LG U+ 인터넷 가입하면 첫 3개월 무료!"
3,"KT 휴대폰 교체 지원금 최대 50만원"
```

## MongoDB 데이터 확인

### 저장된 데이터 조회
```bash
# MongoDB Shell 접속
mongosh

# 데이터베이스 선택
use aos

# 배치 처리로 저장된 문서 조회 (processing_mode로 구분)
db.mmsext.find({"metadata.processing_mode": "batch"})

# 최근 배치 처리 결과 조회
db.mmsext.find({"metadata.processing_mode": "batch"}).sort({_id: -1}).limit(10)

# 배치별 통계
db.mmsext.aggregate([
  {$match: {"metadata.processing_mode": "batch"}},
  {$group: {
    _id: "$metadata.settings.llm_model",
    count: {$sum: 1},
    avg_processing_time: {$avg: "$metadata.processing_time_seconds"}
  }}
])
```

## 성능 최적화

### 1. 워커 수 조정
```bash
# CPU 코어 수에 맞게 조정
python batch.py --batch-size 100 --max-workers 8 --save-to-mongodb

# 메모리 부족 시 워커 수 줄이기
python batch.py --batch-size 100 --max-workers 2 --save-to-mongodb
```

### 2. 배치 크기 조정
```bash
# 대용량 처리
python batch.py --batch-size 1000 --max-workers 16 --save-to-mongodb

# 안정적인 처리
python batch.py --batch-size 50 --max-workers 4 --save-to-mongodb
```

### 3. 순차 처리 (안정성 우선)
```bash
python batch.py --batch-size 20 --disable-multiprocessing --save-to-mongodb
```

## 모니터링 및 로깅

### 1. 로그 파일 확인
```bash
# 배치 처리 로그 확인
tail -f ./logs/batch_processing.log

# 실시간 로그 모니터링
watch -n 1 "tail -20 ./logs/batch_processing.log"
```

### 2. 진행 상황 모니터링
```bash
# MongoDB에서 처리 진행 상황 확인
mongosh --eval "
use aos;
db.mmsext.aggregate([
  {\$match: {'metadata.processing_mode': 'batch'}},
  {\$group: {
    _id: {\$dateToString: {format: '%Y-%m-%d %H', date: '\$metadata.timestamp'}},
    count: {\$sum: 1}
  }},
  {\$sort: {_id: -1}}
])
"
```

### 3. 성능 메트릭
```bash
# 처리 속도 분석
mongosh --eval "
use aos;
db.mmsext.aggregate([
  {\$match: {'metadata.processing_mode': 'batch'}},
  {\$group: {
    _id: null,
    avg_time: {\$avg: '\$metadata.processing_time_seconds'},
    min_time: {\$min: '\$metadata.processing_time_seconds'},
    max_time: {\$max: '\$metadata.processing_time_seconds'},
    total_count: {\$sum: 1}
  }}
])
"
```

## 문제 해결

### 1. 메모리 부족
```bash
# 워커 수 줄이기
python batch.py --batch-size 50 --max-workers 2 --save-to-mongodb

# 순차 처리로 전환
python batch.py --batch-size 20 --disable-multiprocessing --save-to-mongodb
```

### 2. MongoDB 연결 문제
```bash
# 연결 테스트
python batch.py --test-mongodb

# MongoDB 서버 상태 확인
ps aux | grep mongod
```

### 3. 처리 속도 개선
```bash
# DAG 추출 비활성화 (속도 향상)
python batch.py --batch-size 100 --save-to-mongodb

# 병렬 처리 최적화
python batch.py --batch-size 200 --max-workers 12 --save-to-mongodb
```

## 배치 처리 vs 단일 처리 비교

| 기능 | batch.py | mms_extractor.py | demo_streamlit.py |
|------|----------|------------------|-------------------|
| 처리 방식 | 배치 (대용량) | 단일/배치 | 단일 (UI) |
| 병렬 처리 | ✅ 멀티프로세싱 | ✅ 멀티스레드 | ❌ 단일 스레드 |
| MongoDB 저장 | ✅ 자동 저장 | ✅ 옵션 저장 | ✅ 자동 저장 |
| CSV 출력 | ✅ 기본 제공 | ✅ JSON 파일 | ❌ 없음 |
| 사용 용도 | 대량 데이터 처리 | 테스트/개발 | 데모/분석 |

이제 모든 실행 방식에서 MongoDB 저장이 지원됩니다! 🎉
