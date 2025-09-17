# MMS Extractor CLI MongoDB 사용 가이드

## 개요
`python mms_extractor.py` 명령어로 MMS 추출기를 실행하면서 결과를 MongoDB에 자동으로 저장할 수 있습니다.

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

## 사용법

### 1. MongoDB 연결 테스트
```bash
python mms_extractor.py --test-mongodb
```
**출력 예시**:
```
🔌 MongoDB 연결 테스트 중...
✅ MongoDB 연결 성공!
```

### 2. 단일 메시지 처리 + MongoDB 저장
```bash
python mms_extractor.py --message "SKT 5G 요금제 할인 이벤트! 월 39,000원" --save-to-mongodb
```

### 3. DAG 추출 포함 + MongoDB 저장
```bash
python mms_extractor.py --message "광고 메시지" --extract-entity-dag --save-to-mongodb
```

### 4. 배치 처리 + MongoDB 저장
```bash
# 메시지 파일 생성
echo "SKT 5G 요금제 할인 이벤트" > messages.txt
echo "LG U+ 인터넷 가입 혜택" >> messages.txt

# 배치 처리 실행
python mms_extractor.py --batch-file messages.txt --save-to-mongodb --max-workers 2
```

### 5. 다양한 옵션 조합
```bash
python mms_extractor.py \
  --message "광고 메시지" \
  --llm-model ax \
  --entity-matching-mode llm \
  --extract-entity-dag \
  --save-to-mongodb \
  --log-level INFO
```

## 주요 옵션

### 기본 옵션
- `--message`: 처리할 단일 메시지
- `--batch-file`: 배치 처리할 메시지 파일 (한 줄에 하나씩)
- `--max-workers`: 배치 처리 시 최대 워커 수

### LLM 설정
- `--llm-model`: 사용할 LLM 모델 (`ax`, `gem`, `cld`, `gen`, `gpt`)
- `--entity-matching-mode`: 엔티티 매칭 모드 (`logic`, `llm`)
- `--product-info-extraction-mode`: 상품 추출 모드 (`nlp`, `llm`, `rag`)

### 데이터 소스
- `--offer-data-source`: 데이터 소스 (`local`, `db`)

### 고급 기능
- `--extract-entity-dag`: Entity DAG 추출 활성화
- `--save-to-mongodb`: MongoDB 저장 활성화 ⭐
- `--test-mongodb`: MongoDB 연결 테스트만 수행
- `--log-level`: 로그 레벨 (`DEBUG`, `INFO`, `WARNING`, `ERROR`)

## 출력 예시

### 성공적인 실행
```bash
$ python mms_extractor.py --message "SKT 할인" --save-to-mongodb

INFO:__main__:MMS 추출기 초기화 중...
INFO:__main__:단일 메시지 처리 시작 (멀티스레드)

📄 MongoDB 저장 중...
📄 결과가 MongoDB에 저장되었습니다. (ID: 68ca1e98...)
📄 MongoDB 저장 완료!

==================================================
🎯 최종 추출된 정보
==================================================
{
    "title": "SKT 할인 이벤트",
    "purpose": ["할인 혜택 제공"],
    "product": ["5G 요금제"],
    ...
}

==================================================
📊 처리 완료
==================================================
✅ 제목: SKT 할인 이벤트
✅ 목적: 1개
✅ 상품: 1개
✅ 채널: 0개
✅ 프로그램: 0개
```

### MongoDB 연결 실패
```bash
$ python mms_extractor.py --save-to-mongodb --message "test"

❌ MongoDB 저장이 요청되었지만 mongodb_utils를 찾을 수 없습니다.
# 또는
⚠️ MongoDB 저장에 실패했습니다.
```

## MongoDB 데이터 확인

### 저장된 데이터 조회
```bash
# MongoDB Shell 접속
mongosh

# 데이터베이스 선택
use aos

# 문서 개수 확인
db.mmsext.countDocuments()

# 최근 문서 조회
db.mmsext.find().sort({_id: -1}).limit(3)

# 특정 필드만 조회
db.mmsext.find({}, {message: 1, "ext_result.title": 1, "metadata.timestamp": 1})
```

### 통계 쿼리
```javascript
// 성공률 통계
db.mmsext.aggregate([
  {$group: {
    _id: null,
    total: {$sum: 1},
    success: {$sum: {$cond: ["$metadata.success", 1, 0]}}
  }},
  {$project: {
    total: 1,
    success: 1,
    success_rate: {$multiply: [{$divide: ["$success", "$total"]}, 100]}
  }}
])

// 모델별 통계
db.mmsext.aggregate([
  {$group: {
    _id: "$metadata.settings.llm_model",
    count: {$sum: 1}
  }}
])
```

## 문제 해결

### 1. MongoDB 연결 실패
```bash
# MongoDB 서버 상태 확인
ps aux | grep mongod

# MongoDB 서버 시작
mongod --dbpath ~/data/db
```

### 2. 패키지 누락
```bash
# pymongo 설치
pip install pymongo

# mongodb_utils.py 파일 확인
ls -la mongodb_utils.py
```

### 3. 권한 문제
```bash
# 데이터 디렉토리 권한 확인
ls -la ~/data/db

# 권한 수정 (필요시)
chmod 755 ~/data/db
```

## 성능 팁

### 배치 처리 최적화
```bash
# 워커 수 조정 (CPU 코어 수에 맞게)
python mms_extractor.py --batch-file large_messages.txt --max-workers 8 --save-to-mongodb
```

### 로그 레벨 조정
```bash
# 상세 로그 (디버깅용)
python mms_extractor.py --message "test" --log-level DEBUG --save-to-mongodb

# 최소 로그 (성능 향상)
python mms_extractor.py --batch-file messages.txt --log-level ERROR --save-to-mongodb
```

이제 `python mms_extractor.py`로 실행할 때도 Streamlit과 동일하게 MongoDB에 결과가 자동 저장됩니다! 🎉
