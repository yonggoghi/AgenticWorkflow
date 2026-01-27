# HDFS Transfer Script - 파티션 통합 기능 가이드

## 개요

`hdfs_transfer.py` 스크립트에 PyArrow Streaming 방식의 파티션 통합 기능이 추가되었습니다.

### 주요 기능
- ✅ 파티션된 Parquet 파일들을 단일 파일로 통합
- ✅ 파티션 정보를 컬럼으로 자동 추가 (Hive 스타일 지원)
- ✅ 메모리 효율적인 스트리밍 처리 (OOM 방지)
- ✅ 배치 크기 및 압축 알고리즘 조정 가능
- ✅ 대용량 데이터 처리 최적화 (16GB 테스트 완료)

---

## 설치 요구사항

### 필수 패키지
```bash
pip install pyarrow python-dotenv
```

### 권장 환경
- Python 3.7+
- PyArrow 10.0+
- 메모리: 8GB 이상 (16GB 권장)
- 디스크 여유 공간: 데이터 크기의 2배 이상

---

## 사용법

### 1. 기본 사용 (파티션 통합 없음)
기존 방식대로 파티션 구조를 그대로 유지하여 전송:

```bash
# 기본 동작: tar.gz 압축 파일만 전송 (v2.8+)
python hdfs_transfer.py

# tar.gz 압축 해제까지 수행 (기존 방식)
python hdfs_transfer.py --extract-remote

# 로컬 파일 유지 (재사용 가능)
python hdfs_transfer.py --skip-cleanup
```

**중요 (v2.8+):**
- **기본 동작**: tar.gz 파일만 전송하고 압축 해제 안 함 (변경됨)
- **원격 파일**: `data.tar.gz` (압축 상태 유지)
- **장점**: 디스크 공간 75% 절약, 원격 CPU 사용 없음, 백업 파일 활용
- **압축 해제 필요 시**: `--extract-remote` 옵션 사용

**중요 (v2.9+):**
- **압축 해제 시**: tar.gz 이름과 같은 디렉토리 생성 후 그 안에 파티션 풀림
- **원격 구조**: `/remote/path/propensityScoreDF/send_ym=202512/`
- **장점**: 명확한 디렉토리 구조, 데이터셋 간 분리, 관리 용이
- **예시**:
  ```
  원격 서버:
  /remote/path/
  └── propensityScoreDF/     # tar.gz 이름과 같은 디렉토리
      └── send_ym=202512/     # 파티션
          └── *.parquet
  ```

### 2. 파티션 통합 활성화
파티션 파일들을 단일 Parquet 파일로 통합하여 전송:

```bash
# 파티션 통합 + 전송 후 로컬 파일 삭제
python hdfs_transfer.py --merge-partitions

# 파티션 통합 + 로컬 파일 유지 (권장)
python hdfs_transfer.py --merge-partitions --skip-cleanup
```

**중요 (v2.7+):**
- 원본 파티션 디렉토리는 **유지**됩니다 (`LOCAL_TMP_PATH/{DIR_NAME}/`)
- 통합 파일은 별도 디렉토리에 생성됩니다 (`LOCAL_TMP_PATH/{DIR_NAME}_merged/`)
- **압축 없이 직접 전송**됩니다 (tar.gz 생성 안 됨) ⭐
- 원격지 압축 해제 단계 불필요
- `--skip-cleanup` 사용 시 원본과 통합 파일 모두 유지

**로컬 파일 구조:**
```
LOCAL_TMP_PATH/
├── propensityScoreDF/          # 원본 파티션 (유지됨)
│   ├── year=2024/
│   └── ...
└── propensityScoreDF_merged/   # 통합 파일
    └── merged.parquet
```

**전송 방식:**
- 파티션 통합 모드 (`--merge-partitions`): `merged.parquet` 직접 전송 (압축 없음)
- 파티션 구조 유지 (기본): `tar.gz` 압축 후 전송
  - 기본: 압축 파일만 전송 (압축 해제 안 함)
  - `--extract-remote`: 압축 해제까지 수행

### 3. 환경 변수로 파일명 설정

`.env` 파일에서 파일명을 미리 설정할 수 있습니다:

```bash
# .env 파일
OUTPUT_FILENAME=mth_mms_rcv_ract_score_202601.parquet  # 통합 후 parquet 파일명
ARCHIVE_NAME=mth_mms_rcv_ract_score_202601.tar.gz      # tar.gz 압축 파일명
```

```bash
# .env 설정만으로 실행 (파일명 자동 적용)
python hdfs_transfer.py --merge-partitions --skip-cleanup
```

**우선순위:**
- 명령행 옵션 > 환경변수 (.env) > 기본값

### 4. 고급 옵션 사용

#### 배치 크기 조정 (메모리 최적화)
```bash
# 메모리가 부족한 경우 배치 크기를 줄임 (기본값: 100,000)
python hdfs_transfer.py --merge-partitions --batch-size 50000

# 메모리가 충분한 경우 배치 크기를 늘림
python hdfs_transfer.py --merge-partitions --batch-size 200000
```

#### 압축 알고리즘 선택
```bash
# Snappy (기본값): 빠르지만 압축률 낮음
python hdfs_transfer.py --merge-partitions --compression snappy

# Gzip: 느리지만 압축률 높음 (네트워크 전송 최적화)
python hdfs_transfer.py --merge-partitions --compression gzip

# Zstd (권장): 속도와 압축률의 균형
python hdfs_transfer.py --merge-partitions --compression zstd

# 압축 없음: 가장 빠르지만 파일 크기 큼
python hdfs_transfer.py --merge-partitions --compression none
```

#### 출력 파일명 지정
```bash
python hdfs_transfer.py \
  --merge-partitions \
  --output-filename mth_mms_rcv_ract_score_202601.parquet
```

#### 전체 옵션 조합 예제
```bash
python hdfs_transfer.py \
  --merge-partitions \
  --batch-size 150000 \
  --compression zstd \
  --output-filename merged_data.parquet \
  --archive-name result_202601.tar.gz
```

---

## 옵션 상세 설명

### 파티션 통합 관련 옵션

| 옵션 | 설명 | 기본값 / 환경변수 |
|------|------|------------------|
| `--merge-partitions` | 파티션 통합 기능 활성화 | False |
| `--batch-size` | 한 번에 처리할 row 개수 | 100,000 |
| `--compression` | 압축 알고리즘 (snappy/gzip/zstd/none) | snappy |
| `--output-filename` | 출력 parquet 파일명 | merged.parquet / OUTPUT_FILENAME |

### 일반 옵션

| 옵션 | 설명 | 환경변수 |
|------|------|---------|
| `--skip-download` | HDFS 다운로드 단계 건너뛰기 | - |
| `--skip-remove` | 원격 서버 기존 파일 삭제 건너뛰기 (OUTPUT_FILENAME, EOF 파일) | - |
| `--skip-cleanup` | 로컬 파일 유지 (다운로드 전 + 전송 후 정리 건너뛰기) | - |
| `--extract-remote` | 원격 서버에서 tar.gz 압축 해제 수행 (기본: 압축 파일 유지) ⭐ v2.8+ | - |
| `--archive-name` | tar.gz 압축 파일명 지정 | ARCHIVE_NAME |
| `--env-file` | .env 파일 경로 지정 | - |

---

## 성능 가이드

### 배치 크기 선택 기준

| 데이터 크기 | 서버 메모리 | 권장 배치 크기 |
|------------|------------|---------------|
| < 5GB | 8GB | 50,000 |
| 5-10GB | 16GB | 100,000 (기본) |
| 10-20GB | 32GB | 150,000 |
| > 20GB | 64GB | 200,000 |

### 압축 알고리즘 비교

| 알고리즘 | 압축 속도 | 압축률 | 해제 속도 | 권장 용도 |
|---------|----------|--------|----------|----------|
| **snappy** | 매우 빠름 | 낮음 | 매우 빠름 | 로컬 처리, 빠른 전송 |
| **gzip** | 느림 | 높음 | 보통 | 네트워크 대역폭 제한 |
| **zstd** | 빠름 | 높음 | 빠름 | 균형잡힌 선택 (권장) |
| **none** | 없음 | 없음 | 없음 | 디버깅, 테스트 |

### 예상 처리 시간 (16GB 데이터 기준)

| 압축 알고리즘 | 처리 시간 | 최종 파일 크기 |
|-------------|----------|---------------|
| snappy | 10-15분 | ~14GB |
| gzip | 20-30분 | ~8GB |
| zstd | 15-20분 | ~10GB |
| none | 5-10분 | ~16GB |

*실제 시간은 서버 스펙, 디스크 속도, 파티션 개수에 따라 달라질 수 있습니다.*

---

## 동작 방식

### 원격 파일 삭제 (--skip-remove 미사용 시)

전송 전에 원격 서버의 기존 파일을 삭제합니다. 삭제 방식은 `--merge-partitions` 옵션 사용 여부에 따라 다릅니다.

#### 모드 1: 파티션 통합 모드 (`--merge-partitions` 사용)

**삭제되는 파일:**
1. `{REMOTE_PATH}/{DIR_NAME}/{OUTPUT_FILENAME}` - 특정 parquet 파일만
2. `{REMOTE_PATH}/{base_name}.eof` - EOF 파일

**유지되는 것:**
- 디렉토리 구조
- 다른 파일들 (다른 월 데이터 등)

**예시:**
```bash
# .env 설정
OUTPUT_FILENAME=data_202601.parquet
ARCHIVE_NAME=data_202601.tar.gz

# 삭제: 특정 파일만
/remote/path/table_name/data_202601.parquet  ❌
/remote/path/data_202601.eof                 ❌

# 유지: 디렉토리 및 다른 파일
/remote/path/table_name/                     ✅
/remote/path/table_name/data_202512.parquet  ✅
```

#### 모드 2: 파티션 구조 유지 모드 (기본값)

**삭제되는 것:**
1. `{REMOTE_PATH}/{DIR_NAME}/` - 디렉토리 전체
2. `{REMOTE_PATH}/{ARCHIVE_NAME}` - tar.gz 파일
3. `{REMOTE_PATH}/{base_name}.eof` - EOF 파일

**유지되는 것:**
- `{REMOTE_PATH}` 디렉토리 자체
- 다른 디렉토리들
- 다른 tar.gz 파일들

**예시:**
```bash
# .env 설정
ARCHIVE_NAME=raw_data_202601.tar.gz

# 삭제: ARCHIVE_NAME 관련 파일들
/remote/path/table_name/             (DIR_NAME)     ❌
/remote/path/raw_data_202601.tar.gz  (tar.gz)       ❌
/remote/path/raw_data_202601.eof     (EOF)          ❌

# 유지: REMOTE_PATH 및 다른 파일들
/remote/path/                        (디렉토리)     ✅
/remote/path/other_table/            (다른 디렉토리) ✅
/remote/path/raw_data_202512.tar.gz  (다른 tar.gz)  ✅
```

**이유:** ARCHIVE_NAME에 해당하는 모든 관련 파일을 깔끔하게 대체

### 파티션 구조 예시

**원본 HDFS 구조:**
```
/user/data/table/
├── year=2024/
│   ├── month=01/
│   │   ├── part-00000.parquet
│   │   ├── part-00001.parquet
│   │   └── part-00002.parquet
│   └── month=02/
│       ├── part-00000.parquet
│       └── part-00001.parquet
└── _SUCCESS
```

**통합 후 결과:**
- 단일 파일: `merged.parquet`
- 파티션 정보가 컬럼으로 추가됨:

| id | value | year | month |
|----|-------|------|-------|
| 1 | 100 | 2024 | 01 |
| 2 | 200 | 2024 | 01 |
| 3 | 300 | 2024 | 02 |

### 처리 흐름

```
1. HDFS 다운로드
   ↓
2. 파티션 통합 (--merge-partitions 활성화 시)
   - 파티션 파일들을 순차적으로 읽기
   - 배치 단위로 처리 (메모리 효율)
   - 파티션 컬럼 자동 추가
   - 단일 Parquet 파일로 점진적 쓰기
   ↓
3. 압축 (tar.gz)
   ↓
4. 원격 서버 전송
   ↓
5. 압축 해제
   ↓
6. EOF 파일 생성
```

---

## 문제 해결

### Q1. PyArrow import 오류
```
Error: pyarrow가 설치되어 있지 않습니다.
```
**해결:**
```bash
pip install pyarrow
```

### Q2. 메모리 부족 (OOM)
```
MemoryError: Unable to allocate...
```
**해결:**
```bash
# 배치 크기를 줄이세요
python hdfs_transfer.py --merge-partitions --batch-size 50000
```

### Q3. 처리 시간이 너무 오래 걸림
**해결:**
- 압축 알고리즘을 `snappy` 또는 `none`으로 변경
- 배치 크기를 늘림 (메모리가 충분한 경우)
```bash
python hdfs_transfer.py \
  --merge-partitions \
  --batch-size 200000 \
  --compression snappy
```

### Q4. 파티션 컬럼이 추가되지 않음
**원인:** Hive 스타일 파티션 구조가 아닌 경우 (`key=value` 형식)

**확인:**
```bash
# HDFS 구조 확인
hdfs dfs -ls -R /path/to/data
```

파티션 구조가 다음과 같아야 합니다:
- ✅ `year=2024/month=01/` (Hive 스타일)
- ❌ `2024/01/` (일반 디렉토리)

### Q5. 디스크 공간 부족
**해결:**
- `LOCAL_TMP_PATH`의 디스크 공간 확인
- 최소 데이터 크기의 2배 이상 필요
- 또는 `--skip-download`로 기존 파일 재사용

---

## 주의사항

### 1. 메모리 관리
- 16GB 데이터를 처리할 때 피크 메모리는 약 2-4GB
- 배치 크기를 너무 크게 설정하면 OOM 발생 가능
- 안전하게 기본값(100,000)부터 시작 권장

### 2. 디스크 공간
- 임시로 원본 + 통합본 + 압축본 공간 필요
- 16GB 데이터 기준: 최소 32GB 여유 공간 필요

### 3. 파티션 구조
- Hive 스타일 파티션만 자동 인식 (`key=value` 형식)
- 일반 디렉토리 구조는 파티션 컬럼이 추가되지 않음

### 4. 처리 시간
- 급하지 않은 작업에 적합 (10-30분 소요 가능)
- 실시간 처리가 필요한 경우 파티션 통합 비활성화 권장

---

## 옵션 조합 가이드

### --skip-cleanup 동작 (v2.5+)

`--skip-cleanup` 옵션은 전체 프로세스에서 로컬 파일을 유지합니다:

1. **다운로드 전**: 기존 로컬 파일 유지 (삭제 안함)
2. **다운로드**: 기존 파일 덮어씀
3. **전송 후**: 로컬 파일 유지 (삭제 안함)

**사용 패턴:**

**패턴 1: 파일 재사용 (권장)**
```bash
# 첫 실행: 다운로드 + 로컬 파일 유지
python hdfs_transfer.py --skip-cleanup

# 재실행: 다운로드 건너뛰기 + 로컬 파일 재사용
python hdfs_transfer.py --skip-download --skip-cleanup
# 시간 절약: ~80%
```

**패턴 2: 항상 최신 데이터 + 로컬 파일 유지**
```bash
# 매번 다운로드하되 로컬 파일은 유지
python hdfs_transfer.py --skip-cleanup
python hdfs_transfer.py --skip-cleanup  # 기존 파일 덮어씀
```

**패턴 3: 매번 깔끔하게 (기본)**
```bash
# 다운로드 전 삭제 + 전송 후 삭제
python hdfs_transfer.py
```

---

## 사용 예제

### 예제 1: 프로덕션 환경 (권장 설정)

**방법 A: 환경 변수 사용 (권장)**
```bash
# .env 파일
OUTPUT_FILENAME=mth_mms_rcv_ract_score_202601.parquet
ARCHIVE_NAME=mth_mms_rcv_ract_score_202601.tar.gz

# 실행
python hdfs_transfer.py \
  --merge-partitions \
  --compression zstd \
  --skip-cleanup
```

**방법 B: 명령행 옵션 사용**
```bash
python hdfs_transfer.py \
  --merge-partitions \
  --batch-size 100000 \
  --compression zstd \
  --output-filename mth_mms_rcv_ract_score_202601.parquet \
  --archive-name mth_mms_rcv_ract_score_202601.tar.gz \
  --skip-cleanup
```

### 예제 2: 메모리 제한 환경
```bash
python hdfs_transfer.py \
  --merge-partitions \
  --batch-size 50000 \
  --compression snappy \
  --skip-cleanup
```

### 예제 3: 네트워크 대역폭 제한
```bash
python hdfs_transfer.py \
  --merge-partitions \
  --compression gzip \  # 최대 압축
  --skip-cleanup
```

### 예제 4: 로컬 파일 재처리
```bash
# HDFS 다운로드 없이 로컬 파일만 통합
python hdfs_transfer.py \
  --skip-download \
  --merge-partitions \
  --skip-cleanup
```

### 예제 5: 로컬 파일 유지 (재사용)
```bash
# 첫 실행: 다운로드 + 통합 + 전송 + 로컬 파일 유지
python hdfs_transfer.py --merge-partitions --skip-cleanup

# 두 번째 실행: 로컬 파일 재사용 (다운로드 건너뜀)
python hdfs_transfer.py --skip-download --skip-cleanup
```

### 예제 5: 디버깅/테스트
```bash
python hdfs_transfer.py \
  --merge-partitions \
  --compression none \
  --batch-size 10000 \
  --output-filename test.parquet
```

---

## FAQ

**Q: 파티션 통합이 필수인가요?**
A: 아니요. `--merge-partitions` 옵션을 사용하지 않으면 기존 방식대로 파티션 구조를 유지합니다.

**Q: 단일 파일의 장단점은?**
A: 
- 장점: 관리 편리, 다운스트림 시스템 단순화
- 단점: 병렬 처리 불가, 부분 읽기 비효율

**Q: 대용량 데이터(100GB+)도 가능한가요?**
A: 배치 크기를 적절히 조정하면 가능하지만, Spark에서 직접 처리하는 것이 더 효율적입니다.

**Q: 파티션 순서가 보장되나요?**
A: 파일 시스템의 순서대로 처리되며, 특정 순서가 필요하면 정렬은 다운스트림에서 수행하세요.

---

## 문의 및 지원

문제가 발생하면 다음 정보와 함께 문의하세요:
1. 데이터 크기
2. 파티션 개수 및 구조
3. 서버 메모리
4. 사용한 명령어
5. 에러 메시지 전체 내용
