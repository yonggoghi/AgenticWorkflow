# 변경 사항 (v2.4)

**날짜**: 2026-01-26  
**변경자**: AI Assistant

---

## 주요 변경 사항

### 원격 디렉토리 구조 제거 ✅

**문제점:**
- v2.3까지는 압축 해제 시 디렉토리 구조가 그대로 유지됨
- 원격 서버에 `{REMOTE_PATH}/{DIR_NAME}/파일들` 형태로 생성됨
- 사용자가 원하는 구조: `{REMOTE_PATH}/파일들` (디렉토리 없이 바로)

**해결책:**
- 압축 시 `-C` 옵션 사용하여 디렉토리 구조 제거
- 압축 해제 시 파일들이 `{REMOTE_PATH}` 바로 밑에 생성됨
- 삭제 로직도 이에 맞게 수정

---

## 상세 변경 내역

### 1. 압축 방식 변경

**Before (v2.3):**
```bash
tar -czf archive.tar.gz dir_name
```

**압축 내용:**
```
archive.tar.gz
└── dir_name/
    ├── file1.parquet
    └── file2.parquet
```

**압축 해제 결과:**
```
/remote/path/
└── dir_name/  ← 불필요한 디렉토리
    ├── file1.parquet
    └── file2.parquet
```

---

**After (v2.4):**
```bash
tar -czf archive.tar.gz -C dir_name .
```

**압축 내용:**
```
archive.tar.gz
├── file1.parquet  ← 디렉토리 구조 없음
└── file2.parquet
```

**압축 해제 결과:**
```
/remote/path/
├── file1.parquet  ← 바로 이 레벨에
└── file2.parquet
```

---

### 2. 삭제 로직 수정

#### 파티션 통합 모드 (`--merge-partitions`)

**Before (v2.3):**
```bash
rm -f /remote/path/dir_name/output_file.parquet  # 디렉토리 포함
rm -f /remote/path/data.eof
```

**After (v2.4):**
```bash
rm -f /remote/path/output_file.parquet  # 직접 경로
rm -f /remote/path/data.eof
```

---

#### 파티션 구조 유지 모드 (기본값)

**Before (v2.3):**
```bash
rm -rf /remote/path/dir_name  # 디렉토리 삭제
rm -f /remote/path/data.tar.gz
rm -f /remote/path/data.eof
```

**After (v2.4):**
```bash
rm -f /remote/path/*.parquet  # 파일들 직접 삭제
rm -f /remote/path/_SUCCESS
rm -f /remote/path/data.tar.gz
rm -f /remote/path/data.eof
```

---

## 비교표

### 원격 서버 파일 구조

| 버전 | 구조 | 예시 |
|------|------|------|
| **v2.3** | `{REMOTE_PATH}/{DIR_NAME}/파일들` | `/data/MMS_SCORE/propensityScoreDF/data.parquet` |
| **v2.4** | `{REMOTE_PATH}/파일들` | `/data/MMS_SCORE/data.parquet` ⭐ |

### 삭제 대상 (파티션 통합 모드)

| 버전 | 파일 경로 |
|------|----------|
| **v2.3** | `{REMOTE_PATH}/{DIR_NAME}/{OUTPUT_FILENAME}` |
| **v2.4** | `{REMOTE_PATH}/{OUTPUT_FILENAME}` ⭐ |

### 삭제 대상 (파티션 구조 유지 모드)

| 항목 | v2.3 | v2.4 |
|------|------|------|
| **디렉토리** | `{REMOTE_PATH}/{DIR_NAME}/` 삭제 | ❌ 해당 없음 |
| **Parquet 파일** | (디렉토리와 함께 삭제) | `{REMOTE_PATH}/*.parquet` 삭제 ⭐ |
| **_SUCCESS** | (디렉토리와 함께 삭제) | `{REMOTE_PATH}/_SUCCESS` 삭제 ⭐ |
| **tar.gz** | `{REMOTE_PATH}/{ARCHIVE_NAME}` | `{REMOTE_PATH}/{ARCHIVE_NAME}` |
| **EOF** | `{REMOTE_PATH}/{base_name}.eof` | `{REMOTE_PATH}/{base_name}.eof` |

---

## 사용 예제

### 예제 1: 파티션 통합 모드

**실행:**
```bash
OUTPUT_FILENAME=mth_mms_rcv_ract_score_202601.parquet \
python hdfs_transfer.py --merge-partitions --skip-cleanup
```

**원격 서버 결과 (v2.3):**
```
/data/TOSIFDAT/LAKE/MMS_SCORE/
└── propensityScoreDF/  ← 불필요한 디렉토리
    └── mth_mms_rcv_ract_score_202601.parquet
```

**원격 서버 결과 (v2.4):**
```
/data/TOSIFDAT/LAKE/MMS_SCORE/
├── mth_mms_rcv_ract_score_202601.parquet  ← 바로 이 레벨
└── mth_mms_rcv_ract_score_202601.eof
```

---

### 예제 2: 파티션 구조 유지 모드

**실행:**
```bash
python hdfs_transfer.py --skip-cleanup
```

**원격 서버 결과 (v2.3):**
```
/data/TOSIFDAT/LAKE/MMS_SCORE/
└── table_name/  ← 불필요한 디렉토리
    ├── part-00000.parquet
    ├── part-00001.parquet
    └── _SUCCESS
```

**원격 서버 결과 (v2.4):**
```
/data/TOSIFDAT/LAKE/MMS_SCORE/
├── part-00000.parquet  ← 바로 이 레벨
├── part-00001.parquet
├── _SUCCESS
└── data.eof
```

---

## 코드 변경 요약

### 1. compress_data() 함수

```python
# Before (v2.3)
tar_cmd = f"tar -czf {archive_name} {dir_name}"

# After (v2.4)
tar_cmd = f"tar -czf {archive_name} -C {dir_name} ."
```

**설명:** `-C dir_name .` 은 "dir_name 디렉토리로 이동해서 그 안의 내용(.)을 압축"하라는 의미

---

### 2. remove_remote_files() 함수

**파티션 통합 모드:**
```python
# Before (v2.3)
output_file_path = f"{remote_path}/{dir_name}/{output_filename}"

# After (v2.4)
output_file_path = f"{remote_path}/{output_filename}"
```

**파티션 구조 유지 모드:**
```python
# Before (v2.3)
rm_cmd = f"... rm -rf {dir_path} && rm -f {tar_gz_path} {eof_file_path}"

# After (v2.4)
rm_cmd = f"... rm -f {remote_path}/*.parquet {remote_path}/_SUCCESS {tar_gz_path} {eof_file_path}"
```

---

## 장점

### 1. 깔끔한 파일 구조
- 불필요한 디렉토리 없이 파일들이 원하는 위치에 바로 생성됨
- 다운스트림 시스템에서 파일 접근 경로가 단순해짐

### 2. 직관적인 경로
```bash
# Before
/data/TOSIFDAT/LAKE/MMS_SCORE/propensityScoreDF/mth_mms_rcv_ract_score_202601.parquet

# After
/data/TOSIFDAT/LAKE/MMS_SCORE/mth_mms_rcv_ract_score_202601.parquet
```

### 3. 일관성
- `--merge-partitions` 사용/미사용 모두 동일한 구조
- EOF 파일과 데이터 파일이 같은 레벨에 위치

---

## 주의사항

### 1. 파티션 구조 유지 모드 삭제

와일드카드 사용으로 의도하지 않은 파일이 삭제될 수 있음:

```bash
rm -f /remote/path/*.parquet  # 모든 .parquet 파일 삭제
```

**안전 장치:**
- `--skip-remove` 옵션으로 삭제 단계 건너뛰기 가능
- 다른 파일은 다른 디렉토리에 관리 권장

---

### 2. 여러 데이터셋 관리

같은 `REMOTE_PATH`에 여러 데이터셋을 관리하는 경우 주의:

**권장 구조:**
```
/data/TOSIFDAT/LAKE/
├── MMS_SCORE/  ← dataset 1
│   ├── data_202601.parquet
│   └── data_202601.eof
├── PROPENSITY/  ← dataset 2
│   ├── score_202601.parquet
│   └── score_202601.eof
└── CHURN/  ← dataset 3
    ├── pred_202601.parquet
    └── pred_202601.eof
```

각 데이터셋마다 별도의 `REMOTE_PATH` 사용

---

## 마이그레이션 가이드

### v2.3에서 v2.4로

**변경 사항:**
- 원격 서버에서 디렉토리 구조가 한 단계 제거됨

**영향:**
- 기존 데이터가 있는 경우 경로가 변경됨
- 다운스트림 시스템의 경로 설정 업데이트 필요

**조치:**

**옵션 A: 기존 파일 수동 이동**
```bash
# 원격 서버에서 수동 이동
ssh user@remote
mv /data/MMS_SCORE/propensityScoreDF/*.parquet /data/MMS_SCORE/
rmdir /data/MMS_SCORE/propensityScoreDF
```

**옵션 B: 재전송**
```bash
# v2.4로 재전송 (기존 파일 자동 정리됨)
python hdfs_transfer.py --merge-partitions
```

---

## 테스트 시나리오

### 시나리오 1: 파티션 통합

**단계:**
1. v2.4로 파티션 통합 전송
2. 원격 서버에서 파일 위치 확인

**예상 결과:**
```bash
ls /data/TOSIFDAT/LAKE/MMS_SCORE/
# mth_mms_rcv_ract_score_202601.parquet
# mth_mms_rcv_ract_score_202601.eof
```

**확인:** propensityScoreDF 디렉토리가 없음 ✅

---

### 시나리오 2: 파티션 구조 유지

**단계:**
1. v2.4로 파티션 구조 유지 전송
2. 원격 서버에서 파일 위치 확인

**예상 결과:**
```bash
ls /data/TOSIFDAT/LAKE/MMS_SCORE/
# part-00000-xxx.parquet
# part-00001-xxx.parquet
# _SUCCESS
# data.eof
```

**확인:** 디렉토리 없이 파일들만 존재 ✅

---

## 파일 목록

### 수정된 파일

1. **hdfs_transfer.py**
   - `compress_data()`: `-C` 옵션 추가
   - `remove_remote_files()`: 경로 수정, 와일드카드 삭제 추가

2. **CHANGELOG_v2.4.md** (신규)
   - v2.4 변경 사항 문서

---

## 요약

### 변경 내용
✅ 압축 시 디렉토리 구조 제거 (`-C` 옵션)  
✅ 원격 서버에서 파일들이 `REMOTE_PATH` 바로 밑에 생성  
✅ 삭제 로직 수정 (직접 경로 사용)  

### 최종 구조
```
{REMOTE_PATH}/
├── data.parquet  ← 디렉토리 없이 바로
└── data.eof
```

### 권장 사용법
```bash
# 파티션 통합
python hdfs_transfer.py --merge-partitions --skip-cleanup

# 파티션 구조 유지
python hdfs_transfer.py --skip-cleanup
```

**결과:** 두 경우 모두 파일들이 `REMOTE_PATH` 바로 밑에 생성됨

---

**이전 버전**: v2.3 (tar.gz 삭제 추가)  
**현재 버전**: v2.4 (디렉토리 구조 제거)  
**다음 업데이트**: TBD
