# 변경 사항 (v2.2)

**날짜**: 2026-01-26  
**변경자**: AI Assistant

---

## 주요 변경 사항

### --merge-partitions 옵션에 따른 삭제 방식 구분 ✅

**문제점:**
- v2.1에서는 `--merge-partitions` 옵션을 고려하지 않음
- 모든 경우에 OUTPUT_FILENAME만 삭제
- 파티션 구조 유지 모드에서는 불필요한 동작

**해결책:**
- `--merge-partitions` 사용 여부에 따라 다른 삭제 로직 적용
- 파티션 통합 모드: OUTPUT_FILENAME만 삭제 (선택적)
- 파티션 구조 유지 모드: 디렉토리 전체 삭제 (깔끔한 대체)

---

## 상세 변경 내역

### 1. 삭제 로직 분기

**함수 시그니처 변경:**
```python
# Before (v2.1)
def remove_remote_files(remote_user, remote_password, remote_ip, remote_path, 
                       dir_name, output_filename, archive_name):

# After (v2.2)
def remove_remote_files(remote_user, remote_password, remote_ip, remote_path, 
                       dir_name, output_filename, archive_name, 
                       merge_partitions=False):  # 추가됨
```

**로직:**
```python
if merge_partitions:
    # 파티션 통합 모드: 특정 파일만 삭제
    output_file_path = f"{remote_path}/{dir_name}/{output_filename}"
    rm_cmd = f"rm -f {output_file_path} {eof_file_path}"
else:
    # 파티션 구조 유지 모드: 디렉토리 전체 삭제
    dir_path = f"{remote_path}/{dir_name}"
    rm_cmd = f"rm -rf {dir_path} && rm -f {eof_file_path}"
```

### 2. 두 가지 모드

#### 모드 1: 파티션 통합 (`--merge-partitions`)

**특징:**
- OUTPUT_FILENAME만 선택적으로 삭제
- 디렉토리 및 다른 파일 유지
- 여러 월 데이터 동시 관리 가능

**삭제 대상:**
- `{REMOTE_PATH}/{DIR_NAME}/{OUTPUT_FILENAME}`
- `{REMOTE_PATH}/{base_name}.eof`

**사용 예:**
```bash
# 1월 데이터
OUTPUT_FILENAME=data_202601.parquet python hdfs_transfer.py --merge-partitions

# 2월 데이터 (1월은 유지됨)
OUTPUT_FILENAME=data_202602.parquet python hdfs_transfer.py --merge-partitions
```

---

#### 모드 2: 파티션 구조 유지 (기본값)

**특징:**
- 디렉토리 전체 삭제
- 파티션 파일들이 많아 개별 삭제 불가능
- 깔끔한 전체 대체

**삭제 대상:**
- `{REMOTE_PATH}/{DIR_NAME}/` (디렉토리 전체)
- `{REMOTE_PATH}/{base_name}.eof`

**사용 예:**
```bash
# 전체 데이터 재생성
python hdfs_transfer.py
```

---

## 비교표

### v2.1 vs v2.2

| 항목 | v2.1 | v2.2 |
|------|------|------|
| **옵션 고려** | ❌ 미고려 | ✅ 고려 |
| **파티션 통합 모드** | OUTPUT_FILENAME만 삭제 | OUTPUT_FILENAME만 삭제 ✅ |
| **파티션 구조 유지 모드** | OUTPUT_FILENAME만 삭제 ❌ | 디렉토리 전체 삭제 ✅ |
| **로직 정확성** | ❌ 불완전 | ✅ 완전 |

### 모드별 삭제 방식

| 항목 | 파티션 통합 모드 | 파티션 구조 유지 모드 |
|------|----------------|-------------------|
| **옵션** | `--merge-partitions` | (미사용) |
| **삭제 명령** | `rm -f {file}` | `rm -rf {dir}` |
| **OUTPUT_FILENAME** | ✅ 삭제 | ❌ 해당 없음 |
| **디렉토리** | ✅ 유지 | ❌ 삭제 |
| **파티션 파일들** | ✅ 유지 (다른 월) | ❌ 삭제 (전체) |
| **EOF 파일** | ✅ 삭제 | ✅ 삭제 |
| **여러 월 관리** | ✅ 가능 | ❌ 불가능 |

---

## 사용 예제

### 예제 1: 파티션 통합 모드 (월별 데이터 누적)

```bash
# .env
OUTPUT_FILENAME=data_202601.parquet

# 1월 데이터 전송
python hdfs_transfer.py --merge-partitions --skip-cleanup

# 결과: /remote/data/table_name/data_202601.parquet

# 2월 데이터 전송
OUTPUT_FILENAME=data_202602.parquet python hdfs_transfer.py --merge-partitions --skip-cleanup

# 결과: 
#   /remote/data/table_name/data_202601.parquet (유지)
#   /remote/data/table_name/data_202602.parquet (추가)
```

### 예제 2: 파티션 구조 유지 모드 (전체 대체)

```bash
# .env
ARCHIVE_NAME=raw_data_202601.tar.gz

# 첫 실행
python hdfs_transfer.py --skip-cleanup

# 결과: /remote/data/table_name/ (디렉토리 생성, 파티션 구조 포함)

# 재실행 (다음 달 데이터)
ARCHIVE_NAME=raw_data_202602.tar.gz python hdfs_transfer.py --skip-cleanup

# 동작:
#   1. /remote/data/table_name/ 전체 삭제
#   2. /remote/data/table_name/ 재생성 (새 데이터)
#   3. /remote/data/raw_data_202602.eof 생성

# 결과: 202601 데이터는 완전히 대체됨
```

---

## 테스트 시나리오

### 시나리오 1: 파티션 통합 - 여러 월 관리

**초기 상태:**
```
/remote/data/table_name/  (빈 디렉토리)
```

**1월 데이터 전송:**
```bash
OUTPUT_FILENAME=data_202601.parquet python hdfs_transfer.py --merge-partitions
```

**상태:**
```
/remote/data/table_name/
└── data_202601.parquet ✅
```

**2월 데이터 전송:**
```bash
OUTPUT_FILENAME=data_202602.parquet python hdfs_transfer.py --merge-partitions
```

**삭제 동작:**
- `data_202602.parquet` 삭제 시도 (파일 없음, 에러 없이 진행)
- EOF 파일 삭제

**최종 상태:**
```
/remote/data/table_name/
├── data_202601.parquet ✅ (유지)
└── data_202602.parquet ✅ (추가)
```

**결과:** ✅ 성공 (여러 월 데이터 동시 관리)

---

### 시나리오 2: 파티션 구조 유지 - 전체 대체

**초기 상태:**
```
/remote/data/table_name/
├── year=2024/
│   ├── month=01/...
│   └── month=12/...
└── _SUCCESS
```

**재전송:**
```bash
python hdfs_transfer.py
```

**삭제 동작:**
- `/remote/data/table_name/` 디렉토리 전체 삭제
- EOF 파일 삭제

**전송 후:**
```
/remote/data/table_name/  (새로 생성)
├── year=2024/
│   └── month=01/...  (새 데이터만)
└── _SUCCESS
```

**결과:** ✅ 성공 (깔끔한 전체 대체)

---

## 장점

### 1. 정확한 로직
- 각 모드에 맞는 적절한 삭제 방식 적용
- 불필요한 파일 조회 제거

### 2. 파티션 통합 모드
- 여러 월/버전 데이터 동시 관리
- 선택적 업데이트 가능
- 디스크 공간 효율적 관리

### 3. 파티션 구조 유지 모드
- 깔끔한 전체 대체
- 파티션 구조 변경 가능
- 간단한 롤백 (재전송)

---

## 마이그레이션 가이드

### v2.1에서 v2.2로

**변경 사항:**
- 함수 호출 시 `merge_partitions` 인자 추가
- 내부 로직 자동 분기

**코드 수정:**
```python
# Before (v2.1)
remove_remote_files(user, pwd, ip, path, dir_name, output_file, archive)

# After (v2.2)
remove_remote_files(user, pwd, ip, path, dir_name, output_file, archive, 
                   merge_partitions=args.merge_partitions)  # 추가
```

**사용자 영향:**
- 명령어는 동일
- 동작이 더 정확해짐
- 파티션 구조 유지 모드에서 이전에 OUTPUT_FILENAME을 삭제하려던 불필요한 동작 제거

---

## 주의사항

### 1. 파티션 구조 유지 모드
디렉토리 전체가 삭제되므로 주의:

```bash
# 백업이 필요하면 먼저 수동으로 처리
ssh user@remote "cp -r /remote/path/table_name /remote/path/table_name.backup"

# 전송
python hdfs_transfer.py
```

### 2. 모드 선택
- **여러 파일 관리**: `--merge-partitions` 사용
- **전체 대체**: `--merge-partitions` 미사용

### 3. --skip-remove
기존 데이터 유지가 필요하면:

```bash
# 기존 데이터 유지하고 추가
python hdfs_transfer.py --merge-partitions --skip-remove
```

---

## 파일 목록

### 수정된 파일
1. **hdfs_transfer.py**
   - `remove_remote_files()` 함수에 `merge_partitions` 인자 추가
   - 모드별 분기 로직 구현
   - 함수 호출 시 `args.merge_partitions` 전달

2. **REMOTE_FILE_DELETION_GUIDE.md**
   - 두 가지 모드 설명 추가
   - 비교표 추가
   - 예제 업데이트

3. **CHANGELOG_v2.2.md** (신규)
   - v2.2 변경 사항 문서

---

## 요약

### 변경 내용
✅ `--merge-partitions` 옵션 고려  
✅ 모드별 적절한 삭제 방식 적용  
✅ 정확한 로직 구현  

### 모드별 동작
- **파티션 통합**: OUTPUT_FILENAME만 삭제 (선택적)
- **파티션 구조 유지**: 디렉토리 전체 삭제 (전체 대체)

### 권장 사용법

**월별 데이터 누적:**
```bash
python hdfs_transfer.py --merge-partitions --skip-cleanup
```

**전체 데이터 재생성:**
```bash
python hdfs_transfer.py --skip-cleanup
```

---

**이전 버전**: v2.1 (원격 파일 선택적 삭제)  
**현재 버전**: v2.2 (--merge-partitions 옵션 고려)  
**다음 업데이트**: TBD
