# 변경 사항 (v2.1)

**날짜**: 2026-01-26  
**변경자**: AI Assistant

---

## 주요 변경 사항

### 원격 파일 삭제 로직 개선 ✅

**문제점:**
- 기존: 원격 서버의 디렉토리 전체를 삭제 (`rm -rf {remote_path}/{dir_name}`)
- 디렉토리 안에 다른 파일들도 함께 삭제됨
- 불필요한 데이터 손실 위험

**해결책:**
- 변경 후: 특정 파일만 선택적으로 삭제
- OUTPUT_FILENAME과 EOF 파일만 삭제
- 디렉토리 구조와 다른 파일들은 유지

---

## 상세 변경 내역

### 1. 함수 수정

**Before:**
```python
def remove_remote_directory(remote_user, remote_password, remote_ip, remote_path, dir_name):
    """원격 디렉토리 삭제"""
    ssh_cmd = f'... rm -rf {remote_path}/{dir_name}'
    return run_command(ssh_cmd)
```

**After:**
```python
def remove_remote_files(remote_user, remote_password, remote_ip, remote_path, dir_name, output_filename, archive_name):
    """원격 서버의 기존 파일 삭제 (디렉토리는 유지)"""
    # 1. 디렉토리 내의 OUTPUT_FILENAME
    output_file_path = f"{remote_path}/{dir_name}/{output_filename}"
    
    # 2. EOF 파일
    base_name = archive_name.replace('.parquet', '').replace('.tar.gz', '')
    eof_file_path = f"{remote_path}/{base_name}.eof"
    
    # 특정 파일만 삭제
    rm_cmd = f'... rm -f {output_file_path} {eof_file_path}'
    return run_command(rm_cmd)
```

### 2. 삭제되는 파일

**파일 목록:**
1. `{REMOTE_PATH}/{DIR_NAME}/{OUTPUT_FILENAME}`
   - 예: `/remote/data/table_name/data_202601.parquet`
   
2. `{REMOTE_PATH}/{base_name}.eof`
   - 예: `/remote/data/data_202601.eof`
   - base_name = ARCHIVE_NAME에서 .tar.gz 제거

**유지되는 것:**
- 디렉토리: `{REMOTE_PATH}/{DIR_NAME}/`
- 다른 파일들: `{REMOTE_PATH}/{DIR_NAME}/other_*.parquet`
- 관련 없는 EOF 파일들

---

## 사용 예제

### 예제 1: 기본 동작 (파일만 삭제)

```bash
# .env 설정
OUTPUT_FILENAME=mth_mms_rcv_ract_score_202601.parquet
ARCHIVE_NAME=mth_mms_rcv_ract_score_202601.tar.gz
REMOTE_PATH=/home/user/data

# 실행
python hdfs_transfer.py --merge-partitions --skip-cleanup

# 삭제되는 파일
/home/user/data/table_name/mth_mms_rcv_ract_score_202601.parquet
/home/user/data/mth_mms_rcv_ract_score_202601.eof

# 유지되는 것
/home/user/data/table_name/  (디렉토리)
/home/user/data/table_name/other_file.parquet  (다른 파일)
```

### 예제 2: 파일 삭제 건너뛰기

```bash
# 기존 파일을 유지하고 싶은 경우
python hdfs_transfer.py --merge-partitions --skip-remove --skip-cleanup
```

### 예제 3: 여러 월 데이터 관리

```bash
# 1월 데이터 전송
OUTPUT_FILENAME=data_202601.parquet python hdfs_transfer.py --merge-partitions --skip-cleanup

# 2월 데이터 전송 (1월 데이터는 유지됨)
OUTPUT_FILENAME=data_202602.parquet python hdfs_transfer.py --merge-partitions --skip-cleanup

# 원격 서버에 두 파일 모두 존재
/remote/data/table_name/data_202601.parquet
/remote/data/table_name/data_202602.parquet
```

---

## 비교표

### Before vs After

| 항목 | v2.0 (Before) | v2.1 (After) |
|------|--------------|-------------|
| 삭제 방식 | 디렉토리 전체 삭제 | 특정 파일만 삭제 |
| 명령어 | `rm -rf {dir}` | `rm -f {file1} {file2}` |
| 디렉토리 | 삭제됨 | 유지됨 |
| 다른 파일 | 삭제됨 | 유지됨 |
| 데이터 손실 위험 | 높음 | 낮음 |

### 삭제 대상

| 파일 유형 | Before | After |
|---------|--------|-------|
| OUTPUT_FILENAME | ✅ 삭제 | ✅ 삭제 |
| EOF 파일 | ✅ 삭제 | ✅ 삭제 |
| 디렉토리 | ❌ 삭제 | ✅ 유지 |
| 다른 parquet 파일 | ❌ 삭제 | ✅ 유지 |
| 다른 EOF 파일 | ❌ 삭제 | ✅ 유지 |

---

## 장점

### 1. 데이터 안전성 향상
- 실수로 다른 데이터를 삭제할 위험 제거
- 여러 월/년도 데이터를 한 디렉토리에 관리 가능

### 2. 유연한 파일 관리
```bash
# 같은 디렉토리에 여러 파일 보관 가능
/remote/data/table_name/
  ├── data_202401.parquet  (1월 데이터)
  ├── data_202402.parquet  (2월 데이터)
  ├── data_202403.parquet  (3월 데이터)
  └── backup_old.parquet   (백업 파일)
```

### 3. 점진적 업데이트
- 특정 월 데이터만 업데이트 가능
- 다른 데이터는 영향 없음

---

## 마이그레이션 가이드

### 기존 사용자

**v2.0 동작:**
```bash
# 디렉토리 전체 삭제
rm -rf /remote/path/table_name
```

**v2.1 동작:**
```bash
# 특정 파일만 삭제
rm -f /remote/path/table_name/data_202601.parquet
rm -f /remote/path/data_202601.eof
```

**변경 필요 사항:**
- 없음 (자동으로 새로운 방식 적용)
- 기존 스크립트 그대로 사용 가능
- 더 안전하게 동작

---

## 주의사항

### 1. 파일명 충돌
같은 파일명을 사용하면 이전 파일이 덮어씌워집니다:

```bash
# 첫 실행
OUTPUT_FILENAME=data.parquet python hdfs_transfer.py --merge-partitions

# 두 번째 실행 (같은 파일명)
OUTPUT_FILENAME=data.parquet python hdfs_transfer.py --merge-partitions
# → 이전 data.parquet가 덮어씌워짐
```

**권장:** 날짜나 버전을 파일명에 포함
```bash
OUTPUT_FILENAME=data_202601_v1.parquet
OUTPUT_FILENAME=data_202601_v2.parquet
```

### 2. 디렉토리 정리
디렉토리가 남아있으므로 필요시 수동 정리:

```bash
# 원격 서버에서 직접 정리
ssh user@remote "rm -rf /remote/path/table_name"

# 또는 특정 월만 정리
ssh user@remote "rm -f /remote/path/table_name/data_202401.parquet"
```

### 3. --skip-remove 옵션
기존 파일을 유지하고 싶으면:

```bash
# 기존 파일 유지 (추가 전송)
python hdfs_transfer.py --merge-partitions --skip-remove
```

---

## 테스트 결과

### 기능 테스트
✅ 특정 파일만 삭제 확인  
✅ 디렉토리 유지 확인  
✅ 다른 파일 보존 확인  
✅ EOF 파일 정상 삭제 확인  
✅ 하위 호환성 보장  

### 시나리오 테스트

**시나리오 1: 여러 월 데이터 관리**
```bash
# 1월 데이터
OUTPUT_FILENAME=data_202601.parquet python hdfs_transfer.py --merge-partitions
# 결과: /remote/data/table/data_202601.parquet ✅

# 2월 데이터
OUTPUT_FILENAME=data_202602.parquet python hdfs_transfer.py --merge-partitions
# 결과: 
#   /remote/data/table/data_202601.parquet ✅ (유지)
#   /remote/data/table/data_202602.parquet ✅ (추가)
```

**시나리오 2: 동일 파일 업데이트**
```bash
# 첫 실행
OUTPUT_FILENAME=data.parquet python hdfs_transfer.py --merge-partitions
# 결과: /remote/data/table/data.parquet (v1)

# 재실행
OUTPUT_FILENAME=data.parquet python hdfs_transfer.py --merge-partitions
# 결과: /remote/data/table/data.parquet (v2, 덮어씌움) ✅
```

**시나리오 3: 다른 파일 보존**
```bash
# 원격 서버 초기 상태
/remote/data/table/backup.parquet
/remote/data/table/test.parquet

# 실행
OUTPUT_FILENAME=new_data.parquet python hdfs_transfer.py --merge-partitions

# 결과
/remote/data/table/backup.parquet ✅ (유지)
/remote/data/table/test.parquet ✅ (유지)
/remote/data/table/new_data.parquet ✅ (추가)
```

---

## 파일 목록

### 수정된 파일
1. **hdfs_transfer.py**
   - `remove_remote_directory()` → `remove_remote_files()` 함수명 변경
   - 디렉토리 전체 삭제 → 특정 파일만 삭제
   - 함수 인자 추가: `output_filename`, `archive_name`

2. **HDFS_TRANSFER_GUIDE.md**
   - 원격 파일 삭제 동작 설명 추가
   - 예제 업데이트

3. **CHANGELOG_v2.1.md** (신규)
   - v2.1 변경 사항 문서

---

## 요약

### 변경 내용
✅ 원격 디렉토리 전체 삭제 → 특정 파일만 삭제  
✅ 디렉토리 및 다른 파일 보존  
✅ 더 안전한 파일 관리  

### 주요 장점
- 🛡️ 데이터 손실 위험 감소
- 📁 여러 파일 동시 관리 가능
- 🔄 점진적 업데이트 지원

### 권장 사용법
```bash
# 월별 데이터 관리
OUTPUT_FILENAME=data_202601.parquet python hdfs_transfer.py --merge-partitions --skip-cleanup
OUTPUT_FILENAME=data_202602.parquet python hdfs_transfer.py --merge-partitions --skip-cleanup
OUTPUT_FILENAME=data_202603.parquet python hdfs_transfer.py --merge-partitions --skip-cleanup
```

---

**이전 버전**: v2.0 (OUTPUT_FILENAME 환경 변수 추가, --skip-cleanup 옵션 추가)  
**현재 버전**: v2.1 (원격 파일 선택적 삭제)  
**다음 업데이트**: TBD
