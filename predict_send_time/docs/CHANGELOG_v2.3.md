# 변경 사항 (v2.3)

**날짜**: 2026-01-26  
**변경자**: AI Assistant

---

## 주요 변경 사항

### 파티션 구조 유지 모드 삭제 로직 수정 ✅

**문제점:**
- v2.2에서는 파티션 구조 유지 모드에서 디렉토리와 EOF만 삭제
- tar.gz 파일을 삭제하지 않음
- 기존 tar.gz가 남아있을 경우 문제 발생 가능

**해결책:**
- 파티션 구조 유지 모드에서 **tar.gz 파일도 함께 삭제**
- ARCHIVE_NAME에 해당하는 모든 파일 삭제 (디렉토리, tar.gz, EOF)

---

## 상세 변경 내역

### 1. 삭제 대상 추가

**Before (v2.2):**
```python
# --merge-partitions 미사용
rm -rf {dir_path}      # 디렉토리
rm -f {eof_file_path}  # EOF
```

**After (v2.3):**
```python
# --merge-partitions 미사용
rm -rf {dir_path}         # 디렉토리
rm -f {tar_gz_path}       # tar.gz 추가!
rm -f {eof_file_path}     # EOF
```

### 2. 모드별 삭제 대상

#### 파티션 통합 모드 (`--merge-partitions`)

**삭제 대상:**
1. `{REMOTE_PATH}/{DIR_NAME}/{OUTPUT_FILENAME}` - parquet 파일
2. `{REMOTE_PATH}/{base_name}.eof` - EOF 파일

**변경 없음** - v2.2와 동일

---

#### 파티션 구조 유지 모드 (기본값)

**삭제 대상:**
1. `{REMOTE_PATH}/{DIR_NAME}/` - 디렉토리 전체
2. `{REMOTE_PATH}/{ARCHIVE_NAME}` - **tar.gz 파일 (추가됨)** ⭐
3. `{REMOTE_PATH}/{base_name}.eof` - EOF 파일

**변경:** tar.gz 파일 삭제 추가

---

## 비교표

### v2.2 vs v2.3 (파티션 구조 유지 모드)

| 항목 | v2.2 | v2.3 |
|------|------|------|
| **DIR_NAME 디렉토리** | ✅ 삭제 | ✅ 삭제 |
| **tar.gz 파일** | ❌ 미삭제 | ✅ 삭제 ⭐ |
| **EOF 파일** | ✅ 삭제 | ✅ 삭제 |
| **REMOTE_PATH** | ✅ 유지 | ✅ 유지 |

---

## 이유 및 근거

### 왜 tar.gz를 삭제해야 하나?

**시나리오:**
1. 첫 실행: `raw_data_202601.tar.gz` 전송 및 압축 해제
2. 압축 해제 후 tar.gz 자동 삭제됨
3. **재실행**: 같은 이름으로 다시 전송

**v2.2 문제점:**
- 재전송 시 디렉토리와 EOF만 삭제
- 만약 이전 tar.gz가 남아있으면 (압축 해제 실패 등) 충돌 가능

**v2.3 해결:**
- 전송 전에 tar.gz도 함께 삭제
- ARCHIVE_NAME 관련 모든 파일 깔끔하게 정리

### --skip-remove 옵션과의 관계

**--skip-remove 미사용 (기본값):**
```bash
# 전송 전 삭제
rm -rf /remote/path/table_name
rm -f /remote/path/data.tar.gz
rm -f /remote/path/data.eof

# 전송 및 압축 해제
scp data.tar.gz remote:/remote/path/
ssh remote "cd /remote/path && tar -xzf data.tar.gz && rm data.tar.gz"
```

**--skip-remove 사용:**
```bash
# 전송 전 삭제 안함

# 전송 및 압축 해제
scp data.tar.gz remote:/remote/path/  # 기존 파일 덮어씀
ssh remote "cd /remote/path && tar -xzf data.tar.gz && rm data.tar.gz"
```

---

## 사용 예제

### 예제 1: 정상적인 재전송 (v2.3)

**초기 상태:**
```
/remote/path/
├── table_name/  (기존 디렉토리)
├── data_202601.tar.gz  (이전 tar.gz, 있을 수도)
└── data_202601.eof  (기존 EOF)
```

**실행:**
```bash
python hdfs_transfer.py  # --merge-partitions 미사용
```

**전송 전 삭제:**
```
❌ /remote/path/table_name/
❌ /remote/path/data_202601.tar.gz  ⭐ 삭제됨
❌ /remote/path/data_202601.eof
```

**전송 후:**
```
/remote/path/
├── table_name/  (새로 생성)
└── data_202601.eof  (새로 생성)
```

**결과:** 깔끔하게 대체됨 ✅

---

### 예제 2: --skip-remove 사용

**실행:**
```bash
python hdfs_transfer.py --skip-remove
```

**동작:**
- 전송 전 삭제 안함
- tar.gz 전송 시 기존 파일 덮어씀
- 압축 해제 시 디렉토리에 파일 추가/덮어씀
- 압축 해제 후 tar.gz 자동 삭제

---

## 전체 흐름 비교

### v2.2 흐름 (파티션 구조 유지)

```
1. 전송 전 삭제 (--skip-remove 미사용)
   - DIR_NAME 디렉토리 삭제
   - EOF 파일 삭제
   - tar.gz 파일 ❌ 미삭제

2. 전송
   - tar.gz 전송 (기존 파일이 있으면 덮어씀)

3. 압축 해제
   - tar.gz 압축 해제
   - tar.gz 자동 삭제

4. EOF 생성
```

### v2.3 흐름 (파티션 구조 유지)

```
1. 전송 전 삭제 (--skip-remove 미사용)
   - DIR_NAME 디렉토리 삭제
   - tar.gz 파일 삭제 ⭐ 추가
   - EOF 파일 삭제

2. 전송
   - tar.gz 전송

3. 압축 해제
   - tar.gz 압축 해제
   - tar.gz 자동 삭제

4. EOF 생성
```

---

## 명령어 변경

### 파티션 통합 모드 (변경 없음)

```bash
rm -f /remote/path/table_name/data.parquet
rm -f /remote/path/data.eof
```

### 파티션 구조 유지 모드 (tar.gz 추가)

**Before (v2.2):**
```bash
rm -rf /remote/path/table_name
rm -f /remote/path/data.eof
```

**After (v2.3):**
```bash
rm -rf /remote/path/table_name
rm -f /remote/path/data.tar.gz  # 추가!
rm -f /remote/path/data.eof
```

---

## 코드 변경

```python
# v2.2
else:
    # --merge-partitions 미사용
    dir_path = f"{remote_path}/{dir_name}"
    
    rm_cmd = f"rm -rf {dir_path} && rm -f {eof_file_path}"

# v2.3
else:
    # --merge-partitions 미사용
    dir_path = f"{remote_path}/{dir_name}"
    tar_gz_path = f"{remote_path}/{archive_name}"  # 추가
    
    rm_cmd = f"rm -rf {dir_path} && rm -f {tar_gz_path} {eof_file_path}"
```

---

## 테스트 시나리오

### 시나리오 1: 정상 재전송

**단계:**
1. 첫 전송: 성공
2. 재전송: 같은 ARCHIVE_NAME

**v2.2 동작:**
- 디렉토리, EOF 삭제
- tar.gz 전송 (덮어씀)
- 정상 동작 ✅

**v2.3 동작:**
- 디렉토리, tar.gz, EOF 삭제
- tar.gz 전송
- 더 깔끔함 ✅

---

### 시나리오 2: 압축 해제 실패 후 재시도

**단계:**
1. 첫 전송: 압축 해제 중 실패 (tar.gz 남음)
2. 재전송: 같은 ARCHIVE_NAME

**v2.2 동작:**
- 디렉토리, EOF 삭제
- 기존 tar.gz 남아있음 ⚠️
- 새 tar.gz 전송 (덮어씀)
- 동작은 하지만 비효율적

**v2.3 동작:**
- 디렉토리, 기존 tar.gz, EOF 삭제 ✅
- 새 tar.gz 전송
- 깔끔하게 해결 ✅

---

## 마이그레이션 가이드

### v2.2에서 v2.3으로

**변경 사항:**
- 파티션 구조 유지 모드에서 tar.gz 파일도 삭제

**영향:**
- 기존 tar.gz가 남아있던 경우 삭제됨
- 더 깔끔한 파일 관리

**조치:**
- 업그레이드 권장
- 기존 명령어 그대로 사용 가능
- 동작 개선됨

---

## 파일 목록

### 수정된 파일

1. **hdfs_transfer.py**
   - `remove_remote_files()` 함수에 tar.gz 삭제 추가
   - 파티션 구조 유지 모드 로직 개선

2. **REMOTE_FILE_DELETION_GUIDE.md**
   - 파티션 구조 유지 모드 설명 업데이트
   - 삭제 대상에 tar.gz 추가

3. **HDFS_TRANSFER_GUIDE.md**
   - 원격 파일 삭제 섹션 업데이트

4. **CHANGELOG_v2.3.md** (신규)
   - v2.3 변경 사항 문서

---

## 요약

### 변경 내용
✅ 파티션 구조 유지 모드: tar.gz 파일 삭제 추가  
✅ ARCHIVE_NAME 관련 모든 파일 깔끔하게 정리  
✅ 재전송 시 더 안정적  

### 삭제 대상 (파티션 구조 유지)
1. DIR_NAME 디렉토리
2. **ARCHIVE_NAME (tar.gz)** ⭐ 추가
3. EOF 파일

### 권장 사용법
```bash
# 파티션 구조 유지 모드
python hdfs_transfer.py --skip-cleanup

# 파티션 통합 모드 (변경 없음)
python hdfs_transfer.py --merge-partitions --skip-cleanup
```

---

**이전 버전**: v2.2 (--merge-partitions 옵션 고려)  
**현재 버전**: v2.3 (tar.gz 삭제 추가)  
**다음 업데이트**: TBD
