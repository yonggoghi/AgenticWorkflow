# 변경 사항 (v2.9)

**날짜**: 2026-01-26  
**변경자**: AI Assistant

---

## 주요 변경 사항

### 압축 해제 시 디렉토리 구조 유지 ✅

**문제점:**
- v2.8까지는 tar.gz 압축 해제 시 파티션들이 원격 경로 루트에 직접 풀림
- 디렉토리 구조 없이 파일들만 풀려서 관리 어려움
- 예: `/remote/path/send_ym=202512/` (바로 루트에 생성)

**해결책:**
- tar.gz 파일 이름과 같은 디렉토리 생성 후 그 안에 압축 해제
- 명확한 디렉토리 구조 제공
- 예: `/remote/path/data/send_ym=202512/`

---

## 상세 변경 내역

### Before (v2.8) - 파일들만 바로 풀림

```bash
python hdfs_transfer.py --extract-remote
```

**압축 방식:**
```bash
# 디렉토리 내용물만 압축
tar -czf data.tar.gz -C propensityScoreDF .
```

**압축 해제 결과:**
```
원격 서버: /remote/path/
├── send_ym=202512/          # 바로 루트에 풀림
│   └── *.parquet
├── _SUCCESS
└── data.eof
```

**문제점:**
- 파티션 디렉토리가 루트에 직접 생성
- 여러 데이터셋을 전송하면 파일들이 섞임
- 디렉토리 구조 관리 어려움

---

### After (v2.9) - 디렉토리 포함 압축

```bash
python hdfs_transfer.py --extract-remote
```

**압축 방식:**
```bash
# 디렉토리 자체를 압축
tar -czf data.tar.gz propensityScoreDF
```

**압축 해제 결과:**
```
원격 서버: /remote/path/
├── propensityScoreDF/       # tar.gz 이름과 같은 디렉토리 생성 ✅
│   └── send_ym=202512/      # 그 안에 파티션
│       └── *.parquet
├── _SUCCESS
└── data.eof
```

**장점:**
- 명확한 디렉토리 구조
- 데이터셋 간 분리
- 관리 용이

---

## 압축 방식 비교

### tar 명령어

**v2.8 (이전):**
```bash
tar -czf data.tar.gz -C propensityScoreDF .
# -C 옵션: propensityScoreDF 디렉토리 안으로 이동
# . : 현재 디렉토리의 내용물만 압축
```

**v2.9 (현재):**
```bash
tar -czf data.tar.gz propensityScoreDF
# propensityScoreDF 디렉토리 자체를 압축
```

---

### 압축 해제 결과

**v2.8:**
```
압축 파일: data.tar.gz
내용:
├── send_ym=202512/
│   └── *.parquet
└── _SUCCESS

압축 해제 후:
/remote/path/
├── send_ym=202512/  ← 바로 루트에 생성
└── _SUCCESS
```

**v2.9:**
```
압축 파일: data.tar.gz
내용:
└── propensityScoreDF/
    ├── send_ym=202512/
    │   └── *.parquet
    └── _SUCCESS

압축 해제 후:
/remote/path/
└── propensityScoreDF/  ← 디렉토리 생성 후 그 안에 풀림
    ├── send_ym=202512/
    └── _SUCCESS
```

---

## 코드 변경

### 1. compress_data() 함수

**Before (v2.8):**
```python
def compress_data(local_tmp_path, dir_name, archive_name):
    """데이터 압축 (디렉토리 구조 제외, 내용물만 압축)"""
    os.chdir(local_tmp_path)
    tar_cmd = f"tar -czf {archive_name} -C {dir_name} ."  # 내용물만
    run_command(tar_cmd)
```

**After (v2.9):**
```python
def compress_data(local_tmp_path, dir_name, archive_name):
    """데이터 압축 (디렉토리 포함)"""
    os.chdir(local_tmp_path)
    tar_cmd = f"tar -czf {archive_name} {dir_name}"  # 디렉토리 포함 ✅
    run_command(tar_cmd)
```

---

### 2. remove_remote_files() 함수

**Before (v2.8):**
```python
if extract_remote:
    # *.parquet, _SUCCESS 개별 파일 삭제
    rm_cmd = f'... rm -f {remote_path}/*.parquet {remote_path}/_SUCCESS {tar_gz_path} {eof_file_path}'
```

**After (v2.9):**
```python
if extract_remote:
    # 압축 해제된 디렉토리 전체 삭제 ✅
    extracted_dir = f"{remote_path}/{dir_name}"
    rm_cmd = f'... rm -rf {extracted_dir} && rm -f {tar_gz_path} {eof_file_path}'
```

---

## 사용 예제

### 예제 1: 압축 파일만 전송 (기본)

```bash
python hdfs_transfer.py
```

**원격 서버:**
```
/remote/path/
├── propensityScoreDF.tar.gz  (압축 파일)
└── propensityScoreDF.eof
```

**동작:** 변경 없음

---

### 예제 2: 압축 해제 (v2.9+)

```bash
python hdfs_transfer.py --extract-remote
```

**원격 서버:**
```
/remote/path/
├── propensityScoreDF/        # 디렉토리 생성 ✅
│   └── send_ym=202512/       # 파티션
│       └── *.parquet
└── propensityScoreDF.eof
```

**장점:**
- 명확한 디렉토리 구조
- `propensityScoreDF` 디렉토리로 데이터셋 식별 용이

---

### 예제 3: 여러 데이터셋 관리

**v2.8 (문제):**
```bash
# 첫 번째 데이터셋
python hdfs_transfer.py --extract-remote

# 두 번째 데이터셋
HDFS_PATH=/data/other_dataset python hdfs_transfer.py --extract-remote
```

**결과:**
```
/remote/path/
├── send_ym=202512/           # 첫 번째 데이터셋
├── date=2024/                # 두 번째 데이터셋
└── ...                       # 파일들이 섞임 ❌
```

---

**v2.9 (해결):**
```bash
# 첫 번째 데이터셋
python hdfs_transfer.py --extract-remote

# 두 번째 데이터셋
HDFS_PATH=/data/other_dataset python hdfs_transfer.py --extract-remote
```

**결과:**
```
/remote/path/
├── propensityScoreDF/        # 첫 번째 데이터셋
│   └── send_ym=202512/
├── other_dataset/            # 두 번째 데이터셋
│   └── date=2024/
└── ...                       # 명확하게 분리 ✅
```

---

## 원격 파일 삭제 동작

### --skip-remove 미사용 (기본)

**v2.8:**
```
삭제 대상:
- {remote_path}/*.parquet
- {remote_path}/_SUCCESS
- {remote_path}/data.tar.gz
- {remote_path}/data.eof
```

**v2.9:**
```
삭제 대상:
- {remote_path}/propensityScoreDF/  (디렉토리 전체) ✅
- {remote_path}/data.tar.gz
- {remote_path}/data.eof
```

**명령어:**
```bash
rm -rf {remote_path}/propensityScoreDF && rm -f {tar_gz_path} {eof_file_path}
```

---

## 파일 구조 비교

### 로컬 (변경 없음)

```
LOCAL_TMP_PATH/
├── propensityScoreDF/
│   └── send_ym=202512/
│       └── *.parquet
└── propensityScoreDF.tar.gz
```

---

### 원격 (압축 파일만 전송)

**v2.8 & v2.9 (동일):**
```
REMOTE_PATH/
├── propensityScoreDF.tar.gz
└── propensityScoreDF.eof
```

---

### 원격 (압축 해제)

**v2.8:**
```
REMOTE_PATH/
├── send_ym=202512/          # 루트에 바로 풀림
│   └── *.parquet
├── _SUCCESS
└── propensityScoreDF.eof
```

**v2.9:**
```
REMOTE_PATH/
├── propensityScoreDF/       # 디렉토리 생성 후 풀림 ✅
│   └── send_ym=202512/
│       └── *.parquet
└── propensityScoreDF.eof
```

---

## 장점

### 1. 명확한 구조
- ✅ 디렉토리로 데이터셋 식별
- ✅ tar.gz 파일명과 디렉토리명 일치
- ✅ 혼란 없음

### 2. 관리 용이
- ✅ 데이터셋 간 명확한 분리
- ✅ 디렉토리 단위 관리 가능
- ✅ 삭제 간편 (`rm -rf propensityScoreDF`)

### 3. 표준 준수
- ✅ 일반적인 tar.gz 압축 해제 방식
- ✅ 사용자 기대에 부합

### 4. 다중 데이터셋 지원
- ✅ 여러 데이터셋을 같은 원격 경로에 전송 가능
- ✅ 각 데이터셋이 독립적인 디렉토리에 위치

---

## 주의사항

### 1. 기존 스크립트 영향

**v2.8 스크립트:**
```bash
# 압축 해제 후 파일 접근
cat /remote/path/send_ym=202512/file.parquet
```

**v2.9 스크립트 (수정 필요):**
```bash
# 디렉토리 경로 추가 필요
cat /remote/path/propensityScoreDF/send_ym=202512/file.parquet
```

⚠️ **원격 서버의 기존 스크립트가 파일 경로를 하드코딩한 경우 수정 필요**

---

### 2. 디스크 공간

**변경 없음:**
- 디렉토리 구조만 변경, 파일 크기는 동일
- 압축 파일 크기도 동일

---

### 3. 압축 파일만 전송 (기본)

**v2.8 & v2.9 (동일):**
```bash
python hdfs_transfer.py
# tar.gz만 전송, 압축 해제 안 함
```

**영향 없음:** 기본 모드는 변경 사항 없음

---

## 마이그레이션 가이드

### v2.8에서 v2.9로

**변경 사항:**
- 압축 해제 시 디렉토리 구조 포함

**영향 받는 경우:**
- `--extract-remote` 옵션 사용하는 경우만 영향
- 기본 모드(압축 파일만 전송)는 영향 없음

**조치 사항:**

1. **원격 서버 스크립트 확인:**
   - 파일 경로를 하드코딩했는지 확인
   - 디렉토리 경로 수정 필요

2. **테스트:**
   ```bash
   # 테스트 실행
   python hdfs_transfer.py --extract-remote
   
   # 파일 구조 확인
   ssh user@remote "ls -la /remote/path/propensityScoreDF/"
   ```

3. **스크립트 수정 예시:**
   ```bash
   # Before (v2.8)
   DATA_PATH="/remote/path/send_ym=202512"
   
   # After (v2.9)
   DATA_PATH="/remote/path/propensityScoreDF/send_ym=202512"
   ```

---

## 호환성

### 영향 없는 경우

✅ 기본 모드 사용자:
```bash
python hdfs_transfer.py
# tar.gz만 전송 → 변경 없음
```

✅ 파티션 통합 모드 사용자:
```bash
python hdfs_transfer.py --merge-partitions
# parquet 직접 전송 → 변경 없음
```

---

### 영향 있는 경우

⚠️ 압축 해제 모드 사용자:
```bash
python hdfs_transfer.py --extract-remote
# 파일 경로 변경됨
```

**대응 방안:**
- 원격 서버 스크립트 수정
- 또는 v2.8 사용 유지

---

## 테스트

### 테스트 케이스 1: 압축 파일만 전송

```bash
python hdfs_transfer.py
```

**확인 사항:**
- [ ] tar.gz 파일 생성
- [ ] 원격 서버에 tar.gz 전송
- [ ] EOF 파일 생성
- [ ] 압축 해제 안 됨

**결과:** 변경 없음 ✅

---

### 테스트 케이스 2: 압축 해제

```bash
python hdfs_transfer.py --extract-remote
```

**확인 사항:**
- [ ] tar.gz 파일 생성
- [ ] 원격 서버에 전송
- [ ] 압축 해제 시 디렉토리 생성 ✅
- [ ] 파티션이 디렉토리 안에 위치 ✅
- [ ] tar.gz 삭제
- [ ] EOF 파일 생성

**디렉토리 구조:**
```
/remote/path/
└── propensityScoreDF/  ✅
    └── send_ym=202512/
```

---

### 테스트 케이스 3: 원격 파일 삭제

```bash
python hdfs_transfer.py --extract-remote
# 재실행
python hdfs_transfer.py --extract-remote
```

**확인 사항:**
- [ ] 기존 디렉토리 삭제 ✅
- [ ] 새 디렉토리 생성
- [ ] 충돌 없음

---

## 파일 목록

### 수정된 파일

1. **hdfs_transfer.py**
   - `compress_data()`: `-C` 옵션 제거, 디렉토리 포함 압축
   - `remove_remote_files()`: 디렉토리 전체 삭제로 변경

2. **CHANGELOG_v2.9.md** (신규)
   - v2.9 변경 사항 문서

---

## 요약

### 변경 내용
✅ 압축 시 디렉토리 포함  
✅ 압축 해제 시 디렉토리 구조 유지  
✅ 명확한 파일 관리 구조  
✅ 다중 데이터셋 지원

### 핵심 개선
- **Before (v2.8):** 파일들이 루트에 직접 풀림
- **After (v2.9):** tar.gz 이름과 같은 디렉토리 생성 후 풀림

### 압축 명령어
- **Before:** `tar -czf data.tar.gz -C propensityScoreDF .`
- **After:** `tar -czf data.tar.gz propensityScoreDF`

### 원격 구조
**v2.8:**
```
/remote/path/
├── send_ym=202512/  (루트에 바로)
└── ...
```

**v2.9:**
```
/remote/path/
└── propensityScoreDF/  (디렉토리 생성)
    └── send_ym=202512/
```

### 사용 예
```bash
# 압축 파일만 전송 (변경 없음)
python hdfs_transfer.py

# 압축 해제 (디렉토리 구조 유지)
python hdfs_transfer.py --extract-remote

# 파티션 통합 (변경 없음)
python hdfs_transfer.py --merge-partitions
```

---

**이전 버전**: v2.8 (압축 해제 선택화)  
**현재 버전**: v2.9 (디렉토리 구조 유지)  
**다음 업데이트**: TBD
