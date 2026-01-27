# 원격 파일 삭제 가이드

## 개요

hdfs_transfer.py 스크립트는 데이터를 원격 서버로 전송하기 전에 기존 파일을 선택적으로 삭제합니다.

---

## 동작 방식 (v2.2)

### --merge-partitions 옵션에 따른 삭제 방식

삭제 동작은 `--merge-partitions` 옵션 사용 여부에 따라 다릅니다.

#### 모드 1: 파티션 통합 모드 (`--merge-partitions` 사용)

**삭제되는 파일:**
1. **Parquet 파일**: `{REMOTE_PATH}/{DIR_NAME}/{OUTPUT_FILENAME}`
2. **EOF 파일**: `{REMOTE_PATH}/{base_name}.eof`

**유지되는 것:**
- ✅ 디렉토리 구조: `{REMOTE_PATH}/{DIR_NAME}/`
- ✅ 다른 파일들: `{REMOTE_PATH}/{DIR_NAME}/other_*.parquet`
- ✅ 다른 EOF 파일들: `{REMOTE_PATH}/other_*.eof`

**특징:** 특정 파일만 선택적으로 삭제, 여러 월 데이터 관리 가능

---

#### 모드 2: 파티션 구조 유지 모드 (`--merge-partitions` 미사용)

**삭제되는 것:**
1. **디렉토리**: `{REMOTE_PATH}/{DIR_NAME}/` (모든 하위 파일 포함)
2. **tar.gz 파일**: `{REMOTE_PATH}/{ARCHIVE_NAME}`
3. **EOF 파일**: `{REMOTE_PATH}/{base_name}.eof`

**유지되는 것:**
- ✅ `{REMOTE_PATH}` 디렉토리 자체
- ✅ 다른 디렉토리들
- ✅ 다른 tar.gz 파일들
- ✅ 다른 EOF 파일들

**특징:** ARCHIVE_NAME에 해당하는 모든 관련 파일 삭제

**참고:** 
- tar.gz는 전송 후 압축 해제 시 자동 삭제되지만, 전송 전 기존 파일이 있다면 먼저 삭제
- `--skip-remove` 사용 시: 삭제 단계를 건너뛰고, 압축 해제 시에만 tar.gz 자동 삭제

---

## 구체적인 예시

### 예시 1: 파티션 통합 모드 (`--merge-partitions`)

**환경 설정:**
```bash
# .env
REMOTE_PATH=/home/user/data
OUTPUT_FILENAME=mth_mms_rcv_ract_score_202601.parquet
ARCHIVE_NAME=mth_mms_rcv_ract_score_202601.tar.gz
```

**원격 서버 초기 상태:**
```
/home/user/data/
├── table_name/
│   ├── mth_mms_rcv_ract_score_202601.parquet  (이전 버전)
│   ├── mth_mms_rcv_ract_score_202512.parquet  (12월 데이터)
│   └── backup_old.parquet                      (백업 파일)
├── mth_mms_rcv_ract_score_202601.eof          (이전 EOF)
└── mth_mms_rcv_ract_score_202512.eof          (12월 EOF)
```

**실행:**
```bash
python hdfs_transfer.py --merge-partitions
```

**삭제되는 파일:**
```
✅ /home/user/data/table_name/mth_mms_rcv_ract_score_202601.parquet
✅ /home/user/data/mth_mms_rcv_ract_score_202601.eof
```

**유지되는 파일:**
```
✅ /home/user/data/table_name/                              (디렉토리)
✅ /home/user/data/table_name/mth_mms_rcv_ract_score_202512.parquet
✅ /home/user/data/table_name/backup_old.parquet
✅ /home/user/data/mth_mms_rcv_ract_score_202512.eof
```

**전송 후 최종 상태:**
```
/home/user/data/
├── table_name/
│   ├── mth_mms_rcv_ract_score_202601.parquet  (새 버전) ⬅️ 업데이트됨
│   ├── mth_mms_rcv_ract_score_202512.parquet  (유지)
│   └── backup_old.parquet                      (유지)
├── mth_mms_rcv_ract_score_202601.eof          (새로 생성) ⬅️ 업데이트됨
└── mth_mms_rcv_ract_score_202512.eof          (유지)
```

---

### 예시 2: 파티션 구조 유지 모드 (`--merge-partitions` 미사용)

**환경 설정:**
```bash
# .env
REMOTE_PATH=/home/user/data
ARCHIVE_NAME=raw_data_202601.tar.gz
# DIR_NAME은 HDFS_PATH에서 추출됨 (예: table_name)
```

**원격 서버 초기 상태:**
```
/home/user/data/
├── table_name/  (기존 디렉토리)
│   ├── year=2024/
│   │   └── month=12/...
│   └── _SUCCESS
├── raw_data_202601.tar.gz  (이전 tar.gz, 있을 수도 있음)
├── raw_data_202601.eof  (기존 EOF)
└── raw_data_202512.eof  (다른 월 EOF)
```

**실행:**
```bash
# --merge-partitions 미사용 (기본값)
python hdfs_transfer.py
```

**전송 전 삭제되는 것:**
```
❌ /home/user/data/table_name/  (DIR_NAME 디렉토리)
❌ /home/user/data/raw_data_202601.tar.gz  (ARCHIVE_NAME)
❌ /home/user/data/raw_data_202601.eof  (EOF)
```

**유지되는 것:**
```
✅ /home/user/data/  (REMOTE_PATH 디렉토리 자체)
✅ /home/user/data/raw_data_202512.eof  (다른 EOF)
```

**전송 및 압축 해제 후 최종 상태:**
```
/home/user/data/
├── table_name/  (새로 생성, 압축 해제됨)
│   ├── year=2024/
│   │   └── month=01/
│   │       ├── part-00000.parquet
│   │       └── part-00001.parquet
│   └── _SUCCESS
├── raw_data_202601.eof  (새로 생성)
└── raw_data_202512.eof  (유지됨)
```

**참고:**
- raw_data_202601.tar.gz는 압축 해제 후 자동 삭제됨
- ARCHIVE_NAME에 해당하는 모든 파일 (디렉토리, tar.gz, EOF) 깔끔하게 대체

---

### 예시 3: 여러 월 데이터 관리 (파티션 통합 모드)

**시나리오:** 매월 새로운 데이터를 같은 디렉토리에 추가

**1월 데이터 전송:**
```bash
OUTPUT_FILENAME=data_202601.parquet python hdfs_transfer.py --merge-partitions
```

**결과:**
```
/home/user/data/table_name/
└── data_202601.parquet ✅
```

**2월 데이터 전송:**
```bash
OUTPUT_FILENAME=data_202602.parquet python hdfs_transfer.py --merge-partitions
```

**결과:**
```
/home/user/data/table_name/
├── data_202601.parquet ✅ (유지됨)
└── data_202602.parquet ✅ (추가됨)
```

**3월 데이터 전송:**
```bash
OUTPUT_FILENAME=data_202603.parquet python hdfs_transfer.py --merge-partitions
```

**결과:**
```
/home/user/data/table_name/
├── data_202601.parquet ✅ (유지됨)
├── data_202602.parquet ✅ (유지됨)
└── data_202603.parquet ✅ (추가됨)
```

---

### 예시 4: 동일 파일 업데이트 (파티션 통합 모드)

**시나리오:** 같은 파일명으로 재전송 (데이터 수정 후)

**첫 실행:**
```bash
OUTPUT_FILENAME=monthly_data.parquet python hdfs_transfer.py --merge-partitions
```

**결과:**
```
/home/user/data/table_name/monthly_data.parquet (버전 1)
```

**재실행 (데이터 수정 후):**
```bash
OUTPUT_FILENAME=monthly_data.parquet python hdfs_transfer.py --merge-partitions
```

**동작:**
1. 기존 `monthly_data.parquet` 삭제
2. 새로운 `monthly_data.parquet` 전송

**결과:**
```
/home/user/data/table_name/monthly_data.parquet (버전 2) ⬅️ 업데이트됨
```

---

## 모드별 비교

### 삭제 방식 비교표

| 항목 | 파티션 통합 모드<br/>(`--merge-partitions`) | 파티션 구조 유지 모드<br/>(미사용) |
|------|-------------------------------------|--------------------------|
| **삭제 대상** | OUTPUT_FILENAME, EOF | DIR_NAME, tar.gz, EOF |
| **OUTPUT_FILENAME** | ✅ 삭제 | ❌ 해당 없음 |
| **DIR_NAME 디렉토리** | ✅ 유지 | ❌ 삭제 |
| **tar.gz 파일** | ❌ 해당 없음 | ✅ 삭제 |
| **EOF 파일** | ✅ 삭제 | ✅ 삭제 |
| **REMOTE_PATH** | ✅ 유지 | ✅ 유지 |
| **다른 파일/디렉토리** | ✅ 유지 | ✅ 유지 |
| **여러 월 관리** | ✅ 가능 | ❌ 불가능 (덮어씀) |
| **용도** | 월별 데이터 누적 | 전체 데이터 대체 |

### 명령어 비교

**파티션 통합 모드:**
```bash
rm -f /remote/path/table_name/data_202601.parquet  # OUTPUT_FILENAME
rm -f /remote/path/data_202601.eof                 # EOF
```

**파티션 구조 유지 모드:**
```bash
rm -rf /remote/path/table_name           # DIR_NAME 디렉토리
rm -f /remote/path/data_202601.tar.gz    # ARCHIVE_NAME (tar.gz)
rm -f /remote/path/data_202601.eof       # EOF
```

### 사용 시나리오

| 시나리오 | 권장 모드 | 이유 |
|---------|----------|------|
| 월별 데이터 누적 관리 | 파티션 통합 | 특정 월만 업데이트, 다른 월 유지 |
| 전체 데이터 재생성 | 파티션 구조 유지 | 깔끔한 대체 |
| 단일 파일 업데이트 | 파티션 통합 | 선택적 업데이트 |
| 파티션 구조 변경 | 파티션 구조 유지 | 디렉토리 전체 재생성 |

---

## 옵션 사용법

### --skip-remove 옵션

기존 파일을 삭제하지 않고 유지하려면:

```bash
python hdfs_transfer.py --merge-partitions --skip-remove
```

**주의:** 같은 파일명이 이미 존재하면 덮어씌워집니다.

---

## Base Name 계산 규칙

EOF 파일명은 ARCHIVE_NAME에서 계산됩니다:

```python
base_name = archive_name.replace('.parquet', '').replace('.tar.gz', '')
eof_filename = f"{base_name}.eof"
```

**예시:**

| ARCHIVE_NAME | base_name | EOF 파일명 |
|--------------|-----------|-----------|
| `data_202601.tar.gz` | `data_202601` | `data_202601.eof` |
| `data_202601.parquet.tar.gz` | `data_202601` | `data_202601.eof` |
| `mth_data.tar.gz` | `mth_data` | `mth_data.eof` |

---

## 삭제 명령어

실제 실행되는 SSH 명령어:

```bash
rm -f {REMOTE_PATH}/{DIR_NAME}/{OUTPUT_FILENAME} {REMOTE_PATH}/{base_name}.eof
```

**옵션 설명:**
- `-f`: 파일이 없어도 에러 없이 진행 (force)
- 디렉토리는 건드리지 않음
- 지정된 파일만 삭제

---

## 장점

### 1. 데이터 안전성
- 실수로 다른 데이터 삭제 방지
- 디렉토리 구조 유지
- 백업 파일 보존

### 2. 유연한 관리
- 여러 월/년도 데이터를 한 곳에 관리
- 필요한 파일만 선택적으로 업데이트
- 점진적 데이터 축적

### 3. 롤백 가능성
- 이전 버전 데이터 유지 가능
- 문제 발생 시 이전 파일 사용 가능

---

## 비교: v2.0 vs v2.1

| 항목 | v2.0 (이전) | v2.1 (현재) |
|------|------------|------------|
| **삭제 방식** | 디렉토리 전체 삭제 | 특정 파일만 삭제 |
| **명령어** | `rm -rf {dir}` | `rm -f {file1} {file2}` |
| **디렉토리** | ❌ 삭제됨 | ✅ 유지됨 |
| **다른 파일** | ❌ 삭제됨 | ✅ 유지됨 |
| **여러 월 관리** | ❌ 불가능 | ✅ 가능 |
| **데이터 손실 위험** | 🔴 높음 | 🟢 낮음 |

---

## 주의사항

### 1. 파일명 중복
같은 `OUTPUT_FILENAME`을 사용하면 덮어씌워집니다:

```bash
# 같은 파일명 재사용
OUTPUT_FILENAME=data.parquet python hdfs_transfer.py --merge-partitions
# → 이전 data.parquet 삭제 후 새 파일 전송
```

**권장:** 유니크한 파일명 사용
```bash
OUTPUT_FILENAME=data_202601_v1.parquet
OUTPUT_FILENAME=data_202601_v2.parquet
OUTPUT_FILENAME=data_$(date +%Y%m%d_%H%M%S).parquet
```

### 2. 디스크 공간
여러 파일을 누적하면 디스크 공간 관리 필요:

```bash
# 주기적으로 오래된 파일 정리
ssh user@remote "find /remote/path/table_name -name '*.parquet' -mtime +90 -delete"
```

### 3. 수동 정리
필요시 수동으로 특정 파일 삭제:

```bash
# 특정 월 데이터만 삭제
ssh user@remote "rm -f /remote/path/table_name/data_202601.parquet"
ssh user@remote "rm -f /remote/path/data_202601.eof"

# 디렉토리 전체 삭제 (필요한 경우)
ssh user@remote "rm -rf /remote/path/table_name"
```

---

## 테스트

제공된 테스트 스크립트로 동작 확인:

```bash
# 환경 변수 설정
export REMOTE_USER=your_user
export REMOTE_IP=your_server_ip
export REMOTE_PATH=/tmp/test_hdfs_transfer

# 테스트 실행
./test_remote_file_deletion.sh
```

**테스트 내용:**
1. SSH 연결 확인
2. 테스트 파일 생성
3. 파일 삭제 실행
4. 결과 검증
   - 디렉토리 유지 확인
   - 다른 파일 보존 확인
   - 대상 파일만 삭제 확인

---

## FAQ

**Q: 디렉토리 전체를 삭제하고 싶으면?**
A: SSH로 직접 실행하거나 스크립트를 수정하세요:
```bash
ssh user@remote "rm -rf /remote/path/table_name"
```

**Q: 여러 파일을 한 번에 삭제하려면?**
A: 와일드카드를 사용하여 직접 실행:
```bash
ssh user@remote "rm -f /remote/path/table_name/data_2024*.parquet"
```

**Q: 파일이 삭제되지 않으면?**
A: 다음을 확인하세요:
1. SSH 연결 및 권한
2. 파일 경로가 정확한지
3. 파일이 실제로 존재하는지
4. 로그 메시지 확인

**Q: --skip-remove와 다른 점은?**
A: 
- `--skip-remove`: 삭제 단계를 완전히 건너뜀
- 기본 동작: 특정 파일만 선택적으로 삭제

---

## 요약

### v2.1의 핵심 개선사항

✅ **선택적 삭제**: 특정 파일만 삭제, 디렉토리 유지  
✅ **데이터 안전**: 다른 파일들 보존  
✅ **유연한 관리**: 여러 데이터 버전 동시 관리  
✅ **점진적 업데이트**: 필요한 파일만 업데이트  

### 권장 사용 패턴

```bash
# 월별 데이터 관리
OUTPUT_FILENAME=table_202601.parquet python hdfs_transfer.py --merge-partitions --skip-cleanup
OUTPUT_FILENAME=table_202602.parquet python hdfs_transfer.py --merge-partitions --skip-cleanup
OUTPUT_FILENAME=table_202603.parquet python hdfs_transfer.py --merge-partitions --skip-cleanup

# 결과: 3개월 데이터 모두 보존됨
```

---

**관련 문서:**
- HDFS_TRANSFER_GUIDE.md - 전체 가이드
- CHANGELOG_v2.1.md - 변경 사항 상세
- IMPLEMENTATION_SUMMARY.md - 구현 요약
