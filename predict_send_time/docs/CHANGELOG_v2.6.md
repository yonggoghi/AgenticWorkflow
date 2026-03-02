# 변경 사항 (v2.6)

**날짜**: 2026-01-26  
**변경자**: AI Assistant

---

## 주요 변경 사항

### --merge-partitions 사용 시 원본 파티션 디렉토리 유지 ✅

**문제점:**
- v2.5까지는 `--merge-partitions` 사용 시 원본 파티션 디렉토리를 삭제
- 통합된 파일을 새 디렉토리로 이동하는 불필요한 작업 수행
- 원본 데이터 손실 위험

**해결책:**
- 통합 파일을 별도 디렉토리(`{DIR_NAME}_merged`)에 직접 생성
- 원본 파티션 디렉토리는 그대로 유지
- 파일 이동 없이 LOCAL_TMP_PATH에서 직접 압축

---

## 상세 변경 내역

### Before (v2.5) - 문제 있는 동작

```bash
python hdfs_transfer.py --merge-partitions
```

**처리 과정:**
```
1. 원본 파티션 디렉토리: /tmp/propensityScoreDF/
   ├── partition1/*.parquet
   ├── partition2/*.parquet
   └── ...

2. 통합 파일 생성: /tmp/merged.parquet

3. 원본 파티션 디렉토리 삭제 ❌
   - /tmp/propensityScoreDF/ 삭제됨

4. 새 디렉토리 생성: /tmp/propensityScoreDF/
   
5. 통합 파일 이동: /tmp/merged.parquet → /tmp/propensityScoreDF/merged.parquet

6. 압축: /tmp/propensityScoreDF/ → propensityScoreDF.tar.gz
```

**문제점:**
- 원본 파티션 데이터 손실
- 불필요한 파일 이동 작업
- 동일한 디렉토리명 재사용으로 인한 혼란

---

### After (v2.6) - 개선된 동작

```bash
python hdfs_transfer.py --merge-partitions
```

**처리 과정:**
```
1. 원본 파티션 디렉토리: /tmp/propensityScoreDF/ ✅
   ├── partition1/*.parquet
   ├── partition2/*.parquet
   └── ...
   (유지됨)

2. 통합 파일용 별도 디렉토리 생성: /tmp/propensityScoreDF_merged/

3. 통합 파일 직접 생성: /tmp/propensityScoreDF_merged/merged.parquet
   (파일 이동 없음)

4. 압축: /tmp/propensityScoreDF_merged/ → propensityScoreDF_merged.tar.gz

5. 원격 전송 및 압축 해제
```

**결과:**
```
로컬:
/tmp/
├── propensityScoreDF/           # 원본 파티션 디렉토리 (유지) ✅
│   ├── partition1/*.parquet
│   └── ...
└── propensityScoreDF_merged/    # 통합 파일 디렉토리
    └── merged.parquet

원격:
/remote/path/
└── merged.parquet
```

**장점:**
- 원본 데이터 보존
- 파일 이동 없음
- 명확한 디렉토리 구분

---

## 코드 변경

### hdfs_transfer.py - 파티션 통합 로직

**Before (v2.5):**
```python
if args.merge_partitions:
    source_dir = os.path.join(LOCAL_TMP_PATH, DIR_NAME)
    output_file = os.path.join(LOCAL_TMP_PATH, output_filename)
    
    # 통합 실행
    merge_partitioned_parquet(...)
    
    # 원본 디렉토리 삭제 ❌
    shutil.rmtree(source_dir)
    
    # 새 디렉토리 생성 및 파일 이동 ❌
    merged_dir = os.path.join(LOCAL_TMP_PATH, DIR_NAME)
    os.makedirs(merged_dir, exist_ok=True)
    shutil.move(output_file, final_output)
```

**After (v2.6):**
```python
if args.merge_partitions:
    source_dir = os.path.join(LOCAL_TMP_PATH, DIR_NAME)
    
    # 통합 파일용 별도 디렉토리 생성 ✅
    merged_dir_name = f"{DIR_NAME}_merged"
    merged_dir = os.path.join(LOCAL_TMP_PATH, merged_dir_name)
    os.makedirs(merged_dir, exist_ok=True)
    
    # 통합 파일을 별도 디렉토리에 직접 생성 ✅
    output_file = os.path.join(merged_dir, output_filename)
    
    # 통합 실행
    merge_partitioned_parquet(...)
    
    # 원본 디렉토리 유지 ✅
    print(f"원본 파티션 디렉토리 유지: {source_dir}")
    
    # 압축 시 merged 디렉토리 사용
    DIR_NAME = merged_dir_name
```

---

## 사용 예제

### 예제 1: 기본 사용

```bash
python hdfs_transfer.py --merge-partitions
```

**결과:**
- 원본 파티션: `LOCAL_TMP_PATH/propensityScoreDF/` (유지)
- 통합 파일: `LOCAL_TMP_PATH/propensityScoreDF_merged/merged.parquet`
- 원격 전송: `merged.parquet`

---

### 예제 2: 로컬 파일 유지

```bash
python hdfs_transfer.py --merge-partitions --skip-cleanup
```

**로컬 파일 상태:**
```
LOCAL_TMP_PATH/
├── propensityScoreDF/           # 원본 파티션 (유지)
│   ├── year=2024/
│   │   └── month=01/*.parquet
│   └── ...
└── propensityScoreDF_merged/    # 통합 파일 (유지)
    └── merged.parquet
```

**장점:**
- 원본과 통합 파일 모두 유지
- 재실행 시 `--skip-download` 사용 가능
- 디버깅 및 검증 용이

---

### 예제 3: 다운로드 건너뛰기 + 파티션 통합

```bash
# 첫 실행: 다운로드 + 로컬 파일 유지
python hdfs_transfer.py --skip-cleanup

# 두 번째 실행: 파티션 통합 (다운로드 건너뛰기)
python hdfs_transfer.py --skip-download --merge-partitions
```

**동작:**
1. 첫 실행: 원본 파티션 다운로드 및 유지
2. 두 번째 실행: 기존 파티션을 통합하여 전송
3. 원본 파티션은 여전히 유지됨

---

## 파일 구조 비교

### v2.5 (Before)

**처리 중:**
```
LOCAL_TMP_PATH/
├── propensityScoreDF/      # 원본
└── merged.parquet           # 임시
```

**처리 후:**
```
LOCAL_TMP_PATH/
└── propensityScoreDF/      # 통합 파일로 대체 (원본 손실!)
    └── merged.parquet
```

---

### v2.6 (After)

**처리 중:**
```
LOCAL_TMP_PATH/
├── propensityScoreDF/          # 원본 (유지)
└── propensityScoreDF_merged/   # 통합 (생성 중)
```

**처리 후 (--skip-cleanup 미사용):**
```
LOCAL_TMP_PATH/
└── propensityScoreDF/          # 원본만 유지
```

**처리 후 (--skip-cleanup 사용):**
```
LOCAL_TMP_PATH/
├── propensityScoreDF/          # 원본 유지
└── propensityScoreDF_merged/   # 통합 파일 유지
    └── merged.parquet
```

---

## 원격 서버 파일 구조

### --merge-partitions 미사용

```
REMOTE_PATH/
├── year=2024/
│   └── month=01/*.parquet
└── ...
```

### --merge-partitions 사용

```
REMOTE_PATH/
└── merged.parquet
```

**참고:** 원격 서버에는 통합 파일만 전송되며, 디렉토리 구조 없음

---

## 장점

### 1. 데이터 보존
- 원본 파티션 디렉토리 유지
- 통합 실패 시 원본 데이터로 복구 가능

### 2. 명확한 구분
- 원본: `{DIR_NAME}/`
- 통합: `{DIR_NAME}_merged/`
- 혼란 없음

### 3. 효율성
- 불필요한 파일 이동 제거
- 직접 원하는 위치에 생성

### 4. 디버깅
- 원본과 통합 파일 비교 가능
- 문제 발생 시 원인 분석 용이

---

## 주의사항

### 1. 디스크 공간

`--merge-partitions` 사용 시 필요한 공간:

```
원본 파티션: 16GB
통합 파일: 16GB
압축 파일: ~4GB (압축률에 따라)
──────────────────
총 필요 공간: ~36GB
```

**--skip-cleanup 사용 시:**
- 원본 + 통합 + 압축 파일 모두 유지
- 디스크 공간 충분히 확보 필요

**--skip-cleanup 미사용 시:**
- 처리 완료 후 자동 정리
- 원본 파티션만 유지

---

### 2. 파티션 재사용

원본 파티션 디렉토리가 유지되므로 재사용 가능:

```bash
# 첫 실행: 다운로드 + 통합
python hdfs_transfer.py --merge-partitions --skip-cleanup

# 재실행: 다운로드 건너뛰기 + 통합
python hdfs_transfer.py --skip-download --merge-partitions

# 또는: 파티션 구조로 전송
python hdfs_transfer.py --skip-download
```

---

## 업그레이드 가이드

### v2.5에서 v2.6으로

**변경 사항:**
- 통합 파일 디렉토리명 변경: `{DIR_NAME}` → `{DIR_NAME}_merged`
- 원본 파티션 디렉토리 유지

**영향:**
- 기존 스크립트: 수정 불필요
- 로컬 파일: `{DIR_NAME}_merged` 디렉토리 추가 생성
- 원격 서버: 변경 없음

**권장 조치:**
- v2.6으로 업그레이드
- 디스크 공간 확인 (원본 + 통합 파일)
- 필요 시 `--skip-cleanup` 사용

---

## 테스트

### 테스트 케이스 1: 기본 통합

```bash
python hdfs_transfer.py --merge-partitions
```

**확인 사항:**
- [ ] 원본 파티션 디렉토리 유지
- [ ] 통합 파일이 `{DIR_NAME}_merged/`에 생성
- [ ] 원격 서버에 통합 파일만 전송
- [ ] 로컬 파일 자동 정리 (--skip-cleanup 미사용)

---

### 테스트 케이스 2: 로컬 파일 유지

```bash
python hdfs_transfer.py --merge-partitions --skip-cleanup
```

**확인 사항:**
- [ ] 원본 파티션 디렉토리 유지
- [ ] 통합 파일 디렉토리 유지
- [ ] 압축 파일 유지

---

### 테스트 케이스 3: 재실행

```bash
# 첫 실행
python hdfs_transfer.py --skip-cleanup

# 재실행
python hdfs_transfer.py --skip-download --merge-partitions
```

**확인 사항:**
- [ ] 원본 파티션 재사용
- [ ] 통합 파일 새로 생성
- [ ] 두 디렉토리 모두 존재

---

## 파일 목록

### 수정된 파일

1. **hdfs_transfer.py**
   - 파티션 통합 로직 개선
   - 원본 디렉토리 삭제 제거
   - 별도 디렉토리에 통합 파일 생성

2. **CHANGELOG_v2.6.md** (신규)
   - v2.6 변경 사항 문서

---

## 요약

### 변경 내용
✅ 원본 파티션 디렉토리 유지  
✅ 통합 파일을 별도 디렉토리에 직접 생성  
✅ 불필요한 파일 이동 제거  
✅ 데이터 보존 및 재사용 가능

### 핵심 개선
- **Before:** 원본 파티션 삭제 후 통합 파일로 대체
- **After:** 원본 파티션 유지, 통합 파일은 별도 디렉토리

### 사용 예
```bash
# 통합 + 원본 유지
python hdfs_transfer.py --merge-partitions --skip-cleanup

# 원본 재사용
python hdfs_transfer.py --skip-download --merge-partitions
```

---

**이전 버전**: v2.5 (--skip-cleanup 일관성 개선)  
**현재 버전**: v2.6 (원본 파티션 유지)  
**다음 업데이트**: TBD
