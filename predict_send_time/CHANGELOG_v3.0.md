# 변경 사항 (v3.0)

**날짜**: 2026-01-26  
**변경자**: AI Assistant

---

## 주요 변경 사항

### 기본 동작 변경: 로컬 파일 유지 ✅

**문제점:**
- v2.9까지는 기본적으로 전송 후 로컬 파일 삭제
- 파일 재사용이나 디버깅을 위해 매번 `--skip-cleanup` 옵션 필요
- 대부분의 사용 사례에서 로컬 파일 유지가 더 유용

**해결책:**
- 기본 동작을 로컬 파일 유지로 변경
- `--skip-cleanup` → `--cleanup` 옵션으로 변경
- 로컬 파일 삭제가 필요한 경우에만 `--cleanup` 사용

---

## 상세 변경 내역

### Before (v2.9) - 기본은 삭제

```bash
# 기본 동작: 전송 후 로컬 파일 삭제
python hdfs_transfer.py

# 로컬 파일 유지하려면 옵션 필요
python hdfs_transfer.py --skip-cleanup
```

**문제점:**
- 파일 재사용 시 매번 옵션 필요
- 디버깅 시 불편
- 대부분의 경우 로컬 파일 유지가 필요

---

### After (v3.0) - 기본은 유지

```bash
# 기본 동작: 전송 후 로컬 파일 유지 (변경) ✅
python hdfs_transfer.py

# 로컬 파일 삭제하려면 옵션 사용
python hdfs_transfer.py --cleanup
```

**장점:**
- 기본적으로 로컬 파일 유지
- 파일 재사용 용이
- 디버깅 편리
- 옵션 사용 빈도 감소

---

## 옵션 변경

### 옵션명

**Before (v2.9):**
```bash
--skip-cleanup    # 로컬 파일 유지 (옵션)
```

**After (v3.0):**
```bash
--cleanup         # 로컬 파일 삭제 (옵션) ✅
```

---

### 동작 비교

| 버전 | 옵션 없음 | 옵션 사용 |
|------|----------|----------|
| v2.9 | 파일 삭제 | `--skip-cleanup` → 파일 유지 |
| v3.0 | 파일 유지 ✅ | `--cleanup` → 파일 삭제 |

---

## 사용 예제

### 예제 1: 기본 사용 (v3.0+)

```bash
python hdfs_transfer.py
```

**동작:**
1. HDFS 다운로드
2. 압축 (tar.gz)
3. 전송
4. **로컬 파일 유지** ✅ (기본)

**로컬 파일:**
```
LOCAL_TMP_PATH/
├── propensityScoreDF/
└── propensityScoreDF.tar.gz
```

---

### 예제 2: 로컬 파일 삭제

```bash
python hdfs_transfer.py --cleanup
```

**동작:**
1. HDFS 다운로드
2. 압축 (tar.gz)
3. 전송
4. **로컬 파일 삭제** (--cleanup)

**로컬 파일:**
```
LOCAL_TMP_PATH/
(모두 삭제됨)
```

---

### 예제 3: 파일 재사용 (더 간편해짐)

**v2.9:**
```bash
# 첫 실행
python hdfs_transfer.py --skip-cleanup

# 재실행 (다운로드 건너뛰기)
python hdfs_transfer.py --skip-download --skip-cleanup
```

**v3.0:**
```bash
# 첫 실행 (자동으로 파일 유지)
python hdfs_transfer.py

# 재실행 (다운로드 건너뛰기)
python hdfs_transfer.py --skip-download
```

**장점:** 옵션 사용 간소화

---

### 예제 4: 파티션 통합

**v2.9:**
```bash
python hdfs_transfer.py --merge-partitions --skip-cleanup
```

**v3.0:**
```bash
python hdfs_transfer.py --merge-partitions
# 자동으로 로컬 파일 유지됨 ✅
```

---

## 코드 변경

### 1. 옵션 정의

**Before (v2.9):**
```python
parser.add_argument(
    '--skip-cleanup',
    action='store_true',
    help='로컬 임시 파일 삭제 단계를 건너뜁니다 (다운로드된 파일과 압축 파일을 유지)'
)
```

**After (v3.0):**
```python
parser.add_argument(
    '--cleanup',
    action='store_true',
    help='로컬 임시 파일을 삭제합니다 (기본: 파일 유지)'
)
```

---

### 2. prepare_local_directory() 함수

**Before (v2.9):**
```python
def prepare_local_directory(local_tmp_path, dir_name, archive_name, skip_cleanup=False):
    if skip_cleanup:
        # 파일 유지
    else:
        # 파일 삭제 (기본)
```

**After (v3.0):**
```python
def prepare_local_directory(local_tmp_path, dir_name, archive_name, cleanup=False):
    if cleanup:
        # 파일 삭제 (옵션)
    else:
        # 파일 유지 (기본) ✅
```

---

### 3. cleanup 로직

**Before (v2.9):**
```python
if not args.skip_cleanup:
    cleanup(...)  # 기본: 삭제
else:
    print("파일 유지")  # 옵션: 유지
```

**After (v3.0):**
```python
if args.cleanup:
    cleanup(...)  # 옵션: 삭제
else:
    print("파일 유지 (기본)")  # 기본: 유지 ✅
```

---

## 동작 흐름 비교

### v2.9

```
1. HDFS 다운로드
2. 압축 생성
3. 원격 전송
4. 기본: 로컬 파일 삭제 ❌
   --skip-cleanup: 로컬 파일 유지
```

---

### v3.0

```
1. HDFS 다운로드
2. 압축 생성
3. 원격 전송
4. 기본: 로컬 파일 유지 ✅
   --cleanup: 로컬 파일 삭제
```

---

## 장점

### 1. 직관성
- ✅ 기본 동작이 더 안전 (파일 유지)
- ✅ 데이터 손실 위험 감소
- ✅ 재사용 시 편리

### 2. 개발 효율성
- ✅ 디버깅 용이
- ✅ 재실행 시 다운로드 건너뛰기 간편
- ✅ 옵션 사용 빈도 감소

### 3. 디스크 관리
- ✅ 필요한 경우에만 `--cleanup` 사용
- ✅ 자동 정리를 원하는 환경에서만 옵션 추가

---

## 사용 패턴

### 패턴 1: 개발/테스트 환경 (기본)

```bash
# 기본 사용 (파일 자동 유지)
python hdfs_transfer.py

# 재실행 (다운로드 건너뛰기)
python hdfs_transfer.py --skip-download
```

**장점:** 빠른 반복 개발

---

### 패턴 2: 프로덕션 환경 (cleanup)

```bash
# 자동 정리 필요 시
python hdfs_transfer.py --cleanup
```

**장점:** 디스크 공간 자동 관리

---

### 패턴 3: 배치 작업

```bash
# 크론잡 등에서 사용
0 2 * * * /path/to/python hdfs_transfer.py --cleanup
```

**장점:** 매일 자동 실행, 디스크 정리

---

## 마이그레이션 가이드

### v2.9에서 v3.0으로

#### 영향 받는 스크립트

**v2.9 스크립트:**
```bash
# 로컬 파일 유지 (기존)
python hdfs_transfer.py --skip-cleanup
```

**v3.0 마이그레이션:**
```bash
# --skip-cleanup 제거 (기본 동작이 유지로 변경됨)
python hdfs_transfer.py
```

---

**v2.9 스크립트:**
```bash
# 기본 동작 (삭제)
python hdfs_transfer.py
```

**v3.0 마이그레이션:**
```bash
# --cleanup 추가 (삭제 원하면)
python hdfs_transfer.py --cleanup
```

---

### 마이그레이션 체크리스트

1. **`--skip-cleanup` 사용하는 스크립트:**
   - [ ] `--skip-cleanup` 제거
   - [ ] 기본 동작으로 변경

2. **기본 동작(삭제) 의존하는 스크립트:**
   - [ ] `--cleanup` 추가
   - [ ] 디스크 정리 동작 유지

3. **크론잡/자동화 스크립트:**
   - [ ] 디스크 공간 정책 확인
   - [ ] 필요 시 `--cleanup` 추가

---

## 호환성

### 영향 없는 경우

✅ 옵션을 사용하지 않는 스크립트 (동작만 변경)
✅ `--merge-partitions` 등 다른 옵션만 사용

---

### 영향 있는 경우

⚠️ `--skip-cleanup` 사용하는 스크립트
⚠️ 기본 삭제 동작에 의존하는 스크립트
⚠️ 디스크 공간 관리 자동화 스크립트

---

## 테스트

### 테스트 케이스 1: 기본 동작 (파일 유지)

```bash
python hdfs_transfer.py
```

**확인 사항:**
- [ ] 전송 완료
- [ ] 로컬 파일 유지됨 ✅
- [ ] 디렉토리 존재
- [ ] tar.gz 존재

---

### 테스트 케이스 2: 파일 삭제

```bash
python hdfs_transfer.py --cleanup
```

**확인 사항:**
- [ ] 전송 완료
- [ ] 로컬 파일 삭제됨 ✅
- [ ] 디렉토리 삭제
- [ ] tar.gz 삭제

---

### 테스트 케이스 3: 파일 재사용

```bash
# 첫 실행
python hdfs_transfer.py

# 재실행
python hdfs_transfer.py --skip-download
```

**확인 사항:**
- [ ] 첫 실행 파일 유지
- [ ] 재실행 시 다운로드 건너뜀
- [ ] 전송 성공

---

## 권장 사용법

### 개발 환경

```bash
# 기본 사용 (파일 자동 유지)
python hdfs_transfer.py
```

---

### 프로덕션 환경 (디스크 관리 필요)

```bash
# 자동 정리
python hdfs_transfer.py --cleanup
```

---

### CI/CD 파이프라인

```bash
# 임시 빌드 서버
python hdfs_transfer.py --cleanup
```

---

## 요약

### 변경 내용
✅ 기본 동작: 로컬 파일 유지 (변경)  
✅ 옵션명: `--skip-cleanup` → `--cleanup`  
✅ 로직 반전: 유지가 기본, 삭제가 옵션  
✅ 사용 편의성 향상

### 핵심 개선
- **Before (v2.9):** 기본은 삭제, `--skip-cleanup`으로 유지
- **After (v3.0):** 기본은 유지, `--cleanup`으로 삭제 ✅

### 마이그레이션
**v2.9:**
```bash
python hdfs_transfer.py --skip-cleanup
```

**v3.0:**
```bash
python hdfs_transfer.py  # --skip-cleanup 제거
```

### 사용 예
```bash
# 기본 (파일 유지)
python hdfs_transfer.py

# 파일 삭제 필요 시
python hdfs_transfer.py --cleanup

# 파일 재사용
python hdfs_transfer.py --skip-download
```

---

**이전 버전**: v2.9 (디렉토리 구조 유지)  
**현재 버전**: v3.0 (로컬 파일 유지 기본화)  
**다음 업데이트**: TBD
