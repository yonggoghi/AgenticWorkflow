# 변경 사항 (v2.5)

**날짜**: 2026-01-26  
**변경자**: AI Assistant

---

## 주요 변경 사항

### --skip-cleanup 옵션의 일관성 개선 ✅

**문제점:**
- v2.4까지는 `--skip-cleanup`이 전송 후 정리 단계에만 적용됨
- 다운로드 전 `prepare_local_directory()`는 `--skip-cleanup`을 무시하고 항상 기존 파일 삭제
- 결과: `--skip-cleanup`으로 유지한 파일이 다음 실행에서 삭제되는 모순

**해결책:**
- `prepare_local_directory()` 함수가 `skip_cleanup` 인자를 받도록 수정
- `--skip-cleanup` 사용 시 전체 프로세스에서 로컬 파일 유지
- 옵션의 동작이 일관되고 예측 가능해짐

---

## 상세 변경 내역

### 1. 옵션 충돌 문제

**Before (v2.4):**
```bash
# 첫 실행: 로컬 파일 유지
python hdfs_transfer.py --skip-cleanup
# 결과: LOCAL_TMP_PATH에 파일 유지됨 ✅

# 두 번째 실행: --skip-cleanup 사용
python hdfs_transfer.py --skip-cleanup
# 동작:
#   1. prepare_local_directory() 실행
#   2. 기존 파일 삭제 ❌ (모순!)
#   3. HDFS에서 다시 다운로드
#   4. 전송 후 정리 건너뛰기 (--skip-cleanup 적용)
# 결과: 이전에 유지한 파일이 삭제됨 ❌
```

**After (v2.5):**
```bash
# 첫 실행: 로컬 파일 유지
python hdfs_transfer.py --skip-cleanup
# 결과: LOCAL_TMP_PATH에 파일 유지됨 ✅

# 두 번째 실행: --skip-cleanup 사용
python hdfs_transfer.py --skip-cleanup
# 동작:
#   1. prepare_local_directory(skip_cleanup=True) 실행
#   2. 기존 파일 유지 ✅ (일관성!)
#   3. HDFS에서 다시 다운로드 (기존 파일 덮어씀)
#   4. 전송 후 정리 건너뛰기
# 결과: 파일 유지 정책이 일관되게 적용됨 ✅
```

---

### 2. 올바른 사용 패턴

#### 패턴 A: 파일 재사용 (권장)

**목적:** 다운로드 시간 절약

```bash
# 첫 실행: 다운로드 + 로컬 파일 유지
python hdfs_transfer.py --skip-cleanup

# 두 번째 실행: 다운로드 건너뛰기 + 로컬 파일 재사용
python hdfs_transfer.py --skip-download --skip-cleanup
```

**효과:**
- 첫 실행: HDFS 다운로드 (10분)
- 두 번째 실행: 다운로드 없음 (2분)
- 시간 절약: 80%

---

#### 패턴 B: 항상 새로 다운로드 + 파일 유지

**목적:** 항상 최신 데이터 사용하되 로컬 파일 유지

**Before (v2.4) - 문제 있음:**
```bash
python hdfs_transfer.py --skip-cleanup
# 문제: 이전 파일이 매번 삭제됨
```

**After (v2.5) - 정상 동작:**
```bash
python hdfs_transfer.py --skip-cleanup
# 결과:
#   - 다운로드 시 기존 파일 유지 (덮어씀)
#   - 전송 후 로컬 파일 유지
```

---

#### 패턴 C: 매번 깔끔하게 재시작 (기본)

**목적:** 디스크 공간 관리

```bash
python hdfs_transfer.py
# 결과:
#   - 다운로드 전 기존 파일 삭제
#   - 전송 후 로컬 파일 삭제
```

---

## 코드 변경

### prepare_local_directory() 함수

**Before (v2.4):**
```python
def prepare_local_directory(local_tmp_path, dir_name, archive_name):
    # 항상 기존 파일 삭제
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    if os.path.exists(tar_file):
        os.remove(tar_file)
```

**After (v2.5):**
```python
def prepare_local_directory(local_tmp_path, dir_name, archive_name, skip_cleanup=False):
    if skip_cleanup:
        # 기존 파일 유지
        print("모드: 기존 파일 유지 (--skip-cleanup)")
        # 삭제하지 않음
    else:
        # 기존 파일 삭제
        print("모드: 기존 파일 삭제 (기본)")
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        if os.path.exists(tar_file):
            os.remove(tar_file)
```

### 함수 호출

```python
# skip_cleanup 인자 전달
prepare_local_directory(LOCAL_TMP_PATH, DIR_NAME, ARCHIVE_NAME, 
                       skip_cleanup=args.skip_cleanup)
```

---

## 비교표

### 옵션별 동작 비교

| 단계 | 옵션 없음 | --skip-cleanup | --skip-download | 둘 다 사용 |
|------|----------|---------------|----------------|-----------|
| **다운로드 전** | 파일 삭제 | 파일 유지 ⭐ | 건너뜀 | 건너뜀 |
| **다운로드** | 실행 | 실행 | 건너뜀 | 건너뜀 |
| **전송 후** | 파일 삭제 | 파일 유지 | 파일 삭제 | 파일 유지 |

**v2.5 개선:** "--skip-cleanup"이 "다운로드 전" 단계에도 적용됨

---

## 시나리오별 테스트

### 시나리오 1: 연속 실행 (v2.4 vs v2.5)

**명령:**
```bash
# 첫 실행
python hdfs_transfer.py --skip-cleanup

# 두 번째 실행 (즉시)
python hdfs_transfer.py --skip-cleanup
```

**v2.4 동작:**
```
첫 실행:
1. prepare_local_directory() → 기존 파일 없음
2. HDFS 다운로드 → /tmp/data/ 생성
3. 전송
4. cleanup 건너뛰기 → /tmp/data/ 유지 ✅

두 번째 실행:
1. prepare_local_directory() → /tmp/data/ 삭제 ❌
2. HDFS 다운로드 → /tmp/data/ 재생성
3. 전송
4. cleanup 건너뛰기 → /tmp/data/ 유지

문제: --skip-cleanup의 의도와 맞지 않음!
```

**v2.5 동작:**
```
첫 실행:
1. prepare_local_directory(skip_cleanup=True) → 기존 파일 없음
2. HDFS 다운로드 → /tmp/data/ 생성
3. 전송
4. cleanup 건너뛰기 → /tmp/data/ 유지 ✅

두 번째 실행:
1. prepare_local_directory(skip_cleanup=True) → /tmp/data/ 유지 ✅
2. HDFS 다운로드 → 기존 파일 덮어씀
3. 전송
4. cleanup 건너뛰기 → /tmp/data/ 유지

결과: 일관성 있게 파일 유지 ✅
```

---

### 시나리오 2: 파일 재사용

**명령:**
```bash
# 첫 실행
python hdfs_transfer.py --skip-cleanup

# 두 번째 실행 (다운로드 건너뛰기)
python hdfs_transfer.py --skip-download --skip-cleanup
```

**v2.4 & v2.5 동작:** (동일)
```
첫 실행:
- 다운로드 → 전송 → 파일 유지

두 번째 실행:
- 다운로드 건너뜀 (--skip-download)
- 기존 파일 재사용
- 전송 → 파일 유지

결과: 시간 절약 ✅
```

---

## 장점

### 1. 일관성
- `--skip-cleanup` 옵션이 전체 프로세스에 일관되게 적용됨
- "cleanup을 skip한다" = "로컬 파일을 유지한다"의 의미가 명확

### 2. 예측 가능성
- 사용자가 기대하는 대로 동작
- `--skip-cleanup` 사용 시 항상 로컬 파일 유지

### 3. 효율성
- 불필요한 삭제/재생성 방지
- 디스크 I/O 감소

---

## 주의사항

### 1. 파일 덮어쓰기

`--skip-cleanup` 사용 시 기존 파일이 유지되지만, 다운로드 시 **덮어씌워집니다**:

```bash
# 첫 실행
python hdfs_transfer.py --skip-cleanup
# /tmp/data/file.parquet (버전 1)

# 두 번째 실행 (HDFS 데이터가 변경된 경우)
python hdfs_transfer.py --skip-cleanup
# /tmp/data/file.parquet (버전 2로 덮어씌워짐)
```

**백업이 필요하면:**
```bash
# 백업 후 실행
cp -r /tmp/data /tmp/data.backup
python hdfs_transfer.py --skip-cleanup
```

---

### 2. 디스크 공간

`--skip-cleanup`을 지속적으로 사용하면 디스크 공간이 부족할 수 있음:

```bash
# 주기적으로 수동 정리
rm -rf /tmp/data
```

---

## 마이그레이션 가이드

### v2.4에서 v2.5로

**변경 사항:**
- `prepare_local_directory()` 함수 시그니처 변경
- `--skip-cleanup` 동작 개선

**영향:**
- 기존 사용자: 더 직관적으로 동작
- 스크립트: 수정 불필요

**권장 조치:**
- v2.5로 업그레이드
- 연속 실행 시 `--skip-cleanup` 사용하면 파일 재사용 가능

---

## 사용 권장사항

### 개발/테스트 환경
```bash
# 빠른 반복 개발
python hdfs_transfer.py --skip-download --skip-cleanup --merge-partitions
```

### 프로덕션 환경
```bash
# 매번 깔끔하게
python hdfs_transfer.py --merge-partitions
```

### 디버깅
```bash
# 파일 유지하여 분석
python hdfs_transfer.py --skip-cleanup
```

---

## 파일 목록

### 수정된 파일

1. **hdfs_transfer.py**
   - `prepare_local_directory()`: `skip_cleanup` 인자 추가
   - 함수 호출: `args.skip_cleanup` 전달

2. **CHANGELOG_v2.5.md** (신규)
   - v2.5 변경 사항 문서

---

## 요약

### 변경 내용
✅ `--skip-cleanup` 옵션이 다운로드 전 단계에도 적용  
✅ `prepare_local_directory()`가 `skip_cleanup` 고려  
✅ 로컬 파일 관리의 일관성 개선  

### 핵심 개선
- **Before:** `--skip-cleanup`이 전송 후에만 적용
- **After:** `--skip-cleanup`이 전체 프로세스에 적용

### 권장 사용법

**파일 재사용 (시간 절약):**
```bash
# 첫 실행
python hdfs_transfer.py --skip-cleanup

# 재실행 (다운로드 건너뛰기)
python hdfs_transfer.py --skip-download --skip-cleanup
```

**매번 최신 데이터 + 파일 유지:**
```bash
python hdfs_transfer.py --skip-cleanup
```

---

**이전 버전**: v2.4 (디렉토리 구조 제거)  
**현재 버전**: v2.5 (--skip-cleanup 일관성 개선)  
**다음 업데이트**: TBD
