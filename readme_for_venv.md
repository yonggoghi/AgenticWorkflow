# Linux 오프라인 환경을 위한 Python 가상환경 배포 가이드

macOS에서 개발한 Python 프로젝트를 외부 네트워크 접속이 불가능한 Linux 서버에 배포하기 위한 가이드입니다.

## 개요

macOS와 Linux는 바이너리가 호환되지 않아 가상환경을 직접 복사할 수 없습니다. 대신 macOS에서 Linux용 패키지를 미리 다운로드하여 오프라인 설치를 진행합니다.

## 스크립트 구성

| 파일 | 실행 위치 | 설명 |
|------|----------|------|
| `create_requirements.sh` | macOS | 가상환경에서 requirements.txt 생성 |
| `download_for_linux.sh` | macOS | Linux용 패키지 다운로드 |
| `install_on_linux.sh` | Linux | 오프라인 패키지 설치 |

## 사용 방법

### 1단계: requirements.txt 생성 (macOS)

기존 가상환경의 패키지 목록을 추출합니다.

```bash
cd /Users/yongwook/workspace/AgenticWorkflow

# 현재 프로젝트의 가상환경
./create_requirements.sh ./venv

# 또는 다른 경로의 가상환경
./create_requirements.sh /path/to/other/venv
```

### 2단계: Linux용 패키지 다운로드 (macOS)

requirements.txt를 기반으로 Linux x86_64용 wheel 파일을 다운로드합니다.

```bash
./download_for_linux.sh
```

완료되면 다음 파일이 생성됩니다:
- `linux_packages/` - 다운로드된 wheel 파일들
- `linux_packages.tar.gz` - 압축 파일
- `download.log` - 다운로드 로그

### 3단계: Linux 서버로 파일 전송

다음 파일들을 Linux 서버로 복사합니다.

```bash
scp linux_packages.tar.gz requirements.txt install_on_linux.sh user@server:/path/to/project/
```

**복사할 파일 목록:**
- `linux_packages.tar.gz`
- `requirements.txt`
- `install_on_linux.sh`
- 프로젝트 소스 코드

### 4단계: Linux 서버에서 설치

```bash
cd /path/to/project

# 실행 권한 부여
chmod +x install_on_linux.sh

# 설치 실행
./install_on_linux.sh
```

스크립트가 자동으로 수행하는 작업:
1. `linux_packages.tar.gz` 압축 해제
2. `python3.12 -m venv venv` 가상환경 생성
3. 오프라인 패키지 설치

### 5단계: 가상환경 활성화

```bash
source venv/bin/activate
```

## 요구사항

### macOS
- Python 3.12
- pip

### Linux
- Python 3.12 (`python3.12` 명령으로 실행 가능해야 함)
- venv 모듈

## 문제 해결

### 일부 패키지 다운로드 실패

Linux용 바이너리가 없는 패키지는 다운로드에 실패할 수 있습니다. `download.log`를 확인하고 해당 패키지를 requirements.txt에서 제외하거나 별도로 처리하세요.

```bash
# 다운로드 실패한 패키지 확인
grep -i "error\|failed" download.log
```

### python3.12을 찾을 수 없음

Linux 서버에 Python 3.12가 설치되어 있는지 확인하세요.

```bash
which python3.12
python3.12 --version
```

### 패키지 버전 충돌

특정 패키지 버전이 호환되지 않을 경우 requirements.txt를 수정하세요.

```bash
# 버전 고정 제거 (최신 호환 버전 설치)
sed -i 's/==.*//g' requirements.txt
```

## 전체 워크플로우 요약

```bash
# === macOS에서 ===
cd /Users/yongwook/workspace/AgenticWorkflow

# 1. requirements.txt 생성
./create_requirements.sh ./venv

# 2. Linux용 패키지 다운로드
./download_for_linux.sh

# 3. Linux 서버로 전송
scp linux_packages.tar.gz requirements.txt install_on_linux.sh user@server:/path/to/project/

# === Linux 서버에서 ===
cd /path/to/project

# 4. 설치
chmod +x install_on_linux.sh
./install_on_linux.sh

# 5. 가상환경 활성화
source venv/bin/activate
```
