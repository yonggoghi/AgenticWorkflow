#!/bin/bash

# Linux 서버에서 실행: 오프라인 패키지 설치
# Python 3.12 전용
# 사용법: ./install_on_linux.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQUIREMENTS="$SCRIPT_DIR/requirements.txt"
VENV_DIR="$SCRIPT_DIR/venv"

# Python 3.12 확인
if ! command -v python3.12 &> /dev/null; then
    echo "Error: python3.12을 찾을 수 없습니다!"
    echo "python3.12 설치 후 다시 실행해주세요."
    exit 1
fi

echo "Python 버전: $(python3.12 --version)"

# 배치 압축 파일 확인 및 해제
echo ""
echo "=== 압축 파일 확인 및 해제 ==="
BATCH_FILES=($(ls "$SCRIPT_DIR"/linux_packages_*.tar.gz 2>/dev/null | sort -V))

if [ ${#BATCH_FILES[@]} -eq 0 ]; then
    echo "Error: linux_packages_*.tar.gz 파일을 찾을 수 없습니다!"
    echo "다운로드한 배치 파일들을 먼저 복사해주세요."
    exit 1
fi

echo "발견된 배치 파일: ${#BATCH_FILES[@]}개"
for batch_file in "${BATCH_FILES[@]}"; do
    batch_name=$(basename "$batch_file" .tar.gz)
    batch_dir="$SCRIPT_DIR/$batch_name"
    
    if [ ! -d "$batch_dir" ]; then
        echo "압축 해제 중: $batch_file"
        tar -xzf "$batch_file" -C "$SCRIPT_DIR"
    else
        echo "이미 해제됨: $batch_name"
    fi
done

# 패키지 디렉토리 확인
PACKAGES_DIRS=($(ls -d "$SCRIPT_DIR"/linux_packages_* 2>/dev/null | sort -V))
if [ ${#PACKAGES_DIRS[@]} -eq 0 ]; then
    echo "Error: linux_packages_* 디렉토리가 없습니다!"
    echo "압축 파일 해제에 실패했습니다."
    exit 1
fi

echo ""
echo "사용 가능한 패키지 디렉토리: ${#PACKAGES_DIRS[@]}개"
for dir in "${PACKAGES_DIRS[@]}"; do
    count=$(ls -1 "$dir"/*.whl 2>/dev/null | wc -l | tr -d ' ')
    echo "  - $(basename "$dir"): $count 개 wheel 파일"
done

# requirements.txt 확인
if [ ! -f "$REQUIREMENTS" ]; then
    echo "Error: requirements.txt 파일이 없습니다!"
    exit 1
fi

# 가상환경 생성
if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo "=== 가상환경 생성 (python3.12) ==="
    python3.12 -m venv "$VENV_DIR"
fi

# 가상환경 활성화
echo ""
echo "=== 가상환경 활성화 ==="
source "$VENV_DIR/bin/activate"

# pip 버전 확인
echo "pip 버전: $(pip3.12 --version)"

# pip 업그레이드 시도 (오프라인)
echo ""
echo "=== pip 업그레이드 시도 ==="
# 모든 배치 디렉토리에서 pip 찾기
FIND_LINKS_ARGS=""
for dir in "${PACKAGES_DIRS[@]}"; do
    FIND_LINKS_ARGS="$FIND_LINKS_ARGS --find-links=\"$dir\""
done

eval "pip3.12 install --no-index $FIND_LINKS_ARGS pip --upgrade 2>/dev/null" || echo "pip 업그레이드 스킵 (로컬에 없음)"

# 패키지 설치
echo ""
echo "=== 패키지 설치 시작 ==="
# 모든 배치 디렉토리를 find-links로 추가
INSTALL_CMD="pip3.12 install --no-index"
for dir in "${PACKAGES_DIRS[@]}"; do
    INSTALL_CMD="$INSTALL_CMD --find-links=\"$dir\""
done
INSTALL_CMD="$INSTALL_CMD -r \"$REQUIREMENTS\""

eval "$INSTALL_CMD"

echo ""
echo "=== 설치 완료 ==="
echo ""
echo "설치된 패키지 목록:"
pip3.12 list | head -20
TOTAL=$(pip3.12 list | tail -n +3 | wc -l)
echo "... 총 $TOTAL 개 패키지 설치됨"
echo ""
echo "=========================================="
echo "가상환경 활성화 명령어:"
echo "  source $VENV_DIR/bin/activate"
echo "=========================================="
