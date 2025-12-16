#!/bin/bash

# Mac에서 실행: 가상환경에서 requirements.txt 생성
# 사용법: ./create_requirements.sh /path/to/venv

set -e

# 인자 확인
if [ -z "$1" ]; then
    echo "사용법: $0 <가상환경 경로>"
    echo ""
    echo "예시:"
    echo "  $0 /Users/yongwook/workspace/AgenticWorkflow/venv"
    echo "  $0 ./venv"
    exit 1
fi

VENV_PATH="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_FILE="$SCRIPT_DIR/requirements.txt"

# 가상환경 경로를 절대경로로 변환
if [[ "$VENV_PATH" != /* ]]; then
    VENV_PATH="$(cd "$VENV_PATH" 2>/dev/null && pwd)" || {
        echo "Error: 경로를 찾을 수 없습니다: $1"
        exit 1
    }
fi

# 가상환경 확인
if [ ! -d "$VENV_PATH" ]; then
    echo "Error: 가상환경 디렉토리가 없습니다: $VENV_PATH"
    exit 1
fi

# pip 경로 확인
PIP_PATH="$VENV_PATH/bin/pip"
if [ ! -f "$PIP_PATH" ]; then
    echo "Error: pip을 찾을 수 없습니다: $PIP_PATH"
    echo "올바른 가상환경 경로인지 확인해주세요."
    exit 1
fi

# Python 버전 확인
PYTHON_PATH="$VENV_PATH/bin/python"
PYTHON_VERSION=$("$PYTHON_PATH" --version 2>&1)

echo "=== 가상환경 정보 ==="
echo "경로: $VENV_PATH"
echo "Python: $PYTHON_VERSION"
echo ""

# requirements.txt 생성
echo "=== requirements.txt 생성 중 ==="
"$PIP_PATH" freeze > "$OUTPUT_FILE"

# 패키지 개수 확인
PACKAGE_COUNT=$(wc -l < "$OUTPUT_FILE" | tr -d ' ')

echo ""
echo "=== 완료 ==="
echo "생성된 파일: $OUTPUT_FILE"
echo "패키지 수: $PACKAGE_COUNT 개"
echo ""
echo "--- 패키지 목록 (처음 20개) ---"
head -20 "$OUTPUT_FILE"
if [ "$PACKAGE_COUNT" -gt 20 ]; then
    echo "... (총 $PACKAGE_COUNT 개)"
fi
