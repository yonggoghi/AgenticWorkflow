#!/bin/bash

# Mac에서 실행: Linux용 패키지 일괄 다운로드
# 사용법: ./download_for_linux.sh [가상환경 경로]
# 가상환경 경로를 지정하지 않으면 현재 디렉토리의 venv 사용

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQUIREMENTS="$SCRIPT_DIR/requirements.txt"
BATCH_SIZE=100

# 가상환경 경로 설정
if [ -n "$1" ]; then
    VENV_PATH="$1"
else
    VENV_PATH="$SCRIPT_DIR/venv"
fi

# 상대경로를 절대경로로 변환
if [[ "$VENV_PATH" != /* ]]; then
    VENV_PATH="$(cd "$VENV_PATH" 2>/dev/null && pwd)" || {
        echo "Error: 가상환경 경로를 찾을 수 없습니다: $1"
        exit 1
    }
fi

# pip 경로 확인
PIP_PATH="$VENV_PATH/bin/pip"
if [ ! -f "$PIP_PATH" ]; then
    echo "Error: pip을 찾을 수 없습니다: $PIP_PATH"
    echo "올바른 가상환경 경로인지 확인해주세요."
    echo ""
    echo "사용법: $0 [가상환경 경로]"
    echo "예시: $0 ./venv"
    exit 1
fi

# requirements.txt 확인
if [ ! -f "$REQUIREMENTS" ]; then
    echo "Error: requirements.txt 파일이 없습니다!"
    echo "먼저 create_requirements.sh를 실행해주세요."
    exit 1
fi

# 기존 패키지 디렉토리 정리
rm -rf "$SCRIPT_DIR"/linux_packages_*

echo "=== Linux용 패키지 다운로드 시작 ==="
echo "가상환경: $VENV_PATH"
echo "pip 경로: $PIP_PATH"
echo "requirements.txt: $REQUIREMENTS"
echo "배치 크기: $BATCH_SIZE 패키지"
echo "대상 Python 버전: 3.12"
echo ""

# requirements.txt를 배열로 읽기 (빈 줄과 주석 제외)
PACKAGES=()
while IFS= read -r package || [ -n "$package" ]; do
    [[ -z "$package" || "$package" =~ ^# ]] && continue
    PACKAGES+=("$package")
done < "$REQUIREMENTS"

TOTAL_PACKAGES=${#PACKAGES[@]}
TOTAL_BATCHES=$(( (TOTAL_PACKAGES + BATCH_SIZE - 1) / BATCH_SIZE ))

echo "총 패키지 수: $TOTAL_PACKAGES"
echo "배치 수: $TOTAL_BATCHES"
echo ""

# 전체 실패한 패키지 기록
ALL_FAILED_PACKAGES=""

# 배치별로 처리
for ((batch=1; batch<=TOTAL_BATCHES; batch++)); do
    PACKAGES_DIR="$SCRIPT_DIR/linux_packages_$batch"
    mkdir -p "$PACKAGES_DIR"
    
    START_IDX=$(( (batch - 1) * BATCH_SIZE ))
    END_IDX=$(( batch * BATCH_SIZE ))
    if [ $END_IDX -gt $TOTAL_PACKAGES ]; then
        END_IDX=$TOTAL_PACKAGES
    fi
    
    echo "=== 배치 $batch/$TOTAL_BATCHES 처리 중 (패키지 $((START_IDX + 1))-$END_IDX) ==="
    
    # 실패한 패키지 기록 (배치별)
    FAILED_PACKAGES=""
    
    # 배치 내 패키지를 하나씩 다운로드
    for ((i=START_IDX; i<END_IDX; i++)); do
        package="${PACKAGES[$i]}"
        echo "[$((i + 1))/$TOTAL_PACKAGES] 다운로드 중: $package"
        
        # Linux용 바이너리 다운로드 시도
        "$PIP_PATH" download "$package" \
            --dest "$PACKAGES_DIR" \
            --platform manylinux2014_x86_64 \
            --platform manylinux_2_17_x86_64 \
            --platform manylinux_2_27_x86_64 \
            --platform linux_x86_64 \
            --python-version 312 \
            --implementation cp \
            --abi cp312 \
            --only-binary=:all: \
            --no-deps \
            2>/dev/null
        
        if [ $? -ne 0 ]; then
            # pure Python 패키지로 재시도
            "$PIP_PATH" download "$package" \
                --dest "$PACKAGES_DIR" \
                --platform any \
                --python-version 312 \
                --only-binary=:all: \
                --no-deps \
                2>/dev/null
            
            if [ $? -ne 0 ]; then
                echo "  ⚠ 실패: $package"
                FAILED_PACKAGES="$FAILED_PACKAGES$package\n"
                ALL_FAILED_PACKAGES="$ALL_FAILED_PACKAGES$package\n"
            fi
        fi
    done
    
    # 배치별 의존성 보완 다운로드
    echo ""
    echo "=== 배치 $batch 의존성 보완 다운로드 ==="
    
    # 현재 배치의 패키지 목록을 임시 파일로 생성
    TEMP_REQ="$SCRIPT_DIR/temp_requirements_$batch.txt"
    for ((i=START_IDX; i<END_IDX; i++)); do
        echo "${PACKAGES[$i]}" >> "$TEMP_REQ"
    done
    
    "$PIP_PATH" download \
        -r "$TEMP_REQ" \
        --dest "$PACKAGES_DIR" \
        --platform manylinux2014_x86_64 \
        --platform manylinux_2_17_x86_64 \
        --platform linux_x86_64 \
        --platform any \
        --python-version 312 \
        --implementation cp \
        --abi cp312 \
        --only-binary=:all: \
        2>&1 | grep -v "ERROR\|error\|Traceback" || true
    
    rm -f "$TEMP_REQ"
    
    # 배치별 다운로드된 파일 개수 확인
    COUNT=$(ls -1 "$PACKAGES_DIR"/*.whl 2>/dev/null | wc -l | tr -d ' ')
    echo ""
    echo "=== 배치 $batch 다운로드 완료 ==="
    echo "총 $COUNT 개의 wheel 파일 다운로드됨"
    
    # 배치별 실패한 패키지 출력
    if [ -n "$FAILED_PACKAGES" ]; then
        echo ""
        echo "=== 배치 $batch 다운로드 실패한 패키지 ==="
        echo -e "$FAILED_PACKAGES"
    fi
    
    # 배치별 tar.gz로 압축
    echo ""
    echo "=== 배치 $batch 압축 중 ==="
    cd "$SCRIPT_DIR"
    tar -czf "linux_packages_$batch.tar.gz" "linux_packages_$batch/"
    echo "생성된 파일: $SCRIPT_DIR/linux_packages_$batch.tar.gz"
    echo ""
done

# 전체 실패한 패키지 출력 및 저장
if [ -n "$ALL_FAILED_PACKAGES" ]; then
    echo ""
    echo "=== 전체 다운로드 실패한 패키지 ==="
    echo -e "$ALL_FAILED_PACKAGES"
    echo "위 패키지들은 Linux용 wheel이 없거나 소스 빌드가 필요합니다."
    echo "Linux 서버에서 별도로 빌드하거나 제외해야 할 수 있습니다."
    
    # 실패 목록 파일로 저장
    echo -e "$ALL_FAILED_PACKAGES" > "$SCRIPT_DIR/failed_packages.txt"
    echo "실패 목록 저장: $SCRIPT_DIR/failed_packages.txt"
fi

echo ""
echo "=== 전체 작업 완료 ==="
echo "생성된 파일:"
for ((batch=1; batch<=TOTAL_BATCHES; batch++)); do
    echo "  - linux_packages_$batch.tar.gz"
done
echo ""
echo "Linux 서버로 복사할 파일:"
echo "  - linux_packages_*.tar.gz (모든 배치 파일)"
echo "  - requirements.txt"
echo "  - install_on_linux.sh"
echo ""
echo "Linux 서버에서 실행:"
echo "  ./install_on_linux.sh"
