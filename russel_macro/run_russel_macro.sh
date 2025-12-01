#!/bin/bash
# 메가스터디 러셀 단과 접수 자동화 매크로 실행 스크립트

# 스크립트 디렉토리로 이동
cd "$(dirname "$0")"

echo "=========================================="
echo "메가스터디 러셀 단과 접수 자동화 매크로"
echo "=========================================="
echo ""

# Python 가상환경 확인
if [ -d "../venv" ]; then
    echo "가상환경 활성화 중..."
    source ../venv/bin/activate
fi

# Playwright 설치 확인
if ! python -c "import playwright" 2>/dev/null; then
    echo "❌ Playwright가 설치되어 있지 않습니다."
    echo ""
    echo "설치 방법:"
    echo "  pip install playwright"
    echo "  playwright install"
    exit 1
fi

# 실행 모드 선택
echo "실행 모드를 선택하세요:"
echo "1. 기본 실행 (브라우저 표시)"
echo "2. 헤드리스 모드 (백그라운드)"
echo "3. 빠른 실행 (헤드리스 + 빠름)"
echo "4. 예제 실행 (대화형)"
echo ""
read -p "선택 (1-4): " mode

case $mode in
    1)
        echo ""
        echo "기본 모드로 실행합니다..."
        python russel_macro.py
        ;;
    2)
        echo ""
        echo "헤드리스 모드로 실행합니다..."
        python russel_macro.py --headless
        ;;
    3)
        echo ""
        echo "빠른 실행 모드로 실행합니다..."
        python russel_macro.py --headless --fast
        ;;
    4)
        echo ""
        echo "예제를 실행합니다..."
        python russel_macro_example.py
        ;;
    *)
        echo "잘못된 선택입니다."
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "실행 완료"
echo "=========================================="

