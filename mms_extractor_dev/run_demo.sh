#!/bin/bash

# MMS Extractor API Demo 실행 스크립트
# ====================================

echo "🚀 MMS Extractor API Demo 실행 스크립트"
echo "====================================="

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 현재 디렉토리 확인
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${BLUE}📁 작업 디렉토리: $SCRIPT_DIR${NC}"
echo ""

# Python 가상환경 활성화 (있는 경우)
if [ -d "venv" ]; then
    echo -e "${YELLOW}🐍 Python 가상환경 활성화 중...${NC}"
    source venv/bin/activate
elif [ -d "../venv" ]; then
    echo -e "${YELLOW}🐍 Python 가상환경 활성화 중...${NC}"
    source ../venv/bin/activate
fi

# API 서버가 실행 중인지 확인
check_api_server() {
    local port=${1:-8000}
    if curl -s "http://localhost:$port/health" > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

echo -e "${BLUE}🔍 API 서버 상태 확인 중...${NC}"
if check_api_server 8000; then
    echo -e "${GREEN}✅ MMS 추출기 API 서버가 포트 8000에서 실행 중입니다.${NC}"
else
    echo -e "${YELLOW}⚠️  MMS 추출기 API 서버가 실행되지 않았습니다.${NC}"
    echo -e "${YELLOW}   다음 명령으로 API 서버를 먼저 시작하세요:${NC}"
    echo -e "${YELLOW}   python api.py --host 0.0.0.0 --port 8000${NC}"
    echo ""
    
    read -p "그래도 데모 서버를 시작하시겠습니까? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${RED}❌ 데모 서버 시작을 취소했습니다.${NC}"
        exit 1
    fi
fi

echo ""
echo -e "${BLUE}🌐 데모 웹 서버 시작 중...${NC}"

# 데모 서버 실행
DEMO_PORT=${DEMO_PORT:-8080}
DEMO_HOST=${DEMO_HOST:-0.0.0.0}

echo -e "${GREEN}🎯 데모 서버 정보:${NC}"
echo -e "   📱 웹 인터페이스: http://localhost:$DEMO_PORT"
echo -e "   🖼️  DAG 이미지: http://localhost:$DEMO_PORT/dag_images/"
echo -e "   📊 이미지 API: http://localhost:$DEMO_PORT/api/dag-images"
echo ""
echo -e "${YELLOW}💡 사용 팁:${NC}"
echo -e "   • 브라우저에서 http://localhost:$DEMO_PORT 접속"
echo -e "   • 샘플 메시지를 클릭하여 빠른 테스트"
echo -e "   • DAG 추출 옵션으로 관계 그래프 생성"
echo -e "   • Ctrl+C로 서버 종료"
echo ""
echo -e "${GREEN}🚀 서버 시작...${NC}"

# Python 스크립트 실행
python demo_server.py --host "$DEMO_HOST" --port "$DEMO_PORT" "$@"
