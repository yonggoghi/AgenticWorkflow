#!/bin/bash
# DAG 이미지 저장 모드 테스트 스크립트

echo "========================================"
echo "DAG 이미지 저장 모드 테스트"
echo "========================================"
echo ""

# 색상 정의
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 프로젝트 디렉토리 (스크립트 위치 기반)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
cd "$PROJECT_DIR"

# 테스트 1: 로컬 저장 모드
echo "=========================================="
echo "테스트 1: 로컬 저장 모드 (local)"
echo "=========================================="
echo ""

# 로컬 디렉토리 확인
if [ -d "./dag_images_local" ]; then
    echo -e "${GREEN}✅ dag_images_local 디렉토리 존재${NC}"
else
    echo -e "${YELLOW}⚠️  dag_images_local 디렉토리 생성 중...${NC}"
    mkdir -p ./dag_images_local
    echo -e "${GREEN}✅ dag_images_local 디렉토리 생성 완료${NC}"
fi

# 설정 확인
echo ""
echo "로컬 저장 모드 설정 확인:"
export DAG_STORAGE_MODE=local
python -c "from config.settings import STORAGE_CONFIG; print(f'저장 모드: {STORAGE_CONFIG.dag_storage_mode}'); print(f'저장 경로: {STORAGE_CONFIG.get_dag_images_dir()}'); print(f'설명: {STORAGE_CONFIG.get_storage_description()}')" 2>/dev/null || echo "Python 설정 확인 실패"

echo ""
echo -e "${GREEN}✅ 로컬 저장 모드 설정 완료${NC}"
echo ""

# 테스트 2: NAS 저장 모드
echo "=========================================="
echo "테스트 2: NAS 저장 모드 (nas)"
echo "=========================================="
echo ""

# NAS 마운트 확인
if mount | grep -q "nas_dag_images"; then
    echo -e "${GREEN}✅ NAS 마운트됨${NC}"
    mount | grep nas_dag_images
    NAS_AVAILABLE=true
else
    echo -e "${YELLOW}⚠️  NAS가 마운트되지 않음${NC}"
    echo "   NAS 마운트를 하려면: sudo bash scripts/setup_nas_mount.sh"
    NAS_AVAILABLE=false
fi

# dag_images 디렉토리 확인
if [ -L "./dag_images" ]; then
    echo -e "${GREEN}✅ dag_images 심볼릭 링크 존재${NC}"
    echo "   -> $(readlink ./dag_images)"
elif [ -d "./dag_images" ]; then
    echo -e "${YELLOW}⚠️  dag_images는 일반 디렉토리입니다 (심볼릭 링크 아님)${NC}"
else
    echo -e "${YELLOW}⚠️  dag_images 디렉토리/링크 없음${NC}"
fi

# 설정 확인
echo ""
echo "NAS 저장 모드 설정 확인:"
export DAG_STORAGE_MODE=nas
python -c "from config.settings import STORAGE_CONFIG; print(f'저장 모드: {STORAGE_CONFIG.dag_storage_mode}'); print(f'저장 경로: {STORAGE_CONFIG.get_dag_images_dir()}'); print(f'설명: {STORAGE_CONFIG.get_storage_description()}')" 2>/dev/null || echo "Python 설정 확인 실패"

if [ "$NAS_AVAILABLE" = true ]; then
    echo -e "${GREEN}✅ NAS 저장 모드 사용 가능${NC}"
else
    echo -e "${YELLOW}⚠️  NAS 저장 모드 사용 불가 (마운트 필요)${NC}"
fi

echo ""

# 요약
echo "=========================================="
echo "테스트 요약"
echo "=========================================="
echo ""
echo "사용 가능한 저장 모드:"
echo ""
echo -e "${GREEN}✅ 로컬 저장 (local)${NC}"
echo "   명령어: python api.py --storage local"
echo "   저장 위치: ./dag_images_local/"
echo ""

if [ "$NAS_AVAILABLE" = true ]; then
    echo -e "${GREEN}✅ NAS 저장 (nas)${NC}"
    echo "   명령어: python api.py --storage nas"
    echo "   저장 위치: ./dag_images/ -> /mnt/nas_dag_images/dag_images/"
else
    echo -e "${YELLOW}⚠️  NAS 저장 (nas) - 마운트 필요${NC}"
    echo "   설정 방법:"
    echo "   1. sudo bash scripts/setup_nas_mount.sh"
    echo "   2. bash scripts/setup_symlink.sh"
    echo "   3. python api.py --storage nas"
fi

echo ""
echo "=========================================="
echo "다음 단계:"
echo "=========================================="
echo ""
echo "1. API 서버 시작:"
echo "   python api.py --storage local  # 로컬 저장"
echo "   python api.py --storage nas    # NAS 저장 (마운트 필요)"
echo ""
echo "2. 테스트:"
echo "   python api_test.py"
echo ""
echo "3. 자세한 가이드:"
echo "   cat STORAGE_MODE_GUIDE.md"
echo ""

