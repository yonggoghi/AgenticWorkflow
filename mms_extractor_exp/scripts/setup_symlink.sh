#!/bin/bash
# 심볼릭 링크 설정 스크립트 (sudo 불필요)
# 사용법: bash setup_symlink.sh

set -e

echo "================================"
echo "심볼릭 링크 설정 시작"
echo "================================"

PROJECT_DIR="/Users/yongwook/workspace/AgenticWorkflow/mms_extractor_exp"
# macOS는 /Volumes 사용
MOUNT_POINT="/Volumes/nas_dag_images"

# NAS 마운트 확인
if ! mount | grep -q "$MOUNT_POINT"; then
    echo "❌ NAS가 마운트되지 않았습니다."
    echo "먼저 'sudo bash setup_nas_mount.sh'를 실행하세요."
    exit 1
fi

cd "$PROJECT_DIR"

# Step 1: 기존 dag_images 백업
echo ""
echo "Step 1: 기존 dag_images 백업..."
if [ -d "./dag_images" ] && [ ! -L "./dag_images" ]; then
    backup_name="dag_images_backup_$(date +%Y%m%d_%H%M%S)"
    mv ./dag_images "./$backup_name"
    echo "✅ 기존 dag_images -> $backup_name"
    
    # 백업한 파일을 NAS로 복사 (선택사항)
    echo "기존 이미지를 NAS로 복사 중..."
    cp -r "./$backup_name"/* "$MOUNT_POINT/dag_images/" 2>/dev/null || true
    echo "✅ 기존 이미지 NAS로 복사 완료"
elif [ -L "./dag_images" ]; then
    echo "⚠️  이미 심볼릭 링크가 존재합니다. 재생성합니다."
    rm ./dag_images
else
    echo "⚠️  기존 dag_images 디렉토리 없음"
fi

# Step 2: 심볼릭 링크 생성
echo ""
echo "Step 2: 심볼릭 링크 생성..."
ln -s "$MOUNT_POINT/dag_images" ./dag_images
echo "✅ 심볼릭 링크 생성 완료"

# Step 3: 검증
echo ""
echo "Step 3: 검증..."
if [ -L "./dag_images" ]; then
    echo "✅ 심볼릭 링크 존재 확인"
    ls -la ./dag_images | head -1
    echo ""
    echo "링크 대상:"
    readlink ./dag_images
else
    echo "❌ 심볼릭 링크 생성 실패"
    exit 1
fi

# 파일 목록 확인
echo ""
echo "NAS의 DAG 이미지 파일 수:"
ls -1 "$MOUNT_POINT/dag_images/" | wc -l

echo ""
echo "================================"
echo "✅ 심볼릭 링크 설정 완료!"
echo "================================"
echo ""
echo "현재 dag_images는 NAS를 가리킵니다:"
echo "  로컬: $PROJECT_DIR/dag_images"
echo "  실제: $MOUNT_POINT/dag_images"
echo "  NAS:  172.27.7.58:/aos_ext/dag_images"

