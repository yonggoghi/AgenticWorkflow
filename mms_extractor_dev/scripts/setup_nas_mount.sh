#!/bin/bash
# NAS DAG Images 마운트 설정 스크립트
# 사용법: sudo bash setup_nas_mount.sh

set -e  # 오류 발생 시 중단

echo "================================"
echo "NAS DAG Images 마운트 설정 시작"
echo "================================"

# 변수 설정
NAS_IP="172.27.7.58"
NAS_PATH="/aos_ext"
# Linux는 /mnt 사용
MOUNT_POINT="/mnt/nas_dag_images"
PROJECT_DIR="$(dirname "$(dirname "$(readlink -f "$0")")")"

# Step 1: 마운트 디렉토리 생성
echo ""
echo "Step 1: 마운트 디렉토리 생성..."
if [ ! -d "$MOUNT_POINT" ]; then
    mkdir -p "$MOUNT_POINT"
    echo "✅ $MOUNT_POINT 생성 완료"
else
    echo "✅ $MOUNT_POINT 이미 존재"
fi

# 소유자 설정 (현재 사용자)
chown $(whoami):$(id -gn) "$MOUNT_POINT"
echo "✅ 소유자 설정 완료"

# Step 2: NFS 마운트
echo ""
echo "Step 2: NFS 마운트 실행..."

# 이미 마운트되어 있는지 확인
if mount | grep -q "$MOUNT_POINT"; then
    echo "⚠️  이미 마운트되어 있습니다. 언마운트 후 재마운트합니다."
    umount -f "$MOUNT_POINT" 2>/dev/null || true
fi

# NFS 마운트 (Linux)
mount -t nfs -o rw,hard,intr,nfsvers=3 "$NAS_IP:$NAS_PATH" "$MOUNT_POINT"

if mount | grep -q "$MOUNT_POINT"; then
    echo "✅ NFS 마운트 성공"
    mount | grep "$MOUNT_POINT"
else
    echo "❌ NFS 마운트 실패"
    exit 1
fi

# Step 3: NAS에 dag_images 디렉토리 생성
echo ""
echo "Step 3: NAS에 dag_images 디렉토리 생성..."
if [ ! -d "$MOUNT_POINT/dag_images" ]; then
    mkdir -p "$MOUNT_POINT/dag_images"
    echo "✅ $MOUNT_POINT/dag_images 생성 완료"
else
    echo "✅ dag_images 디렉토리 이미 존재"
fi

chown $(whoami):$(id -gn) "$MOUNT_POINT/dag_images"
chmod 755 "$MOUNT_POINT/dag_images"

# Step 4: 쓰기 권한 테스트
echo ""
echo "Step 4: 쓰기 권한 테스트..."
test_file="$MOUNT_POINT/dag_images/test_$(date +%s).txt"
if touch "$test_file" 2>/dev/null; then
    echo "✅ 쓰기 권한 정상"
    rm "$test_file"
else
    echo "❌ 쓰기 권한 없음"
    exit 1
fi

echo ""
echo "================================"
echo "✅ NAS 마운트 설정 완료!"
echo "================================"
echo ""
echo "다음 단계:"
echo "1. 기존 dag_images 백업 (자동 진행)"
echo "2. 심볼릭 링크 생성 (자동 진행)"
echo ""
echo "마운트 정보:"
df -h | grep "$MOUNT_POINT"

