#!/bin/bash
# NAS 설정 검증 스크립트 (sudo 불필요)
# 사용법: bash verify_nas_setup.sh

echo "======================================="
echo "NAS DAG Images 설정 검증"
echo "======================================="

# Linux는 /mnt 사용
MOUNT_POINT="/mnt/nas_dag_images"
PROJECT_DIR="$(dirname "$(dirname "$(readlink -f "$0")")")"

# 1. NAS 마운트 확인
echo ""
echo "1. NAS 마운트 확인..."
if mount | grep -q "$MOUNT_POINT"; then
    echo "✅ NAS 마운트 성공"
    mount | grep "$MOUNT_POINT"
else
    echo "❌ NAS 마운트 실패"
    echo "   'sudo bash setup_nas_mount.sh'를 실행하세요."
fi

# 2. 심볼릭 링크 확인
echo ""
echo "2. 심볼릭 링크 확인..."
cd "$PROJECT_DIR"
if [ -L "./dag_images" ]; then
    echo "✅ 심볼릭 링크 존재"
    ls -la ./dag_images | head -1
    echo "   링크 대상: $(readlink ./dag_images)"
else
    echo "❌ 심볼릭 링크 없음"
    echo "   'bash setup_symlink.sh'를 실행하세요."
fi

# 3. 쓰기 권한 확인
echo ""
echo "3. 쓰기 권한 확인..."
test_file="$MOUNT_POINT/dag_images/test_$(date +%s).txt"
if touch "$test_file" 2>/dev/null; then
    echo "✅ 쓰기 권한 정상"
    rm "$test_file"
else
    echo "❌ 쓰기 권한 없음"
fi

# 4. fstab 설정 확인
echo ""
echo "4. fstab 영구 마운트 확인..."
if grep -q "172.27.7.58:/aos_ext" /etc/fstab 2>/dev/null; then
    echo "✅ 영구 마운트 설정됨"
else
    echo "⚠️  영구 마운트 미설정 (재부팅 시 수동 마운트 필요)"
    echo "   'sudo bash setup_fstab.sh'를 실행하세요."
fi

# 5. 파일 수 확인
echo ""
echo "5. DAG 이미지 파일 수..."
if [ -d "$MOUNT_POINT/dag_images" ]; then
    file_count=$(ls -1 "$MOUNT_POINT/dag_images" 2>/dev/null | wc -l)
    echo "✅ NAS에 $file_count 개의 파일 존재"
else
    echo "❌ dag_images 디렉토리 없음"
fi

# 6. 디스크 사용량
echo ""
echo "6. NAS 디스크 사용량..."
df -h | grep "$MOUNT_POINT" || echo "⚠️  마운트 정보 없음"

echo ""
echo "======================================="
echo "검증 완료"
echo "======================================="
echo ""
echo "요약:"
echo "  로컬 경로: $PROJECT_DIR/dag_images"
echo "  실제 위치: $MOUNT_POINT/dag_images"
echo "  NAS 서버: 172.27.7.58:/aos_ext/dag_images"

