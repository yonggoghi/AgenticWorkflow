#!/bin/bash
# /etc/fstab 영구 마운트 설정 스크립트
# 사용법: sudo bash setup_fstab.sh

set -e

echo "================================"
echo "영구 마운트 설정 (fstab)"
echo "================================"

# macOS는 /Volumes 사용
FSTAB_LINE="172.27.7.58:/aos_ext /Volumes/nas_dag_images nfs resvport,rw,bg,hard,intr,nolocks,tcp,nofail 0 0"

# fstab 백업
echo "Step 1: /etc/fstab 백업..."
if [ ! -f "/etc/fstab.backup" ]; then
    cp /etc/fstab /etc/fstab.backup
    echo "✅ /etc/fstab -> /etc/fstab.backup"
else
    echo "✅ 백업 파일 이미 존재"
fi

# 이미 설정되어 있는지 확인
if grep -q "172.27.7.58:/aos_ext" /etc/fstab; then
    echo "✅ fstab에 이미 설정되어 있습니다."
    exit 0
fi

# fstab에 추가
echo ""
echo "Step 2: fstab에 NFS 마운트 추가..."
echo "" >> /etc/fstab
echo "# NAS DAG Images - Added on $(date)" >> /etc/fstab
echo "$FSTAB_LINE" >> /etc/fstab
echo "✅ fstab 설정 완료"

echo ""
echo "추가된 내용:"
tail -3 /etc/fstab

echo ""
echo "================================"
echo "✅ 영구 마운트 설정 완료!"
echo "================================"
echo ""
echo "재부팅 후에도 자동으로 마운트됩니다."
echo ""
echo "테스트하려면:"
echo "  sudo umount /Volumes/nas_dag_images"
echo "  sudo mount -a"

