#!/bin/bash
# 원격 파일 삭제 기능 테스트 스크립트

echo "=== 원격 파일 삭제 기능 테스트 (v2.2) ==="
echo "이 테스트는 --merge-partitions 옵션에 따른 삭제 방식을 검증합니다."
echo ""

# 색상 정의
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 테스트 환경 변수
REMOTE_USER="${REMOTE_USER:-testuser}"
REMOTE_IP="${REMOTE_IP:-192.168.1.100}"
REMOTE_PATH="${REMOTE_PATH:-/tmp/test_hdfs_transfer}"
DIR_NAME="test_table"
OUTPUT_FILENAME="test_data_202601.parquet"
ARCHIVE_NAME="test_data_202601.tar.gz"

echo "테스트 설정:"
echo "  REMOTE_USER: $REMOTE_USER"
echo "  REMOTE_IP: $REMOTE_IP"
echo "  REMOTE_PATH: $REMOTE_PATH"
echo "  DIR_NAME: $DIR_NAME"
echo "  OUTPUT_FILENAME: $OUTPUT_FILENAME"
echo "  ARCHIVE_NAME: $ARCHIVE_NAME"
echo ""

# SSH 연결 확인
echo "1. SSH 연결 확인..."
if ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$REMOTE_USER@$REMOTE_IP" "echo 'SSH OK'" &>/dev/null; then
    echo -e "${GREEN}✅ SSH 연결 성공${NC}"
else
    echo -e "${RED}❌ SSH 연결 실패${NC}"
    echo "SSH 연결 정보를 확인하세요."
    exit 1
fi
echo ""

# 테스트 파일 생성
echo "2. 원격 서버에 테스트 파일 생성..."
ssh "$REMOTE_USER@$REMOTE_IP" << EOF
    # 디렉토리 생성
    mkdir -p $REMOTE_PATH/$DIR_NAME
    
    # 테스트 파일들 생성
    echo "test data 1" > $REMOTE_PATH/$DIR_NAME/$OUTPUT_FILENAME
    echo "test data 2" > $REMOTE_PATH/$DIR_NAME/other_file.parquet
    echo "test data 3" > $REMOTE_PATH/$DIR_NAME/backup.parquet
    
    # EOF 파일 생성
    BASE_NAME=\$(echo "$ARCHIVE_NAME" | sed 's/.parquet//g' | sed 's/.tar.gz//g')
    touch $REMOTE_PATH/\${BASE_NAME}.eof
    
    # 다른 EOF 파일
    touch $REMOTE_PATH/other_file.eof
    
    echo "생성된 파일 목록:"
    ls -la $REMOTE_PATH/$DIR_NAME/
    ls -la $REMOTE_PATH/*.eof
EOF
echo -e "${GREEN}✅ 테스트 파일 생성 완료${NC}"
echo ""

# 파일 삭제 전 상태 확인
echo "3. 파일 삭제 전 상태:"
ssh "$REMOTE_USER@$REMOTE_IP" << EOF
    echo "디렉토리 내 파일:"
    ls -1 $REMOTE_PATH/$DIR_NAME/ | while read file; do
        echo "  - \$file"
    done
    
    echo "EOF 파일:"
    ls -1 $REMOTE_PATH/*.eof | while read file; do
        echo "  - \$(basename \$file)"
    done
EOF
echo ""

# 삭제할 파일 목록
BASE_NAME=$(echo "$ARCHIVE_NAME" | sed 's/.parquet//g' | sed 's/.tar.gz//g')
OUTPUT_FILE_PATH="$REMOTE_PATH/$DIR_NAME/$OUTPUT_FILENAME"
EOF_FILE_PATH="$REMOTE_PATH/${BASE_NAME}.eof"

echo "4. 삭제할 파일:"
echo -e "  - ${YELLOW}$OUTPUT_FILE_PATH${NC}"
echo -e "  - ${YELLOW}$EOF_FILE_PATH${NC}"
echo ""

# 파일 삭제 실행
echo "5. 파일 삭제 실행..."
ssh "$REMOTE_USER@$REMOTE_IP" "rm -f $OUTPUT_FILE_PATH $EOF_FILE_PATH"
echo -e "${GREEN}✅ 삭제 완료${NC}"
echo ""

# 파일 삭제 후 상태 확인
echo "6. 파일 삭제 후 상태:"
ssh "$REMOTE_USER@$REMOTE_IP" << EOF
    echo "디렉토리 내 파일 (남아있는 것):"
    if [ -d $REMOTE_PATH/$DIR_NAME ]; then
        ls -1 $REMOTE_PATH/$DIR_NAME/ 2>/dev/null | while read file; do
            echo "  ✅ \$file (유지됨)"
        done
    else
        echo "  ❌ 디렉토리가 삭제됨!"
    fi
    
    echo ""
    echo "EOF 파일 (남아있는 것):"
    ls -1 $REMOTE_PATH/*.eof 2>/dev/null | while read file; do
        echo "  ✅ \$(basename \$file) (유지됨)"
    done
    
    echo ""
    echo "삭제된 파일 확인:"
    if [ ! -f "$OUTPUT_FILE_PATH" ]; then
        echo "  ✅ $OUTPUT_FILENAME (삭제됨)"
    else
        echo "  ❌ $OUTPUT_FILENAME (여전히 존재)"
    fi
    
    if [ ! -f "$EOF_FILE_PATH" ]; then
        echo "  ✅ ${BASE_NAME}.eof (삭제됨)"
    else
        echo "  ❌ ${BASE_NAME}.eof (여전히 존재)"
    fi
EOF
echo ""

# 검증
echo "7. 테스트 결과 검증..."
VERIFICATION_RESULT=$(ssh "$REMOTE_USER@$REMOTE_IP" << EOF
    ERRORS=0
    
    # 디렉토리가 존재하는지 확인
    if [ ! -d $REMOTE_PATH/$DIR_NAME ]; then
        echo "ERROR: 디렉토리가 삭제되었습니다"
        ERRORS=\$((ERRORS + 1))
    fi
    
    # 다른 파일들이 유지되는지 확인
    if [ ! -f $REMOTE_PATH/$DIR_NAME/other_file.parquet ]; then
        echo "ERROR: other_file.parquet가 삭제되었습니다"
        ERRORS=\$((ERRORS + 1))
    fi
    
    if [ ! -f $REMOTE_PATH/$DIR_NAME/backup.parquet ]; then
        echo "ERROR: backup.parquet가 삭제되었습니다"
        ERRORS=\$((ERRORS + 1))
    fi
    
    if [ ! -f $REMOTE_PATH/other_file.eof ]; then
        echo "ERROR: other_file.eof가 삭제되었습니다"
        ERRORS=\$((ERRORS + 1))
    fi
    
    # 대상 파일들이 삭제되었는지 확인
    if [ -f "$OUTPUT_FILE_PATH" ]; then
        echo "ERROR: $OUTPUT_FILENAME가 삭제되지 않았습니다"
        ERRORS=\$((ERRORS + 1))
    fi
    
    if [ -f "$EOF_FILE_PATH" ]; then
        echo "ERROR: ${BASE_NAME}.eof가 삭제되지 않았습니다"
        ERRORS=\$((ERRORS + 1))
    fi
    
    if [ \$ERRORS -eq 0 ]; then
        echo "SUCCESS"
    fi
EOF
)

if echo "$VERIFICATION_RESULT" | grep -q "SUCCESS"; then
    echo -e "${GREEN}✅ 모든 테스트 통과!${NC}"
    echo ""
    echo "검증 완료:"
    echo "  ✅ 디렉토리 유지됨"
    echo "  ✅ 다른 파일들 유지됨"
    echo "  ✅ 대상 파일만 삭제됨"
else
    echo -e "${RED}❌ 테스트 실패!${NC}"
    echo ""
    echo "$VERIFICATION_RESULT"
fi
echo ""

# 정리
echo "8. 테스트 환경 정리..."
read -p "테스트 파일을 삭제하시겠습니까? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    ssh "$REMOTE_USER@$REMOTE_IP" "rm -rf $REMOTE_PATH"
    echo -e "${GREEN}✅ 정리 완료${NC}"
else
    echo -e "${YELLOW}테스트 파일이 유지됩니다: $REMOTE_PATH${NC}"
fi

echo ""
echo "=== 테스트 완료 ==="
