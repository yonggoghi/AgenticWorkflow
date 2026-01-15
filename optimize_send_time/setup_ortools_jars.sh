#!/bin/bash

# ============================================================================
# Google OR-Tools JAR 파일 다운로드 및 설정
# ============================================================================

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

WORK_DIR="/Users/yongwook/workspace/AgenticWorkflow/optimize_send_time"
cd "$WORK_DIR"

# JAR 저장 디렉토리
JAR_DIR="lib"
mkdir -p "$JAR_DIR"

echo "========================================="
echo "Google OR-Tools JAR Setup"
echo "========================================="
echo ""

# OR-Tools 버전 (리눅스 서버와 동일하게)
ORTOOLS_VERSION="9.8.3296"
JNA_VERSION="5.13.0"

# macOS 아키텍처 확인
ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ]; then
    ORTOOLS_PLATFORM="darwin-aarch64"
    echo "Detected: Apple Silicon (M1/M2/M3)"
elif [ "$ARCH" = "x86_64" ]; then
    ORTOOLS_PLATFORM="darwin-x86-64"
    echo "Detected: Intel Mac"
else
    echo -e "${RED}Unknown architecture: $ARCH${NC}"
    exit 1
fi

echo ""
echo "Required JARs:"
echo "  1. ortools-$ORTOOLS_PLATFORM-$ORTOOLS_VERSION.jar"
echo "  2. ortools-java-$ORTOOLS_VERSION.jar"
echo "  3. jna-$JNA_VERSION.jar"
echo "  4. jna-platform-$JNA_VERSION.jar"
echo ""

# Maven Central URLs
MAVEN_CENTRAL="https://repo1.maven.org/maven2"
ORTOOLS_BASE="$MAVEN_CENTRAL/com/google/ortools"
JNA_BASE="$MAVEN_CENTRAL/net/java/dev/jna"

# JAR 파일 목록 (배열 방식)
JAR_FILES=(
    "ortools-$ORTOOLS_PLATFORM-$ORTOOLS_VERSION.jar"
    "ortools-java-$ORTOOLS_VERSION.jar"
    "jna-$JNA_VERSION.jar"
    "jna-platform-$JNA_VERSION.jar"
)

# 다운로드 함수
download_jar() {
    local jar_name="$1"
    local jar_path="$JAR_DIR/$jar_name"
    local url=""
    
    # URL 결정
    if [[ "$jar_name" == ortools-darwin-* ]] || [[ "$jar_name" == ortools-linux-* ]]; then
        local platform=$(echo "$jar_name" | sed -E 's/ortools-([^-]+-[^-]+)-.*/\1/')
        url="$ORTOOLS_BASE/ortools-$platform/$ORTOOLS_VERSION/ortools-$platform-$ORTOOLS_VERSION.jar"
    elif [[ "$jar_name" == ortools-java-* ]]; then
        url="$ORTOOLS_BASE/ortools-java/$ORTOOLS_VERSION/ortools-java-$ORTOOLS_VERSION.jar"
    elif [[ "$jar_name" == jna-platform-* ]]; then
        url="$JNA_BASE/jna-platform/$JNA_VERSION/jna-platform-$JNA_VERSION.jar"
    elif [[ "$jar_name" == jna-* ]]; then
        url="$JNA_BASE/jna/$JNA_VERSION/jna-$JNA_VERSION.jar"
    else
        echo -e "${RED}✗ Unknown JAR type: $jar_name${NC}"
        return 1
    fi
    
    # 다운로드
    if [ -f "$jar_path" ]; then
        echo -e "${GREEN}✓${NC} Already exists: $jar_name"
        return 0
    else
        echo -e "${BLUE}Downloading: $jar_name${NC}"
        echo "  URL: $url"
        
        if curl -L -f -o "$jar_path" "$url" 2>/dev/null; then
            echo -e "${GREEN}✓ Downloaded successfully${NC}"
            return 0
        else
            echo -e "${RED}✗ Failed to download${NC}"
            return 1
        fi
    fi
}

# JAR 파일 다운로드
echo "Downloading JARs..."
echo ""

for jar_name in "${JAR_FILES[@]}"; do
    if ! download_jar "$jar_name"; then
        echo ""
        echo -e "${RED}Error: Failed to download $jar_name${NC}"
        exit 1
    fi
    echo ""
done

echo ""
echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}All JARs ready!${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""

# JAR 파일 크기 확인
echo "Downloaded JARs:"
ls -lh "$JAR_DIR"/*.jar | awk '{print "  " $9, "(" $5 ")"}'
echo ""

# 환경 변수 설정 파일 생성
ORTOOLS_ENV="$WORK_DIR/ortools_env.sh"
cat > "$ORTOOLS_ENV" << EOF
#!/bin/bash

# Google OR-Tools JAR paths for Spark
export ORTOOLS_JARS="$(pwd)/$JAR_DIR/ortools-$ORTOOLS_PLATFORM-$ORTOOLS_VERSION.jar,$(pwd)/$JAR_DIR/ortools-java-$ORTOOLS_VERSION.jar,$(pwd)/$JAR_DIR/jna-$JNA_VERSION.jar,$(pwd)/$JAR_DIR/jna-platform-$JNA_VERSION.jar"

echo "OR-Tools JARs configured:"
echo "  \$ORTOOLS_JARS"
EOF

chmod +x "$ORTOOLS_ENV"

echo "Environment file created: $ORTOOLS_ENV"
echo ""
echo "To use OR-Tools JARs in Spark:"
echo ""
echo "  # Option 1: Source the environment file"
echo "  source $ORTOOLS_ENV"
echo "  spark-shell --jars \$ORTOOLS_JARS -i optimize_ost.scala"
echo ""
echo "  # Option 2: Direct path"
echo "  spark-shell --jars \\"
echo "    $(pwd)/$JAR_DIR/ortools-$ORTOOLS_PLATFORM-$ORTOOLS_VERSION.jar,\\"
echo "    $(pwd)/$JAR_DIR/ortools-java-$ORTOOLS_VERSION.jar,\\"
echo "    $(pwd)/$JAR_DIR/jna-$JNA_VERSION.jar,\\"
echo "    $(pwd)/$JAR_DIR/jna-platform-$JNA_VERSION.jar \\"
echo "    -i optimize_ost.scala"
echo ""

# 리눅스 서버용 JAR 경로 정보 생성
cat > "$WORK_DIR/ortools_jars_linux.txt" << EOF
# Linux Server OR-Tools JAR paths (for reference)
# Copy this to Red Hat server and adjust paths

ORTOOLS_JARS_LINUX="/home/skinet/myfiles/data_bus/ortools-linux-x86-64-9.8.3296.jar,/home/skinet/myfiles/data_bus/ortools-java-9.8.3296.jar,/home/skinet/myfiles/data_bus/jna-5.13.0.jar,/home/skinet/myfiles/data_bus/jna-platform-5.13.0.jar"

# Usage on Linux:
spark-shell --jars \$ORTOOLS_JARS_LINUX -i optimize_ost.scala
EOF

echo "Linux JAR paths saved to: ortools_jars_linux.txt"
echo ""
