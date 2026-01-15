#!/bin/bash
################################################################################
# jMetal 라이브러리 다운로드 스크립트
# 
# 설명: jMetal 관련 JAR 파일을 Maven Central에서 다운로드
# 사용법: ./download_jmetal.sh
################################################################################

set -e

# ==============================================================================
# Version note:
# - jMetal 6.x: requires Java 11+ (class file version 55)
# - Spark 3.1.x environments are often Java 8 (class file version 52)
# So we default to a Java 8-compatible jMetal 5.x version.
#
# Override:
#   JMETAL_VERSION=6.1 ./download_jmetal.sh   # if your spark-shell runs on Java 11+
# ==============================================================================

# NOTE: jMetal 5.10 is compiled for Java 11 (major=55). For Spark(Java8) use an older 5.x.
JMETAL_VERSION="${JMETAL_VERSION:-5.6}"
DOWNLOAD_DIR="./lib"
MAVEN_REPO="https://repo1.maven.org/maven2"

echo "================================================================================"
echo "jMetal Library Downloader"
echo "================================================================================"
echo ""
echo "Version: $JMETAL_VERSION"
echo "Download directory: $DOWNLOAD_DIR"
echo ""

# 다운로드 디렉토리 생성
mkdir -p "$DOWNLOAD_DIR"

# JAR 파일 목록
declare -a JARS=(
    "org/uma/jmetal/jmetal-core/$JMETAL_VERSION/jmetal-core-$JMETAL_VERSION.jar"
    "org/uma/jmetal/jmetal-algorithm/$JMETAL_VERSION/jmetal-algorithm-$JMETAL_VERSION.jar"
    "org/uma/jmetal/jmetal-problem/$JMETAL_VERSION/jmetal-problem-$JMETAL_VERSION.jar"
)

# 다운로드 함수
download_jar() {
    local jar_path=$1
    local filename=$(basename "$jar_path")
    local url="$MAVEN_REPO/$jar_path"
    
    echo "Downloading: $filename"
    echo "  URL: $url"
    
    if [ -f "$DOWNLOAD_DIR/$filename" ]; then
        echo "  ✓ Already exists, skipping"
    else
        if curl -L -f -o "$DOWNLOAD_DIR/$filename" "$url"; then
            echo "  ✓ Downloaded successfully"
        else
            echo "  ✗ Failed to download"
            return 1
        fi
    fi
    echo ""
}

# 모든 JAR 다운로드
echo "Starting downloads..."
echo ""

for jar in "${JARS[@]}"; do
    download_jar "$jar" || {
        echo "Error: Failed to download $jar"
        exit 1
    }
done

echo "================================================================================"
echo "Download completed!"
echo "================================================================================"
echo ""
echo "Downloaded JARs:"
ls -lh "$DOWNLOAD_DIR"/*.jar
echo ""

# JAR 경로 리스트 생성
JAR_LIST=$(ls "$DOWNLOAD_DIR"/*.jar | tr '\n' ',' | sed 's/,$//')

echo "To use in spark-shell:"
echo ""
echo "  spark-shell --jars $JAR_LIST"
echo ""
echo "Or load optimize_ost.scala:"
echo ""
echo "  spark-shell --jars $JAR_LIST -i optimize_ost.scala"
echo ""
echo "================================================================================"

# 편의 스크립트는 루트에 생성 (lib 디렉토리가 아닌)
cat > "spark-shell-with-lib.sh" << EOF
#!/bin/bash
# Spark Shell with all JARs in lib/
SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
LIB_DIR="\$SCRIPT_DIR/lib"

if [ ! -d "\$LIB_DIR" ]; then
    echo "Error: lib directory not found"
    exit 1
fi

JAR_LIST=\$(find "\$LIB_DIR" -name "*.jar" | tr '\n' ',' | sed 's/,\$//')

echo "Starting spark-shell with all optimizer libraries..."
echo "JARs from: \$LIB_DIR"
echo ""

spark-shell --jars "\$JAR_LIST" "\$@"
EOF

chmod +x "spark-shell-with-lib.sh"

echo "Convenience script created: spark-shell-with-lib.sh"
echo ""

# 환경 변수 설정 방법 안내
echo "================================================================================"
echo "Environment Variable Setup"
echo "================================================================================"
echo ""
echo "To set up JMETAL_JARS environment variable:"
echo ""
echo "  source setup_jmetal_env.sh"
echo ""
echo "Then use it with spark-shell:"
echo ""
echo "  spark-shell --jars \$JMETAL_JARS"
echo "  spark-shell --jars \$JMETAL_JARS -i optimize_ost.scala"
echo ""
echo "If you also have OR-Tools, combine them:"
echo ""
echo "  spark-shell --jars \$ORTOOLS_JARS,\$JMETAL_JARS"
echo "  # Or use \$ALL_OPTIMIZER_JARS (automatically set)"
echo ""
echo "================================================================================"
echo ""
echo "Quick start (without environment variable):"
echo "  cd $DOWNLOAD_DIR"
echo "  ./spark-shell-jmetal.sh -i ../optimize_ost.scala"
echo ""
echo "================================================================================"
