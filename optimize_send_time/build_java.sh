#!/bin/bash

# ============================================================================
# Java 버전 빌드 및 실행 스크립트
# ============================================================================

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "========================================"
echo "GreedyAllocator Java Build & Run"
echo "========================================"
echo ""

# 환경 확인
if [ -z "$SPARK_HOME" ]; then
    echo -e "${RED}✗ SPARK_HOME is not set${NC}"
    echo ""
    echo "Please set SPARK_HOME first:"
    echo "  export SPARK_HOME=/path/to/spark"
    echo ""
    exit 1
fi

echo -e "${GREEN}✓ SPARK_HOME: $SPARK_HOME${NC}"

# Java 확인
if ! command -v javac &> /dev/null; then
    echo -e "${RED}✗ javac not found${NC}"
    echo ""
    echo "Please install JDK 11 or higher"
    echo ""
    exit 1
fi

JAVA_VERSION=$(javac -version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✓ Java version: $JAVA_VERSION${NC}"

# 작업 디렉토리
CURRENT_DIR=$(pwd)
BUILD_DIR="$CURRENT_DIR/build"
mkdir -p "$BUILD_DIR"

echo ""
echo "========================================"
echo "Step 1: Compiling Java files"
echo "========================================"

# Spark JARs classpath 생성
SPARK_JARS=$(find "$SPARK_HOME/jars" -name '*.jar' | tr '\n' ':')

echo "Compiling GreedyAllocator.java..."
javac -cp "$SPARK_JARS" \
      -d "$BUILD_DIR" \
      -Xlint:unchecked \
      GreedyAllocator.java

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Compilation failed: GreedyAllocator.java${NC}"
    exit 1
fi

echo -e "${GREEN}✓ GreedyAllocator.java compiled${NC}"

echo "Compiling GreedyAllocatorTest.java..."
javac -cp "$SPARK_JARS:$BUILD_DIR" \
      -d "$BUILD_DIR" \
      -Xlint:unchecked \
      GreedyAllocatorTest.java

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Compilation failed: GreedyAllocatorTest.java${NC}"
    exit 1
fi

echo -e "${GREEN}✓ GreedyAllocatorTest.java compiled${NC}"

echo ""
echo "========================================"
echo "Step 2: Creating JAR"
echo "========================================"

cd "$BUILD_DIR"
jar cf greedy-allocator.jar optimize_send_time/*.class
cd "$CURRENT_DIR"

if [ ! -f "$BUILD_DIR/greedy-allocator.jar" ]; then
    echo -e "${RED}✗ JAR creation failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ JAR created: $BUILD_DIR/greedy-allocator.jar${NC}"

# 실행 옵션
echo ""
echo "========================================"
echo "Build Complete!"
echo "========================================"
echo ""
echo "Build artifacts:"
echo "  Classes: $BUILD_DIR/optimize_send_time/"
echo "  JAR: $BUILD_DIR/greedy-allocator.jar"
echo ""
echo "To run the test:"
echo ""
echo "  ./run_java_allocator.sh"
echo ""
echo "To run with custom settings:"
echo "  ./run_java_allocator.sh <cores> <driver-mem> <max-result-size>"
echo "  Example: ./run_java_allocator.sh 32 150g 50g"
echo ""
echo "Or use spark-submit directly:"
echo "  spark-submit \\"
echo "    --master \"local[*]\" \\"
echo "    --class optimize_send_time.GreedyAllocatorTest \\"
echo "    --driver-cores 16 \\"
echo "    --driver-memory 100g \\"
echo "    --conf spark.driver.maxResultSize=30g \\"
echo "    $BUILD_DIR/greedy-allocator.jar"
echo ""
