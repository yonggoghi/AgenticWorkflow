#!/bin/bash

# ============================================================================
# Java GreedyAllocator 실행 스크립트
# ============================================================================

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

WORK_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$WORK_DIR"

echo "========================================"
echo "GreedyAllocator Java Test Runner"
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

# JAR 파일 확인
BUILD_DIR="$WORK_DIR/build"
JAR_FILE="$BUILD_DIR/greedy-allocator.jar"

if [ ! -f "$JAR_FILE" ]; then
    echo -e "${RED}✗ JAR file not found: $JAR_FILE${NC}"
    echo ""
    echo "Please build first:"
    echo "  ./build_java.sh"
    echo ""
    exit 1
fi

echo -e "${GREEN}✓ JAR file: $JAR_FILE${NC}"

# 데이터 확인
DATA_PATH="aos/sto/propensityScoreDF"
if [ ! -d "$DATA_PATH" ]; then
    echo -e "${YELLOW}⚠ Warning: Data not found at $DATA_PATH${NC}"
    echo ""
    echo "Please generate data first:"
    echo "  ./generate_data_simple.sh 100000"
    echo ""
    read -p "Continue anyway? (y/N): " CONTINUE
    if [[ ! "$CONTINUE" =~ ^[Yy]$ ]]; then
        exit 0
    fi
fi

echo ""

# 메모리 및 코어 설정 (커맨드라인 인자로 조정 가능)
DRIVER_CORES=${1:-16}
DRIVER_MEM=${2:-100g}
MAX_RESULT_SIZE=${3:-30g}

echo "========================================"
echo "Spark Configuration"
echo "========================================"
echo ""
echo "  Master: local[*]"
echo "  Driver cores: $DRIVER_CORES"
echo "  Driver memory: $DRIVER_MEM"
echo "  Max result size: $MAX_RESULT_SIZE"
echo ""
echo "To customize:"
echo "  $0 <cores> <driver-mem> <max-result-size>"
echo "  Example: $0 32 150g 50g"
echo ""

read -p "Press Enter to start, or Ctrl+C to cancel..."

echo ""
echo "========================================"
echo "Running Allocation Test"
echo "========================================"
echo ""

START_TIME=$(date +%s)

$SPARK_HOME/bin/spark-submit \
  --master "local[*]" \
  --class optimize_send_time.GreedyAllocatorTest \
  --driver-cores $DRIVER_CORES \
  --driver-memory $DRIVER_MEM \
  --conf spark.driver.maxResultSize=$MAX_RESULT_SIZE \
  "$JAR_FILE"

EXIT_CODE=$?
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED / 60))
ELAPSED_SEC=$((ELAPSED % 60))

echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ Test Complete!${NC}"
else
    echo -e "${RED}✗ Test Failed!${NC}"
fi
echo "========================================"
echo ""
echo "Execution time: ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "Results saved to: aos/sto/allocation_result"
    echo ""
    echo "To view results:"
    echo "  spark-shell"
    echo "  scala> val result = spark.read.parquet(\"aos/sto/allocation_result\")"
    echo "  scala> result.show(20)"
    echo ""
fi

exit $EXIT_CODE
