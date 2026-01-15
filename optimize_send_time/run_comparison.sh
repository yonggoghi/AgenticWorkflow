#!/bin/bash

# ============================================================================
# Batch vs All-at-Once Comparison Runner
# ============================================================================

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "========================================"
echo "Batch vs All-at-Once Comparison"
echo "========================================"
echo ""

CURRENT_DIR=$(pwd)

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

# 데이터 확인
DATA_PATH="aos/sto/propensityScoreDF"
if [ ! -d "$DATA_PATH" ]; then
    echo -e "${RED}✗ Data not found: $DATA_PATH${NC}"
    echo ""
    echo "Please generate data first:"
    echo "  ./generate_data_simple.sh 100000"
    echo ""
    exit 1
fi

echo -e "${GREEN}✓ Data found: $DATA_PATH${NC}"

# 메모리 설정
DRIVER_MEM=${1:-8g}
EXECUTOR_MEM=${2:-8g}

echo ""
echo "Memory settings:"
echo "  Driver: $DRIVER_MEM"
echo "  Executor: $EXECUTOR_MEM"
echo ""
echo "To use different memory:"
echo "  $0 16g 16g"
echo ""

# Spark Shell 실행
echo -e "${BLUE}Starting Spark Shell...${NC}"
echo ""

$SPARK_HOME/bin/spark-shell \
  --driver-memory $DRIVER_MEM \
  --executor-memory $EXECUTOR_MEM \
  -i compare_batch_vs_all.scala

echo ""
echo "========================================"
echo "Comparison Complete!"
echo "========================================"
