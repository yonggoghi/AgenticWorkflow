#!/bin/bash

# ============================================================================
# 빠른 실행 스크립트 (자동 환경 감지)
# ============================================================================

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

WORK_DIR="/Users/yongwook/workspace/AgenticWorkflow/optimize_send_time"
cd "$WORK_DIR"

echo "========================================="
echo "Quick Run - Optimize Send Time"
echo "========================================="
echo ""

# 1. Spark 환경 확인
if [ -z "$SPARK_HOME" ]; then
    echo -e "${RED}✗ SPARK_HOME not set${NC}"
    echo ""
    echo "Please run one of:"
    echo "  source ~/.zshrc"
    echo "  source ~/spark-local/spark-env.sh"
    exit 1
else
    echo -e "${GREEN}✓ SPARK_HOME: $SPARK_HOME${NC}"
fi

# 2. OR-Tools JAR 설정
JAR_DIR="$WORK_DIR/lib"
ORTOOLS_JARS=""

# 2-1. 환경 파일에서 로드 시도
if [ -f "$WORK_DIR/ortools_env.sh" ]; then
    source "$WORK_DIR/ortools_env.sh"
    echo -e "${GREEN}✓ OR-Tools JARs loaded from environment${NC}"
# 2-2. lib 디렉토리에서 찾기
elif [ -d "$JAR_DIR" ] && [ -n "$(ls -A $JAR_DIR/*.jar 2>/dev/null)" ]; then
    ORTOOLS_JARS=$(ls "$JAR_DIR"/*.jar 2>/dev/null | tr '\n' ',' | sed 's/,$//')
    echo -e "${GREEN}✓ OR-Tools JARs found in lib/${NC}"
# 2-3. JAR 없음 (Simulated Annealing만 사용)
else
    echo -e "${YELLOW}⚠ OR-Tools JARs not found${NC}"
    echo "  Run './setup_ortools_jars.sh' to download JARs"
    echo "  OR-Tools optimizer will not be available"
    echo "  (Simulated Annealing optimizer will still work)"
fi

echo ""

# 3. 실행 모드 선택
echo "Select mode:"
echo "  1) Interactive with OR-Tools (if available)"
echo "  2) Interactive without OR-Tools"
echo "  3) Load data and test"
read -p "Choice [1]: " MODE
MODE=${MODE:-1}

case $MODE in
    1)
        echo ""
        if [ -n "$ORTOOLS_JARS" ]; then
            echo -e "${GREEN}Starting with OR-Tools...${NC}"
            echo ""
            spark-shell \
                --driver-memory 4g \
                --executor-memory 4g \
                --conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
                --jars "$ORTOOLS_JARS" \
                -i optimize_ost.scala
        else
            echo -e "${YELLOW}OR-Tools JARs not found. Starting without OR-Tools...${NC}"
            echo ""
            spark-shell \
                --driver-memory 4g \
                --executor-memory 4g \
                --conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
                -i optimize_ost.scala
        fi
        ;;
        
    2)
        echo ""
        echo -e "${BLUE}Starting without OR-Tools...${NC}"
        echo ""
        spark-shell \
            --driver-memory 4g \
            --executor-memory 4g \
            --conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
            -i optimize_ost.scala
        ;;
        
    3)
        echo ""
        echo -e "${GREEN}Starting with test data...${NC}"
        echo ""
        
        # 샘플 데이터 확인
        if [ ! -d "aos/sto/propensityScoreDF" ]; then
            echo -e "${YELLOW}Sample data not found${NC}"
            read -p "Generate sample data now? [Y/n]: " GEN_DATA
            GEN_DATA=${GEN_DATA:-Y}
            
            if [[ "$GEN_DATA" =~ ^[Yy]$ ]]; then
                ./generate_data_simple.sh 1000
            else
                echo "Please run './generate_data_simple.sh' first"
                exit 1
            fi
        fi
        
        # 테스트 스크립트 생성
        TEST_SCRIPT=$(mktemp /tmp/spark_test.XXXXXX.scala)
        cat > "$TEST_SCRIPT" << 'SCALA_EOF'
// Load optimize_ost.scala
:load optimize_ost.scala

println("\n" + "=" * 80)
println("Quick Test Setup")
println("=" * 80)

// Load sample data
val dfAll = spark.read.parquet("aos/sto/propensityScoreDF").cache()
println(s"\nLoaded ${dfAll.count()} records")

// Test with 1000 users
val df = dfAll.limit(1000)

val capacity = Map(
  9 -> 100, 10 -> 100, 11 -> 100, 12 -> 100, 13 -> 100,
  14 -> 100, 15 -> 100, 16 -> 100, 17 -> 100, 18 -> 100
)

println("\n" + "=" * 80)
println("Running Greedy Allocation Test")
println("=" * 80)

import OptimizeSendTime._
val result = allocateGreedySimple(df, Array(9,10,11,12,13,14,15,16,17,18), capacity)

println("\nResults:")
result.groupBy("assigned_hour").count().orderBy("assigned_hour").show()

println("\n" + "=" * 80)
println("Test Complete!")
println("=" * 80)
println("\nYou can now:")
println("  - Try other algorithms (see QUICK_START.md)")
println("  - Load more data: val df = dfAll.limit(10000)")
println("  - Run SA: allocateUsersWithSimulatedAnnealing(...)")
println()
SCALA_EOF
        
        if [ -n "$ORTOOLS_JARS" ]; then
            spark-shell \
                --driver-memory 4g \
                --executor-memory 4g \
                --conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
                --jars "$ORTOOLS_JARS" \
                -i "$TEST_SCRIPT"
        else
            spark-shell \
                --driver-memory 4g \
                --executor-memory 4g \
                --conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
                -i "$TEST_SCRIPT"
        fi
        
        rm -f "$TEST_SCRIPT"
        ;;
        
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

echo ""
