#!/bin/bash

# ============================================================================
# optimize_ost.scala 실행 스크립트
# ============================================================================

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================="
echo "Optimize Send Time - Spark Job Runner"
echo "========================================="
echo ""

# 환경 변수 확인
if [ -z "$SPARK_HOME" ]; then
    echo -e "${RED}Error: SPARK_HOME is not set${NC}"
    echo "Please run: source ~/.zshrc"
    exit 1
fi

if [ -z "$JAVA_HOME" ]; then
    echo -e "${RED}Error: JAVA_HOME is not set${NC}"
    echo "Please run: source ~/.zshrc"
    exit 1
fi

echo -e "${GREEN}✓ SPARK_HOME: $SPARK_HOME${NC}"
echo -e "${GREEN}✓ JAVA_HOME: $JAVA_HOME${NC}"
echo ""

# Spark 버전 확인
SPARK_VERSION=$($SPARK_HOME/bin/spark-shell --version 2>&1 | grep "version" | head -1)
echo "Spark Version: $SPARK_VERSION"
echo ""

# Google OR-Tools JAR 경로 설정
ORTOOLS_JARS=""
JAR_DIR="$WORK_DIR/lib"

if [ -f "$WORK_DIR/ortools_env.sh" ]; then
    source "$WORK_DIR/ortools_env.sh"
    echo -e "${GREEN}✓ OR-Tools JARs loaded from environment${NC}"
elif [ -d "$JAR_DIR" ] && [ -n "$(ls -A $JAR_DIR/*.jar 2>/dev/null)" ]; then
    # JAR 파일들을 자동으로 찾아서 설정
    ORTOOLS_JARS=$(ls "$JAR_DIR"/*.jar 2>/dev/null | tr '\n' ',' | sed 's/,$//')
    echo -e "${GREEN}✓ OR-Tools JARs found in lib directory${NC}"
else
    echo -e "${YELLOW}⚠ Warning: Google OR-Tools JARs not found${NC}"
    echo "Run './setup_ortools_jars.sh' to download required JARs."
    echo "OR-Tools optimizer requires:"
    echo "  - ortools-darwin-*.jar"
    echo "  - ortools-java-*.jar"
    echo "  - jna-*.jar"
    echo "  - jna-platform-*.jar"
    echo ""
    echo "You can still run the Simulated Annealing optimizer."
    echo ""
fi

# 실행 모드 선택
echo "Select execution mode:"
echo "1) Interactive mode (spark-shell)"
echo "2) Submit as Spark job (spark-submit)"
echo "3) Compile and run with sbt"
read -p "Enter choice [1-3]: " choice

case $choice in
    1)
        echo ""
        echo "Starting Spark Shell with optimize_ost.scala..."
        echo "Once in the shell, use: import OptimizeSendTime._"
        echo ""
        
        if [ -n "$ORTOOLS_JARS" ]; then
            echo "Using OR-Tools JARs: $ORTOOLS_JARS"
            echo ""
            $SPARK_HOME/bin/spark-shell \
                --driver-memory 4g \
                --executor-memory 4g \
                --conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
                --jars "$ORTOOLS_JARS" \
                -i optimize_ost.scala
        else
            echo "Running without OR-Tools (Simulated Annealing only)"
            echo ""
            $SPARK_HOME/bin/spark-shell \
                --driver-memory 4g \
                --executor-memory 4g \
                --conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
                -i optimize_ost.scala
        fi
        ;;
        
    2)
        echo ""
        echo "Compiling and submitting Spark job..."
        
        # Scala 파일 컴파일
        scalac -classpath "$SPARK_HOME/jars/*" optimize_ost.scala
        
        # JAR 패키징 (간단한 방법)
        jar cf optimize_ost.jar *.class
        
        # Spark Submit
        if [ -n "$ORTOOLS_JARS" ]; then
            $SPARK_HOME/bin/spark-submit \
                --class OptimizeSendTime \
                --master local[*] \
                --driver-memory 4g \
                --executor-memory 4g \
                --conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
                --jars "$ORTOOLS_JARS" \
                optimize_ost.jar
        else
            $SPARK_HOME/bin/spark-submit \
                --class OptimizeSendTime \
                --master local[*] \
                --driver-memory 4g \
                --executor-memory 4g \
                --conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
                optimize_ost.jar
        fi
        
        # 정리
        rm -f *.class optimize_ost.jar
        ;;
        
    3)
        echo ""
        echo "Using sbt to compile and run..."
        
        if [ ! -f "build.sbt" ]; then
            echo -e "${RED}Error: build.sbt not found${NC}"
            echo "Please create build.sbt first (use create_sbt_project.sh)"
            exit 1
        fi
        
        sbt run
        ;;
        
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

echo ""
echo "========================================="
echo "Execution Complete"
echo "========================================="
