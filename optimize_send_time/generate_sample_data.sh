#!/bin/bash

# ============================================================================
# 샘플 데이터 생성 스크립트
# ============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "========================================="
echo "Sample Data Generator"
echo "========================================="
echo ""

# 환경 변수 확인
if [ -z "$SPARK_HOME" ]; then
    echo -e "${RED}Error: SPARK_HOME is not set${NC}"
    echo "Please run: source ~/.zshrc"
    exit 1
fi

# 작업 디렉토리
WORK_DIR="/Users/yongwook/workspace/AgenticWorkflow/optimize_send_time"
cd "$WORK_DIR"

# 출력 디렉토리 생성
mkdir -p data/sample

echo -e "${BLUE}[Configuration]${NC}"
echo "  Users: 100,000"
echo "  Hours: 9-18 (10 hours)"
echo "  Total records: 1,000,000"
echo "  Output: aos/sto/propensityScoreDF"
echo ""

# 실행 방법 선택
echo "Select execution mode:"
echo "1) Generate full dataset (100K users, 1M records) - Recommended"
echo "2) Generate small test dataset (1K users, 10K records) - Quick test"
echo "3) Interactive mode (Spark Shell) - Manual generation"
read -p "Enter choice [1-3]: " choice

case $choice in
    1)
        echo ""
        echo -e "${GREEN}Generating full dataset...${NC}"
        echo ""
        
        # Spark Shell을 사용하여 Scala 파일 로드 및 실행
        spark-shell --driver-memory 4g --conf spark.driver.maxResultSize=2g << 'SCALA_EOF'
:load generate_sample_data.scala

import GenerateSampleData._

// Main 메서드 실행
GenerateSampleData.main(Array())

System.exit(0)
SCALA_EOF
        
        if [ $? -eq 0 ] && [ -d "aos/sto/propensityScoreDF" ]; then
            echo ""
            echo -e "${GREEN}✓ Data generation complete!${NC}"
            echo ""
            echo "Output location:"
            echo "  $(pwd)/aos/sto/propensityScoreDF"
            echo ""
            echo "File size:"
            du -sh aos/sto/propensityScoreDF 2>/dev/null || echo "  (calculating...)"
            echo ""
            echo "Parquet files:"
            ls -1 aos/sto/propensityScoreDF/part-*.parquet 2>/dev/null | wc -l | xargs echo "  Count:"
        else
            echo -e "${RED}✗ Data generation failed${NC}"
            exit 1
        fi
        ;;
        
    2)
        echo ""
        echo -e "${GREEN}Generating small test dataset...${NC}"
        echo ""
        
        # Spark Shell로 실행 (빠른 테스트)
        spark-shell --driver-memory 2g << 'SCALA_EOF'
import scala.util.Random

println("=" * 80)
println("Generating small test dataset (1,000 users)")
println("=" * 80)

val numUsers = 1000
val sendYm = "202512"
val hours = 9 to 18

println(s"Creating ${numUsers} users with ${hours.length} hours each...")

val users = (0 until numUsers).map { i =>
  f"s:${i%10}${Random.nextInt(10)}${Random.alphanumeric.take(12).mkString.toLowerCase}${i%10000}%05d"
}

val data = for {
  user <- users
  hour <- hours
} yield {
  val preferredHour = 9 + (user.hashCode().abs % 10)
  val distance = Math.abs(hour - preferredHour)
  val baseScore = 0.5 + (Random.nextDouble() * 0.3)
  val distancePenalty = distance * 0.05
  val score = Math.max(0.1, Math.min(0.99, baseScore - distancePenalty + (Random.nextDouble() * 0.2 - 0.1)))
  
  (user, sendYm, hour, score)
}

val df = data.toDF("svc_mgmt_num", "send_ym", "send_hour", "propensity_score")

println(s"\nGenerated ${df.count()} records")
println("\nSample data (first 20 rows):")
df.orderBy("svc_mgmt_num", "send_hour").show(20, false)

val outputPath = "aos/sto/propensityScoreDF"
println(s"\nSaving to $outputPath...")
df.write.mode("overwrite").parquet(outputPath)
println(s"✓ Saved successfully")

println("\n" + "=" * 80)
println("Small dataset generation complete!")
println("=" * 80)

System.exit(0)
SCALA_EOF
        
        if [ $? -eq 0 ] && [ -d "aos/sto/propensityScoreDF" ]; then
            echo ""
            echo -e "${GREEN}✓ Small dataset generated!${NC}"
            du -sh aos/sto/propensityScoreDF 2>/dev/null
        else
            echo -e "${RED}✗ Generation failed${NC}"
            exit 1
        fi
        ;;
        
    3)
        echo ""
        echo -e "${BLUE}Starting Spark Shell for manual generation...${NC}"
        echo ""
        echo "You can use these functions:"
        echo "  1. generateSampleData(spark, numUsers, sendYm, outputPath)"
        echo "  2. generateSmallSampleData(spark, numUsers)"
        echo ""
        
        spark-shell \
            --driver-memory 4g \
            -i generate_sample_data.scala
        ;;
        
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

echo ""
echo "========================================="
echo "Next Steps"
echo "========================================="
echo ""
echo "To use the generated data in optimize_ost.scala:"
echo ""
echo "1. Start Spark Shell:"
echo "   spark-shell --driver-memory 4g -i optimize_ost.scala"
echo ""
echo "2. Load the data:"
echo "   val dfAll = spark.read.parquet(\"aos/sto/propensityScoreDF\").cache()"
echo ""
echo "3. Run optimization:"
echo "   import OptimizeSendTime._"
echo "   val df = dfAll.limit(10000)  // Test with 10K records first"
echo "   val capacity = Map(9->1000, 10->1000, 11->1000, 12->1000,"
echo "                      13->1000, 14->1000, 15->1000, 16->1000,"
echo "                      17->1000, 18->1000)"
echo "   val result = allocateGreedySimple(df, Array(9,10,11,12,13,14,15,16,17,18), capacity)"
echo "   result.show()"
echo ""
