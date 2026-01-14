#!/bin/bash

# ============================================================================
# 간단한 샘플 데이터 생성 스크립트 (Spark Shell 직접 사용)
# ============================================================================

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

WORK_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$WORK_DIR"

# 출력 디렉토리 생성
mkdir -p aos/sto

# 사용자 수 설정 (기본값: 100000)
NUM_USERS=${1:-100000}

# 기존 데이터 확인
if [ -d "aos/sto/propensityScoreDF" ]; then
    echo -e "${YELLOW}⚠ Warning: Existing data will be OVERWRITTEN${NC}"
    echo "  Location: aos/sto/propensityScoreDF"
    du -sh aos/sto/propensityScoreDF 2>/dev/null
    echo ""
    echo "Options:"
    echo "  1. Continue (overwrite)"
    echo "  2. Cancel"
    echo "  3. Use backup script (./generate_data_with_backup.sh)"
    echo ""
    read -p "Choice [1]: " OVERWRITE_CHOICE
    OVERWRITE_CHOICE=${OVERWRITE_CHOICE:-1}
    
    if [ "$OVERWRITE_CHOICE" != "1" ]; then
        echo "Cancelled"
        exit 0
    fi
    echo ""
fi

echo "========================================="
echo "Simple Data Generator"
echo "========================================="
echo ""
echo -e "${BLUE}Generating data for $NUM_USERS users...${NC}"
echo "Output: aos/sto/propensityScoreDF"
echo ""

# Spark Shell로 직접 데이터 생성
spark-shell --driver-memory 4g << SCALA_EOF

import scala.util.Random

println("=" * 80)
println("Sample Data Generation")
println("=" * 80)

val numUsers = $NUM_USERS
val sendYm = "202512"
val hours = 9 to 18
val random = new Random(42)

println(s"\nGenerating data for \$numUsers users...")
println(s"  Hours: 9-18 (10 hours)")
println(s"  Total records: \${numUsers * 10}")

// 사용자 ID 생성
val users = (0 until numUsers).map { i =>
  f"s:\${i%10}\${random.nextInt(10)}\${random.alphanumeric.take(12).mkString.toLowerCase}\${i%100000}%05d"
}

// 데이터 생성 (현실적인 분포)
val data = for {
  user <- users
  hour <- hours
} yield {
  val preferredHour = 9 + (user.hashCode().abs % 10)
  val distance = Math.abs(hour - preferredHour)
  val baseScore = 0.5 + (random.nextDouble() * 0.3)
  val distancePenalty = distance * 0.05
  val score = Math.max(0.1, Math.min(0.99, 
    baseScore - distancePenalty + (random.nextDouble() * 0.2 - 0.1)))
  
  (user, sendYm, hour, score)
}

// DataFrame 생성
val df = data.toDF("svc_mgmt_num", "send_ym", "send_hour", "propensity_score")

println(s"\n✓ Generated \${df.count()} records")

// 샘플 데이터 표시
println("\nSample data (first 20 rows):")
df.orderBy("svc_mgmt_num", "send_hour").show(20, false)

// 통계
println("\nPropensity score statistics:")
df.select(
  min("propensity_score").as("min"),
  max("propensity_score").as("max"),
  avg("propensity_score").as("avg")
).show(false)

println("\nRecords per hour:")
df.groupBy("send_hour").count().orderBy("send_hour").show(false)

// 저장
val outputPath = "aos/sto/propensityScoreDF"
println(s"\nSaving to \$outputPath...")
df.write.mode("overwrite").parquet(outputPath)
println("✓ Saved successfully")

println("\n" + "=" * 80)
println("Data Generation Complete!")
println("=" * 80)
println("\nTo use this data:")
    println("  val dfAll = spark.read.parquet(\"aos/sto/propensityScoreDF\").cache()")
println()

System.exit(0)
SCALA_EOF

# 결과 확인
if [ $? -eq 0 ] && [ -d "aos/sto/propensityScoreDF" ]; then
    echo ""
    echo -e "${GREEN}✓ Success!${NC}"
    echo ""
    echo "Output: $(pwd)/aos/sto/propensityScoreDF"
    du -sh aos/sto/propensityScoreDF 2>/dev/null
    echo ""
    echo "Next steps:"
    echo "  1. Start Spark Shell: spark-shell --driver-memory 4g -i optimize_ost.scala"
    echo "  2. Load data: val dfAll = spark.read.parquet(\"aos/sto/propensityScoreDF\").cache()"
    echo "  3. Run test: See examples in QUICK_START.md"
else
    echo -e "${RED}✗ Failed${NC}"
    exit 1
fi

echo ""
