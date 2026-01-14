// ============================================================================
// Greedy Allocator Test Script
// ============================================================================
// 
// Usage:
//   spark-shell -i test_greedy.scala
//   또는
//   spark-shell
//   scala> :load test_greedy.scala
// ============================================================================

println("""
================================================================================
Loading Greedy Allocator...
================================================================================
""")

// 1. greedy_allocation.scala 로드
:load greedy_allocation.scala

// 2. 함수 import
import GreedyAllocator._

println("""
================================================================================
Running Greedy Test
================================================================================
""")

// 3. 샘플 데이터 확인
val sampleDataPath = "aos/sto/propensityScoreDF"
val dataExists = try {
  spark.read.parquet(sampleDataPath).limit(1).count()
  true
} catch {
  case _: Exception => false
}

if (!dataExists) {
  println("""
⚠ Sample data not found!

Please generate sample data first:
  ./generate_data_simple.sh 100000

Then run this script again.
================================================================================
""")
} else {
  // 4. 데이터 로드
  println("Loading sample data...")
  val dfAll = spark.read.parquet(sampleDataPath).cache()

  val df = dfAll.filter("svc_mgmt_num like '%0'")
  val totalUsers = df.select("svc_mgmt_num").distinct().count()
  println(s"✓ Loaded $totalUsers unique users")
  
  val totalRecords = df.count()
  println(s"✓ Total records: $totalRecords")

  // 5. 용량 설정
  val capacityPerHour = (totalUsers * 0.11).toInt
  println(s"✓ Capacity per hour: $capacityPerHour")

  // 시간대별 용량 맵 생성 (9시~18시)
  val hours = Array(9, 10, 11, 12, 13, 14, 15, 16, 17, 18)
  val capacity = Map(
    9 -> capacityPerHour,
    10 -> capacityPerHour,
    11 -> capacityPerHour,
    12 -> capacityPerHour,
    13 -> capacityPerHour,
    14 -> capacityPerHour,
    15 -> capacityPerHour,
    16 -> capacityPerHour,
    17 -> capacityPerHour,
    18 -> capacityPerHour
  )

  println("\n" + "=" * 80)
  println("Starting Greedy Allocation")
  println("=" * 80)

  // 6. 실행 시간 측정
  val startTime = System.currentTimeMillis()
  
  val result = allocate(df, hours, capacity)
  
  val endTime = System.currentTimeMillis()
  val totalTimeSeconds = (endTime - startTime) / 1000.0

  println("\n" + "=" * 80)
  println("Allocation Complete!")
  println("=" * 80)
  println(f"Total execution time: $totalTimeSeconds%.2f seconds")
  println("=" * 80 + "\n")

  // 7. 결과 확인
  println("Top 20 assignments:")
  result.show(20, false)
  
  println("\nAllocation by hour:")
  result.groupBy("assigned_hour")
    .agg(
      count("*").as("count"),
      sum("score").as("total_score"),
      avg("score").as("avg_score")
    )
    .orderBy("assigned_hour")
    .show(false)
  
  println("""
================================================================================
Test Complete!
================================================================================

You can now use GreedyAllocator functions:
  - allocate(df, hours, capacity)
  - allocateSimple(df, hours, capacityPerHour)
  - printStatistics(result)
  - quickTest(spark, dataPath, numUsers, capacityPerHour)

Try different configurations:
  val df2 = dfAll.filter("svc_mgmt_num like '%00'")
  val result2 = allocate(df2, hours, capacity)
  printStatistics(result2)

================================================================================
""")
}
