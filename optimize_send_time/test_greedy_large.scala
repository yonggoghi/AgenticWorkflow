// ============================================================================
// Large-Scale Greedy Test (2500만명)
// ============================================================================
// 
// Usage:
//   spark-shell --driver-memory 8g --executor-memory 8g -i test_greedy_large.scala
// ============================================================================

println("""
================================================================================
Loading Greedy Allocator (Large-Scale)...
================================================================================
""")

// 1. greedy_allocation.scala 로드
:load greedy_allocation.scala

// 2. 함수 import
import GreedyAllocator._

println("""
================================================================================
Running Large-Scale Greedy Test (2500만명 대응)
================================================================================
""")

// 3. 데이터 경로
val dataPath = "aos/sto/propensityScoreDF"

val dataExists = try {
  spark.read.parquet(dataPath).limit(1).count()
  true
} catch {
  case _: Exception => false
}

if (!dataExists) {
  println("""
⚠ Data not found!

Please generate data first:
  ./generate_data_simple.sh 25000000

Then run this script again.
================================================================================
""")
} else {
  // 4. 데이터 로드
  println("Loading data...")
  val dfAll = spark.read.parquet(dataPath).cache()
  
  // 전체 사용자 수 확인
  val totalUsers = dfAll.select("svc_mgmt_num").distinct().count()
  println(s"✓ Total unique users: ${numFormatter.format(totalUsers)}")
  
  val totalRecords = dfAll.count()
  println(s"✓ Total records: ${numFormatter.format(totalRecords)}")
  
  // 5. 용량 설정 (시간대당 총 사용자의 11%)
  val capacityPerHour = (totalUsers * 0.11).toInt
  println(s"✓ Capacity per hour: ${numFormatter.format(capacityPerHour)}")
  
  val hours = Array(9, 10, 11, 12, 13, 14, 15, 16, 17, 18)
  val capacity = hours.map(h => h -> capacityPerHour).toMap
  
  val totalCapacity = capacity.values.sum
  println(s"✓ Total capacity: ${numFormatter.format(totalCapacity)}")
  println(f"✓ Capacity ratio: ${totalCapacity.toDouble / totalUsers}%.2fx")
  
  // 6. 배치 크기 설정
  val batchSize = if (totalUsers > 10000000) {
    1000000  // 1000만명 이상: 100만 배치
  } else if (totalUsers > 1000000) {
    500000   // 100만-1000만: 50만 배치
  } else {
    20000   // 100만 이하: 10만 배치
  }
  
  println(s"✓ Batch size: ${numFormatter.format(batchSize)}")
  val numBatches = Math.ceil(totalUsers.toDouble / batchSize).toInt
  println(s"✓ Number of batches: $numBatches")
  
  println("\n" + "=" * 80)
  println("Starting Large-Scale Greedy Allocation")
  println("=" * 80)
  println(s"This will process ${numFormatter.format(totalUsers)} users in $numBatches batches")
  println(f"Estimated time: ${totalUsers / 1000000.0 * 2}%.1f - ${totalUsers / 1000000.0 * 5}%.1f minutes")
  println("=" * 80 + "\n")
  
  // 7. 실행 시간 측정
  val startTime = System.currentTimeMillis()
  
  val result = allocateLargeScale(dfAll, hours, capacity, batchSize)
  
  val endTime = System.currentTimeMillis()
  val totalTimeSeconds = (endTime - startTime) / 1000.0
  val totalTimeMinutes = totalTimeSeconds / 60.0
  
  println("\n" + "=" * 80)
  println("Allocation Complete!")
  println("=" * 80)
  println(f"Total execution time: $totalTimeSeconds%.2f seconds ($totalTimeMinutes%.2f minutes)")
  println(f"Throughput: ${totalUsers / totalTimeSeconds}%.0f users/second")
  println("=" * 80 + "\n")
  
  // 8. 결과 확인
  println("Sample assignments (first 20):")
  result.show(20, false)
  
  // 9. 결과 저장 (옵션)
  println("\nTo save results:")
  result.write.mode("overwrite").parquet("aos/sto/allocation_result")
  
  println("""
================================================================================
Large-Scale Test Complete!
================================================================================

Performance Tips:
  1. For 2500만명: Use --driver-memory 16g --executor-memory 16g
  2. Batch size: Adjust based on memory (500K - 2M)
  3. Save results: result.write.parquet("output/result")

Next steps:
  - Compare with SA: :load load_and_test.scala
  - Analyze results: printStatistics(result)
  - Export to CSV: result.coalesce(1).write.csv("output/result.csv")

================================================================================
""")
}
