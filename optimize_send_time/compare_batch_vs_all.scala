// ============================================================================
// Batch vs All-at-Once Comparison Test
// ============================================================================
// 
// 배치 처리(allocateLargeScale) vs 일괄 처리(allocate) 비교
// 
// 비교 항목:
//   1. 실행 시간
//   2. 총 점수 (품질)
//   3. 평균 점수
//   4. 시간대별 할당 분포
//   5. 품질 저하율
//
// Usage:
//   spark-shell --driver-memory 8g --executor-memory 8g -i compare_batch_vs_all.scala
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
import java.text.DecimalFormat

println("""
================================================================================
Batch vs All-at-Once Comparison Test
================================================================================

This test will compare:
  - Batch processing (allocateLargeScale)
  - All-at-once processing (allocate)

Metrics:
  ✓ Execution time
  ✓ Total score
  ✓ Average score
  ✓ Quality degradation
  ✓ Allocation distribution

================================================================================
""")

// 3. 테스트 설정
val dataPath = "aos/sto/propensityScoreDF"

// 데이터 존재 확인
val testDataExists = try {
  spark.read.parquet(dataPath).limit(1).count()
  true
} catch {
  case _: Exception => false
}

if (!testDataExists) {
  println("""
⚠ Data not found!

Please generate test data first:
  ./generate_data_simple.sh 100000

Then run this script again.
================================================================================
""")
} else {
  // 4. 테스트 데이터 로드
  println("\n[DATA LOADING]")
  println("=" * 80)
  
  val dfAll = spark.read.parquet(dataPath).cache()
  
  // 테스트 규모 설정 (사용자 수 기반)
  // 너무 많으면 일괄 처리가 메모리 부족으로 실패할 수 있음
  val testUsers = Array(10000, 50000, 100000, 200000, 500000)
  
  println("Available test sizes: " + testUsers.map(numFormatter.format).mkString(", "))
  
  // 사용 가능한 최대 테스트 크기 선택
  val totalAvailable = dfAll.select("svc_mgmt_num").distinct().count()
  println(s"Total users available: ${numFormatter.format(totalAvailable)}")
  
  val selectedTestSize = testUsers.filter(_ <= totalAvailable).lastOption.getOrElse(10000)
  println(s"\nSelected test size: ${numFormatter.format(selectedTestSize)} users")
  println("(Use smaller size to ensure both methods can run)")
  
  val testDf = dfAll.limit(selectedTestSize * 10).cache()  // 10 hours per user
  val actualUsers = testDf.select("svc_mgmt_num").distinct().count()
  
  println(s"✓ Loaded: ${numFormatter.format(actualUsers)} users")
  println(s"✓ Records: ${numFormatter.format(testDf.count())}")
  
  // 5. 공통 설정
  val hours = Array(9, 10, 11, 12, 13, 14, 15, 16, 17, 18)
  val capacityPerHour = (actualUsers * 0.11).toInt
  val capacity = hours.map(h => h -> capacityPerHour).toMap
  val totalCapacity = capacity.values.sum
  
  println(s"\n[TEST CONFIGURATION]")
  println("=" * 80)
  println(s"Hours: ${hours.mkString(", ")}")
  println(s"Capacity per hour: ${numFormatter.format(capacityPerHour)}")
  println(s"Total capacity: ${numFormatter.format(totalCapacity)}")
  println(f"Capacity ratio: ${totalCapacity.toDouble / actualUsers}%.2fx")
  
  // 배치 크기 설정
  val batchSize = if (actualUsers > 100000) 50000 else 20000
  val numBatches = Math.ceil(actualUsers.toDouble / batchSize).toInt
  println(s"\nBatch size: ${numFormatter.format(batchSize)}")
  println(s"Number of batches: $numBatches")
  
  println("\n" + "=" * 80)
  println("Starting Comparison Test...")
  println("=" * 80)
  
  // ============================================================================
  // Test 1: All-at-Once (일괄 처리)
  // ============================================================================
  
  println("\n" + "█" * 80)
  println("█" + " " * 78 + "█")
  println("█" + "  TEST 1: All-at-Once Processing".padTo(78, ' ') + "█")
  println("█" + " " * 78 + "█")
  println("█" * 80)
  
  val allStartTime = System.currentTimeMillis()
  var allResult: DataFrame = null
  var allSuccess = false
  var allError = ""
  
  try {
    allResult = allocate(testDf, hours, capacity)
    allSuccess = true
  } catch {
    case e: OutOfMemoryError =>
      allError = "OutOfMemoryError"
      println(s"\n❌ All-at-Once FAILED: ${e.getMessage}")
      println("   (This is expected for large datasets)")
    case e: Exception =>
      allError = e.getClass.getSimpleName
      println(s"\n❌ All-at-Once FAILED: ${e.getMessage}")
  }
  
  val allEndTime = System.currentTimeMillis()
  val allTimeSeconds = (allEndTime - allStartTime) / 1000.0
  val allTimeMinutes = allTimeSeconds / 60.0
  
  // All-at-Once 결과
  val allStats = if (allSuccess) {
    val assigned = allResult.count()
    val totalScore = allResult.agg(sum("score")).first().getDouble(0)
    val avgScore = totalScore / assigned
    val coverage = assigned.toDouble / actualUsers * 100
    
    Map(
      "success" -> true,
      "time_seconds" -> allTimeSeconds,
      "assigned" -> assigned.toDouble,
      "total_score" -> totalScore,
      "avg_score" -> avgScore,
      "coverage" -> coverage
    )
  } else {
    Map(
      "success" -> false,
      "error" -> allError,
      "time_seconds" -> allTimeSeconds
    )
  }
  
  // ============================================================================
  // Test 2: Batch Processing (배치 처리)
  // ============================================================================
  
  println("\n" + "█" * 80)
  println("█" + " " * 78 + "█")
  println("█" + "  TEST 2: Batch Processing".padTo(78, ' ') + "█")
  println("█" + " " * 78 + "█")
  println("█" * 80)
  
  val batchStartTime = System.currentTimeMillis()
  var batchResult: DataFrame = null
  var batchSuccess = false
  var batchError = ""
  
  try {
    batchResult = allocateLargeScale(testDf, hours, capacity, batchSize)
    batchSuccess = true
  } catch {
    case e: OutOfMemoryError =>
      batchError = "OutOfMemoryError"
      println(s"\n❌ Batch Processing FAILED: ${e.getMessage}")
    case e: Exception =>
      batchError = e.getClass.getSimpleName
      println(s"\n❌ Batch Processing FAILED: ${e.getMessage}")
  }
  
  val batchEndTime = System.currentTimeMillis()
  val batchTimeSeconds = (batchEndTime - batchStartTime) / 1000.0
  val batchTimeMinutes = batchTimeSeconds / 60.0
  
  // Batch 결과
  val batchStats = if (batchSuccess) {
    val assigned = batchResult.count()
    val totalScore = batchResult.agg(sum("score")).first().getDouble(0)
    val avgScore = totalScore / assigned
    val coverage = assigned.toDouble / actualUsers * 100
    
    Map(
      "success" -> true,
      "time_seconds" -> batchTimeSeconds,
      "assigned" -> assigned.toDouble,
      "total_score" -> totalScore,
      "avg_score" -> avgScore,
      "coverage" -> coverage
    )
  } else {
    Map(
      "success" -> false,
      "error" -> batchError,
      "time_seconds" -> batchTimeSeconds
    )
  }
  
  // ============================================================================
  // Comparison Results
  // ============================================================================
  
  println("\n\n" + "=" * 80)
  println("=" * 80)
  println("  COMPARISON RESULTS")
  println("=" * 80)
  println("=" * 80)
  
  println(s"\n[TEST CONFIGURATION]")
  println(s"  Users: ${numFormatter.format(actualUsers)}")
  println(s"  Capacity: ${numFormatter.format(totalCapacity)} (${totalCapacity.toDouble / actualUsers}%.2fx)")
  println(s"  Batch size: ${numFormatter.format(batchSize)} (${numBatches} batches)")
  
  // 성공 여부
  println(s"\n[EXECUTION STATUS]")
  println("  ┌─────────────────────┬──────────┬─────────────────┐")
  println("  │ Method              │ Status   │ Time            │")
  println("  ├─────────────────────┼──────────┼─────────────────┤")
  
  val allStatusIcon = if (allSuccess) "✓" else "✗"
  val allStatusText = if (allSuccess) "Success" else s"Failed ($allError)"
  printf("  │ %-19s │ %-8s │ %6.2f sec      │\n", "All-at-Once", allStatusText, allTimeSeconds)
  
  val batchStatusIcon = if (batchSuccess) "✓" else "✗"
  val batchStatusText = if (batchSuccess) "Success" else s"Failed ($batchError)"
  printf("  │ %-19s │ %-8s │ %6.2f sec      │\n", "Batch Processing", batchStatusText, batchTimeSeconds)
  
  println("  └─────────────────────┴──────────┴─────────────────┘")
  
  // 상세 비교 (둘 다 성공한 경우만)
  if (allSuccess && batchSuccess) {
    println(s"\n[PERFORMANCE COMPARISON]")
    println("  ┌─────────────────────┬──────────────┬──────────────┬────────────┐")
    println("  │ Metric              │ All-at-Once  │ Batch        │ Difference │")
    println("  ├─────────────────────┼──────────────┼──────────────┼────────────┤")
    
    // 실행 시간
    val timeDiff = ((batchTimeSeconds - allTimeSeconds) / allTimeSeconds * 100)
    val timeDiffStr = if (timeDiff > 0) f"+$timeDiff%.1f%%" else f"$timeDiff%.1f%%"
    printf("  │ %-19s │ %7.2f sec  │ %7.2f sec  │ %10s │\n", 
      "Execution Time", allTimeSeconds, batchTimeSeconds, timeDiffStr)
    
    // 할당 수
    val allAssigned = allStats("assigned").asInstanceOf[Double].toLong
    val batchAssigned = batchStats("assigned").asInstanceOf[Double].toLong
    val assignedDiff = batchAssigned - allAssigned
    val assignedDiffStr = if (assignedDiff >= 0) f"+$assignedDiff%,d" else f"$assignedDiff%,d"
    printf("  │ %-19s │ %12s │ %12s │ %10s │\n",
      "Assigned Users", numFormatter.format(allAssigned), numFormatter.format(batchAssigned), assignedDiffStr)
    
    // 총 점수
    val allScore = allStats("total_score").asInstanceOf[Double]
    val batchScore = batchStats("total_score").asInstanceOf[Double]
    val scoreDiff = ((batchScore - allScore) / allScore * 100)
    val scoreDiffStr = if (scoreDiff >= 0) f"+$scoreDiff%.2f%%" else f"$scoreDiff%.2f%%"
    printf("  │ %-19s │ %12.2f │ %12.2f │ %10s │\n",
      "Total Score", allScore, batchScore, scoreDiffStr)
    
    // 평균 점수
    val allAvg = allStats("avg_score").asInstanceOf[Double]
    val batchAvg = batchStats("avg_score").asInstanceOf[Double]
    val avgDiff = ((batchAvg - allAvg) / allAvg * 100)
    val avgDiffStr = if (avgDiff >= 0) f"+$avgDiff%.2f%%" else f"$avgDiff%.2f%%"
    printf("  │ %-19s │ %12.6f │ %12.6f │ %10s │\n",
      "Average Score", allAvg, batchAvg, avgDiffStr)
    
    // 커버리지
    val allCov = allStats("coverage").asInstanceOf[Double]
    val batchCov = batchStats("coverage").asInstanceOf[Double]
    val covDiff = batchCov - allCov
    val covDiffStr = if (covDiff >= 0) f"+$covDiff%.2f%%" else f"$covDiff%.2f%%"
    printf("  │ %-19s │ %11.2f%% │ %11.2f%% │ %10s │\n",
      "Coverage", allCov, batchCov, covDiffStr)
    
    println("  └─────────────────────┴──────────────┴──────────────┴────────────┘")
    
    // 품질 분석
    println(s"\n[QUALITY ANALYSIS]")
    val qualityRatio = (batchScore / allScore * 100)
    val qualityLoss = 100 - qualityRatio
    
    println(f"  Quality retention: $qualityRatio%.3f%%")
    println(f"  Quality loss: $qualityLoss%.3f%%")
    
    if (qualityLoss < 1.0) {
      println("  ✓ Excellent: Less than 1% quality loss")
    } else if (qualityLoss < 3.0) {
      println("  ✓ Good: Less than 3% quality loss")
    } else if (qualityLoss < 5.0) {
      println("  ⚠ Acceptable: Less than 5% quality loss")
    } else {
      println("  ⚠ Warning: More than 5% quality loss")
    }
    
    // 시간 효율성
    println(s"\n[TIME EFFICIENCY]")
    val timeRatio = batchTimeSeconds / allTimeSeconds
    
    if (timeRatio < 1.2) {
      println(f"  ✓ Batch is similar speed (${timeRatio}%.2fx)")
    } else if (timeRatio < 2.0) {
      println(f"  ⚠ Batch is slower (${timeRatio}%.2fx)")
    } else {
      println(f"  ⚠ Batch is much slower (${timeRatio}%.2fx)")
    }
    
    // 시간대별 할당 분포 비교
    println(s"\n[ALLOCATION DISTRIBUTION BY HOUR]")
    println("  ┌──────┬──────────────┬──────────────┬────────────┐")
    println("  │ Hour │ All-at-Once  │ Batch        │ Difference │")
    println("  ├──────┼──────────────┼──────────────┼────────────┤")
    
    val allByHour = allResult.groupBy("assigned_hour").count().collect()
      .map(r => r.getInt(0) -> r.getLong(1)).toMap
    val batchByHour = batchResult.groupBy("assigned_hour").count().collect()
      .map(r => r.getInt(0) -> r.getLong(1)).toMap
    
    hours.sorted.foreach { hour =>
      val allCount = allByHour.getOrElse(hour, 0L)
      val batchCount = batchByHour.getOrElse(hour, 0L)
      val diff = batchCount - allCount
      val diffStr = if (diff >= 0) f"+$diff%,d" else f"$diff%,d"
      printf("  │  %2d  │ %12s │ %12s │ %10s │\n",
        hour, numFormatter.format(allCount), numFormatter.format(batchCount), diffStr)
    }
    
    println("  └──────┴──────────────┴──────────────┴────────────┘")
    
    // 결론
    println(s"\n[RECOMMENDATION]")
    
    if (actualUsers < 100000) {
      println(s"  For ${numFormatter.format(actualUsers)} users:")
      println("  → Use All-at-Once: Faster and simpler")
      println(f"    (Quality: 100%%, Time: ${allTimeSeconds}%.2f sec)")
    } else if (actualUsers < 1000000) {
      if (qualityLoss < 2.0 && timeRatio < 1.5) {
        println(s"  For ${numFormatter.format(actualUsers)} users:")
        println("  → Batch is acceptable: Good balance")
        println(f"    (Quality: ${qualityRatio}%.2f%%, Time: ${batchTimeSeconds}%.2f sec)")
      } else {
        println(s"  For ${numFormatter.format(actualUsers)} users:")
        println("  → Use All-at-Once if memory allows")
        println(f"    (Quality: 100%%, Time: ${allTimeSeconds}%.2f sec)")
      }
    } else {
      println(s"  For ${numFormatter.format(actualUsers)} users:")
      println("  → Use Batch: Only practical option")
      println(f"    (Quality: ${qualityRatio}%.2f%%, Time: ${batchTimeSeconds}%.2f sec)")
    }
    
  } else {
    // 하나라도 실패한 경우
    println(s"\n[ANALYSIS]")
    
    if (!allSuccess && batchSuccess) {
      println("  ✓ Batch processing succeeded while All-at-Once failed")
      println("  → This demonstrates the advantage of batch processing for large datasets")
      println(f"  → Batch completed in ${batchTimeSeconds}%.2f seconds")
    } else if (allSuccess && !batchSuccess) {
      println("  ⚠ All-at-Once succeeded but Batch processing failed")
      println("  → This is unexpected. Check implementation or data issues")
    } else {
      println("  ❌ Both methods failed")
      println("  → Check data, memory settings, or reduce test size")
    }
  }
  
  println("\n" + "=" * 80)
  println("=" * 80)
  println("Comparison Test Complete!")
  println("=" * 80 + "\n")
  
  // 캐시 정리
  testDf.unpersist()
  dfAll.unpersist()
  
  println("""
To run with different sizes:
  1. Generate more data: ./generate_data_simple.sh 1000000
  2. Edit this script to change testUsers array
  3. Rerun: spark-shell --driver-memory 16g -i compare_batch_vs_all.scala

================================================================================
""")
}
