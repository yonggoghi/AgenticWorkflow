file://<WORKSPACE>/optimize_send_time/load_and_test.scala
empty definition using pc, found symbol in pc: 
semanticdb not found
empty definition using fallback
non-local guesses:

offset: 1566
uri: file://<WORKSPACE>/optimize_send_time/load_and_test.scala
text:
```scala
// ============================================================================
// Quick Test Script for Spark Shell
// ============================================================================
// 
// Usage:
//   spark-shell -i load_and_test.scala
//   또는
//   spark-shell
//   scala> :load load_and_test.scala
// ============================================================================

println("""
================================================================================
Loading Optimize Send Time...
================================================================================
""")

// 1. optimize_ost.scala 로드
:load optimize_ost.scala

// 2. 함수 import
import OptimizeSendTime._

println("""
================================================================================
Running Quick Test
================================================================================
""")

// 3. 샘플 데이터 확인
val sampleDataPath = "data/sample/propensityScoreDF"
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
  ./generate_data_simple.sh 1000

Then run this script again.
================================================================================
""")
} else {
  // 4. 데이터 로드
  println("Loading sample data...")
  val dfAll = spark.read.parquet(sampleDataPath).cache()

  val df = dfAll.filter("svc_mgmt_num like '%0'")
  val totalRecords = df.select("svc_mgmt_num").distinct().co@@unt()
  println(s"✓ Loaded $totalRecords records")
  
  println(s"Total records: ${df.count()}")

  val safeBatchSize = (totalRecords*1.1).toInt
  println(s"Safe batch size: ${numFormatter.format(safeBatchSize)}")

  // 설정
  val userCnt = df.select("svc_mgmt_num").distinct().count()
  val capacityPerHour = (totalRecords*0.11).toInt

  println(s"Capacity per hour: ${numFormatter.format(capacityPerHour)}")

  // 시간대별 용량 맵 생성 (9시~18시)
  val capacityPerHourMap = Map(
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

  // SA 알고리즘 파라미터
  val maxIterations = 2000000
  val initialTemp = 1000.0  // 적응형 온도 미사용 시에만 적용
  val coolingRate = 0.9995
  val batchSize = safeBatchSize
  
  // 개선 기능 설정
  val useAdaptiveTemp = true      // 적응형 온도 스케일 (점수의 10%)
  val reheatingEnabled = true     // 재가열 메커니즘
  val reheatingThreshold = 1000   // 1000회 개선 없으면 재가열

  println("\n" + "=" * 80)
  println("Starting Simulated Annealing Optimization (Enhanced)")
  println("=" * 80)
  println(s"Max iterations: ${numFormatter.format(maxIterations)}")
  println(s"Cooling rate: $coolingRate")
  println(s"Adaptive temperature: ${if (useAdaptiveTemp) "Enabled (auto-scaling)" else s"Disabled (fixed: $initialTemp)"}")
  println(s"Reheating: ${if (reheatingEnabled) s"Enabled (threshold: $reheatingThreshold)" else "Disabled"}")
  println("=" * 80 + "\n")

  // 실행 시간 측정
  val startTime = System.currentTimeMillis()
  
  val result = allocateUsersWithSimulatedAnnealing(
    df = df,
    capacityPerHour = capacityPerHourMap,
    maxIterations = maxIterations,
    initialTemperature = initialTemp,
    coolingRate = coolingRate,
    batchSize = batchSize,
    useAdaptiveTemp = useAdaptiveTemp,
    reheatingEnabled = reheatingEnabled,
    reheatingThreshold = reheatingThreshold
  )
  
  val endTime = System.currentTimeMillis()
  val totalTimeSeconds = (endTime - startTime) / 1000.0
  val totalTimeMinutes = totalTimeSeconds / 60.0

  println("\n" + "=" * 80)
  println("Optimization Complete!")
  println("=" * 80)
  println(f"Total execution time: $totalTimeSeconds%.2f seconds ($totalTimeMinutes%.2f minutes)")
  println("=" * 80 + "\n")

  // 결과 확인
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
  
}

```


#### Short summary: 

empty definition using pc, found symbol in pc: 