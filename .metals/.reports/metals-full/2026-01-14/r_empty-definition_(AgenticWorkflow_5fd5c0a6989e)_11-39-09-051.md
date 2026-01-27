file://<WORKSPACE>/optimize_send_time/load_and_test.scala
empty definition using pc, found symbol in pc: 
semanticdb not found
empty definition using fallback
non-local guesses:

offset: 1709
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
  val df = spark.read.parquet(sampleDataPath).cache()
  val totalRecords = df.select("svc_mgmt_num").distinct().count()
  println(s"✓ Loaded $totalRecords records")
  
  println(s"Total records: ${df.count()}")

  val safeBatchSize = (totalRecords*1.1).toInt
  println(s"Safe batch size: ${numFormatter.format(s@@afeBatchSize)}")

  // 설정
  val userCnt = df.select("svc_mgmt_num").distinct().count()
  val capacityPerHour = 10000

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
  val maxIterations = 100000
  val initialTemp = 1000.0
  val coolingRate = 0.9995
  val batchSize = safeBatchSize

  // 실행
  val result = allocateUsersWithSimulatedAnnealing(
    df = df,
    capacityPerHour = capacityPerHourMap,
    maxIterations = maxIterations,
    initialTemperature = initialTemp,
    coolingRate = coolingRate,
    batchSize = batchSize
  )

  // 결과 확인
  result.show(20, false)
  
  println("""
================================================================================
Test Complete!
================================================================================

Available functions:
  - allocateGreedySimple(df, hours, capacity)
  - allocateUsersWithSimulatedAnnealing(df, capacity, maxIterations, ...)
  - allocateUsersWithHourlyCapacity(df, capacity, timeLimit, ...)
  - allocateLargeScaleHybrid(df, capacity, batchSize, ...)

Try more tests:
  val df2 = dfAll.limit(10000)
  val result2 = allocateGreedySimple(df2, Array(9,10,11,12,13,14,15,16,17,18), 
                  Map(9->1000, 10->1000, 11->1000, 12->1000, 13->1000,
                      14->1000, 15->1000, 16->1000, 17->1000, 18->1000))

For more examples: QUICK_START.md
================================================================================
""")
}

```


#### Short summary: 

empty definition using pc, found symbol in pc: 