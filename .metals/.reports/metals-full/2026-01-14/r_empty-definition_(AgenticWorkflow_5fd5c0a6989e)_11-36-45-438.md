file://<WORKSPACE>/optimize_send_time/load_and_test.scala
empty definition using pc, found symbol in pc: 
semanticdb not found
empty definition using fallback
non-local guesses:

offset: 1461
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
  val tot@@alRecords = df.count()
  println(s"✓ Loaded $totalRecords records")
  
  // 6. 용량 설정
  val capacity = Map(
    9 -> 100, 10 -> 100, 11 -> 100, 12 -> 100, 13 -> 100,
    14 -> 100, 15 -> 100, 16 -> 100, 17 -> 100, 18 -> 100
  )
  
  println("\n" + "=" * 80)
  println("Running Greedy Allocation Test")
  println("=" * 80)
  
  // 7. Greedy 테스트 실행
  val result = allocateGreedySimple(df, Array(9,10,11,12,13,14,15,16,17,18), capacity)
  
  println("\nResults by hour:")
  result.groupBy("assigned_hour")
    .count()
    .orderBy("assigned_hour")
    .show()
  
  println("\nTop 10 assignments:")
  result.show(10, false)
  
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