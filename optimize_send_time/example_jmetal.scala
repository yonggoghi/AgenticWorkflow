/**
 * jMetal 기반 최적화 사용 예제
 * 
 * 실행 방법:
 * 1. jMetal JAR 다운로드: ./download_jmetal.sh
 * 2. Spark shell 실행: 
 *    spark-shell --jars lib/*.jar
 * 3. 이 파일 로드:
 *    :load example_jmetal.scala
 */

import org.apache.spark.sql.SparkSession
import OptimizeSendTime._

// 데이터 로드
println("Loading data...")
val dfAll = spark.read.parquet("aos/sto/propensityScoreDF").cache()
println(s"Total records: ${dfAll.count()}")

// 샘플 데이터 (테스트용)
val df = dfAll.filter("svc_mgmt_num like '%00'")
println(s"Sample records: ${df.count()}")
println(s"Unique users: ${df.select("svc_mgmt_num").distinct().count()}")

// 시간대별 용량 설정
val capacityPerHour = Map(
  9 -> 10000,
  10 -> 10000,
  11 -> 10000,
  12 -> 10000,
  13 -> 10000,
  14 -> 10000,
  15 -> 10000,
  16 -> 10000,
  17 -> 10000,
  18 -> 10000
)

println("\n" + "=" * 80)
println("Example 1: NSGA-II Algorithm")
println("=" * 80)

val resultNSGAII = allocateUsersWithJMetalNSGAII(
  df = df,
  capacityPerHour = capacityPerHour,
  populationSize = 100,
  maxEvaluations = 25000,
  crossoverProbability = 0.9,
  mutationProbability = 0.1
)

println("\nResult summary:")
resultNSGAII.groupBy("assigned_hour")
  .agg(
    count("*").as("count"),
    sum("score").as("total_score"),
    avg("score").as("avg_score")
  )
  .orderBy("assigned_hour")
  .show()

println("\n" + "=" * 80)
println("Example 2: MOEA/D Algorithm")
println("=" * 80)

val resultMOEAD = allocateUsersWithJMetalMOEAD(
  df = df,
  capacityPerHour = capacityPerHour,
  populationSize = 100,
  maxEvaluations = 25000,
  neighborhoodSize = 20
)

println("\nResult summary:")
resultMOEAD.groupBy("assigned_hour")
  .agg(
    count("*").as("count"),
    sum("score").as("total_score"),
    avg("score").as("avg_score")
  )
  .orderBy("assigned_hour")
  .show()

println("\n" + "=" * 80)
println("Example 3: Hybrid (jMetal + Greedy)")
println("=" * 80)

val resultHybrid = allocateUsersHybridJMetal(
  df = df,
  capacityPerHour = capacityPerHour,
  algorithm = "NSGAII",
  populationSize = 100,
  maxEvaluations = 25000
)

println("\nResult summary:")
resultHybrid.groupBy("assigned_hour")
  .agg(
    count("*").as("count"),
    sum("score").as("total_score"),
    avg("score").as("avg_score")
  )
  .orderBy("assigned_hour")
  .show()

println("\n" + "=" * 80)
println("Comparison: NSGA-II vs MOEA/D vs Hybrid")
println("=" * 80)

def printAlgorithmStats(name: String, result: org.apache.spark.sql.DataFrame): Unit = {
  val totalScore = result.agg(sum("score")).first().getDouble(0)
  val avgScore = result.agg(avg("score")).first().getDouble(0)
  val count = result.count()
  
  // 시간대별 표준편차 계산
  val hourCounts = result.groupBy("assigned_hour").count().collect().map(_.getLong(1))
  val avgCount = hourCounts.sum.toDouble / hourCounts.length
  val variance = hourCounts.map(c => Math.pow(c - avgCount, 2)).sum / hourCounts.length
  val stdDev = Math.sqrt(variance)
  
  println(f"$name%-20s Total Score: $totalScore%,.2f  Avg Score: $avgScore%.4f  Load StdDev: $stdDev%.2f  Assigned: $count%,d")
}

printAlgorithmStats("NSGA-II", resultNSGAII)
printAlgorithmStats("MOEA/D", resultMOEAD)
printAlgorithmStats("Hybrid", resultHybrid)

println("\n" + "=" * 80)
println("Examples completed!")
println("=" * 80)

println("""
To run on full dataset:
  val resultFull = allocateLargeScaleJMetal(
    df = dfAll,
    capacityPerHour = capacityPerHour,
    batchSize = 50000,
    algorithm = "NSGAII",
    populationSize = 100,
    maxEvaluations = 25000
  )
  
To compare with other algorithms:
  val resultOR = allocateUsersWithHourlyCapacity(df, capacityPerHour)
  val resultSA = allocateUsersWithSimulatedAnnealing(df, capacityPerHour)
  val resultGreedy = allocateGreedySimple(df, Array(9,10,11,12,13,14,15,16,17,18), capacityPerHour)
""")
