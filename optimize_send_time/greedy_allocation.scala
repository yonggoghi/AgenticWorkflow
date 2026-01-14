import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import scala.collection.mutable
import java.text.DecimalFormat

// ============================================================================
// Data Structure
// ============================================================================

case class GreedyAllocationResult(
  svc_mgmt_num: String,
  assigned_hour: Int,
  score: Double
)

/**
 * Greedy 기반 최적 발송 시간 할당
 * 
 * 특징:
 * - 빠른 실행 속도 (대규모 데이터 처리 가능)
 * - 직관적인 로직 (최고 점수 우선 할당)
 * - 용량 제약 준수
 * 
 * 사용법:
 *   spark-shell
 *   scala> :load greedy_allocation.scala
 *   scala> import GreedyAllocator._
 */
object GreedyAllocator {

  val numFormatter = new DecimalFormat("#,###")

  /**
   * DataFrame에서 사용자 데이터 수집
   */
  def collectUserData(df: DataFrame): Map[String, Map[Int, Double]] = {
    df.collect()
      .groupBy(_.getAs[String]("svc_mgmt_num"))
      .map { case (userId, rows) =>
        userId -> rows.map { row =>
          row.getAs[Int]("send_hour") -> row.getAs[Double]("propensity_score")
        }.toMap
      }
      .toMap
  }

  /**
   * Greedy 할당 (기본)
   * 
   * 알고리즘:
   * 1. 모든 사용자를 최고 점수 순으로 정렬
   * 2. 각 사용자에 대해:
   *    - 가능한 시간대 중 점수가 가장 높은 시간대 선택
   *    - 용량이 남아있으면 할당
   * 
   * @param df 입력 DataFrame (svc_mgmt_num, send_hour, propensity_score)
   * @param hours 가능한 시간대 배열
   * @param capacity 시간대별 용량 Map
   * @return 할당 결과 DataFrame
   */
  def allocate(
    df: DataFrame,
    hours: Array[Int],
    capacity: Map[Int, Int]
  ): DataFrame = {
    
    import df.sparkSession.implicits._
    
    println("\n" + "=" * 80)
    println("Greedy Allocation")
    println("=" * 80)
    
    println("\nInitial capacity:")
    capacity.toSeq.sortBy(_._1).foreach { case (hour, cap) =>
      println(s"  Hour $hour: ${numFormatter.format(cap)}")
    }
    
    val userData = collectUserData(df)
    val users = userData.keys.toArray
    
    println(s"\nUsers to assign: ${numFormatter.format(users.length)}")
    
    // 시작 시간
    val startTime = System.currentTimeMillis()
    
    // 사용자를 최고 점수 순으로 정렬
    val userBestScores = users.map { user =>
      val bestScore = userData(user).values.max
      (user, bestScore)
    }.sortBy(-_._2)
    
    val hourCapacity = mutable.Map(capacity.toSeq: _*)
    val assignments = mutable.ArrayBuffer[GreedyAllocationResult]()
    
    // 각 사용자에 대해 최선의 시간대 할당
    for ((user, _) <- userBestScores) {
      val choices = userData(user).toSeq.sortBy(-_._2)  // 점수 높은 순
      
      var assigned = false
      for ((hour, score) <- choices if !assigned) {
        val currentCapacity = hourCapacity.getOrElse(hour, 0)
        
        if (currentCapacity > 0) {
          assignments += GreedyAllocationResult(user, hour, score)
          hourCapacity(hour) = currentCapacity - 1
          assigned = true
        }
      }
    }
    
    val elapsedTime = (System.currentTimeMillis() - startTime) / 1000.0
    
    println(s"\nGreedy assigned: ${numFormatter.format(assignments.size)} / ${numFormatter.format(users.length)}")
    println(f"Execution time: $elapsedTime%.2f seconds")
    
    if (assignments.nonEmpty) {
      val totalScore = assignments.map(_.score).sum
      val avgScore = totalScore / assignments.size
      
      println(f"\nTotal score: $totalScore%,.2f")
      println(f"Average score: $avgScore%.4f")
      
      println("\n[ALLOCATION BY HOUR]")
      val hourlyAssignment = assignments.groupBy(_.assigned_hour).mapValues(_.length)
      
      hours.sorted.foreach { hour =>
        val assigned = hourlyAssignment.getOrElse(hour, 0)
        val initialCap = capacity.getOrElse(hour, 0)
        val remaining = hourCapacity.getOrElse(hour, 0)
        val utilizationPct = if (initialCap > 0) assigned.toDouble / initialCap * 100 else 0.0
        
        println(f"  Hour $hour: assigned=${numFormatter.format(assigned).padTo(8, ' ')}, " +
          f"capacity=${numFormatter.format(initialCap).padTo(8, ' ')}, " +
          f"remaining=${numFormatter.format(remaining).padTo(8, ' ')} " +
          f"(${utilizationPct}%5.1f%%)")
      }
      
      println("=" * 80 + "\n")
    }
    
    if (assignments.isEmpty) {
      df.sparkSession.emptyDataFrame
    } else {
      assignments.toSeq.toDF()
    }
  }

  /**
   * Greedy 할당 (간단 버전 - 시간대별 동일 용량)
   * 
   * @param df 입력 DataFrame
   * @param hours 가능한 시간대 배열
   * @param capacityPerHour 시간대당 용량 (모든 시간대 동일)
   * @return 할당 결과 DataFrame
   */
  def allocateSimple(
    df: DataFrame,
    hours: Array[Int],
    capacityPerHour: Int
  ): DataFrame = {
    val capacity = hours.map(h => h -> capacityPerHour).toMap
    allocate(df, hours, capacity)
  }

  /**
   * 할당 결과 통계 출력
   */
  def printStatistics(result: DataFrame): Unit = {
    import result.sparkSession.implicits._
    
    println("\n" + "=" * 80)
    println("Allocation Statistics")
    println("=" * 80)
    
    val totalAssigned = result.count()
    val totalScore = result.agg(sum("score")).first().getDouble(0)
    val avgScore = totalScore / totalAssigned
    
    println(s"\nTotal assigned: ${numFormatter.format(totalAssigned)} users")
    println(f"Total score: $totalScore%,.2f")
    println(f"Average score: $avgScore%.4f")
    
    println("\nHour-wise allocation:")
    result.groupBy("assigned_hour")
      .agg(
        count("*").as("count"),
        sum("score").as("total_score"),
        avg("score").as("avg_score")
      )
      .orderBy("assigned_hour")
      .show(false)
    
    println("=" * 80)
  }

  /**
   * Quick test helper
   */
  def quickTest(
    spark: SparkSession,
    dataPath: String = "aos/sto/propensityScoreDF",
    numUsers: Int = 1000,
    capacityPerHour: Int = 100
  ): Unit = {
    
    import spark.implicits._
    
    println(s"\n${"=" * 80}")
    println("Quick Test - Greedy Allocation")
    println(s"${"=" * 80}")
    
    val dfAll = spark.read.parquet(dataPath).cache()
    val df = dfAll.limit(numUsers)
    
    val hours = Array(9, 10, 11, 12, 13, 14, 15, 16, 17, 18)
    val capacity = hours.map(h => h -> capacityPerHour).toMap
    
    val result = allocate(df, hours, capacity)
    
    println("\nTop 20 assignments:")
    result.show(20, false)
    
    printStatistics(result)
  }
}

// ============================================================================
// 초기화 메시지
// ============================================================================

println("""
================================================================================
Greedy Allocator - Fast User Allocation
================================================================================

✓ Loaded successfully!

To use:
  import GreedyAllocator._

Examples:

  // 1. Basic usage
  val df = spark.read.parquet("aos/sto/propensityScoreDF").cache()
  val hours = Array(9, 10, 11, 12, 13, 14, 15, 16, 17, 18)
  val capacity = Map(9->1000, 10->1000, 11->1000, 12->1000, 13->1000,
                      14->1000, 15->1000, 16->1000, 17->1000, 18->1000)
  val result = allocate(df, hours, capacity)

  // 2. Simple usage (same capacity for all hours)
  val result = allocateSimple(df, hours, 1000)

  // 3. Quick test
  quickTest(spark, "aos/sto/propensityScoreDF", numUsers=1000, capacityPerHour=100)

  // 4. Print statistics
  printStatistics(result)

================================================================================
""")
