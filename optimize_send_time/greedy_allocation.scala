import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window
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
   * Large-Scale Greedy 할당 (배치 처리)
   * 
   * 2500만명 같은 대규모 데이터를 위한 배치 처리
   * 점수 기반 분할로 품질 저하 최소화 (1-3%)
   * 
   * @param df 입력 DataFrame
   * @param hours 가능한 시간대 배열
   * @param capacity 시간대별 용량 Map
   * @param batchSize 배치 크기 (기본값: 500,000)
   * @return 할당 결과 DataFrame
   */
  def allocateLargeScale(
    df: DataFrame,
    hours: Array[Int],
    capacity: Map[Int, Int],
    batchSize: Int = 500000
  ): DataFrame = {
    
    import df.sparkSession.implicits._
    
    println("=" * 80)
    println("Large-Scale Greedy Allocation (Batch Processing)")
    println("=" * 80)
    
    // 전체 사용자 수 확인
    val totalUsers = df.select("svc_mgmt_num").distinct().count()
    val numBatches = Math.ceil(totalUsers.toDouble / batchSize).toInt
    
    println(s"\n[INPUT INFO]")
    println(s"Total users: ${numFormatter.format(totalUsers)}")
    println(s"Batch size: ${numFormatter.format(batchSize)}")
    println(s"Number of batches: $numBatches")
    
    println("\n[INITIAL CAPACITY]")
    capacity.toSeq.sortBy(_._1).foreach { case (hour, cap) =>
      println(s"  Hour $hour: ${numFormatter.format(cap)}")
    }
    val totalCapacity = capacity.values.sum
    println(s"Total capacity: ${numFormatter.format(totalCapacity)}")
    println(s"Capacity ratio: ${totalCapacity.toDouble / totalUsers}%.2fx")
    
    // 사용자별 최고 점수 계산 및 정렬 (Spark 작업)
    println("\nCalculating user priorities...")
    val userPriority = df.groupBy("svc_mgmt_num")
      .agg(max("propensity_score").as("max_score"))
      .withColumn("row_id", row_number().over(Window.orderBy(desc("max_score"))))
      .withColumn("batch_id", (($"row_id" - 1) / batchSize).cast("int"))
      .select("svc_mgmt_num", "batch_id", "max_score")
      .cache()
    
    println("\n[BATCH DISTRIBUTION]")
    val batchCounts = userPriority.groupBy("batch_id").count().collect()
      .map(r => r.getInt(0) -> r.getLong(1))
      .sortBy(_._1)
    
    batchCounts.foreach { case (bid, cnt) =>
      println(s"  Batch $bid: ${numFormatter.format(cnt)} users")
    }
    
    // 배치별 처리
    var remainingCapacity = capacity.toMap
    val allResults = mutable.ArrayBuffer[DataFrame]()
    var totalAssignedSoFar = 0L
    val startTime = System.currentTimeMillis()
    
    for (batchId <- 0 until numBatches) {
      println(s"\n${"=" * 80}")
      println(s"Processing Batch ${batchId + 1}/$numBatches")
      println(s"${"=" * 80}")
      
      val batchUsers = userPriority.filter($"batch_id" === batchId)
      val batchUserCount = batchUsers.count()
      
      // 용량이 남아있는 시간대만 처리
      val availableHours = remainingCapacity.filter(_._2 > 0).keys.toSeq
      
      if (availableHours.isEmpty) {
        println("⚠ No capacity left in any hour. Stopping.")
        val unassignedCount = totalUsers - totalAssignedSoFar
        println(s"Unassigned users: ${numFormatter.format(unassignedCount)}")
      } else {
        println(s"Batch users: ${numFormatter.format(batchUserCount)}")
        println(s"Available hours: ${availableHours.sorted.mkString(", ")}")
        
        println("\nRemaining capacity:")
        remainingCapacity.toSeq.sortBy(_._1).foreach { case (hour, cap) =>
          val status = if (availableHours.contains(hour)) "✓" else "✗"
          println(s"  Hour $hour: ${numFormatter.format(cap)} $status")
        }
        
        // 배치 데이터 준비
        val batchDf = df.join(batchUsers, Seq("svc_mgmt_num"))
          .filter($"send_hour".isin(availableHours: _*))
          .select("svc_mgmt_num", "send_hour", "propensity_score")
        
        val batchStartTime = System.currentTimeMillis()
        
        // 배치 할당 (메모리 내 처리)
        val batchResult = allocate(batchDf, hours, remainingCapacity)
        
        val batchTime = (System.currentTimeMillis() - batchStartTime) / 1000.0
        
        val assignedCount = batchResult.count()
        
        if (assignedCount > 0) {
          totalAssignedSoFar += assignedCount
          
          // 용량 차감
          val allocatedPerHour = batchResult.groupBy("assigned_hour").count().collect()
            .map(row => row.getInt(0) -> row.getLong(1).toInt).toMap
          
          println(s"\n[CAPACITY UPDATE]")
          hours.sorted.foreach { hour =>
            val allocated = allocatedPerHour.getOrElse(hour, 0)
            val before = remainingCapacity.getOrElse(hour, 0)
            val after = Math.max(0, before - allocated)
            
            if (allocated > 0) {
              println(f"  Hour $hour: ${numFormatter.format(before)} - ${numFormatter.format(allocated)} = ${numFormatter.format(after)}")
            }
          }
          
          remainingCapacity = remainingCapacity.map { case (hour, cap) =>
            hour -> Math.max(0, cap - allocatedPerHour.getOrElse(hour, 0))
          }
          
          val batchScore = batchResult.agg(sum("score")).first().getDouble(0)
          println(f"\nBatch time: $batchTime%.2f seconds")
          println(f"Batch score: $batchScore%,.2f")
          println(s"Batch assigned: ${numFormatter.format(assignedCount)}")
          
          allResults += batchResult
        } else {
          println("⚠ No users assigned in this batch")
        }
      }
      
      // 진행률
      val progress = totalAssignedSoFar.toDouble / totalUsers * 100
      val coverageVsCapacity = totalAssignedSoFar.toDouble / totalCapacity * 100
      println(f"\n[PROGRESS]")
      println(f"  Assigned: ${numFormatter.format(totalAssignedSoFar)} / ${numFormatter.format(totalUsers)} users ($progress%.1f%%)")
      println(f"  Capacity used: ${numFormatter.format(totalAssignedSoFar)} / ${numFormatter.format(totalCapacity)} ($coverageVsCapacity%.1f%%)")
    }
    
    userPriority.unpersist()
    
    val totalTime = (System.currentTimeMillis() - startTime) / 1000.0
    val totalMinutes = totalTime / 60.0
    
    // 최종 결과
    if (allResults.isEmpty) {
      println("\n⚠ No results generated!")
      return df.sparkSession.emptyDataFrame
    }
    
    val finalResult = allResults.reduce(_.union(_))
    
    println(s"\n${"=" * 80}")
    println("Large-Scale Allocation Complete")
    println(s"${"=" * 80}")
    println(f"Total execution time: $totalTime%.2f seconds ($totalMinutes%.2f minutes)")
    
    printFinalStatistics(finalResult, totalUsers, totalCapacity)
    
    finalResult
  }

  /**
   * 최종 통계 출력 (Large-Scale)
   */
  def printFinalStatistics(result: DataFrame, totalUsers: Long, totalCapacity: Long): Unit = {
    println(s"\n${"=" * 80}")
    println("Final Allocation Statistics")
    println(s"${"=" * 80}")
    
    val totalAssigned = result.count()
    val coverage = totalAssigned.toDouble / totalUsers * 100
    val capacityUtil = totalAssigned.toDouble / totalCapacity * 100
    
    println(s"\nTotal assigned: ${numFormatter.format(totalAssigned)} / ${numFormatter.format(totalUsers)} ($coverage%.2f%%)")
    println(f"Capacity utilization: ${numFormatter.format(totalAssigned)} / ${numFormatter.format(totalCapacity)} ($capacityUtil%.2f%%)")
    
    if (totalAssigned > 0) {
      val totalScore = result.agg(sum("score")).first().getDouble(0)
      val avgScore = totalScore / totalAssigned
      
      println(f"\nTotal score: $totalScore%,.2f")
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
    }
    
    println("=" * 80)
  }
}

// ============================================================================
// 초기화 메시지
// ============================================================================

println("""
================================================================================
Greedy Allocator - Large-Scale User Allocation (Batch Processing)
================================================================================

✓ Loaded successfully!

To use:
  import GreedyAllocator._

Available Functions:
  1. allocate(df, hours, capacity)              - Internal: Single batch allocation
  2. allocateLargeScale(df, hours, capacity, batchSize) - ⭐ Main: Large-scale batch allocation
  3. printFinalStatistics(result, totalUsers, totalCapacity) - Print statistics
  4. collectUserData(df)                        - Internal: Collect user data

Primary Usage (2500만명 대규모 처리):

  // 1. Load data
  val df = spark.read.parquet("aos/sto/propensityScoreDF").cache()
  val totalUsers = df.select("svc_mgmt_num").distinct().count()

  // 2. Set up hours and capacity
  val hours = Array(9, 10, 11, 12, 13, 14, 15, 16, 17, 18)
  val capacityPerHour = (totalUsers * 0.11).toInt
  val capacity = hours.map(h => h -> capacityPerHour).toMap

  // 3. Run large-scale allocation
  val result = allocateLargeScale(
    df = df,
    hours = hours,
    capacity = capacity,
    batchSize = 1000000  // 100만명씩 배치 (default: 500,000)
  )

  // 4. Save results (optional)
  result.write.mode("overwrite").parquet("output/allocation_result")

Performance Tips:
  - Batch size: 500K-2M users per batch
  - Memory: Use --driver-memory 100g for 25M users
  - For 25M users: ~1 hour, quality loss 1-3%

================================================================================
""")
