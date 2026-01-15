import org.apache.spark.sql.{DataFrame, SparkSession, Row}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import com.google.ortools.Loader
import com.google.ortools.linearsolver.{MPConstraint, MPObjective, MPSolver, MPVariable}
import scala.collection.mutable
import java.text.DecimalFormat
import org.apache.spark.sql.expressions.Window
import scala.util.Random

// jMetal imports
import org.uma.jmetal.algorithm.multiobjective.nsgaii.NSGAIIBuilder
import org.uma.jmetal.operator.impl.crossover.IntegerSBXCrossover
import org.uma.jmetal.operator.impl.mutation.IntegerPolynomialMutation
import org.uma.jmetal.operator.impl.selection.BinaryTournamentSelection
import org.uma.jmetal.problem.Problem
import org.uma.jmetal.problem.IntegerProblem
import org.uma.jmetal.solution.IntegerSolution
import org.uma.jmetal.util.comparator.RankingAndCrowdingDistanceComparator
import java.util.{List => JavaList, ArrayList => JavaArrayList}
import scala.collection.JavaConverters._

// ============================================================================
// Data Structures (must be outside object for Spark encoders)
// ============================================================================

case class AllocationResult(
  svc_mgmt_num: String,
  assigned_hour: Int,
  score: Double
)

case class Solution(
  assignments: Map[String, Int],  // userId -> hour
  score: Double,
  hourUsage: Map[Int, Int]        // hour -> count
) {
  def isValid(capacityPerHour: Map[Int, Int]): Boolean = {
    hourUsage.forall { case (hour, count) =>
      count <= capacityPerHour.getOrElse(hour, 0)
    }
  }
}

/**
 * 최적 발송 시간 할당 시스템
 * Google OR-Tools 및 Simulated Annealing 기반 최적화
 */
object OptimizeSendTime {

  // ============================================================================
  // Helper Functions
  // ============================================================================

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
   * 의사결정 변수 생성
   */
  def createVariables(
    solver: MPSolver,
    users: Array[String],
    hours: Array[Int],
    userData: Map[String, Map[Int, Double]]
  ): mutable.Map[(String, Int), MPVariable] = {
    
    val variables = mutable.Map[(String, Int), MPVariable]()
    
    for {
      user <- users
      hour <- hours
      if userData(user).contains(hour)
    } {
      val varName = s"x_${user}_$hour"
      variables((user, hour)) = solver.makeBoolVar(varName)
    }
    
    variables
  }

  /**
   * 목적 함수 설정
   */
  def setObjective(
    solver: MPSolver,
    variables: mutable.Map[(String, Int), MPVariable],
    userData: Map[String, Map[Int, Double]]
  ): Unit = {
    
    val objective: MPObjective = solver.objective()
    
    for (((user, hour), variable) <- variables) {
      val score = userData(user)(hour)
      objective.setCoefficient(variable, score)
    }
    
    objective.setMaximization()
  }

  /**
   * 제약 조건 추가
   */
  def addConstraints(
    solver: MPSolver,
    variables: mutable.Map[(String, Int), MPVariable],
    users: Array[String],
    hours: Array[Int],
    capacityPerHour: Int
  ): Unit = {
    
    // Constraint 1: 각 사용자는 정확히 1개 시간대에 할당
    for (user <- users) {
      val constraint: MPConstraint = solver.makeConstraint(
        1.0, 1.0, s"user_$user"
      )
      
      for (hour <- hours if variables.contains((user, hour))) {
        constraint.setCoefficient(variables((user, hour)), 1.0)
      }
    }
    
    // Constraint 2: 시간대별 용량 제약
    for (hour <- hours) {
      val constraint: MPConstraint = solver.makeConstraint(
        0.0, capacityPerHour.toDouble, s"hour_$hour"
      )
      
      for (user <- users if variables.contains((user, hour))) {
        constraint.setCoefficient(variables((user, hour)), 1.0)
      }
    }
  }

  /**
   * 솔루션 추출
   */
  def extractSolution(
    variables: mutable.Map[(String, Int), MPVariable],
    userData: Map[String, Map[Int, Double]],
    users: Array[String],
    hours: Array[Int]
  ): Array[AllocationResult] = {
    
    val results = mutable.ArrayBuffer[AllocationResult]()
    
    for (user <- users) {
      for (hour <- hours if variables.contains((user, hour))) {
        val variable = variables((user, hour))
        if (variable.solutionValue() > 0.5) {
          val score = userData(user)(hour)
          results += AllocationResult(user, hour, score)
        }
      }
    }
    
    results.toArray
  }

  /**
   * 통계 출력
   */
  def printStatistics(
    results: Array[AllocationResult],
    hours: Array[Int],
    capacityPerHour: Int
  ): Unit = {
    
    println("\n" + "=" * 80)
    println("Allocation Statistics")
    println("=" * 80)
    
    val totalAssigned = results.length
    val totalScore = results.map(_.score).sum
    val avgScore = totalScore / totalAssigned
    
    println(s"Total assigned: ${numFormatter.format(totalAssigned)} users")
    println(f"Total score: $totalScore%,.2f")
    println(f"Average score: $avgScore%.4f")
    
    println("\nHour-wise allocation:")
    println("-" * 60)
    println(f"${"Hour"}%-8s ${"Count"}%12s ${"Total Score"}%15s ${"Avg Score"}%12s")
    println("-" * 60)
    
    val hourStats = results.groupBy(_.assigned_hour)
    
    for (hour <- hours) {
      val hourResults = hourStats.getOrElse(hour, Array.empty)
      val count = hourResults.length
      val hourTotalScore = hourResults.map(_.score).sum
      val hourAvgScore = if (count > 0) hourTotalScore / count else 0.0
      val utilization = count.toDouble / capacityPerHour * 100
      
      println(f"$hour%-8d ${numFormatter.format(count)}%12s $hourTotalScore%,15.2f $hourAvgScore%12.4f ($utilization%5.1f%%)")
    }
    println("-" * 60)
  }

  // ============================================================================
  // Google OR-Tools Based Optimization
  // ============================================================================

  /**
   * Preprocessing: 사용자별 선택지 제한
   * 각 사용자의 상위 N개 시간대만 선택지로 제한
   */
  def preprocessTopChoices(
    df: DataFrame,
    topN: Int = 5
  ): DataFrame = {
    
    import df.sparkSession.implicits._
    
    println(s"\n${"=" * 80}")
    println(s"Preprocessing: Limiting to top $topN choices per user")
    println(s"${"=" * 80}")
    
    val originalCount = df.count()
    val originalUsers = df.select("svc_mgmt_num").distinct().count()
    
    println(s"Original data:")
    println(s"  Total rows: ${numFormatter.format(originalCount)}")
    println(s"  Unique users: ${numFormatter.format(originalUsers)}")
    
    // 각 사용자별로 propensity_score 상위 topN개만 선택
    val windowSpec = Window.partitionBy("svc_mgmt_num")
      .orderBy(desc("propensity_score"))
    
    val filtered = df
      .withColumn("rank", row_number().over(windowSpec))
      .filter($"rank" <= topN)
      .drop("rank")
    
    val filteredCount = filtered.count()
    val filteredUsers = filtered.select("svc_mgmt_num").distinct().count()
    
    println(s"\nFiltered data:")
    println(s"  Total rows: ${numFormatter.format(filteredCount)}")
    println(s"  Unique users: ${numFormatter.format(filteredUsers)}")
    println(s"  Reduction: ${(1 - filteredCount.toDouble / originalCount) * 100}%.1f%%")
    
    // 시간대별 분포 확인
    println(s"\nHour distribution after filtering:")
    val hourDist = filtered.groupBy("send_hour")
      .agg(countDistinct("svc_mgmt_num").as("user_count"))
      .orderBy("send_hour")
      .collect()
    
    hourDist.foreach { row =>
      val hour = row.getInt(0)
      val count = row.getLong(1)
      println(f"  Hour $hour: ${numFormatter.format(count)} users")
    }
    
    // 사용자별 평균 선택지 수
    val avgChoices = filtered.groupBy("svc_mgmt_num")
      .count()
      .agg(avg("count"))
      .first()
      .getDouble(0)
    
    println(f"\nAverage choices per user: $avgChoices%.2f (max: $topN)")
    println(s"${"=" * 80}\n")
    
    filtered
  }

  /**
   * 1. 시간대별 차등 용량을 적용한 최적화 (FEASIBLE 수용 + Preprocessing)
   */
  def allocateUsersWithHourlyCapacity(
    df: DataFrame,
    capacityPerHour: Map[Int, Int],
    timeLimit: Int = 300,
    batchSize: Int = 500000,
    topChoices: Int = 5,
    enablePreprocessing: Boolean = true
  ): DataFrame = {
    
    import df.sparkSession.implicits._
    
    println("=" * 80)
    println("Robust Allocation with hourly capacity")
    println(s"Preprocessing: ${if (enablePreprocessing) s"Enabled (top $topChoices)" else "Disabled"}")
    println("FEASIBLE solution: Accepted")
    println("=" * 80)
    
    // Preprocessing: 상위 N개 선택지만 사용
    val processedDf = if (enablePreprocessing) {
      preprocessTopChoices(df, topChoices)
    } else {
      df
    }
    
    Loader.loadNativeLibraries()
    
    val userData = collectUserData(processedDf)
    val users = userData.keys.toArray.sorted
    val hours = processedDf.select("send_hour").distinct().collect().map(_.getInt(0)).sorted
    
    println(s"\n[INPUT INFO]")
    println(s"Users: ${numFormatter.format(users.length)}")
    println(s"Hours: ${hours.length}")
    
    if (users.length > batchSize) {
      println(s"⚠ Warning: User count (${users.length}) exceeds batchSize ($batchSize)")
    }
    
    // 데이터 분포 분석
    println(s"\n[DATA DISTRIBUTION]")
    val hourUserCounts = userData.values.flatMap(_.keys).groupBy(identity).mapValues(_.size)
    hours.sorted.foreach { hour =>
      val userCount = hourUserCounts.getOrElse(hour, 0)
      println(f"  Hour $hour: $userCount%,6d users have this hour as an option")
    }
    
    val avgChoicesPerUser = userData.values.map(_.size).sum.toDouble / users.length
    println(f"\n  Average choices per user: $avgChoicesPerUser%.2f hours")
    
    // Solver 설정
    val solver = MPSolver.createSolver("SCIP")
    solver.setTimeLimit(timeLimit * 1000L)
    
    val variables = createVariables(solver, users, hours, userData)
    setObjective(solver, variables, userData)
    
    // 제약 조건 1: 사용자당 정확히 1회 할당
    println(s"\n[CONSTRAINT 1: User Assignment]")
    for (user <- users) {
      val constraint = solver.makeConstraint(1.0, 1.0, s"user_$user")
      for (hour <- hours if variables.contains((user, hour))) {
        constraint.setCoefficient(variables((user, hour)), 1.0)
      }
    }
    println(s"  ✓ Each user must be assigned to exactly 1 hour")
    
    // 제약 조건 2: 시간대별 용량 (독립 제약)
    println(s"\n[CONSTRAINT 2: Hourly Capacity]")
    for (hour <- hours) {
      val hourCapacity = capacityPerHour.getOrElse(hour, 0)
      val constraint = solver.makeConstraint(0.0, hourCapacity.toDouble, s"hour_$hour")
      
      var varCount = 0
      for (user <- users if variables.contains((user, hour))) {
        constraint.setCoefficient(variables((user, hour)), 1.0)
        varCount += 1
      }
      
      println(f"  Hour $hour: capacity = ${numFormatter.format(hourCapacity)}, candidates = ${numFormatter.format(varCount)}")
    }
    
    val totalCapacity = capacityPerHour.values.sum
    println(f"\n  Total capacity: ${numFormatter.format(totalCapacity)}")
    println(f"  Users to assign: ${numFormatter.format(users.length)}")
    println(f"  Capacity ratio: ${totalCapacity.toDouble / users.length}%.2fx")
    
    println(s"\n[SOLVER INFO]")
    println(s"  Variables: ${numFormatter.format(variables.size)}")
    println(s"  Constraints: ${numFormatter.format(solver.numConstraints())}")
    println(s"  Time limit: ${timeLimit}s")
    
    println("\nSolving...")
    val startTime = System.currentTimeMillis()
    val status = solver.solve()
    val solveTime = (System.currentTimeMillis() - startTime) / 1000.0
    
    status match {
      case MPSolver.ResultStatus.OPTIMAL =>
        println(s"✓ OPTIMAL solution found in ${solveTime}s")
        val objValue = solver.objective().value()
        println(f"Objective value: $objValue%,.2f")
        
        val results = extractSolution(variables, userData, users, hours)
        validateAndPrintResults(results, users, hours, capacityPerHour)
        results.toSeq.toDF()
        
      case MPSolver.ResultStatus.FEASIBLE =>
        println(s"⚠ FEASIBLE solution found in ${solveTime}s (timeout, but solution accepted)")
        val objValue = solver.objective().value()
        println(f"Objective value: $objValue%,.2f")
        
        val results = extractSolution(variables, userData, users, hours)
        val assignedCount = results.length
        
        println(f"\nFeasible solution quality:")
        println(f"  Assigned: ${numFormatter.format(assignedCount)} / ${numFormatter.format(users.length)}")
        println(f"  Coverage: ${assignedCount.toDouble / users.length * 100}%.2f%%")
        
        if (assignedCount > 0) {
          validateAndPrintResults(results, users, hours, capacityPerHour)
          results.toSeq.toDF()
        } else {
          println("  ✗ No assignments in feasible solution")
          throw new RuntimeException("Empty feasible solution")
        }
        
      case _ =>
        println(s"✗ Solver failed with status: $status (${solveTime}s)")
        throw new RuntimeException(s"Solver failed: $status")
    }
  }

  /**
   * 결과 검증 및 출력
   */
  def validateAndPrintResults(
    results: Array[AllocationResult],
    users: Array[String],
    hours: Array[Int],
    capacityPerHour: Map[Int, Int]
  ): Unit = {
    
    println(s"\n${"=" * 80}")
    println("[VALIDATION: Hourly Assignment vs Capacity]")
    println(s"${"=" * 80}")
    
    val hourlyAssignment = results.groupBy(_.assigned_hour).mapValues(_.length)
    val totalCapacity = capacityPerHour.values.sum
    
    var totalAssigned = 0
    var violationDetected = false
    
    hours.sorted.foreach { hour =>
      val assigned = hourlyAssignment.getOrElse(hour, 0)
      val capacity = capacityPerHour.getOrElse(hour, 0)
      val utilizationPct = if (capacity > 0) assigned.toDouble / capacity * 100 else 0.0
      
      val status = if (assigned <= capacity) "✓" else "✗ VIOLATION"
      
      println(f"  Hour $hour: assigned=${numFormatter.format(assigned).padTo(8, ' ')} / capacity=${numFormatter.format(capacity).padTo(8, ' ')} (${utilizationPct}%5.1f%%) $status")
      
      totalAssigned += assigned
      
      if (assigned > capacity) {
        violationDetected = true
        println(s"    ⚠⚠⚠ ERROR: Over capacity by ${numFormatter.format(assigned - capacity)}!")
      }
    }
    
    println(s"\n[SUMMARY]")
    println(f"  Total assigned: ${numFormatter.format(totalAssigned)}")
    println(f"  Total users: ${numFormatter.format(users.length)}")
    println(f"  Assignment rate: ${totalAssigned.toDouble / users.length * 100}%.2f%%")
    println(f"  Total capacity: ${numFormatter.format(totalCapacity)}")
    println(f"  Capacity utilization: ${totalAssigned.toDouble / totalCapacity * 100}%.2f%%")
    
    if (violationDetected) {
      println("\n✗✗✗ CRITICAL ERROR: Capacity constraints violated!")
      throw new RuntimeException("Capacity constraints not enforced correctly")
    } else {
      println("\n✓✓✓ All capacity constraints satisfied")
    }
    
    println(s"${"=" * 80}\n")
  }

  /**
   * 2. Hybrid 할당: OR-Tools + Greedy 조합
   */
  def allocateUsersHybrid(
    df: DataFrame,
    capacityPerHour: Map[Int, Int],
    timeLimit: Int = 300,
    batchSize: Int = 500000,
    topChoices: Int = 5,
    enablePreprocessing: Boolean = true
  ): DataFrame = {
    
    import df.sparkSession.implicits._
    
    try {
      val optimizerResult = allocateUsersWithHourlyCapacity(
        df, 
        capacityPerHour, 
        timeLimit, 
        batchSize,
        topChoices,
        enablePreprocessing
      )
      
      // 미할당 사용자 확인
      val assignedUsers = optimizerResult.select("svc_mgmt_num").collect().map(_.getString(0)).toSet
      val allUsers = df.select("svc_mgmt_num").distinct().collect().map(_.getString(0)).toSet
      val unassignedUsers = allUsers -- assignedUsers
      
      if (unassignedUsers.isEmpty) {
        println("✓ All users assigned by optimizer")
        return optimizerResult
      }
      
      println(s"\n${numFormatter.format(unassignedUsers.size)} users unassigned, running Greedy for remainder...")
      
      // 남은 용량 계산
      val usedCapacity = optimizerResult.groupBy("assigned_hour").count().collect()
        .map(r => r.getInt(0) -> r.getLong(1).toInt).toMap
      
      val remainingCapacity = capacityPerHour.map { case (hour, cap) =>
        hour -> Math.max(0, cap - usedCapacity.getOrElse(hour, 0))
      }
      
      println("\nRemaining capacity after optimizer:")
      remainingCapacity.toSeq.sortBy(_._1).foreach { case (hour, cap) =>
        println(s"  Hour $hour: ${numFormatter.format(cap)}")
      }
      
      // 미할당 사용자를 원본 데이터에서 가져옴 (preprocessing 없이)
      val unassignedDf = df.filter($"svc_mgmt_num".isin(unassignedUsers.toSeq: _*))
      val hours = df.select("send_hour").distinct().collect().map(_.getInt(0)).sorted
      val greedyResult = allocateGreedySimple(unassignedDf, hours, remainingCapacity)
      
      if (greedyResult.count() == 0) {
        println("Greedy assigned 0 users")
        return optimizerResult
      }
      
      val combined = optimizerResult.union(greedyResult)
      val totalAssigned = combined.count()
      println(s"\nTotal assigned: ${numFormatter.format(totalAssigned)} / ${numFormatter.format(allUsers.size)}")
      
      combined
      
    } catch {
      case e: Exception =>
        println(s"\nOptimizer failed: ${e.getMessage}")
        println("Running full Greedy allocation...")
        val hours = df.select("send_hour").distinct().collect().map(_.getInt(0)).sorted
        allocateGreedySimple(df, hours, capacityPerHour)
    }
  }

  /**
   * 3. 안정적인 배치 처리 함수 - Hybrid 방식 (가치 기반 정렬)
   */
  def allocateLargeScaleHybrid(
    df: DataFrame,
    capacityPerHour: Map[Int, Int],
    batchSize: Int = 500000,
    timeLimit: Int = 300,
    topChoices: Int = 5,
    enablePreprocessing: Boolean = true
  ): DataFrame = {
    
    import df.sparkSession.implicits._
    
    println("=" * 80)
    println("Batch Allocation for Large Scale Data (Hybrid Mode)")
    println(s"Preprocessing: ${if (enablePreprocessing) s"Enabled (top $topChoices)" else "Disabled"}")
    println("FEASIBLE solution: Accepted")
    println("=" * 80)
    
    // 가치 기반 정렬
    val userPriority = df.groupBy("svc_mgmt_num")
      .agg(max("propensity_score").as("max_prob"))

    val totalUsers = userPriority.count()
    val numBatches = Math.ceil(totalUsers.toDouble / batchSize).toInt
    
    println(s"\n[BATCH SETUP]")
    println(s"Total users: ${numFormatter.format(totalUsers)}")
    println(s"Batch size: ${numFormatter.format(batchSize)}")
    println(s"Number of batches: $numBatches")
    
    // 시간대별 용량 출력
    println("\n[CAPACITY PER HOUR]")
    capacityPerHour.toSeq.sortBy(_._1).foreach { case (hour, cap) =>
      println(s"  Hour $hour: ${numFormatter.format(cap)}")
    }
    val totalCapacity = capacityPerHour.values.sum
    println(s"Total capacity: ${numFormatter.format(totalCapacity)}")
    
    // 가치 기반 배치 분할
    val allUsers = userPriority
      .withColumn("row_id", row_number().over(Window.orderBy(desc("max_prob"))))
      .withColumn("batch_id", (($"row_id" - 1) / batchSize).cast("int"))
      .select("svc_mgmt_num", "batch_id")
      .cache()
    
    println("\n[BATCH DISTRIBUTION]")
    val batchCounts = allUsers.groupBy("batch_id").count().collect()
      .map(r => r.getInt(0) -> r.getLong(1))
      .sortBy(_._1)
    
    batchCounts.foreach { case (bid, cnt) =>
      println(s"  Batch $bid: ${numFormatter.format(cnt)} users")
    }
    
    var remainingCapacity = capacityPerHour.toMap
    val allResults = mutable.ArrayBuffer[DataFrame]()
    var totalAssignedSoFar = 0L
    
    // 각 배치별로 최적화
    for (batchId <- 0 until numBatches) {
      println(s"\n${"=" * 80}")
      println(s"Processing Batch ${batchId + 1}/$numBatches")
      println(s"${"=" * 80}")
      
      val batchUsers = allUsers.filter($"batch_id" === batchId)
      
      // 용량이 남아있는 시간대만 처리
      val availableHours = remainingCapacity.filter(_._2 > 0).keys.toSeq
      
      if (availableHours.isEmpty) {
        println("⚠ No capacity left in any hour. Stopping.")
        val unassignedCount = totalUsers - totalAssignedSoFar
        println(s"Unassigned users: ${numFormatter.format(unassignedCount)}")
        
      } else {
        println(s"\nAvailable hours: ${availableHours.sorted.mkString(", ")}")
        println("\nRemaining capacity:")
        remainingCapacity.toSeq.sortBy(_._1).foreach { case (hour, cap) =>
          val status = if (availableHours.contains(hour)) "✓" else "✗"
          println(s"  Hour $hour: ${numFormatter.format(cap)} $status")
        }
        
        val batchDf = df.join(batchUsers, Seq("svc_mgmt_num"))
          .filter($"send_hour".isin(availableHours: _*))
        
        val batchUserCount = batchDf.select("svc_mgmt_num").distinct().count()
        println(s"\nBatch users with available hours: ${numFormatter.format(batchUserCount)}")
        
        if (batchUserCount == 0) {
          println("⚠ No users can be assigned")
        } else {
          val startTime = System.currentTimeMillis()
          val batchResult = allocateUsersHybrid(
            batchDf, 
            remainingCapacity, 
            timeLimit, 
            batchSize,
            topChoices,
            enablePreprocessing
          )
          val solveTime = (System.currentTimeMillis() - startTime) / 1000.0
          
          println(f"\nBatch completed in $solveTime%.2f seconds")
          
          val assignedCount = batchResult.count()
          
          if (assignedCount > 0) {
            totalAssignedSoFar += assignedCount
            
            // 용량 차감
            val allocatedPerHour = batchResult.groupBy("assigned_hour").count().collect()
              .map(row => row.getInt(0) -> row.getLong(1).toInt).toMap
            
            println("\n[CAPACITY UPDATE]")
            val hours = df.select("send_hour").distinct().collect().map(_.getInt(0)).sorted
            hours.foreach { hour =>
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
            println(f"\nBatch score: $batchScore%,.2f")
            println(s"Batch assigned: ${numFormatter.format(assignedCount)}")
            
            allResults += batchResult
          } else {
            println("⚠ No users assigned in this batch")
          }
        }
      }
      
      // 진행률
      val progress = totalAssignedSoFar.toDouble / totalUsers * 100
      val coverageVsCapacity = totalAssignedSoFar.toDouble / totalCapacity * 100
      println(f"\n[PROGRESS]")
      println(f"  Assigned: ${numFormatter.format(totalAssignedSoFar)} / ${numFormatter.format(totalUsers)} users ($progress%.1f%%)")
      println(f"  Capacity used: ${numFormatter.format(totalAssignedSoFar)} / ${numFormatter.format(totalCapacity)} ($coverageVsCapacity%.1f%%)")
    }
    
    allUsers.unpersist()
    
    // 최종 결과
    if (allResults.isEmpty) {
      println("\n⚠ No results generated!")
      return df.sparkSession.emptyDataFrame
    }
    
    val finalResult = allResults.reduce(_.union(_))
    printFinalStatistics(finalResult, totalUsers)
    
    finalResult
  }

  /**
   * 4. 간단한 Greedy 할당
   */
  def allocateGreedySimple(
    df: DataFrame,
    hours: Array[Int],
    initialCapacity: Map[Int, Int]
  ): DataFrame = {
    
    import df.sparkSession.implicits._
    
    println("\n" + "=" * 80)
    println("Running Greedy allocation")
    println("=" * 80)
    
    println("\nInitial capacity:")
    initialCapacity.toSeq.sortBy(_._1).foreach { case (hour, cap) =>
      println(s"  Hour $hour: ${numFormatter.format(cap)}")
    }
    
    val userData = collectUserData(df)
    val users = userData.keys.toArray
    
    println(s"\nUsers to assign: ${numFormatter.format(users.length)}")
    
    // 사용자를 최고 점수 순으로 정렬
    val userBestScores = users.map { user =>
      val bestScore = userData(user).values.max
      (user, bestScore)
    }.sortBy(-_._2)
    
    val hourCapacity = mutable.Map(initialCapacity.toSeq: _*)
    val assignments = mutable.ArrayBuffer[AllocationResult]()
    
    for ((user, _) <- userBestScores) {
      val choices = userData(user).toSeq.sortBy(-_._2)
      
      var assigned = false
      for ((hour, score) <- choices if !assigned) {
        val currentCapacity = hourCapacity.getOrElse(hour, 0)
        
        if (currentCapacity > 0) {
          assignments += AllocationResult(user, hour, score)
          hourCapacity(hour) = currentCapacity - 1
          assigned = true
        }
      }
    }
    
    println(s"\nGreedy assigned: ${numFormatter.format(assignments.size)} / ${numFormatter.format(users.length)}")
    
    if (assignments.nonEmpty) {
      println("\n[GREEDY ALLOCATION BY HOUR]")
      val hourlyAssignment = assignments.groupBy(_.assigned_hour).mapValues(_.length)
      
      hours.sorted.foreach { hour =>
        val assigned = hourlyAssignment.getOrElse(hour, 0)
        val initialCap = initialCapacity.getOrElse(hour, 0)
        val remaining = hourCapacity.getOrElse(hour, 0)
        
        println(f"  Hour $hour: assigned=${numFormatter.format(assigned)}, capacity=${numFormatter.format(initialCap)}, remaining=${numFormatter.format(remaining)}")
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
   * 5. 최종 통계 출력
   */
  def printFinalStatistics(result: DataFrame, totalUsers: Long): Unit = {
    println(s"\n${"=" * 80}")
    println("Final Allocation Statistics")
    println(s"${"=" * 80}")
    
    val totalAssigned = result.count()
    val coverage = totalAssigned.toDouble / totalUsers * 100
    
    println(s"\nTotal assigned: ${numFormatter.format(totalAssigned)} / ${numFormatter.format(totalUsers)} ($coverage%.2f%%)")
    
    if (totalAssigned > 0) {
      val totalScore = result.agg(sum("score")).first().getDouble(0)
      val avgScore = totalScore / totalAssigned
      
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
    }
    
    println("=" * 80)
  }

  // ============================================================================
  // Simulated Annealing Based Optimization
  // ============================================================================

  /**
   * 1. Simulated Annealing 기반 최적화 (적응형 온도 + 재가열)
   */
  def allocateUsersWithSimulatedAnnealing(
    df: DataFrame,
    capacityPerHour: Map[Int, Int],
    maxIterations: Int = 100000,
    initialTemperature: Double = 100.0,
    coolingRate: Double = 0.995,
    batchSize: Int = 500000,
    useAdaptiveTemp: Boolean = true,
    reheatingEnabled: Boolean = true,
    reheatingThreshold: Int = 1000
  ): DataFrame = {
    
    import df.sparkSession.implicits._
    
    println("=" * 80)
    println("Simulated Annealing Optimization (Enhanced)")
    println("=" * 80)
    
    val userData = collectUserData(df)
    val users = userData.keys.toArray.sorted
    val hours = df.select("send_hour").distinct().collect().map(_.getInt(0)).sorted
    
    println(s"\n[INPUT INFO]")
    println(s"Users: ${numFormatter.format(users.length)}")
    println(s"Hours: ${hours.length}")
    
    if (users.length > batchSize) {
      println(s"⚠ Warning: User count (${users.length}) exceeds batchSize ($batchSize)")
    }
    
    // 데이터 분포 분석
    println(s"\n[DATA DISTRIBUTION]")
    val hourUserCounts = userData.values.flatMap(_.keys).groupBy(identity).mapValues(_.size)
    hours.sorted.foreach { hour =>
      val userCount = hourUserCounts.getOrElse(hour, 0)
      println(f"  Hour $hour: $userCount%,6d users have this hour as an option")
    }
    
    println(s"\n[CAPACITY PER HOUR]")
    capacityPerHour.toSeq.sortBy(_._1).foreach { case (hour, cap) =>
      println(s"  Hour $hour: ${numFormatter.format(cap)}")
    }
    val totalCapacity = capacityPerHour.values.sum
    println(f"\n  Total capacity: ${numFormatter.format(totalCapacity)}")
    println(f"  Users to assign: ${numFormatter.format(users.length)}")
    println(f"  Capacity ratio: ${totalCapacity.toDouble / users.length}%.2fx")
    
    // 초기 해 생성 (Greedy)
    println("\nGenerating initial solution...")
    val startTime = System.currentTimeMillis()
    val initialSolution = generateInitialSolution(users, userData, capacityPerHour, hours)
    val initTime = (System.currentTimeMillis() - startTime) / 1000.0
    
    println(f"Initial solution generated in $initTime%.2f seconds")
    println(f"  Initial score: ${initialSolution.score}%,.2f")
    println(f"  Assigned users: ${numFormatter.format(initialSolution.assignments.size)}")
    
    // 적응형 온도 설정
    val adaptiveInitialTemp = if (useAdaptiveTemp) {
      val temp = initialSolution.score * 0.1  // 초기 점수의 10%
      println(f"\n✓ Adaptive temperature enabled: $temp%.2f (based on initial score)")
      temp
    } else {
      initialTemperature
    }
    
    println(s"\n[ALGORITHM PARAMETERS]")
    println(s"  Max iterations: ${numFormatter.format(maxIterations)}")
    println(f"  Initial temperature: $adaptiveInitialTemp%.1f ${if (useAdaptiveTemp) "(adaptive)" else ""}")
    println(f"  Cooling rate: $coolingRate%.4f")
    println(s"  Reheating: ${if (reheatingEnabled) s"Enabled (threshold: $reheatingThreshold)" else "Disabled"}")
    
    // Simulated Annealing 실행
    println("\nRunning Simulated Annealing...")
    val saStartTime = System.currentTimeMillis()
    val bestSolution = runSimulatedAnnealing(
      initialSolution,
      userData,
      capacityPerHour,
      hours,
      maxIterations,
      adaptiveInitialTemp,
      coolingRate,
      reheatingEnabled,
      reheatingThreshold
    )
    val saTime = (System.currentTimeMillis() - saStartTime) / 1000.0
    
    println(f"\nOptimization completed in $saTime%.2f seconds")
    println(f"  Best score: ${bestSolution.score}%,.2f")
    println(f"  Improvement: ${(bestSolution.score - initialSolution.score)}%,.2f")
    println(f"  Improvement rate: ${((bestSolution.score - initialSolution.score) / initialSolution.score * 100)}%.2f%%")
    
    // 결과 검증
    validateSolution(bestSolution, capacityPerHour, hours, users.length)
    
    // DataFrame 변환
    val results = bestSolution.assignments.map { case (user, hour) =>
      val score = userData(user)(hour)
      AllocationResult(user, hour, score)
    }.toSeq
    
    results.toDF()
  }

  /**
   * 2. 초기 해 생성 (Greedy 방식)
   */
  def generateInitialSolution(
    users: Array[String],
    userData: Map[String, Map[Int, Double]],
    capacityPerHour: Map[Int, Int],
    hours: Array[Int]
  ): Solution = {
    
    val hourCapacity = mutable.Map(capacityPerHour.toSeq: _*)
    val assignments = mutable.Map[String, Int]()
    val hourUsage = mutable.Map[Int, Int]().withDefaultValue(0)
    var totalScore = 0.0
    
    // 사용자를 최고 점수 순으로 정렬
    val sortedUsers = users.sortBy { user =>
      -userData(user).values.max
    }
    
    for (user <- sortedUsers) {
      val choices = userData(user).toSeq.sortBy(-_._2)
      
      var assigned = false
      for ((hour, score) <- choices if !assigned) {
        if (hourCapacity(hour) > 0) {
          assignments(user) = hour
          hourUsage(hour) += 1
          hourCapacity(hour) -= 1
          totalScore += score
          assigned = true
        }
      }
    }
    
    Solution(assignments.toMap, totalScore, hourUsage.toMap)
  }

  /**
   * 3. Simulated Annealing 메인 루프 (개선된 재가열)
   */
  def runSimulatedAnnealing(
    initialSolution: Solution,
    userData: Map[String, Map[Int, Double]],
    capacityPerHour: Map[Int, Int],
    hours: Array[Int],
    maxIterations: Int,
    initialTemp: Double,
    coolingRate: Double,
    reheatingEnabled: Boolean = true,
    reheatingThreshold: Int = 1000
  ): Solution = {
    
    val random = new Random(42)
    var currentSolution = initialSolution
    var bestSolution = initialSolution
    var temperature = initialTemp
    
    var acceptedMoves = 0
    var rejectedMoves = 0
    var improvements = 0
    var iterationsWithoutImprovement = 0
    var reheatingCount = 0
    var consecutiveNoProgress = 0
    
    val reportInterval = maxIterations / 10
    val maxReheatingCount = 10  // 최대 재가열 횟수 제한
    
    for (iteration <- 1 to maxIterations) {
      // 이웃 해 생성 (점수 기반 전략 사용)
      val neighborSolution = generateSmartNeighbor(
        currentSolution,
        userData,
        capacityPerHour,
        hours,
        random
      )
      
      if (neighborSolution.isDefined) {
        val neighbor = neighborSolution.get
        val delta = neighbor.score - currentSolution.score
        
        // 수락 여부 결정
        if (delta > 0 || random.nextDouble() < Math.exp(delta / temperature)) {
          currentSolution = neighbor
          acceptedMoves += 1
          
          if (neighbor.score > bestSolution.score) {
            bestSolution = neighbor
            improvements += 1
            iterationsWithoutImprovement = 0
            consecutiveNoProgress = 0
          } else {
            iterationsWithoutImprovement += 1
          }
        } else {
          rejectedMoves += 1
          iterationsWithoutImprovement += 1
        }
      } else {
        iterationsWithoutImprovement += 1
      }
      
      // 개선된 재가열 메커니즘
      if (reheatingEnabled && 
          iterationsWithoutImprovement > reheatingThreshold && 
          reheatingCount < maxReheatingCount &&
          temperature < initialTemp * 0.05) {  // 온도가 충분히 식었을 때만
        
        temperature = initialTemp * 0.2  // 더 낮은 온도로 재가열
        iterationsWithoutImprovement = 0
        reheatingCount += 1
        consecutiveNoProgress += 1
        
        // 3번 연속 재가열해도 개선 없으면 중단
        if (consecutiveNoProgress >= 3) {
          println(f"\n  Early stopping: No improvement after $reheatingCount reheating cycles")
          // 재가열 비활성화하고 계속 진행 (조기 종료는 하지 않음)
          temperature *= coolingRate
        }
      } else {
        // 온도 감소
        temperature *= coolingRate
      }
      
      // 진행 상황 출력
      if (iteration % reportInterval == 0) {
        val progress = iteration.toDouble / maxIterations * 100
        val acceptRate = acceptedMoves.toDouble / (acceptedMoves + rejectedMoves) * 100
        println(f"  Iteration ${numFormatter.format(iteration)} ($progress%.0f%%) - " +
          f"Best: ${bestSolution.score}%,.2f, Current: ${currentSolution.score}%,.2f, " +
          f"Temp: $temperature%.2f, Accept: $acceptRate%.1f%%, " +
          f"Improvements: $improvements, Reheating: $reheatingCount")
      }
    }
    
    val totalMoves = acceptedMoves + rejectedMoves
    val finalAcceptRate = if (totalMoves > 0) acceptedMoves.toDouble / totalMoves * 100 else 0.0
    
    println(f"\n[SA STATISTICS]")
    println(f"  Total moves evaluated: ${numFormatter.format(totalMoves)}")
    println(f"  Accepted moves: ${numFormatter.format(acceptedMoves)} ($finalAcceptRate%.1f%%)")
    println(f"  Improvements found: ${numFormatter.format(improvements)}")
    println(f"  Reheating count: ${numFormatter.format(reheatingCount)}")
    
    bestSolution
  }

  /**
   * 4. 이웃 해 생성 (여러 전략)
   */
  def generateNeighbor(
    current: Solution,
    userData: Map[String, Map[Int, Double]],
    capacityPerHour: Map[Int, Int],
    hours: Array[Int],
    random: Random
  ): Option[Solution] = {
    
    val strategy = random.nextInt(3)
    
    strategy match {
      case 0 => reassignSingleUser(current, userData, capacityPerHour, random)
      case 1 => swapTwoUsers(current, userData, random)
      case 2 => reassignMultipleUsers(current, userData, capacityPerHour, random, 3)
      case _ => reassignSingleUser(current, userData, capacityPerHour, random)
    }
  }

  /**
   * 4-1. 스마트 이웃 해 생성 (점수 기반 전략)
   */
  def generateSmartNeighbor(
    current: Solution,
    userData: Map[String, Map[Int, Double]],
    capacityPerHour: Map[Int, Int],
    hours: Array[Int],
    random: Random
  ): Option[Solution] = {
    
    // 70% 확률로 점수 기반 전략, 30% 확률로 랜덤 전략
    val useSmartStrategy = random.nextDouble() < 0.7
    
    if (useSmartStrategy) {
      val strategy = random.nextInt(2)
      strategy match {
        case 0 => smartReassignUser(current, userData, capacityPerHour, random)
        case 1 => smartSwapUsers(current, userData, capacityPerHour, random)
        case _ => smartReassignUser(current, userData, capacityPerHour, random)
      }
    } else {
      // 기존 랜덤 전략
      generateNeighbor(current, userData, capacityPerHour, hours, random)
    }
  }

  /**
   * 스마트 전략 1: 점수가 낮은 사용자를 더 나은 시간대로 재할당
   */
  def smartReassignUser(
    current: Solution,
    userData: Map[String, Map[Int, Double]],
    capacityPerHour: Map[Int, Int],
    random: Random
  ): Option[Solution] = {
    
    if (current.assignments.isEmpty) return None
    
    // 현재 점수가 낮은 사용자 찾기 (하위 20%)
    val userScores = current.assignments.map { case (user, hour) =>
      (user, hour, userData(user)(hour))
    }.toArray.sortBy(_._3)
    
    val lowScoreUsers = userScores.take(Math.max(1, userScores.length / 5))
    
    if (lowScoreUsers.isEmpty) return None
    
    // 랜덤하게 하나 선택
    val (user, currentHour, currentScore) = lowScoreUsers(random.nextInt(lowScoreUsers.length))
    
    // 더 나은 시간대 찾기
    val betterHours = userData(user).filter { case (hour, score) =>
      hour != currentHour &&
      score > currentScore &&  // 점수가 더 좋고
      current.hourUsage.getOrElse(hour, 0) < capacityPerHour.getOrElse(hour, 0)  // 용량 여유
    }
    
    if (betterHours.isEmpty) return None
    
    // 가장 좋은 시간대 선택 (70% 확률) 또는 랜덤 (30%)
    val newHour = if (random.nextDouble() < 0.7) {
      betterHours.maxBy(_._2)._1
    } else {
      val hoursArray = betterHours.keys.toArray
      hoursArray(random.nextInt(hoursArray.length))
    }
    
    val newScore = userData(user)(newHour)
    
    // 새로운 해 생성
    val newAssignments = current.assignments + (user -> newHour)
    val newHourUsage = current.hourUsage +
      (currentHour -> (current.hourUsage(currentHour) - 1)) +
      (newHour -> (current.hourUsage.getOrElse(newHour, 0) + 1))
    val newTotalScore = current.score - currentScore + newScore
    
    Some(Solution(newAssignments, newTotalScore, newHourUsage))
  }

  /**
   * 스마트 전략 2: 서로에게 이득인 교환 찾기
   */
  def smartSwapUsers(
    current: Solution,
    userData: Map[String, Map[Int, Double]],
    capacityPerHour: Map[Int, Int],
    random: Random
  ): Option[Solution] = {
    
    if (current.assignments.size < 2) return None
    
    // 현재 점수가 낮은 사용자들 중에서 선택 (하위 30%)
    val userScores = current.assignments.map { case (user, hour) =>
      (user, hour, userData(user)(hour))
    }.toArray.sortBy(_._3)
    
    val candidates = userScores.take(Math.max(2, userScores.length / 3))
    
    if (candidates.length < 2) return None
    
    // 무작위로 2명 선택
    val idx1 = random.nextInt(candidates.length)
    var idx2 = random.nextInt(candidates.length)
    while (idx2 == idx1 && candidates.length > 1) {
      idx2 = random.nextInt(candidates.length)
    }
    
    val (user1, hour1, score1) = candidates(idx1)
    val (user2, hour2, score2) = candidates(idx2)
    
    if (hour1 == hour2) return None
    
    // 서로의 시간대를 가질 수 있는지 확인
    if (!userData(user1).contains(hour2) || !userData(user2).contains(hour1)) {
      return None
    }
    
    val newScore1 = userData(user1)(hour2)
    val newScore2 = userData(user2)(hour1)
    
    val oldScore = score1 + score2
    val newScore = newScore1 + newScore2
    
    // 교환이 이득인 경우에만 (또는 약간의 손해 허용)
    if (newScore >= oldScore * 0.95) {  // 5% 이내 손해 허용
      val newAssignments = current.assignments +
        (user1 -> hour2) +
        (user2 -> hour1)
      val newTotalScore = current.score - oldScore + newScore
      
      Some(Solution(newAssignments, newTotalScore, current.hourUsage))
    } else {
      None
    }
  }

  /**
   * 전략 1: 단일 사용자 재할당
   */
  def reassignSingleUser(
    current: Solution,
    userData: Map[String, Map[Int, Double]],
    capacityPerHour: Map[Int, Int],
    random: Random
  ): Option[Solution] = {
    
    if (current.assignments.isEmpty) return None
    
    val users = current.assignments.keys.toArray
    val user = users(random.nextInt(users.length))
    val currentHour = current.assignments(user)
    val currentScore = userData(user)(currentHour)
    
    // 다른 시간대 중 선택
    val availableHours = userData(user).keys.filter { hour =>
      hour != currentHour &&
      current.hourUsage.getOrElse(hour, 0) < capacityPerHour.getOrElse(hour, 0)
    }.toArray
    
    if (availableHours.isEmpty) return None
    
    val newHour = availableHours(random.nextInt(availableHours.length))
    val newScore = userData(user)(newHour)
    
    // 새로운 해 생성
    val newAssignments = current.assignments + (user -> newHour)
    val newHourUsage = current.hourUsage +
      (currentHour -> (current.hourUsage(currentHour) - 1)) +
      (newHour -> (current.hourUsage.getOrElse(newHour, 0) + 1))
    val newTotalScore = current.score - currentScore + newScore
    
    Some(Solution(newAssignments, newTotalScore, newHourUsage))
  }

  /**
   * 전략 2: 두 사용자 교환
   */
  def swapTwoUsers(
    current: Solution,
    userData: Map[String, Map[Int, Double]],
    random: Random
  ): Option[Solution] = {
    
    if (current.assignments.size < 2) return None
    
    val users = current.assignments.keys.toArray
    val user1 = users(random.nextInt(users.length))
    val user2 = users(random.nextInt(users.length))
    
    if (user1 == user2) return None
    
    val hour1 = current.assignments(user1)
    val hour2 = current.assignments(user2)
    
    if (hour1 == hour2) return None
    
    // 두 사용자가 서로의 시간대를 가질 수 있는지 확인
    if (!userData(user1).contains(hour2) || !userData(user2).contains(hour1)) {
      return None
    }
    
    val oldScore = userData(user1)(hour1) + userData(user2)(hour2)
    val newScore = userData(user1)(hour2) + userData(user2)(hour1)
    
    val newAssignments = current.assignments +
      (user1 -> hour2) +
      (user2 -> hour1)
    val newTotalScore = current.score - oldScore + newScore
    
    Some(Solution(newAssignments, newTotalScore, current.hourUsage))
  }

  /**
   * 전략 3: 다중 사용자 재할당
   */
  def reassignMultipleUsers(
    current: Solution,
    userData: Map[String, Map[Int, Double]],
    capacityPerHour: Map[Int, Int],
    random: Random,
    numUsers: Int
  ): Option[Solution] = {
    
    if (current.assignments.size < numUsers) return None
    
    val users = current.assignments.keys.toArray
    val selectedUsers = random.shuffle(users.toSeq).take(numUsers)
    
    var newSolution = current
    var success = false
    
    for (user <- selectedUsers) {
      val result = reassignSingleUser(newSolution, userData, capacityPerHour, random)
      if (result.isDefined) {
        newSolution = result.get
        success = true
      }
    }
    
    if (success) Some(newSolution) else None
  }

  /**
   * 5. 해 검증
   */
  def validateSolution(
    solution: Solution,
    capacityPerHour: Map[Int, Int],
    hours: Array[Int],
    totalUsers: Int
  ): Unit = {
    
    println(s"\n${"=" * 80}")
    println("[SOLUTION VALIDATION]")
    println(s"${"=" * 80}")
    
    val assignedUsers = solution.assignments.size
    println(f"\nTotal assigned: ${numFormatter.format(assignedUsers)} / ${numFormatter.format(totalUsers)}")
    println(f"Assignment rate: ${assignedUsers.toDouble / totalUsers * 100}%.2f%%")
    
    println("\n[HOURLY ALLOCATION vs CAPACITY]")
    var violationDetected = false
    
    hours.sorted.foreach { hour =>
      val assigned = solution.hourUsage.getOrElse(hour, 0)
      val capacity = capacityPerHour.getOrElse(hour, 0)
      val utilizationPct = if (capacity > 0) assigned.toDouble / capacity * 100 else 0.0
      
      val status = if (assigned <= capacity) "✓" else "✗ VIOLATION"
      
      println(f"  Hour $hour: assigned=${numFormatter.format(assigned).padTo(8, ' ')} / " +
        f"capacity=${numFormatter.format(capacity).padTo(8, ' ')} (${utilizationPct}%5.1f%%) $status")
      
      if (assigned > capacity) {
        violationDetected = true
        println(s"    ⚠⚠⚠ ERROR: Over capacity by ${numFormatter.format(assigned - capacity)}!")
      }
    }
    
    if (violationDetected) {
      println("\n✗✗✗ CRITICAL ERROR: Capacity constraints violated!")
      throw new RuntimeException("Capacity constraints not satisfied")
    } else {
      println("\n✓✓✓ All capacity constraints satisfied")
    }
    
    println(s"${"=" * 80}\n")
  }

  /**
   * 6. Hybrid 할당: SA + Greedy 조합
   */
  def allocateUsersHybridSA(
    df: DataFrame,
    capacityPerHour: Map[Int, Int],
    maxIterations: Int = 100000,
    batchSize: Int = 500000,
    useAdaptiveTemp: Boolean = true,
    reheatingEnabled: Boolean = true
  ): DataFrame = {
    
    import df.sparkSession.implicits._
    
    try {
      val saResult = allocateUsersWithSimulatedAnnealing(
        df,
        capacityPerHour,
        maxIterations,
        100.0,
        0.995,
        batchSize,
        useAdaptiveTemp,
        reheatingEnabled
      )
      
      val assignedUsers = saResult.select("svc_mgmt_num").collect().map(_.getString(0)).toSet
      val allUsers = df.select("svc_mgmt_num").distinct().collect().map(_.getString(0)).toSet
      val unassignedUsers = allUsers -- assignedUsers
      
      if (unassignedUsers.isEmpty) {
        println("✓ All users assigned by SA optimizer")
        return saResult
      }
      
      println(s"\n${numFormatter.format(unassignedUsers.size)} users unassigned, running Greedy for remainder...")
      
      val usedCapacity = saResult.groupBy("assigned_hour").count().collect()
        .map(r => r.getInt(0) -> r.getLong(1).toInt).toMap
      
      val remainingCapacity = capacityPerHour.map { case (hour, cap) =>
        hour -> Math.max(0, cap - usedCapacity.getOrElse(hour, 0))
      }
      
      val unassignedDf = df.filter($"svc_mgmt_num".isin(unassignedUsers.toSeq: _*))
      val hours = df.select("send_hour").distinct().collect().map(_.getInt(0)).sorted
      val greedyResult = allocateGreedySimple(unassignedDf, hours, remainingCapacity)
      
      if (greedyResult.count() == 0) {
        return saResult
      }
      
      saResult.union(greedyResult)
      
    } catch {
      case e: Exception =>
        println(s"\nSA optimizer failed: ${e.getMessage}")
        println("Running full Greedy allocation...")
        val hours = df.select("send_hour").distinct().collect().map(_.getInt(0)).sorted
        allocateGreedySimple(df, hours, capacityPerHour)
    }
  }

  /**
   * 7. 대규모 배치 처리 (SA 기반)
   */
  def allocateLargeScaleSA(
    df: DataFrame,
    capacityPerHour: Map[Int, Int],
    batchSize: Int = 500000,
    maxIterations: Int = 100000,
    useAdaptiveTemp: Boolean = true,
    reheatingEnabled: Boolean = true
  ): DataFrame = {
    
    import df.sparkSession.implicits._
    
    println("=" * 80)
    println("Batch Allocation for Large Scale Data (SA Mode)")
    println("=" * 80)
    
    val userPriority = df.groupBy("svc_mgmt_num")
      .agg(max("propensity_score").as("max_prob"))

    val totalUsers = userPriority.count()
    val numBatches = Math.ceil(totalUsers.toDouble / batchSize).toInt
    
    println(s"\n[BATCH SETUP]")
    println(s"Total users: ${numFormatter.format(totalUsers)}")
    println(s"Batch size: ${numFormatter.format(batchSize)}")
    println(s"Number of batches: $numBatches")
    
    val allUsers = userPriority
      .withColumn("row_id", row_number().over(Window.orderBy(desc("max_prob"))))
      .withColumn("batch_id", (($"row_id" - 1) / batchSize).cast("int"))
      .select("svc_mgmt_num", "batch_id")
      .cache()
    
    var remainingCapacity = capacityPerHour.toMap
    val allResults = mutable.ArrayBuffer[DataFrame]()
    var totalAssignedSoFar = 0L
    
    for (batchId <- 0 until numBatches) {
      println(s"\n${"=" * 80}")
      println(s"Processing Batch ${batchId + 1}/$numBatches")
      println(s"${"=" * 80}")
      
      val batchUsers = allUsers.filter($"batch_id" === batchId)
      val availableHours = remainingCapacity.filter(_._2 > 0).keys.toSeq
      
      if (availableHours.isEmpty) {
        println("⚠ No capacity left in any hour.")
        
      } else {
        val batchDf = df.join(batchUsers, Seq("svc_mgmt_num"))
          .filter($"send_hour".isin(availableHours: _*))
        
        val batchResult = allocateUsersHybridSA(
          batchDf, 
          remainingCapacity, 
          maxIterations, 
          batchSize,
          useAdaptiveTemp,
          reheatingEnabled
        )
        val assignedCount = batchResult.count()
        
        if (assignedCount > 0) {
          totalAssignedSoFar += assignedCount
          
          val allocatedPerHour = batchResult.groupBy("assigned_hour").count().collect()
            .map(row => row.getInt(0) -> row.getLong(1).toInt).toMap
          
          remainingCapacity = remainingCapacity.map { case (hour, cap) =>
            hour -> Math.max(0, cap - allocatedPerHour.getOrElse(hour, 0))
          }
          
          allResults += batchResult
        }
      }
    }
    
    allUsers.unpersist()
    
    if (allResults.isEmpty) {
      df.sparkSession.emptyDataFrame
    } else {
      val finalResult = allResults.reduce(_.union(_))
      printFinalStatistics(finalResult, totalUsers)
      finalResult
    }
  }

  // ============================================================================
  // jMetal Multi-Objective Optimization
  // ============================================================================

  /**
   * jMetal Problem 정의: 시간대 할당 문제를 다목적 최적화로 모델링
   * 
   * 목적 1: 총 propensity score 최대화
   * 목적 2: 시간대별 균등 분배 (부하 분산, 표준편차 최소화)
   * 
   * 제약: 
   * - 시간대별 용량 제한
   * - 각 사용자는 사용 가능한 시간대 중에서만 선택
   */
  class SendTimeAllocationProblem(
    users: Array[String],
    userData: Map[String, Map[Int, Double]],
    capacityPerHour: Map[Int, Int],
    hours: Array[Int]
  ) extends IntegerProblem {
    
    private val numUsers = users.length
    private val hourToIndex = hours.zipWithIndex.toMap
    private val indexToHour = hours
    
    // 각 사용자가 선택 가능한 시간대 인덱스 범위
    private val userHourRanges: Array[(Int, Int)] = users.map { user =>
      val availableHours = userData(user).keys.toArray.sorted
      if (availableHours.isEmpty) {
        (0, 0)  // 불가능한 경우
      } else {
        (hourToIndex(availableHours.head), hourToIndex(availableHours.last))
      }
    }
    
    // ============================================================================
    // jMetal 5.10 Problem API
    // ============================================================================
    override def getNumberOfVariables(): Int = numUsers
    override def getNumberOfObjectives(): Int = 2
    // NOTE: constraints are handled as penalties in objectives (so: 0 constraints)
    override def getNumberOfConstraints(): Int = 0
    override def getName(): String = "SendTimeAllocationProblem"
    
    override def createSolution(): IntegerSolution = {
      // jMetal 5.6: DefaultIntegerSolution(IntegerProblem)
      val solution = new org.uma.jmetal.solution.impl.DefaultIntegerSolution(this)
      
      // 초기 해: 각 사용자를 최고 점수 시간대에 할당 (용량 고려)
      val hourCapacity = mutable.Map(capacityPerHour.toSeq: _*)
      
      for (i <- 0 until numUsers) {
        val user = users(i)
        val choices = userData(user).toSeq.sortBy(-_._2)
        
        var assigned = false
        var selectedHourIdx = 0
        
        for ((hour, score) <- choices if !assigned) {
          if (hourCapacity.getOrElse(hour, 0) > 0) {
            selectedHourIdx = hourToIndex(hour)
            hourCapacity(hour) = hourCapacity(hour) - 1
            assigned = true
          }
        }
        
        // 용량이 없으면 일단 최고 점수 시간대로 (제약 위반)
        if (!assigned && choices.nonEmpty) {
          selectedHourIdx = hourToIndex(choices.head._1)
        }
        
        solution.setVariableValue(i, Integer.valueOf(selectedHourIdx))
      }
      
      solution
    }
    
    override def evaluate(solution: IntegerSolution): Unit = {
      val assignments = (0 until numUsers).map { i =>
        val hourIdx = solution.getVariableValue(i).intValue()
        val hour = indexToHour(hourIdx)
        (users(i), hour)
      }
      
      // 목적 1: 총 propensity score (최대화 -> 음수로 변환)
      var totalScore = 0.0
      val hourUsage = mutable.Map[Int, Int]().withDefaultValue(0)
      var validAssignments = 0
      var invalidAssignments = 0
      
      assignments.foreach { case (user, hour) =>
        if (userData(user).contains(hour)) {
          totalScore += userData(user)(hour)
          hourUsage(hour) += 1
          validAssignments += 1
        } else {
          invalidAssignments += 1
        }
      }
      
      // Penalty 1: invalid assignment (user-hour not available)
      val invalidPenalty = invalidAssignments.toDouble * 1000.0
      
      // Penalty 2: capacity violation (over-capacity count)
      val overCap = hours.map { hour =>
        Math.max(0, hourUsage.getOrElse(hour, 0) - capacityPerHour.getOrElse(hour, 0))
      }.sum
      val capPenalty = overCap.toDouble * 100.0

      // Objective 0: minimize (-score + penalties)
      solution.setObjective(0, -totalScore + invalidPenalty + capPenalty)
      
      // 목적 2: 시간대별 분포의 표준편차 (최소화)
      val avgUsage = validAssignments.toDouble / hours.length
      val variance = hours.map { hour =>
        val usage = hourUsage.getOrElse(hour, 0)
        Math.pow(usage - avgUsage, 2)
      }.sum / hours.length
      val stdDev = Math.sqrt(variance)
      
      // Objective 1: minimize stdDev (+ small penalties to prefer feasible solutions)
      solution.setObjective(1, stdDev + (invalidAssignments.toDouble * 10.0) + (overCap.toDouble * 1.0))
    }

    // ============================================================================
    // IntegerProblem bounds (jMetal 5.6)
    // ============================================================================
    override def getLowerBound(index: Int): Integer = Integer.valueOf(0)
    override def getUpperBound(index: Int): Integer = Integer.valueOf(hours.length - 1)
  }

  /**
   * jMetal NSGA-II 알고리즘을 사용한 다목적 최적화
   */
  def allocateUsersWithJMetalNSGAII(
    df: DataFrame,
    capacityPerHour: Map[Int, Int],
    populationSize: Int = 100,
    maxEvaluations: Int = 25000,
    crossoverProbability: Double = 0.9,
    mutationProbability: Double = 0.1
  ): DataFrame = {
    
    import df.sparkSession.implicits._
    
    println("=" * 80)
    println("jMetal NSGA-II Multi-Objective Optimization")
    println("=" * 80)
    
    val userData = collectUserData(df)
    val users = userData.keys.toArray.sorted
    val hours = df.select("send_hour").distinct().collect().map(_.getInt(0)).sorted
    
    println(s"\n[INPUT INFO]")
    println(s"Users: ${numFormatter.format(users.length)}")
    println(s"Hours: ${hours.length}")
    
    println(s"\n[CAPACITY PER HOUR]")
    capacityPerHour.toSeq.sortBy(_._1).foreach { case (hour, cap) =>
      println(s"  Hour $hour: ${numFormatter.format(cap)}")
    }
    
    val totalCapacity = capacityPerHour.values.sum
    println(f"\n  Total capacity: ${numFormatter.format(totalCapacity)}")
    println(f"  Users to assign: ${numFormatter.format(users.length)}")
    println(f"  Capacity ratio: ${totalCapacity.toDouble / users.length}%.2fx")
    
    println(s"\n[NSGA-II PARAMETERS]")
    println(s"  Population size: $populationSize")
    println(s"  Max evaluations: ${numFormatter.format(maxEvaluations)}")
    println(f"  Crossover probability: $crossoverProbability%.2f")
    println(f"  Mutation probability: $mutationProbability%.2f")
    
    // Problem 정의
    val problem = new SendTimeAllocationProblem(users, userData, capacityPerHour, hours)
    
    // Operators
    val crossover = new IntegerSBXCrossover(crossoverProbability, 20.0)
    val mutation = new IntegerPolynomialMutation(
      problem,
      mutationProbability / users.length
    )
    val selection = new BinaryTournamentSelection[IntegerSolution](
      new RankingAndCrowdingDistanceComparator[IntegerSolution]()
    )
    
    // NSGA-II 알고리즘
    val algorithm = new NSGAIIBuilder[IntegerSolution](
      problem, 
      crossover, 
      mutation
    )
      .setSelectionOperator(selection)
      .setPopulationSize(populationSize)
      .setMaxEvaluations(maxEvaluations)
      .build()
    
    println("\nRunning NSGA-II...")
    val startTime = System.currentTimeMillis()
    algorithm.run()
    val runTime = (System.currentTimeMillis() - startTime) / 1000.0
    
    // jMetal 5.x: getResult(), jMetal 6.x: result()
    val paretoFront: JavaList[IntegerSolution] = {
      val cls = algorithm.getClass
      val m =
        try cls.getMethod("result")
        catch { case _: NoSuchMethodException => cls.getMethod("getResult") }
      m.invoke(algorithm).asInstanceOf[JavaList[IntegerSolution]]
    }
    println(f"\nOptimization completed in $runTime%.2f seconds")
    println(s"Pareto front size: ${paretoFront.size()}")
    
    // Pareto front에서 최선의 해 선택 (총 점수 기준)
    val bestSolution = paretoFront.asScala.minBy((sol: IntegerSolution) => sol.getObjective(0))(Ordering.Double)
    
    val bestScore = -bestSolution.getObjective(0)  // 원래 값으로 복원(패널티 포함일 수 있음)
    val bestStdDev = bestSolution.getObjective(1)
    
    println(f"\nBest solution (highest total score):")
    println(f"  Total score: $bestScore%,.2f")
    println(f"  Load balance (std dev): $bestStdDev%.2f")
    
    // 제약 위반 확인
    // Constraints are modeled as penalties (numberOfConstraints = 0)
    
    // 해를 AllocationResult로 변환
    val results = (0 until users.length).map { i =>
      val user = users(i)
      val hourIdx = bestSolution.getVariableValue(i).intValue()
      val hour = hours(hourIdx)
      val score = userData(user).getOrElse(hour, 0.0)
      AllocationResult(user, hour, score)
    }.toArray
    
    validateAndPrintResults(results, users, hours, capacityPerHour)
    
    results.toSeq.toDF()
  }

  /**
   * jMetal MOEA/D 알고리즘을 사용한 다목적 최적화
   */
  def allocateUsersWithJMetalMOEAD(
    df: DataFrame,
    capacityPerHour: Map[Int, Int],
    populationSize: Int = 100,
    maxEvaluations: Int = 25000,
    neighborhoodSize: Int = 20
  ): DataFrame = {
    
    import df.sparkSession.implicits._
    
    // NOTE:
    // jMetal 6.1의 MOEADBuilder는 기본적으로 DoubleSolution 기반 Problem을 기대합니다.
    // 현재 구현은 IntegerSolution 기반이므로 spark-shell에서 타입 에러가 발생합니다.
    // 우선 컴파일/실행 안정성을 위해 NSGA-II로 graceful fallback 합니다.
    println("=" * 80)
    println("jMetal MOEA/D requested (fallback to NSGA-II)")
    println("=" * 80)
    println("⚠ jMetal 6.1 MOEA/D is DoubleSolution-oriented; current IntegerSolution encoding falls back to NSGA-II.")
    println(s"  (populationSize=$populationSize, maxEvaluations=$maxEvaluations, neighborhoodSize=$neighborhoodSize ignored)")

    allocateUsersWithJMetalNSGAII(
      df = df,
      capacityPerHour = capacityPerHour,
      populationSize = populationSize,
      maxEvaluations = maxEvaluations
    )
  }

  /**
   * Hybrid: jMetal + Greedy 조합
   */
  def allocateUsersHybridJMetal(
    df: DataFrame,
    capacityPerHour: Map[Int, Int],
    algorithm: String = "NSGAII",  // "NSGAII" or "MOEAD"
    populationSize: Int = 100,
    maxEvaluations: Int = 25000
  ): DataFrame = {
    
    import df.sparkSession.implicits._
    
    try {
      val jmetalResult = algorithm.toUpperCase match {
        case "NSGAII" => 
          allocateUsersWithJMetalNSGAII(df, capacityPerHour, populationSize, maxEvaluations)
        case "MOEAD" => 
          allocateUsersWithJMetalMOEAD(df, capacityPerHour, populationSize, maxEvaluations)
        case _ => 
          println(s"Unknown algorithm: $algorithm, using NSGA-II")
          allocateUsersWithJMetalNSGAII(df, capacityPerHour, populationSize, maxEvaluations)
      }
      
      val assignedUsers = jmetalResult.select("svc_mgmt_num").collect().map(_.getString(0)).toSet
      val allUsers = df.select("svc_mgmt_num").distinct().collect().map(_.getString(0)).toSet
      val unassignedUsers = allUsers -- assignedUsers
      
      if (unassignedUsers.isEmpty) {
        println("✓ All users assigned by jMetal optimizer")
        return jmetalResult
      }
      
      println(s"\n${numFormatter.format(unassignedUsers.size)} users unassigned, running Greedy for remainder...")
      
      val usedCapacity = jmetalResult.groupBy("assigned_hour").count().collect()
        .map(r => r.getInt(0) -> r.getLong(1).toInt).toMap
      
      val remainingCapacity = capacityPerHour.map { case (hour, cap) =>
        hour -> Math.max(0, cap - usedCapacity.getOrElse(hour, 0))
      }
      
      val unassignedDf = df.filter($"svc_mgmt_num".isin(unassignedUsers.toSeq: _*))
      val hours = df.select("send_hour").distinct().collect().map(_.getInt(0)).sorted
      val greedyResult = allocateGreedySimple(unassignedDf, hours, remainingCapacity)
      
      if (greedyResult.count() == 0) {
        return jmetalResult
      }
      
      jmetalResult.union(greedyResult)
      
    } catch {
      case e: Exception =>
        println(s"\njMetal optimizer failed: ${e.getMessage}")
        println("Running full Greedy allocation...")
        val hours = df.select("send_hour").distinct().collect().map(_.getInt(0)).sorted
        allocateGreedySimple(df, hours, capacityPerHour)
    }
  }

  /**
   * 대규모 배치 처리 (jMetal 기반)
   */
  def allocateLargeScaleJMetal(
    df: DataFrame,
    capacityPerHour: Map[Int, Int],
    batchSize: Int = 50000,  // jMetal은 메모리 사용량이 크므로 작은 배치 크기
    algorithm: String = "NSGAII",
    populationSize: Int = 100,
    maxEvaluations: Int = 25000
  ): DataFrame = {
    
    import df.sparkSession.implicits._
    
    println("=" * 80)
    println(s"Batch Allocation for Large Scale Data (jMetal $algorithm Mode)")
    println("=" * 80)
    
    val userPriority = df.groupBy("svc_mgmt_num")
      .agg(max("propensity_score").as("max_prob"))
    
    val totalUsers = userPriority.count()
    val numBatches = Math.ceil(totalUsers.toDouble / batchSize).toInt
    
    println(s"\n[BATCH SETUP]")
    println(s"Total users: ${numFormatter.format(totalUsers)}")
    println(s"Batch size: ${numFormatter.format(batchSize)}")
    println(s"Number of batches: $numBatches")
    
    val allUsers = userPriority
      .withColumn("row_id", row_number().over(Window.orderBy(desc("max_prob"))))
      .withColumn("batch_id", (($"row_id" - 1) / batchSize).cast("int"))
      .select("svc_mgmt_num", "batch_id")
      .cache()
    
    var remainingCapacity = capacityPerHour.toMap
    val allResults = mutable.ArrayBuffer[DataFrame]()
    var totalAssignedSoFar = 0L
    
    for (batchId <- 0 until numBatches) {
      println(s"\n${"=" * 80}")
      println(s"Processing Batch ${batchId + 1}/$numBatches")
      println(s"${"=" * 80}")
      
      val batchUsers = allUsers.filter($"batch_id" === batchId)
      val availableHours = remainingCapacity.filter(_._2 > 0).keys.toSeq
      
      if (availableHours.isEmpty) {
        println("⚠ No capacity left in any hour.")
      } else {
        val batchDf = df.join(batchUsers, Seq("svc_mgmt_num"))
          .filter($"send_hour".isin(availableHours: _*))
        
        val batchResult = allocateUsersHybridJMetal(
          batchDf,
          remainingCapacity,
          algorithm,
          populationSize,
          maxEvaluations
        )
        
        val assignedCount = batchResult.count()
        
        if (assignedCount > 0) {
          totalAssignedSoFar += assignedCount
          
          val allocatedPerHour = batchResult.groupBy("assigned_hour").count().collect()
            .map(row => row.getInt(0) -> row.getLong(1).toInt).toMap
          
          remainingCapacity = remainingCapacity.map { case (hour, cap) =>
            hour -> Math.max(0, cap - allocatedPerHour.getOrElse(hour, 0))
          }
          
          allResults += batchResult
        }
      }
      
      val progress = totalAssignedSoFar.toDouble / totalUsers * 100
      println(f"\n[PROGRESS] Assigned: ${numFormatter.format(totalAssignedSoFar)} / ${numFormatter.format(totalUsers)} users ($progress%.1f%%)")
    }
    
    allUsers.unpersist()
    
    if (allResults.isEmpty) {
      df.sparkSession.emptyDataFrame
    } else {
      val finalResult = allResults.reduce(_.union(_))
      printFinalStatistics(finalResult, totalUsers)
      finalResult
    }
  }

  // ============================================================================
  // Usage Examples (for Interactive Spark Shell)
  // ============================================================================
  
  /**
   * 사용 예제:
   * 
   * spark-shell에서:
   * scala> :load optimize_ost.scala
   * scala> import OptimizeSendTime._
   * 
   * 이후 모든 함수를 직접 사용 가능
   */
  
  // Main 메서드는 spark-submit용으로만 사용 (주석 처리됨)
  /*
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("Optimize Send Time")
      .getOrCreate()

    import spark.implicits._

    // 데이터 로드
    val dfAll = spark.read.parquet("aos/sto/propensityScoreDF").cache()
    val df = dfAll.filter("svc_mgmt_num like '%00'")

    println(s"Total records: ${df.count()}")

    val runtime = Runtime.getRuntime
    val maxMemoryGB = runtime.maxMemory() / (1024.0 * 1024 * 1024)
    println(f"Available memory: $maxMemoryGB%.2f GB")

    val safeBatchSize = (maxMemoryGB * 1000000 / 10 * 0.05).toInt
    println(s"Safe batch size: ${numFormatter.format(safeBatchSize)}")

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

    // 시간대별 할당 현황
    result.groupBy("assigned_hour")
      .agg(
        count("*").as("count"),
        sum("score").as("total_score"),
        avg("score").as("avg_score")
      )
      .orderBy("assigned_hour")
      .show(false)

    spark.stop()
  }
  */
  
  // ============================================================================
  // 초기화 메시지
  // ============================================================================
  
  println("""
================================================================================
Optimize Send Time - User Allocation Optimizer
================================================================================

✓ Loaded successfully!

To use the optimizer functions:
  import OptimizeSendTime._

Available algorithms:
  1. allocateGreedySimple                   - Fastest, simple greedy allocation
  2. allocateUsersWithHourlyCapacity        - OR-Tools optimization (accurate)
  3. allocateUsersWithSimulatedAnnealing    - SA optimization (balanced)
  4. allocateUsersHybrid                    - Hybrid (OR-Tools + Greedy)
  5. allocateLargeScaleHybrid               - Large scale batch processing (OR-Tools)
  6. allocateUsersHybridSA                  - Hybrid (SA + Greedy)
  7. allocateLargeScaleSA                   - Large scale SA batch processing
  8. allocateUsersWithJMetalNSGAII          - jMetal NSGA-II multi-objective
  9. allocateUsersWithJMetalMOEAD           - jMetal MOEA/D multi-objective
  10. allocateUsersHybridJMetal             - Hybrid (jMetal + Greedy)
  11. allocateLargeScaleJMetal              - Large scale jMetal batch processing

Quick start example:
  import OptimizeSendTime._
  val dfAll = spark.read.parquet("aos/sto/propensityScoreDF").cache()
  val df = dfAll.limit(1000)
  val capacity = Map(9->100, 10->100, 11->100, 12->100, 13->100,
                      14->100, 15->100, 16->100, 17->100, 18->100)
  val result = allocateGreedySimple(df, Array(9,10,11,12,13,14,15,16,17,18), capacity)
  result.groupBy("assigned_hour").count().show()

For more examples, see: QUICK_START.md and JMETAL_SETUP.md
================================================================================
""")
}