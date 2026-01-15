// ============================================================================
// Compare Script: jMetal NSGA-II vs Greedy (same conditions)
// ============================================================================
//
// Usage (권장):
//   SPARK_SHELL_XSS=8m ./spark-shell-with-lib.sh \
//     -i optimize_ost.scala \
//     -i greedy_allocation.scala \
//     -i compare_jmetal_vs_greedy.scala \
//     -i compare_jmetal_vs_greedy.sc
//
// Notes:
// - This file assumes OptimizeSendTime and GreedyAllocator are already loaded.
// - Runs both algorithms on the same df / hours / capacityPerHourMap.
// ============================================================================

object CompareJMetalVsGreedy {
  /**
   * @param capacityRatios hour당 capacity 비율 (예: 0.05 => users*0.05)
   * @param sampleFilter   Spark SQL filter 조건 (기본: 전체)
   * @param populationSize NSGA-II population size
   * @param maxEvaluations NSGA-II max evaluations
   */
  def run(
    capacityRatios: Seq[Double] = Seq(0.05, 0.08, 0.11),
    sampleFilter: String = "svc_mgmt_num like '%%'",
    populationSize: Int = 80,
    maxEvaluations: Int = 8000
  ): Unit = {
    println("""
================================================================================
Compare: jMetal(NSGA-II) vs Greedy (same conditions)
================================================================================
""")

    import OptimizeSendTime._
    import GreedyAllocator._

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
  ./generate_data_simple.sh 1000
================================================================================
""")
      return
    }

    println("Loading sample data...")
    val dfAll = spark.read.parquet(sampleDataPath).cache()

    // Same sampling rule as load_and_test_jmetal.scala (adjust if needed)
    val df = dfAll.filter(sampleFilter)

    val userCnt = df.select("svc_mgmt_num").distinct().count()
    val recordCnt = df.count()

    println(s"✓ Users: ${OptimizeSendTime.numFormatter.format(userCnt)}")
    println(s"✓ Records: ${OptimizeSendTime.numFormatter.format(recordCnt)}")

    val hours = df.select("send_hour").distinct().collect().map(_.getInt(0)).sorted
    println(s"Hours: ${hours.mkString(", ")}")

    println(s"Capacity ratios: ${capacityRatios.mkString(", ")}")
    println(s"NSGA-II params: populationSize=$populationSize, maxEvaluations=${OptimizeSendTime.numFormatter.format(maxEvaluations)}")

    case class Row(
      ratio: Double,
      capacityPerHour: Int,
      totalCapacity: Long,
      greedySec: Double,
      greedyScore: Double,
      greedyAssigned: Long,
      nsga2Sec: Double,
      nsga2Score: Double,
      nsga2Assigned: Long
    )

    val rows = scala.collection.mutable.ArrayBuffer[Row]()

    capacityRatios.foreach { ratio =>
      val capacityPerHour = Math.max(1, Math.floor(userCnt.toDouble * ratio).toInt)
      val capacityPerHourMap = hours.map(h => h -> capacityPerHour).toMap

      println("\n" + "=" * 80)
      println(f"CASE: capacityRatio=$ratio%.3f  (capacityPerHour=${OptimizeSendTime.numFormatter.format(capacityPerHour)})")
      println("=" * 80)
      println(s"Total capacity: ${OptimizeSendTime.numFormatter.format(capacityPerHourMap.values.sum)}")

      // ----------------------------------------------------------------------
      // 1) Greedy
      // ----------------------------------------------------------------------
      println("\n" + "-" * 80)
      println("Running Greedy")
      println("-" * 80)

      val t0 = System.currentTimeMillis()
      val greedyResult = GreedyAllocator.allocate(df, hours, capacityPerHourMap)
      val t1 = System.currentTimeMillis()
      val greedySec = (t1 - t0) / 1000.0

      val greedyScore = greedyResult.agg(sum("score")).first().getDouble(0)
      val greedyAssigned = greedyResult.count()

      println(f"✓ Greedy done in $greedySec%.2f sec")
      println(f"✓ Greedy assigned: ${OptimizeSendTime.numFormatter.format(greedyAssigned)} users")
      println(f"✓ Greedy total score: $greedyScore%,.2f")

      // ----------------------------------------------------------------------
      // 2) jMetal NSGA-II
      // ----------------------------------------------------------------------
      println("\n" + "-" * 80)
      println("Running jMetal NSGA-II")
      println("-" * 80)

      val t2 = System.currentTimeMillis()
      val nsga2Result = allocateUsersWithJMetalNSGAII(
        df = df,
        capacityPerHour = capacityPerHourMap,
        populationSize = populationSize,
        maxEvaluations = maxEvaluations
      )
      val t3 = System.currentTimeMillis()
      val nsga2Sec = (t3 - t2) / 1000.0

      val nsga2Score = nsga2Result.agg(sum("score")).first().getDouble(0)
      val nsga2Assigned = nsga2Result.count()

      println(f"✓ NSGA-II done in $nsga2Sec%.2f sec")
      println(f"✓ NSGA-II assigned: ${OptimizeSendTime.numFormatter.format(nsga2Assigned)} users")
      println(f"✓ NSGA-II total score: $nsga2Score%,.2f")

      rows += Row(
        ratio = ratio,
        capacityPerHour = capacityPerHour,
        totalCapacity = capacityPerHourMap.values.sum.toLong,
        greedySec = greedySec,
        greedyScore = greedyScore,
        greedyAssigned = greedyAssigned,
        nsga2Sec = nsga2Sec,
        nsga2Score = nsga2Score,
        nsga2Assigned = nsga2Assigned
      )

      println("\nGreedy by hour:")
      greedyResult.groupBy("assigned_hour").count().orderBy("assigned_hour").show(false)

      println("\nNSGA-II by hour:")
      nsga2Result.groupBy("assigned_hour").count().orderBy("assigned_hour").show(false)
    }

    // ------------------------------------------------------------------------
    // Summary (all ratios)
    // ------------------------------------------------------------------------
    println("\n" + "=" * 80)
    println("Summary (all ratios)")
    println("=" * 80)

    def pct(delta: Double, base: Double): Double = if (base == 0) 0.0 else (delta / base) * 100.0

    println(f"${"ratio"}%7s  ${"cap/hr"}%10s  ${"Greedy(s)"}%10s  ${"NSGA(s)"}%10s  ${"Δtime%"}%9s  ${"GreedyScore"}%14s  ${"NSGAScore"}%14s  ${"Δscore%"}%9s")
    rows.foreach { r =>
      val timeDelta = r.nsga2Sec - r.greedySec
      val scoreDelta = r.nsga2Score - r.greedyScore
      println(
        f"${r.ratio}%7.3f  ${OptimizeSendTime.numFormatter.format(r.capacityPerHour)}%10s  " +
          f"${r.greedySec}%10.2f  ${r.nsga2Sec}%10.2f  ${pct(timeDelta, r.greedySec)}%9.2f  " +
          f"${r.greedyScore}%14.2f  ${r.nsga2Score}%14.2f  ${pct(scoreDelta, r.greedyScore)}%9.2f"
      )
    }

    println("""
================================================================================
Compare complete.
================================================================================
""")
  }
}

