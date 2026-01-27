error id: file://<WORKSPACE>/optimize_send_time/compare_jmetal_vs_greedy.scala:distinct
file://<WORKSPACE>/optimize_send_time/compare_jmetal_vs_greedy.scala
empty definition using pc, found symbol in pc: 
semanticdb not found

found definition using fallback; symbol distinct
offset: 1757
uri: file://<WORKSPACE>/optimize_send_time/compare_jmetal_vs_greedy.scala
text:
```scala
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
  def run(): Unit = {
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
    val df = dfAll.filter("svc_mgmt_num like '%%'")

    val userCnt = df.select("svc_mgmt_num").distinct@@().count()
    val recordCnt = df.count()

    println(s"✓ Users: ${OptimizeSendTime.numFormatter.format(userCnt)}")
    println(s"✓ Records: ${OptimizeSendTime.numFormatter.format(recordCnt)}")

    val hours = df.select("send_hour").distinct().collect().map(_.getInt(0)).sorted
    println(s"Hours: ${hours.mkString(", ")}")

    // Same capacity rule for both algorithms
    val capacityPerHour = Math.max(1, (userCnt * 0.11).toInt)
    val capacityPerHourMap = hours.map(h => h -> capacityPerHour).toMap

    println(s"Capacity per hour: ${OptimizeSendTime.numFormatter.format(capacityPerHour)}")
    println(s"Total capacity: ${OptimizeSendTime.numFormatter.format(capacityPerHourMap.values.sum)}")

    // jMetal parameters (keep moderate)
    val populationSize = 80
    val maxEvaluations = 8000

    // --------------------------------------------------------------------------
    // 1) Greedy
    // --------------------------------------------------------------------------
    println("\n" + "=" * 80)
    println("Running Greedy")
    println("=" * 80)

    val t0 = System.currentTimeMillis()
    val greedyResult = GreedyAllocator.allocate(df, hours, capacityPerHourMap)
    val t1 = System.currentTimeMillis()
    val greedySec = (t1 - t0) / 1000.0

    val greedyScore = greedyResult.agg(sum("score")).first().getDouble(0)
    val greedyAssigned = greedyResult.count()

    println(f"✓ Greedy done in $greedySec%.2f sec")
    println(f"✓ Greedy assigned: ${OptimizeSendTime.numFormatter.format(greedyAssigned)} users")
    println(f"✓ Greedy total score: $greedyScore%,.2f")

    // --------------------------------------------------------------------------
    // 2) jMetal NSGA-II
    // --------------------------------------------------------------------------
    println("\n" + "=" * 80)
    println("Running jMetal NSGA-II")
    println("=" * 80)
    println(s"Population size: $populationSize")
    println(s"Max evaluations: ${OptimizeSendTime.numFormatter.format(maxEvaluations)}")

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

    // --------------------------------------------------------------------------
    // Summary
    // --------------------------------------------------------------------------
    println("\n" + "=" * 80)
    println("Summary (same df / capacity)")
    println("=" * 80)

    def pct(delta: Double, base: Double): Double = if (base == 0) 0.0 else (delta / base) * 100.0

    val scoreDelta = nsga2Score - greedyScore
    val timeDelta = nsga2Sec - greedySec

    println(f"Greedy  : time=$greedySec%.2f sec, score=$greedyScore%,.2f, assigned=${OptimizeSendTime.numFormatter.format(greedyAssigned)}")
    println(f"NSGA-II : time=$nsga2Sec%.2f sec, score=$nsga2Score%,.2f, assigned=${OptimizeSendTime.numFormatter.format(nsga2Assigned)}")
    println(f"Δ score : $scoreDelta%,.2f (${pct(scoreDelta, greedyScore)}%.2f%% vs Greedy)")
    println(f"Δ time  : $timeDelta%.2f sec (${pct(timeDelta, greedySec)}%.2f%% vs Greedy)")

    println("\nGreedy by hour:")
    greedyResult.groupBy("assigned_hour").count().orderBy("assigned_hour").show(false)

    println("\nNSGA-II by hour:")
    nsga2Result.groupBy("assigned_hour").count().orderBy("assigned_hour").show(false)

    println("""
================================================================================
Compare complete.
================================================================================
""")
  }
}


```


#### Short summary: 

empty definition using pc, found symbol in pc: 