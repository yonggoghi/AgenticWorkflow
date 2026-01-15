// ============================================================================
// Quick Test Script for jMetal (NSGA-II / MOEA-D) in Spark Shell
// ============================================================================
//
// Usage:
//   (권장) stack을 올려 spark-shell :load 크래시를 줄이기
//     SPARK_SHELL_XSS=8m ./spark-shell-with-lib.sh -i optimize_ost.scala -i load_and_test_jmetal.scala
//
//   또는 (환경/스크립트에 따라)
//     spark-shell --jars "$ALL_OPTIMIZER_JARS" --driver-java-options "-Xss8m" -i optimize_ost.scala -i load_and_test_jmetal.scala
//
// 주의: 이 파일은 OptimizeSendTime이 이미 로드되었다는 가정하에 동작합니다.
// ============================================================================

object LoadAndTestJMetal {
  def run(): Unit = {
    println("""
================================================================================
Loading Optimize Send Time (jMetal test)...
================================================================================
""")

    // 1) 함수 import (optimize_ost.scala 가 먼저 로드되어 있어야 함)
    import OptimizeSendTime._

    println("""
================================================================================
Running jMetal Quick Test (NSGA-II only)
================================================================================
""")

    // 3) 샘플 데이터 확인
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

Then run this script again.
================================================================================
""")
    } else {
      try {
        println("Loading sample data...")
        val dfAll = spark.read.parquet(sampleDataPath).cache()

        // NOTE: jMetal은 메모리/시간 사용량이 크므로 기본은 작은 샘플로 실행
        // - suffix 필터는 데이터 생성 방식에 따라 조정하세요.
        val df = dfAll.filter("svc_mgmt_num like '%0'")

        val userCnt = df.select("svc_mgmt_num").distinct().count()
        val recordCnt = df.count()

        println(s"✓ Loaded users: ${numFormatter.format(userCnt)}")
        println(s"✓ Loaded records: ${numFormatter.format(recordCnt)}")

        // 시간대 배열
        val hours = df.select("send_hour").distinct().collect().map(_.getInt(0)).sorted
        println(s"Hours: ${hours.mkString(", ")}")

        // 용량 설정 (시간대별 동일 용량)
        val capacityPerHour = Math.max(1, (userCnt * 0.2).toInt)
        val capacityPerHourMap = hours.map(h => h -> capacityPerHour).toMap

        println(s"Capacity per hour: ${numFormatter.format(capacityPerHour)}")

        // jMetal 파라미터 (기본은 가볍게)
        val populationSize = 80
        val maxEvaluations = 8000

        println("\n" + "=" * 80)
        println("Running jMetal NSGA-II")
        println("=" * 80)
        println(s"Population size: $populationSize")
        println(s"Max evaluations: ${numFormatter.format(maxEvaluations)}")
        println("")

        val t0 = System.currentTimeMillis()
        val resultNSGAII = allocateUsersWithJMetalNSGAII(
          df = df,
          capacityPerHour = capacityPerHourMap,
          populationSize = populationSize,
          maxEvaluations = maxEvaluations
        )
        val t1 = System.currentTimeMillis()
        println(f"✓ NSGA-II done in ${(t1 - t0) / 1000.0}%.2f sec")

        println("\nNSGA-II: Allocation by hour")
        resultNSGAII.groupBy("assigned_hour")
          .agg(
            count("*").as("count"),
            sum("score").as("total_score"),
            avg("score").as("avg_score")
          )
          .orderBy("assigned_hour")
          .show(false)

        println("""
================================================================================
jMetal Quick Test Complete!
================================================================================

Tips:
  - Greedy와 성능 비교는 compare_jmetal_vs_greedy.scala 를 사용하세요.
  - :load 크래시가 반복되면:
      SPARK_SHELL_XSS=8m ./spark-shell-with-lib.sh -i optimize_ost.scala -i load_and_test_jmetal.scala -i load_and_test_jmetal.sc
    처럼 stack을 올려 실행하세요.
================================================================================
""")
      } catch {
        case e: java.lang.UnsupportedClassVersionError =>
          println("""
================================================================================
✗ UnsupportedClassVersionError (Java 버전 불일치)
================================================================================
현재 spark-shell의 Java 런타임이 낮아서(jdk8=52) jMetal JAR(jdk11=55)을 로드할 수 없습니다.

해결 방법(둘 중 하나):
  1) jMetal을 Java 8 호환 버전(권장: 5.6)으로 재다운로드
       rm -f lib/jmetal-*.jar
       JMETAL_VERSION=5.6 ./download_jmetal.sh

  2) spark-shell을 Java 11+로 실행
       (JAVA_HOME을 11+로 설정한 뒤 spark-shell 재실행)

================================================================================
""")
          throw e
      }
    }
  }
}

