import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.sql.functions._
import scala.util.Random

/**
 * 테스트용 샘플 데이터 생성 스크립트
 * 
 * 스키마:
 * - svc_mgmt_num: String (사용자 ID)
 * - send_ym: String (발송 년월, 202512로 고정)
 * - send_hour: Int (발송 시간, 9~18시)
 * - propensity_score: Double (반응 확률, 0.1~0.99)
 * 
 * 생성 데이터:
 * - 사용자: 100,000명
 * - 각 사용자당 시간대: 10개 (9~18시)
 * - 총 레코드: 1,000,000개
 */
object GenerateSampleData {
  
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("Generate Sample Data")
      .master("local[*]")
      .config("spark.driver.memory", "4g")
      .getOrCreate()

    import spark.implicits._

    println("=" * 80)
    println("Sample Data Generation for Optimize Send Time")
    println("=" * 80)
    
    val numUsers = 100000
    val sendYm = "202512"
    val hours = 9 to 18
    
    println(s"\n[Configuration]")
    println(s"  Users: ${numUsers}")
    println(s"  Send YM: ${sendYm}")
    println(s"  Hours: ${hours.mkString(", ")}")
    println(s"  Total records: ${numUsers * hours.length}")
    
    println("\n[Generating data...]")
    val startTime = System.currentTimeMillis()
    
    // 사용자 ID 생성 (s:0063c2994b5452d... 형식)
    val users = (0 until numUsers).map { i =>
      f"s:${i%10}${Random.nextInt(10)}${Random.alphanumeric.take(12).mkString.toLowerCase}${i%100000}%05d"
    }
    
    // 데이터 생성
    val data = for {
      user <- users
      hour <- hours
    } yield {
      // 각 사용자마다 선호 시간대를 설정하여 현실적인 분포 생성
      val preferredHour = 9 + (user.hashCode().abs % 10)
      val distance = Math.abs(hour - preferredHour)
      
      // 선호 시간대에 가까울수록 높은 점수
      val baseScore = 0.5 + (Random.nextDouble() * 0.3)  // 0.5 ~ 0.8
      val distancePenalty = distance * 0.05
      val score = Math.max(0.1, Math.min(0.99, baseScore - distancePenalty + (Random.nextDouble() * 0.2 - 0.1)))
      
      (user, sendYm, hour, score)
    }
    
    // DataFrame 생성
    val df = data.toDF("svc_mgmt_num", "send_ym", "send_hour", "propensity_score")
    
    val generateTime = (System.currentTimeMillis() - startTime) / 1000.0
    println(f"  ✓ Data generated in $generateTime%.2f seconds")
    
    // 통계 출력
    println("\n[Data Statistics]")
    println(s"  Total records: ${df.count()}")
    println(s"  Unique users: ${df.select("svc_mgmt_num").distinct().count()}")
    
    println("\n[Sample data (first 20 rows)]")
    df.orderBy("svc_mgmt_num", "send_hour").show(20, truncate = false)
    
    println("\n[Propensity score statistics]")
    df.select(
      min("propensity_score").as("min"),
      max("propensity_score").as("max"),
      avg("propensity_score").as("avg"),
      stddev("propensity_score").as("stddev")
    ).show(false)
    
    println("\n[Records per hour]")
    df.groupBy("send_hour")
      .count()
      .orderBy("send_hour")
      .show(false)
    
    println("\n[Records per user (sample of 5 users)]")
    val sampleUsers = df.select("svc_mgmt_num").distinct().limit(5).collect().map(_.getString(0))
    df.filter($"svc_mgmt_num".isin(sampleUsers: _*))
      .groupBy("svc_mgmt_num")
      .count()
      .show(false)
    
    // Parquet로 저장
    val outputPath = "aos/sto/propensityScoreDF"
    println(s"\n[Saving to Parquet]")
    println(s"  Output path: $outputPath")
    
    val saveStartTime = System.currentTimeMillis()
    df.write
      .mode("overwrite")
      .parquet(outputPath)
    val saveTime = (System.currentTimeMillis() - saveStartTime) / 1000.0
    
    println(f"  ✓ Data saved in $saveTime%.2f seconds")
    
    // 저장된 데이터 검증
    println("\n[Verifying saved data]")
    val loadedDf = spark.read.parquet(outputPath)
    println(s"  Loaded records: ${loadedDf.count()}")
    println(s"  Schema:")
    loadedDf.printSchema()
    
    println("\n" + "=" * 80)
    println("Sample Data Generation Complete!")
    println("=" * 80)
    println(s"\nYou can now use this data with:")
    println(s"""  val dfAll = spark.read.parquet("$outputPath").cache()""")
    println()
    
    spark.stop()
  }
  
  /**
   * Spark Shell에서 사용할 수 있는 함수 버전
   */
  def generateSampleData(
    spark: SparkSession,
    numUsers: Int = 100000,
    sendYm: String = "202512",
    outputPath: String = "aos/sto/propensityScoreDF"
  ): DataFrame = {
    
    import spark.implicits._
    
    val hours = 9 to 18
    val random = new Random(42)  // 재현 가능한 결과를 위해 seed 사용
    
    println(s"Generating sample data for $numUsers users...")
    
    // 사용자 ID 생성
    val users = (0 until numUsers).map { i =>
      f"s:${i%10}${random.nextInt(10)}${random.alphanumeric.take(12).mkString.toLowerCase}${i%100000}%05d"
    }
    
    // 데이터 생성
    val data = for {
      user <- users
      hour <- hours
    } yield {
      val preferredHour = 9 + (user.hashCode().abs % 10)
      val distance = Math.abs(hour - preferredHour)
      val baseScore = 0.5 + (random.nextDouble() * 0.3)
      val distancePenalty = distance * 0.05
      val score = Math.max(0.1, Math.min(0.99, baseScore - distancePenalty + (random.nextDouble() * 0.2 - 0.1)))
      
      (user, sendYm, hour, score)
    }
    
    val df = data.toDF("svc_mgmt_num", "send_ym", "send_hour", "propensity_score")
    
    // 저장
    df.write.mode("overwrite").parquet(outputPath)
    
    println(s"✓ Generated ${df.count()} records")
    println(s"✓ Saved to $outputPath")
    
    df
  }
  
  /**
   * 소규모 테스트용 데이터 생성 (메모리에만 유지)
   */
  def generateSmallSampleData(
    spark: SparkSession,
    numUsers: Int = 1000
  ): DataFrame = {
    
    import spark.implicits._
    
    val hours = 9 to 18
    val sendYm = "202512"
    val random = new Random(42)
    
    val users = (0 until numUsers).map { i =>
      f"s:${i%10}${random.nextInt(10)}${random.alphanumeric.take(12).mkString.toLowerCase}${i%10000}%05d"
    }
    
    val data = for {
      user <- users
      hour <- hours
    } yield {
      val preferredHour = 9 + (user.hashCode().abs % 10)
      val distance = Math.abs(hour - preferredHour)
      val baseScore = 0.5 + (random.nextDouble() * 0.3)
      val distancePenalty = distance * 0.05
      val score = Math.max(0.1, Math.min(0.99, baseScore - distancePenalty + (random.nextDouble() * 0.2 - 0.1)))
      
      (user, sendYm, hour, score)
    }
    
    val df = data.toDF("svc_mgmt_num", "send_ym", "send_hour", "propensity_score")
    
    println(s"✓ Generated ${df.count()} records (${numUsers} users)")
    df.show(20, false)
    
    df
  }
}
