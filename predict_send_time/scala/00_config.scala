// =====================================================================================
// predict_ost - spark-shell scala scripts
//
// Usage (spark-shell):
//   :load predict_send_time/scala/00_config.scala
//   :load predict_send_time/scala/10_response_extract_save.scala
//   :load predict_send_time/scala/20_build_joined_datasets.scala
//   :load predict_send_time/scala/30_vectorize_fit_transform_save.scala
//   :load predict_send_time/scala/40_train_eval_and_save_models.scala
//   :load predict_send_time/scala/50_build_pred_inputs.scala
//   :load predict_send_time/scala/60_score_and_save.scala
//
// Environment overrides (optional):
//   SEND_MONTH=202512 FEATURE_MONTH=202511 PERIOD_M=3 \
//   PREDICTION_DT_STA=20251101 PREDICTION_DT_END=20251201 \
//   PRED_DT=20251201 SUFFIX=0,1,2 \
//   spark-shell ...
// =====================================================================================

object PredictOstConfig {
  // [ZEPPELIN_PARAGRAPH] 00. Imports & Spark base settings
  import org.apache.spark.sql.DataFrame

  import java.time.{LocalDate => JLocalDate}
  import java.time.YearMonth
  import java.time.format.DateTimeFormatter
  import java.time.temporal.ChronoUnit
  import scala.collection.mutable.ListBuffer

  def initSpark(): Unit = {
    spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")
    spark.sparkContext.setCheckpointDir(sys.env.getOrElse("CHECKPOINT_DIR", "hdfs://scluster/user/g1110566/checkpoint"))
  }

  // [ZEPPELIN_PARAGRAPH] 01. Common params (override via env)
  val sendMonth: String = sys.env.getOrElse("SEND_MONTH", "202512") // yyyyMM
  val featureMonth: String = sys.env.getOrElse("FEATURE_MONTH", "202511") // yyyyMM
  val periodM: Int = sys.env.getOrElse("PERIOD_M", "3").toInt

  val predictionDTSta: String = sys.env.getOrElse("PREDICTION_DT_STA", "20251101") // yyyyMMdd
  val predictionDTEnd: String = sys.env.getOrElse("PREDICTION_DT_END", "20251201") // yyyyMMdd

  val predDT: String = sys.env.getOrElse("PRED_DT", "20251201") // yyyyMMdd (inference 기준일)

  // comma-separated suffixes (e.g. "0,1,2,a,b,c"). Empty or "%" means "no filter".
  val suffixCsv: String = sys.env.getOrElse("SUFFIX", "%")

  // Time window
  val startHour: Int = sys.env.getOrElse("START_HOUR", "9").toInt
  val endHour: Int = sys.env.getOrElse("END_HOUR", "18").toInt
  val hourRange: List[Int] = (startHour to endHour).toList

  // Storage paths
  val responsePath: String = sys.env.getOrElse("PATH_RESPONSE", "aos/sto/response")
  val trainDFRevPath: String = sys.env.getOrElse("PATH_TRAIN_DF_REV", "aos/sto/trainDFRev")
  val testDFRevPath: String = sys.env.getOrElse("PATH_TEST_DF_REV", "aos/sto/testDFRev")

  val transformerClickPath: String = sys.env.getOrElse("PATH_TRANSFORMER_CLICK", "aos/sto/transformPipelineXDRClick")
  val transformerGapPath: String = sys.env.getOrElse("PATH_TRANSFORMER_GAP", "aos/sto/transformPipelineXDRGap")
  val transformedTrainPath: String = sys.env.getOrElse("PATH_TRANSFORMED_TRAIN", "aos/sto/transformedTrainDFXDR")
  val transformedTestPath: String = sys.env.getOrElse("PATH_TRANSFORMED_TEST", "aos/sto/transformedTestDFXDF")

  val propensityScorePath: String = sys.env.getOrElse("PATH_PROPENSITY_SCORE", "aos/sto/propensityScoreDF")

  val modelBasePath: String = sys.env.getOrElse("MODEL_BASE_PATH", "aos/sto/models/predict_ost")
  val modelClickPath: String = s"${modelBasePath}/click"
  val modelGapPath: String = s"${modelBasePath}/gap"
  val modelRegPath: String = s"${modelBasePath}/reg"

  // [ZEPPELIN_PARAGRAPH] 02. Date helpers (from notebook)
  def getPreviousMonths(startMonthStr: String, period: Int): Array[String] = {
    val formatter = DateTimeFormatter.ofPattern("yyyyMM")
    val startMonth = YearMonth.parse(startMonthStr, formatter)
    val resultMonths = ListBuffer[String]()
    var currentMonth = startMonth
    for (_ <- 0 until period) {
      resultMonths.prepend(currentMonth.format(formatter))
      currentMonth = currentMonth.minusMonths(1)
    }
    resultMonths.toArray
  }

  def getPreviousDays(startDayStr: String, period: Int): Array[String] = {
    val formatter = DateTimeFormatter.ofPattern("yyyyMMdd")
    val startDay = JLocalDate.parse(startDayStr, formatter)
    val resultDays = ListBuffer[String]()
    var currentDay = startDay
    for (_ <- 0 until period) {
      resultDays.prepend(currentDay.format(formatter))
      currentDay = currentDay.minusDays(1)
    }
    resultDays.toArray
  }

  def getDaysBetween(startDayStr: String, endDayStr: String): Array[String] = {
    val formatter = DateTimeFormatter.ofPattern("yyyyMMdd")
    val start = JLocalDate.parse(startDayStr, formatter)
    val end = JLocalDate.parse(endDayStr, formatter)
    val numOfDays = ChronoUnit.DAYS.between(start, end).toInt
    val resultDays = ListBuffer[String]()
    for (i <- 0 to numOfDays) resultDays.append(start.plusDays(i).format(formatter))
    resultDays.toArray
  }

  // Derived date lists
  val sendYmList: Array[String] = getPreviousMonths(sendMonth, periodM + 2)
  val featureYmList: Array[String] = getPreviousMonths(featureMonth, periodM + 2)
  val featureDTList: Array[String] = getDaysBetween(featureYmList(0) + "01", sendMonth + "01")

  // [ZEPPELIN_PARAGRAPH] 03. Misc helpers
  def buildSuffixLikeCond(colName: String, csv: String): String = {
    val parts = csv.split(",").map(_.trim).filter(_.nonEmpty)
    if (parts.isEmpty || parts.contains("%")) "1=1"
    else parts.map(s => s"$colName like '%$s'").mkString(" or ")
  }

  val smnCond: String = buildSuffixLikeCond("svc_mgmt_num", suffixCsv)

  def suffixListFromCsv(csv: String): List[String] = {
    val parts = csv.split(",").map(_.trim.toLowerCase).filter(_.nonEmpty).toList
    if (parts.isEmpty || parts.contains("%")) (0 to 15).map(_.toHexString).toList
    else parts
  }

  def showN(df: DataFrame, n: Int = 20): Unit = df.show(n, truncate = false)
}

