object PredictOst60ScoreAndSave {
  import PredictOstConfig._

  case class ScoreConfig(
    suffixScoreCsv: String = "%",                 // e.g. "%", "0,1,2"
    trainPosRate: Option[Double] = None,          // TRAIN_POS_RATE
    truePosRate: Option[Double] = None            // TRUE_POS_RATE
  )

  @volatile private var overrideConfig: Option[ScoreConfig] = None
  def setConfig(cfg: ScoreConfig): Unit = { overrideConfig = Some(cfg) }
  def clearConfig(): Unit = { overrideConfig = None }

  // Convenience for spark-shell
  def setConfig(
    suffixScoreCsv: String,
    trainPosRate: Option[Double] = None,
    truePosRate: Option[Double] = None
  ): Unit = setConfig(ScoreConfig(suffixScoreCsv = suffixScoreCsv, trainPosRate = trainPosRate, truePosRate = truePosRate))

  private def loadConfig(): ScoreConfig = overrideConfig.getOrElse(loadConfigFromEnv())

  private def loadConfigFromEnv(): ScoreConfig = {
    val suffixScoreCsv = sys.env.getOrElse("SUFFIX_SCORE", sys.env.getOrElse("SUFFIX_PRED", "%"))
    val piTrainOpt = sys.env.get("TRAIN_POS_RATE").map(_.toDouble).filter(p => p > 0.0 && p < 1.0)
    val piTrueOpt = sys.env.get("TRUE_POS_RATE").map(_.toDouble).filter(p => p > 0.0 && p < 1.0)
    ScoreConfig(suffixScoreCsv = suffixScoreCsv, trainPosRate = piTrainOpt, truePosRate = piTrueOpt)
  }

  def run(): Unit = {
    initSpark()

    // [ZEPPELIN_PARAGRAPH] 60. Score predDFRev with saved transformer/model and write CLICK probabilities
    // Focus: P(C=1 | T=t, X=x)
    import org.apache.spark.ml.PipelineModel
    import org.apache.spark.ml.linalg.Vector
    import org.apache.spark.sql.functions._

    // [ZEPPELIN_PARAGRAPH] 61. Load transformers + models
    val transformerClick = PipelineModel.load(transformerClickPath)
    val pipelineModelClick = PipelineModel.load(modelClickPath)

    val vectorToArrayUdf = udf((v: Vector) => v.toArray)

    // Optional prior correction (same as eval script):
    //   TRAIN_POS_RATE=0.3333 TRUE_POS_RATE=0.0075
    // If set, we save `prob_click` as corrected probability and also keep `prob_click_raw`.
    val cfg = loadConfig()
    val piTrainOpt = cfg.trainPosRate
    val piTrueOpt = cfg.truePosRate

    def priorCorrectedProb(p: org.apache.spark.sql.Column, piTrain: Double, piTrue: Double): org.apache.spark.sql.Column = {
      val eps = 1e-12
      val pClamped = greatest(lit(eps), least(lit(1.0 - eps), p))
      val odds = pClamped / (lit(1.0) - pClamped)
      val k = lit((piTrue * (1.0 - piTrain)) / (piTrain * (1.0 - piTrue)))
      val oddsAdj = odds * k
      oddsAdj / (lit(1.0) + oddsAdj)
    }

    // [ZEPPELIN_PARAGRAPH] 62. Load predDFRev (from previous step)
    val predDFRev = spark.table("pred_df_rev")

    // [ZEPPELIN_PARAGRAPH] 63. Score per suffix (0-f) and save
    val suffixScoreCsv = cfg.suffixScoreCsv
    val suffixesToScore = suffixListFromCsv(suffixScoreCsv)

    suffixesToScore.foreach { suffix =>
      println(s"Scoring suffix=$suffix")

      val predPart = predDFRev.filter(s"svc_mgmt_num like '%$suffix'")
      val transformedPredDF = transformerClick.transform(predPart)
      val predictions = pipelineModelClick.transform(
        transformedPredDF.dropDuplicates("svc_mgmt_num", "chnl_typ", "cmpgn_typ", "send_ym", "send_hournum_cd", "click_yn")
      )

      val probCol = s"prob_${pipelineModelClick.stages.head.uid}"
      val scoredClickRawDF = predictions.withColumn("prob_click_raw", vectorToArrayUdf(col(probCol)).getItem(1))

      val scoredClickDF = (piTrainOpt, piTrueOpt) match {
        case (Some(piTrain), Some(piTrue)) =>
          scoredClickRawDF.withColumn("prob_click", priorCorrectedProb(col("prob_click_raw"), piTrain, piTrue))
        case _ =>
          scoredClickRawDF.withColumn("prob_click", col("prob_click_raw"))
      }

      scoredClickDF
        .selectExpr(
          "svc_mgmt_num",
          "send_ym",
          "send_hournum_cd send_hour",
          "ROUND(prob_click, 6) prob_click",
          "ROUND(prob_click_raw, 6) prob_click_raw"
        )
        .withColumn("suffix", expr("right(svc_mgmt_num, 1)"))
        .repartition(10)
        .write
        .mode("overwrite")
        .partitionBy("send_ym", "send_hour", "suffix")
        .parquet(propensityScorePath)
    }

    // [ZEPPELIN_PARAGRAPH] 64. Quick load check
    val scored = spark.read.parquet(propensityScorePath).cache()
    println(s"propensityScorePath count=${scored.count()}")
  }
}
