object PredictOst40TrainEvalAndSaveModels {
  // CLICK ONLY: P(C=1 | T=t, X=x)
  //
  // Entrypoints:
  //   PredictOst40TrainEvalAndSaveModels.runTrain()
  //   PredictOst40TrainEvalAndSaveModels.runEval()
  //   PredictOst40TrainEvalAndSaveModels.runAll()
  // Backward compat:
  //   PredictOst40TrainEvalAndSaveModels.run()

  import PredictOstConfig._
  import ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier
  import org.apache.spark.ml.{Pipeline, PipelineModel}
  import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
  import org.apache.spark.ml.feature.{IndexToString, StringIndexerModel}
  import org.apache.spark.ml.linalg.Vector
  import org.apache.spark.mllib.evaluation.MulticlassMetrics
  import org.apache.spark.sql.{DataFrame, functions => F}
  import org.apache.spark.sql.functions._

  case class DiagConfig(sampleFrac: Double, topK: Int, thresholds: List[Double])
  case class CalibrationConfig(
    trainPosRate: Option[Double] = None,
    truePosRate: Option[Double] = None,
    useWeightCorrection: Boolean = false
  )
  case class RunConfig(
    logPath: String,
    cacheTransformed: Boolean,
    useSampleBy: Boolean,
    diag: DiagConfig,
    calib: CalibrationConfig = CalibrationConfig()
  )

  // Allow overriding config inside spark-shell without restarting.
  // If not set, config is loaded from sys.env.
  @volatile private var overrideConfig: Option[RunConfig] = None

  def setConfig(cfg: RunConfig): Unit = { overrideConfig = Some(cfg) }
  def clearConfig(): Unit = { overrideConfig = None }

  // Convenience setter for spark-shell usage.
  def setConfig(
    logPath: String,
    cacheTransformed: Boolean = false,
    useSampleBy: Boolean = false,
    diagSampleFrac: Double = 0.02,
    diagTopK: Int = 50000,
    diagThresholdsCsv: String = "0.05,0.1,0.2,0.3,0.5,0.7",
    trainPosRate: Option[Double] = None,
    truePosRate: Option[Double] = None,
    useWeightCorrection: Boolean = false
  ): Unit = {
    val thresholds = diagThresholdsCsv.split(",").toList.map(_.trim).filter(_.nonEmpty).map(_.toDouble)
    setConfig(
      RunConfig(
        logPath = logPath,
        cacheTransformed = cacheTransformed,
        useSampleBy = useSampleBy,
        diag = DiagConfig(diagSampleFrac, diagTopK, thresholds),
        calib = CalibrationConfig(trainPosRate = trainPosRate, truePosRate = truePosRate, useWeightCorrection = useWeightCorrection)
      )
    )
  }

  final class FileLogger(val path: String) {
    private val writer = new java.io.PrintWriter(new java.io.BufferedWriter(new java.io.FileWriter(path, /*append=*/ true)))
    private def nowTs: String = java.time.ZonedDateTime.now().toString

    def close(): Unit = {
      writer.flush()
      writer.close()
    }

    def line(s: String): Unit = {
      writer.println(s"[$nowTs] $s")
      writer.flush()
    }

    def section(title: String): Unit = {
      line("")
      line("=" * 110)
      line(title)
      line("=" * 110)
      line("")
    }

    private def captureStdout(body: => Unit): String = {
      val baos = new java.io.ByteArrayOutputStream()
      val ps = new java.io.PrintStream(baos)
      try {
        Console.withOut(ps) { Console.withErr(ps) { body } }
      } finally {
        ps.flush()
        ps.close()
      }
      baos.toString("UTF-8")
    }

    def df(df: DataFrame, title: String, n: Int, truncate: Int = 0): Unit = {
      val truncateArg: Any = if (truncate <= 0) false else truncate
      val s = captureStdout {
        truncateArg match {
          case b: Boolean => df.show(n, b)
          case i: Int => df.show(n, i)
        }
      }
      section(title)
      s.split("\n", -1).foreach(line)
    }
  }

  def loadRunConfig(): RunConfig = {
    overrideConfig.getOrElse(loadRunConfigFromEnv())
  }

  def loadRunConfigFromEnv(): RunConfig = {
    val logPath = sys.env.getOrElse("LOG_PATH", s"/data/myfiles/aos_ost/predict/logs/predict_ost_click_train_${System.currentTimeMillis()}.log")
    val cacheTransformed = sys.env.getOrElse("CACHE_TRANSFORMED", "false").toBoolean
    val useSampleBy = sys.env.getOrElse("USE_SAMPLE_BY", "false").toBoolean
    val diagSampleFrac = sys.env.getOrElse("DIAG_SAMPLE_FRAC", "0.02").toDouble
    val diagTopK = sys.env.getOrElse("DIAG_TOPK", "50000").toInt
    val diagThresholds = sys.env
      .getOrElse("DIAG_THRESHOLDS", "0.05,0.1,0.2,0.3,0.5,0.7")
      .split(",")
      .toList
      .map(_.trim)
      .filter(_.nonEmpty)
      .map(_.toDouble)

    val trainPosRate = sys.env.get("TRAIN_POS_RATE").map(_.toDouble).filter(p => p > 0.0 && p < 1.0)
    val truePosRate = sys.env.get("TRUE_POS_RATE").map(_.toDouble).filter(p => p > 0.0 && p < 1.0)
    val useWeightCorrection = sys.env.getOrElse("USE_WEIGHT_CORRECTION", "false").toBoolean

    RunConfig(
      logPath = logPath,
      cacheTransformed = cacheTransformed,
      useSampleBy = useSampleBy,
      diag = DiagConfig(diagSampleFrac, diagTopK, diagThresholds),
      calib = CalibrationConfig(trainPosRate = trainPosRate, truePosRate = truePosRate, useWeightCorrection = useWeightCorrection)
    )
  }

  private def loadTransformed(cfg: RunConfig): (DataFrame, DataFrame) = {
    var train = spark.read.parquet(transformedTrainPath)
    var test = spark.read.parquet(transformedTestPath)
    if (cfg.cacheTransformed) {
      train = train.cache()
      test = test.cache()
      train.count()
      test.count()
    }
    (train, test)
  }

  private def buildClickEstimator(numWorkers: Int): XGBoostClassifier =
    new XGBoostClassifier("xgbc_click")
      .setLabelCol("indexedLabelClick")
      .setFeaturesCol("indexedFeaturesClick")
      .setMissing(0)
      .setSeed(0)
      .setMaxDepth(4)
      .setObjective("binary:logistic")
      .setNumRound(50)
      .setNumWorkers(numWorkers)
      .setEvalMetric("auc")
      .setProbabilityCol("prob_xgbc_click")
      .setPredictionCol("pred_xgbc_click")
      .setRawPredictionCol("pred_raw_xgbc_click")

  private def logCounts(logger: FileLogger, df: DataFrame, labelCol: String, name: String): Unit = {
    val agg = df
      .agg(
        count(lit(1)).as("n"),
        sum(col(labelCol)).as("pos"),
        (count(lit(1)) - sum(col(labelCol))).as("neg")
      )
      .collect()(0)
    val n = agg.getAs[Long]("n")
    val pos = Option(agg.get(1)).map(_.toString.toDouble).getOrElse(0.0)
    val neg = Option(agg.get(2)).map(_.toString.toDouble).getOrElse(0.0)
    val rate = if (n == 0) 0.0 else pos / n.toDouble
    val scalePosWeight = if (pos <= 0) Double.PositiveInfinity else (neg / pos)
    logger.line(f"[$name] n=$n%,d pos=$pos%.0f neg=$neg%.0f pos_rate=${rate * 100}%.4f%%  (neg/pos scalePosWeight≈$scalePosWeight%.4f)")
  }

  private def trainClickModel(train: DataFrame, cfg: RunConfig, logger: FileLogger): PipelineModel = {
    val numWorkers = sys.env.getOrElse("XGB_WORKERS", "10").toInt
    val xgbc = buildClickEstimator(numWorkers)

    val clickTrainBase = train.filter("cmpgn_typ=='Sales'")
    logger.section("TRAIN: click model dataset distribution (Sales)")
    logCounts(logger, clickTrainBase, "indexedLabelClick", "clickTrainBase (Sales, indexedLabelClick)")

    val clickTrainFinal =
      if (cfg.useSampleBy) {
        val sampled = clickTrainBase.stat.sampleBy(col("indexedLabelClick"), Map(0.0 -> 0.5, 1.0 -> 1.0), 42L)
        logCounts(logger, sampled, "indexedLabelClick", "clickTrainSampled (Sales, indexedLabelClick)")
        sampled
      } else {
        logger.line("[train] USE_SAMPLE_BY=false (no additional down-sampling)")
        clickTrainBase
      }

    // Optional importance weighting to undo rebalancing for better-calibrated probabilities.
    // If your training data was rebalanced (e.g. pos≈33%) but true prior is much smaller,
    // weighting negatives can help the optimizer learn a more realistic decision boundary.
    //
    // Enable via env:
    //   USE_WEIGHT_CORRECTION=true  TRUE_POS_RATE=0.0075
    val useWeightCorrection = cfg.calib.useWeightCorrection
    val piTrueOpt = cfg.calib.truePosRate

    val trainForFit =
      if (useWeightCorrection && piTrueOpt.nonEmpty) {
        val piTrue = piTrueOpt.get
        val piTrain = clickTrainFinal.agg(avg(col("indexedLabelClick").cast("double")).as("pi_train")).collect()(0).getAs[Double]("pi_train")
        // weight for negatives so that effective neg/pos matches true neg/pos
        // kNeg = ((1-piTrue)/piTrue) / ((1-piTrain)/piTrain)
        val kNeg = ((1.0 - piTrue) / piTrue) / ((1.0 - piTrain) / piTrain)
        logger.line(f"[train] USE_WEIGHT_CORRECTION=true pi_train=$piTrain%.6f pi_true=$piTrue%.6f kNeg=$kNeg%.4f (neg weight, pos weight=1.0)")

        val weighted = clickTrainFinal
          .withColumn(
            "sample_weight",
            when(col("indexedLabelClick") === 1.0, lit(1.0)).otherwise(lit(kNeg))
          )

        xgbc.setWeightCol("sample_weight")
        weighted
      } else {
        if (useWeightCorrection && piTrueOpt.isEmpty) {
          logger.line("[train] USE_WEIGHT_CORRECTION=true but TRUE_POS_RATE is not set; skipping weight correction")
        }
        clickTrainFinal
      }

    new Pipeline().setStages(Array(xgbc)).fit(trainForFit)
  }

  private def evalClickModel(model: PipelineModel, transformerClick: PipelineModel, test: DataFrame, cfg: RunConfig, logger: FileLogger): Unit = {
    val vectorToArrayUdf = udf((v: Vector) => v.toArray)
    def withProb(df: DataFrame, probCol: String): DataFrame = df.withColumn("_prob1", vectorToArrayUdf(col(probCol)).getItem(1))

    // Optional prior correction:
    // If your model was trained on rebalanced data (e.g., 1:2 => pi_train≈0.3333) but TS is real prior (e.g., 0.0075),
    // you can adjust probability via odds correction.
    //
    // Set via env (optional):
    //   TRAIN_POS_RATE=0.3333   TRUE_POS_RATE=0.0075
    val piTrainOpt = cfg.calib.trainPosRate
    val piTrueOpt = cfg.calib.truePosRate

    def priorCorrectedProb(p: org.apache.spark.sql.Column, piTrain: Double, piTrue: Double): org.apache.spark.sql.Column = {
      // odds_true = odds_model * (piTrue*(1-piTrain)) / (piTrain*(1-piTrue))
      val eps = 1e-12
      val pClamped = greatest(lit(eps), least(lit(1.0 - eps), p))
      val odds = pClamped / (lit(1.0) - pClamped)
      val k = lit((piTrue * (1.0 - piTrain)) / (piTrain * (1.0 - piTrue)))
      val oddsAdj = odds * k
      oddsAdj / (lit(1.0) + oddsAdj)
    }

    def logProbSummary(dfWithProb: DataFrame, name: String): Unit = {
      val qs = dfWithProb.stat.approxQuantile("_prob1", Array(0.0, 0.01, 0.05, 0.1, 0.5, 0.9, 0.99, 1.0), 0.01)
      logger.line(s"[$name] prob quantiles: p00=${qs(0)} p01=${qs(1)} p05=${qs(2)} p10=${qs(3)} p50=${qs(4)} p90=${qs(5)} p99=${qs(6)} p100=${qs(7)}")
    }

    def logCalibration(dfWithProb: DataFrame, labelCol: String, name: String, bins: Int = 20): Unit = {
      val b = bins.toDouble
      val cal = dfWithProb
        .sample(cfg.diag.sampleFrac)
        .withColumn("_bin", floor(col("_prob1") * b) / b)
        .groupBy(col("_bin"))
        .agg(count(lit(1)).as("n"), avg(col("_prob1")).as("avg_prob"), avg(col(labelCol)).as("emp_rate"))
        .orderBy(col("_bin"))
      logger.df(cal, s"[$name] calibration table (bins=$bins sampleFrac=${cfg.diag.sampleFrac})", 200, truncate = 0)
    }

    def logConfusion(dfWithProb: DataFrame, labelCol: String, name: String): Unit = {
      cfg.diag.thresholds.foreach { thr =>
        val m = dfWithProb
          .withColumn("_pred", when(col("_prob1") >= lit(thr), lit(1.0)).otherwise(lit(0.0)))
          .agg(
            sum(when(col("_pred") === 1.0 && col(labelCol) === 1.0, 1).otherwise(0)).as("tp"),
            sum(when(col("_pred") === 1.0 && col(labelCol) === 0.0, 1).otherwise(0)).as("fp"),
            sum(when(col("_pred") === 0.0 && col(labelCol) === 0.0, 1).otherwise(0)).as("tn"),
            sum(when(col("_pred") === 0.0 && col(labelCol) === 1.0, 1).otherwise(0)).as("fn")
          )
          .collect()(0)
        val tp = m.getAs[Long]("tp").toDouble
        val fp = m.getAs[Long]("fp").toDouble
        val tn = m.getAs[Long]("tn").toDouble
        val fn = m.getAs[Long]("fn").toDouble
        val prec = if (tp + fp == 0) 0.0 else tp / (tp + fp)
        val rec = if (tp + fn == 0) 0.0 else tp / (tp + fn)
        val f1 = if (prec + rec == 0) 0.0 else 2 * prec * rec / (prec + rec)
        val fpr = if (fp + tn == 0) 0.0 else fp / (fp + tn)
        logger.line(f"[$name] thr=$thr%.2f  TP=$tp%.0f FP=$fp%.0f TN=$tn%.0f FN=$fn%.0f  precision=$prec%.4f recall=$rec%.4f f1=$f1%.4f fpr=$fpr%.4f")
      }
    }

    def logTopK(dfWithProb: DataFrame, labelCol: String, name: String): Unit = {
      def toDoubleAny(v: Any): Double = v match {
        case null => 0.0
        case d: java.lang.Double => d.doubleValue()
        case f: java.lang.Float => f.doubleValue()
        case l: java.lang.Long => l.doubleValue()
        case i: java.lang.Integer => i.doubleValue()
        case s: java.lang.Short => s.doubleValue()
        case bd: java.math.BigDecimal => bd.doubleValue()
        case bd: scala.math.BigDecimal => bd.doubleValue
        case n: Number => n.doubleValue()
        case other => other.toString.toDouble
      }

      val baseRow = dfWithProb.agg(avg(col(labelCol)).as("base_rate")).collect()(0)
      val base = toDoubleAny(baseRow.getAs[Any]("base_rate"))

      val topRow = dfWithProb
        .orderBy(desc("_prob1"))
        .limit(cfg.diag.topK)
        .agg(avg(col(labelCol)).as("top_rate"))
        .collect()(0)
      val top = toDoubleAny(topRow.getAs[Any]("top_rate"))

      val lift = if (base == 0.0) Double.PositiveInfinity else top / base
      logger.line(f"[$name] TopK=${cfg.diag.topK} base_rate=${base * 100}%.4f%% top_rate=${top * 100}%.4f%% lift=${lift}%.2f")
    }

    val labelIndexersMap: Map[String, StringIndexerModel] =
      transformerClick.stages.collect { case sim: StringIndexerModel => sim.uid -> sim }.toMap
    val labelConverterClick = new IndexToString()
      .setInputCol("prediction_click")
      .setOutputCol("predictedLabelClick")
      .setLabels(labelIndexersMap("indexer_click").labelsArray(0))

    logger.section("EVAL: transformedTestDF distribution & by-hour click rate")
    logCounts(logger, test, "click_yn", "transformedTestDF (raw click_yn)")
    val byHour = test
      .groupBy(col("send_hournum_cd"))
      .agg(count(lit(1)).as("n"), sum(col("click_yn")).as("pos"))
      .withColumn("pos_rate", col("pos") / col("n"))
      .orderBy(desc("n"))
    logger.df(byHour, "[transformedTestDF] click rate by send_hournum_cd", 50, truncate = 0)

    val predictionsClickDev = model.transform(
      test
        .filter("cmpgn_typ=='Sales'")
        .dropDuplicates("svc_mgmt_num", "chnl_typ", "cmpgn_typ", "send_ym", "send_hournum_cd", "click_yn")
    )

    logger.section("EVAL: dev/test (row-level) distribution")
    logCounts(logger, predictionsClickDev, "click_yn", "predictionsClickDev (Sales, row-level click_yn)")

    val clickProbCol = s"prob_${model.stages.head.uid}"
    val clickDevWithProb = withProb(predictionsClickDev, clickProbCol).cache()

    logger.section(s"EVAL: probability diagnostics ($clickProbCol)")
    logProbSummary(clickDevWithProb, s"predictionsClickDev (row-level) $clickProbCol")
    logCalibration(clickDevWithProb, "click_yn", s"predictionsClickDev (row-level) $clickProbCol")
    logConfusion(clickDevWithProb, "click_yn", s"predictionsClickDev (row-level) $clickProbCol")
    logTopK(clickDevWithProb, "click_yn", s"predictionsClickDev (row-level) $clickProbCol")

    // Prior-corrected probability diagnostics (optional)
    (piTrainOpt, piTrueOpt) match {
      case (Some(piTrain), Some(piTrue)) =>
        val clickDevAdj = clickDevWithProb.withColumn("_prob1", priorCorrectedProb(col("_prob1"), piTrain, piTrue)).cache()
        logger.section(s"EVAL: prior-corrected probability diagnostics (TRAIN_POS_RATE=$piTrain TRUE_POS_RATE=$piTrue)")
        logProbSummary(clickDevAdj, s"predictionsClickDev (row-level) prior-corrected")
        logCalibration(clickDevAdj, "click_yn", s"predictionsClickDev (row-level) prior-corrected")
        logConfusion(clickDevAdj, "click_yn", s"predictionsClickDev (row-level) prior-corrected")
        logTopK(clickDevAdj, "click_yn", s"predictionsClickDev (row-level) prior-corrected")
      case _ =>
        logger.section("EVAL: prior-corrected probability diagnostics (skipped)")
        logger.line("To enable: set env TRAIN_POS_RATE and TRUE_POS_RATE (both in (0,1)). Example: TRAIN_POS_RATE=0.3333 TRUE_POS_RATE=0.0075")
    }

    logger.section("EVAL: aggregated user-hour view (matches original evaluation style)")
    val clickAgg = clickDevWithProb
      .groupBy("svc_mgmt_num", "send_ym", "send_hournum_cd")
      .agg(sum(col("indexedLabelClick")).as("indexedLabelClick"), max(col("_prob1")).as("_prob1"))
      .withColumn("indexedLabelClick", expr("case when indexedLabelClick>0 then 1.0 else 0.0 end"))
      .cache()
    logCounts(logger, clickAgg, "indexedLabelClick", "clickAgg (user-hour, binarized label)")
    logProbSummary(clickAgg, "clickAgg (user-hour)")
    logConfusion(clickAgg, "indexedLabelClick", "clickAgg (user-hour)")
    logTopK(clickAgg, "indexedLabelClick", "clickAgg (user-hour)")

    logger.section("EVAL: AUC metrics (row-level)")
    val rocAuc = new BinaryClassificationEvaluator().setLabelCol("click_yn").setRawPredictionCol(clickProbCol).setMetricName("areaUnderROC").evaluate(predictionsClickDev)
    val prAuc = new BinaryClassificationEvaluator().setLabelCol("click_yn").setRawPredictionCol(clickProbCol).setMetricName("areaUnderPR").evaluate(predictionsClickDev)
    logger.line(f"[predictionsClickDev] ROC-AUC=$rocAuc%.6f  PR-AUC=$prAuc%.6f  (TS is highly imbalanced; PR-AUC is usually more informative)")

    logger.section("EVAL: original-style MulticlassMetrics (aggregated user-hour, threshold=0.5)")
    val thresholdProb = 0.5
    model.stages.foreach { stage =>
      val modelName = stage.uid
      val predictionAndLabels = labelConverterClick
        .transform(
          predictionsClickDev
            .withColumn("prob", vectorToArrayUdf(col(s"prob_$modelName")).getItem(1))
            .groupBy("svc_mgmt_num", "send_ym", "send_hournum_cd")
            .agg(sum(col("indexedLabelClick")).alias("indexedLabelClick"), max(col("prob")).alias("prob"))
            .withColumn("indexedLabelClick", expr("case when indexedLabelClick>0 then 1.0 else 0.0 end"))
            .withColumn("prediction_click", expr(s"case when prob>=$thresholdProb then 1.0 else 0.0 end"))
        )
        .selectExpr("prediction_click", "cast(indexedLabelClick as double)")
        .rdd
        .map(row => (row.getDouble(0), row.getDouble(1)))

      val metrics = new MulticlassMetrics(predictionAndLabels)
      logger.line(s"######### $modelName click 예측 결과 #########")
      metrics.labels.foreach { label =>
        logger.line(f"Label $label: Precision=${metrics.precision(label)}%.4f Recall=${metrics.recall(label)}%.4f F1=${metrics.fMeasure(label)}%.4f")
      }
      logger.line(s"Weighted Precision: ${metrics.weightedPrecision}")
      logger.line(s"Weighted Recall: ${metrics.weightedRecall}")
      logger.line(s"Accuracy: ${metrics.accuracy}")
      logger.line("Confusion Matrix:")
      logger.line(metrics.confusionMatrix.toString())
    }
  }

  // [ZEPPELIN_PARAGRAPH] 49. Entrypoints
  def runTrain(): Unit = {
    initSpark()
    val cfg = loadRunConfig()
    val logger = new FileLogger(cfg.logPath)
    try {
      logger.section("RUN: train (click-only)")
      logger.line(s"[config] cacheTransformed=${cfg.cacheTransformed} useSampleBy=${cfg.useSampleBy} transformedTrainPath=$transformedTrainPath modelClickPath=$modelClickPath")

      // Ensure transformer exists in classpath/env
      val transformerClick = PipelineModel.load(transformerClickPath)
      val (train, _) = loadTransformed(cfg)

      val model = trainClickModel(train, cfg, logger)
      model.write.overwrite().save(modelClickPath)
      logger.line(s"Saved click model to: $modelClickPath")
      logger.line(s"Log saved to: ${cfg.logPath}")
      if (transformerClick.uid.nonEmpty) ()
    } finally {
      logger.close()
    }
  }

  def runEval(): Unit = {
    initSpark()
    val cfg = loadRunConfig()
    val logger = new FileLogger(cfg.logPath)
    try {
      logger.section("RUN: eval (click-only)")
      logger.line(s"[config] cacheTransformed=${cfg.cacheTransformed} transformedTestPath=$transformedTestPath modelClickPath=$modelClickPath")

      val transformerClick = PipelineModel.load(transformerClickPath)
      val model = PipelineModel.load(modelClickPath)
      val (_, test) = loadTransformed(cfg)

      evalClickModel(model, transformerClick, test, cfg, logger)
      logger.line(s"Log saved to: ${cfg.logPath}")
    } finally {
      logger.close()
    }
  }

  def runAll(): Unit = {
    initSpark()
    val cfg = loadRunConfig()
    val logger = new FileLogger(cfg.logPath)
    try {
      logger.section("RUN: train + eval (click-only)")
      logger.line(s"[config] cacheTransformed=${cfg.cacheTransformed} useSampleBy=${cfg.useSampleBy} LOG_PATH=${cfg.logPath}")

      val transformerClick = PipelineModel.load(transformerClickPath)
      val (train, test) = loadTransformed(cfg)

      val model = trainClickModel(train, cfg, logger)
      model.write.overwrite().save(modelClickPath)
      logger.line(s"Saved click model to: $modelClickPath")

      evalClickModel(model, transformerClick, test, cfg, logger)
      logger.line(s"Log saved to: ${cfg.logPath}")
    } finally {
      logger.close()
    }
  }

  def run(): Unit = runAll()
}
