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
  import org.apache.spark.ml.classification.{GBTClassifier, ProbabilisticClassifier}
  import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
  import org.apache.spark.ml.feature.{IndexToString, StringIndexerModel}
  import org.apache.spark.ml.linalg.Vector
  import org.apache.spark.mllib.evaluation.MulticlassMetrics
  import org.apache.spark.sql.{DataFrame, functions => F}
  import org.apache.spark.sql.functions._

  // Model type selection
  object ModelType extends Enumeration {
    type ModelType = Value
    val XGBoost, SparkGBT = Value
    
    def fromString(s: String): ModelType = s.toLowerCase match {
      case "xgboost" | "xgb" => XGBoost
      case "sparkgbt" | "gbt" | "spark_gbt" => SparkGBT
      case _ => throw new IllegalArgumentException(s"Unknown model type: $s. Use 'xgboost' or 'sparkgbt'")
    }
  }
  import ModelType._

  case class DiagConfig(sampleFrac: Double, topK: Int, thresholds: List[Double])
  case class CalibrationConfig(
    trainPosRate: Option[Double] = None,  // Training data positive rate (used for both weight correction and prior correction)
    truePosRate: Option[Double] = None,   // True population positive rate
    useWeightCorrection: Boolean = false
  )
  case class RunConfig(
    logPath: String,
    cacheTransformed: Boolean,
    useSampleBy: Boolean,
    diag: DiagConfig,
    calib: CalibrationConfig = CalibrationConfig(),
    modelType: ModelType = ModelType.XGBoost,
    enableTrainingDiagnostics: Boolean = false
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
    useWeightCorrection: Boolean = false,
    modelType: String = "xgboost",
    enableTrainingDiagnostics: Boolean = false
  ): Unit = {
    val thresholds = diagThresholdsCsv.split(",").toList.map(_.trim).filter(_.nonEmpty).map(_.toDouble)
    setConfig(
      RunConfig(
        logPath = logPath,
        cacheTransformed = cacheTransformed,
        useSampleBy = useSampleBy,
        diag = DiagConfig(diagSampleFrac, diagTopK, thresholds),
        calib = CalibrationConfig(trainPosRate = trainPosRate, truePosRate = truePosRate, useWeightCorrection = useWeightCorrection),
        modelType = ModelType.fromString(modelType),
        enableTrainingDiagnostics = enableTrainingDiagnostics
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
    val modelTypeStr = sys.env.getOrElse("MODEL_TYPE", "xgboost")
    val enableTrainingDiagnostics = sys.env.getOrElse("ENABLE_TRAINING_DIAGNOSTICS", "false").toBoolean

    RunConfig(
      logPath = logPath,
      cacheTransformed = cacheTransformed,
      useSampleBy = useSampleBy,
      diag = DiagConfig(diagSampleFrac, diagTopK, diagThresholds),
      calib = CalibrationConfig(trainPosRate = trainPosRate, truePosRate = truePosRate, useWeightCorrection = useWeightCorrection),
      modelType = ModelType.fromString(modelTypeStr),
      enableTrainingDiagnostics = enableTrainingDiagnostics
    )
  }

  private def loadTransformed(cfg: RunConfig, logger: FileLogger): (DataFrame, DataFrame) = {
    logger.line("[FLOW] loadTransformed: START")
    logger.line(s"[CONFIG] Train path: $transformedTrainPath")
    logger.line(s"[CONFIG] Test path: $transformedTestPath")
    
    var train = spark.read.parquet(transformedTrainPath)
    logger.line(s"[DATA] Train dataset loaded (partitions: ${train.rdd.getNumPartitions})")
    
    var test = spark.read.parquet(transformedTestPath)
    logger.line(s"[DATA] Test dataset loaded (partitions: ${test.rdd.getNumPartitions})")
    
    if (cfg.cacheTransformed) {
      logger.line("[FLOW] Caching enabled - caching train dataset...")
      train = train.cache()
      val trainCount = train.count()
      logger.line(s"[DATA] Train dataset cached (count: ${trainCount.formatted("%,d")})")
      
      logger.line("[FLOW] Caching test dataset...")
      test = test.cache()
      val testCount = test.count()
      logger.line(s"[DATA] Test dataset cached (count: ${testCount.formatted("%,d")})")
    } else {
      logger.line("[CONFIG] Caching disabled")
    }
    
    logger.line("[FLOW] loadTransformed: DONE")
    (train, test)
  }

  private def buildXGBoostEstimator(numWorkers: Int): XGBoostClassifier =
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
  
  private def buildSparkGBTEstimator(): GBTClassifier =
    new GBTClassifier()
      .setLabelCol("indexedLabelClick")
      .setFeaturesCol("indexedFeaturesClick")
      .setSeed(0)
      .setMaxDepth(4)
      .setMaxIter(50)  // equivalent to numRound
      .setStepSize(0.1)  // learning rate
      .setSubsamplingRate(0.8)  // subsample
      .setFeatureSubsetStrategy("sqrt")  // colsample_bytree equivalent
      .setProbabilityCol("prob_gbt_click")
      .setPredictionCol("pred_gbt_click")
      .setRawPredictionCol("pred_raw_gbt_click")

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
    logger.line(f"[STAT] $name: n=$n%,d pos=$pos%.0f neg=$neg%.0f pos_rate=${rate * 100}%.4f%%  (neg/pos scalePosWeight≈$scalePosWeight%.4f)")
  }

  private def trainClickModel(train: DataFrame, cfg: RunConfig, logger: FileLogger): PipelineModel = {
    logger.line("[FLOW] trainClickModel: START")
    logger.line(s"[CONFIG] Model type: ${cfg.modelType}")
    
    val numWorkers = sys.env.getOrElse("XGB_WORKERS", "10").toInt
    logger.line(s"[CONFIG] XGB workers: $numWorkers")

    logger.line("[DATA] Filtering data: cmpgn_typ=='Sales'")
    val clickTrainBase = train.filter("cmpgn_typ=='Sales'")
    
    if (cfg.enableTrainingDiagnostics) {
      logger.section(s"TRAIN: click model dataset distribution (Sales) - Model: ${cfg.modelType}")
      logCounts(logger, clickTrainBase, "indexedLabelClick", "clickTrainBase (Sales, indexedLabelClick)")
    } else {
      logger.line("[CONFIG] Training diagnostics disabled (ENABLE_TRAINING_DIAGNOSTICS=false)")
    }

    val clickTrainFinal =
      if (cfg.useSampleBy) {
        logger.line("[FLOW] Applying down-sampling (sampleBy)")
        val sampled = clickTrainBase.stat.sampleBy(col("indexedLabelClick"), Map(0.0 -> 0.5, 1.0 -> 1.0), 42L)
        logger.line("[DATA] Down-sampling completed")
        if (cfg.enableTrainingDiagnostics) {
          logCounts(logger, sampled, "indexedLabelClick", "clickTrainSampled (Sales, indexedLabelClick)")
        }
        sampled
      } else {
        logger.line("[CONFIG] USE_SAMPLE_BY=false (no additional down-sampling)")
        clickTrainBase
      }

    // Optional importance weighting to undo rebalancing for better-calibrated probabilities.
    // If your training data was rebalanced (e.g. pos≈33%) but true prior is much smaller,
    // weighting negatives can help the optimizer learn a more realistic decision boundary.
    //
    // Enable via env:
    //   USE_WEIGHT_CORRECTION=true  TRUE_POS_RATE=0.0075
    //   TRAIN_POS_RATE=0.333333  (optional: if known, skips driver computation)
    val useWeightCorrection = cfg.calib.useWeightCorrection
    val piTrueOpt = cfg.calib.truePosRate

    val trainForFit =
      if (useWeightCorrection && piTrueOpt.nonEmpty) {
        logger.line("[FLOW] Applying weight correction")
        val piTrue = piTrueOpt.get
        
        // Use provided trainPosRate if available, otherwise compute it
        val piTrain = cfg.calib.trainPosRate match {
          case Some(rate) =>
            logger.line(f"[CONFIG] Using provided TRAIN_POS_RATE=$rate%.6f (skipping driver computation)")
            rate
          case None =>
            logger.line("[DATA] Computing train positive rate (driver action)...")
            val computed = clickTrainFinal.agg(avg(col("indexedLabelClick").cast("double")).as("pi_train")).collect()(0).getAs[Double]("pi_train")
            logger.line(f"[DATA] Computed train positive rate: $computed%.6f")
            computed
        }
        
        // weight for negatives so that effective neg/pos matches true neg/pos
        // kNeg = ((1-piTrue)/piTrue) / ((1-piTrain)/piTrain)
        val kNeg = ((1.0 - piTrue) / piTrue) / ((1.0 - piTrain) / piTrain)
        logger.line(f"[CONFIG] USE_WEIGHT_CORRECTION=true pi_train=$piTrain%.6f pi_true=$piTrue%.6f kNeg=$kNeg%.4f (neg weight, pos weight=1.0)")

        logger.line("[DATA] Adding sample_weight column...")
        clickTrainFinal.withColumn(
          "sample_weight",
          when(col("indexedLabelClick") === 1.0, lit(1.0)).otherwise(lit(kNeg))
        )
      } else {
        if (useWeightCorrection && piTrueOpt.isEmpty) {
          logger.line("[CONFIG] USE_WEIGHT_CORRECTION=true but TRUE_POS_RATE is not set; skipping weight correction")
        } else {
          logger.line("[CONFIG] Weight correction disabled")
        }
        clickTrainFinal
      }

    // Build and train model based on type
    logger.line(s"[MODEL] Building ${cfg.modelType} estimator...")
    val model = cfg.modelType match {
      case ModelType.XGBoost =>
        logger.line("[MODEL] Creating XGBoost classifier")
        val estimator = buildXGBoostEstimator(numWorkers)
        if (useWeightCorrection && piTrueOpt.nonEmpty) {
          logger.line("[CONFIG] Setting weight column for XGBoost")
          estimator.setWeightCol("sample_weight")
        }
        logger.line("[MODEL] Starting XGBoost training (this may take a while)...")
        val trainedModel = new Pipeline().setStages(Array(estimator)).fit(trainForFit)
        logger.line("[MODEL] XGBoost training completed!")
        trainedModel
      
      case ModelType.SparkGBT =>
        logger.line("[MODEL] Creating Spark GBT classifier")
        val estimator = buildSparkGBTEstimator()
        if (useWeightCorrection && piTrueOpt.nonEmpty) {
          logger.line("[CONFIG] Setting weight column for Spark GBT")
          estimator.setWeightCol("sample_weight")
        }
        logger.line("[MODEL] Starting Spark GBT training (this may take a while)...")
        val trainedModel = new Pipeline().setStages(Array(estimator)).fit(trainForFit)
        logger.line("[MODEL] Spark GBT training completed!")
        trainedModel
    }
    
    logger.line("[FLOW] trainClickModel: DONE")
    model
  }

  private def evalClickModel(model: PipelineModel, transformerClick: PipelineModel, test: DataFrame, cfg: RunConfig, logger: FileLogger): Unit = {
    logger.line("[FLOW] evalClickModel: START")
    
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
      // 0.01 → 0.05: 정확도 완화로 안정성 향상 (노드 부하 감소)
      val qs = dfWithProb.stat.approxQuantile("_prob1", Array(0.0, 0.01, 0.05, 0.1, 0.5, 0.9, 0.99, 1.0), 0.05)
      logger.line(s"[STAT] $name prob quantiles: p00=${qs(0)} p01=${qs(1)} p05=${qs(2)} p10=${qs(3)} p50=${qs(4)} p90=${qs(5)} p99=${qs(6)} p100=${qs(7)}")
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
        logger.line(f"[PERF] $name thr=$thr%.2f  TP=$tp%.0f FP=$fp%.0f TN=$tn%.0f FN=$fn%.0f  precision=$prec%.4f recall=$rec%.4f f1=$f1%.4f fpr=$fpr%.4f")
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
      logger.line(f"[PERF] $name TopK=${cfg.diag.topK} base_rate=${base * 100}%.4f%% top_rate=${top * 100}%.4f%% lift=${lift}%.2f")
    }

    val labelIndexersMap: Map[String, StringIndexerModel] =
      transformerClick.stages.collect { case sim: StringIndexerModel => sim.uid -> sim }.toMap
    val labelConverterClick = new IndexToString()
      .setInputCol("prediction_click")
      .setOutputCol("predictedLabelClick")
      .setLabels(labelIndexersMap("indexer_click").labelsArray(0))

    logger.section("EVAL: transformedTestDF distribution & by-hour click rate")
    logger.line("[DATA] Computing test dataset statistics...")
    logCounts(logger, test, "click_yn", "transformedTestDF (raw click_yn)")
    
    logger.line("[STAT] Computing by-hour click rates...")
    val byHour = test
      .groupBy(col("send_hournum_cd"))
      .agg(count(lit(1)).as("n"), sum(col("click_yn")).as("pos"))
      .withColumn("pos_rate", col("pos") / col("n"))
      .orderBy(desc("n"))
    logger.df(byHour, "[transformedTestDF] click rate by send_hournum_cd", 50, truncate = 0)

    logger.line("[DATA] Preparing test data (filtering Sales, deduplicating)...")
    val testPrep = test
      .filter("cmpgn_typ=='Sales'")
      .dropDuplicates("svc_mgmt_num", "chnl_typ", "cmpgn_typ", "send_ym", "send_hournum_cd", "click_yn")
    logger.line("[DATA] Test data prepared")
    
    logger.line("[MODEL] Applying model transformation (prediction)...")
    val predictionsClickDev = model.transform(testPrep)
    logger.line("[MODEL] Predictions completed")

    logger.section("EVAL: dev/test (row-level) distribution")
    logger.line("[DATA] Computing prediction statistics...")
    logCounts(logger, predictionsClickDev, "click_yn", "predictionsClickDev (Sales, row-level click_yn)")

    val clickProbCol = s"prob_${model.stages.head.uid}"
    logger.line(s"[DATA] Extracting probabilities from column: $clickProbCol")
    val clickDevWithProb = withProb(predictionsClickDev, clickProbCol).cache()
    logger.line("[DATA] Probabilities extracted and cached")

    logger.section(s"EVAL: probability diagnostics ($clickProbCol)")
    logger.line("[FLOW] Computing probability quantiles...")
    logProbSummary(clickDevWithProb, s"predictionsClickDev (row-level) $clickProbCol")
    
    logger.line("[FLOW] Computing calibration table...")
    logCalibration(clickDevWithProb, "click_yn", s"predictionsClickDev (row-level) $clickProbCol")
    
    logger.line("[FLOW] Computing confusion matrices at multiple thresholds...")
    logConfusion(clickDevWithProb, "click_yn", s"predictionsClickDev (row-level) $clickProbCol")
    
    logger.line("[FLOW] Computing top-K lift...")
    logTopK(clickDevWithProb, "click_yn", s"predictionsClickDev (row-level) $clickProbCol")

    // Prior-corrected probability diagnostics (optional)
    logger.line("[FLOW] Checking for prior correction settings...")
    (piTrainOpt, piTrueOpt) match {
      case (Some(piTrain), Some(piTrue)) =>
        logger.line(s"[FLOW] Applying prior correction (piTrain=$piTrain, piTrue=$piTrue)")
        val clickDevAdj = clickDevWithProb.withColumn("_prob1", priorCorrectedProb(col("_prob1"), piTrain, piTrue)).cache()
        logger.section(s"EVAL: prior-corrected probability diagnostics (TRAIN_POS_RATE=$piTrain TRUE_POS_RATE=$piTrue)")
        logger.line("[FLOW] Computing prior-corrected metrics...")
        logProbSummary(clickDevAdj, s"predictionsClickDev (row-level) prior-corrected")
        logCalibration(clickDevAdj, "click_yn", s"predictionsClickDev (row-level) prior-corrected")
        logConfusion(clickDevAdj, "click_yn", s"predictionsClickDev (row-level) prior-corrected")
        logTopK(clickDevAdj, "click_yn", s"predictionsClickDev (row-level) prior-corrected")
      case _ =>
        logger.section("EVAL: prior-corrected probability diagnostics (skipped)")
        logger.line("[CONFIG] Prior correction disabled (TRAIN_POS_RATE or TRUE_POS_RATE not set)")
        logger.line("To enable: set env TRAIN_POS_RATE and TRUE_POS_RATE (both in (0,1)). Example: TRAIN_POS_RATE=0.3333 TRUE_POS_RATE=0.0075")
    }

    logger.section("EVAL: aggregated user-hour view (matches original evaluation style)")
    logger.line("[DATA] Aggregating predictions by user-hour...")
    val clickAgg = clickDevWithProb
      .groupBy("svc_mgmt_num", "send_ym", "send_hournum_cd")
      .agg(sum(col("indexedLabelClick")).as("indexedLabelClick"), max(col("_prob1")).as("_prob1"))
      .withColumn("indexedLabelClick", expr("case when indexedLabelClick>0 then 1.0 else 0.0 end"))
      .cache()
    logger.line("[DATA] User-hour aggregation completed and cached")
    
    logger.line("[FLOW] Computing aggregated metrics...")
    logCounts(logger, clickAgg, "indexedLabelClick", "clickAgg (user-hour, binarized label)")
    logProbSummary(clickAgg, "clickAgg (user-hour)")
    logConfusion(clickAgg, "indexedLabelClick", "clickAgg (user-hour)")
    logTopK(clickAgg, "indexedLabelClick", "clickAgg (user-hour)")

    logger.section("EVAL: AUC metrics (row-level)")
    logger.line("[FLOW] Computing ROC-AUC...")
    val rocAuc = new BinaryClassificationEvaluator().setLabelCol("click_yn").setRawPredictionCol(clickProbCol).setMetricName("areaUnderROC").evaluate(predictionsClickDev)
    logger.line(f"[PERF] ROC-AUC=$rocAuc%.6f")
    
    logger.line("[FLOW] Computing PR-AUC...")
    val prAuc = new BinaryClassificationEvaluator().setLabelCol("click_yn").setRawPredictionCol(clickProbCol).setMetricName("areaUnderPR").evaluate(predictionsClickDev)
    logger.line(f"[PERF] PR-AUC=$prAuc%.6f")
    
    logger.line(f"[PERF] predictionsClickDev: ROC-AUC=$rocAuc%.6f  PR-AUC=$prAuc%.6f  (TS is highly imbalanced; PR-AUC is usually more informative)")

    logger.section("EVAL: original-style MulticlassMetrics (aggregated user-hour, threshold=0.5)")
    logger.line("[FLOW] Computing multiclass metrics with threshold=0.5...")
    val thresholdProb = 0.5
    model.stages.foreach { stage =>
      val modelName = stage.uid
      logger.line(s"[FLOW] Processing stage: $modelName")
      
      logger.line("[DATA] Preparing prediction and labels...")
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
      logger.line("[DATA] Predictions prepared")

      logger.line("[FLOW] Computing MulticlassMetrics...")
      val metrics = new MulticlassMetrics(predictionAndLabels)
      logger.line(s"######### $modelName click 예측 결과 #########")
      metrics.labels.foreach { label =>
        logger.line(f"[PERF] Label $label: Precision=${metrics.precision(label)}%.4f Recall=${metrics.recall(label)}%.4f F1=${metrics.fMeasure(label)}%.4f")
      }
      logger.line(f"[PERF] Weighted Precision: ${metrics.weightedPrecision}%.4f")
      logger.line(f"[PERF] Weighted Recall: ${metrics.weightedRecall}%.4f")
      logger.line(f"[PERF] Accuracy: ${metrics.accuracy}%.4f")
      logger.line("[PERF] Confusion Matrix:")
      logger.line(metrics.confusionMatrix.toString())
      logger.line("[FLOW] MulticlassMetrics computed successfully")
    }
    
    logger.line("[FLOW] evalClickModel: DONE")
  }

  // [ZEPPELIN_PARAGRAPH] 49. Entrypoints
  def runTrain(): Unit = {
    initSpark()
    val cfg = loadRunConfig()
    val logger = new FileLogger(cfg.logPath)
    try {
      logger.section("RUN: train (click-only)")
      logger.line(s"[FLOW] runTrain: START")
      logger.line(s"[CONFIG] modelType=${cfg.modelType} cacheTransformed=${cfg.cacheTransformed} useSampleBy=${cfg.useSampleBy}")
      logger.line(s"[CONFIG] enableTrainingDiagnostics=${cfg.enableTrainingDiagnostics}")
      logger.line(s"[CONFIG] useWeightCorrection=${cfg.calib.useWeightCorrection}")
      cfg.calib.trainPosRate.foreach(rate => logger.line(f"[CONFIG] trainPosRate=$rate%.6f (provided, will skip driver computation)"))
      cfg.calib.truePosRate.foreach(rate => logger.line(f"[CONFIG] truePosRate=$rate%.6f"))
      logger.line(s"[CONFIG] transformedTrainPath=$transformedTrainPath")
      logger.line(s"[CONFIG] modelClickPath=$modelClickPath")

      // Ensure transformer exists in classpath/env
      logger.line(s"[MODEL] Loading transformer from: $transformerClickPath")
      val transformerClick = PipelineModel.load(transformerClickPath)
      logger.line("[MODEL] Transformer loaded successfully")
      
      val (train, _) = loadTransformed(cfg, logger)

      val model = trainClickModel(train, cfg, logger)
      
      logger.line(s"[MODEL] Saving model to: $modelClickPath")
      model.write.overwrite().save(modelClickPath)
      logger.line(s"[MODEL] Model saved successfully")
      logger.line(s"[FLOW] runTrain: DONE")
      logger.line(s"Log saved to: ${cfg.logPath}")
      if (transformerClick.uid.nonEmpty) ()
    } catch {
      case e: Exception =>
        logger.line(s"[ERROR] runTrain failed: ${e.getClass.getName}: ${e.getMessage}")
        logger.line(s"[ERROR] Stack trace:")
        e.getStackTrace.take(20).foreach(st => logger.line(s"[ERROR]   at $st"))
        throw e
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
      logger.line(s"[FLOW] runEval: START")
      logger.line(s"[CONFIG] cacheTransformed=${cfg.cacheTransformed}")
      cfg.calib.trainPosRate.foreach(rate => logger.line(f"[CONFIG] trainPosRate=$rate%.6f (for prior correction)"))
      cfg.calib.truePosRate.foreach(rate => logger.line(f"[CONFIG] truePosRate=$rate%.6f (for prior correction)"))
      logger.line(s"[CONFIG] transformedTestPath=$transformedTestPath")
      logger.line(s"[CONFIG] modelClickPath=$modelClickPath")

      logger.line(s"[MODEL] Loading transformer from: $transformerClickPath")
      val transformerClick = PipelineModel.load(transformerClickPath)
      logger.line("[MODEL] Transformer loaded")
      
      logger.line(s"[MODEL] Loading model from: $modelClickPath")
      val model = PipelineModel.load(modelClickPath)
      logger.line("[MODEL] Model loaded")
      
      val (_, test) = loadTransformed(cfg, logger)

      evalClickModel(model, transformerClick, test, cfg, logger)
      logger.line(s"[FLOW] runEval: DONE")
      logger.line(s"Log saved to: ${cfg.logPath}")
    } catch {
      case e: Exception =>
        logger.line(s"[ERROR] runEval failed: ${e.getClass.getName}: ${e.getMessage}")
        logger.line(s"[ERROR] Stack trace:")
        e.getStackTrace.take(20).foreach(st => logger.line(s"[ERROR]   at $st"))
        throw e
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
      logger.line(s"[FLOW] runAll: START")
      logger.line(s"[CONFIG] modelType=${cfg.modelType} cacheTransformed=${cfg.cacheTransformed} useSampleBy=${cfg.useSampleBy}")
      logger.line(s"[CONFIG] enableTrainingDiagnostics=${cfg.enableTrainingDiagnostics}")
      logger.line(s"[CONFIG] useWeightCorrection=${cfg.calib.useWeightCorrection}")
      cfg.calib.trainPosRate.foreach(rate => logger.line(f"[CONFIG] trainPosRate=$rate%.6f (provided, will skip driver computation)"))
      cfg.calib.truePosRate.foreach(rate => logger.line(f"[CONFIG] truePosRate=$rate%.6f"))
      logger.line(s"[CONFIG] LOG_PATH=${cfg.logPath}")

      logger.line(s"[MODEL] Loading transformer from: $transformerClickPath")
      val transformerClick = PipelineModel.load(transformerClickPath)
      logger.line("[MODEL] Transformer loaded")
      
      val (train, test) = loadTransformed(cfg, logger)

      logger.line("[FLOW] ===== PHASE 1: TRAINING =====")
      val model = trainClickModel(train, cfg, logger)
      
      logger.line(s"[MODEL] Saving model to: $modelClickPath")
      model.write.overwrite().save(modelClickPath)
      logger.line(s"[MODEL] Model saved successfully to: $modelClickPath")

      logger.line("[FLOW] ===== PHASE 2: EVALUATION =====")
      evalClickModel(model, transformerClick, test, cfg, logger)
      
      logger.line(s"[FLOW] runAll: DONE")
      logger.line(s"Log saved to: ${cfg.logPath}")
    } catch {
      case e: Exception =>
        logger.line(s"[ERROR] runAll failed: ${e.getClass.getName}: ${e.getMessage}")
        logger.line(s"[ERROR] Stack trace:")
        e.getStackTrace.take(20).foreach(st => logger.line(s"[ERROR]   at $st"))
        throw e
    } finally {
      logger.close()
    }
  }

  def run(): Unit = runAll()
}
