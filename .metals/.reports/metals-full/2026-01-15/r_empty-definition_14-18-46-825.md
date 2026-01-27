error id: file://<WORKSPACE>/predict_send_time/scala/40_train_eval_and_save_models.scala:
file://<WORKSPACE>/predict_send_time/scala/40_train_eval_and_save_models.scala
empty definition using pc, found symbol in pc: 
empty definition using semanticdb
empty definition using fallback
non-local guesses:
	 -PredictOstConfig.baos.
	 -PredictOstConfig.baos#
	 -PredictOstConfig.baos().
	 -org/apache/spark/sql/functions/baos.
	 -org/apache/spark/sql/functions/baos#
	 -org/apache/spark/sql/functions/baos().
	 -baos.
	 -baos#
	 -baos().
	 -scala/Predef.baos.
	 -scala/Predef.baos#
	 -scala/Predef.baos().
offset: 1663
uri: file://<WORKSPACE>/predict_send_time/scala/40_train_eval_and_save_models.scala
text:
```scala
object PredictOst40TrainEvalAndSaveModels {
  def run(): Unit = {
    import PredictOstConfig._
    initSpark()

    // [ZEPPELIN_PARAGRAPH] 40. Train + evaluate + save models (CLICK ONLY)
    // Focus: P(C=1 | T=t, X=x)
    import ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier
    import org.apache.spark.ml.{Pipeline, PipelineModel}
    import org.apache.spark.ml.feature.StringIndexerModel
    import org.apache.spark.ml.feature.IndexToString
    import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
    import org.apache.spark.mllib.evaluation.MulticlassMetrics
    import org.apache.spark.sql.functions._
    import org.apache.spark.ml.linalg.Vector

    // [ZEPPELIN_PARAGRAPH] 41-0. File logger (append)
    // Set via env:
    //   LOG_PATH=/path/to/file.log
    // If not set, a timestamped file is created under /tmp.
    val logPath = sys.env.getOrElse("LOG_PATH", s"/data/myfiles/aos_ost/predict/logs/predict_ost_click_train_${System.currentTimeMillis()}.log")
    val logWriter = new java.io.PrintWriter(new java.io.BufferedWriter(new java.io.FileWriter(logPath, /*append=*/ true)))

    try {
      def nowTs: String = java.time.ZonedDateTime.now().toString
      def logLine(s: String): Unit = {
        logWriter.println(s"[$nowTs] $s")
        logWriter.flush()
      }

    def logBlock(title: String, content: String): Unit = {
      logLine("")
      logLine("=" * 110)
      logLine(title)
      logLine("=" * 110)
      content.split("\n", -1).foreach { ln => logLine(ln) }
    }

    def captureStdout(body: => Unit): String = {
      val baos = new java.io.ByteArrayOutputStream()
      val ps = new java.io.PrintStream(ba@@os)
      try {
        Console.withOut(ps) {
          Console.withErr(ps) {
            body
          }
        }
      } finally {
        ps.flush()
        ps.close()
      }
      baos.toString("UTF-8")
    }

    def dfShowString(df: org.apache.spark.sql.DataFrame, n: Int, truncate: Int = 0): String = {
      // Capture df.show() output without relying on private Spark internals.
      // truncate=0 means no truncation in Spark show().
      val truncateArg = if (truncate <= 0) false else truncate
      captureStdout {
        truncateArg match {
          case b: Boolean => df.show(n, b)
          case i: Int => df.show(n, i)
        }
      }
    }

    def logDf(df: org.apache.spark.sql.DataFrame, title: String, n: Int, truncate: Int = 0): Unit =
      logBlock(title, dfShowString(df, n, truncate))

      // [ZEPPELIN_PARAGRAPH] 41. Load transformer + transformed datasets
      val transformerClick = PipelineModel.load(transformerClickPath)

      val cacheTransformed = sys.env.getOrElse("CACHE_TRANSFORMED", "false").toBoolean
      val useSampleBy = sys.env.getOrElse("USE_SAMPLE_BY", "false").toBoolean

      var transformedTrainDF = spark.read.parquet(transformedTrainPath)
      var transformedTestDF = spark.read.parquet(transformedTestPath)
      if (cacheTransformed) {
        transformedTrainDF = transformedTrainDF.cache()
        transformedTestDF = transformedTestDF.cache()
        // materialize caches early to avoid surprises mid-training
        transformedTrainDF.count()
        transformedTestDF.count()
      }

      // [ZEPPELIN_PARAGRAPH] 41-1. Diagnostics helpers (imbalance / sampling / calibration)
    val diagSampleFrac = sys.env.getOrElse("DIAG_SAMPLE_FRAC", "0.02").toDouble // heavy logs use sampling
    val diagTopK = sys.env.getOrElse("DIAG_TOPK", "50000").toInt
    val diagThresholds = sys.env.getOrElse("DIAG_THRESHOLDS", "0.05,0.1,0.2,0.3,0.5,0.7").split(",").toList.map(_.trim).filter(_.nonEmpty).map(_.toDouble)

    def logSection(title: String): Unit = {
      logLine("")
      logLine("=" * 110)
      logLine(title)
      logLine("=" * 110)
      logLine("")
    }

    def logCounts(df: org.apache.spark.sql.DataFrame, labelCol: String, name: String): Unit = {
      val agg = df.agg(
        count(lit(1)).as("n"),
        sum(col(labelCol)).as("pos"),
        (count(lit(1)) - sum(col(labelCol))).as("neg")
      ).collect()(0)
      val n = agg.getAs[Long]("n")
      val pos = Option(agg.get(1)).map(_.toString.toDouble).getOrElse(0.0)
      val neg = Option(agg.get(2)).map(_.toString.toDouble).getOrElse(0.0)
      val rate = if (n == 0) 0.0 else pos / n.toDouble
      val scalePosWeight = if (pos <= 0) Double.PositiveInfinity else (neg / pos)
      logLine(f"[$name] n=$n%,d pos=$pos%.0f neg=$neg%.0f pos_rate=${rate * 100}%.4f%%  (neg/pos scalePosWeight≈$scalePosWeight%.4f)")
    }

    def logByHour(df: org.apache.spark.sql.DataFrame, labelCol: String, name: String, limit: Int = 50): Unit = {
      logLine(s"[$name] click rate by send_hournum_cd (top $limit rows)")
      val byHour = df.groupBy(col("send_hournum_cd"))
        .agg(count(lit(1)).as("n"), sum(col(labelCol)).as("pos"))
        .withColumn("pos_rate", col("pos") / col("n"))
        .orderBy(desc("n"))
      logDf(byHour, s"[$name] by-hour table", limit, truncate = 0)
    }

    def withProb(df: org.apache.spark.sql.DataFrame, probCol: String): org.apache.spark.sql.DataFrame =
      df.withColumn("_prob1", expr(s"vector_to_array($probCol)[1]"))

    def logProbSummary(dfWithProb: org.apache.spark.sql.DataFrame, name: String): Unit = {
      val qs = dfWithProb.stat.approxQuantile("_prob1", Array(0.0, 0.01, 0.05, 0.1, 0.5, 0.9, 0.99, 1.0), 0.01)
      logLine(s"[$name] prob quantiles: " +
        s"p00=${qs(0)} p01=${qs(1)} p05=${qs(2)} p10=${qs(3)} p50=${qs(4)} p90=${qs(5)} p99=${qs(6)} p100=${qs(7)}")
    }

    def logCalibration(dfWithProb: org.apache.spark.sql.DataFrame, labelCol: String, name: String, bins: Int = 20): Unit = {
      // Bin by probability (0..1) into [0,1/bins),...
      val b = bins.toDouble
      logLine(s"[$name] calibration by prob bin (bins=$bins, sampled=${diagSampleFrac})")
      val cal = dfWithProb
        .sample(diagSampleFrac)
        .withColumn("_bin", floor(col("_prob1") * b) / b)
        .groupBy(col("_bin"))
        .agg(count(lit(1)).as("n"), avg(col("_prob1")).as("avg_prob"), avg(col(labelCol)).as("emp_rate"))
        .orderBy(col("_bin"))
      logDf(cal, s"[$name] calibration table", 200, truncate = 0)
    }

    def logConfusion(dfWithProb: org.apache.spark.sql.DataFrame, labelCol: String, name: String): Unit = {
      diagThresholds.foreach { thr =>
        val m = dfWithProb
          .withColumn("_pred", when(col("_prob1") >= lit(thr), lit(1.0)).otherwise(lit(0.0)))
          .agg(
            sum(when(col("_pred") === 1.0 && col(labelCol) === 1.0, 1).otherwise(0)).as("tp"),
            sum(when(col("_pred") === 1.0 && col(labelCol) === 0.0, 1).otherwise(0)).as("fp"),
            sum(when(col("_pred") === 0.0 && col(labelCol) === 0.0, 1).otherwise(0)).as("tn"),
            sum(when(col("_pred") === 0.0 && col(labelCol) === 1.0, 1).otherwise(0)).as("fn")
          ).collect()(0)
        val tp = m.getAs[Long]("tp").toDouble
        val fp = m.getAs[Long]("fp").toDouble
        val tn = m.getAs[Long]("tn").toDouble
        val fn = m.getAs[Long]("fn").toDouble
        val prec = if (tp + fp == 0) 0.0 else tp / (tp + fp)
        val rec = if (tp + fn == 0) 0.0 else tp / (tp + fn)
        val f1 = if (prec + rec == 0) 0.0 else 2 * prec * rec / (prec + rec)
        val fpr = if (fp + tn == 0) 0.0 else fp / (fp + tn)
        logLine(f"[$name] thr=$thr%.2f  TP=$tp%.0f FP=$fp%.0f TN=$tn%.0f FN=$fn%.0f  precision=$prec%.4f recall=$rec%.4f f1=$f1%.4f fpr=$fpr%.4f")
      }
    }

    def logTopK(dfWithProb: org.apache.spark.sql.DataFrame, labelCol: String, name: String): Unit = {
      val base = dfWithProb.agg(avg(col(labelCol)).as("base_rate")).collect()(0).getAs[Double]("base_rate")
      val top = dfWithProb
        .orderBy(desc("_prob1"))
        .limit(diagTopK)
        .agg(avg(col(labelCol)).as("top_rate"))
        .collect()(0)
        .getAs[Double]("top_rate")
      val lift = if (base == 0.0) Double.PositiveInfinity else top / base
      logLine(f"[$name] TopK=$diagTopK base_rate=${base * 100}%.4f%% top_rate=${top * 100}%.4f%% lift=${lift}%.2f")
    }

    // Column names (must match vectorization step)
    val indexedLabelColClick = "indexedLabelClick"

    val indexedFeatureColClick = "indexedFeaturesClick"

    // [ZEPPELIN_PARAGRAPH] 42. Label indexers map + label converters
    val labelIndexersMap: Map[String, StringIndexerModel] =
      transformerClick.stages.collect { case sim: StringIndexerModel => sim.uid -> sim }.toMap

    val labelConverterClick = new IndexToString()
      .setInputCol("prediction_click")
      .setOutputCol("predictedLabelClick")
      .setLabels(labelIndexersMap("indexer_click").labelsArray(0))

      // [ZEPPELIN_PARAGRAPH] 42-1. Data imbalance diagnostics (before any sampling)
      logSection("DIAG: label distribution (saved transformedTrain/Test DF) - BEFORE model sampling")
      logLine(s"[config] cacheTransformed=$cacheTransformed useSampleBy=$useSampleBy LOG_PATH=$logPath")
      logCounts(transformedTrainDF, "click_yn", "transformedTrainDF (raw click_yn)")
      logCounts(transformedTestDF, "click_yn", "transformedTestDF (raw click_yn)")
      logByHour(transformedTestDF, "click_yn", "transformedTestDF (raw click_yn)")

    // [ZEPPELIN_PARAGRAPH] 43. Define models (from notebook)
    val numWorkers = sys.env.getOrElse("XGB_WORKERS", "10").toInt

    val xgbc = new XGBoostClassifier("xgbc_click")
  .setLabelCol(indexedLabelColClick)
  .setFeaturesCol(indexedFeatureColClick)
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

    // [ZEPPELIN_PARAGRAPH] 44. Fit models
    // NOTE: Training uses additional down-sampling on the (already vectorized) dataset.
    // We print BEFORE/AFTER distributions to avoid accidental over-biasing.
      val clickTrainBase = transformedTrainDF.filter("cmpgn_typ=='Sales'")

      val clickTrainFinal =
        if (useSampleBy) {
          val sampled = clickTrainBase.stat.sampleBy(
            col(indexedLabelColClick),
            Map(0.0 -> 0.5, 1.0 -> 1.0),
            42L
          )
          logSection("DIAG: click model training dataset distribution (Sales) - BEFORE vs AFTER sampleBy")
          logCounts(clickTrainBase, indexedLabelColClick, "clickTrainBase (Sales, indexedLabelClick)")
          logCounts(sampled, indexedLabelColClick, "clickTrainSampled (Sales, indexedLabelClick)")
          sampled
        } else {
          logSection("DIAG: click model training dataset distribution (Sales) - sampleBy disabled")
          logCounts(clickTrainBase, indexedLabelColClick, "clickTrainBase (Sales, indexedLabelClick)")
          clickTrainBase
        }

      val pipelineModelClick = new Pipeline().setStages(Array(xgbc)).fit(clickTrainFinal)

    // [ZEPPELIN_PARAGRAPH] 45. Dev predictions (test)
    val predictionsClickDev = pipelineModelClick.transform(
  transformedTestDF
    .filter("cmpgn_typ=='Sales'")
    .dropDuplicates("svc_mgmt_num", "chnl_typ", "cmpgn_typ", "send_ym", "send_hournum_cd", "click_yn")
    )

    // [ZEPPELIN_PARAGRAPH] 45-1. Dev-set diagnostics (row-level, BEFORE aggregation)
    logSection("DIAG: dev/test (row-level) click_yn distribution & prob diagnostics")
    logCounts(predictionsClickDev, "click_yn", "predictionsClickDev (Sales, row-level click_yn)")

    val clickProbCol = s"prob_${pipelineModelClick.stages.head.uid}"
    val clickDevWithProb = withProb(predictionsClickDev, clickProbCol).cache()
    logProbSummary(clickDevWithProb, s"predictionsClickDev (row-level) $clickProbCol")
    logCalibration(clickDevWithProb, "click_yn", s"predictionsClickDev (row-level) $clickProbCol")
    logConfusion(clickDevWithProb, "click_yn", s"predictionsClickDev (row-level) $clickProbCol")
    logTopK(clickDevWithProb, "click_yn", s"predictionsClickDev (row-level) $clickProbCol")

    // [ZEPPELIN_PARAGRAPH] 45-2. Dev-set diagnostics (aggregated to svc_mgmt_num + hour)
    // This matches how the original notebook evaluates (max prob per user-hour, label summed then binarized).
    logSection("DIAG: dev/test (aggregated user-hour) metrics (matches original evaluation style)")
    val clickAgg = clickDevWithProb
      .groupBy("svc_mgmt_num", "send_ym", "send_hournum_cd")
      .agg(sum(col(indexedLabelColClick)).as(indexedLabelColClick), max(col("_prob1")).as("_prob1"))
      .withColumn(indexedLabelColClick, expr(s"case when $indexedLabelColClick>0 then 1.0 else 0.0 end"))
      .cache()

    logCounts(clickAgg, indexedLabelColClick, "clickAgg (user-hour, binarized label)")
    logProbSummary(clickAgg, "clickAgg (user-hour)")
    logConfusion(clickAgg, indexedLabelColClick, "clickAgg (user-hour)")
    logTopK(clickAgg, indexedLabelColClick, "clickAgg (user-hour)")

    // [ZEPPELIN_PARAGRAPH] 45-3. AUC metrics (ROC-AUC / PR-AUC)
    logSection("DIAG: AUC metrics (dev/test, row-level)")
    val rocAuc = new BinaryClassificationEvaluator()
      .setLabelCol("click_yn")
      .setRawPredictionCol(clickProbCol) // probability vector is acceptable in Spark evaluator
      .setMetricName("areaUnderROC")
      .evaluate(predictionsClickDev)

    val prAuc = new BinaryClassificationEvaluator()
      .setLabelCol("click_yn")
      .setRawPredictionCol(clickProbCol)
      .setMetricName("areaUnderPR")
      .evaluate(predictionsClickDev)

    logLine(f"[predictionsClickDev] ROC-AUC=$rocAuc%.6f  PR-AUC=$prAuc%.6f  (NOTE: TS is highly imbalanced; PR-AUC is usually more informative)")

    // [ZEPPELIN_PARAGRAPH] 46. Evaluation (classification)
    spark.udf.register("vector_to_array", (v: Vector) => v.toArray)
    val thresholdProb = 0.5

    pipelineModelClick.stages.foreach { stage =>
  val modelName = stage.uid

  val predictionAndLabels = labelConverterClick.transform(
    predictionsClickDev
      .withColumn("prob", expr(s"vector_to_array(prob_$modelName)[1]"))
      .groupBy("svc_mgmt_num", "send_ym", "send_hournum_cd")
      .agg(sum(col(indexedLabelColClick)).alias(indexedLabelColClick), max(col("prob")).alias("prob"))
      .withColumn(indexedLabelColClick, expr(s"case when $indexedLabelColClick>0 then cast(1.0 AS DOUBLE) else cast(0.0 AS DOUBLE) end"))
      .withColumn("prediction_click", expr(s"case when prob>=$thresholdProb then cast(1.0 AS DOUBLE) else cast(0.0 AS DOUBLE) end"))
  )
    .selectExpr("prediction_click", s"cast($indexedLabelColClick as double)")
    .rdd
    .map(row => (row.getDouble(0), row.getDouble(1)))

  val metrics = new MulticlassMetrics(predictionAndLabels)
  logLine(s"######### $modelName click 예측 결과 #########")
  metrics.labels.foreach { label =>
    logLine(f"Label $label: Precision=${metrics.precision(label)}%.4f Recall=${metrics.recall(label)}%.4f F1=${metrics.fMeasure(label)}%.4f")
  }
  logLine(s"Weighted Precision: ${metrics.weightedPrecision}")
  logLine(s"Weighted Recall: ${metrics.weightedRecall}")
  logLine(s"Accuracy: ${metrics.accuracy}")
  logLine("Confusion Matrix:")
  logLine(metrics.confusionMatrix.toString())
    }

    // [ZEPPELIN_PARAGRAPH] 48. Save click model (for inference in another session)
    pipelineModelClick.write.overwrite().save(modelClickPath)

      logLine(s"Saved click model to: $modelClickPath")
      logLine(s"Log saved to: $logPath")
    } finally {
      // Ensure log is flushed even on Spark job failures
      logWriter.flush()
      logWriter.close()
    }
  }
}

```


#### Short summary: 

empty definition using pc, found symbol in pc: 