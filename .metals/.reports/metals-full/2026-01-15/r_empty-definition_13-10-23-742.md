error id: file://<WORKSPACE>/predict_send_time/scala/40_train_eval_and_save_models.scala:spark.
file://<WORKSPACE>/predict_send_time/scala/40_train_eval_and_save_models.scala
empty definition using pc, found symbol in pc: 
empty definition using semanticdb
empty definition using fallback
non-local guesses:
	 -PredictOstConfig.org.apache.spark.
	 -org/apache/spark/sql/functions/org/apache/spark.
	 -org/apache/spark.
	 -scala/Predef.org.apache.spark.
offset: 616
uri: file://<WORKSPACE>/predict_send_time/scala/40_train_eval_and_save_models.scala
text:
```scala
object PredictOst40TrainEvalAndSaveModels {
  def run(): Unit = {
    import PredictOstConfig._
    initSpark()

    // [ZEPPELIN_PARAGRAPH] 40. Train + evaluate + save models
    import ml.dmlc.xgboost4j.scala.spark.{XGBoostClassifier, XGBoostRegressor}
    import org.apache.spark.ml.{Pipeline, PipelineModel}
    import org.apache.spark.ml.feature.StringIndexerModel
    import org.apache.spark.ml.feature.IndexToString
    import org.apache.spark.ml.evaluation.RegressionEvaluator
    import org.apache.spark.mllib.evaluation.MulticlassMetrics
    import org.apache.spark.sql.functions._
    import org.apache.sp@@ark.ml.linalg.Vector

    // [ZEPPELIN_PARAGRAPH] 41. Load transformers + transformed datasets
    val transformerClick = PipelineModel.load(transformerClickPath)
    val transformerGap = PipelineModel.load(transformerGapPath)

    val transformedTrainDF = spark.read.parquet(transformedTrainPath).cache()
    val transformedTestDF = spark.read.parquet(transformedTestPath).cache()

    // Column names (must match vectorization step)
    val indexedLabelColClick = "indexedLabelClick"
    val indexedLabelColGap = "indexedLabelGap"
    val indexedLabelColReg = "res_utility"

    val indexedFeatureColClick = "indexedFeaturesClick"
    val indexedFeatureColGap = "indexedFeaturesGap"

    // [ZEPPELIN_PARAGRAPH] 42. Label indexers map + label converters
    val labelIndexersMap: Map[String, StringIndexerModel] =
      transformerClick.stages.collect { case sim: StringIndexerModel => sim.uid -> sim }.toMap ++
        transformerGap.stages.collect { case sim: StringIndexerModel => sim.uid -> sim }.toMap

    val labelConverterClick = new IndexToString()
      .setInputCol("prediction_click")
      .setOutputCol("predictedLabelClick")
      .setLabels(labelIndexersMap("indexer_click").labelsArray(0))

    val labelConverterGap = new IndexToString()
      .setInputCol("prediction_gap")
      .setOutputCol("predictedLabelGap")
      .setLabels(labelIndexersMap("indexer_gap").labelsArray(0))

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

    val xgbg = new XGBoostClassifier("xgbc_gap")
  .setFeaturesCol(indexedFeatureColGap)
  .setLabelCol(indexedLabelColGap)
  .setMissing(0)
  .setSeed(0)
  .setEta(0.81)
  .setMaxDepth(6)
  .setObjective("binary:logistic")
  .setNumRound(100)
  .setNumWorkers(numWorkers)
  .setEvalMetric("error")
  .setProbabilityCol("prob_xgbc_gap")
  .setPredictionCol("pred_xgbc_gap")
  .setRawPredictionCol("pred_raw_xgbc_gap")

    val xgbr = new XGBoostRegressor("xgbr_reg")
  .setFeaturesCol(indexedFeatureColGap)
  .setLabelCol(indexedLabelColReg)
  .setMissing(0)
  .setSeed(0)
  .setEta(0.01)
  .setMaxDepth(6)
  .setObjective("reg:squarederror")
  .setNumRound(100)
  .setNumWorkers(numWorkers)
  .setEvalMetric("rmse")
  .setPredictionCol("pred_xgbr_gap")

    // [ZEPPELIN_PARAGRAPH] 44. Fit models
    val pipelineModelClick = new Pipeline().setStages(Array(xgbc)).fit(
  transformedTrainDF
    .filter("cmpgn_typ=='Sales'")
    .stat
    .sampleBy(
      col(indexedLabelColClick),
      Map(
        0.0 -> 0.5,
        1.0 -> 1.0
      ),
      42L
    )
    )

    val pipelineModelGap = new Pipeline().setStages(Array(xgbg)).fit(
  transformedTrainDF
    .filter("click_yn>0")
    .stat
    .sampleBy(
      col("hour_gap"),
      Map(
        0.0 -> 1.0,
        1.0 -> 0.45
      ),
      42L
    )
    )

    val pipelineModelReg = new Pipeline().setStages(Array(xgbr)).fit(
      transformedTrainDF.filter("click_yn>0")
    )

    // [ZEPPELIN_PARAGRAPH] 45. Dev predictions (test)
    val predictionsClickDev = pipelineModelClick.transform(
  transformedTestDF
    .filter("cmpgn_typ=='Sales'")
    .dropDuplicates("svc_mgmt_num", "chnl_typ", "cmpgn_typ", "send_ym", "send_hournum_cd", "click_yn")
    )

    val predictionsGapDev = pipelineModelGap.transform(
  transformedTestDF
    .filter("cmpgn_typ=='Sales'")
    .dropDuplicates("svc_mgmt_num", "chnl_typ", "cmpgn_typ", "send_ym", "send_hournum_cd", "click_yn")
    )

    val predictionsRegDev = pipelineModelReg.transform(
      transformedTestDF.filter("click_yn>0")
    )

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
  println(s"######### $modelName click 예측 결과 #########")
  metrics.labels.foreach { label =>
    println(f"Label $label: Precision=${metrics.precision(label)}%.4f Recall=${metrics.recall(label)}%.4f F1=${metrics.fMeasure(label)}%.4f")
  }
  println(s"Weighted Precision: ${metrics.weightedPrecision}")
  println(s"Weighted Recall: ${metrics.weightedRecall}")
  println(s"Accuracy: ${metrics.accuracy}")
  println(metrics.confusionMatrix)
    }

    pipelineModelGap.stages.foreach { stage =>
  val modelName = stage.uid

  val predictionAndLabels = labelConverterGap.transform(
    predictionsGapDev
      .filter("click_yn>0")
      .withColumn("prob", expr(s"vector_to_array(prob_$modelName)[1]"))
      .groupBy("svc_mgmt_num", "send_ym", "send_hournum_cd")
      .agg(sum(col(indexedLabelColGap)).alias(indexedLabelColGap), max(col("prob")).alias("prob"))
      .withColumn(indexedLabelColGap, expr(s"case when $indexedLabelColGap>0 then cast(1.0 AS DOUBLE) else cast(0.0 AS DOUBLE) end"))
      .withColumn("prediction_gap", expr("case when prob>=0.5 then cast(1.0 AS DOUBLE) else cast(0.0 AS DOUBLE) end"))
  )
    .selectExpr("prediction_gap", s"cast($indexedLabelColGap as double)")
    .rdd
    .map(row => (row.getDouble(0), row.getDouble(1)))

  val metrics = new MulticlassMetrics(predictionAndLabels)
  println(s"######### $modelName gap 예측 결과 #########")
  metrics.labels.foreach { label =>
    println(f"Label $label: Precision=${metrics.precision(label)}%.4f Recall=${metrics.recall(label)}%.4f F1=${metrics.fMeasure(label)}%.4f")
  }
  println(s"Weighted Precision: ${metrics.weightedPrecision}")
  println(s"Weighted Recall: ${metrics.weightedRecall}")
  println(s"Accuracy: ${metrics.accuracy}")
  println(metrics.confusionMatrix)
    }

    // [ZEPPELIN_PARAGRAPH] 47. Evaluation (regression)
    val rmse = new RegressionEvaluator()
      .setLabelCol(indexedLabelColReg)
      .setPredictionCol("pred_xgbr_gap")
      .setMetricName("rmse")
      .evaluate(predictionsRegDev)
    println(f"######### xgbr_reg regression RMSE: $rmse%.4f")

    // [ZEPPELIN_PARAGRAPH] 48. Save models (for inference in another session)
    pipelineModelClick.write.overwrite().save(modelClickPath)
    pipelineModelGap.write.overwrite().save(modelGapPath)
    pipelineModelReg.write.overwrite().save(modelRegPath)

    println(s"Saved models to: click=$modelClickPath gap=$modelGapPath reg=$modelRegPath")
  }
}

```


#### Short summary: 

empty definition using pc, found symbol in pc: 