error id: file://<WORKSPACE>/predict_send_time/scala/60_score_and_save.scala:PipelineModel.
file://<WORKSPACE>/predict_send_time/scala/60_score_and_save.scala
empty definition using pc, found symbol in pc: 
empty definition using semanticdb
empty definition using fallback
non-local guesses:
	 -PredictOstConfig.PipelineModel.
	 -org/apache/spark/ml/PipelineModel.
	 -org/apache/spark/sql/functions/PipelineModel.
	 -PipelineModel.
	 -scala/Predef.PipelineModel.
offset: 500
uri: file://<WORKSPACE>/predict_send_time/scala/60_score_and_save.scala
text:
```scala
object PredictOst60ScoreAndSave {
  def run(): Unit = {
    import PredictOstConfig._
    initSpark()

    // [ZEPPELIN_PARAGRAPH] 60. Score predDFRev with saved transformers/models and write propensity scores
    import org.apache.spark.ml.PipelineModel
    import org.apache.spark.ml.linalg.Vector
    import org.apache.spark.sql.functions._

    // [ZEPPELIN_PARAGRAPH] 61. Load transformers + models
    val transformerClick = PipelineModel.load(transformerClickPath)
    val transformerGap = Pip@@elineModel.load(transformerGapPath)

    val pipelineModelClick = PipelineModel.load(modelClickPath)
    val pipelineModelGap = PipelineModel.load(modelGapPath)

    spark.udf.register("vector_to_array", (v: Vector) => v.toArray)

    // [ZEPPELIN_PARAGRAPH] 62. Load predDFRev (from previous step)
    val predDFRev = spark.table("pred_df_rev")

    // [ZEPPELIN_PARAGRAPH] 63. Score per suffix (0-f) and save
    (0 to 15).map(_.toHexString).foreach { suffix =>
      println(s"Scoring suffix=$suffix")

      val predPart = predDFRev.filter(s"svc_mgmt_num like '%$suffix'")
      val transformedPredDF = transformerGap.transform(transformerClick.transform(predPart))

      val predictionsSVCClick = pipelineModelClick.transform(
        transformedPredDF.dropDuplicates("svc_mgmt_num", "chnl_typ", "cmpgn_typ", "send_ym", "send_hournum_cd", "click_yn")
      )

      val predictionsSVCFinal = pipelineModelGap.transform(predictionsSVCClick)

      val probClickExpr = pipelineModelClick.stages.map(m => s"vector_to_array(prob_${m.uid})[1]").mkString(",")
      val probGapExpr = pipelineModelGap.stages.map(m => s"vector_to_array(prob_${m.uid})[1]").mkString(",")

      val predictedPropensityScoreDF = predictionsSVCFinal
        .withColumn("prob_click", expr(s"aggregate(array($probClickExpr), 0D, (acc, x) -> acc + x)"))
        .withColumn("prob_gap", expr(s"aggregate(array($probGapExpr), 0D, (acc, x) -> acc + x)"))
        .withColumn("propensity_score", expr("prob_click*prob_gap"))

      predictedPropensityScoreDF
        .selectExpr(
          "svc_mgmt_num",
          "send_ym",
          "send_hournum_cd send_hour",
          "ROUND(prob_click, 4) prob_click",
          "ROUND(prob_gap, 4) prob_gap",
          "ROUND(propensity_score, 4) propensity_score"
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

```


#### Short summary: 

empty definition using pc, found symbol in pc: 