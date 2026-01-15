object PredictOst30VectorizeFitTransformSave {
  def run(): Unit = {
    import PredictOstConfig._
    initSpark()

    // [ZEPPELIN_PARAGRAPH] 30. Vectorize (fit transformers) + transform train/test + save
    import org.apache.spark.sql.types.{ArrayType, NumericType, StringType}
    import org.apache.spark.sql.functions._
    import org.apache.spark.ml.{Pipeline, PipelineStage}

    // [ZEPPELIN_PARAGRAPH] 31. Load train/test rev datasets
    val trainDFRev = spark.read.parquet(trainDFRevPath)
      .withColumn("hour_gap", expr("case when res_utility>=1.0 then 1 else 0 end"))
      .cache()

    val testDFRev = spark.read.parquet(testDFRevPath)
      .withColumn("hour_gap", expr("case when res_utility>=1.0 then 1 else 0 end"))
      .cache()

    // [ZEPPELIN_PARAGRAPH] 32. Infer feature columns from schema
    val noFeatureCols = Set("click_yn", "hour_gap", "chnl_typ", "cmpgn_typ")
    val baseCols = Set("cmpgn_num", "svc_mgmt_num", "chnl_typ", "cmpgn_typ", "send_ym", "send_dt", "feature_ym", "click_yn", "res_utility", "suffix")

    val tokenCols = trainDFRev.schema.fields
      .collect { case f if f.name.endsWith("_token") && f.dataType.isInstanceOf[ArrayType] => f.name }
      .distinct

    val vectorCols = trainDFRev.columns.filter(_.endsWith("_vec")).distinct

    val continuousCols = trainDFRev.schema.fields
      .collect { case f if f.dataType.isInstanceOf[NumericType] && !tokenCols.contains(f.name) && !vectorCols.contains(f.name) && !baseCols.contains(f.name) && !noFeatureCols.contains(f.name) => f.name }
      .distinct

    val categoryCols = trainDFRev.schema.fields
      .collect { case f if f.dataType == StringType && !tokenCols.contains(f.name) && !vectorCols.contains(f.name) && !baseCols.contains(f.name) && !noFeatureCols.contains(f.name) => f.name }
      .distinct

    // [ZEPPELIN_PARAGRAPH] 33. Pipeline builder (from notebook)
    def makePipeline(
  labelCols: Array[Map[String, String]] = Array.empty,
  indexedFeatureCol: String = "indexed_features",
  scaledFeatureCol: String = "scaled_features",
  selectedFeatureCol: String = "selected_features",
  tokenCols: Array[String] = Array.empty,
  vectorCols: Array[String] = Array.empty,
  continuousCols: Array[String] = Array.empty,
  categoryCols: Array[String] = Array.empty,
  doNotHashingCateCols: Array[String] = Array.empty,
  doNotHashingContCols: Array[String] = Array.empty,
  useSelector: Boolean = false,
  useScaling: Boolean = false,
  tokenColsEmbCols: Array[String] = Array.empty,
  featureHasherNumFeature: Int = 256,
  featureHashColNm: String = "feature_hashed",
  colNmSuffix: String = "#",
  params: Map[String, Any]
) = {

  val minDF = params.getOrElse("minDF", 1).asInstanceOf[Int]
  val minTF = params.getOrElse("minTF", 5).asInstanceOf[Int]
  val embSize = params.getOrElse("embSize", 10).asInstanceOf[Int]
  val vocabSize = params.getOrElse("vocabSize", 30).asInstanceOf[Int]
  val numParts = params.getOrElse("numParts", 10).asInstanceOf[Int]

  var featureListForAssembler = continuousCols ++ vectorCols
  val transformPipeline = new Pipeline().setStages(Array[PipelineStage]())

  if (labelCols.nonEmpty) {
    labelCols.foreach { map =>
      val stage = new StringIndexer(map("indexer_nm"))
        .setInputCol(map("col_nm"))
        .setOutputCol(map("label_nm"))
        .setHandleInvalid("skip")
      transformPipeline.setStages(transformPipeline.getStages ++ Array(stage))
    }
  }

  val tokenColsEmb = tokenCols.filter(x => tokenColsEmbCols.exists(x.contains))
  val tokenColsCnt = tokenCols.filterNot(tokenColsEmb.contains)

  if (embSize > 0 && tokenColsEmb.nonEmpty) {
    val embEncoder = tokenColsEmb.map { c =>
      new Word2Vec()
        .setNumPartitions(numParts)
        .setSeed(46)
        .setVectorSize(embSize)
        .setMinCount(minTF)
        .setInputCol(c)
        .setOutputCol(c + "_embvec")
    }
    transformPipeline.setStages(transformPipeline.getStages ++ embEncoder)
    featureListForAssembler ++= tokenColsEmb.map(_ + "_" + colNmSuffix + "_embvec")
  }

  if (tokenColsCnt.nonEmpty) {
    val cntVectorizer = tokenColsCnt.map { c =>
      new HashingTF()
        .setInputCol(c)
        .setOutputCol(c + "_" + colNmSuffix + "_cntvec")
        .setNumFeatures(vocabSize)
    }
    transformPipeline.setStages(transformPipeline.getStages ++ cntVectorizer)
    featureListForAssembler ++= tokenColsCnt.map(_ + "_" + colNmSuffix + "_cntvec")
  }

  if (featureHasherNumFeature > 0 && categoryCols.nonEmpty) {
    val featureHasher = new FeatureHasher()
      .setNumFeatures(featureHasherNumFeature)
      .setInputCols(
        (continuousCols ++ categoryCols)
          .filterNot(doNotHashingContCols.contains)
          .filterNot(doNotHashingCateCols.contains)
      )
      .setOutputCol(featureHashColNm)

    transformPipeline.setStages(transformPipeline.getStages ++ Array(featureHasher))
    featureListForAssembler = featureListForAssembler.filterNot(continuousCols.contains)
    featureListForAssembler ++= Array(featureHashColNm)

    if (doNotHashingCateCols.nonEmpty) {
      val categoryIndexerList = doNotHashingCateCols.map(c => new StringIndexer().setInputCol(c).setOutputCol(c + "_" + colNmSuffix + "_index").setHandleInvalid("keep"))
      val encoder = new OneHotEncoder().setInputCols(doNotHashingCateCols.map(c => c + "_" + colNmSuffix + "_index")).setOutputCols(doNotHashingCateCols.map(c => c + "_" + colNmSuffix + "_enc")).setHandleInvalid("keep")
      transformPipeline.setStages(transformPipeline.getStages ++ categoryIndexerList ++ Array(encoder))
      featureListForAssembler ++= doNotHashingCateCols.map(_ + "_" + colNmSuffix + "_enc")
    }

    if (doNotHashingContCols.nonEmpty) {
      featureListForAssembler ++= doNotHashingContCols
    }
  } else if (featureHasherNumFeature < 1 && categoryCols.nonEmpty) {
    val categoryIndexerList = categoryCols.map(c => new StringIndexer().setInputCol(c).setOutputCol(c + "_" + colNmSuffix + "_index").setHandleInvalid("keep"))
    val encoder = new OneHotEncoder().setInputCols(categoryCols.map(c => c + "_" + colNmSuffix + "_index")).setOutputCols(categoryCols.map(c => c + "_" + colNmSuffix + "_enc")).setHandleInvalid("keep")
    transformPipeline.setStages(transformPipeline.getStages ++ categoryIndexerList ++ Array(encoder))
    featureListForAssembler ++= categoryCols.map(_ + "_" + colNmSuffix + "_enc")
  }

  val assembler = new VectorAssembler().setInputCols(featureListForAssembler).setOutputCol(indexedFeatureCol).setHandleInvalid("keep")
  transformPipeline.setStages(transformPipeline.getStages ++ Array(assembler))

  if (useSelector) {
    val selector = new VarianceThresholdSelector().setVarianceThreshold(8.0).setFeaturesCol(indexedFeatureCol).setOutputCol(selectedFeatureCol)
    transformPipeline.setStages(transformPipeline.getStages ++ Array(selector))
  }

  if (useScaling) {
    val inputFeatureCol = if (useSelector) selectedFeatureCol else indexedFeatureCol
    val scaler = new MinMaxScaler().setInputCol(inputFeatureCol).setOutputCol(scaledFeatureCol)
    transformPipeline.setStages(transformPipeline.getStages ++ Array(scaler))
  }

      transformPipeline
    }

    // [ZEPPELIN_PARAGRAPH] 34. Fit transformers + transform train/test
    val tokenColsEmbCols = Array("app_usage_token")
    val featureHasherNumFeature = 128

    var nodeNumber = 10
    var coreNumber = 32
    try {
      nodeNumber = spark.conf.get("spark.executor.instances").toInt
      coreNumber = spark.conf.get("spark.executor.cores").toInt
    } catch { case _: Exception => () }

    val params: Map[String, Any] = Map("minDF" -> 1, "minTF" -> 5, "embSize" -> 30, "vocabSize" -> 30, "numParts" -> nodeNumber)

    val indexedLabelColClick = "indexedLabelClick"
    val indexedLabelColGap = "indexedLabelGap"
    val indexedLabelColReg = "res_utility"

    val labelColsClick = Array(Map("indexer_nm" -> "indexer_click", "col_nm" -> "click_yn", "label_nm" -> indexedLabelColClick))
    val labelColsGap = Array(Map("indexer_nm" -> "indexer_gap", "col_nm" -> "hour_gap", "label_nm" -> indexedLabelColGap))

    val indexedFeatureColClick = "indexedFeaturesClick"
    val indexedFeatureColGap = "indexedFeaturesGap"

    val doNotHashingCateCols = Array("send_hournum_cd")
    val doNotHashingContCols = Array("click_cnt")

    val onlyGapFeature = Array[String]()

    val transformPipelineClick = makePipeline(
  labelCols = labelColsClick,
  indexedFeatureCol = indexedFeatureColClick,
  tokenCols = Array("app_usage_token"),
  vectorCols = vectorCols.filterNot(onlyGapFeature.contains),
  continuousCols = continuousCols.filterNot(onlyGapFeature.contains),
  categoryCols = categoryCols.filterNot(onlyGapFeature.contains),
  doNotHashingCateCols = doNotHashingCateCols,
  doNotHashingContCols = doNotHashingContCols,
  params = params,
  useSelector = false,
  featureHasherNumFeature = featureHasherNumFeature,
  featureHashColNm = "feature_hashed_click",
  colNmSuffix = "click"
    )

    val transformPipelineGap = makePipeline(
  labelCols = labelColsGap,
  indexedFeatureCol = indexedFeatureColGap,
  tokenCols = Array("app_usage_token"),
  vectorCols = vectorCols,
  continuousCols = continuousCols,
  categoryCols = categoryCols,
  doNotHashingCateCols = doNotHashingCateCols,
  doNotHashingContCols = doNotHashingContCols,
  params = params,
  useSelector = false,
  featureHasherNumFeature = featureHasherNumFeature,
  featureHashColNm = "feature_hashed_gap",
  colNmSuffix = "gap"
    )

    val transformerClick = transformPipelineClick.fit(trainDFRev.sample(0.3))
    // [CLICK_ONLY] gap/fast-response transformer is disabled for now
    // val transformerGap = transformPipelineGap.fit(trainDFRev.sample(0.3))

    var transformedTrainDF = transformerClick.transform(trainDFRev)
    var transformedTestDF = transformerClick.transform(testDFRev)
    // transformedTrainDF = transformerGap.transform(transformedTrainDF)
    // transformedTestDF = transformerGap.transform(transformedTestDF)

    // [ZEPPELIN_PARAGRAPH] 35. Save transformers + transformed datasets
    transformerClick.write.overwrite().save(transformerClickPath)
    // transformerGap.write.overwrite().save(transformerGapPath)

    transformedTrainDF.write
      .mode("overwrite")
      .partitionBy("send_ym", "send_hournum_cd", "suffix")
      .parquet(transformedTrainPath)

    transformedTestDF.write
      .mode("overwrite")
      .partitionBy("send_ym", "send_hournum_cd", "suffix")
      .parquet(transformedTestPath)
  }
}
