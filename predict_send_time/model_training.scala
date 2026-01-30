// =============================================================================
// MMS Click Prediction - Model Training and Service Data Generation
// =============================================================================
// 이 코드는 data_transformation.scala에서 생성된 transformed 데이터를 읽어서:
// 1. Transformed Train/Test 데이터 로딩
// 2. 학습, 테스트, 서비스용 데이터 준비
// 3. Click 예측 모델 학습
// 4. Gap 예측 모델 학습
// 5. 학습된 모델 저장
//
// [실행 방법]
// 1. data_transformation.scala 완료 후 실행
// 2. Paragraph 1-8: 순차 실행
// =============================================================================


// ===== Paragraph 1: Imports and Configuration =====

import com.microsoft.azure.synapse.ml.causal
import ml.dmlc.xgboost4j.scala.spark.{XGBoostClassificationModel, XGBoostClassifier, XGBoostRegressor}
import org.apache.spark.ml.classification._
import org.apache.spark.ml.regression._
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.{Pipeline, PipelineModel, linalg}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.explode
import org.apache.spark.sql.types.{DateType, StringType, TimestampType}
import org.apache.spark.sql.{DataFrame, SparkSession, functions => F}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator, RegressionEvaluator}

import java.sql.Date
import java.text.DecimalFormat
import java.time.format.DateTimeFormatter
import java.time.{ZoneId, ZonedDateTime}
import com.microsoft.azure.synapse.{ml => sml}
import com.microsoft.azure.synapse.ml.lightgbm.LightGBMClassifier

import collection.JavaConverters._
import scala.collection.JavaConverters._
import org.apache.spark.ml.linalg.Vector

// 대용량 처리 최적화를 위한 Spark 설정
spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
spark.conf.set("spark.sql.adaptive.advisoryPartitionSizeInBytes", "128MB")
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "50m")

// // 메모리 최적화 설정 (OOM 방지)
// spark.conf.set("spark.executor.memoryOverhead", "4g")
// spark.conf.set("spark.memory.fraction", "0.8")
// spark.conf.set("spark.memory.storageFraction", "0.3")

// 동적 파티션 개수 계산
val executorInstances = spark.sparkContext.getConf.getInt("spark.executor.instances", 10)
val executorCores = spark.sparkContext.getConf.getInt("spark.executor.cores", 5)
val optimalPartitions = executorInstances * executorCores
spark.conf.set("spark.sql.shuffle.partitions", (optimalPartitions * 4).toString)

println(s"Spark configuration set for model training")
println(s"  Optimal partitions: $optimalPartitions (Executors: $executorInstances × Cores: $executorCores)")


// ===== Paragraph 2: Configuration Variables =====

// =============================================================================
// 데이터 버전 및 경로 설정
// =============================================================================

// ⚠️  중요: data_transformation.scala에서 저장한 버전과 일치해야 합니다.
val transformedDataVersion = "1"  // data_transformation.scala의 transformedDataVersion과 일치

// data_transformation.scala에서 사용한 날짜 설정 (재현을 위해 동일하게 설정)
val predictionDTSta = "20251201"  // 테스트 시작일
val predictionDTEnd = "20260101"  // 테스트 종료일

// 학습 기간 정보 (data_transformation.scala와 동일)
val trainPeriod = 3  // 학습 개월 수
val trainSendMonth = "202511"  // 학습 기준 월

// Helper functions (data_transformation.scala에서 사용한 것과 동일)
def getPreviousMonths(startMonthStr: String, periodM: Int): Array[String] = {
  val formatter = DateTimeFormatter.ofPattern("yyyyMM")
  val startMonth = java.time.YearMonth.parse(startMonthStr, formatter)
  var resultMonths = scala.collection.mutable.ListBuffer[String]()
  var currentMonth = startMonth

  for (i <- 0 until periodM) {
    resultMonths.prepend(currentMonth.format(formatter))
    currentMonth = currentMonth.minusMonths(1)
  }
  resultMonths.toArray
}

val trainSendYmList = getPreviousMonths(trainSendMonth, trainPeriod)

spark.udf.register("vector_to_array", (v: Vector) => v.toArray)

// 경로 구성 (data_transformation.scala와 일치)
val transformerClickPath = s"aos/sto/transformPipelineClick_v${transformedDataVersion}_${trainSendYmList.head}-${trainSendYmList.last}"
val transformerGapPath = s"aos/sto/transformPipelineGap_v${transformedDataVersion}_${trainSendYmList.head}-${trainSendYmList.last}"
val trainDataPath = s"aos/sto/transformedTrainDF_v${transformedDataVersion}_${trainSendYmList.head}-${trainSendYmList.last}"
val testDataPath = s"aos/sto/transformedTestDF_v${transformedDataVersion}_${predictionDTSta}-${predictionDTEnd}"

// 모델 저장 버전
val modelVersion = "1"  // 모델 저장 버전 번호

// Feature column 이름 (data_transformation.scala와 일치)
val indexedLabelColClick = "click_yn"
val indexedLabelColGap = "hour_gap"
val indexedLabelColReg = "res_utility"

val indexedFeatureColClick = "indexedFeaturesClick"
val scaledFeatureColClick = "scaledFeaturesClick"
val selectedFeatureColClick = "selectedFeaturesClick"

val indexedFeatureColGap = "indexedFeaturesGap"
val scaledFeatureColGap = "scaledFeaturesGap"
val selectedFeatureColGap = "selectedFeaturesGap"

println("=" * 80)
println("Configuration Summary")
println("=" * 80)
println(s"Data Configuration:")
println(s"  - Transformed Data Version: $transformedDataVersion")
println(s"  - Model Version: $modelVersion")
println(s"  - Training Period: ${trainSendYmList.head} ~ ${trainSendYmList.last} ($trainPeriod months)")
println(s"  - Test Period: $predictionDTSta ~ $predictionDTEnd")
println()
println(s"Data Paths:")
println(s"  - Click Transformer: $transformerClickPath")
println(s"  - Gap Transformer: $transformerGapPath")
println(s"  - Training Data: $trainDataPath")
println(s"  - Test Data: $testDataPath")
println("=" * 80)


// ===== Paragraph 3: Load Transformers and Transformed Data =====

println("=" * 80)
println("Loading Transformers and Transformed Data")
println("=" * 80)

// 1. Transformer 로딩
println(s"Loading Click transformer from: $transformerClickPath")
val transformerClick = PipelineModel.load(transformerClickPath)
println("Click transformer loaded successfully")

println(s"Loading Gap transformer from: $transformerGapPath")
val transformerGap = PipelineModel.load(transformerGapPath)
println("Gap transformer loaded successfully")

// 2. Transformed training data 로딩
// ✅ persist 필요: Click/Gap/Reg 모델에서 여러 번 재사용 (3회)
println(s"Loading transformed training data from: $trainDataPath")
val transformedTrainDF = spark.read.parquet(trainDataPath)
    .persist(StorageLevel.MEMORY_AND_DISK_SER)

println("Transformed training data loaded and cached")

// 3. Transformed test data 로딩
// ✅ persist 필요: Paragraph 9에서 예측 생성 시 재사용 (2회)
println(s"Loading transformed test data from: $testDataPath")
val transformedTestDF = spark.read.parquet(testDataPath)
    .persist(StorageLevel.MEMORY_AND_DISK_SER)

println("Transformed test data loaded and cached")
println("=" * 80)


// ===== Paragraph 4: ML Model Definitions =====

println("=" * 80)
println("Defining ML Models")
println("=" * 80)

// ========================================
// Click 예측 모델 정의
// ========================================

// GBTClassifier for Click
val gbtc = new GBTClassifier("gbtc_click")
  .setLabelCol(indexedLabelColClick)
  .setFeaturesCol(indexedFeatureColClick)
  .setMaxIter(100)
  .setMaxDepth(4)
  .setFeatureSubsetStrategy("auto")
  .setPredictionCol("pred_gbtc_click")
  .setProbabilityCol("prob_gbtc_click")
  .setRawPredictionCol("pred_raw_gbtc_click")

println("GBT Click model defined")

// FMClassifier for Click
val fmc = new FMClassifier("fmc_click")
  .setLabelCol(indexedLabelColClick)
  .setFeaturesCol(indexedFeatureColClick)
  .setStepSize(0.01)
  .setPredictionCol("pred_fmc_click")
  .setProbabilityCol("prob_fmc_click")
  .setRawPredictionCol("pred_raw_fmc_click")

println("FM Click model defined")

// XGBoost for Click
val xgbc = new XGBoostClassifier("xgbc_click")
    .setLabelCol(indexedLabelColClick)
    .setFeaturesCol(indexedFeatureColClick)
    .setMissing(0)
    .setSeed(0)
    .setMaxDepth(4)
    .setObjective("binary:logistic")
    .setNumRound(50)
    .setWeightCol("sample_weight")
    .setNumWorkers(10)
    .setEvalMetric("auc")
    .setProbabilityCol("prob_xgbc_click")
    .setPredictionCol("pred_xgbc_click")
    .setRawPredictionCol("pred_raw_xgbc_click")

println("XGBoost Click model defined")

// LightGBM for Click
val lgbmc = new LightGBMClassifier("lgbmc_click")
  .setLabelCol(indexedLabelColClick)
  .setFeaturesCol(indexedFeatureColClick)
  .setObjective("binary")
  .setNumIterations(50)
  .setMaxDepth(4)
  .setNumLeaves(63)
  .setNumTasks(10)
  .setSeed(0)
  .setProbabilityCol("prob_lgbmc_click")
  .setPredictionCol("pred_lgbmc_click")
  .setRawPredictionCol("pred_raw_lgbmc_click")
  .setMetric("binary_error")
  .setBoostingType("gbdt")
  .setFeatureFraction(0.8)
  .setBaggingFraction(0.8)
  .setBaggingFreq(5)

println("LightGBM Click model defined")

// ========================================
// Gap 예측 모델 정의
// ========================================

// GBT for Gap
val gbtg = new GBTClassifier("gbtc_gap")
  .setLabelCol(indexedLabelColGap)
  .setFeaturesCol(indexedFeatureColGap)
  .setMaxIter(100)
  .setMaxDepth(6)
  .setStepSize(0.1)
  .setSubsamplingRate(0.8)
  .setFeatureSubsetStrategy("sqrt")
  .setMinInstancesPerNode(10)
  .setMinInfoGain(0.001)
  .setPredictionCol("pred_gbtc_gap")
  .setProbabilityCol("prob_gbtc_gap")
  .setRawPredictionCol("pred_raw_gbtc_gap")

println("GBT Gap model defined")

// XGBoost for Gap
val xgbg = new XGBoostClassifier("xgbc_gap")
    .setLabelCol(indexedLabelColGap)
    .setFeaturesCol(indexedFeatureColGap)
    .setMissing(0)
    .setSeed(0)
    .setMaxDepth(4)
    .setObjective("binary:logistic")
    .setNumRound(50)
    .setNumWorkers(10)
    .setEvalMetric("auc")
    .setProbabilityCol("prob_xgbc_gap")
    .setPredictionCol("pred_xgbc_gap")
    .setRawPredictionCol("pred_raw_xgbc_gap")

println("XGBoost Gap model defined")

// ========================================
// Response Utility Regression 모델 정의
// ========================================

// XGBoost for Regression
val xgbr = new XGBoostRegressor("xgbr")
  .setLabelCol(indexedLabelColReg)
  .setFeaturesCol(indexedFeatureColGap)
  .setMissing(0)
  .setSeed(0)
  .setMaxDepth(6)
  .setObjective("reg:squarederror")
  .setNumRound(100)
  .setNumWorkers(10)
  .setEvalMetric("rmse")
  .setPredictionCol("pred_xgbr")

println("XGBoost Regression model defined")

println("=" * 80)
println("All models defined successfully")
println("=" * 80)


// ===== Paragraph 5: Click Prediction Model Training =====

println("=" * 80)
println("Training Click Prediction Model")
println("=" * 80)

// 학습에 사용할 모델 선택 (gbtc, xgbc, fmc, lgbmc 중 선택)
val modelClickforCV = gbtc  // 기본값: GBT

val pipelineMLClick = new Pipeline().setStages(Array(modelClickforCV))

// 학습 데이터 샘플링 최적화
// 샘플링 비율 설정 (neg:pos)
// - 0.3 (3:1) = 권장 (F1 최적)
val negSampleRatioClick = 0.3

println(s"Preparing training samples for Click model...")
println(s"  - Negative sampling ratio: $negSampleRatioClick")
println(s"  - Campaign type filter: Sales")

// ❌ repartition 제거: ML fit 전 repartition은 AQE가 자동 최적화
// ❌ count() 제거: 드라이버 액션 불필요 (로깅용)
val trainSampleClick = transformedTrainDF
    .filter("cmpgn_typ=='Sales'")
    .stat.sampleBy(
        F.col(indexedLabelColClick),
        Map(
            0.0 -> negSampleRatioClick,  // Negative 샘플링
            1.0 -> 1.0                   // Positive 전체 사용
        ),
        42L
    )
    .withColumn("sample_weight", F.expr(s"case when $indexedLabelColClick>0.0 then 10.0 else 1.0 end"))

println("Training samples prepared")

println("Training Click prediction model...")
val startTimeClick = System.currentTimeMillis()

val pipelineModelClick = pipelineMLClick.fit(trainSampleClick)

val elapsedClick = (System.currentTimeMillis() - startTimeClick) / 1000
println(s"Click model training completed in ${elapsedClick}s")

println("=" * 80)


// ===== Paragraph 6: Click-to-Action Gap Model Training =====

println("=" * 80)
println("Training Gap Prediction Model")
println("=" * 80)

// 학습에 사용할 모델 선택
val modelGapforCV = xgbg  // 기본값: XGBoost

val pipelineMLGap = new Pipeline().setStages(Array(modelGapforCV))

// Gap 모델 학습 데이터 샘플링
val posSampleRatioGap = 0.45

println(s"Preparing training samples for Gap model...")
println(s"  - Positive sampling ratio: $posSampleRatioGap")
println(s"  - Click filter: click_yn > 0")

// ❌ repartition 제거: AQE가 자동 최적화
// ❌ count() 제거: 드라이버 액션 불필요
val trainSampleGap = transformedTrainDF
    .filter("click_yn>0")
    .stat.sampleBy(
        F.col("hour_gap"),
        Map(
            0.0 -> 1.0,                  // Negative 전체 사용
            1.0 -> posSampleRatioGap     // Positive 샘플링
        ),
        42L
    )

println("Training samples prepared")

println("Training Gap prediction model...")
val startTimeGap = System.currentTimeMillis()

val pipelineModelGap = pipelineMLGap.fit(trainSampleGap)

val elapsedGap = (System.currentTimeMillis() - startTimeGap) / 1000
println(s"Gap model training completed in ${elapsedGap}s")

println("=" * 80)


// ===== Paragraph 7: Response Utility Regression Model Training =====

println("=" * 80)
println("Training Response Utility Regression Model")
println("=" * 80)

// 학습에 사용할 모델 선택
val modelRegforCV = xgbr  // 기본값: XGBoost Regressor

val pipelineMLReg = new Pipeline().setStages(Array(modelRegforCV))

println(s"Preparing training samples for Regression model...")
println(s"  - Click filter: click_yn > 0")

// ✅ persist 필요: fit()에서 여러 번 읽음 (2회)
// ❌ repartition 제거: AQE가 자동 최적화
// ❌ count() 제거: 드라이버 액션 불필요
val trainSampleReg = transformedTrainDF
    .filter("click_yn>0")
    .persist(StorageLevel.MEMORY_AND_DISK_SER)

println("Training samples prepared")

println("Training Regression model...")
val startTimeReg = System.currentTimeMillis()

val pipelineModelReg = pipelineMLReg.fit(trainSampleReg)

val elapsedReg = (System.currentTimeMillis() - startTimeReg) / 1000
println(s"Regression model training completed in ${elapsedReg}s")

trainSampleReg.unpersist()

println("=" * 80)


// ===== Paragraph 8: Save Trained Models =====

println("=" * 80)
println("Saving Trained Models")
println("=" * 80)

// 모델 저장 경로
val modelClickPath = s"aos/sto/pipelineModelClick_${modelClickforCV.uid}_v${modelVersion}_${trainSendYmList.head}-${trainSendYmList.last}"
val modelGapPath = s"aos/sto/pipelineModelGap_${modelGapforCV.uid}_v${modelVersion}_${trainSendYmList.head}-${trainSendYmList.last}"
// val modelRegPath = s"aos/sto/pipelineModelReg_${modelRegforCV.uid}_v${modelVersion}_${trainSendYmList.head}-${trainSendYmList.last}"

println(s"Saving Click model to: $modelClickPath")
pipelineModelClick.write.overwrite().save(modelClickPath)
println("Click model saved successfully")

println(s"Saving Gap model to: $modelGapPath")
pipelineModelGap.write.overwrite().save(modelGapPath)
println("Gap model saved successfully")

// println(s"Saving Regression model to: $modelRegPath")
// pipelineModelReg.write.overwrite().save(modelRegPath)
// println("Regression model saved successfully")

println("=" * 80)
println("All models saved successfully")
println("=" * 80)


// ===== Paragraph 9: Model Prediction on Test Dataset =====

// 테스트 데이터 중복 제거 후 예측 - 파티션 최적화
val testDataForPred = transformedTestDF
    .filter("cmpgn_typ=='Sales'")
    .dropDuplicates("svc_mgmt_num", "chnl_typ", "cmpgn_typ", "send_ym", "send_hournum_cd", "click_yn")
    .repartition(400)
    .persist(StorageLevel.MEMORY_AND_DISK_SER)

println("Test data for prediction prepared and cached")

println("Generating Click predictions...")
val predictionsClickDev = pipelineModelClick.transform(testDataForPred)
    .persist(StorageLevel.MEMORY_AND_DISK_SER)

println("Click predictions cached")

println("Generating Gap predictions...")
val predictionsGapDev = pipelineModelGap.transform(testDataForPred)
    .persist(StorageLevel.MEMORY_AND_DISK_SER)

println("Gap predictions cached")


// ===== Paragraph 10: Click Model Performance Evaluation (Precision@K per Hour & MAP) =====

val stagesClick = pipelineModelClick.stages

stagesClick.foreach { stage => 

    val evalModelName = stage.uid  // 학습에 사용된 모델의 UID 자동 추출
    val evalModelShortName = evalModelName.replace("_click", "").toUpperCase
        
    println("\n" + "=" * 80)
    println(s"실제 서비스 평가: Precision@K per Hour & MAP")
    println("-" * 80)
    println(s"평가 모델: $evalModelShortName (UID: $evalModelName)")
    
    // 모델별 하이퍼파라미터 출력
    evalModelName match {
        case "gbtc_click" =>
            println(s"  - maxIter: ${gbtc.getMaxIter}")
            println(s"  - maxDepth: ${gbtc.getMaxDepth}")
            println(s"  - featureSubsetStrategy: ${gbtc.getFeatureSubsetStrategy}")
        case "xgbc_click" =>
            println(s"  - numRound: ${xgbc.getNumRound}")
            println(s"  - maxDepth: ${xgbc.getMaxDepth}")
            println(s"  - objective: ${xgbc.getObjective}")
        case "fmc_click" =>
            println(s"  - stepSize: ${fmc.getStepSize}")
        case "lgbmc_click" =>
            println(s"  - numIterations: ${lgbmc.getNumIterations}")
            println(s"  - learningRate: ${lgbmc.getLearningRate}")
        case _ =>
            println(s"  - Model: $evalModelName")
    }
    println("=" * 80 + "\n")
    
    // ========================================
    // Part 1: Precision@K per Hour (시간대별 평가)
    // ========================================
    
    println("=" * 80)
    println("Part 1: Precision@K per Hour (시간대별 상위 K명 선택 시 클릭률)")
    println("=" * 80)
    
    // 시간대별 사용자 확률 생성
    // suffix 기반 샘플링 (shuffle 없이 빠르게 처리)
    val suffixRange = (0 to 8).map(_.toHexString)  // 0~9: 약 62.5% 샘플링, 0~f: 100%
    
    val hourlyUserPredictions = predictionsClickDev
        .filter("click_yn >= 0")
        .withColumn("suffix", F.substring(F.col("svc_mgmt_num"), -1, 1))  // 마지막 자리 추출
        .where(s"""suffix in ('${suffixRange.mkString("', '")}')""")  // suffix 필터링 (shuffle 없음!)
        .repartition(400)  // 파티션 증가로 메모리 분산
        .select(
            F.col("svc_mgmt_num"),
            F.col("send_ym"),
            F.col("send_hournum_cd").cast("int").alias("hour"),
            F.col(indexedLabelColClick).alias("actual_click"),
            F.expr(s"vector_to_array(prob_$evalModelName)[1]").alias("click_prob")
        )
        .groupBy("svc_mgmt_num", "send_ym", "hour")
        .agg(
            F.max("click_prob").alias("click_prob"),
            F.max("actual_click").alias("actual_click")
        )
        .repartition(200)  // 집계 후 파티션 조정
        .persist(StorageLevel.MEMORY_AND_DISK_SER)  // cache → persist로 변경
    
    // K 값들
    val kValues = Array(100, 500, 1000, 2000, 5000, 10000)
    
    println("\n시간대별 Precision@K:")
    println("-" * 100)
    println(f"Hour | ${"K=100"}%8s | ${"K=500"}%8s | ${"K=1000"}%9s | ${"K=2000"}%9s | ${"K=5000"}%9s | ${"K=10000"}%10s |")
    println("-" * 100)
    
    // 각 시간대별로 계산
    val hours = (9 to 18).toArray
    
    hours.foreach { hour =>
        val hourData = hourlyUserPredictions.filter(s"hour = $hour")
        
        val precisions = kValues.map { k =>
            // 확률 상위 K명 선택
            val topK = hourData
                .orderBy(F.desc("click_prob"))
                .limit(k)
            
            val totalK = topK.count().toDouble
            val clickedK = topK.filter("actual_click > 0").count().toDouble
            
            if (totalK > 0) clickedK / totalK else 0.0
        }
        
        println(f"$hour%4d | ${precisions(0) * 100}%7.2f%% | ${precisions(1) * 100}%7.2f%% | ${precisions(2) * 100}%8.2f%% | ${precisions(3) * 100}%8.2f%% | ${precisions(4) * 100}%8.2f%% | ${precisions(5) * 100}%9.2f%% |")
    }
    
    println("-" * 100)
    
    // 전체 평균 (시간대별 평균)
    println("\n전체 평균 Precision@K (시간대별 평균):")
    kValues.foreach { k =>
        val avgPrecision = hours.map { hour =>
            val hourData = hourlyUserPredictions.filter(s"hour = $hour")
            val topK = hourData.orderBy(F.desc("click_prob")).limit(k)
            val totalK = topK.count().toDouble
            val clickedK = topK.filter("actual_click > 0").count().toDouble
            if (totalK > 0) clickedK / totalK else 0.0
        }.sum / hours.length
        
        println(f"  Precision@$k%5d: $avgPrecision%.4f (${avgPrecision * 100}%.2f%%)")
    }
    
    // ========================================
    // Part 1.5: Recall@K per Hour (시간대별 커버리지)
    // ========================================
    
    println("\n" + "=" * 80)
    println("Part 1.5: Recall@K per Hour (시간대별 상위 K명이 전체 클릭자 중 차지하는 비율)")
    println("=" * 80)
    
    println("\n시간대별 Recall@K:")
    println("-" * 100)
    println(f"Hour | ${"K=100"}%8s | ${"K=500"}%8s | ${"K=1000"}%9s | ${"K=2000"}%9s | ${"K=5000"}%9s | ${"K=10000"}%10s |")
    println("-" * 100)
    
    hours.foreach { hour =>
        val hourData = hourlyUserPredictions.filter(s"hour = $hour")
        val totalClicked = hourData.filter("actual_click > 0").count().toDouble
        
        val recalls = kValues.map { k =>
            // 확률 상위 K명 선택
            val topK = hourData
                .orderBy(F.desc("click_prob"))
                .limit(k)
            
            val clickedK = topK.filter("actual_click > 0").count().toDouble
            
            if (totalClicked > 0) clickedK / totalClicked else 0.0
        }
        
        println(f"$hour%4d | ${recalls(0) * 100}%7.2f%% | ${recalls(1) * 100}%7.2f%% | ${recalls(2) * 100}%8.2f%% | ${recalls(3) * 100}%8.2f%% | ${recalls(4) * 100}%8.2f%% | ${recalls(5) * 100}%9.2f%% |")
    }
    
    println("-" * 100)
    
    // 전체 평균 Recall
    println("\n전체 평균 Recall@K (시간대별 평균):")
    kValues.foreach { k =>
        val avgRecall = hours.map { hour =>
            val hourData = hourlyUserPredictions.filter(s"hour = $hour")
            val totalClicked = hourData.filter("actual_click > 0").count().toDouble
            val topK = hourData.orderBy(F.desc("click_prob")).limit(k)
            val clickedK = topK.filter("actual_click > 0").count().toDouble
            if (totalClicked > 0) clickedK / totalClicked else 0.0
        }.sum / hours.length
        
        println(f"  Recall@$k%5d: $avgRecall%.4f (${avgRecall * 100}%.2f%%)")
    }
    
    // Precision-Recall 트레이드오프 분석
    println("\n" + "=" * 80)
    println("Precision-Recall 트레이드오프 (시간대별 평균)")
    println("=" * 80)
    println(f"${"K"}%8s | ${"Precision"}%10s | ${"Recall"}%8s | ${"F1-Score"}%10s | ${"클릭자/발송"}%12s |")
    println("-" * 80)
    
    kValues.foreach { k =>
        val (avgPrec, avgRec, avgClicked) = hours.map { hour =>
            val hourData = hourlyUserPredictions.filter(s"hour = $hour")
            val totalClicked = hourData.filter("actual_click > 0").count().toDouble
            val topK = hourData.orderBy(F.desc("click_prob")).limit(k)
            val totalK = topK.count().toDouble
            val clickedK = topK.filter("actual_click > 0").count().toDouble
            
            val prec = if (totalK > 0) clickedK / totalK else 0.0
            val rec = if (totalClicked > 0) clickedK / totalClicked else 0.0
            
            (prec, rec, clickedK)
        }.reduce((a, b) => (a._1 + b._1, a._2 + b._2, a._3 + b._3))
        
        val finalPrec = avgPrec / hours.length
        val finalRec = avgRec / hours.length
        val f1 = if (finalPrec + finalRec > 0) 2 * (finalPrec * finalRec) / (finalPrec + finalRec) else 0.0
        val avgClickedPerHour = avgClicked / hours.length
        
        println(f"$k%8d | ${finalPrec * 100}%9.2f%% | ${finalRec * 100}%7.2f%% | $f1%10.4f | ${avgClickedPerHour}%12.1f |")
    }
    
    println("-" * 80)
    println("\n💡 해석:")
    println("- K 증가 → Precision 감소, Recall 증가 (정상)")
    println("- F1-Score 최대 지점 = 최적 K 값")
    println("- 비즈니스 목표에 따라 K 선택:")
    println("  • 효율 우선 (높은 Precision): 작은 K")
    println("  • 커버리지 우선 (높은 Recall): 큰 K")
    
    // ========================================
    // Part 2: MAP (Mean Average Precision) - IR 표준 방식
    // ========================================
    
    println("\n" + "=" * 80)
    println("Part 2: MAP (정보검색 표준 방식) - 시간대별 AP → 평균")
    println("=" * 80)
    
    println("\n각 시간대(질의어)별 Average Precision:")
    println("-" * 80)
    println(f"${"Hour"}%5s | ${"클릭자 수"}%10s | ${"AP"}%8s | ${"설명"}%40s")
    println("-" * 80)
    
    // 각 시간대별로 AP 계산
    val hourlyAPs = hours.map { hour =>
        val hourData = hourlyUserPredictions
            .filter(s"hour = $hour")
            .orderBy(F.desc("click_prob"))  // 확률 순 정렬
            .withColumn("rank", F.row_number().over(Window.orderBy(F.desc("click_prob"))).cast("long"))
            .cache()
        
        val totalClicked = hourData.filter("actual_click > 0").count()
        
        if (totalClicked > 0) {
            // 클릭한 사용자들의 순위
            val clickedRanks = hourData
                .filter("actual_click > 0")
                .select("rank")
                .collect()
                .map { row =>
                    // 타입 안전하게 처리
                    val rank = row.get(0) match {
                        case i: Int => i.toLong
                        case l: Long => l
                        case _ => row.getLong(0)
                    }
                    rank.toDouble
                }
            
            // AP 계산: sum(precision@rank) / total_relevant
            val ap = clickedRanks.zipWithIndex.map { case (rank, idx) =>
                (idx + 1).toDouble / rank  // precision@rank = (idx+1) / rank
            }.sum / totalClicked
            
            hourData.unpersist()
            (hour, totalClicked, ap)
        } else {
            hourData.unpersist()
            (hour, 0L, 0.0)
        }
    }
    
    // 시간대별 AP 출력
    hourlyAPs.foreach { case (hour, clicked, ap) =>
        val desc = if (clicked > 0) {
            f"상위 순위에 클릭자 배치 품질"
        } else {
            "클릭 없음"
        }
        println(f"$hour%5d | $clicked%10d | $ap%8.4f | $desc%-40s")
    }
    
    println("-" * 80)
    
    // MAP 계산 (클릭이 있는 시간대만)
    val validAPs = hourlyAPs.filter(_._2 > 0)
    val map = if (validAPs.nonEmpty) {
        validAPs.map(_._3).sum / validAPs.length
    } else {
        0.0
    }
    
    println(f"\n★ MAP (Mean Average Precision): $map%.4f")
    println(f"   평가 대상 시간대: ${validAPs.length}/10")
    println(f"   해석: 각 시간대별로 클릭자를 얼마나 상위에 랭킹했는지 평균")
    println(f"   기준: MAP > 0.3 (양호), > 0.5 (우수), > 0.7 (매우 우수)")
    
    // ========================================
    // Part 2.5: 사용자별 MAP (보조 지표 - 비표준)
    // ========================================
    
    println("\n" + "=" * 80)
    println("Part 2.5: 사용자별 MAP (보조 지표 - 사용자 관점)")
    println("=" * 80)
    
    // 각 사용자별 시간대별 확률 및 클릭 여부
    val userAPData = predictionsClickDev
        .filter("click_yn >= 0")
        .select(
            F.col("svc_mgmt_num"),
            F.col("send_ym"),
            F.col("send_hournum_cd").cast("int").alias("hour"),
            F.col(indexedLabelColClick).alias("actual_click"),
            F.expr("vector_to_array(prob_gbtc_click)[1]").alias("click_prob")
        )
        .groupBy("svc_mgmt_num", "send_ym")
        .agg(
            F.collect_list(
                F.struct(
                    F.col("hour"),
                    F.col("click_prob"),
                    F.col("actual_click")
                )
            ).alias("hourly_data")
        )
        .withColumn("hourly_data_sorted", 
            F.expr("array_sort(hourly_data, (left, right) -> case when left.click_prob > right.click_prob then -1 when left.click_prob < right.click_prob then 1 else 0 end)")
        )
        .withColumn("clicked_hours_count",
            F.expr("size(filter(hourly_data, x -> x.actual_click > 0))")
        )
        .filter("clicked_hours_count > 0")  // 실제 클릭이 있는 사용자만
        .withColumn("ap", 
            F.expr("""
                aggregate(
                    sequence(0, size(hourly_data_sorted) - 1),
                    cast(0.0 as double),
                    (acc, i) -> case 
                        when element_at(hourly_data_sorted, i + 1).actual_click > 0 
                        then acc + (
                            size(filter(slice(hourly_data_sorted, 1, i + 1), x -> x.actual_click > 0)) 
                            / cast(i + 1 as double)
                        )
                        else acc
                    end
                ) / clicked_hours_count
            """)
        )
        .cache()
    
    val userMAP = userAPData
        .agg(F.avg("ap"))
        .first()
        .getDouble(0)
    
    val totalUsersMAP = userAPData.count()
    
    println(f"\n사용자별 MAP: $userMAP%.4f")
    println(f"  → 평가 대상 사용자 수: $totalUsersMAP")
    println(f"  → 의미: 각 사용자별로 클릭 시간대를 얼마나 상위에 예측했는지")
    println(f"  → 주의: IR 표준 MAP과 다름! 보조 지표로만 활용")
    
    // AP 분포
    println("\n사용자별 AP 분포:")
    userAPData
        .selectExpr("floor(ap * 10) / 10 as ap_bucket")
        .groupBy("ap_bucket")
        .count()
        .orderBy("ap_bucket")
        .withColumn("percentage", F.expr(s"count * 100.0 / $totalUsersMAP"))
        .show(10, false)
    
    // ========================================
    // Part 3: 보조 지표 (참고용)
    // ========================================
    
    println("\n" + "=" * 80)
    println("Part 3: 보조 지표 (사용자별 Top-K Accuracy - 참고용)")
    println("=" * 80)
    
    val userMetrics = userAPData
        .withColumn("top1_hour", F.expr("hourly_data_sorted[0].hour"))
        .withColumn("actual_click_hours", 
            F.expr("transform(filter(hourly_data, x -> x.actual_click > 0), x -> x.hour)")
        )
        .withColumn("top1_match", 
            F.expr("array_contains(actual_click_hours, top1_hour)")
        )
        .withColumn("top3_hours",
            F.expr("transform(slice(hourly_data_sorted, 1, 3), x -> x.hour)")
        )
        .withColumn("top3_match",
            F.expr("size(array_intersect(top3_hours, actual_click_hours)) > 0")
        )
        .cache()
    
    val top1Acc = userMetrics.filter("top1_match").count().toDouble / totalUsersMAP
    val top3Acc = userMetrics.filter("top3_match").count().toDouble / totalUsersMAP
    
    println(f"Top-1 Accuracy: $top1Acc%.4f (${top1Acc * 100}%.2f%%)")
    println(f"  → 랜덤 대비: ${top1Acc / 0.1}%.2f배")
    println(f"Top-3 Accuracy: $top3Acc%.4f (${top3Acc * 100}%.2f%%)")
    
    println("\n" + "=" * 80)
    println("💡 종합 해석 가이드 (정보검색 관점)")
    println("=" * 80)
    println("\n★ 정보검색 시스템 매핑:")
    println("  질의어(Query)  → 시간대 (9시, 10시, ..., 18시)")
    println("  문서(Document) → 사용자")
    println("  관련성         → 해당 시간대 클릭 여부")
    println("  검색 결과      → 확률 순 사용자 리스트")
    
    println("\n1. Precision@K per Hour (IR 표준 ✓):")
    println("   - 의미: 질의어(시간대) q에 대해 상위 K개 문서(사용자) 중 관련 문서 비율")
    println("   - 활용: 각 시간대별 발송 인원(K) 결정")
    println("   - 전략: 높은 Precision 시간대에 더 많은 예산")
    
    println("\n2. Recall@K per Hour (IR 표준 ✓):")
    println("   - 의미: 질의어(시간대) q에 대해 전체 관련 문서 중 상위 K개에 포함된 비율")
    println("   - 활용: K에 따른 커버리지 파악")
    println("   - 전략: 목표 Recall 달성을 위한 최소 K 결정")
    
    println("\n3. Precision-Recall 트레이드오프:")
    println("   - F1-Score 최대 지점 = 효율과 커버리지 균형점")
    println("   - 비즈니스 목표에 따라 K 선택")
    
    println("\n4. MAP - 시간대별 (IR 표준 ✓):")
    println("   - 의미: 각 질의어(시간대)별 AP를 평균")
    println("   - AP = 각 관련 문서(클릭자)의 순위에서 precision 평균")
    println("   - 활용: 전체 모델 품질 평가, 모델 간 비교")
    println("   - 기준: MAP > 0.3 (양호), > 0.5 (우수), > 0.7 (매우 우수)")
    
    println("\n5. 사용자별 MAP (비표준, 보조):")
    println("   - 의미: 각 사용자별로 클릭 시간대를 얼마나 상위에 예측했는지")
    println("   - 주의: IR 표준과 다름! 관점이 반대 (사용자 중심)")
    println("   - 활용: Top-K Accuracy와 유사, 보조 지표로만 사용")
    
    println("\n6. Top-K Accuracy (보조):")
    println("   - 사용자 관점 평가")
    println("   - 랜덤 대비 성능 확인")
    
    println("\n★ 표준 IR 평가 vs 우리 평가:")
    println("  ✓ Precision@K per Hour: 완벽히 일치")
    println("  ✓ Recall@K per Hour: 완벽히 일치")
    println("  ✓ MAP (시간대별): 완벽히 일치")
    println("  ⚠ 사용자별 MAP: 비표준 (보조용)")
    
    println("\n실전 활용 예시:")
    println("  목표: 시간당 1000명 발송, 최소 Recall 20%")
    println("  → Recall@1000 >= 20%인 시간대 선택")
    println("  → 해당 시간대의 Precision@1000 확인 (예상 클릭률)")
    println("  → MAP으로 전체 모델 품질 추적")
    
    // ========================================
    // 로그 저장용 데이터 사전 수집 (unpersist 전에 모든 계산 완료)
    // ========================================
    println("\n로그 저장을 위한 데이터 수집 중...")
    
    // Precision@K와 Recall@K를 한 번에 계산하여 로컬로 수집
    val precisionRecallResults = hours.map { hour =>
        val hourData = hourlyUserPredictions.filter(s"hour = $hour")
        val totalClicked = hourData.filter("actual_click > 0").count().toDouble
        
        val metricsPerK = kValues.map { k =>
            val topK = hourData.orderBy(F.desc("click_prob")).limit(k)
            val totalK = topK.count().toDouble
            val clickedK = topK.filter("actual_click > 0").count().toDouble
            
            val precision = if (totalK > 0) clickedK / totalK else 0.0
            val recall = if (totalClicked > 0) clickedK / totalClicked else 0.0
            
            (precision, recall)
        }
        
        (hour, metricsPerK)
    }
    
    // MAP 및 User Metrics를 로컬 변수로 저장
    val mapValue = map
    val validAPsLength = validAPs.length
    val userMAPValue = userMAP
    val totalUsersMAPValue = totalUsersMAP
    
    println("데이터 수집 완료. 메모리 해제 중...")
    
    // 이제 안전하게 unpersist
    hourlyUserPredictions.unpersist()
    userAPData.unpersist()
    userMetrics.unpersist()
    
    println("메모리 해제 완료. 로그 저장 시작...")
    
    // ========================================
    // 평가 결과 로그 저장 (수집된 로컬 데이터 사용)
    // ========================================
    import java.io.{File, PrintWriter}
    import java.time.LocalDateTime
    import java.time.format.DateTimeFormatter
    
    val logDir = new File("/data/myfiles/aos_ost/predict")
    if (!logDir.exists()) logDir.mkdirs()
    
    val timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"))
    val logFile = new File(logDir, s"click_model_eval_${evalModelName}_${timestamp}.log")
    val writer = new PrintWriter(logFile)
    
    try {
        writer.println("=" * 80)
        writer.println(s"Click Model Evaluation Log - $timestamp")
        writer.println("=" * 80)
        writer.println()
        
        // 모델 정보
        writer.println("[Model Information]")
        writer.println(s"Model: $evalModelShortName (UID: $evalModelName)")
        evalModelName match {
            case "gbtc_click" =>
                writer.println(s"  - maxIter: ${gbtc.getMaxIter}")
                writer.println(s"  - maxDepth: ${gbtc.getMaxDepth}")
                writer.println(s"  - featureSubsetStrategy: ${gbtc.getFeatureSubsetStrategy}")
            case "xgbc_click" =>
                writer.println(s"  - numRound: ${xgbc.getNumRound}")
                writer.println(s"  - maxDepth: ${xgbc.getMaxDepth}")
                writer.println(s"  - objective: ${xgbc.getObjective}")
            case "fmc_click" =>
                writer.println(s"  - stepSize: ${fmc.getStepSize}")
            case "lgbmc_click" =>
                writer.println(s"  - numIterations: ${lgbmc.getNumIterations}")
                writer.println(s"  - learningRate: ${lgbmc.getLearningRate}")
            case _ =>
                writer.println(s"  - Model: $evalModelName")
        }
        writer.println()
        
        // 학습 데이터 정보
        writer.println("[Training Data Information]")
        writer.println(s"Negative Sample Ratio: $negSampleRatioClick")
        writer.println(s"Positive Sample Ratio: 1.0 (전체 사용)")
        writer.println(s"Sample Strategy: stat.sampleBy")
        writer.println(s"Estimated neg:pos ratio: ${(negSampleRatioClick * 100).toInt}:100")
        writer.println()
        
        // Precision@K 결과 저장 (수집된 데이터 사용)
        writer.println("[Precision@K per Hour]")
        writer.println(f"${"Hour"}%5s | ${"K=100"}%7s | ${"K=500"}%7s | ${"K=1000"}%8s | ${"K=2000"}%8s | ${"K=5000"}%8s | ${"K=10000"}%9s")
        writer.println("-" * 75)
        
        precisionRecallResults.foreach { case (hour, metricsPerK) =>
            val precisions = metricsPerK.map(_._1)
            writer.println(f"$hour%5d | ${precisions(0)*100}%6.2f%% | ${precisions(1)*100}%6.2f%% | ${precisions(2)*100}%7.2f%% | ${precisions(3)*100}%7.2f%% | ${precisions(4)*100}%7.2f%% | ${precisions(5)*100}%8.2f%%")
        }
        
        // 평균 Precision@K
        writer.println("-" * 75)
        val avgPrecisions = kValues.indices.map { idx =>
            precisionRecallResults.map(_._2(idx)._1).sum / precisionRecallResults.length
        }
        writer.print(f"${"Avg"}%5s")
        avgPrecisions.foreach { avg =>
            writer.print(f" | ${avg*100}%6.2f%%")
        }
        writer.println()
        writer.println()
        
        // Recall@K 결과 저장 (수집된 데이터 사용)
        writer.println("[Recall@K per Hour]")
        writer.println(f"${"Hour"}%5s | ${"K=100"}%7s | ${"K=500"}%7s | ${"K=1000"}%8s | ${"K=2000"}%8s | ${"K=5000"}%8s | ${"K=10000"}%9s")
        writer.println("-" * 75)
        
        precisionRecallResults.foreach { case (hour, metricsPerK) =>
            val recalls = metricsPerK.map(_._2)
            writer.println(f"$hour%5d | ${recalls(0)*100}%6.2f%% | ${recalls(1)*100}%6.2f%% | ${recalls(2)*100}%7.2f%% | ${recalls(3)*100}%7.2f%% | ${recalls(4)*100}%7.2f%% | ${recalls(5)*100}%8.2f%%")
        }
        
        // 평균 Recall@K
        writer.println("-" * 75)
        val avgRecalls = kValues.indices.map { idx =>
            precisionRecallResults.map(_._2(idx)._2).sum / precisionRecallResults.length
        }
        writer.print(f"${"Avg"}%5s")
        avgRecalls.foreach { avg =>
            writer.print(f" | ${avg*100}%6.2f%%")
        }
        writer.println()
        writer.println()
        
        // MAP 결과 저장 (수집된 데이터 사용)
        writer.println("[MAP - Mean Average Precision]")
        writer.println(f"MAP (시간대별): $mapValue%.4f")
        writer.println(f"Evaluated Hours: $validAPsLength/10")
        writer.println(f"User-based MAP: $userMAPValue%.4f")
        writer.println(f"Evaluated Users: $totalUsersMAPValue")
        writer.println()
        
        // 해석 가이드
        writer.println("[Interpretation Guide]")
        writer.println("- Precision@K: 상위 K명 발송 시 클릭률")
        writer.println("- Recall@K: 전체 클릭자 중 상위 K명에 포함된 비율")
        writer.println("- MAP > 0.3: 양호, > 0.5: 우수, > 0.7: 매우 우수")
        writer.println()
        
        writer.println("=" * 80)
        writer.println("Log saved successfully!")
        
        println(s"\n✅ Click Model 평가 결과 로그 저장: ${logFile.getAbsolutePath}")
        
    } finally {
        writer.close()
    }
}


// ===== Paragraph 11: Gap Model Performance Evaluation (Precision, Recall, F1) =====

val stagesGap = pipelineModelGap.stages

stagesGap.foreach { stage => 
    
    val modelName = stage.uid
    
    println(s"Evaluating model: $modelName")
    
    // 집계 및 평가용 데이터 준비 - 파티션 최적화
    val aggregatedPredictionsGap = predictionsGapDev
        .filter("click_yn>0")
        .withColumn("prob", F.expr(s"vector_to_array(prob_$modelName)[1]"))
        .repartition(100, F.col("svc_mgmt_num"))  // GroupBy 전 파티셔닝
        .groupBy("svc_mgmt_num", "send_ym", "send_hournum_cd")
        .agg(F.sum(indexedLabelColGap).alias(indexedLabelColGap), F.max("prob").alias("prob"))
        .withColumn(indexedLabelColGap, F.expr(s"case when $indexedLabelColGap>0 then cast(1.0 AS DOUBLE) else cast(0.0 AS DOUBLE) end"))
        .withColumn("prediction_gap", F.expr("case when prob>=0.5 then cast(1.0 AS DOUBLE) else cast(0.0 AS DOUBLE) end"))
        .sample(false, 0.3, 42)  // ← 30% 샘플링 추가
        .repartition(50)  // 샘플링 후 파티션 재조정
        .persist(StorageLevel.MEMORY_AND_DISK_SER)
    
    // ========================================
    // 완전 분산 평가 (Driver 수집 없음, 매우 빠름!)
    // ========================================
    
    println(s"######### $modelName 예측 결과 #########")
    
    // 1. DataFrame 연산으로 Confusion Matrix 계산 (완전 분산)
    val confusionDF = aggregatedPredictionsGap
        .groupBy(indexedLabelColGap, "prediction_gap")
        .count()
        .cache()
    
    // 2. TP, FP, TN, FN 계산 (분산 연산)
    val tp = confusionDF.filter(s"$indexedLabelColGap = 1.0 AND prediction_gap = 1.0").select("count").first().getLong(0).toDouble
    val fp = confusionDF.filter(s"$indexedLabelColGap = 0.0 AND prediction_gap = 1.0").select("count").first().getLong(0).toDouble
    val tn = confusionDF.filter(s"$indexedLabelColGap = 0.0 AND prediction_gap = 0.0").select("count").first().getLong(0).toDouble
    val fn = confusionDF.filter(s"$indexedLabelColGap = 1.0 AND prediction_gap = 0.0").select("count").first().getLong(0).toDouble
    
    // 3. 지표 계산 (Driver에서 간단한 계산만)
    val precision_1 = if (tp + fp > 0) tp / (tp + fp) else 0.0
    val recall_1 = if (tp + fn > 0) tp / (tp + fn) else 0.0
    val f1_1 = if (precision_1 + recall_1 > 0) 2 * (precision_1 * recall_1) / (precision_1 + recall_1) else 0.0
    
    val precision_0 = if (tn + fn > 0) tn / (tn + fn) else 0.0
    val recall_0 = if (tn + fp > 0) tn / (tn + fp) else 0.0
    val f1_0 = if (precision_0 + recall_0 > 0) 2 * (precision_0 * recall_0) / (precision_0 + recall_0) else 0.0
    
    val accuracy = (tp + tn) / (tp + tn + fp + fn)
    val total = tp + tn + fp + fn
    
    val weightedPrecision = (precision_1 * (tp + fn) + precision_0 * (tn + fp)) / total
    val weightedRecall = (recall_1 * (tp + fn) + recall_0 * (tn + fp)) / total
    
    // 4. BinaryClassificationEvaluator로 AUC 계산 (분산)
    val binaryEvaluator = new BinaryClassificationEvaluator()
        .setLabelCol(indexedLabelColGap)
        .setRawPredictionCol("prob")
        .setMetricName("areaUnderROC")
    val auc = binaryEvaluator.evaluate(aggregatedPredictionsGap)
    
    // 5. 결과 출력
    println("--- 레이블별 성능 지표 ---")
    println(f"Label 0.0 (클래스): Precision = $precision_0%.4f, Recall = $recall_0%.4f, F1 = $f1_0%.4f")
    println(f"Label 1.0 (클래스): Precision = $precision_1%.4f, Recall = $recall_1%.4f, F1 = $f1_1%.4f")
    
    println(f"\nWeighted Precision (전체 평균): $weightedPrecision%.4f")
    println(f"Weighted Recall (전체 평균): $weightedRecall%.4f")
    println(f"Accuracy (전체 정확도): $accuracy%.4f")
    println(f"AUC (Area Under ROC): $auc%.4f")
    
    println("\n--- Confusion Matrix (혼동 행렬) ---")
    println(f"              Predicted 0    Predicted 1")
    println(f"Actual 0:     ${tn}%.0f         ${fp}%.0f")
    println(f"Actual 1:     ${fn}%.0f         ${tp}%.0f")
    
    // ========================================
    // 로그 저장용 데이터 사전 수집 (unpersist 전에 모든 값을 로컬로)
    // ========================================
    
    // 성능 지표들을 로컬 변수로 저장
    val precision_0_local = precision_0
    val recall_0_local = recall_0
    val f1_0_local = f1_0
    val precision_1_local = precision_1
    val recall_1_local = recall_1
    val f1_1_local = f1_1
    val weightedPrecision_local = weightedPrecision
    val weightedRecall_local = weightedRecall
    val accuracy_local = accuracy
    val auc_local = auc
    val tp_local = tp
    val fp_local = fp
    val tn_local = tn
    val fn_local = fn
    val total_local = total
    
    // 메모리 해제
    confusionDF.unpersist()
    aggregatedPredictionsGap.unpersist()
    
    // ========================================
    // Gap Model 평가 결과 로그 저장 (수집된 로컬 데이터 사용)
    // ========================================
    import java.io.{File, PrintWriter}
    import java.time.LocalDateTime
    import java.time.format.DateTimeFormatter
    
    val logDir = new File("/data/myfiles/aos_ost/predict")
    if (!logDir.exists()) logDir.mkdirs()
    
    val timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"))
    val logFile = new File(logDir, s"gap_model_eval_${modelName}_${timestamp}.log")
    val writer = new PrintWriter(logFile)
    
    try {
        writer.println("=" * 80)
        writer.println(s"Gap Model Evaluation Log - $timestamp")
        writer.println("=" * 80)
        writer.println()
        
        // 모델 정보
        writer.println("[Model Information]")
        writer.println(s"Model: ${modelName.replace("_gap", "").toUpperCase} (UID: $modelName)")
        modelName match {
            case "gbtc_gap" =>
                writer.println(s"  - maxIter: ${gbtg.getMaxIter}")
                writer.println(s"  - maxDepth: ${gbtg.getMaxDepth}")
                writer.println(s"  - featureSubsetStrategy: ${gbtg.getFeatureSubsetStrategy}")
            case "xgbc_gap" =>
                writer.println(s"  - numRound: ${xgbg.getNumRound}")
                writer.println(s"  - maxDepth: ${xgbg.getMaxDepth}")
                writer.println(s"  - objective: ${xgbg.getObjective}")
            case _ =>
                writer.println(s"  - Model: $modelName")
        }
        writer.println()
        
        // 학습 데이터 정보
        writer.println("[Training Data Information]")
        writer.println(s"Data Filter: click_yn > 0 (클릭 발생 케이스만)")
        writer.println(s"Positive Sample Ratio (hour_gap > 0): $posSampleRatioGap")
        writer.println(s"Negative Sample Ratio (hour_gap = 0): 1.0")
        writer.println(s"Sample Strategy: stat.sampleBy")
        writer.println()
        
        // 성능 지표 (로컬 변수 사용)
        writer.println("[Performance Metrics]")
        writer.println(f"Precision (Label 0): $precision_0_local%.4f")
        writer.println(f"Recall (Label 0): $recall_0_local%.4f")
        writer.println(f"F1-Score (Label 0): $f1_0_local%.4f")
        writer.println()
        writer.println(f"Precision (Label 1): $precision_1_local%.4f")
        writer.println(f"Recall (Label 1): $recall_1_local%.4f")
        writer.println(f"F1-Score (Label 1): $f1_1_local%.4f")
        writer.println()
        writer.println(f"Weighted Precision: $weightedPrecision_local%.4f")
        writer.println(f"Weighted Recall: $weightedRecall_local%.4f")
        writer.println(f"Accuracy: $accuracy_local%.4f")
        writer.println(f"AUC: $auc_local%.4f")
        writer.println()
        
        // Confusion Matrix (로컬 변수 사용)
        writer.println("[Confusion Matrix]")
        writer.println(f"              Predicted 0    Predicted 1")
        writer.println(f"Actual 0:     ${tn_local}%.0f         ${fp_local}%.0f")
        writer.println(f"Actual 1:     ${fn_local}%.0f         ${tp_local}%.0f")
        writer.println()
        
        // 해석
        writer.println("[Interpretation]")
        writer.println("- Label 0: 즉시 클릭 (hour_gap = 0)")
        writer.println("- Label 1: 지연 클릭 (hour_gap > 0)")
        writer.println(s"- 총 평가 샘플: ${total_local.toLong}")
        writer.println()
        
        writer.println("=" * 80)
        writer.println("Log saved successfully!")
        
        println(s"\n✅ Gap Model 평가 결과 로그 저장: ${logFile.getAbsolutePath}")
        
    } finally {
        writer.close()
    }
}

