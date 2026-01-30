// =============================================================================
// MMS Click Prediction - Service Prediction (Production Version)
// =============================================================================
// 이 코드는 실시간 서비스 환경에서 Propensity Score를 생성합니다:
// 1. 저장된 rawDF 로딩 (aos/sto/rawDF{version})
// 2. 예측 날짜 조건 필터링 (predDT, predSendYM)
// 3. Suffix 기반 배치 처리 (메모리 최적화)
// 4. 앙상블 모델 지원 (여러 모델의 평균)
// 5. Propensity Score 계산 (prob_click × prob_gap)
// 6. 실제 서비스 저장 포맷
//
// [실행 방법]
// 1. raw_data_generation.scala로 rawDF 생성
// 2. data_transformation.scala로 transformer 학습
// 3. model_training.scala로 모델 학습
// 4. 이 스크립트로 서비스 Propensity Score 생성
//
// [조건 설정]
// - predDT: 예측 날짜 (yyyyMMdd) - 예: "20260101"
// - predSendYM: 예측 월 (자동 계산) - 예: "202601"
// - predFeatureYM: Feature 기간 (자동 계산, 2개월 전) - 예: "202511"
// =============================================================================


// ===== Paragraph 1: Imports and Configuration =====

import ml.dmlc.xgboost4j.scala.spark.{XGBoostClassificationModel, XGBoostClassifier, XGBoostRegressor}
import org.apache.spark.ml.classification._
import org.apache.spark.ml.regression._
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.{Pipeline, PipelineModel, linalg}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.types.{DateType, StringType, TimestampType}
import org.apache.spark.sql.{DataFrame, SparkSession, functions => F}
import org.apache.spark.storage.StorageLevel

import java.sql.Date
import java.text.DecimalFormat
import java.time.format.DateTimeFormatter
import java.time.{ZoneId, ZonedDateTime, LocalDate, YearMonth}

// Spark 설정
spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
spark.conf.set("spark.sql.adaptive.advisoryPartitionSizeInBytes", "128MB")
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "50m")

spark.conf.set("spark.executor.memoryOverhead", "4g")
spark.conf.set("spark.memory.fraction", "0.8")
spark.conf.set("spark.memory.storageFraction", "0.3")

val executorInstances = spark.sparkContext.getConf.getInt("spark.executor.instances", 10)
val executorCores = spark.sparkContext.getConf.getInt("spark.executor.cores", 5)
val optimalPartitions = executorInstances * executorCores
spark.conf.set("spark.sql.shuffle.partitions", (optimalPartitions * 4).toString)

println(s"Spark configuration set for service prediction (Production)")
println(s"  Optimal partitions: $optimalPartitions")


// ===== Paragraph 2: Configuration Variables =====

// =============================================================================
// 서비스 예측 설정
// =============================================================================

// 모델 및 Transformer 버전
val transformedDataVersion = "1"
val modelVersion = "1"

// Raw data 버전
val rawDataVersion = "1"

// 예측 날짜 설정 (조건)
val predDT = "20260101"  // 예측 날짜 (yyyyMMdd)
val predSendYM = predDT.take(6)  // 예측 월 (yyyyMM)

// Helper functions
def getPreviousMonths(startMonthStr: String, periodM: Int): Array[String] = {
  val formatter = DateTimeFormatter.ofPattern("yyyyMM")
  val startMonth = YearMonth.parse(startMonthStr, formatter)
  var resultMonths = scala.collection.mutable.ListBuffer[String]()
  var currentMonth = startMonth

  for (i <- 0 until periodM) {
    resultMonths.prepend(currentMonth.format(formatter))
    currentMonth = currentMonth.minusMonths(1)
  }
  resultMonths.toArray
}

// Feature 기간 계산 (예측 날짜 기준 1개월 전)
val predFeatureYM = getPreviousMonths(predSendYM, 1)(0)

// Raw data 경로
val rawDataPath = s"aos/sto/rawDF${rawDataVersion}"

// 시간대 설정
val startHour = 9
val endHour = 18
val hourRange = (startHour to endHour).toList

// 모델이 학습된 기간 정보
val trainPeriod = 3
val trainSendMonth = "202511"
val trainSendYmList = getPreviousMonths(trainSendMonth, trainPeriod)

// 경로 설정
val transformerClickPath = s"aos/sto/transformPipelineClick_v${transformedDataVersion}_${trainSendYmList.head}-${trainSendYmList.last}"
val transformerGapPath = s"aos/sto/transformPipelineGap_v${transformedDataVersion}_${trainSendYmList.head}-${trainSendYmList.last}"
val modelClickPath = s"aos/sto/pipelineModelClick_v${modelVersion}_${trainSendYmList.head}-${trainSendYmList.last}"
val modelGapPath = s"aos/sto/pipelineModelGap_v${modelVersion}_${trainSendYmList.head}-${trainSendYmList.last}"

// 출력 경로
val outputPath = "aos/sto/mms_score"

// Feature column 이름
val categoryColNameList = Array("_cd", "_yn", "_rank", "_type", "_typ")
val numericColNameList = Array("_cnt","_amt","_arpu","_mb","_qty","_age","_score","_price","_ratio","_duration","_avg","_distance","_entropy")

// UDF 등록
spark.udf.register("vector_to_array", (v: Vector) => v.toArray)

println("=" * 80)
println("Service Prediction Configuration (Production)")
println("=" * 80)
println(s"Prediction Configuration:")
println(s"  - Prediction Date: $predDT")
println(s"  - Prediction Month: $predSendYM")
println(s"  - Feature Month: $predFeatureYM (1 months before)")
println(s"  - Model Version: $modelVersion")
println(s"  - Transformer Version: $transformedDataVersion")
println(s"  - Raw Data Version: $rawDataVersion")
println()
println(s"Data Paths:")
println(s"  - Raw Data: $rawDataPath")
println(s"  - Click Model: $modelClickPath")
println(s"  - Gap Model: $modelGapPath")
println(s"  - Click Transformer: $transformerClickPath")
println(s"  - Gap Transformer: $transformerGapPath")
println()
println(s"Output:")
println(s"  - Path: $outputPath")
println(s"  - Partition: send_ym, send_hour, suffix")
println()
println(s"Processing:")
println(s"  - Suffix batching: 0-f (16 batches)")
println(s"  - Hour range: ${hourRange.mkString(", ")}")
println("=" * 80)


// ===== Paragraph 3: Load Models and Transformers =====

println("=" * 80)
println("Loading Models and Transformers")
println("=" * 80)

// 1. Transformer 로딩
println(s"Loading Click transformer from: $transformerClickPath")
val transformerClick = PipelineModel.load(transformerClickPath)
println("Click transformer loaded successfully")

println(s"Loading Gap transformer from: $transformerGapPath")
val transformerGap = PipelineModel.load(transformerGapPath)
println("Gap transformer loaded successfully")

// 2. 모델 로딩
println(s"Loading Click model from: $modelClickPath")
val pipelineModelClick = PipelineModel.load(modelClickPath)
println("Click model loaded successfully")

println(s"Loading Gap model from: $modelGapPath")
val pipelineModelGap = PipelineModel.load(modelGapPath)
println("Gap model loaded successfully")

println("=" * 80)
println("All models and transformers loaded successfully")
println("=" * 80)


// ===== Paragraph 4: Load Raw Data =====

println("=" * 80)
println("Loading Raw Data")
println("=" * 80)

println(s"Loading raw data from: $rawDataPath")
val rawDF = spark.read.parquet(rawDataPath)

// 예측 날짜 조건으로 필터링 (predDT, predSendYM)
println(s"Filtering raw data for prediction:")
println(s"  - Prediction Date: $predDT")
println(s"  - Prediction Month: $predSendYM")
println(s"  - Feature Month: $predFeatureYM")

// rawDF 구조 파악을 위한 컬럼 정보
val noFeatureCols = Array("click_yn", "hour_gap", "chnl_typ", "cmpgn_typ")
val baseColumnsFromRaw = Array("cmpgn_num", "svc_mgmt_num", "chnl_typ", "cmpgn_typ", 
                                "send_ym", "send_dt", "feature_ym", "click_yn", "res_utility")

val tokenCols = rawDF.columns.filter(x => x.endsWith("_token")).distinct
val continuousCols = (rawDF.columns
    .filter(x => numericColNameList.map(x.endsWith(_)).reduceOption(_ || _).getOrElse(false))
    .distinct
    .filter(x => !tokenCols.contains(x) && !noFeatureCols.contains(x))
).distinct
val categoryCols = (rawDF.columns
    .filter(x => categoryColNameList.map(x.endsWith(_)).reduceOption(_ || _).getOrElse(false))
    .distinct
    .filter(x => !tokenCols.contains(x) && !noFeatureCols.contains(x) && !continuousCols.contains(x))
).distinct
val vectorCols = rawDF.columns.filter(x => x.endsWith("_vec")).distinct

// 서비스 예측용 데이터 준비
// 조건: send_dt = predDT, send_ym = predSendYM 또는 feature_ym = predFeatureYM
val serviceDFBase = rawDF
    .filter(F.col("feature_ym") === predFeatureYM)  // Feature 기간 조건
    .withColumn("send_ym", F.lit(predSendYM))       // 예측 월 설정
    .withColumn("send_dt", F.lit(predDT))           // 예측 날짜 설정
    .withColumn("cmpgn_num", F.lit("#"))            // 서비스용 더미값
    .withColumn("cmpgn_typ", F.lit("#"))
    .withColumn("chnl_typ", F.lit("#"))
    .withColumn("click_yn", F.lit(0))
    .withColumn("hour_gap", F.lit(0))
    .withColumn("res_utility", F.lit(0.0))
    .persist(StorageLevel.MEMORY_AND_DISK_SER)

println(s"  - Feature month: $predFeatureYM")
println(s"  - Prediction date: $predDT")

println("=" * 80)


// ===== Paragraph 5: Propensity Score Calculation (Suffix Batch Processing) =====

println("=" * 80)
println("Starting Propensity Score Calculation (Suffix-based Batch Processing)")
println("=" * 80)

// Suffix 기반 배치 처리 (0-f, 총 16개)
(0 to 15).map(_.toHexString).foreach { suffix =>

    println(s"\n[Suffix: $suffix] Starting batch processing...")
    val batchStartTime = System.currentTimeMillis()

    val prdSuffixCond = suffix.map(c => s"svc_mgmt_num like '%${c}'").mkString(" or ")

    // ========================================
    // Step 1: Suffix별 base 데이터 필터링
    // ========================================
    
    val suffixBaseDF = serviceDFBase
        .filter(prdSuffixCond)

    println(s"[Suffix: $suffix] Base data filtered")

    // ========================================
    // Step 2: 시간대별 explode (예측용 데이터 생성)
    // ========================================
    
    // rawDF에는 이미 모든 feature가 포함되어 있으므로
    // send_hournum_cd를 explode하여 각 시간대별 예측 데이터 생성
    val predDF = suffixBaseDF
        .withColumn("send_hournum_cd", F.explode(F.expr(s"array(${hourRange.mkString(",")})")))

    println(s"[Suffix: $suffix] Prediction DataFrame created with ${hourRange.size} hours")

    // ========================================
    // Step 3: Feature 선택 및 정리
    // ========================================
    
    // rawDF에는 이미 feature engineering이 완료되어 있으므로
    // 필요한 컬럼만 선택하고 정리
    val predDFRev = predDF.select(
        (Array("cmpgn_num", "svc_mgmt_num", "chnl_typ", "cmpgn_typ",
            "send_ym", "send_dt", "feature_ym", "send_hournum_cd", "hour_gap",
            "click_yn", "res_utility").map(F.col(_))
        ++ tokenCols.map(cl => F.coalesce(F.col(cl), F.array(F.lit("#"))).alias(cl))
        ++ vectorCols.map(cl => F.col(cl).alias(cl))
        ++ categoryCols.map(cl =>
            F.when(F.col(cl) === "", F.lit("UKV"))
                .otherwise(F.coalesce(F.col(cl).cast("string"), F.lit("UKV")))
                .alias(cl)
        )
        ++ continuousCols.map(cl => F.coalesce(F.col(cl).cast("float"), F.lit(Double.NaN)).alias(cl))
        ): _*
    )

    println(s"[Suffix: $suffix] Features prepared")

    // ========================================
    // Step 4: Transformer 적용
    // ========================================
    
    val transformedPredDF = transformerGap.transform(transformerClick.transform(predDFRev))

    println(s"[Suffix: $suffix] Transformations applied")

    // ========================================
    // Step 5: 모델 예측 (Click + Gap)
    // ========================================
    
    val predictionsSVCClick = pipelineModelClick.transform(transformedPredDF)
    val predictionsSVCFinal = pipelineModelGap.transform(predictionsSVCClick)

    println(s"[Suffix: $suffix] Model predictions generated")

    // ========================================
    // Step 6: Propensity Score 계산 (앙상블)
    // ========================================
    
    // 앙상블: 여러 모델의 확률 평균
    val predictedPropensityScoreDF = predictionsSVCFinal
        .withColumn("prob_click", 
            F.expr(s"""aggregate(array(${pipelineModelClick.stages.map(m => s"vector_to_array(prob_${m.uid})[1]").mkString(",")}), 0D, (acc, x) -> acc + x)""")
        )
        .withColumn("prob_gap", 
            F.expr(s"""aggregate(array(${pipelineModelGap.stages.map(m => s"vector_to_array(prob_${m.uid})[1]").mkString(",")}), 0D, (acc, x) -> acc + x)""")
        )
        .withColumn("propensity_score", F.expr("prob_click * prob_gap"))

    println(s"[Suffix: $suffix] Propensity scores calculated")

    // ========================================
    // Step 7: 결과 저장
    // ========================================
    
    // 실제 서비스 형식으로 저장
    // default.decodekey()는 실제 환경의 암호화 함수
    predictedPropensityScoreDF
        .selectExpr(
            "svc_mgmt_num",  // 실제: default.decodekey(svc_mgmt_num, svc_mgmt_num)
            "send_ym",
            "send_hournum_cd AS send_hour",
            "ROUND(prob_click, 4) AS prob_click",
            "ROUND(prob_gap, 4) AS prob_gap",
            "ROUND(propensity_score, 4) AS propensity_score"
        )
        .withColumn("suffix", F.lit(suffix))
        .coalesce(optimalPartitions)  // 파일 개수 최적화 (작은 파일 문제 방지)
        .write
        .mode("overwrite")
        .partitionBy("send_ym", "send_hour", "suffix")
        .parquet(outputPath)

    val batchElapsed = (System.currentTimeMillis() - batchStartTime) / 1000
    println(s"[Suffix: $suffix] Batch completed in ${batchElapsed}s")
    println("=" * 80)
}

println("\n" + "=" * 80)
println("All Suffix Batches Completed!")
println("=" * 80)


// ===== Summary =====

println("\n" + "=" * 80)
println("Service Prediction Summary (Production)")
println("=" * 80)
println()
println("Prediction Configuration:")
println(s"  - Prediction Date: $predDT")
println(s"  - Prediction Month: $predSendYM")
println(s"  - Feature Month: $predFeatureYM")
println()
println("Processing:")
println(s"  - Suffix batches: 16 (0-f)")
println(s"  - Hours per batch: ${hourRange.size} (${hourRange.head}-${hourRange.last})")
println(s"  - Total predictions: [suffix × hours × users]")
println()
println("Output:")
println(s"  - Path: $outputPath")
println(s"  - Format: Parquet (Snappy)")
println(s"  - Partitions: send_ym={$predSendYM}, send_hour={9-18}, suffix={0-f}")
println()
println("Model Information:")
println(s"  - Model Version: $modelVersion")
println(s"  - Training Period: ${trainSendYmList.head} ~ ${trainSendYmList.last}")
println(s"  - Ensemble: Click + Gap models")
println()
println("Propensity Score:")
println(s"  - Formula: prob_click × prob_gap")
println(s"  - Range: 0.0000 ~ 1.0000")
println()
println("=" * 80)
println("Service prediction completed successfully!")
println("=" * 80)

// 메모리 정리
serviceDFBase.unpersist()

println("\nMemory cleaned up")


// =============================================================================
// End of Service Prediction (Production)
// =============================================================================
// 
// 데이터 흐름:
// 1. rawDF 로딩 (aos/sto/rawDF{version})
// 2. 예측 조건 필터링 (predDT, predSendYM, predFeatureYM)
// 3. Suffix별 배치 처리 (0-f, 16개)
// 4. Transformer 적용 (Click → Gap)
// 5. 모델 예측 (앙상블)
// 6. Propensity Score 계산 및 저장
//
// 서비스 예측 결과 활용:
// 1. Propensity Score: prob_click × prob_gap (사용자별 클릭 성향 점수)
// 2. Partition 구조: send_ym/send_hour/suffix (효율적인 쿼리)
// 3. Suffix 배치: 메모리 효율적인 대용량 처리
//
// 확인 방법:
//   // 전체 결과 확인
//   val scorePath = "aos/sto/mms_score"
//   val scoreDF = spark.read.parquet(scorePath)
//   scoreDF.printSchema()
//   scoreDF.show(10)
//   
//   // 특정 시간대/Suffix 확인
//   val hourlyScore = spark.read.parquet(s"$scorePath/send_ym=202601/send_hour=9/suffix=0")
//   hourlyScore.orderBy(F.desc("propensity_score")).show(10)
//   
//   // 점수 분포
//   scoreDF.select("prob_click", "prob_gap", "propensity_score").describe().show()
//   
//   // Top-K 사용자 선정
//   scoreDF
//     .filter("send_hour = 9")
//     .orderBy(F.desc("propensity_score"))
//     .limit(1000)
//     .show()
//
// 조건 변경 방법:
//   // 다른 날짜로 예측하려면
//   val predDT = "20260201"  // 새로운 예측 날짜
//   // predSendYM과 predFeatureYM은 자동 계산됨
//
// =============================================================================
