// =============================================================================
// MMS Click Prediction - Service Prediction (Production Version v3)
// =============================================================================
// 이 코드는 predict_ost_2MC68ADVY_260130.scala의 Paragraph들을 재활용하여
// 실시간 서비스 환경에서 Propensity Score를 생성합니다.
//
// [재활용 Paragraph]
// - Paragraph 1: Imports and Configuration
// - Paragraph 2: Helper Functions
// - Paragraph 6: Response Data Loading
// - Paragraph 8: MMKT Data Loading
// - Paragraph 14: Click Count Feature
// - Paragraph 24, 30: Model Loading
// - Paragraph 37: Propensity Score Calculation
//
// [실행 방법]
// 1. Response data 준비 (aos/sto/response)
// 2. 이 스크립트 실행
//
// [데이터 흐름]
// Response data → clickCountDF
// MMKT data → mmktDF
// XDR data (Suffix별 로딩)
// → Feature 조인 → Transformer → 모델 예측 → Propensity Score
// =============================================================================


// =============================================================================
// Paragraph 1: Imports and Configuration
// =============================================================================
// 재활용: predict_ost_2MC68ADVY_260130.scala Line 11-50

import ml.dmlc.xgboost4j.scala.spark.{XGBoostClassificationModel, XGBoostClassifier, XGBoostRegressor}
import org.apache.spark.ml.classification._
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.{Pipeline, PipelineModel, linalg}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.types.{DateType, StringType, TimestampType}
import org.apache.spark.sql.{DataFrame, SparkSession, functions => F}
import org.apache.spark.storage.StorageLevel

import java.sql.Date
import java.text.DecimalFormat
import java.time.format.DateTimeFormatter
import java.time.{ZoneId, ZonedDateTime}

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

// Checkpoint 설정 (환경에 맞게 조정)
spark.sparkContext.setCheckpointDir("hdfs://scluster/user/g1110566/checkpoint")

println(s"Spark configuration set for service prediction")
println(s"  Optimal partitions: $optimalPartitions")


// =============================================================================
// Paragraph 2: Helper Functions
// =============================================================================
// 재활용: predict_ost_2MC68ADVY_260130.scala Line 53-128

import java.time.YearMonth
import java.time.format.DateTimeFormatter
import java.time.LocalDate
import scala.collection.mutable.ListBuffer
import java.time.temporal.ChronoUnit
import org.apache.spark.ml.linalg.Vector

spark.udf.register("vector_to_array", (v: Vector) => v.toArray)

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

def getPreviousDays(startDayStr: String, periodD: Int): Array[String] = {
  val formatter = DateTimeFormatter.ofPattern("yyyyMMdd")
  val startDay = LocalDate.parse(startDayStr, formatter)
  val resultDays = ListBuffer[String]()
  var currentDay = startDay

  for (i <- 0 until periodD) {
    resultDays.prepend(currentDay.format(formatter))
    currentDay = currentDay.minusDays(1)
  }
  resultDays.toArray
}

def getDaysBetween(startDayStr: String, endDayStr: String): Array[String] = {
  val formatter = DateTimeFormatter.ofPattern("yyyyMMdd")
  val start = LocalDate.parse(startDayStr, formatter)
  val end = LocalDate.parse(endDayStr, formatter)
  val numOfDays = ChronoUnit.DAYS.between(start, end).toInt
  val resultDays = ListBuffer[String]()

  for (i <- 0 to numOfDays) {
    resultDays.append(start.plusDays(i).format(formatter))
  }
  resultDays.toArray
}


// =============================================================================
// Paragraph 3: Configuration Variables
// =============================================================================

println("=" * 80)
println("Service Prediction Configuration")
println("=" * 80)

// 예측 날짜 설정
val predDT = "20260101"                             // 예측 날짜 (yyyyMMdd)
val predSendYM = predDT.take(6)                     // 예측 월 (yyyyMM)
val predFeatureYM = getPreviousMonths(predSendYM, 2)(0)  // Feature 기간 (2개월 전)

// 시간대 설정
val startHour = 9
val endHour = 18
val hourRange = (startHour to endHour).toList

// 모델 및 Transformer 버전
val transformedDataVersion = "1"
val modelVersion = "1"

// 모델 UID
val modelClickUID = "gbtc_click"
val modelGapUID = "xgbc_gap"

// 학습 데이터 기간 정보 (모델 경로 생성용)
val trainPeriod = 3
val trainSendMonth = "202511"
val trainSendYmList = getPreviousMonths(trainSendMonth, trainPeriod)

// 경로 설정
val responseDataPath = "aos/sto/response"
val transformedTrainDataPath = s"aos/sto/transformedTrainDF_v${transformedDataVersion}_${trainSendYmList.head}-${trainSendYmList.last}"
val transformerClickPath = s"aos/sto/transformPipelineClick_v${transformedDataVersion}_${trainSendYmList.head}-${trainSendYmList.last}"
val transformerGapPath = s"aos/sto/transformPipelineGap_v${transformedDataVersion}_${trainSendYmList.head}-${trainSendYmList.last}"
val modelClickPath = s"aos/sto/pipelineModelClick_${modelClickUID}_v${modelVersion}_${trainSendYmList.head}-${trainSendYmList.last}"
val modelGapPath = s"aos/sto/pipelineModelGap_${modelGapUID}_v${modelVersion}_${trainSendYmList.head}-${trainSendYmList.last}"

// 출력 경로
val outputPath = "aos/sto/mms_score"

// Feature column 이름
val categoryColNameList = Array("_cd", "_yn", "_rank", "_type", "_typ")
val numericColNameList = Array("_cnt","_amt","_arpu","_mb","_qty","_age","_score","_price","_ratio","_duration","_avg","_distance","_entropy")

println(s"Prediction Configuration:")
println(s"  - Prediction Date: $predDT")
println(s"  - Prediction Month: $predSendYM")
println(s"  - Feature Month: $predFeatureYM")
println(s"  - Hour Range: ${hourRange.mkString(", ")}")
println(s"  - Model Version: $modelVersion")
println(s"  - Transformer Version: $transformedDataVersion")
println()
println(s"Paths:")
println(s"  - Response Data: $responseDataPath")
println(s"  - Transformed Train Data: $transformedTrainDataPath")
println(s"  - Click Model: $modelClickPath")
println(s"  - Gap Model: $modelGapPath")
println(s"  - Output: $outputPath")
println("=" * 80)


// =============================================================================
// Paragraph 4: Response Data Loading
// =============================================================================
// 재활용: predict_ost_2MC68ADVY_260130.scala Paragraph 6

println("=" * 80)
println("Loading Response Data")
println("=" * 80)

val resDF = spark.read.parquet(responseDataPath)
  .persist(StorageLevel.MEMORY_AND_DISK_SER)

println(s"Response data loaded from: $responseDataPath")
println("=" * 80)


// =============================================================================
// Paragraph 5: User Feature Data Loading (MMKT)
// =============================================================================
// 재활용: predict_ost_2MC68ADVY_260130.scala Paragraph 8 (수정)

println("=" * 80)
println("Loading User Feature Data (MMKT)")
println("=" * 80)

val allFeaturesMMKT = spark.sql("describe wind_tmt.mmkt_svc_bas_f")
    .select("col_name")
    .collect()
    .map(_.getString(0))

val sigFeaturesMMKT = spark.read
    .option("header", "true")
    .csv("feature_importance/table=mmkt_bas/creation_dt=20230407")
    .filter("rank<=100")
    .select("col")
    .collect()
    .map(_(0).toString())
    .map(_.trim)

val colListForMMKT = (Array(
    "svc_mgmt_num", "strd_ym feature_ym", "mst_work_dt", "cust_birth_dt", 
    "prcpln_last_chg_dt", "fee_prod_id", "eqp_mdl_cd", "eqp_acqr_dt", 
    "equip_chg_cnt", "svc_scrb_dt", "chg_dt", "cust_age_cd", "sex_cd", 
    "equip_chg_day", "last_equip_chg_dt", "prev_equip_chg_dt", "rten_pen_amt", 
    "agrmt_brch_amt", "eqp_mfact_cd", "allot_mth_cnt", "mbr_use_cnt", 
    "mbr_use_amt", "tyr_mbr_use_cnt", "tyr_mbr_use_amt", "mth_cnsl_cnt", 
    "dsat_cnsl_cnt", "simpl_ref_cnsl_cnt", "arpu", "bf_m1_arpu", "voc_arpu", 
    "bf_m3_avg_arpu", "tfmly_nh39_scrb_yn", "prcpln_chg_cnt", "email_inv_yn", 
    "copn_use_psbl_cnt", "data_gift_send_yn", "data_gift_recv_yn", 
    "equip_chg_mth_cnt", "dom_tot_pckt_cnt", "scrb_sale_chnl_cl_cd", 
    "op_sale_chnl_cl_cd", "agrmt_dc_end_dt", "svc_cd", "svc_st_cd", "pps_yn", 
    "svc_use_typ_cd", "indv_corp_cl_cd", "frgnr_yn", "nm_cust_num", "wlf_dc_cd"
) ++ sigFeaturesMMKT)
    .filter(c => allFeaturesMMKT.contains(c.trim.split(" ")(0).trim))
    .distinct

val mmktDFTemp = spark.sql(s"""
    SELECT ${colListForMMKT.mkString(",")}, strd_ym 
    FROM wind_tmt.mmkt_svc_bas_f 
    WHERE strd_ym = '$predFeatureYM'
""")
// .repartition() 제거: AQE가 자동 최적화 (checkpoint 전 불필요)

val prodDF = spark.sql("""
    SELECT prod_id fee_prod_id, prod_nm fee_prod_nm 
    FROM wind.td_zprd_prod
""")

val mmktDF = mmktDFTemp
    .join(F.broadcast(prodDF), Seq("fee_prod_id"), "left")
    .filter("cust_birth_dt not like '9999%'")
    .checkpoint()

println(s"MMKT data loaded: Feature month = $predFeatureYM")
println("=" * 80)


// =============================================================================
// Paragraph 6: Historical Click Count Feature Engineering
// =============================================================================
// 재활용: predict_ost_2MC68ADVY_260130.scala Paragraph 14

println("=" * 80)
println("Creating Click Count Feature")
println("=" * 80)

import org.apache.spark.sql.functions._

val n = 3  // 이전 3개월

// Response data에서 feature_ym 계산
val df = resDF
    .withColumn("feature_ym",
        F.date_format(F.add_months(F.unix_timestamp($"send_dt", "yyyyMMdd")
            .cast(TimestampType), -1), "yyyyMM").cast(StringType))
// .repartition() 제거: AQE가 자동 최적화 (Self-join이 자동 셔플)

// Self-join으로 과거 클릭 횟수 계산
val clickCountDF = df.as("current")
    .join(
        df.as("previous").hint("shuffle_replicate_nl"),
        col("current.svc_mgmt_num") === col("previous.svc_mgmt_num") &&
        col("current.send_hournum") === col("previous.send_hournum") &&
        col("previous.feature_ym") < col("current.feature_ym"),
        "left"
    )
    .where(
        months_between(
            to_date(col("current.feature_ym"), "yyyyMM"),
            to_date(col("previous.feature_ym"), "yyyyMM")
        ) <= n &&
        months_between(
            to_date(col("current.feature_ym"), "yyyyMM"),
            to_date(col("previous.feature_ym"), "yyyyMM")
        ) >= 0
    )
    // .repartition() 제거: AQE가 GroupBy 전 자동 최적화
    .groupBy("current.svc_mgmt_num", "current.feature_ym","current.send_hournum")
    .agg(
        sum(coalesce(col("previous.click_yn"), lit(0.0))).alias("click_cnt")
    )
    .select(
        col("svc_mgmt_num"),
        col("feature_ym"),
        col("send_hournum").alias("send_hournum_cd"),
        col("click_cnt")
    )
    .filter(s"feature_ym = '$predFeatureYM'")
    .persist(StorageLevel.MEMORY_AND_DISK_SER)

println(s"Click count feature created: Feature month = $predFeatureYM")
println("=" * 80)


// =============================================================================
// Paragraph 7: Load Models and Transformers
// =============================================================================
// 재활용: predict_ost_2MC68ADVY_260130.scala Paragraph 24, 30

println("=" * 80)
println("Loading Models and Transformers")
println("=" * 80)

println(s"Loading Click transformer from: $transformerClickPath")
val transformerClick = PipelineModel.load(transformerClickPath)
println("Click transformer loaded successfully")

println(s"Loading Gap transformer from: $transformerGapPath")
val transformerGap = PipelineModel.load(transformerGapPath)
println("Gap transformer loaded successfully")

println(s"Loading Click model from: $modelClickPath")
val pipelineModelClick = PipelineModel.load(modelClickPath)
println("Click model loaded successfully")

println(s"Loading Gap model from: $modelGapPath")
val pipelineModelGap = PipelineModel.load(modelGapPath)
println("Gap model loaded successfully")

println("=" * 80)
println("All models and transformers loaded successfully")
println("=" * 80)


// =============================================================================
// Paragraph 7.5: Load Transformed Data Schema (Feature Column Detection)
// =============================================================================

println("=" * 80)
println("Loading Transformed Data Schema for Feature Column Detection")
println("=" * 80)

// Transformed Train Data의 스키마만 읽기 (데이터 로딩 없이 메타데이터만)
val transformedSchemaDF = spark.read.parquet(transformedTrainDataPath).limit(0)
val transformedColumns = transformedSchemaDF.columns

println(s"Schema loaded from: $transformedTrainDataPath")
println(s"Total columns: ${transformedColumns.length}")

// Feature 컬럼 자동 감지 (스키마 기반)
val noFeatureCols = Array("click_yn", "hour_gap", "chnl_typ", "cmpgn_typ")
val tokenCols = transformedColumns.filter(x => x.endsWith("_token")).distinct
val continuousCols = (transformedColumns
    .filter(x => numericColNameList.map(x.endsWith(_)).reduceOption(_ || _).getOrElse(false))
    .distinct
    .filter(x => !tokenCols.contains(x) && !noFeatureCols.contains(x))
).distinct
val categoryCols = (transformedColumns
    .filter(x => categoryColNameList.map(x.endsWith(_)).reduceOption(_ || _).getOrElse(false))
    .distinct
    .filter(x => !tokenCols.contains(x) && !noFeatureCols.contains(x) && !continuousCols.contains(x))
).distinct
val vectorCols = transformedColumns.filter(x => x.endsWith("_vec")).distinct

println(s"Feature columns detected:")
println(s"  - Token columns: ${tokenCols.length}")
println(s"  - Continuous columns: ${continuousCols.length}")
println(s"  - Category columns: ${categoryCols.length}")
println(s"  - Vector columns: ${vectorCols.length}")
println("=" * 80)


// =============================================================================
// Paragraph 8: Propensity Score Calculation (Batch by Suffix)
// =============================================================================
// 재활용: predict_ost_2MC68ADVY_260130.scala Paragraph 37

println("=" * 80)
println("Starting Propensity Score Calculation (Suffix-based Batch Processing)")
println("=" * 80)

val pivotColumns = hourRange.toList.map(h => f"$h, total_traffic_$h%02d").mkString(", ")

// Suffix별 배치 처리
(0 to 15).map(_.toHexString).foreach { suffix =>

    println(s"\n[Suffix: $suffix] Starting batch processing...")
    val batchStartTime = System.currentTimeMillis()

    val prdSuffixCond = suffix.map(c => s"svc_mgmt_num like '%${c}'").mkString(" or ")

    // ========================================
    // Step 1: XDR 데이터 로딩 (앱 사용 패턴)
    // ========================================
    
    val xdrDFPred = spark.sql(s"""
    SELECT
        svc_mgmt_num,
        ym AS feature_ym,
        COALESCE(rep_app_title, app_uid) AS app_nm,
        CAST(hour.send_hournum_cd AS STRING) AS send_hournum_cd,
        hour.traffic
    FROM dprobe.mst_app_svc_app_monthly
    LATERAL VIEW STACK(
        ${hourRange.size},
        ${pivotColumns}
    ) hour AS send_hournum_cd, traffic
    WHERE hour.traffic > 1000
        AND ym = '$predFeatureYM'
        AND ($prdSuffixCond)
    """)
    // .repartition() 제거: AQE가 GroupBy 전 자동 최적화
    .groupBy("svc_mgmt_num", "feature_ym", "send_hournum_cd")
    .agg(
        F.collect_set("app_nm").alias("app_usage_token"),
        F.count(F.when(F.col("traffic") > 100000, 1)).alias("heavy_usage_app_cnt"),
        F.count(F.when(F.col("traffic").between(10000, 100000), 1)).alias("medium_usage_app_cnt"),
        F.count(F.when(F.col("traffic") < 10000, 1)).alias("light_usage_app_cnt"),
        F.sum("traffic").alias("total_traffic_mb"),
        F.count("*").alias("app_cnt")
    )

    println(s"[Suffix: $suffix] XDR hourly data loaded")

    // ========================================
    // Step 2: XDR 집계 피처 생성
    // ========================================
    
    val xdrPredAggregatedFeatures = xdrDFPred
        .groupBy("svc_mgmt_num", "feature_ym")
        .agg(
            F.max(F.struct(F.col("app_cnt"), F.col("send_hournum_cd"))).alias("peak_hour_struct"),
            F.count(F.when(F.col("app_cnt") > 0, 1)).alias("active_hour_cnt"),
            F.avg("app_cnt").alias("avg_hourly_app_avg"),
            F.sum("total_traffic_mb").alias("total_daily_traffic_mb"),
            F.sum("heavy_usage_app_cnt").alias("total_heavy_apps_cnt"),
            F.sum("medium_usage_app_cnt").alias("total_medium_apps_cnt"),
            F.sum("light_usage_app_cnt").alias("total_light_apps_cnt")
        )
        .withColumn("peak_usage_hour_cd", F.col("peak_hour_struct.send_hournum_cd"))
        .withColumn("peak_hour_app_cnt", F.col("peak_hour_struct.app_cnt"))
        .drop("peak_hour_struct")

    println(s"[Suffix: $suffix] XDR aggregated features created")

    // ========================================
    // Step 3: XDR Pivot (시간대별 앱 사용)
    // ========================================
    
    val xdrPredDF = xdrDFPred
        // .repartition() 제거: AQE가 Pivot 전 자동 최적화
        .groupBy("svc_mgmt_num", "feature_ym")
        .pivot("send_hournum_cd", hourRange.map(_.toString))
        .agg(F.first("app_usage_token"))
        .select(
            F.col("svc_mgmt_num") +: F.col("feature_ym") +:
            hourRange.map(h =>
                F.coalesce(F.col(s"$h"), F.array(F.lit("#"))).alias(s"app_usage_${h}_token")
            ): _*
        )
        .join(xdrPredAggregatedFeatures, Seq("svc_mgmt_num", "feature_ym"), "left")

    println(s"[Suffix: $suffix] XDR monthly pivot data created")

    // ========================================
    // Step 4: 데이터 조인 (작은 테이블부터)
    // ========================================
    
    val predDF = mmktDF
        .filter(prdSuffixCond)
        .filter(s"strd_ym = '$predFeatureYM'")
        .withColumn("feature_ym", F.col("strd_ym"))
        .withColumn("send_ym", F.lit(predSendYM))
        .withColumn("send_dt", F.lit(predDT))
        .withColumn("cmpgn_num", F.lit("#"))
        .withColumn("cmpgn_typ", F.lit("#"))
        .withColumn("chnl_typ", F.lit("#"))
        .withColumn("click_yn", F.lit(0))
        .withColumn("hour_gap", F.lit(0))
        .withColumn("res_utility", F.lit(0.0))
        .withColumn("send_hournum_cd", F.explode(F.expr(s"array(${hourRange.mkString(",")})")))
        // .repartition() 제거: AQE가 Join 전 자동 최적화
        .join(clickCountDF, Seq("svc_mgmt_num", "feature_ym", "send_hournum_cd"), "left")
        .na.fill(Map("click_cnt" -> 0.0))
        .join(xdrDFPred, Seq("svc_mgmt_num", "feature_ym", "send_hournum_cd"), "left")
        // .repartition() 제거: AQE가 Join 전 자동 최적화
        .join(xdrPredDF, Seq("svc_mgmt_num", "feature_ym"), "left")

    println(s"[Suffix: $suffix] Data joined")

    // ========================================
    // Step 5: Feature 선택 및 정리
    // ========================================
    
    val predDFRev = predDF.select(
        (Array("cmpgn_num", "svc_mgmt_num", "chnl_typ", "cmpgn_typ",
            "send_ym", "send_dt", "feature_ym", "hour_gap",
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
    // Step 6: Transformer 적용
    // ========================================
    
    val transformedPredDF = transformerGap.transform(transformerClick.transform(predDFRev))

    println(s"[Suffix: $suffix] Transformations applied")

    // ========================================
    // Step 7: 모델 예측 (Click + Gap)
    // ========================================
    
    val predictionsSVCClick = pipelineModelClick.transform(transformedPredDF)
    val predictionsSVCFinal = pipelineModelGap.transform(predictionsSVCClick)

    println(s"[Suffix: $suffix] Model predictions generated")

    // ========================================
    // Step 8: Propensity Score 계산 및 저장
    // ========================================
    
    val predictedPropensityScoreDF = predictionsSVCFinal
        .withColumn("prob_click", 
            F.expr(s"""aggregate(array(${pipelineModelClick.stages.map(m => s"vector_to_array(prob_${m.uid})[1]").mkString(",")}), 0D, (acc, x) -> acc + x)""")
        )
        .withColumn("prob_gap", 
            F.expr(s"""aggregate(array(${pipelineModelGap.stages.map(m => s"vector_to_array(prob_${m.uid})[1]").mkString(",")}), 0D, (acc, x) -> acc + x)""")
        )
        .withColumn("propensity_score", F.expr("prob_click * prob_gap"))

    predictedPropensityScoreDF
        .selectExpr(
            "default.decodekey(svc_mgmt_num, svc_mgmt_num) svc_mgmt_num",
            "send_ym",
            "send_hournum_cd AS send_hour",
            "ROUND(prob_click, 4) AS prob_click",
            "ROUND(prob_gap, 4) AS prob_gap",
            "ROUND(propensity_score, 4) AS propensity_score"
        )
        .withColumn("suffix", F.lit(suffix))
        .coalesce(optimalPartitions)
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


// =============================================================================
// Summary and Cleanup
// =============================================================================

println("\n" + "=" * 80)
println("Service Prediction Summary")
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
println()
println("Output:")
println(s"  - Path: $outputPath")
println(s"  - Format: Parquet (Snappy)")
println(s"  - Partitions: send_ym, send_hour, suffix")
println()
println("=" * 80)
println("Service prediction completed successfully!")
println("=" * 80)

// 메모리 정리
resDF.unpersist()
mmktDF.unpersist()
clickCountDF.unpersist()

println("\nMemory cleaned up")


// =============================================================================
// End of Service Prediction (Production v3)
// =============================================================================
