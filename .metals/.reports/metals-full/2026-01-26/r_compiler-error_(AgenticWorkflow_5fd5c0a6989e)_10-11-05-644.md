error id: 7B24F233620AC68017BC6B6C3469744D
file://<WORKSPACE>/predict_send_time/raw_data_generation.scala
### dotty.tools.dotc.core.CyclicReference: Cyclic reference involving method wrapString

 Run with -explain-cyclic for more details.

occurred in the presentation compiler.



action parameters:
offset: 6398
uri: file://<WORKSPACE>/predict_send_time/raw_data_generation.scala
text:
```scala
// =============================================================================
// MMS Click Prediction - Raw Data Generation (Unified Version)
// =============================================================================
// 이 코드는 train/test 구분 없이 통합 raw 데이터를 생성합니다.
// Train/Test split 및 undersampling은 transformed data 생성 시 수행됩니다.
//
// [주요 변경사항]
// - Train/Test split 제거 (P7)
// - Undersampling 제거 (P8-P9)
// - 통합 raw 데이터로 저장
// - Suffix별 배치 처리 유지
//
// [실행 방법]
// 1. Paragraph 1-6: 전체 실행 (1회)
// 2. Paragraph 7-12: Suffix별 파라미터와 함께 실행 (suffix: 0-f, 16회)
// =============================================================================


// ===== Paragraph 1: Imports and Configuration =====

import com.microsoft.azure.synapse.ml.causal
import com.skt.mno.dt.utils.commfunc._
import ml.dmlc.xgboost4j.scala.spark.{XGBoostClassificationModel, XGBoostClassifier, XGBoostRegressor}
import org.apache.spark.ml.classification._
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.{Pipeline, PipelineModel, linalg}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.explode
import org.apache.spark.sql.types.{DateType, StringType, TimestampType}
import org.apache.spark.sql.{DataFrame, SparkSession, functions => F}
import org.apache.spark.storage.StorageLevel
import org.joda.time.{LocalDate, Years}
import org.riversun.ml.spark.FeatureImportance.Order
import org.riversun.ml.spark.{FeatureImportance, Importance}

import java.sql.Date
import java.text.DecimalFormat
import java.time.format.DateTimeFormatter
import java.time.{ZoneId, ZonedDateTime}
import com.microsoft.azure.synapse.{ml => sml}

import collection.JavaConverters._
import scala.collection.JavaConverters._

// 대용량 처리 최적화를 위한 Spark 설정
spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
spark.conf.set("spark.sql.shuffle.partitions", "400")
spark.conf.set("spark.sql.files.maxPartitionBytes", "128MB")

// 메모리 최적화 설정 (OOM 방지)
spark.conf.set("spark.executor.memoryOverhead", "4g")
spark.conf.set("spark.memory.fraction", "0.8")
spark.conf.set("spark.memory.storageFraction", "0.3")

spark.sparkContext.setCheckpointDir("hdfs://scluster/user/g1110566/checkpoint")

println("Spark configuration completed")


// ===== Paragraph 2: Helper Functions =====

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

println("Helper functions registered")


// ===== Paragraph 3: Time Condition Variables Configuration =====
// =============================================================================
// 시간 조건 변수 통합 관리
// =============================================================================
// 이 섹션에서 모든 시간 조건을 중앙 관리합니다.
// 
// [통합 Raw Data 생성 설정]
// - Train/Test split 없이 전체 기간 데이터 생성
// - Split 및 sampling은 transformed data 생성 시 수행
// =============================================================================

// -----------------------------------------------------------------------------
// 기본 시간 범위 설정
// -----------------------------------------------------------------------------
// Zeppelin 파라미터로 받기 (기본값: 202512)
val sendMonth = z.input("sendMonth", "202512").toString
val period = 1                            // 데이터 수집 기간 (개월)

// sendMonth 기준으로 이전 period+2개월의 발송 월 리스트 생성
val sendYmList = getPreviousMonths(sendMonth, period)

// featureYmList는 sendYmList의 각 월에서 1개월 전으로 자동 계산
// 예: sendYmList = ["202512", "202511", ...] → featureYmList = ["202511", "202510", ...]
val featureYmList = {
  val formatter = DateTimeFormatter.ofPattern("yyyyMM")
  sendYmList.map { ym =>
    val month = YearMonth.parse(ym, formatter)
    month.minusMonths(1).format(formatter)
  }
}

// 피처 추출을 위한 날짜 범위 (가장 이른 feature month부터 sendMonth까지)
val featureDTList = getDaysBetween(featureYmList(0)+"01", sendMonth+"01")

// -----------------------------------------------------------------------------
// 필터링 조건
// -----------------------------------------------------------------------------
val responseHourGapMax = 5                // 최대 클릭 시간차 (시간 단위)
val startHour = 9                         // 분석 대상 시작 시간
val endHour = 18                          // 분석 대상 종료 시간
val hourRange = (startHour to endHour).toList

// -----------------------------------------------------------------------------
// 저장 설정
// -----------------------------------------------------------------------------
val rawDataVersion = "1"                  // Raw data 버전
val rawDataSavePath = s"aos/sto/rawDF${rawDataVersion}"

// 시간대별 저장 배치 설정
val hourGroupSize = 3                     // 한 번에 처리할 시간대 개수
val hourSlide = 3                         @@// Slide 크기 (groupSize와 같으면 overlap 없음)

// -----------------------------------------------------------------------------
// 설정 요약 출력
// -----------------------------------------------------------------------------
println("=" * 80)
println("Time Condition Variables - Raw Data Generation (Unified)")
println("=" * 80)
println(s"Data Collection Period:")
println(s"  - Send Month (base): $sendMonth")
println(s"  - Period: $period months")
println(s"  - Send YM List: ${sendYmList.mkString(", ")}")
println(s"  - Feature YM List (auto-calculated): ${featureYmList.mkString(", ")}")
println(s"  - Feature extraction range: ${featureDTList(0)} ~ ${featureDTList.last}")
println()
println(s"Filtering Conditions:")
println(s"  - Response Hour Gap Max: $responseHourGapMax hours")
println(s"  - Hour Range: $startHour ~ $endHour")
println()
println(s"Save Configuration:")
println(s"  - Raw Data Version: $rawDataVersion")
println(s"  - Save Path: $rawDataSavePath")
println(s"  - Partition By: send_ym, send_hournum_cd, suffix")
println(s"  - Hour Group Size: $hourGroupSize")
println(s"  - Hour Slide: $hourSlide")
println()
println("NOTE: Feature months are automatically calculated as (send_month - 1)")
println("NOTE: Hour groups are processed using sliding window approach")
println("NOTE: Train/Test split will be performed during transformed data generation")
println("=" * 80)


// ===== Paragraph 4: Response Data Loading from HDFS =====
// =============================================================================
// Campaign 반응 데이터 로딩
// =============================================================================
// 전체 기간의 response data를 로딩합니다.
// Train/Test 구분 없이 모든 데이터를 포함합니다.
// =============================================================================

println("=" * 80)
println("Response Data Loading from HDFS")
println("=" * 80)
println(s"Loading response data...")
println(s"  - Base month: $sendMonth")
println(s"  - Period: $period months")
println(s"  - Target months: ${sendYmList.mkString(", ")}")
println("=" * 80)

// 대용량 데이터는 MEMORY_AND_DISK_SER 사용하여 메모리 절약
val resDF = spark.read.parquet("aos/sto/cmpgn_res").filter(s"send_ym in (${sendYmList.mkString("'","','","'")})")
  .persist(StorageLevel.MEMORY_AND_DISK_SER)

resDF.createOrReplaceTempView("res_df")

println("Response data loaded and cached")
println("=" * 80)


// ===== Paragraph 5: Response Data Filtering and Preparation =====
// =============================================================================
// Response Data 필터링 및 준비
// =============================================================================
// Train/Test split 없이 전체 데이터를 필터링하고 준비합니다.
// - 시간 범위 필터링
// - 중복 제거
// - send_dt 컬럼 유지 (추후 split용)
// =============================================================================

println("=" * 80)
println("Response Data Filtering and Preparation")
println("=" * 80)
println(s"Applying filters:")
println(s"  - Send months: ${sendYmList.mkString(", ")}")
println(s"  - Hour gap: 0 ~ $responseHourGapMax")
println(s"  - Send hour: $startHour ~ $endHour")
println("=" * 80)

// 필터링 및 기본 변환
val resDFFiltered = resDF
    .filter(s"""send_ym in (${sendYmList.mkString("'","','","'")})""")
    .filter(s"hour_gap is null or (hour_gap between 0 and $responseHourGapMax)")
    .filter(s"send_hournum between $startHour and $endHour")
    .selectExpr(
        "cmpgn_num",
        "svc_mgmt_num",
        "chnl_typ",
        "cmpgn_typ",
        "send_ym",
        "send_dt",
        "send_time",
        "send_daynum",
        "send_hournum",
        "click_dt",
        "click_time",
        "click_daynum",
        "click_hournum",
        "case when hour_gap is null then 0 else 1 end as click_yn",
        "hour_gap"
    )
    .withColumn("res_utility", F.expr(s"case when hour_gap is null then 0.0 else 1.0 / (1 + hour_gap) end"))
    .dropDuplicates()
    .repartition(200, F.col("svc_mgmt_num"))
    .persist(StorageLevel.MEMORY_AND_DISK_SER)

println("Filtered response data cached")

// Feature month 추가 및 중복 제거
val resDFSelected = resDFFiltered
    .withColumn("feature_ym", F.date_format(F.add_months(F.unix_timestamp(F.col("send_dt"), "yyyyMMdd").cast(TimestampType), -1), "yyyyMM").cast(StringType))
    .selectExpr(
        "cmpgn_num",
        "svc_mgmt_num",
        "chnl_typ",
        "cmpgn_typ",
        "send_ym",
        "send_dt",
        "feature_ym",
        "send_daynum",
        "send_hournum as send_hournum_cd",
        "hour_gap",
        "click_yn",
        "res_utility"
    )
    .repartition(200, F.col("svc_mgmt_num"), F.col("chnl_typ"), F.col("cmpgn_typ"), F.col("send_ym"), F.col("send_hournum_cd"))
    .dropDuplicates("svc_mgmt_num", "chnl_typ", "cmpgn_typ", "send_ym", "send_hournum_cd", "click_yn")
    .persist(StorageLevel.MEMORY_AND_DISK_SER)

println("Response data prepared with feature_ym")
println("=" * 80)

// 이전 캐시 정리
resDFFiltered.unpersist()


// ===== Paragraph 6: User Feature Data Loading (MMKT_SVC_BAS) =====
// =============================================================================
// User Feature Data 로딩
// =============================================================================
// 사용자 기본 정보 및 중요 피처를 로딩합니다.
// 전체 suffix에 대해 로딩 (suffix 필터링은 나중에 수행)
// =============================================================================

println("=" * 80)
println("User Feature Data Loading (MMKT_SVC_BAS)")
println("=" * 80)

// Feature 컬럼 정의
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
    "svc_mgmt_num", "strd_ym as feature_ym", 
    "mst_work_dt", "cust_birth_dt", "prcpln_last_chg_dt", 
    "fee_prod_id", "eqp_mdl_cd", "eqp_acqr_dt", 
    "equip_chg_cnt", "svc_scrb_dt", "chg_dt", 
    "cust_age_cd", "sex_cd", "equip_chg_day", 
    "last_equip_chg_dt", "prev_equip_chg_dt", 
    "rten_pen_amt", "agrmt_brch_amt", "eqp_mfact_cd",
    "allot_mth_cnt", "mbr_use_cnt", "mbr_use_amt", 
    "tyr_mbr_use_cnt", "tyr_mbr_use_amt", "mth_cnsl_cnt", 
    "dsat_cnsl_cnt", "simpl_ref_cnsl_cnt", 
    "arpu", "bf_m1_arpu", "voc_arpu", "bf_m3_avg_arpu",
    "tfmly_nh39_scrb_yn", "prcpln_chg_cnt", "email_inv_yn", 
    "copn_use_psbl_cnt", "data_gift_send_yn", "data_gift_recv_yn", 
    "equip_chg_mth_cnt", "dom_tot_pckt_cnt", 
    "scrb_sale_chnl_cl_cd", "op_sale_chnl_cl_cd", "agrmt_dc_end_dt",
    "svc_cd", "svc_st_cd", "pps_yn", "svc_use_typ_cd", 
    "indv_corp_cl_cd", "frgnr_yn", "nm_cust_num", "wlf_dc_cd"
) ++ sigFeaturesMMKT)
  .filter(c => allFeaturesMMKT.contains(c.trim.split(" ")(0).trim))
  .distinct

println(s"Loading ${colListForMMKT.length} features from MMKT")

// MMKT 데이터 로딩
val mmktDFTemp = spark.sql(s"""
    SELECT ${colListForMMKT.mkString(",")}, strd_ym 
    FROM wind_tmt.mmkt_svc_bas_f 
    WHERE strd_ym IN (${featureYmList.mkString("'","','","'")})
""")
  .repartition(200, F.col("svc_mgmt_num"))

// Product 정보 조인
val prodDF = spark.sql("SELECT prod_id AS fee_prod_id, prod_nm AS fee_prod_nm FROM wind.td_zprd_prod")

val mmktDF = mmktDFTemp
    .join(F.broadcast(prodDF), Seq("fee_prod_id"), "left")
    .filter("cust_birth_dt NOT LIKE '9999%'")
    .checkpoint()

println(s"MMKT user data loaded and checkpointed")
println("=" * 80)


// ===== Paragraph 7: Suffix Parameter Setup for Batch Processing =====
// =============================================================================
// Suffix별 배치 처리 준비
// =============================================================================
// 이 단계부터 suffix별로 실행됩니다.
// Zeppelin input 파라미터로 suffix를 받아 처리합니다.
//
// 실행 방법:
// - config_raw_data.py에서 PARAMS = [f"suffix:{i}" for i in range(16)]
// - 또는 수동으로 suffix 파라미터 입력 (예: "0", "1", ..., "f")
// =============================================================================

val smnSuffix = z.input("suffix", "0").toString
val smnCond = smnSuffix.split(",").map(c => s"svc_mgmt_num LIKE '%${c}'").mkString(" OR ")

println("=" * 80)
println(s"Suffix Batch Processing Setup")
println("=" * 80)
println(s"Processing suffix: $smnSuffix")
println(s"Condition: $smnCond")
println("=" * 80)

// Suffix별 user-ym pair 추출
val userYmDF = resDFSelected
    .select("svc_mgmt_num", "feature_ym")
    .distinct()
    .filter(smnCond)
    .repartition(100, F.col("svc_mgmt_num"))
    .persist(StorageLevel.MEMORY_AND_DISK_SER)

println(s"User-ym pairs extracted for suffix '$smnSuffix'")

userYmDF.createOrReplaceTempView("user_ym_df")


// ===== Paragraph 8: App Usage Data Loading (Suffix Filtered) =====
// =============================================================================
// App Usage Data 로딩 (Suffix별)
// =============================================================================
// 시간대별 앱 사용 데이터를 로딩합니다.
// - Hourly app usage (시간대별 앱 사용 토큰)
// - Aggregated features (일별 요약 피처)
// =============================================================================

println("=" * 80)
println(s"App Usage Data Loading for suffix: $smnSuffix")
println("=" * 80)

val hourCols = hourRange.toList.map(h => f"$h, a.total_traffic_$h%02d").mkString(", ")

// Hourly app usage data 로딩
val xdrDF = spark.sql(s"""
    SELECT
        a.svc_mgmt_num,
        a.ym AS feature_ym,
        COALESCE(a.rep_app_title, a.app_uid) AS app_nm,
        hour.send_hournum_cd,
        hour.traffic
    FROM (
        SELECT * 
        FROM dprobe.mst_app_svc_app_monthly
        WHERE ym IN (${featureYmList.mkString("'","','","'")})
          AND ($smnCond)
    ) a
    INNER JOIN user_ym_df b
        ON a.svc_mgmt_num = b.svc_mgmt_num
        AND a.ym = b.feature_ym
    LATERAL VIEW STACK(
        ${hourRange.size},
        $hourCols
    ) hour AS send_hournum_cd, traffic
    WHERE hour.traffic > 1000
""")
.repartition(300, F.col("svc_mgmt_num"), F.col("feature_ym"), F.col("send_hournum_cd"))
.groupBy("svc_mgmt_num", "feature_ym", "send_hournum_cd")
.agg(
  F.collect_set("app_nm").alias("app_usage_token"),
  // 트래픽 정보 기반 피처
  F.count(F.when(F.col("traffic") > 100000, 1)).alias("heavy_usage_app_cnt"),
  F.count(F.when(F.col("traffic").between(10000, 100000), 1)).alias("medium_usage_app_cnt"),
  F.count(F.when(F.col("traffic") < 10000, 1)).alias("light_usage_app_cnt"),
  F.sum("traffic").alias("total_traffic_mb"),
  F.count("*").alias("app_cnt")
)
.persist(StorageLevel.MEMORY_AND_DISK_SER)

println("XDR hourly data loaded and cached")

// 시간대 집계 피처 생성
val xdrAggregatedFeatures = xdrDF
    .groupBy("svc_mgmt_num", "feature_ym")
    .agg(
      // 가장 활발한 시간대
      F.max(F.struct(F.col("app_cnt"), F.col("send_hournum_cd"))).alias("peak_hour_struct"),
      // 활동 시간대 개수
      F.count(F.when(F.col("app_cnt") > 0, 1)).alias("active_hour_cnt"),
      // 평균 시간당 앱 사용 개수
      F.avg("app_cnt").alias("avg_hourly_app_avg"),
      // 전체 트래픽 합계
      F.sum("total_traffic_mb").alias("total_daily_traffic_mb"),
      // 트래픽 구간별 집계
      F.sum("heavy_usage_app_cnt").alias("total_heavy_apps_cnt"),
      F.sum("medium_usage_app_cnt").alias("total_medium_apps_cnt"),
      F.sum("light_usage_app_cnt").alias("total_light_apps_cnt")
    )
    .withColumn("peak_usage_hour_cd", F.col("peak_hour_struct.send_hournum_cd"))
    .withColumn("peak_hour_app_cnt", F.col("peak_hour_struct.app_cnt"))
    .drop("peak_hour_struct")
    .persist(StorageLevel.MEMORY_AND_DISK_SER)

println("XDR aggregated features created")

// Pivot 작업으로 시간대별 앱 사용 토큰 생성
val xdrDFMon = xdrDF
    .repartition(200, F.col("svc_mgmt_num"), F.col("feature_ym"))
    .groupBy("svc_mgmt_num", "feature_ym")
    .pivot("send_hournum_cd", hourRange.map(_.toString))
    .agg(F.first("app_usage_token"))
    .select(
        F.col("svc_mgmt_num") +: F.col("feature_ym") +:
        hourRange.map(h =>
            F.coalesce(F.col(s"$h"), F.array(F.lit("#"))).alias(s"app_usage_${h}_token")
        ): _*
    )
    .join(xdrAggregatedFeatures, Seq("svc_mgmt_num", "feature_ym"), "left")
    .persist(StorageLevel.MEMORY_AND_DISK_SER)

println("XDR monthly pivot data created and cached")
println("=" * 80)


// ===== Paragraph 9: Historical Click Count Feature (Suffix Filtered) =====
// =============================================================================
// Historical Click Count 피처 생성 (Suffix별)
// =============================================================================
// 이전 N개월 동안의 클릭 이력을 집계합니다.
// =============================================================================

println("=" * 80)
println(s"Historical Click Count Feature for suffix: $smnSuffix")
println("=" * 80)

val n = 3  // 이전 3개월

// 현재 및 이전 N개월 데이터 준비
val clickHistoryDF = resDFSelected
    .filter(smnCond)
    .select("svc_mgmt_num", "feature_ym", "send_hournum_cd", "click_yn")
    .withColumn(
        "prev_feature_yms",
        F.expr(s"transform(sequence(1, $n), x -> date_format(add_months(to_date(feature_ym, 'yyyyMM'), -x), 'yyyyMM'))")
    )
    .withColumn("prev_feature_ym", F.explode(F.col("prev_feature_yms")))
    .select("svc_mgmt_num", "prev_feature_ym", "send_hournum_cd", "click_yn")
    .groupBy("svc_mgmt_num", "prev_feature_ym", "send_hournum_cd")
    .agg(F.sum("click_yn").alias("prev_click_cnt"))
    .withColumnRenamed("prev_feature_ym", "feature_ym")

// 현재 feature_ym과 조인
val clickCountDF = resDFSelected
    .filter(smnCond)
    .select("svc_mgmt_num", "feature_ym", "send_hournum_cd")
    .distinct()
    .join(clickHistoryDF, Seq("svc_mgmt_num", "feature_ym", "send_hournum_cd"), "left")
    .na.fill(Map("prev_click_cnt" -> 0.0))
    .withColumnRenamed("prev_click_cnt", "click_cnt")
    .persist(StorageLevel.MEMORY_AND_DISK_SER)

println("Click count features created and cached")
println("=" * 80)


// ===== Paragraph 10: Feature Integration via Multi-way Joins (Suffix Filtered) =====
// =============================================================================
// Feature Integration (Suffix별)
// =============================================================================
// Response data에 모든 feature를 조인합니다.
// 조인 순서: Response → Click Count → XDR (hourly) → XDR (monthly) → MMKT
// =============================================================================

println("=" * 80)
println(s"Feature Integration for suffix: $smnSuffix")
println("=" * 80)

// MMKT 필터링
val mmktDFFiltered = mmktDF
    .filter(smnCond)
    .persist(StorageLevel.MEMORY_AND_DISK_SER)

println("MMKT data filtered and cached")

// Response data 필터링
val resDFSelectedFiltered = resDFSelected
    .filter(smnCond)
    .repartition(200, F.col("svc_mgmt_num"), F.col("feature_ym"), F.col("send_hournum_cd"))
    .persist(StorageLevel.MEMORY_AND_DISK_SER)

println("Response data filtered and cached")

// Multi-way join (조인 순서 최적화)
println("Performing multi-way join...")

val rawDF = resDFSelectedFiltered
    // 1단계: Click count 조인 (가장 작은 테이블)
    .join(clickCountDF, Seq("svc_mgmt_num", "feature_ym", "send_hournum_cd"), "left")
    .na.fill(Map("click_cnt" -> 0.0))
    // 2단계: XDR hourly 조인
    .join(xdrDF, Seq("svc_mgmt_num", "feature_ym", "send_hournum_cd"), "inner")
    // 3단계: XDR monthly 조인 (파티셔닝 변경)
    .repartition(200, F.col("svc_mgmt_num"), F.col("feature_ym"))
    .join(xdrDFMon, Seq("svc_mgmt_num", "feature_ym"), "inner")
    // 4단계: MMKT 조인 (가장 큰 테이블)
    .join(mmktDFFiltered, Seq("svc_mgmt_num", "feature_ym"), "inner")
    .persist(StorageLevel.MEMORY_AND_DISK_SER)

println("Feature integration completed and cached")
println("=" * 80)


// ===== Paragraph 11: Data Type Conversion and Column Standardization =====
// =============================================================================
// 데이터 타입 변환 및 컬럼 표준화
// =============================================================================
// 모델 학습을 위한 데이터 타입 변환을 수행합니다.
// =============================================================================

println("=" * 80)
println(s"Data Type Conversion for suffix: $smnSuffix")
println("=" * 80)

// 제외할 컬럼 (label 등)
val noFeatureCols = Array("click_yn", "hour_gap")

// 컬럼 분류
val tokenCols = rawDF.columns.filter(x => x.endsWith("_token")).distinct
val continuousCols = rawDF.columns
    .filter(x => numericColNameList.map(x.endsWith(_)).reduceOption(_ || _).getOrElse(false))
    .distinct
    .filter(x => !tokenCols.contains(x) && !noFeatureCols.contains(x))
    .distinct

val categoryCols = rawDF.columns
    .filter(x => categoryColNameList.map(x.endsWith(_)).reduceOption(_ || _).getOrElse(false))
    .distinct
    .filter(x => !tokenCols.contains(x) && !noFeatureCols.contains(x) && !continuousCols.contains(x))
    .distinct

val vectorCols = rawDF.columns.filter(x => x.endsWith("_vec"))

println(s"Column classification:")
println(s"  - Token columns: ${tokenCols.length}")
println(s"  - Continuous columns: ${continuousCols.length}")
println(s"  - Category columns: ${categoryCols.length}")
println(s"  - Vector columns: ${vectorCols.length}")

// 데이터 타입 변환
val rawDFRev = rawDF.select(
    (Array("cmpgn_num", "svc_mgmt_num", "send_ym", "send_dt", "feature_ym", "click_yn", "res_utility")
        .map(F.col(_))
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
.distinct()
.withColumn("suffix", F.expr("right(svc_mgmt_num, 1)"))
.repartition(200, F.col("send_ym"), F.col("send_hournum_cd"), F.col("suffix"))
.persist(StorageLevel.MEMORY_AND_DISK_SER)

println("Data type conversion completed and cached")
println("=" * 80)

// 메모리 정리
rawDF.unpersist()
resDFSelectedFiltered.unpersist()
mmktDFFiltered.unpersist()


// ===== Paragraph 12: Raw Data Persistence =====
// =============================================================================
// 통합 Raw Data 저장 (Suffix별, Hour Group별)
// =============================================================================
// Suffix별로 raw data를 저장합니다.
// 시간대를 sliding window 방식으로 group화하여 메모리 효율적으로 처리합니다.
// 
// [설정 파라미터]
// - hourGroupSize: 한 번에 처리할 시간대 개수
// - hourSlide: Slide 크기
//   * hourSlide = hourGroupSize: Overlap 없음 (예: [9,10,11], [12,13,14], ...)
//   * hourSlide < hourGroupSize: Overlap 있음 (예: [9,10,11], [11,12,13], ...)
// =============================================================================

println("=" * 80)
println(s"Saving Raw Data for suffix: $smnSuffix")
println("=" * 80)
println(s"Save path: $rawDataSavePath")
println(s"Partition by: send_ym, send_hournum_cd, suffix")
println(s"Hour range: ${hourRange.mkString(", ")}")
println(s"Hour group size: $hourGroupSize, Slide: $hourSlide")
println("=" * 80)

// Sliding window로 시간대 그룹 생성
val hourGroups = hourRange.sliding(hourGroupSize, hourSlide).toList

println(s"Total hour groups: ${hourGroups.length}")
hourGroups.zipWithIndex.foreach { case (hourGroup, groupIdx) =>
  println(s"  Group ${groupIdx + 1}/${hourGroups.length}: Hours ${hourGroup.mkString(", ")}")
}
println("=" * 80)

// Hour group별로 loop를 돌면서 저장
hourGroups.zipWithIndex.foreach { case (hourGroup, groupIdx) =>
  println(s"Processing group ${groupIdx + 1}/${hourGroups.length}: Hours ${hourGroup.mkString(", ")}")
  
  val rawDFRevHourGroup = rawDFRev
    .filter(F.col("send_hournum_cd").isin(hourGroup: _*))
  
  // 데이터가 있는지 확인 (take(1)로 빠르게 체크)
  if (rawDFRevHourGroup.take(1).nonEmpty) {
    rawDFRevHourGroup
      .repartition(10 * hourGroup.size)  // 그룹 크기에 비례한 파티션 수
      .write
      .mode("overwrite")  // 동적 파티션 덮어쓰기
      .partitionBy("send_ym", "send_hournum_cd", "suffix")
      .option("compression", "snappy")
      .parquet(rawDataSavePath)
    
    println(s"  Group ${groupIdx + 1} saved successfully")
  } else {
    println(s"  Group ${groupIdx + 1} - no data, skipped")
  }
}

println("=" * 80)
println(s"Raw data saved successfully for suffix: $smnSuffix")
println("=" * 80)

// 최종 메모리 정리
rawDFRev.unpersist()
xdrDF.unpersist()
xdrDFMon.unpersist()
xdrAggregatedFeatures.unpersist()
clickCountDF.unpersist()
userYmDF.unpersist()

println(s"Suffix '$smnSuffix' processing completed!")
println("=" * 80)


// =============================================================================
// End of Raw Data Generation
// =============================================================================
// 
// 다음 단계:
// 1. 모든 suffix (0-f) 처리 완료 후 rawDF{version} 데이터 확인
// 2. Transformed data 생성 시 train/test split 및 undersampling 수행
// 
// 확인 방법:
//   // Suffix별 레코드 수
//   spark.read.parquet("aos/sto/rawDF1").groupBy("suffix").count().show()
//   
//   // 시간대별 레코드 수
//   spark.read.parquet("aos/sto/rawDF1").groupBy("send_hournum_cd").count().orderBy("send_hournum_cd").show()
//   
//   // Suffix + 시간대별 레코드 수
//   spark.read.parquet("aos/sto/rawDF1").groupBy("suffix", "send_hournum_cd").count().orderBy("suffix", "send_hournum_cd").show()
// 
// =============================================================================

```


presentation compiler configuration:
Scala version: 3.8.0-bin-nonbootstrapped
Classpath:
<WORKSPACE>/.scala-build/AgenticWorkflow_d5c0a6989e/classes/main [exists ], <HOME>/Library/Caches/Coursier/v1/https/repo1.maven.org/maven2/org/scala-lang/scala3-library_3/3.8.0/scala3-library_3-3.8.0.jar [exists ], <HOME>/Library/Caches/Coursier/v1/https/repo1.maven.org/maven2/org/scala-lang/scala-library/3.8.0/scala-library-3.8.0.jar [exists ], <HOME>/Library/Caches/Coursier/v1/https/repo1.maven.org/maven2/com/sourcegraph/semanticdb-javac/0.10.0/semanticdb-javac-0.10.0.jar [exists ], <WORKSPACE>/.scala-build/AgenticWorkflow_d5c0a6989e/classes/main/META-INF/best-effort [missing ]
Options:
-Xsemanticdb -sourceroot <WORKSPACE> -release 8 -Ywith-best-effort-tasty




#### Error stacktrace:

```

```
#### Short summary: 

dotty.tools.dotc.core.CyclicReference: Cyclic reference involving method wrapString

 Run with -explain-cyclic for more details.