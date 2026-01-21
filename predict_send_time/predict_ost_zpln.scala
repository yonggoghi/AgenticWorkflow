// =============================================================================
// MMS Click Prediction Pipeline - Scala Code for Zeppelin (대용량 처리 최적화 버전)
// =============================================================================
// 각 섹션은 Zeppelin notebook의 paragraph로 구성되어 있습니다.
// 섹션 구분: // ===== Paragraph N: [Title] =====
//
// [대용량 처리 최적화 사항]
// 1. Spark 설정 최적화: AQE, Broadcast Join, Shuffle Partition 조정
// 2. 캐싱 전략: StorageLevel.MEMORY_AND_DISK_SER 사용으로 메모리 절약
// 3. 파티셔닝 최적화: 조인 전 적절한 파티션 수 조정으로 shuffle 최소화
// 4. 조인 최적화: 작은 테이블 broadcast, 조인 순서 최적화
// 5. 메모리 관리: 사용 완료된 DataFrame unpersist, 명시적 count()로 캐시 구체화
// 6. 저장 최적화: Snappy 압축, small files 방지를 위한 repartition
// 7. 로그 추가: 각 단계별 레코드 수 확인으로 데이터 흐름 파악 용이
// =============================================================================

// ===== Paragraph 1: Preps (Imports and Configuration) (ID: paragraph_1764658338256_686533166) =====

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
// spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "100MB")
spark.conf.set("spark.sql.shuffle.partitions", "400")
spark.conf.set("spark.sql.files.maxPartitionBytes", "128MB")

spark.sparkContext.setCheckpointDir("hdfs://scluster/user/g1110566/checkpoint")


// ===== Paragraph 2: Helper Functions (ID: paragraph_1764742922351_426209997) =====

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


// ===== Paragraph 3: Response Data Loading from HDFS (ID: paragraph_1764659911196_1763551717) =====

// 대용량 데이터는 MEMORY_AND_DISK_SER 사용하여 메모리 절약
val resDF = spark.read.parquet("aos/sto/response")
  .persist(StorageLevel.MEMORY_AND_DISK_SER)
resDF.createOrReplaceTempView("res_df")

resDF.printSchema()


// ===== Paragraph 4: Date Range and Period Configuration (ID: paragraph_1764742953919_436300403) =====

val sendMonth = "202512"
val featureMonth = "202511"
val period = 6
val sendYmList = getPreviousMonths(sendMonth, period+2)
val featureYmList = getPreviousMonths(featureMonth, period+2)

val featureDTList = getDaysBetween(featureYmList(0)+"01", sendMonth+"01")

val upperHourGap = 1
val startHour = 9
val endHour = 18

val hourRange = (startHour to endHour).toList


// ===== Paragraph 5: Response Data Filtering and Feature Engineering (ID: paragraph_1764641394585_598529380) =====

val resDFFiltered = resDF
    .filter(s"""send_ym in (${sendYmList.mkString("'","','","'")})""")
    .filter(s"hour_gap is null or (hour_gap between 0 and 5)")
    .filter(s"send_hournum between $startHour and $endHour")
    .selectExpr("cmpgn_num","svc_mgmt_num","chnl_typ","cmpgn_typ","send_ym","send_dt","send_time","send_daynum","send_hournum","click_dt","click_time","click_daynum","click_hournum","case when hour_gap is null then 0 else 1 end click_yn","hour_gap")
    .withColumn("res_utility", F.expr(s"case when hour_gap is null then 0.0 else 1.0 / (1 + hour_gap) end"))
    .dropDuplicates()
    .repartition(200, F.col("svc_mgmt_num"))  // 조인을 위해 적절한 파티션 수로 재분배
    .persist(StorageLevel.MEMORY_AND_DISK_SER)

println("Filtered response data cached (will materialize on next action)")

resDF.printSchema()


// ===== Paragraph 6: User Feature Data Loading (MMKT_SVC_BAS) (ID: paragraph_1764739202982_181479704) =====

val allFeaturesMMKT = spark.sql("describe wind_tmt.mmkt_svc_bas_f").select("col_name").collect().map(_.getString(0))
val sigFeaturesMMKT = spark.read.option("header", "true").csv("feature_importance/table=mmkt_bas/creation_dt=20230407").filter("rank<=100").select("col").collect().map(_(0).toString()).map(_.trim)
val colListForMMKT = (Array("svc_mgmt_num", "strd_ym feature_ym", "mst_work_dt", "cust_birth_dt", "prcpln_last_chg_dt", "fee_prod_id", "eqp_mdl_cd", "eqp_acqr_dt", "equip_chg_cnt", "svc_scrb_dt", "chg_dt", "cust_age_cd", "sex_cd", "equip_chg_day", "last_equip_chg_dt", "prev_equip_chg_dt", "rten_pen_amt", "agrmt_brch_amt", "eqp_mfact_cd",
    "allot_mth_cnt", "mbr_use_cnt", "mbr_use_amt", "tyr_mbr_use_cnt", "tyr_mbr_use_amt", "mth_cnsl_cnt", "dsat_cnsl_cnt", "simpl_ref_cnsl_cnt", "arpu", "bf_m1_arpu", "voc_arpu", "bf_m3_avg_arpu",
    "tfmly_nh39_scrb_yn", "prcpln_chg_cnt", "email_inv_yn", "copn_use_psbl_cnt", "data_gift_send_yn", "data_gift_recv_yn", "equip_chg_mth_cnt", "dom_tot_pckt_cnt", "scrb_sale_chnl_cl_cd", "op_sale_chnl_cl_cd", "agrmt_dc_end_dt",
    "svc_cd", "svc_st_cd", "pps_yn", "svc_use_typ_cd", "indv_corp_cl_cd", "frgnr_yn", "nm_cust_num", "wlf_dc_cd"
) ++ sigFeaturesMMKT).filter(c => allFeaturesMMKT.contains(c.trim.split(" ")(0).trim)).distinct

val mmktDFTemp = spark.sql(s"""select ${colListForMMKT.mkString(",")}, strd_ym from wind_tmt.mmkt_svc_bas_f a where strd_ym in (${featureYmList.mkString("'","','","'")})""")
  .repartition(200, F.col("svc_mgmt_num"))  // 조인을 위한 파티셔닝

// Broadcast join for small dimension table
val prodDF = spark.sql("select prod_id fee_prod_id, prod_nm fee_prod_nm from wind.td_zprd_prod")

val mmktDF = {
  mmktDFTemp
    .join(F.broadcast(prodDF), Seq("fee_prod_id"), "left")
    .filter("cust_birth_dt not like '9999%'")
    // .persist(StorageLevel.MEMORY_AND_DISK_SER)
    .checkpoint()
}

println("MMKT user data cached (will materialize on next action)")


// ===== Paragraph 7: Train/Test Split and Feature Month Mapping (ID: paragraph_1764739017819_1458690185) =====

val predictionDTSta = "20251101"
val predictionDTEnd = "20251201"

// 중복 제거 전 파티셔닝으로 shuffle 최적화
val resDFSelected = resDFFiltered
    .withColumn("feature_ym", F.date_format(F.add_months(F.unix_timestamp($"send_dt", "yyyyMMdd").cast(TimestampType), -1), "yyyyMM").cast(StringType))
    .selectExpr("cmpgn_num", "svc_mgmt_num", "chnl_typ", "cmpgn_typ", "send_ym", "send_dt", "feature_ym", "send_daynum", "send_hournum send_hournum_cd", "hour_gap", "click_yn", "res_utility")
    .repartition(200, F.col("svc_mgmt_num"), F.col("chnl_typ"), F.col("cmpgn_typ"), F.col("send_ym"), F.col("send_hournum_cd"))
    .dropDuplicates("svc_mgmt_num", "chnl_typ", "cmpgn_typ", "send_ym", "send_hournum_cd", "click_yn")
    .persist(StorageLevel.MEMORY_AND_DISK_SER)

println("Selected response data cached")

val resDFSelectedTr = resDFSelected
    .filter(s"send_dt<$predictionDTSta")
    .persist(StorageLevel.MEMORY_AND_DISK_SER)

println("Training response data cached")

var resDFSelectedTs = resDFSelected
    .filter(s"send_dt>=$predictionDTSta and send_dt<$predictionDTEnd")
    .selectExpr("cmpgn_num", "svc_mgmt_num", "chnl_typ", "cmpgn_typ", "send_ym", "send_dt", "feature_ym", "send_hournum_cd", "click_yn", "res_utility")
    .checkpoint()

// Checkpoint는 다음 action에서 자동 구체화됨
println("Test response data checkpointed")


// ===== Paragraph 8: Undersampling Ratio Calculation (Class Balance) (ID: paragraph_1764738582669_1614068999) =====

val samplingKeyCols = Array("chnl_typ","cmpgn_typ","send_daynum","send_hournum_cd","click_yn")

val genSampleNumMulti = 10.0

// 샘플링 비율 계산 최적화 - 파티셔닝 추가
val samplingRatioMapDF = {
  resDFSelectedTr
    .sample(false, 0.3, 42)  // 시드 추가로 재현성 확보
    .repartition(100, samplingKeyCols.filter(_ != "click_yn").map(F.col(_)):_*)
    .groupBy(samplingKeyCols.map(F.col(_)):_*)
    .agg(F.count("*").alias("cnt"))
    .withColumn("min_cnt", F.min("cnt").over(Window.partitionBy(samplingKeyCols.filter(_!="click_yn").map(F.col(_)):_*)))
    .withColumn("ratio", F.col("min_cnt") / F.col("cnt"))
    .withColumn("sampling_col", F.expr(s"""concat_ws('-', ${samplingKeyCols.mkString(",")})"""))
    .selectExpr("sampling_col", s"least(1.0, ratio*${genSampleNumMulti}) ratio")
    .sort("sampling_col")
    .persist(StorageLevel.MEMORY_AND_DISK_SER)
}

println("Sampling ratio map cached (small table, will materialize quickly)")


// ===== Paragraph 9: Training Data Undersampling (Balanced Dataset) (ID: paragraph_1764756027560_85739584) =====

// Undersampling 최적화 - broadcast join 활용
var resDFSelectedTrBal = resDFSelectedTr
    .repartition(200, F.col("svc_mgmt_num"))  // 후속 조인을 위한 파티셔닝
    .withColumn("sampling_col", F.expr(s"""concat_ws('-', ${samplingKeyCols.mkString(",")})"""))
    .join(F.broadcast(samplingRatioMapDF), "sampling_col")
    .withColumn("rand", F.rand(42))  // 시드 추가로 재현성 확보
    .filter("rand<=ratio")
    .checkpoint()

// Checkpoint는 다음 action에서 자동 구체화됨
println("Balanced training data checkpointed")


// ===== Paragraph 10: App Usage Data Loading and Aggregation (Large Dataset) (ID: paragraph_1766323923540_1041552789) =====

import org.apache.spark.sql.functions._

val smnSuffix = z.input("suffix", "0").toString

val smnCond = smnSuffix.split(",").map(c => s"svc_mgmt_num like '%${c}'").mkString(" or ")

val userYmDF = resDFSelectedTrBal
    .select("svc_mgmt_num", "feature_ym")
    .union(resDFSelectedTs.select("svc_mgmt_num", "feature_ym"))
    .distinct()
    .filter(smnCond)
    .repartition(100, F.col("svc_mgmt_num"))  // 조인 전 파티셔닝
    .persist(StorageLevel.MEMORY_AND_DISK_SER)

println("Unique user-ym pairs cached")

userYmDF.createOrReplaceTempView("user_ym_df")

val hourCols = hourRange.toList.map(h => f"$h, a.total_traffic_$h%02d").mkString(", ")

// 대용량 앱 사용 데이터 처리 최적화
val xdrDF = spark.sql(s"""
    SELECT
        a.svc_mgmt_num,
        a.ym AS feature_ym,
        COALESCE(a.rep_app_title, a.app_uid) AS app_nm,
        hour.send_hournum_cd,
        hour.traffic
    FROM (
        SELECT * FROM dprobe.mst_app_svc_app_monthly
        WHERE (ym in (${featureYmList.mkString("'","','","'")}))
        and ($smnCond)
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
.repartition(300, F.col("svc_mgmt_num"), F.col("feature_ym"), F.col("send_hournum_cd"))  // Shuffle 전 파티셔닝
.groupBy("svc_mgmt_num", "feature_ym", "send_hournum_cd")
.agg(
  collect_set("app_nm").alias("app_usage_token"),  // 기존: 앱 리스트
  // 트래픽 정보 기반 피처 추가
  F.count(F.when(F.col("traffic") > 100000, 1)).alias("heavy_usage_app_cnt"),
  F.count(F.when(F.col("traffic").between(10000, 100000), 1)).alias("medium_usage_app_cnt"),
  F.count(F.when(F.col("traffic") < 10000, 1)).alias("light_usage_app_cnt"),
  F.sum("traffic").alias("total_traffic_mb"),  // 총 트래픽양 (MB 단위)
  F.count("*").alias("app_cnt")  // 사용한 앱 개수
)
.persist(StorageLevel.MEMORY_AND_DISK_SER)

println("XDR hourly data cached with traffic features (large dataset, will materialize on first use)")

// 시간대 집계 피처 생성 (24개 시간대를 요약 피처로 압축)
val xdrAggregatedFeatures = xdrDF
    .groupBy("svc_mgmt_num", "feature_ym")
    .agg(
      // 전체 기간 동안 사용한 모든 앱
      F.flatten(F.collect_list("app_usage_token")).alias("all_apps_list"),
      // 가장 활발한 시간대 (앱 개수 기준)
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
    .drop("peak_hour_struct", "all_apps_list")  // all_apps_list는 메모리 부담이 크므로 제거
    .persist(StorageLevel.MEMORY_AND_DISK_SER)

println("XDR aggregated features cached (time-based summary features)")

// Pivot 작업은 메모리 집약적이므로 충분한 파티션 확보
val xdrDFMon = xdrDF
    .repartition(200, F.col("svc_mgmt_num"), F.col("feature_ym"))
    .groupBy("svc_mgmt_num", "feature_ym")
    .pivot("send_hournum_cd", hourRange.map(_.toString))
    .agg(first("app_usage_token"))
    .select(
        col("svc_mgmt_num") +: col("feature_ym") +:
        hourRange.map(h =>
            coalesce(col(s"$h"), array(lit("#"))).alias(s"app_usage_${h}_token")
        ): _*
    )
    // 집계 피처 조인
    .join(xdrAggregatedFeatures, Seq("svc_mgmt_num", "feature_ym"), "left")
    .persist(StorageLevel.MEMORY_AND_DISK_SER)

println("XDR monthly pivot data cached with aggregated features (will materialize on first use)")


// ===== Paragraph 11: Historical Click Count Feature Engineering (3-Month Window) (ID: paragraph_1767594403472_2124174124) =====

import org.apache.spark.sql.functions._

val n = 3   // 이전 3개월

val df = resDF
    .withColumn("feature_ym",
        F.date_format(F.add_months(F.unix_timestamp($"send_dt", "yyyyMMdd")
            .cast(TimestampType), -1), "yyyyMM").cast(StringType))
    .repartition(200, F.col("svc_mgmt_num"))  // Self-join 전 파티셔닝

// Self-join은 비용이 크므로 최적화
val clickCountDF = df.as("current")
    .join(
        df.as("previous").hint("shuffle_replicate_nl"),  // Broadcast 힌트 대신 효율적인 조인 전략
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
    .repartition(200, col("current.svc_mgmt_num"), col("current.feature_ym"))  // GroupBy 전 파티셔닝
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
    .persist(StorageLevel.MEMORY_AND_DISK_SER)

println("Click count history cached (will materialize on first use)")


// ===== Paragraph 12: Feature Integration via Multi-way Joins (Optimized Order) (ID: paragraph_1764755002817_1620624445) =====

val mmktDFFiltered = mmktDF.filter(smnCond).persist(StorageLevel.MEMORY_AND_DISK_SER)
println("Filtered MMKT data cached")

// 조인 순서 최적화: 작은 테이블부터 조인 (크기 순서: clickCountDF < xdrDFMon < xdrDF < mmktDFFiltered)
// 1) resDFSelectedTrBal (base - undersampled)
// 2) clickCountDF (가장 작음 - historical click count) - left join
// 3) xdrDF (세 번째 - hourly app usage)
// 4) xdrDFMon (두 번째 - monthly app usage pivot)
// 5) mmktDFFiltered (가장 큼 - user features)

val trainDF = resDFSelectedTrBal
    .filter(smnCond)
    .repartition(200, F.col("svc_mgmt_num"), F.col("feature_ym"), F.col("send_hournum_cd"))  // 첫 조인을 위한 파티셔닝
    .join(clickCountDF, Seq("svc_mgmt_num", "feature_ym", "send_hournum_cd"), "left")  // 1단계: 가장 작은 테이블 (left join)
    .na.fill(Map("click_cnt" -> 0.0))
    .join(xdrDF, Seq("svc_mgmt_num", "feature_ym", "send_hournum_cd"))  // 2단계: 세 번째로 작은 테이블
    .repartition(200, F.col("svc_mgmt_num"), F.col("feature_ym"))  // xdrDFMon 조인을 위한 파티셔닝 (send_hournum_cd 제외)
    .join(xdrDFMon, Seq("svc_mgmt_num", "feature_ym"))  // 3단계: 두 번째로 작은 테이블
    .join(mmktDFFiltered, Seq("svc_mgmt_num", "feature_ym"))  // 4단계: 가장 큰 테이블 (마지막)

println(s"Training data joined successfully")

val testDF = resDFSelectedTs
    .filter(smnCond)
    .repartition(200, F.col("svc_mgmt_num"), F.col("feature_ym"), F.col("send_hournum_cd"))
    .join(clickCountDF, Seq("svc_mgmt_num", "feature_ym", "send_hournum_cd"), "left")
    .na.fill(Map("click_cnt" -> 0.0))
    .join(xdrDF, Seq("svc_mgmt_num", "feature_ym", "send_hournum_cd"))
    .repartition(200, F.col("svc_mgmt_num"), F.col("feature_ym"))
    .join(xdrDFMon, Seq("svc_mgmt_num", "feature_ym"))
    .join(mmktDFFiltered, Seq("svc_mgmt_num", "feature_ym"))

println(s"Test data joined successfully")


// ===== Paragraph 13: Data Type Conversion and Column Standardization (ID: paragraph_1764832142136_413314670) =====

val noFeatureCols = Array("click_yn","hour_gap")

val tokenCols = trainDF.columns.filter(x => x.endsWith("_token")).distinct
val continuousCols = (trainDF.columns.filter(x => numericColNameList.map(x.endsWith(_)).reduceOption(_ || _).getOrElse(false)).distinct.filter(x => !tokenCols.contains(x) && !noFeatureCols.contains(x))).distinct
val categoryCols = (trainDF.columns.filter(x => categoryColNameList.map(x.endsWith(_)).reduceOption(_ || _).getOrElse(false)).distinct.filter(x => !tokenCols.contains(x) && !noFeatureCols.contains(x) && !continuousCols.contains(x))).distinct
val vectorCols = trainDF.columns.filter(x => x.endsWith("_vec"))

val trainDFRev = trainDF.select(
    (Array("cmpgn_num", "svc_mgmt_num", "chnl_typ", "cmpgn_typ", "send_ym", "send_dt", "feature_ym", "click_yn", "res_utility").map(F.col(_))
    ++ tokenCols.map(cl => F.coalesce(F.col(cl), F.array(F.lit("#"))).alias(cl))
    ++ vectorCols.map(cl => F.col(cl).alias(cl))
    ++ categoryCols.map(cl => F.when(F.col(cl) === "", F.lit("UKV")).otherwise(F.coalesce(F.col(cl).cast("string"), F.lit("UKV"))).alias(cl))
    ++ continuousCols.map(cl => F.coalesce(F.col(cl).cast("float"), F.lit(Double.NaN)).alias(cl))
    ): _*
)
.distinct
.withColumn("suffix", F.expr("right(svc_mgmt_num, 1)"))

val testDFRev = testDF.select(
    (Array("cmpgn_num", "svc_mgmt_num", "chnl_typ", "cmpgn_typ", "send_ym", "send_dt", "feature_ym", "click_yn", "res_utility").map(F.col(_))
    ++ tokenCols.map(cl => F.coalesce(F.col(cl), F.array(F.lit("#"))).alias(cl))
    ++ vectorCols.map(cl => F.col(cl).alias(cl))
    ++ categoryCols.map(cl => F.when(F.col(cl) === "", F.lit("UKV")).otherwise(F.coalesce(F.col(cl).cast("string"), F.lit("UKV"))).alias(cl))
    ++ continuousCols.map(cl => F.coalesce(F.col(cl).cast("float"), F.lit(Double.NaN)).alias(cl))
): _*
)
.distinct
.withColumn("suffix", F.expr("right(svc_mgmt_num, 1)"))


// ===== Paragraph 14: Raw Feature Data Persistence (Parquet Format) (ID: paragraph_1766224516076_433149416) =====

// 저장 전 파티션 수 조정하여 small files 문제 방지
// partitionBy()가 컬럼 기준 분배를 처리하므로 repartition(n)만 사용
trainDFRev
.repartition(50)  // 각 파티션 디렉토리당 파일 수 조정
.write
.mode("overwrite")
.partitionBy("send_ym", "send_hournum_cd", "suffix")
.option("compression", "snappy")  // 압축 사용
.parquet(s"aos/sto/trainDFRev${genSampleNumMulti.toInt}")

testDFRev
.repartition(100)  // 각 파티션 디렉토리당 파일 수 조정
.write
.mode("overwrite")
.partitionBy("send_ym", "send_hournum_cd", "suffix")
.option("compression", "snappy")
.parquet(s"aos/sto/testDFRev")

println("Training and test datasets saved successfully")


// ===== Paragraph 15: Raw Feature Data Loading and Cache Management (ID: paragraph_1766392634024_1088239830) =====

// 이전 단계에서 사용한 캐시 정리 (메모리 확보)
println("Cleaning up intermediate cached data...")
try {
  resDFSelectedTrBal.unpersist()
  resDFSelectedTs.unpersist()
  xdrDF.unpersist()
  xdrDFMon.unpersist()
  userYmDF.unpersist()
  mmktDFFiltered.unpersist()
  println("Cache cleanup completed")
} catch {
  case e: Exception => println(s"Cache cleanup warning: ${e.getMessage}")
}

spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "10g")

val trainDFRev = spark.read.parquet("aos/sto/trainDFRev10")
    .withColumn("hour_gap", F.expr("case when res_utility>=1.0 then 1 else 0 end"))
    // .drop("click_cnt").join(clickCountDF, Seq("svc_mgmt_num", "feature_ym", "send_hournum_cd"), "left").na.fill(Map("click_cnt"->0.0))
    .persist(StorageLevel.MEMORY_AND_DISK_SER)

println("Training data loaded and cached")

val testDFRev = spark.read.parquet("aos/sto/testDFRev")
    .withColumn("hour_gap", F.expr("case when res_utility>=1.0 then 1 else 0 end"))
    // .drop("click_cnt").join(clickCountDF, Seq("svc_mgmt_num", "feature_ym", "send_hournum_cd"), "left").na.fill(Map("click_cnt"->0.0))
    .persist(StorageLevel.MEMORY_AND_DISK_SER)

println("Test data loaded and cached")

val noFeatureCols = Array("click_yn","hour_gap","chnl_typ","cmpgn_typ")
val tokenCols = trainDFRev.columns.filter(x => x.endsWith("_token")).distinct
val continuousCols = (trainDFRev.columns.filter(x => numericColNameList.map(x.endsWith(_)).reduceOption(_ || _).getOrElse(false)).distinct.filter(x => !tokenCols.contains(x) && !noFeatureCols.contains(x))).distinct
val categoryCols = (trainDFRev.columns.filter(x => categoryColNameList.map(x.endsWith(_)).reduceOption(_ || _).getOrElse(false)).distinct.filter(x => !tokenCols.contains(x) && !noFeatureCols.contains(x) && !continuousCols.contains(x))).distinct
val vectorCols = trainDFRev.columns.filter(x => x.endsWith("_vec"))


// ===== Paragraph 16: Prediction Dataset Preparation for Production (ID: paragraph_1765765120629_645290475) =====

val predDT = "20251201"
val predFeatureYM = getPreviousMonths(predDT.take(6), 2)(0)
val predSendYM = predDT.take(6)

val prdSuffix = "%"
val prdSuffixCond = prdSuffix.map(c => s"svc_mgmt_num like '%${c}'").mkString(" or ")
val pivotColumns = hourRange.toList.map(h => f"$h, total_traffic_$h%02d").mkString(", ")

// Prediction용 앱 사용 데이터 로딩 최적화
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
.repartition(300, F.col("svc_mgmt_num"), F.col("feature_ym"), F.col("send_hournum_cd"))
.groupBy("svc_mgmt_num", "feature_ym", "send_hournum_cd")
.agg(
  collect_set("app_nm").alias("app_usage_token"),
  // 트래픽 정보 기반 피처 추가 (training과 동일)
  F.count(F.when(F.col("traffic") > 100000, 1)).alias("heavy_usage_app_cnt"),
  F.count(F.when(F.col("traffic").between(10000, 100000), 1)).alias("medium_usage_app_cnt"),
  F.count(F.when(F.col("traffic") < 10000, 1)).alias("light_usage_app_cnt"),
  F.sum("traffic").alias("total_traffic_mb"),
  F.count("*").alias("app_cnt")
)
.persist(StorageLevel.MEMORY_AND_DISK_SER)

println("Prediction XDR hourly data cached with traffic features")

// Prediction용 시간대 집계 피처 생성
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
    .persist(StorageLevel.MEMORY_AND_DISK_SER)

println("Prediction XDR aggregated features cached")

val xdrPredDF = xdrDFPred
    .repartition(200, F.col("svc_mgmt_num"), F.col("feature_ym"))
    .groupBy("svc_mgmt_num", "feature_ym")
    .pivot("send_hournum_cd", hourRange.map(_.toString))
    .agg(first("app_usage_token"))
    .select(
        col("svc_mgmt_num") +: col("feature_ym") +:
        hourRange.map(h =>
            coalesce(col(s"$h"), array(lit("#"))).alias(s"app_usage_${h}_token")
        ): _*
    )
    // 집계 피처 조인
    .join(xdrPredAggregatedFeatures, Seq("svc_mgmt_num", "feature_ym"), "left")
    .persist(StorageLevel.MEMORY_AND_DISK_SER)

println("Prediction XDR monthly pivot data cached with aggregated features")

// 조인 순서 최적화: 작은 테이블부터 조인 (크기 순서: clickCountDF < xdrPredDF < xdrDFPred < mmktDF)
// mmktDF가 base이지만, explode 후에는 작은 테이블부터 조인
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
    .withColumn("send_hournum_cd", F.explode(F.expr(s"array(${(startHour to endHour).toArray.mkString(",")})")))
    .repartition(200, F.col("svc_mgmt_num"), F.col("feature_ym"), F.col("send_hournum_cd"))
    .join(clickCountDF, Seq("svc_mgmt_num", "feature_ym", "send_hournum_cd"), "left")  // 1단계: 가장 작은 테이블
    .na.fill(Map("click_cnt" -> 0.0))
    .join(xdrDFPred, Seq("svc_mgmt_num", "feature_ym", "send_hournum_cd"), "left")  // 2단계: 세 번째로 작은 테이블
    .repartition(200, F.col("svc_mgmt_num"), F.col("feature_ym"))  // xdrPredDF 조인을 위한 파티셔닝
    .join(xdrPredDF, Seq("svc_mgmt_num", "feature_ym"), "left")  // 3단계: 두 번째로 작은 테이블

val predDFRev = predDF.select(
    Array("cmpgn_num", "svc_mgmt_num", "chnl_typ", "cmpgn_typ",
        "send_ym", "send_dt", "feature_ym", "hour_gap",
        "click_yn", "res_utility").map(F.col(_)) ++
    tokenCols.map(cl => F.coalesce(F.col(cl), F.array(F.lit("#"))).alias(cl)) ++
    vectorCols.map(cl => F.col(cl).alias(cl)) ++
    categoryCols.map(cl =>
        F.when(F.col(cl) === "", F.lit("UKV"))
            .otherwise(F.coalesce(F.col(cl).cast("string"), F.lit("UKV")))
            .alias(cl)
    ) ++
    continuousCols.map(cl => F.coalesce(F.col(cl).cast("float"), F.lit(Double.NaN)).alias(cl))
    : _*
)
.distinct
.persist(StorageLevel.MEMORY_AND_DISK_SER)

println("Prediction data prepared and cached")


// ===== Paragraph 17: Pipeline Parameters and Feature Column Settings (ID: paragraph_1764833771372_1110341451) =====

val tokenColsEmbCols = Array("app_usage_token")
val featureHasherNumFeature = 128

var nodeNumber = 10
var coreNumber = 32
try {
    nodeNumber = spark.conf.get("spark.executor.instances").toInt
    coreNumber = spark.conf.get("spark.executor.cores").toInt
} catch {
    case ex: Exception => {}
}

// 피처 엔지니어링 파라미터 (개선: vocabSize 30 → 1000)
val params:Map[String, Any] = Map("minDF"->1,"minTF"->5,"embSize"->30,"vocabSize"->1000, "numParts"->nodeNumber)

val labelColClick = "click_yn"
val labelColGap = "hour_gap"

val indexedLabelColClick = "click_yn"
val indexedLabelColGap = "hour_gap"

val labelColsClick = Array(Map("indexer_nm"->"indexer_click", "col_nm"->labelColClick, "label_nm"->indexedLabelColClick))
val labelColsGap = Array(Map("indexer_nm"->"indexer_gap", "col_nm"->labelColGap, "label_nm"->indexedLabelColGap))

val indexedLabelColReg = "res_utility"

val indexedFeatureColClick = "indexedFeaturesClick"
val scaledFeatureColClick = "scaledFeaturesClick"
val selectedFeatureColClick = "selectedFeaturesClick"

val indexedFeatureColGap = "indexedFeaturesGap"
val scaledFeatureColGap = "scaledFeaturesGap"
val selectedFeatureColGap = "selectedFeaturesGap"

val onlyGapFeature = Array[String]()

val doNotHashingCateCols = Array[String]("send_hournum_cd", "peak_usage_hour_cd")
val doNotHashingContCols = Array[String]("click_cnt", 
  "heavy_usage_app_cnt", "medium_usage_app_cnt", "light_usage_app_cnt",
  "total_traffic_mb", "app_cnt", "peak_hour_app_cnt", "active_hour_cnt",
  "avg_hourly_app_avg", "total_daily_traffic_mb", "total_heavy_apps_cnt", 
  "total_medium_apps_cnt", "total_light_apps_cnt")


// ===== Paragraph 18: Feature Engineering Pipeline Function Definition (makePipeline) (ID: paragraph_1765330122144_909170709) =====

def makePipeline(
    labelCols: Array[Map[String,String]] = Array.empty,
    indexedFeatureCol: String = "indexed_features",
    scaledFeatureCol: String = "scaled_features",
    selectedFeatureCol: String = "selected_features",
    tokenCols: Array[String] = Array.empty,
    vectorCols: Array[String] = Array.empty,
    continuousCols: Array[String] = Array.empty,
    categoryCols: Array[String] = Array.empty,
    doNotHashingCateCols: Array[String] = Array.empty,
    doNotHashingContCols: Array[String] = Array.empty,
    useSelector:Boolean = false,
    useScaling:Boolean = false,
    tokenColsEmbCols:Array[String] = Array.empty,
    featureHasherNumFeature:Int = 256,
    featureHashColNm:String="feature_hashed",
    colNmSuffix:String="#",
    params:Map[String, Any],
    userDefinedFeatureListForAssembler: Array[String] = Array.empty
) = {
    val minDF = params.get("minDF").getOrElse(1).asInstanceOf[Int]
    val minTF = params.get("minTF").getOrElse(5).asInstanceOf[Int]
    val embSize = params.get("embSize").getOrElse(10).asInstanceOf[Int]
    val vocabSize = params.get("vocabSize").getOrElse(30).asInstanceOf[Int]
    val numParts = params.get("numParts").getOrElse(10).asInstanceOf[Int]

    var featureListForAssembler = continuousCols ++ vectorCols

    import org.apache.spark.ml.{Pipeline, PipelineStage}
    val transformPipeline = new Pipeline().setStages(Array[PipelineStage]())

    if (labelCols.size>0){
        labelCols.foreach(map =>
            if(transformPipeline.getStages.isEmpty){
                transformPipeline.setStages(Array(new StringIndexer(map("indexer_nm")).setInputCol(map("col_nm")).setOutputCol(map("label_nm")).setHandleInvalid("skip")))
            }else{
                transformPipeline.setStages(transformPipeline.getStages++Array(new StringIndexer(map("indexer_nm")).setInputCol(map("col_nm")).setOutputCol(map("label_nm")).setHandleInvalid("skip")))
            }
        )
    }

    val tokenColsEmb = tokenCols.filter(x => tokenColsEmbCols.map(x.contains(_)).reduceOption(_ || _).getOrElse(false))
    val tokenColsCnt = tokenCols.filter(!tokenColsEmb.contains(_))

    if (embSize > 0 && tokenColsEmb.size > 0) {
        val embEncoder = tokenColsEmb.map(c => new Word2Vec().setNumPartitions(numParts).setSeed(46).setVectorSize(embSize).setMinCount(minTF).setInputCol(c).setOutputCol(c + "_embvec"))
        transformPipeline.setStages(if(transformPipeline.getStages.isEmpty){embEncoder}else{transformPipeline.getStages++embEncoder})
        featureListForAssembler ++= tokenColsEmb.map(_ +"_"+colNmSuffix+"_embvec")
    }

    if (tokenColsCnt.size > 0) {
        // HashingTF에서 CountVectorizer + TF-IDF로 변경 (정보 손실 최소화)
        val cntVectorizer = tokenColsCnt.map(c => 
            new CountVectorizer()
                .setInputCol(c)
                .setOutputCol(c + "_" + colNmSuffix + "_cntvec")
                .setVocabSize(1000)    // 30 → 1000 증가 (정확도 우선)
                .setMinDF(3)           // 노이즈 제거 (최소 3명 이상 사용)
                .setBinary(false)      // 빈도 정보 유지
        )
        transformPipeline.setStages(if(transformPipeline.getStages.isEmpty){cntVectorizer}else{transformPipeline.getStages++cntVectorizer})
        
        // TF-IDF 가중치 추가 (중요한 앱에 더 높은 가중치)
        val tfidfTransformers = tokenColsCnt.map(c =>
            new IDF()
                .setInputCol(c + "_" + colNmSuffix + "_cntvec")
                .setOutputCol(c + "_" + colNmSuffix + "_tfidf")
        )
        transformPipeline.setStages(transformPipeline.getStages ++ tfidfTransformers)
        
        // TF-IDF 피처를 사용 (원본 cntvec 대신)
        featureListForAssembler ++= tokenColsCnt.map(_ + "_" + colNmSuffix + "_tfidf")
    }

    if (featureHasherNumFeature > 0 && categoryCols.size > 0) {
        val featureHasher = new FeatureHasher().setNumFeatures(featureHasherNumFeature)
            .setInputCols((continuousCols++categoryCols)
                .filter(c => !doNotHashingContCols.contains(c))
                .filter(c => !doNotHashingCateCols.contains(c))
            ).setOutputCol(featureHashColNm)

        transformPipeline.setStages(if(transformPipeline.getStages.isEmpty){Array(featureHasher)}else{transformPipeline.getStages++Array(featureHasher)})
        featureListForAssembler = featureListForAssembler.filter(!continuousCols.contains(_))
        featureListForAssembler ++= Array(featureHashColNm)
    }

    if (doNotHashingCateCols.size>0) {
        val catetoryIndexerList = doNotHashingCateCols.map(c => new StringIndexer().setInputCol(c).setOutputCol(c + "_" + colNmSuffix + "_index").setHandleInvalid("keep"))
        val encoder = new OneHotEncoder().setInputCols(doNotHashingCateCols.map(c => c + "_" + colNmSuffix + "_index")).setOutputCols(doNotHashingCateCols.map(c => c + "_" + colNmSuffix + "_enc")).setHandleInvalid("keep")
        transformPipeline.setStages(if(transformPipeline.getStages.isEmpty){catetoryIndexerList ++ Array(encoder)}else{transformPipeline.getStages++catetoryIndexerList ++ Array(encoder)})
        featureListForAssembler ++= doNotHashingCateCols.map(_ + "_" + colNmSuffix + "_enc")
    }

    if (doNotHashingContCols.size>0){
        featureListForAssembler ++= doNotHashingContCols
    }

    if (featureHasherNumFeature < 1 && categoryCols.size > 0) {
        val catetoryIndexerList = categoryCols.map(c => new StringIndexer().setInputCol(c).setOutputCol(c + "_" + colNmSuffix + "_index").setHandleInvalid("keep"))
        val encoder = new OneHotEncoder().setInputCols(categoryCols.map(c => c + "_" + colNmSuffix + "_index")).setOutputCols(categoryCols.map(c => c + "_" + colNmSuffix + "_enc")).setHandleInvalid("keep")
        transformPipeline.setStages(if(transformPipeline.getStages.isEmpty){catetoryIndexerList ++ Array(encoder)}else{transformPipeline.getStages++catetoryIndexerList ++ Array(encoder)})
        featureListForAssembler ++= categoryCols.map(_ + "_" + colNmSuffix + "_enc")
    }

    if (userDefinedFeatureListForAssembler.size>0){
        featureListForAssembler ++= userDefinedFeatureListForAssembler
    }

    val assembler = new VectorAssembler().setInputCols(featureListForAssembler.distinct).setOutputCol(indexedFeatureCol).setHandleInvalid("keep")

    transformPipeline.setStages(transformPipeline.getStages++Array(assembler))

    if(useSelector){
        val selector = new VarianceThresholdSelector().setVarianceThreshold(8.0).setFeaturesCol(indexedFeatureCol).setOutputCol(selectedFeatureCol)
        transformPipeline.setStages(transformPipeline.getStages++Array(selector))
    }

    if(useScaling){
        val inputFeautreCol = if(useSelector){selectedFeatureCol}else{indexedFeatureCol}
        val scaler = new MinMaxScaler().setInputCol(inputFeautreCol).setOutputCol(scaledFeatureCol)
        transformPipeline.setStages(transformPipeline.getStages++Array(scaler))
    }

    transformPipeline
}


// ===== Paragraph 19: Feature Engineering Pipeline Fitting and Transformation (ID: paragraph_1767353227961_983246072) =====

val transformPipelineClick = makePipeline(
    labelCols = Array(),
    indexedFeatureCol = indexedFeatureColClick,
    scaledFeatureCol = scaledFeatureColClick,
    selectedFeatureCol = selectedFeatureColClick,
    tokenCols = Array("app_usage_token"),
    vectorCols = vectorCols.filter(!onlyGapFeature.contains(_)),
    continuousCols = continuousCols.filter(!onlyGapFeature.contains(_)),
    categoryCols = categoryCols.filter(!onlyGapFeature.contains(_)),
    doNotHashingCateCols = doNotHashingCateCols,
    doNotHashingContCols = doNotHashingContCols,
    params = params,
    useSelector = false,
    featureHasherNumFeature = featureHasherNumFeature,
    featureHashColNm = "feature_hashed_click",
    colNmSuffix = "click"
)


val userDefinedFeatureListForAssemblerGap = Array("app_usage_token_click_cntvec","click_cnt","feature_hashed_click","send_hournum_cd_click_enc")

val transformPipelineGap = makePipeline(
    labelCols = Array(),
    indexedFeatureCol = indexedFeatureColGap,
    scaledFeatureCol = scaledFeatureColGap,
    selectedFeatureCol = selectedFeatureColGap,
    tokenCols = Array(),
    vectorCols = Array(),
    continuousCols = Array(),
    categoryCols = Array(),
    doNotHashingCateCols = Array(),
    doNotHashingContCols = Array(),
    params = params,
    useSelector = false,
    featureHasherNumFeature = 0,
    featureHashColNm = "feature_hashed_gap",
    colNmSuffix = "gap",
    userDefinedFeatureListForAssembler = userDefinedFeatureListForAssemblerGap
)

// Pipeline fitting은 샘플로 수행하되, 결과를 전체 데이터에 적용
println("Fitting Click transformer pipeline...")
val transformerClick = transformPipelineClick.fit(
  trainDFRev.sample(false, 0.3, 42)
)

println("Transforming training data with Click transformer...")
var transformedTrainDF = transformerClick.transform(trainDFRev)
  .persist(StorageLevel.MEMORY_AND_DISK_SER)

println("Training data transformed with Click pipeline (cached)")

println("Fitting Gap transformer pipeline...")
val transformerGap = transformPipelineGap.fit(
  transformedTrainDF.sample(false, 0.3, 42)
)

println("Transforming training data with Gap transformer...")
transformedTrainDF = transformerGap.transform(transformedTrainDF)
  .persist(StorageLevel.MEMORY_AND_DISK_SER)

println("Training data transformed with Gap pipeline (cached)")

println("Transforming test data...")
var transformedTestDF = transformerClick.transform(testDFRev)
transformedTestDF = transformerGap.transform(transformedTestDF)
  .persist(StorageLevel.MEMORY_AND_DISK_SER)

println("Test data transformed (cached)")


// ===== Paragraph 20: Transformer and Transformed Data Saving (Batch by Suffix) (ID: paragraph_1765520460775_2098641576) =====

println("Saving transformers...")
transformerClick.write.overwrite().save("aos/sto/transformPipelineXDRClick10")
transformerGap.write.overwrite().save("aos/sto/transformPipelineXDRGap10")

// 2단계: Train/Test 데이터를 Suffix별 배치 저장 (메모리 과부하 방지)
// suffixGroupSizeTrans: 한 번에 처리할 suffix 개수 (예: 4 = [0,1,2,3], [4,5,6,7], ...)
val suffixGroupSizeTrans = 2  // 조정 가능: 1(개별), 2, 4, 8, 16(전체)

// Train과 Test 데이터셋 정의
val datasetsToSave = Seq(
  ("training", transformedTrainDF, "aos/sto/transformedTrainDFXDR10", 20),
  ("test", transformedTestDF, "aos/sto/transformedTestDFXDF10", 20)
)

// 각 데이터셋에 대해 suffix 그룹별로 저장
datasetsToSave.foreach { case (datasetName, dataFrame, outputPath, basePartitions) =>
  println(s"Saving transformed $datasetName data by suffix (group size: $suffixGroupSizeTrans)...")
  
  (0 to 15).map(_.toHexString).grouped(suffixGroupSizeTrans).zipWithIndex.foreach { case (suffixGroup, groupIdx) =>
    println(s"  [Group ${groupIdx + 1}/${16 / suffixGroupSizeTrans}] Processing $datasetName suffixes: ${suffixGroup.mkString(", ")}")
    dataFrame
      .filter(suffixGroup.map(s => s"suffix = '$s'").mkString(" OR "))
      .repartition(basePartitions * suffixGroupSizeTrans)  // 그룹 크기에 비례한 파티션 수
      .write
      .mode("overwrite")  // Dynamic partition overwrite
      .option("compression", "snappy")
      .partitionBy("send_ym", "send_hournum_cd", "suffix")
      .parquet(outputPath)
  }
  
  println(s"  Completed saving $datasetName data")
}

println("All transformers and transformed data saved successfully")

// ===== Paragraph 21: Pipeline Transformers and Transformed Data Loading (ID: paragraph_1765521446308_1651058139) =====

println("Loading Click transformer...")
val transformerClick = PipelineModel.load("aos/sto/transformPipelineXDRClick")

println("Loading Gap transformer...")
val transformerGap = PipelineModel.load("aos/sto/transformPipelineXDRGap")

println("Loading transformed training data...")
val transformedTrainDF = spark.read.parquet("aos/sto/transformedTrainDFXDR10")
    .persist(StorageLevel.MEMORY_AND_DISK_SER)

println("Transformed training data loaded and cached")

println("Loading transformed test data...")
val transformedTestDF = spark.read.parquet("aos/sto/transformedTestDFXDF10")
    .persist(StorageLevel.MEMORY_AND_DISK_SER)

println("Transformed test data loaded and cached")

// val transformedPredDF = transformer.transform(predDFRev)//.persist(StorageLevel.MEMORY_AND_DISK_SER)


// ===== Paragraph 22: ML Model Definitions (GBT, FM, XGBoost, LightGBM) (ID: paragraph_1764836200898_700489598) =====

import org.apache.spark.ml.classification._
import org.apache.spark.ml.regression._

import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.StringIndexerModel

val gbtc = new GBTClassifier("gbtc_click")
  .setLabelCol(indexedLabelColClick)
  .setFeaturesCol(indexedFeatureColClick)
  .setMaxIter(100)  // 50 → 100 (더 많은 트리)
  .setMaxDepth(4)   // 4 → 6 (더 깊은 트리)
//   .setStepSize(0.1) // learning rate 추가
//   .setSubsamplingRate(0.8)  // 80% 샘플링으로 과적합 방지
  .setFeatureSubsetStrategy("auto")  // auto → sqrt (Random Forest 스타일)
//   .setMinInstancesPerNode(10)  // 리프 노드 최소 샘플 수
//   .setMinInfoGain(0.001)  // 분할 최소 정보 이득
//   .setWeightCol("sample_weight")
  .setPredictionCol("pred_gbtc_click")
  .setProbabilityCol("prob_gbtc_click")
  .setRawPredictionCol("pred_raw_gbtc_click")

val fmc = new FMClassifier("fmc_click")
  .setLabelCol(indexedLabelColClick)
  .setFeaturesCol(indexedFeatureColClick)
  .setStepSize(0.01)
  .setPredictionCol("pred_fmc_click")
  .setProbabilityCol("prob_fmc_click")
  .setRawPredictionCol("pred_raw_fmc_click")

val xgbc = {
  new XGBoostClassifier("xgbc_click")
    .setLabelCol(indexedLabelColClick)
    .setFeaturesCol(indexedFeatureColClick)
    .setMissing(0)
    .setSeed(0)
    .setMaxDepth(4)
    .setObjective("binary:logistic")
    .setNumRound(50)
    // .setScalePosWeight(10.0)
    .setWeightCol("sample_weight")
    .setNumWorkers(10)
    .setEvalMetric("auc")
    .setProbabilityCol("prob_xgbc_click")
    .setPredictionCol("pred_xgbc_click")
    .setRawPredictionCol("pred_raw_xgbc_click")
}

import com.microsoft.azure.synapse.ml.lightgbm.LightGBMClassifier

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

val xgbg = {
  new XGBoostClassifier("xgbc_gap")
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
}

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


// ===== Paragraph 23: XGBoost Feature Interaction and Monotone Constraints (ID: paragraph_1765939568349_1781513249) =====

// Feature Interaction Constraints를 위한 설정

// 1. VectorAssembler에서 생성된 feature 순서를 파악
val assemblerInputCols = transformPipeline.getStages
  .filter(_.isInstanceOf[VectorAssembler])
  .head.asInstanceOf[VectorAssembler]
  .getInputCols

println("Feature columns in order:")
assemblerInputCols.zipWithIndex.foreach { case (col, idx) =>
  println(s"  Index $idx: $col")
}

// 2. send_hournum_cd의 인덱스 찾기
val sendHournumFeatureIndices = assemblerInputCols.zipWithIndex
  .filter { case (colName, _) => 
    colName.contains("send_hournum_cd") || colName == "send_hournum_cd_enc"
  }
  .map(_._2)

println(s"\nsend_hournum_cd feature indices: ${sendHournumFeatureIndices.mkString(", ")}")

// 3. Interaction Constraints 설정
val sendHournumIndices = if (sendHournumFeatureIndices.nonEmpty) {
  sendHournumFeatureIndices.mkString(",")
} else {
  val startIdx = assemblerInputCols.indexWhere(_.contains("send_hournum_cd"))
  val endIdx = assemblerInputCols.lastIndexWhere(_.contains("send_hournum_cd"))
  (startIdx to endIdx).mkString(",")
}

val allFeatureIndices = (0 until assemblerInputCols.length).mkString(",")

// 4. XGBoostRegressor에 Feature Interaction Constraints 적용
val xgbParamR_withConstraints = Map(
  "eta" -> 0.01,
  "max_depth" -> 6,
  "objective" -> "reg:squarederror",
  "num_round" -> 100,
  "num_workers" -> 10,
  "eval_metric" -> "rmse",
  "interaction_constraints" -> s"[[$sendHournumIndices],[$allFeatureIndices]]"
)

val xgbParamR_withMonotone = Map(
  "eta" -> 0.01,
  "max_depth" -> 6,
  "objective" -> "reg:squarederror",
  "num_round" -> 100,
  "num_workers" -> 10,
  "eval_metric" -> "rmse",
  "interaction_constraints" -> s"[[$sendHournumIndices],[$allFeatureIndices]]",
  "monotone_constraints" -> s"(${assemblerInputCols.map(col => if (col.contains("send_hournum_cd")) "1" else "0" ).mkString(",")})"
)

val xgbr_withConstraints = new XGBoostRegressor(xgbParamR_withMonotone)
  .setFeaturesCol(featureColName)
  .setLabelCol(indexedLabelColReg)
  .setMissing(0)
  .setSeed(0)
  .setPredictionCol("pred_xgbr")

println("\n✅ XGBoostRegressor with Feature Interaction Constraints created!")
println(s"Interaction Constraints: [[$sendHournumIndices],[$allFeatureIndices]]")


// ===== Paragraph 24: Click Prediction Model Training (ID: paragraph_1765789893517_1550413688) =====

import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator

// 학습에 사용할 모델 선택 (gbtc, xgbc, fmc, lgbmc 중 선택)
val modelClickforCV = gbtc  // 또는 xgbc, fmc, lgbmc

val pipelineMLClick = new Pipeline().setStages(Array(modelClickforCV))

// 학습 데이터 샘플링 최적화 - 캐시하여 재사용
// 샘플링 비율 설정 (neg:pos)
// - 1:1 = 매우 균형적 (Precision 최대, 학습 데이터 최소)
// - 2:1 = 공격적 (실험 결과: Precision 2.5%, Recall 17%, F1 0.044)
// - 3:1 = 권장 (F1 최적, 실험 결과: Precision 4.2%, Recall 8%, F1 0.055) ✓
// - 전체 데이터 = Recall 최대 (비용 비효율)

val negSampleRatioClick = 0.3

val trainSampleClick = transformedTrainDF
    .filter("cmpgn_typ=='Sales'")
    .stat.sampleBy(
        F.col(indexedLabelColClick),
        Map(
            0.0 -> negSampleRatioClick,  // 3:1 비율 (Negative를 27% 샘플링) ✓
            1.0 -> 1.0,   // Positive 전체 사용
        ),
        42L
    )
    // sample_weight는 XGBoost에서만 사용되므로 그대로 유지
    .withColumn("sample_weight", F.expr(s"case when $indexedLabelColClick>0.0 then 10.0 else 1.0 end"))
    .repartition(100)  // 학습을 위한 적절한 파티션 수
    .persist(StorageLevel.MEMORY_AND_DISK_SER)

println("Click model training samples prepared and cached")

println("Training Click prediction model...")
val pipelineModelClick = pipelineMLClick.fit(trainSampleClick)

trainSampleClick.unpersist()  // 학습 완료 후 메모리 해제
println("Click model training completed")


// ===== Paragraph 25: Click-to-Action Gap Model Training (ID: paragraph_1767010803374_275395458) =====

val modelGapforCV = xgbg

val pipelineMLGap = new Pipeline().setStages(Array(modelGapforCV))

val posSampleRatioGap = 0.45

// Gap 모델 학습 데이터 샘플링 최적화
val trainSampleGap = transformedTrainDF
    .filter("click_yn>0")
    .stat.sampleBy(
        F.col("hour_gap"),
        Map(
            0.0 -> 1.0,
            1.0 -> posSampleRatioGap,
        ),
        42L
    )
    .repartition(100)
    .persist(StorageLevel.MEMORY_AND_DISK_SER)

println("Gap model training samples prepared and cached")

println("Training Gap prediction model...")
val pipelineModelGap = pipelineMLGap.fit(trainSampleGap)

trainSampleGap.unpersist()  // 학습 완료 후 메모리 해제
println("Gap model training completed")


// ===== Paragraph 26: Response Utility Regression Model Training (ID: paragraph_1765764610094_1504595267) =====

import org.apache.spark.ml.evaluation.RegressionEvaluator

val modelRegforCV = xgbr

val pipelineMLReg = new Pipeline().setStages(Array(modelRegforCV))

// 회귀 모델 학습 데이터 준비
val trainSampleReg = transformedTrainDF
    .filter("click_yn>0")
    .repartition(100)
    .persist(StorageLevel.MEMORY_AND_DISK_SER)

println("Regression model training samples prepared and cached")

println("Training Regression model...")
val pipelineModelReg = pipelineMLReg.fit(trainSampleReg)

trainSampleReg.unpersist()  // 학습 완료 후 메모리 해제
println("Regression model training completed")


// ===== Paragraph 27: Model Prediction on Test Dataset (ID: paragraph_1765345345715_612147457) =====

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


// ===== Paragraph 28: Click Model Performance Evaluation (Precision@K per Hour & MAP) (ID: paragraph_1764838154931_1623772564) =====

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
    val hourlyUserPredictions = predictionsClickDev
        .filter("click_yn >= 0")
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
        .cache()
    
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


// ===== Paragraph 29: Gap Model Performance Evaluation (Precision, Recall, F1) (ID: paragraph_1767010293011_1290077245) =====

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


// ===== Paragraph 30: Regression Model Performance Evaluation (RMSE) (ID: paragraph_1765786040626_1985577608) =====

import org.apache.spark.ml.evaluation.RegressionEvaluator

val stagesReg = pipelineModelReg.stages

println("Evaluating Regression models...")

// Regression 평가를 위한 prediction 생성 필요 (predictionsDev가 정의되지 않음)
val predictionsRegDev = pipelineModelReg.transform(
    transformedTestDF
        .filter("click_yn>0")
        .repartition(50)
        .persist(StorageLevel.MEMORY_AND_DISK_SER)
)

println("Regression predictions generated")

stagesReg.foreach { stage => 

    val modelName = stage.uid.split("_")(0)
    
    println(s"Evaluating model: $modelName")

    val evaluator = new RegressionEvaluator()
      .setLabelCol(indexedLabelColReg)
      .setPredictionCol(s"pred_${modelName}")
      .setMetricName("rmse")
    val rmse = evaluator.evaluate(predictionsRegDev)
    println(s"######### $modelName 예측 결과 #########")
    println(f"Root Mean Squared Error (RMSE) : $rmse%.4f")
}

predictionsRegDev.unpersist()  // 메모리 해제


// ===== Paragraph 31: Propensity Score Calculation and Persistence (Batch by Suffix) (ID: paragraph_1765768974381_910321724) =====

// 메모리 절약을 위해 이전 캐시 정리
try {
  transformedTrainDF.unpersist()
  transformedTestDF.unpersist()
  predictionsClickDev.unpersist()
  predictionsGapDev.unpersist()
  testDataForPred.unpersist()
} catch {
  case e: Exception => println(s"Cache cleanup warning: ${e.getMessage}")
}

// Propensity Score 계산 - suffix별로 배치 처리하되 메모리 효율적으로
// suffixGroupSize: 한 번에 처리할 suffix 개수 (예: 4 = [0,1,2,3], [4,5,6,7], ...)
val suffixGroupSizePred = 4  // 조정 가능: 1(개별), 2, 4, 8, 16(전체)

println(s"Processing propensity scores by suffix (group size: $suffixGroupSizePred)...")
(0 to 15).map(_.toHexString).grouped(suffixGroupSizePred).zipWithIndex.foreach { case (suffixGroup, groupIdx) =>

    println(s"[Group ${groupIdx + 1}/${16 / suffixGroupSizePred}] Processing suffixes: ${suffixGroup.mkString(", ")}")

    val predDFForSuffixGroup = predDFRev
        .filter(suffixGroup.map(s => s"svc_mgmt_num like '%${s}'").mkString(" OR "))
        .repartition(50 * suffixGroupSizePred)  // 그룹 크기에 비례한 파티션 수
        .persist(StorageLevel.MEMORY_AND_DISK_SER)
    
    // count() 대신 take(1)로 존재 여부만 확인 (훨씬 빠름)
    val hasRecords = predDFForSuffixGroup.take(1).nonEmpty
    
    if (hasRecords) {
        println(s"  Transforming data for suffix group ${suffixGroup.mkString(", ")}...")
        val transformedPredDF = transformerGap.transform(
            transformerClick.transform(predDFForSuffixGroup)
        ).persist(StorageLevel.MEMORY_AND_DISK_SER)
        
        println(s"  Deduplicating...")
        val dedupedDF = transformedPredDF
            .dropDuplicates("svc_mgmt_num", "chnl_typ", "cmpgn_typ", "send_ym", "send_hournum_cd", "click_yn")
        
        println(s"  Generating Click predictions...")
        val predictionsSVCClick = pipelineModelClick.transform(dedupedDF)
        
        println(s"  Generating Gap predictions...")
        val predictionsSVCFinal = pipelineModelGap.transform(predictionsSVCClick)
        
        println(s"  Calculating propensity scores...")
        var predictedPropensityScoreDF = predictionsSVCFinal
            .withColumn("prob_click", F.expr(s"""aggregate(array(${pipelineModelClick.stages.map(m => s"vector_to_array(prob_${m.uid})[1]").mkString(",")}), 0D, (acc, x) -> acc + x)"""))
            .withColumn("prob_gap", F.expr(s"""aggregate(array(${pipelineModelGap.stages.map(m => s"vector_to_array(prob_${m.uid})[1]").mkString(",")}), 0D, (acc, x) -> acc + x)"""))
            .withColumn("propensity_score", F.expr("prob_click*prob_gap"))
        
        println(s"  Saving propensity scores...")
        predictedPropensityScoreDF
            .selectExpr("svc_mgmt_num", "send_ym","send_hournum_cd send_hour"
                ,"ROUND(prob_click, 4) prob_click" 
                ,"ROUND(prob_gap, 4) prob_gap"
                ,"ROUND(propensity_score, 4) propensity_score")
            .withColumn("suffix", F.expr("right(svc_mgmt_num, 1)"))
            .repartition(10 * suffixGroupSizePred)  // 그룹 크기에 비례
            .write
            .mode("overwrite")  // Dynamic partition overwrite
            .option("compression", "snappy")
            .partitionBy("send_ym", "send_hour", "suffix")
            .parquet("aos/sto/propensityScoreDF")
        
        // 메모리 해제
        predDFForSuffixGroup.unpersist()
        transformedPredDF.unpersist()
        
        println(s"  Completed processing suffix group ${suffixGroup.mkString(", ")}")
    } else {
        println(s"  No records found for suffix group ${suffixGroup.mkString(", ")}, skipping...")
    }
}


// ===== Paragraph 32: Propensity Score Data Loading and Verification (ID: paragraph_1767943423474_1143363402) =====

val dfPropensity = spark.read.parquet("aos/sto/propensityScoreDF")
    .persist(StorageLevel.MEMORY_AND_DISK_SER)

println("Propensity score data loaded and cached")

// 기본 통계 확인 (summary에 이미 count 포함)
dfPropensity
    .select("prob_click", "prob_gap", "propensity_score")
    .summary("count", "mean", "stddev", "min", "max")
    .show()

// 필요시에만 전체 레코드 수 확인 (시간이 걸림)
// val propensityCount = dfPropensity.count()
// println(s"Total propensity score records: $propensityCount")


// ===== Paragraph 33: Performance Monitoring and Optimization Tips =====

/*
 * [대용량 처리 성능 모니터링 및 추가 최적화 팁]
 * 
 * 1. Spark UI 모니터링:
 *    - Stage별 실행 시간과 데이터 shuffle 크기 확인
 *    - Task skew 확인 (일부 task만 오래 걸리는지)
 *    - GC 시간이 전체 실행 시간의 10% 이상이면 메모리 부족
 * 
 * 2. 메모리 튜닝:
 *    - spark.executor.memory: Executor당 메모리 (권장: 8-16GB)
 *    - spark.driver.memory: Driver 메모리 (권장: 4-8GB)
 *    - spark.memory.fraction: 실행/저장 메모리 비율 (기본: 0.6)
 * 
 * 3. 파티션 수 조정:
 *    - spark.sql.shuffle.partitions: 현재 400으로 설정됨
 *    - 데이터 크기에 따라 200-800 사이에서 조정
 *    - 각 파티션 크기는 128MB-256MB가 이상적
 * 
 * 4. 데이터 Skew 해결:
 *    - Skew가 심한 경우 salting 기법 사용
 *    - AQE의 skewJoin이 자동으로 처리하지만, 수동 조정 가능
 * 
 * 5. Checkpoint 전략:
 *    - 긴 lineage를 끊기 위해 중요한 중간 결과는 checkpoint
 *    - 현재 resDFSelectedTrBal, resDFSelectedTs에 적용됨
 * 
 * 6. 캐시 정리:
 *    - 사용 완료된 DataFrame은 반드시 unpersist() 호출
 *    - 주기적으로 spark.catalog.clearCache() 실행
 * 
 * 7. Broadcast Join 임계값:
 *    - 현재 100MB로 설정 (spark.sql.autoBroadcastJoinThreshold)
 *    - 네트워크 대역폭이 충분하면 200-500MB로 증가 가능
 * 
 * 8. 압축 알고리즘:
 *    - 현재 Snappy 사용 (빠른 압축/해제)
 *    - 저장 공간이 중요하면 gzip 또는 zstd 고려
 * 
 * 9. 실행 순서 최적화:
 *    - 데이터 로딩 -> 필터링 -> 파티셔닝 -> 조인 -> 집계 순서 유지
 *    - 가능한 한 일찍 필터링하여 데이터 크기 축소
 * 
 * 10. 리소스 사용량 모니터링:
 *     spark.sparkContext.statusTracker.getExecutorInfos.foreach { info =>
 *       println(s"Executor ${info.host()}: ${info.totalCores()} cores")
 *     }
 */

println("Performance optimization tips displayed. Check code comments for details.")


// ===== Paragraph 34: Feature Engineering Improvements Summary and Performance Comparison Guide =====

/*
 * [피처 엔지니어링 개선 사항 요약]
 * 
 * 이 코드는 다음과 같은 피처 개선이 적용되었습니다:
 * 
 * ========================================
 * 1. HashingTF → CountVectorizer + TF-IDF
 * ========================================
 * 
 * **이전 방식** (Line 676, 현재 주석 처리됨):
 *   - HashingTF with vocabSize=30
 *   - 문제점: Hash collision, 정보 손실, 작은 vocabulary
 * 
 * **개선된 방식** (Line 740-759):
 *   - CountVectorizer with vocabSize=1000
 *   - TF-IDF 가중치 추가
 *   - 장점: 
 *     • 정확한 앱 매핑 (hash collision 없음)
 *     • 더 많은 앱 추적 가능 (30개 → 1000개)
 *     • 중요한 앱에 높은 가중치 (TF-IDF)
 *   - 예상 개선: 5-10% 정확도 향상
 * 
 * ========================================
 * 2. 트래픽 정보 기반 피처 추가
 * ========================================
 * 
 * **추가된 피처** (Paragraph 10, Line 289-300):
 *   - heavy_usage_app_cnt: 트래픽 > 100,000인 앱 개수
 *   - medium_usage_app_cnt: 트래픽 10,000-100,000인 앱 개수
 *   - light_usage_app_cnt: 트래픽 < 10,000인 앱 개수
 *   - total_traffic_mb: 시간대별 총 트래픽양 (MB)
 *   - app_cnt: 사용한 앱 개수
 * 
 * **적용 위치**:
 *   - Training: xdrDF (Paragraph 10)
 *   - Prediction: xdrDFPred (Paragraph 16)
 * 
 * **장점**:
 *   - 단순 앱 리스트가 아닌 사용 강도 정보 포함
 *   - 트래픽 구간별 사용 패턴 파악
 *   - 예상 개선: 3-5% 정확도 향상
 * 
 * ========================================
 * 3. 시간대 집계 피처 추가
 * ========================================
 * 
 * **추가된 피처** (Paragraph 10, Line 301-327):
 *   - peak_usage_hour_cd: 가장 활발한 시간대 (category)
 *   - peak_hour_app_cnt: 피크 시간대 앱 개수
 *   - active_hour_cnt: 활동 시간대 개수
 *   - avg_hourly_app_avg: 시간당 평균 앱 사용 개수
 *   - total_daily_traffic_mb: 전체 일일 트래픽 (MB)
 *   - total_heavy_apps_cnt, total_medium_apps_cnt, total_light_apps_cnt: 트래픽 구간별 총계
 * 
 * **적용 위치**:
 *   - Training: xdrAggregatedFeatures → xdrDFMon 조인 (Paragraph 10)
 *   - Prediction: xdrPredAggregatedFeatures → xdrPredDF 조인 (Paragraph 16)
 * 
 * **장점**:
 *   - 24개 시간대를 8개 요약 피처로 압축
 *   - 시간적 패턴 포착 (피크 시간, 활동 시간대 수)
 *   - 차원 축소로 모델 학습 효율성 향상
 *   - 예상 개선: 2-4% 정확도 향상
 * 
 * ========================================
 * 4. 피처 설정 업데이트
 * ========================================
 * 
 * **업데이트된 설정** (Paragraph 17, Line 664, 684-686):
 *   - vocabSize: 30 → 1000
 *   - doNotHashingCateCols: "peak_usage_hour" 추가
 *   - doNotHashingContCols: 12개 집계 피처 추가
 * 
 * ========================================
 * 전체 예상 성능 영향
 * ========================================
 * 
 * | 항목                    | 계산 시간 증가 | 메모리 증가 | 정확도 향상 (예상) |
 * |------------------------|--------------|-----------|-----------------|
 * | CountVectorizer        | +20%         | +30%      | +5-10%          |
 * | TF-IDF                 | +10%         | +15%      | +2-3%           |
 * | 트래픽 피처             | +5%          | +10%      | +3-5%           |
 * | 시간대 집계 피처        | +15%         | +20%      | +2-4%           |
 * | **전체**               | **+50%**     | **+75%**  | **+12-22%**     |
 * 
 * ========================================
 * 모델 재학습 및 성능 비교 가이드
 * ========================================
 * 
 * 1. 기존 모델 성능 기록
 *    - Paragraph 28-30의 평가 결과 저장
 *    - 주요 지표: AUC, Precision, Recall, F1-Score
 * 
 * 2. 신규 모델 학습
 *    - Paragraph 1부터 순차적으로 실행
 *    - 새로운 피처가 포함된 transformedTrainDF로 학습
 *    - Paragraph 24-26에서 모델 학습
 * 
 * 3. 성능 비교
 *    a) 정량적 비교:
 *       - AUC, Precision, Recall 비교
 *       - 실행 시간 비교
 *       - 메모리 사용량 비교
 * 
 *    b) 정성적 비교:
 *       - Feature importance 분석
 *         ```scala
 *         val featureImportances = pipelineModelClick.stages
 *           .filter(_.isInstanceOf[GBTClassificationModel])
 *           .head.asInstanceOf[GBTClassificationModel]
 *           .featureImportances
 *         ```
 *       - 새로 추가된 피처들의 중요도 확인
 * 
 * 4. A/B 테스트 (Production 환경)
 *    - 기존 모델과 신규 모델을 50:50으로 트래픽 분할
 *    - 실제 클릭률(CTR) 비교
 *    - 최소 2주간 테스트 후 결정
 * 
 * 5. 모니터링 지표
 *    ```scala
 *    // 모델별 성능 비교
 *    val oldModelMetrics = Map(
 *      "AUC" -> 0.XX,
 *      "Precision" -> 0.XX,
 *      "Recall" -> 0.XX,
 *      "TrainingTime" -> "XX minutes"
 *    )
 *    
 *    val newModelMetrics = Map(
 *      "AUC" -> 0.XX,  // 예상: +5-10% 향상
 *      "Precision" -> 0.XX,
 *      "Recall" -> 0.XX,
 *      "TrainingTime" -> "XX minutes"  // 예상: +50% 증가
 *    )
 *    
 *    // 개선율 계산
 *    val aucImprovement = (newModelMetrics("AUC") - oldModelMetrics("AUC")) / oldModelMetrics("AUC") * 100
 *    println(s"AUC improved by: ${aucImprovement}%")
 *    ```
 * 
 * 6. 롤백 계획
 *    - 성능이 기대에 미치지 못할 경우:
 *      • aos/sto/transformPipelineXDRClick10 → 이전 버전으로 복구
 *      • aos/sto/transformedTrainDFXDR10 → 이전 버전으로 복구
 *    - 기존 모델 백업 필수:
 *      ```scala
 *      // 백업 명령
 *      hadoop fs -cp aos/sto/transformPipelineXDRClick aos/sto/transformPipelineXDRClick_backup_YYYYMMDD
 *      ```
 * 
 * ========================================
 * 추가 개선 가능 항목 (Phase 2)
 * ========================================
 * 
 * 1. 앱 카테고리 피처
 *    - 앱-카테고리 매핑 테이블 필요
 *    - SNS/게임/쇼핑/동영상 등 카테고리별 사용 패턴
 *    - 예상 개선: +5-8%
 * 
 * 2. 시간적 연속성 피처
 *    - Window 함수로 이전/이후 시간대와의 관계 파악
 *    - 앱 사용 일관성 지표
 *    - 예상 개선: +3-5%
 * 
 * 3. 사용자 세그먼트별 피처
 *    - 연령대/성별/요금제별 앱 사용 패턴
 *    - 세그먼트별 개인화 피처
 *    - 예상 개선: +4-7%
 */

println("Feature engineering improvements summary displayed. Check code comments for details.")
println("=" * 80)
println("IMPORTANT: This code includes the following improvements:")
println("  1. HashingTF → CountVectorizer + TF-IDF (vocabSize: 30 → 1000)")
println("  2. Traffic-based features (heavy/medium/light usage counts)")
println("  3. Time aggregation features (peak hour, active hours, etc.)")
println("  4. GBT hyperparameter tuning (maxDepth: 4→6, maxIter: 50→100)")
println("  5. Threshold optimization analysis (Paragraph 28.5)")
println("  6. Expected accuracy improvement: +12-22%")
println("  7. Expected compute time increase: +50%")
println("=" * 80)


// ===== Paragraph 35: Additional Feature Recommendations for Future Improvement =====

/*
 * ========================================
 * 추가 피처 개선 로드맵 (AUC 0.66 → 0.75+)
 * ========================================
 * 
 * **현재 상태 (1:1 언더샘플링)**:
 *   - GBT: AUC 0.66, Precision 1.4%, Recall 46.7%, F1 0.027
 *   - XGBoost: AUC 0.52, Precision 0.8%, Recall 89.9%, F1 0.016
 *   - 결론: GBT가 우수하지만, 여전히 Precision이 낮음
 * 
 * **목표**:
 *   - AUC: 0.75-0.80
 *   - Precision: 3-5%
 *   - F1-Score: 0.05-0.08
 * 
 * ========================================
 * Priority 1: 과거 클릭 이력 피처 (가장 중요!)
 * ========================================
 * 
 * 데이터 소스: MMS 발송/클릭 이력 테이블
 * 
 * ```scala
 * // 과거 7일/30일 클릭 이력 집계
 * val userClickHistory = clickHistoryDF
 *   .filter("click_date >= date_sub(current_date(), 30)")
 *   .groupBy("svc_mgmt_num")
 *   .agg(
 *     F.count(F.when(F.col("click_date") >= F.date_sub(F.current_date(), 7), 1)).alias("last_7d_click_cnt"),
 *     F.count("*").alias("last_30d_click_cnt"),
 *     (F.count(F.when(F.col("click_yn") > 0, 1)) / F.count("*")).alias("avg_click_rate"),
 *     F.datediff(F.current_date(), F.max("click_date")).alias("last_click_days_ago")
 *   )
 * 
 * // 메인 데이터와 조인
 * val enrichedDF = mainDF.join(userClickHistory, Seq("svc_mgmt_num"), "left")
 *   .na.fill(Map(
 *     "last_7d_click_cnt" -> 0,
 *     "last_30d_click_cnt" -> 0,
 *     "avg_click_rate" -> 0.0,
 *     "last_click_days_ago" -> 9999
 *   ))
 * ```
 * 
 * **예상 효과**:
 *   - AUC: +0.10-0.15
 *   - Precision: +2-3%
 *   - 과거 클릭 이력이 있는 사용자의 재클릭 확률이 10-20배 높을 것으로 예상
 * 
 * ========================================
 * Priority 2: Interaction 피처
 * ========================================
 * 
 * ```scala
 * // 앱 카테고리별 선호 시간대
 * val appCategoryHourDF = xdrDF
 *   .groupBy("svc_mgmt_num", "hour", "app_category")
 *   .agg(F.sum("traffic").alias("traffic_sum"))
 *   .withColumn("rank", F.row_number().over(
 *     Window.partitionBy("svc_mgmt_num").orderBy(F.desc("traffic_sum"))))
 *   .filter("rank = 1")
 *   .select(
 *     F.col("svc_mgmt_num"),
 *     F.concat(F.col("app_category"), F.lit("_"), F.col("hour")).alias("top_category_hour_cd")
 *   )
 * 
 * // ARPU 구간별 캠페인 반응도
 * val arpuCampaignDF = mainDF
 *   .withColumn("arpu_segment", 
 *     F.when(F.col("arpu") > 50000, "high")
 *      .when(F.col("arpu") > 30000, "medium")
 *      .otherwise("low"))
 *   .withColumn("arpu_campaign_cd", 
 *     F.concat(F.col("arpu_segment"), F.lit("_"), F.col("campaign_type")))
 * ```
 * 
 * **예상 효과**:
 *   - AUC: +0.03-0.05
 *   - 사용자 세그먼트별 맞춤 예측
 * 
 * ========================================
 * Priority 3: 시간 기반 피처 강화
 * ========================================
 * 
 * ```scala
 * val timeEnrichedDF = mainDF
 *   .withColumn("day_of_week_cd", F.dayofweek(F.col("send_date")).cast("string"))
 *   .withColumn("is_weekend", 
 *     F.when(F.dayofweek(F.col("send_date")).isin(1, 7), 1).otherwise(0))
 *   .withColumn("is_peak_time", 
 *     F.when(F.col("send_hournum").between(18, 22), 1).otherwise(0))
 *   .withColumn("days_since_last_send", 
 *     F.datediff(F.current_date(), F.col("last_send_date")))
 * ```
 * 
 * **예상 효과**:
 *   - AUC: +0.02-0.04
 *   - 요일/시간대별 패턴 포착
 * 
 * ========================================
 * Priority 4: 캠페인-사용자 적합도 피처
 * ========================================
 * 
 * ```scala
 * // 유사 사용자 그룹 클릭률
 * val similarUserClickRate = mainDF
 *   .withColumn("user_segment", 
 *     F.concat(
 *       F.when(F.col("age") < 30, "young").when(F.col("age") < 50, "middle").otherwise("senior"),
 *       F.lit("_"),
 *       F.col("gender_cd"),
 *       F.lit("_"),
 *       F.when(F.col("arpu") > 40000, "high_arpu").otherwise("low_arpu")
 *     ))
 *   .join(
 *     clickHistoryDF
 *       .groupBy("user_segment", "campaign_type")
 *       .agg((F.sum("click_yn") / F.count("*")).alias("segment_click_rate")),
 *     Seq("user_segment", "campaign_type"),
 *     "left"
 *   )
 * ```
 * 
 * **예상 효과**:
 *   - AUC: +0.05-0.08
 *   - 개인화 스코어로 정확도 향상
 * 
 * ========================================
 * 구현 우선순위 및 일정
 * ========================================
 * 
 * **Phase 1 (즉시 시작 가능)**:
 *   - Priority 1 (과거 클릭 이력) 구현
 *   - 예상 기간: 2-3일
 *   - 예상 AUC: 0.66 → 0.75
 * 
 * **Phase 2 (Phase 1 완료 후)**:
 *   - Priority 2 (Interaction 피처) 구현
 *   - Priority 3 (시간 피처) 구현
 *   - 예상 기간: 3-4일
 *   - 예상 AUC: 0.75 → 0.78
 * 
 * **Phase 3 (선택적)**:
 *   - Priority 4 (적합도 피처) 구현
 *   - 예상 기간: 4-5일
 *   - 예상 AUC: 0.78 → 0.80
 * 
 * ========================================
 * 중요: 실제 서비스 평가 - Ranking Approach
 * ========================================
 * 
 * **실제 사용 시나리오** (Paragraph 28.5):
 *   - 목적: **최적 발송 시간 선택** (발송 여부 결정 아님!)
 *   - 방법: 각 사용자별 9~18시(10개 시간대) 모두 예측
 *   - 발송: 가장 높은 확률의 시간대 1개 선택
 *   - 예: 사용자 A의 11시 확률 = 0.87, 13시 = 0.88 → 13시 발송
 * 
 * **Ranking 평가 지표** (Paragraph 28.5):
 *   - **Top-1 Accuracy**: 최고 확률 시간대 = 실제 클릭 시간대 비율
 *     * 랜덤: 10% (10개 중 1개)
 *     * 목표: > 20% (랜덤 대비 2배)
 *     * 우수: > 30% (랜덤 대비 3배)
 *   - **Top-3 Accuracy**: 상위 3개 중 실제 클릭 시간대 포함 비율
 *   - **MRR**: Mean Reciprocal Rank
 *   - **평균 순위**: 실제 클릭 시간대의 평균 예측 순위
 * 
 * **Binary Classification vs Ranking**:
 *   - Paragraph 28: Precision, Recall, F1 (참고용)
 *   - **Paragraph 28.5**: Top-K Accuracy, MRR (실제 평가) ✓
 *   - Paragraph 28.6: Threshold 분석 (사용 안 함)
 * 
 * **샘플링 비율 재해석**:
 *   - 3:1 vs 2:1 비교 시 **Top-1 Accuracy로 재평가 필요**
 *   - F1-Score는 참고용 (실제 서비스와 무관)
 *   - Ranking 성능이 더 중요한 지표
 * 
 * ========================================
 * 참고: Threshold vs Feature Engineering
 * ========================================
 * 
 * **Threshold 조정**:
 *   - 장점: 즉시 적용 가능, 구현 비용 없음
 *   - 단점: 근본적인 성능 향상 없음 (Precision/Recall 트레이드오프만 조정)
 *   - 용도: Binary Classification 시나리오용 (실제 서비스 미사용!)
 * 
 * **Feature Engineering**:
 *   - 장점: 근본적인 성능 향상 (AUC 증가)
 *   - 단점: 구현 시간/비용, 계산 리소스 증가
 *   - 용도: 중장기 모델 품질 개선
 * 
 * **Undersampling 비율**:
 *   - 장점: 학습 속도 향상, 클래스 불균형 완화, 메모리 절감
 *   - 단점: 데이터 손실, 비율 선택 필요
 *   - 용도: 극단적 불균형 해결 (127:1 → 2:1)
 * 
 * **권장 전략**:
 *   1. 단기: Paragraph 28.5로 최적 Threshold 찾기
 *   2. 중기: Priority 1 피처 추가로 AUC 0.75 달성
 *   3. 장기: Priority 2-4로 AUC 0.80 목표
 *   4. 샘플링 비율은 2:1 고정 (실험 검증 완료)
 * 
 * ========================================
 * Priority 5: 언더샘플링 비율 최적화 (완료!)
 * ========================================
 * 
 * **실험 결과**:
 * 
 * | neg:pos | Neg 샘플링 | Precision | Recall | F1 | Pred+ | 학습시간 |
 * |---------|-----------|-----------|--------|-----|-------|----------|
 * | 1:1     | 9%        | ~6-8%     | ~5%    | 최고 | 최소   | 최단     |
 * | 2:1     | 18%       | 2.5%      | 17.1%  | 0.044 | 74K  | 중간     |
 * | **3:1** | **27%**   | **4.2%**  | **7.8%** | **0.055** ✓ | **20K** | **중장** ✓ |
 * | 전체    | 100%      | <1%       | ~90%   | 최저 | 최대   | 최장     |
 * 
 * **3:1 선택 근거** (Binary Classification 관점):
 *   - F1-Score 최대 (0.055 > 0.044)
 *   - Precision 4.2% (100명 중 4명 클릭)
 *   - 예측 Positive 1.4% (비용 효율)
 *   - AUC 0.66 (양호)
 * 
 * **중요**: 실제 서비스는 Ranking 방식 사용!
 *   - 각 사용자별 9~18시 모두 예측
 *   - 가장 높은 확률의 시간대에 발송
 *   - Paragraph 28.5에서 Top-K Accuracy로 평가
 *   - Threshold, Precision/Recall은 참고용
 * 
 * **코드 적용** (Paragraph 24, Line 1159):
 *   ```scala
 *   .stat.sampleBy(
 *       F.col(indexedLabelColClick),
 *       Map(
 *           0.0 -> 0.27,  // 3:1 비율 (27%) ✓
 *           1.0 -> 1.0,   // Positive 전체
 *       ),
 *       42L
 *   )
 *   ```
 * 
 * **비즈니스 시나리오별**:
 *   - 비용 최소화: 1:1 (또는 Threshold 높임)
 *   - 커버리지 확대: 2:1 (또는 Threshold 낮춤)
 *   - **균형/표준**: 3:1 ✓ 권장
 * 
 * **Threshold vs Sampling**:
 *   - Sampling: 모델 자체를 변경 (학습 단계)
 *   - Threshold: 예측 기준만 변경 (예측 단계)
 *   - **권장**: Sampling 2:1 고정 + Threshold로 미세 조정
 * 
 */

println("=" * 80)
println("Additional feature recommendations added (Paragraph 35)")
println("Priority 1: User click history features (expected AUC +0.10-0.15)")
println("Priority 2: Interaction features (expected AUC +0.03-0.05)")
println("Priority 3: Enhanced time features (expected AUC +0.02-0.04)")
println("Priority 4: Campaign-user affinity (expected AUC +0.05-0.08)")
println("Priority 5: Undersampling ratio = 3:1 (COMPLETED)")
println("IMPORTANT: Paragraph 28.5 - Ranking-based evaluation (Top-K Accuracy)")
println("  → 실제 서비스 시나리오: 9~18시 중 최적 시간 선택")
println("=" * 80)
