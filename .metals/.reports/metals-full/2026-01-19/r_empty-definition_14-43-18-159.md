error id: file://<WORKSPACE>/optimize_send_time/predict_ost_zpln.scala:filter.
file://<WORKSPACE>/optimize_send_time/predict_ost_zpln.scala
empty definition using pc, found symbol in pc: filter.
empty definition using semanticdb
empty definition using fallback
non-local guesses:

offset: 14856
uri: file://<WORKSPACE>/optimize_send_time/predict_ost_zpln.scala
text:
```scala
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

// ===== Paragraph 1: Preps (Imports and Configuration) =====

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
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "100MB")
spark.conf.set("spark.sql.shuffle.partitions", "400")
spark.conf.set("spark.sql.files.maxPartitionBytes", "128MB")

spark.sparkContext.setCheckpointDir("hdfs://scluster/user/g1110566/checkpoint")


// ===== Paragraph 2: Helper Functions =====

import java.time.YearMonth
import java.time.format.DateTimeFormatter
import java.time.LocalDate
import scala.collection.mutable.ListBuffer
import java.time.temporal.ChronoUnit

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


// ===== Paragraph 3: Response Data Loading - HDFS =====

// 대용량 데이터는 MEMORY_AND_DISK_SER 사용하여 메모리 절약
val resDF = spark.read.parquet("aos/sto/response")
  .persist(StorageLevel.MEMORY_AND_DISK_SER)
resDF.createOrReplaceTempView("res_df")

resDF.printSchema()


// ===== Paragraph 4: Dates Setting =====

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


// ===== Paragraph 5: Filtering Response History Data =====

val resDFFiltered = resDF
    .filter(s"""send_ym in (${sendYmList.mkString("'","','","'")})""")
    .filter(s"hour_gap is null or (hour_gap between 0 and 5)")
    .filter(s"send_hournum between $startHour and $endHour")
    .selectExpr("cmpgn_num","svc_mgmt_num","chnl_typ","cmpgn_typ","send_ym","send_dt","send_time","send_daynum","send_hournum","click_dt","click_time","click_daynum","click_hournum","case when hour_gap is null then 0 else 1 end click_yn","hour_gap")
    .withColumn("res_utility", F.expr(s"case when hour_gap is null then 0.0 else 1.0 / (1 + hour_gap) end"))
    .dropDuplicates()
    .repartition(200, F.col("svc_mgmt_num"))  // 조인을 위해 적절한 파티션 수로 재분배
    .persist(StorageLevel.MEMORY_AND_DISK_SER)

resDFFiltered.count()  // 캐시 구체화
println(s"Filtered response records: ${resDFFiltered.count()}")

resDF.printSchema()


// ===== Paragraph 6: Loading User Info Data =====

val allFeaturesMMKT = spark.sql("describe wind_tmt.mmkt_svc_bas_f").select("col_name").collect().map(_.getString(0))
val sigFeaturesMMKT = spark.read.option("header", "true").csv("feature_importance/table=mmkt_bas/creation_dt=20230407").filter("rank<100").select("col").collect().map(_(0).toString()).map(_.trim)
val collistForMMKT = (Array("svc_mgmt_num", "strd_ym feature_ym", "mst_work_dt", "cust_birth_dt", "prcpln_last_chg_dt", "fee_prod_id", "eqp_mdl_cd", "eqp_acqr_cnt", "equip_chg_cnt", "svc_scrb_dt", "chg_dt", "cust_age_cd", "sex_cd", "equip_chg_day", "last_equip_chg_dt", "prev_equip_chg_dt", "term_pen_amt", "agrmt_brrh_amt",
    "allot_mth_cnt", "mbr_use_cnt", "mbr_use_amt", "tyr_mbr_use_amt", "mth_cnsl_cnt", "dsst_cnsl_cnt", "simpl_ref_cnsl_cnt", "arpu", "bf_m1_arpu", "voc_arpu", "bf_m3_avg_arpu",
    "tfsly_nh39_scrb_yn", "prcpln_chg_cnt", "email_inv_yn", "cpn_use_psbl_cnt", "data_gift_send_yn", "data_gift_recv_yn", "equip_chg_mth_cnt", "dom_tot_pckt_cnt", "scrb_sale_chnl_cl_cd", "op_sale_chnl_cl_cd", "agrmt_dc_end_dt",
    "svc_cd", "svc_st_cd", "pps_yn", "svc_use_typ_cd", "lndp_corp_cl_cd", "frgrn_yn", "mn_cust_num", "wlf_dc_cd"
) ++ sigFeaturesMMKT.filter(c => allFeaturesMMKT.contains(c.trim.split(" ")(0).trim))).distinct

val mmktDFTemp = spark.sql(s"""select ${collistForMMKT.mkString(",")}, strd_ym from wind_tmt.mmkt_svc_bas_f a where strd_ym in (${featureYmList.mkString("'","','","'")})""")
  .repartition(200, F.col("svc_mgmt_num"))  // 조인을 위한 파티셔닝

// Broadcast join for small dimension table
val prodDF = spark.sql("select prod_id fee_prod_id, prod_nm fee_prod_nm from wind.td_zprd_prod")

val mmktDF = {
  mmktDFTemp
    .join(F.broadcast(prodDF), Seq("fee_prod_id"), "left")
    .filter("cust_birth_dt not like '9999%'")
    .persist(StorageLevel.MEMORY_AND_DISK_SER)
}

mmktDF.count()  // 캐시 구체화
println(s"MMKT user records: ${mmktDF.count()}")


// ===== Paragraph 7: Training and Test Dataset Setting =====

val predictionDTSta = "20251101"
val predictionDTEnd = "20251201"

// 중복 제거 전 파티셔닝으로 shuffle 최적화
val resDFSelected = resDFFiltered
    .withColumn("feature_ym", F.date_format(F.add_months(F.unix_timestamp($"send_dt", "yyyyMMdd").cast(TimestampType), -1), "yyyyMM").cast(StringType))
    .selectExpr("cmpgn_num", "svc_mgmt_num", "chnl_typ", "cmpgn_typ", "send_ym", "send_dt", "feature_ym", "send_daynum", "send_hournum send_hournum_cd", "hour_gap", "click_yn", "res_utility")
    .repartition(200, F.col("svc_mgmt_num"), F.col("chnl_typ"), F.col("cmpgn_typ"), F.col("send_ym"), F.col("send_hournum_cd"))
    .dropDuplicates("svc_mgmt_num", "chnl_typ", "cmpgn_typ", "send_ym", "send_hournum_cd", "click_yn")
    .persist(StorageLevel.MEMORY_AND_DISK_SER)

resDFSelected.count()  // 캐시 구체화
println(s"Selected response records: ${resDFSelected.count()}")

val resDFSelectedTr = resDFSelected
    .filter(s"send_dt<$predictionDTSta")
    .persist(StorageLevel.MEMORY_AND_DISK_SER)

resDFSelectedTr.count()  // 캐시 구체화
println(s"Training response records: ${resDFSelectedTr.count()}")

var resDFSelectedTs = resDFSelected
    .filter(s"send_dt>=$predictionDTSta and send_dt<$predictionDTEnd")
    .selectExpr("cmpgn_num", "svc_mgmt_num", "chnl_typ", "cmpgn_typ", "send_ym", "send_dt", "feature_ym", "send_hournum_cd", "click_yn", "res_utility")
    .checkpoint()

resDFSelectedTs.count()  // 체크포인트 구체화
println(s"Test response records: ${resDFSelectedTs.count()}")


// ===== Paragraph 8: Sampling Ratio for Undersampling Training Dataset =====

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

samplingRatioMapDF.count()  // 캐시 구체화
println(s"Sampling ratio map size: ${samplingRatioMapDF.count()}")


// ===== Paragraph 9: Undersampling Training Dataset =====

// Undersampling 최적화 - broadcast join 활용
var resDFSelectedTrBal = resDFSelectedTr
    .repartition(200, F.col("svc_mgmt_num"))  // 후속 조인을 위한 파티셔닝
    .withColumn("sampling_col", F.expr(s"""concat_ws('-', ${samplingKeyCols.mkString(",")})"""))
    .join(F.broadcast(samplingRatioMapDF), "sampling_col")
    .withColumn("rand", F.rand(42))  // 시드 추가로 재현성 확보
    .filter("rand<=ratio")
    .checkpoint()

resDFSelectedTrBal.count()  // 체크포인트 구체화
println(s"Balanced training records: ${resDFSelectedTrBal.count()}")


// ===== Paragraph 10: Loading App Usage Dataset (Big) =====

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

userYmDF.count()  // 캐시 구체화
println(s"Unique user-ym pairs: ${userYmDF.count()}")

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
.agg(collect_set("app_nm").alias("app_usage_token"))  // collect_set은 메모리 효율적으로 사용
.persist(StorageLevel.MEMORY_AND_DISK_SER)

xdrDF.count()  // 캐시 구체화
println(s"XDR hourly records: ${xdrDF.count()}")

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
    .persist(StorageLevel.MEMORY_AND_DISK_SER)

xdrDFMon.count()  // 캐시 구체화
println(s"XDR monthly records: ${xdrDFMon.count()}")


// ===== Paragraph 11: Making Click Count History from Response History Data =====

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

clickCountDF.count()  // 캐시 구체화
println(s"Click count records: ${clickCountDF.count()}")


// ===== Paragraph 12: Building Training and Test Data by Joining Feature DF =====

val mmktDFFiltered = mmktDF.fi@@lter(smnCond).persist(StorageLevel.MEMORY_AND_DISK_SER)
println(s"Filtered MMKT records: ${mmktDFFiltered.count()}")

// 조인 순서 최적화: 작은 테이블부터 조인
// 1) resDFSelectedTrBal (undersampled) 
// 2) mmktDFFiltered (user features)
// 3) xdrDF (hourly app usage)
// 4) xdrDFMon (monthly app usage pivot)
// 5) clickCountDF (historical click count)

val trainDF = resDFSelectedTrBal
    .filter(smnCond)
    .repartition(200, F.col("svc_mgmt_num"), F.col("feature_ym"))  // 조인 키로 파티셔닝
    .join(mmktDFFiltered, Seq("svc_mgmt_num", "feature_ym"))  // 첫 번째 조인
    .repartition(200, F.col("svc_mgmt_num"), F.col("feature_ym"), F.col("send_hournum_cd"))  // 다음 조인을 위한 파티셔닝
    .join(xdrDF, Seq("svc_mgmt_num", "feature_ym", "send_hournum_cd"))  // 두 번째 조인
    .join(xdrDFMon, Seq("svc_mgmt_num", "feature_ym"))  // 세 번째 조인
    .join(clickCountDF, Seq("svc_mgmt_num", "feature_ym", "send_hournum_cd"), "left")  // 네 번째 조인 (left)
    .na.fill(Map("click_cnt" -> 0.0))

val testDF = resDFSelectedTs
    .filter(smnCond)
    .repartition(200, F.col("svc_mgmt_num"), F.col("feature_ym"))
    .join(mmktDFFiltered, Seq("svc_mgmt_num", "feature_ym"))
    .repartition(200, F.col("svc_mgmt_num"), F.col("feature_ym"), F.col("send_hournum_cd"))
    .join(xdrDF, Seq("svc_mgmt_num", "feature_ym", "send_hournum_cd"))
    .join(xdrDFMon, Seq("svc_mgmt_num", "feature_ym"))
    .join(clickCountDF, Seq("svc_mgmt_num", "feature_ym", "send_hournum_cd"), "left")
    .na.fill(Map("click_cnt" -> 0.0))


// ===== Paragraph 13: Revising Training and Test dataset =====

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


// ===== Paragraph 14: Saving Training and Test dataset =====

// 저장 전 적절한 파티션 수로 조정하여 small files 문제 방지
trainDFRev
.repartition(100, F.col("send_ym"), F.col("send_hournum_cd"), F.col("suffix"))
.write
.mode("overwrite")
.partitionBy("send_ym", "send_hournum_cd", "suffix")
.option("compression", "snappy")  // 압축 사용
.parquet(s"aos/sto/trainDFRev${genSampleNumMulti.toInt}")

testDFRev
.repartition(50, F.col("send_ym"), F.col("send_hournum_cd"), F.col("suffix"))
.write
.mode("overwrite")
.partitionBy("send_ym", "send_hournum_cd", "suffix")
.option("compression", "snappy")
.parquet(s"aos/sto/testDFRev")

println("Training and test datasets saved successfully")


// ===== Paragraph 15: Loading Training and Test dataset =====

// 이전 단계에서 사용한 캐시 정리 (메모리 확보)
try {
  resDFSelectedTrBal.unpersist()
  resDFSelectedTs.unpersist()
  xdrDF.unpersist()
  xdrDFMon.unpersist()
  userYmDF.unpersist()
  mmktDFFiltered.unpersist()
} catch {
  case e: Exception => println(s"Cache cleanup warning: ${e.getMessage}")
}

spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "10g")

val trainDFRev = spark.read.parquet("aos/sto/trainDFRev10")
    .withColumn("hour_gap", F.expr("case when res_utility>=1.0 then 1 else 0 end"))
    .drop("click_cnt")
    .join(clickCountDF, Seq("svc_mgmt_num", "feature_ym", "send_hournum_cd"), "left")
    .na.fill(Map("click_cnt"->0.0))
    .persist(StorageLevel.MEMORY_AND_DISK_SER)

trainDFRev.count()  // 캐시 구체화
println(s"Training records loaded: ${trainDFRev.count()}")

val testDFRev = spark.read.parquet("aos/sto/testDFRev")
    .withColumn("hour_gap", F.expr("case when res_utility>=1.0 then 1 else 0 end"))
    .drop("click_cnt")
    .join(clickCountDF, Seq("svc_mgmt_num", "feature_ym", "send_hournum_cd"), "left")
    .na.fill(Map("click_cnt"->0.0))
    .persist(StorageLevel.MEMORY_AND_DISK_SER)

testDFRev.count()  // 캐시 구체화
println(s"Test records loaded: ${testDFRev.count()}")

val noFeatureCols = Array("click_yn","hour_gap","chnl_typ","cmpgn_typ")
val tokenCols = trainDFRev.columns.filter(x => x.endsWith("_token")).distinct
val continuousCols = (trainDFRev.columns.filter(x => numericColNameList.map(x.endsWith(_)).reduceOption(_ || _).getOrElse(false)).distinct.filter(x => !tokenCols.contains(x) && !noFeatureCols.contains(x))).distinct
val categoryCols = (trainDFRev.columns.filter(x => categoryColNameList.map(x.endsWith(_)).reduceOption(_ || _).getOrElse(false)).distinct.filter(x => !tokenCols.contains(x) && !noFeatureCols.contains(x) && !continuousCols.contains(x))).distinct
val vectorCols = trainDFRev.columns.filter(x => x.endsWith("_vec"))


// ===== Paragraph 16: Making Dataset for Real Service =====

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
.agg(collect_set("app_nm").alias("app_usage_token"))
.persist(StorageLevel.MEMORY_AND_DISK_SER)

xdrDFPred.count()  // 캐시 구체화
println(s"Prediction XDR hourly records: ${xdrDFPred.count()}")

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
    .persist(StorageLevel.MEMORY_AND_DISK_SER)

xdrPredDF.count()  // 캐시 구체화
println(s"Prediction XDR monthly records: ${xdrPredDF.count()}")

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
    .join(xdrDFPred, Seq("svc_mgmt_num", "feature_ym", "send_hournum_cd"), "left")
    .join(xdrPredDF, Seq("svc_mgmt_num", "feature_ym"), "left")
    .join(clickCountDF, Seq("svc_mgmt_num", "feature_ym", "send_hournum_cd"), "left")
    .na.fill(Map("click_cnt" -> 0.0))

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

predDFRev.count()  // 캐시 구체화
println(s"Prediction records prepared: ${predDFRev.count()}")


// ===== Paragraph 17: Param setting for Pipeline =====

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

val params:Map[String, Any] = Map("minDF"->1,"minTF"->5,"embSize"->30,"vocabSize"->30, "numParts"->nodeNumber)

val indexedLabelColClick = "indexedLabelClick"
val indexedLabelColGap = "indexedLabelGap"

val labelColsClick = Array(Map("indexer_nm"->"indexer_click", "col_nm"->"click_yn", "label_nm"->indexedLabelColClick))
val labelColsGap = Array(Map("indexer_nm"->"indexer_gap", "col_nm"->"hour_gap", "label_nm"->indexedLabelColGap))

val indexedLabelColReg = "res_utility"

val indexedFeatureColClick = "indexedFeaturesClick"
val scaledFeatureColClick = "scaledFeaturesClick"
val selectedFeatureColClick = "selectedFeaturesClick"

val indexedFeatureColGap = "indexedFeaturesGap"
val scaledFeatureColGap = "scaledFeaturesGap"
val selectedFeatureColGap = "selectedFeaturesGap"

val onlyGapFeature = Array[String]()

val doNotHashingCateCols = Array[String]("send_hournum_cd")
val doNotHashingContCols = Array[String]("click_cnt")


// ===== Paragraph 18: Pipeline Function (makePipeline) =====

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
        val cntVectorizer = tokenColsCnt.map(c => new HashingTF().setInputCol(c).setOutputCol(c +"_"+colNmSuffix+ "_cntvec").setNumFeatures(vocabSize))
        transformPipeline.setStages(if(transformPipeline.getStages.isEmpty){cntVectorizer}else{transformPipeline.getStages++cntVectorizer})
        featureListForAssembler ++= tokenColsCnt.map(_ +"_"+colNmSuffix+ "_cntvec")
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

    print(featureListForAssembler.distinct)

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


// ===== Paragraph 19: Pipeline Fitting =====

val transformPipelineClick = makePipeline(
    labelCols = labelColsClick,
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
  trainDFRev
    .sample(false, 0.3, 42)
    .persist(StorageLevel.MEMORY_AND_DISK_SER)
)

println("Transforming training data with Click transformer...")
var transformedTrainDF = transformerClick.transform(trainDFRev)
  .persist(StorageLevel.MEMORY_AND_DISK_SER)

transformedTrainDF.count()  // 캐시 구체화
println(s"Transformed training records (Click): ${transformedTrainDF.count()}")

println("Fitting Gap transformer pipeline...")
val transformerGap = transformPipelineGap.fit(
  transformedTrainDF
    .sample(false, 0.3, 42)
    .persist(StorageLevel.MEMORY_AND_DISK_SER)
)

println("Transforming training data with Gap transformer...")
transformedTrainDF = transformerGap.transform(transformedTrainDF)
  .persist(StorageLevel.MEMORY_AND_DISK_SER)

transformedTrainDF.count()  // 캐시 구체화
println(s"Transformed training records (Gap): ${transformedTrainDF.count()}")

println("Transforming test data...")
var transformedTestDF = transformerClick.transform(testDFRev)
transformedTestDF = transformerGap.transform(transformedTestDF)
  .persist(StorageLevel.MEMORY_AND_DISK_SER)

transformedTestDF.count()  // 캐시 구체화
println(s"Transformed test records: ${transformedTestDF.count()}")


// ===== Paragraph 20: Transformer Saving =====

println("Saving Click transformer...")
transformerClick.write.overwrite().save("aos/sto/transformPipelineXDRClick10")

println("Saving Gap transformer...")
transformerGap.write.overwrite().save("aos/sto/transformPipelineXDRGap10")

println("Saving transformed training data...")
transformedTrainDF
    .repartition(100, F.col("send_ym"), F.col("send_hournum_cd"), F.col("suffix"))
    .write
    .mode("overwrite")
    .option("compression", "snappy")
    .partitionBy("send_ym", "send_hournum_cd", "suffix")
    .parquet("aos/sto/transformedTrainDFXDR10")

println("Saving transformed test data...")
transformedTestDF
    .repartition(50, F.col("send_ym"), F.col("send_hournum_cd"), F.col("suffix"))
    .write
    .mode("overwrite")
    .option("compression", "snappy")
    .partitionBy("send_ym", "send_hournum_cd", "suffix")
    .parquet("aos/sto/transformedTestDFXDF10")

println("All transformers and transformed data saved successfully")


// ===== Paragraph 21: Transformer Loading =====

println("Loading Click transformer...")
val transformerClick = PipelineModel.load("aos/sto/transformPipelineXDRClick")

println("Loading Gap transformer...")
val transformerGap = PipelineModel.load("aos/sto/transformPipelineXDRGap")

println("Loading transformed training data...")
val transformedTrainDF = spark.read.parquet("aos/sto/transformedTrainDFXDR")
    .persist(StorageLevel.MEMORY_AND_DISK_SER)

transformedTrainDF.count()  // 캐시 구체화
println(s"Transformed training records: ${transformedTrainDF.count()}")

println("Loading transformed test data...")
val transformedTestDF = spark.read.parquet("aos/sto/transformedTestDFXDF")
    .persist(StorageLevel.MEMORY_AND_DISK_SER)

transformedTestDF.count()  // 캐시 구체화
println(s"Transformed test records: ${transformedTestDF.count()}")

// val transformedPredDF = transformer.transform(predDFRev)//.persist(StorageLevel.MEMORY_AND_DISK_SER)


// ===== Paragraph 22: Model Definitions =====

import org.apache.spark.ml.classification._
import org.apache.spark.ml.regression._

import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.StringIndexerModel

val labelIndexersMap: Map[String, StringIndexerModel] = transformerClick.stages.collect {
  case sim: StringIndexerModel => sim.uid -> sim
}.toMap ++ transformerGap.stages.collect {
  case sim: StringIndexerModel => sim.uid -> sim
}.toMap

val gbtc = new GBTClassifier("gbtc_click")
  .setLabelCol(indexedLabelColClick)
  .setFeaturesCol(indexedFeatureColClick)
  .setMaxIter(50)
  .setMaxDepth(4)
  .setFeatureSubsetStrategy("auto")
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
  .setFeatureSubsetStrategy("auto")
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


// ===== Paragraph 23: XGB Feature Interaction Constraints =====

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


// ===== Paragraph 24: Click Model Training with Cross Validation =====

import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator

val modelClickforCV = xgbc

val pipelineMLClick = new Pipeline().setStages(Array(modelClickforCV))

// 학습 데이터 샘플링 최적화 - 캐시하여 재사용
val trainSampleClick = transformedTrainDF
    .filter("cmpgn_typ=='Sales'")
    .stat.sampleBy(
        F.col(indexedLabelColClick),
        Map(
            0.0 -> 0.5,
            1.0 -> 1.0,
        ),
        42L
    )
    .repartition(100)  // 학습을 위한 적절한 파티션 수
    .persist(StorageLevel.MEMORY_AND_DISK_SER)

trainSampleClick.count()  // 캐시 구체화
println(s"Click model training samples: ${trainSampleClick.count()}")

println("Training Click prediction model...")
val pipelineModelClick = pipelineMLClick.fit(trainSampleClick)

trainSampleClick.unpersist()  // 학습 완료 후 메모리 해제


// ===== Paragraph 25: Gap Model Training =====

val modelGapforCV = xgbg

val pipelineMLGap = new Pipeline().setStages(Array(modelGapforCV))

// Gap 모델 학습 데이터 샘플링 최적화
val trainSampleGap = transformedTrainDF
    .filter("click_yn>0")
    .stat.sampleBy(
        F.col("hour_gap"),
        Map(
            0.0 -> 1.0,
            1.0 -> 0.45,
        ),
        42L
    )
    .repartition(100)
    .persist(StorageLevel.MEMORY_AND_DISK_SER)

trainSampleGap.count()  // 캐시 구체화
println(s"Gap model training samples: ${trainSampleGap.count()}")

println("Training Gap prediction model...")
val pipelineModelGap = pipelineMLGap.fit(trainSampleGap)

trainSampleGap.unpersist()  // 학습 완료 후 메모리 해제


// ===== Paragraph 26: Gap Regression =====

import org.apache.spark.ml.evaluation.RegressionEvaluator

val modelRegforCV = xgbr

val pipelineMLReg = new Pipeline().setStages(Array(modelRegforCV))

// 회귀 모델 학습 데이터 준비
val trainSampleReg = transformedTrainDF
    .filter("click_yn>0")
    .repartition(100)
    .persist(StorageLevel.MEMORY_AND_DISK_SER)

trainSampleReg.count()  // 캐시 구체화
println(s"Regression model training samples: ${trainSampleReg.count()}")

println("Training Regression model...")
val pipelineModelReg = pipelineMLReg.fit(trainSampleReg)

trainSampleReg.unpersist()  // 학습 완료 후 메모리 해제


// ===== Paragraph 27: Predictions on Test Data =====

// 테스트 데이터 중복 제거 후 예측 - 파티션 최적화
val testDataForPred = transformedTestDF
    .filter("cmpgn_typ=='Sales'")
    .dropDuplicates("svc_mgmt_num", "chnl_typ", "cmpgn_typ", "send_ym", "send_hournum_cd", "click_yn")
    .repartition(100)
    .persist(StorageLevel.MEMORY_AND_DISK_SER)

testDataForPred.count()  // 캐시 구체화
println(s"Test data for prediction: ${testDataForPred.count()}")

println("Generating Click predictions...")
val predictionsClickDev = pipelineModelClick.transform(testDataForPred)
    .persist(StorageLevel.MEMORY_AND_DISK_SER)

predictionsClickDev.count()
println(s"Click predictions generated: ${predictionsClickDev.count()}")

println("Generating Gap predictions...")
val predictionsGapDev = pipelineModelGap.transform(testDataForPred)
    .persist(StorageLevel.MEMORY_AND_DISK_SER)

predictionsGapDev.count()
println(s"Gap predictions generated: ${predictionsGapDev.count()}")


// ===== Paragraph 28: Click Model Evaluation =====

import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.functions._
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}

import org.apache.spark.ml.linalg.Vector

spark.udf.register("vector_to_array", (v: Vector) => v.toArray)

val topK = 50000
val thresholdProb = 0.5

val stagesClick = pipelineModelClick.stages

println("Evaluating Click models...")

stagesClick.foreach { stage => 
    
    val modelName = stage.uid
    
    println(s"Evaluating model: $modelName")
    
    // 집계 및 평가용 데이터 준비 - 파티션 최적화
    val aggregatedPredictions = predictionsClickDev
        .withColumn("prob", F.expr(s"vector_to_array(prob_$modelName)[1]"))
        .repartition(50, F.col("svc_mgmt_num"))  // GroupBy 전 파티셔닝
        .groupBy("svc_mgmt_num", "send_ym","send_hournum_cd")
        .agg(F.sum(indexedLabelColClick).alias(indexedLabelColClick), F.max("prob").alias("prob"))
        .withColumn(indexedLabelColClick, F.expr(s"case when $indexedLabelColClick>0 then cast(1.0 AS DOUBLE) else cast(0.0 AS DOUBLE) end"))
        .withColumn("prediction_click", F.expr(s"case when prob>=$thresholdProb then cast(1.0 AS DOUBLE) else cast(0.0 AS DOUBLE) end"))
        .persist(StorageLevel.MEMORY_AND_DISK_SER)
    
    aggregatedPredictions.count()  // 캐시 구체화
    
    val predictionAndLabels = labelConverterClick.transform(aggregatedPredictions)
        .selectExpr("prediction_click", s"cast($indexedLabelColClick as double)")
        .rdd
        .map(row => (row.getDouble(0), row.getDouble(1)))
    
    val metrics = new MulticlassMetrics(predictionAndLabels)
    val labels = metrics.labels
    
    println(s"######### $modelName 예측 결과 #########")
    
    println("--- 레이블별 성능 지표 ---")
    labels.foreach { label =>
      val precision = metrics.precision(label)
      val recall = metrics.recall(label)
      val f1 = metrics.fMeasure(label)
    
      println(f"Label $label (클래스): Precision = $precision%.4f, Recall = $recall%.4f, F1 = $f1%.4f")
    }
    
    println(s"\nWeighted Precision (전체 평균): ${metrics.weightedPrecision}")
    println(s"Weighted Recall (전체 평균): ${metrics.weightedRecall}")
    println(s"Accuracy (전체 정확도): ${metrics.accuracy}")
    
    println("\n--- Confusion Matrix (혼동 행렬) ---")
    println(metrics.confusionMatrix)
    
    aggregatedPredictions.unpersist()  // 메모리 해제
}


// ===== Paragraph 29: Gap Model Evaluation =====

val stagesGap = pipelineModelGap.stages

println("Evaluating Gap models...")

stagesGap.foreach { stage => 
    
    val modelName = stage.uid
    
    println(s"Evaluating model: $modelName")
    
    // 집계 및 평가용 데이터 준비 - 파티션 최적화
    val aggregatedPredictionsGap = predictionsGapDev
        .filter("click_yn>0")
        .withColumn("prob", F.expr(s"vector_to_array(prob_$modelName)[1]"))
        .repartition(50, F.col("svc_mgmt_num"))  // GroupBy 전 파티셔닝
        .groupBy("svc_mgmt_num", "send_ym", "send_hournum_cd")
        .agg(F.sum(indexedLabelColGap).alias(indexedLabelColGap), F.max("prob").alias("prob"))
        .withColumn(indexedLabelColGap, F.expr(s"case when $indexedLabelColGap>0 then cast(1.0 AS DOUBLE) else cast(0.0 AS DOUBLE) end"))
        .withColumn("prediction_gap", F.expr("case when prob>=0.5 then cast(1.0 AS DOUBLE) else cast(0.0 AS DOUBLE) end"))
        .persist(StorageLevel.MEMORY_AND_DISK_SER)
    
    aggregatedPredictionsGap.count()  // 캐시 구체화
    
    val predictionAndLabels = labelConverterGap.transform(aggregatedPredictionsGap)
        .selectExpr("prediction_gap", s"cast($indexedLabelColGap as double)")
        .rdd
        .map(row => (row.getDouble(0), row.getDouble(1)))
    
    val metrics = new MulticlassMetrics(predictionAndLabels)
    val labels = metrics.labels
    
    println(s"######### $modelName 예측 결과 #########")
    
    println("--- 레이블별 성능 지표 ---")
    labels.foreach { label =>
      val precision = metrics.precision(label)
      val recall = metrics.recall(label)
      val f1 = metrics.fMeasure(label)
    
      println(f"Label $label (클래스): Precision = $precision%.4f, Recall = $recall%.4f, F1 = $f1%.4f")
    }
    
    println(s"\nWeighted Precision (전체 평균): ${metrics.weightedPrecision}")
    println(s"Weighted Recall (전체 평균): ${metrics.weightedRecall}")
    println(s"Accuracy (전체 정확도): ${metrics.accuracy}")
    
    println("\n--- Confusion Matrix (혼동 행렬) ---")
    println(metrics.confusionMatrix)
    
    aggregatedPredictionsGap.unpersist()  // 메모리 해제
}


// ===== Paragraph 30: Regression Model Evaluation =====

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

predictionsRegDev.count()
println(s"Regression predictions generated: ${predictionsRegDev.count()}")

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


// ===== Paragraph 31: Propensity Score Calculation and Saving =====

import org.apache.spark.ml.linalg.Vector

spark.udf.register("vector_to_array", (v: Vector) => v.toArray)

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
(0 to 15).map(_.toHexString).foreach { suffix => 

    println(s"Processing suffix: $suffix")

    val predDFForSuffix = predDFRev
        .filter(s"svc_mgmt_num like '%${suffix}'")
        .repartition(50)  // 적절한 파티션 수로 조정
        .persist(StorageLevel.MEMORY_AND_DISK_SER)
    
    val recordCount = predDFForSuffix.count()
    println(s"Records for suffix $suffix: $recordCount")
    
    if (recordCount > 0) {
        println(s"Transforming data for suffix $suffix...")
        val transformedPredDF = transformerGap.transform(
            transformerClick.transform(predDFForSuffix)
        ).persist(StorageLevel.MEMORY_AND_DISK_SER)
        
        println(s"Deduplicating for suffix $suffix...")
        val dedupedDF = transformedPredDF
            .dropDuplicates("svc_mgmt_num", "chnl_typ", "cmpgn_typ", "send_ym", "send_hournum_cd", "click_yn")
        
        println(s"Generating Click predictions for suffix $suffix...")
        val predictionsSVCClick = pipelineModelClick.transform(dedupedDF)
        
        println(s"Generating Gap predictions for suffix $suffix...")
        val predictionsSVCFinal = pipelineModelGap.transform(predictionsSVCClick)
        
        println(s"Calculating propensity scores for suffix $suffix...")
        var predictedPropensityScoreDF = predictionsSVCFinal
            .withColumn("prob_click", F.expr(s"""aggregate(array(${pipelineModelClick.stages.map(m => s"vector_to_array(prob_${m.uid})[1]").mkString(",")}), 0D, (acc, x) -> acc + x)"""))
            .withColumn("prob_gap", F.expr(s"""aggregate(array(${pipelineModelGap.stages.map(m => s"vector_to_array(prob_${m.uid})[1]").mkString(",")}), 0D, (acc, x) -> acc + x)"""))
            .withColumn("propensity_score", F.expr("prob_click*prob_gap"))
        
        println(s"Saving propensity scores for suffix $suffix...")
        predictedPropensityScoreDF
            .selectExpr("svc_mgmt_num", "send_ym","send_hournum_cd send_hour"
                ,"ROUND(prob_click, 4) prob_click" 
                ,"ROUND(prob_gap, 4) prob_gap"
                ,"ROUND(propensity_score, 4) propensity_score")
            .withColumn("suffix", F.expr("right(svc_mgmt_num, 1)"))
            .repartition(10)
            .write
            .mode("overwrite")
            .option("compression", "snappy")
            .partitionBy("send_ym", "send_hour", "suffix")
            .parquet("aos/sto/propensityScoreDF")
        
        // 메모리 해제
        predDFForSuffix.unpersist()
        transformedPredDF.unpersist()
        
        println(s"Completed processing suffix $suffix")
    } else {
        println(s"No records found for suffix $suffix, skipping...")
    }
}


// ===== Paragraph 32: Load and Verify Propensity Score Data =====

val dfPropensity = spark.read.parquet("aos/sto/propensityScoreDF")
    .persist(StorageLevel.MEMORY_AND_DISK_SER)

val propensityCount = dfPropensity.count()
println(s"Total propensity score records: $propensityCount")

// 기본 통계 확인
dfPropensity
    .select("prob_click", "prob_gap", "propensity_score")
    .summary("count", "mean", "stddev", "min", "max")
    .show()

propensityCount


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

```


#### Short summary: 

empty definition using pc, found symbol in pc: filter.