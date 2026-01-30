// =============================================================================
// Converted from Zeppelin Notebook
// =============================================================================
// Original file: predict_ost_2MC68ADVY_260130.zpln
// Converted on: 2026-01-30 10:41:31
// Total paragraphs: 37
// =============================================================================


// =============================================================================
// Paragraph 1: Preps (Imports and Configuration)
// =============================================================================

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
// spark.conf.set("spark.sql.shuffle.partitions", "400")
spark.conf.set("spark.sql.files.maxPartitionBytes", "128MB")

spark.sparkContext.setCheckpointDir("hdfs://scluster/user/g1110566/checkpoint")


// =============================================================================
// Paragraph 2: Helper Functions
// =============================================================================

import java.time.YearMonth
import java.time.format.DateTimeFormatter
import java.time.LocalDate
import scala.collection.mutable.ListBuffer
import java.time.temporal.ChronoUnit
import org.apache.spark.ml.linalg.Vector

spark.udf.register("vector_to_array", (v: Vector) => v.toArray)

def getPreviousMonths(startMonthStr: String, periodM: Int): Array[String] = {
  val formatter = DateTimeFormatter.ofPattern("yyyyMM")

  // 입력 월 파싱
  val startMonth = YearMonth.parse(startMonthStr, formatter)

  // 결과를 저장할 가변 리스트 (순서를 위해 ListBuffer 사용)
  var resultMonths = scala.collection.mutable.ListBuffer[String]()
  var currentMonth = startMonth

  // M번 반복하여 월을 계산하고 리스트에 추가
  for (i <- 0 until periodM) {
    // 현재 월을 리스트 맨 앞에 추가
    resultMonths.prepend(currentMonth.format(formatter))
    // 다음 반복을 위해 이전 달로 이동
    currentMonth = currentMonth.minusMonths(1)
  }

  // 리스트를 Array로 반환
  resultMonths.toArray
}

def getPreviousDays(startDayStr: String, periodD: Int): Array[String] = {
  // yyyyMMdd 형식의 포맷터 설정
  val formatter = DateTimeFormatter.ofPattern("yyyyMMdd")

  // 1. 입력된 날짜 문자열 파싱
  val startDay = LocalDate.parse(startDayStr, formatter)

  // 2. 결과를 저장할 가변 리스트 (순서를 위해 ListBuffer 사용)
  val resultDays = ListBuffer[String]()
  var currentDay = startDay

  // 3. periodD번 반복하여 날짜를 계산하고 리스트에 추가
  for (i <- 0 until periodD) {
    // 현재 날짜를 리스트 맨 앞에 추가 (최신 날짜가 뒤로 가게 하려면 append 사용)
    resultDays.prepend(currentDay.format(formatter))

    // 다음 반복을 위해 이전 날짜(1일 전)로 이동
    currentDay = currentDay.minusDays(1)
  }

  // 4. 리스트를 Array로 반환
  resultDays.toArray
}

def getDaysBetween(startDayStr: String, endDayStr: String): Array[String] = {
  val formatter = DateTimeFormatter.ofPattern("yyyyMMdd")

  val start = LocalDate.parse(startDayStr, formatter)
  val end = LocalDate.parse(endDayStr, formatter)

  // 두 날짜 사이의 일수 차이 계산
  val numOfDays = ChronoUnit.DAYS.between(start, end).toInt

  val resultDays = ListBuffer[String]()

  // 에러 수정: Scala의 for 루프는 'i <- 시작 to 끝' 형식을 사용합니다.
  for (i <- 0 to numOfDays) {
    resultDays.append(start.plusDays(i).format(formatter))
  }

  resultDays.toArray
}


// =============================================================================
// Paragraph 3: Response Data Loading - Hive
// =============================================================================

val cmpgnYM = z.input("cmpgn_ym", "202512,202601").toString
val cmpgnYMList = cmpgnYM.split(",").mkString("','")

val cmpgnResDF = spark.sql(s"""
with ract as
(
select A.cmpgn_num
    ,A.cmpgn_obj_num
    ,B.svc_mgmt_num
    ,A.extrt_seq
    ,CASE WHEN A.ract_typ_cd = '0802' THEN 1 ELSE 0 END AS send_yn
    ,CASE WHEN A.ract_typ_cd = '0810' THEN 1 ELSE 0 END AS click_yn
    ,CASE WHEN A.ract_typ_cd = '0811' THEN 1 ELSE 0 END AS read_yn
    ,CASE WHEN A.ract_typ_cd = '0802' THEN A.cont_dt ELSE null END AS send_dt
    ,CASE WHEN A.ract_typ_cd = '0810' THEN A.cont_dt ELSE null END AS click_dt
    ,CASE WHEN A.ract_typ_cd = '0811' THEN A.cont_dt ELSE null END AS read_dt
    ,CASE WHEN A.ract_typ_cd = '0802' THEN A.cont_tm ELSE null END AS send_tm
    ,CASE WHEN A.ract_typ_cd = '0810' THEN A.cont_tm ELSE null END AS click_tm
    ,CASE WHEN A.ract_typ_cd = '0811' THEN A.cont_tm ELSE null END AS read_tm
    ,CASE WHEN A.cont_chnl_cd = 'C18001' THEN 'MMS'
            WHEN A.cont_chnl_cd = 'C28001' THEN 'RCS' END AS chnl_typ
    ,CASE WHEN C.cmpgn_purp_typ_cd IN ('S01', 'S06') THEN 'Sales'
                WHEN C.cmpgn_purp_typ_cd = 'C01' THEN 'Care'
                WHEN C.cmpgn_purp_typ_cd = 'B01' THEN 'Bmarketing' END AS cmpgn_typ
    ,CASE WHEN D.cmpgn_num IS NOT NULL THEN 'url_Y' ELSE 'url_N' END AS url_yn
from tos.od_tcam_cmpgn_obj_cont as A
LEFT JOIN (SELECT DISTINCT cmpgn_num, cmpgn_obj_num, svc_mgmt_num FROM tos.od_tcam_cmpgn_obj) AS B
ON A.cmpgn_obj_num = B.cmpgn_obj_num AND A.cmpgn_num = B.cmpgn_num
LEFT JOIN tos.od_tcam_cmpgn_brief AS C
ON A.cmpgn_num = C.cmpgn_num
LEFT JOIN (SELECT DISTINCT cmpgn_num FROM tos.od_tcam_cmpgn_obj WHERE ract_typ_cd = '0810') AS D
ON A.cmpgn_num = D.cmpgn_num
where A.ract_typ_cd in ('0802','0810', '0811')
AND A.cont_chnl_cd in ('C18001','C28001')
AND B.svc_mgmt_num != '0'
AND substring(A.cont_dt, 1, 6) in ('${cmpgnYMList}')
)

,ract2 as(

select cmpgn_num
    ,svc_mgmt_num
    ,extrt_seq
    ,chnl_typ
    ,cmpgn_typ
    ,url_yn
    ,send_yn
    ,read_yn
    ,click_yn
    ,send_dt
    ,read_dt
    ,click_dt
    ,CASE WHEN LENGTH(send_dt) = 8 AND LENGTH(send_tm) = 6 THEN TO_TIMESTAMP(send_dt || send_tm, 'yyyyMMddHHmmss') ELSE NULL END AS send_time
    ,CASE WHEN LENGTH(read_dt) = 8 AND LENGTH(read_tm) = 6 THEN TO_TIMESTAMP(read_dt || read_tm, 'yyyyMMddHHmmss') ELSE NULL END AS read_time
    ,CASE WHEN LENGTH(click_dt) = 8 AND LENGTH(click_tm) = 6 THEN TO_TIMESTAMP(click_dt || click_tm, 'yyyyMMddHHmmss') ELSE NULL END AS click_time
from ract
where (send_dt IS NULL OR LENGTH(send_dt) = 8)
and (send_tm IS NULL OR LENGTH(send_tm) = 6)
and (read_dt IS NULL OR LENGTH(read_dt) = 8)
and (read_tm IS NULL OR LENGTH(read_tm) = 6)
and (click_dt IS NULL OR LENGTH(click_dt) = 8)
and (click_tm IS NULL OR LENGTH(click_tm) = 6)
)

,ract3 as(
select cmpgn_num
    , svc_mgmt_num
    , extrt_seq
    , chnl_typ
    , cmpgn_typ
    , url_yn
    , sum(send_yn) as send_yn
    , sum(read_yn) as read_yn
    , sum(click_yn) as click_yn
    , min(send_dt) as send_dt
    , min(send_time) as send_time
    , min(read_dt) as read_dt
    , min(read_time) as read_time
    , min(click_dt) as click_dt
    , min(click_time) as click_time
from ract2
group by cmpgn_num, svc_mgmt_num, extrt_seq, chnl_typ, cmpgn_typ, url_yn
),

tmp AS
(
select *
from ract3
where (send_yn = 1 OR (send_yn = 2 AND chnl_typ = 'MMS'))
and (read_yn IS NULL OR read_yn <= 1)
)


SELECT *
    ,dayofweek(to_date(send_dt, 'yyyyMMdd')) as send_daynum
    ,dayofweek(to_date(click_dt, 'yyyyMMdd')) as click_daynum
    ,hour(send_time) as send_hournum
    ,hour(click_time) as click_hournum
    ,CAST(datediff(click_time, send_time) AS INTEGER) AS day_gap
    ,CAST((unix_timestamp(click_time) - unix_timestamp(send_time))/3600 AS INTEGER) AS hour_gap
    ,CAST((unix_timestamp(click_time) - unix_timestamp(send_time))/60 AS INTEGER) AS minute_gap
    ,CAST(unix_timestamp(click_time) - unix_timestamp(send_time) AS INTEGER) AS second_gap
    ,substring(send_dt,1,6) as send_ym
FROM tmp

""")//.cache()

// cmpgnResDF.createOrReplaceTempView("cmpgn_df_view")


// =============================================================================
// Paragraph 4: Response Data Saving
// =============================================================================

cmpgnResDF.repartition(100).write.mode("overwrite").partitionBy("send_ym").parquet("aos/sto/response")


// =============================================================================
// Paragraph 5: Date Range and Period Configuration
// =============================================================================

// -----------------------------------------------------------------------------
// 작업 흐름 1-2: Campaign 반응 데이터 로딩 및 Train/Test Raw 생성 (Paragraph 3-14)
// -----------------------------------------------------------------------------
// 반응 데이터는 학습/테스트 데이터의 기반이 되므로 동일한 시간 조건을 사용합니다.
//
// 학습 및 테스트 데이터 생성을 위한 시간 조건을 정의합니다.
val sendMonth = "202512"                  // 학습/테스트 데이터 기준 발송 월
val featureMonth = "202511"               // 피처 추출 기준 월 (발송 월 이전)
val period = 3                            // 학습 데이터 기간 (개월 수)
val sendYmList = getPreviousMonths(sendMonth, period+2)
val featureYmList = getPreviousMonths(featureMonth, period+2)

// 피처 추출을 위한 날짜 범위
val featureDTList = getDaysBetween(featureYmList(0)+"01", sendMonth+"01")

// Train/Test 분할 기준 날짜
val predictionDTSta = "20251201"          // 테스트 데이터 시작 날짜
val predictionDTEnd = "20260101"          // 테스트 데이터 종료 날짜

// 반응 데이터 필터링 조건
val responseHourGapMax = 5                // 최대 클릭 시간차 (시간 단위)

// 시간대 설정 (반응 데이터 필터링 및 분석에 공통 사용)
val upperHourGap = 1                      // 상위 시간차 임계값
val startHour = 9                         // 분석 대상 시작 시간
val endHour = 18                          // 분석 대상 종료 시간
val hourRange = (startHour to endHour).toList

// -----------------------------------------------------------------------------
// 작업 흐름 3: Transformed Data 생성 및 저장 (Paragraph 15-20)
// -----------------------------------------------------------------------------
// Raw 데이터를 로딩하여 transformation pipeline을 적용하는 단계의 시간 조건입니다.
// 주의: 이 단계는 작업 흐름 2에서 생성된 데이터를 사용하므로,
//       저장 경로의 버전/suffix와 일치해야 합니다.
val transformRawDataVersion = "10"        // 로딩할 raw training data 버전 (trainDFRev10)
val transformTestDataPath = "testDFRev"   // 로딩할 raw test data 경로
val transformSampleRate = 0.3             // Pipeline fitting 시 샘플링 비율 (메모리 절약)
val transformSuffixGroupSize = 2          // Suffix별 배치 저장 시 그룹 크기 (1,2,4,8,16)

// Transformed data 저장 경로 버전 관리
val transformedTrainSaveVersion = "10"    // Transformed training data 저장 버전
val transformedTestSaveVersion = "10"     // Transformed test data 저장 버전

// -----------------------------------------------------------------------------
// 작업 흐름 4: Transformed Data를 통한 학습 및 테스트 (Paragraph 21-30)
// -----------------------------------------------------------------------------
// Transformed data를 로딩하여 모델 학습 및 평가를 수행하는 단계의 시간 조건입니다.
// 주의: 이 단계는 작업 흐름 3에서 생성된 데이터를 사용하므로,
//       로딩 경로가 작업 흐름 3의 저장 경로와 일치해야 합니다.
val modelTrainDataVersion = "10"          // 로딩할 transformed training data 버전
val modelTestDataVersion = "10"           // 로딩할 transformed test data 버전
val modelTransformerVersion = ""          // 로딩할 transformer 버전 (빈 문자열이면 버전 없음)

// 모델 학습 설정
val modelMaxIter = 50                     // 모델 학습 최대 반복 횟수
val modelCrossValidationFolds = 3         // Cross-validation fold 수

// -----------------------------------------------------------------------------
// 작업 흐름 5: 실제 서비스용 데이터 생성 및 예측 (Paragraph 16 + 31-32)
// -----------------------------------------------------------------------------
// 실제 서비스에 사용할 예측 데이터를 생성하고 propensity score를 계산하는 단계입니다.
val predDT = "20251201"                   // 예측 기준 날짜 (발송 예정일)
val predFeatureYM = getPreviousMonths(predDT.take(6), 2)(0)  // 피처 추출 기준 월
val predSendYM = predDT.take(6)           // 예측 발송 월

// 예측 데이터 처리 설정
val predSuffix = "%"                      // 처리할 suffix 패턴 (% = 전체)
val predSuffixGroupSize = 4               // Propensity score 계산 시 suffix 배치 크기
val predRepartitionSize = 50              // 예측 데이터 repartition 크기

// 예측 결과 저장 경로
val predOutputPath = "aos/sto/propensityScoreDF"  // Propensity score 저장 경로
val predCompression = "snappy"            // 압축 방식

// -----------------------------------------------------------------------------
// 공통 설정 검증 (Configuration Validation)
// -----------------------------------------------------------------------------
// 시간 조건 변수 간의 일관성을 검증합니다.
println("=" * 80)
println("Time Condition Variables Configuration Summary")
println("=" * 80)
println(s"[작업 흐름 1-2] Response Data Loading & Train/Test Raw Data Generation")
println(s"  - Send Month: $sendMonth")
println(s"  - Feature Month: $featureMonth")
println(s"  - Period: $period months")
println(s"  - Send YM List: ${sendYmList.mkString(", ")}")
println(s"  - Feature YM List: ${featureYmList.mkString(", ")}")
println(s"  - Train/Test Split: < $predictionDTSta (train) / >= $predictionDTSta (test)")
println(s"  - Response Hour Gap Max: $responseHourGapMax hours")
println(s"  - Hour Range: $startHour ~ $endHour")
println()
println(s"[작업 흐름 3] Transformed Data Generation")
println(s"  - Raw Train Data Version: $transformRawDataVersion")
println(s"  - Raw Test Data Path: $transformTestDataPath")
println(s"  - Transformed Train Version: $transformedTrainSaveVersion")
println(s"  - Transformed Test Version: $transformedTestSaveVersion")
println(s"  - Suffix Group Size: $transformSuffixGroupSize")
println(s"  - Pipeline Sample Rate: $transformSampleRate")
println()
println(s"[작업 흐름 4] Model Training and Evaluation")
println(s"  - Train Data Version: $modelTrainDataVersion")
println(s"  - Test Data Version: $modelTestDataVersion")
println(s"  - Transformer Version: ${if (modelTransformerVersion.isEmpty) "default" else modelTransformerVersion}")
println(s"  - Max Iterations: $modelMaxIter")
println(s"  - CV Folds: $modelCrossValidationFolds")
println()
println(s"[작업 흐름 5] Production Prediction")
println(s"  - Prediction Date: $predDT")
println(s"  - Feature Month: $predFeatureYM")
println(s"  - Send Month: $predSendYM")
println(s"  - Suffix Pattern: $predSuffix")
println(s"  - Suffix Group Size: $predSuffixGroupSize")
println(s"  - Repartition Size: $predRepartitionSize")
println(s"  - Output Path: $predOutputPath")
println(s"  - Compression: $predCompression")
println()

// 검증: 작업 흐름 3과 4의 버전 일치 확인
if (transformedTrainSaveVersion != modelTrainDataVersion || transformedTestSaveVersion != modelTestDataVersion) {
  println("⚠️  WARNING: Version mismatch detected!")
  println(s"   Transformed data versions (save): Train=$transformedTrainSaveVersion, Test=$transformedTestSaveVersion")
  println(s"   Model training versions (load): Train=$modelTrainDataVersion, Test=$modelTestDataVersion")
  println("   Please ensure the versions are consistent across workflows 3 and 4.")
  println()
}

// 검증: Raw data 버전 일치 확인
if (transformRawDataVersion != transformedTrainSaveVersion) {
  println("⚠️  WARNING: Raw data version and transformed data version mismatch!")
  println(s"   Raw data version (load in P15): $transformRawDataVersion")
  println(s"   Transformed data version (save in P20): $transformedTrainSaveVersion")
  println("   Consider using the same version number for consistency.")
  println()
}

println("=" * 80)


// =============================================================================
// Paragraph 6: Response Data Loading from HDFS
// =============================================================================

println("=" * 80)
println("[작업 흐름 1-2] Response Data Loading from HDFS")
println("=" * 80)
println(s"Loading response data...")
println(s"  - Base month: $sendMonth")
println(s"  - Period: $period months")
println(s"  - Target months: ${sendYmList.mkString(", ")}")
println("=" * 80)

// 대용량 데이터는 MEMORY_AND_DISK_SER 사용하여 메모리 절약
val resDF = spark.read.parquet("aos/sto/response")
  .persist(StorageLevel.MEMORY_AND_DISK_SER)
resDF.createOrReplaceTempView("res_df")

// val totalRecords = resDF.count()
// println(s"Response data loaded: $totalRecords records")
println("=" * 80)

resDF.printSchema()


// =============================================================================
// Paragraph 7: Response Data Filtering and Feature Engineering
// =============================================================================

// =============================================================================
// [작업 흐름 1-2] Response Data 필터링 및 피처 엔지니어링
// =============================================================================
// 이 단계는 Paragraph 4에서 정의된 시간 조건 변수를 사용합니다:
// - sendYmList: 필터링할 월 리스트
// - responseHourGapMax: 최대 클릭 시간차
// - startHour, endHour: 발송 시간대 범위
//
// 이 필터링된 데이터는 Train/Test raw 데이터 생성의 기반이 됩니다.
// =============================================================================

println("=" * 80)
println("[작업 흐름 1-2] Response Data Filtering and Feature Engineering")
println("=" * 80)
println(s"Applying filters:")
println(s"  - Send months: ${sendYmList.mkString(", ")}")
println(s"  - Hour gap: 0 ~ $responseHourGapMax")
println(s"  - Send hour: $startHour ~ $endHour")
println("=" * 80)

val resDFFiltered = resDF
    .filter(s"""send_ym in (${sendYmList.mkString("'","','","'")})""")
    .filter(s"hour_gap is null or (hour_gap between 0 and $responseHourGapMax)")
    .filter(s"send_hournum between $startHour and $endHour")
    .selectExpr("cmpgn_num","svc_mgmt_num","chnl_typ","cmpgn_typ","send_ym","send_dt","send_time","send_daynum","send_hournum","click_dt","click_time","click_daynum","click_hournum","case when hour_gap is null then 0 else 1 end click_yn","hour_gap")
    .withColumn("res_utility", F.expr(s"case when hour_gap is null then 0.0 else 1.0 / (1 + hour_gap) end"))
    .dropDuplicates()
    .repartition(200, F.col("svc_mgmt_num"))  // 조인을 위해 적절한 파티션 수로 재분배
    .persist(StorageLevel.MEMORY_AND_DISK_SER)

println("Filtered response data cached (will materialize on next action)")

val filteredRecords = resDFFiltered.count()
println(s"Filtered response data: $filteredRecords records")
println("=" * 80)

resDF.printSchema()


// =============================================================================
// Paragraph 8: User Feature Data Loading (MMKT_SVC_BAS)
// =============================================================================

val allFeaturesMMKT = spark.sql("describe wind_tmt.mmkt_svc_bas_f").select("col_name").collect().map(_.getString(0))
val sigFeaturesMMKT = spark.read.option("header", "true").csv("feature_importance/table=mmkt_bas/creation_dt=20230407").filter("rank<=100").select("col").collect().map(_(0).toString()).map(_.trim)
val colListForMMKT = (Array("svc_mgmt_num", "strd_ym feature_ym", "mst_work_dt", "cust_birth_dt", "prcpln_last_chg_dt", "fee_prod_id", "eqp_mdl_cd", "eqp_acqr_dt", "equip_chg_cnt", "svc_scrb_dt", "chg_dt", "cust_age_cd", "sex_cd", "equip_chg_day", "last_equip_chg_dt", "prev_equip_chg_dt", "rten_pen_amt", "agrmt_brch_amt", "eqp_mfact_cd",
    "allot_mth_cnt", "mbr_use_cnt", "mbr_use_amt", "tyr_mbr_use_cnt", "tyr_mbr_use_amt", "mth_cnsl_cnt", "dsat_cnsl_cnt", "simpl_ref_cnsl_cnt", "arpu", "bf_m1_arpu", "voc_arpu", "bf_m3_avg_arpu",
    "tfmly_nh39_scrb_yn", "prcpln_chg_cnt", "email_inv_yn", "copn_use_psbl_cnt", "data_gift_send_yn", "data_gift_recv_yn", "equip_chg_mth_cnt", "dom_tot_pckt_cnt", "scrb_sale_chnl_cl_cd", "op_sale_chnl_cl_cd", "agrmt_dc_end_dt",
    "svc_cd", "svc_st_cd", "pps_yn", "svc_use_typ_cd", "indv_corp_cl_cd", "frgnr_yn", "nm_cust_num", "wlf_dc_cd"
) ++ sigFeaturesMMKT).filter(c => allFeaturesMMKT.contains(c.trim.split(" ")(0).trim)).distinct

val mmktDFTemp = spark.sql(s"""select ${colListForMMKT.mkString(",")}, strd_ym from wind_tmt.mmkt_svc_bas_f a where strd_ym in (${(featureYmList++sendYmList).distinct.sorted.mkString("'","','","'")})""")
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


// =============================================================================
// Paragraph 9: Train/Test Split and Feature Month Mapping
// =============================================================================

// =============================================================================
// [작업 흐름 2] Train/Test Split 및 Feature Month Mapping
// =============================================================================
// 이 단계는 Paragraph 4에서 정의된 시간 조건 변수를 사용합니다:
// - predictionDTSta: 테스트 데이터 시작 날짜
// - predictionDTEnd: 테스트 데이터 종료 날짜
//
// Train: send_dt < predictionDTSta
// Test:  send_dt >= predictionDTSta and send_dt < predictionDTEnd
// =============================================================================

println("=" * 80)
println("[작업 흐름 2] Train/Test Split and Feature Month Mapping")
println("=" * 80)
println(s"Split configuration:")
println(s"  - Training: send_dt < $predictionDTSta")
println(s"  - Testing:  send_dt >= $predictionDTSta and send_dt < $predictionDTEnd")
println("=" * 80)

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


// =============================================================================
// Paragraph 10: Undersampling Ratio Calculation (Class Balance)
// =============================================================================

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


// =============================================================================
// Paragraph 11: Training Data Undersampling (Balanced Dataset)
// =============================================================================

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


// =============================================================================
// Paragraph 12: App Usage Data Loading and Aggregation (Large Dataset)
// =============================================================================

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


// =============================================================================
// Paragraph 13
// =============================================================================

%spark.sql
SELECT
    -- date_format(
    --     from_utc_timestamp(
    --         from_unixtime(summary_create_time),
    --         "Asia/Seoul"
    --     ),
    --     "yyyy-MM-dd HH:mm:ss"
    -- ) AS summary_create_time,

    distinct
    date_format(
        from_utc_timestamp(
            from_unixtime(traffic_first_time),
            "Asia/Seoul"
        ),
        "yyyy-MM-dd HH:mm:ss"
    ) AS traffic_first_time,

    svc_mgmt_num,
    app_id,
    app_title_ko
FROM dprobe_raw.xdr_app
WHERE svc_mgmt_num = 's:e4e78b17ec7efcbc554478829a9272da96a34a40d73dbdf39a9bbad8dc9d83b7'
    AND dt = '20251216'
    AND hh >= 10 and hh <= 19
    and app_id = 'H032'
ORDER BY traffic_first_time
LIMIT 100;


// =============================================================================
// Paragraph 14: Historical Click Count Feature Engineering (3-Month Window)
// =============================================================================

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


// =============================================================================
// Paragraph 15: Feature Integration via Multi-way Joins (Optimized Order)
// =============================================================================

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


// =============================================================================
// Paragraph 16: Data Type Conversion and Column Standardization
// =============================================================================

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


// =============================================================================
// Paragraph 17: Raw Feature Data Persistence (Parquet Format)
// =============================================================================

// =============================================================================
// [작업 흐름 2] Train/Test Raw 데이터 저장
// =============================================================================
// 이 단계에서 생성된 데이터는 Paragraph 15에서 로딩됩니다.
// 저장 경로와 버전은 transformRawDataVersion과 일치해야 합니다.
//
// ⚠️  중요: 저장 경로는 Paragraph 15의 로딩 경로와 일치해야 합니다.
// =============================================================================

println("=" * 80)
println("[작업 흐름 2] Saving Train/Test Raw Feature Data")
println("=" * 80)

// 저장 전 파티션 수 조정하여 small files 문제 방지
// partitionBy()가 컬럼 기준 분배를 처리하므로 repartition(n)만 사용
val trainSavePath = s"aos/sto/trainDFRev${genSampleNumMulti.toInt}"
val testSavePath = s"aos/sto/testDFRev"

println(s"Saving training data to: $trainSavePath")
trainDFRev
.repartition(50)  // 각 파티션 디렉토리당 파일 수 조정
.write
.mode("overwrite")
.partitionBy("send_ym", "send_hournum_cd", "suffix")
.option("compression", "snappy")  // 압축 사용
.parquet(trainSavePath)

println(s"Training data saved successfully")

println(s"Saving test data to: $testSavePath")
testDFRev
.repartition(100)  // 각 파티션 디렉토리당 파일 수 조정
.write
.mode("overwrite")
.partitionBy("send_ym", "send_hournum_cd", "suffix")
.option("compression", "snappy")
.parquet(testSavePath)

println(s"Test data saved successfully")

println("=" * 80)
println("Raw feature data persistence completed")
println(s"  - Training: $trainSavePath")
println(s"  - Test: $testSavePath")
println(s"  - Period: ${sendYmList.mkString(", ")}")
println(s"  - Train/Test Split: < $predictionDTSta (train) / >= $predictionDTSta (test)")
println("=" * 80)


// =============================================================================
// Paragraph 18: Raw Feature Data Loading and Cache Management
// =============================================================================

// =============================================================================
// [작업 흐름 3] Transformed Data 생성을 위한 Raw Data 로딩
// =============================================================================
// 이 단계는 Paragraph 4에서 정의된 시간 조건 변수를 사용합니다:
// - transformRawDataVersion: 로딩할 raw training data 버전
// - transformTestDataPath: 로딩할 raw test data 경로
//
// ⚠️  중요: 이 경로들은 Paragraph 14에서 저장한 경로와 일치해야 합니다.
// =============================================================================

// 이전 단계에서 사용한 캐시 정리 (메모리 확보)
println("=" * 80)
println("[작업 흐름 3] Raw Feature Data Loading for Transformation")
println("=" * 80)
println("Cleaning up intermediate cached data...")
// try {
//   resDFSelectedTrBal.unpersist()
//   resDFSelectedTs.unpersist()
//   xdrDF.unpersist()
//   xdrDFMon.unpersist()
//   userYmDF.unpersist()
//   mmktDFFiltered.unpersist()
//   println("Cache cleanup completed")
// } catch {
//   case e: Exception => println(s"Cache cleanup warning: ${e.getMessage}")
// }

// spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "10g")

// Paragraph 4의 시간 조건 변수를 사용하여 데이터 로딩
val trainDataPath = s"aos/sto/trainDFRev${transformRawDataVersion}"
val testDataPath = s"aos/sto/${transformTestDataPath}"

println(s"Loading training data from: $trainDataPath")
val trainDFRev = spark.read.parquet(trainDataPath)
    .withColumn("hour_gap", F.expr("case when res_utility>=1.0 then 1 else 0 end"))
    // .drop("click_cnt").join(clickCountDF, Seq("svc_mgmt_num", "feature_ym", "send_hournum_cd"), "left").na.fill(Map("click_cnt"->0.0))
    .persist(StorageLevel.MEMORY_AND_DISK_SER)

// println(s"Training data loaded and cached: ${trainDFRev.count()} records")

println(s"Loading test data from: $testDataPath")
val testDFRev = spark.read.parquet(testDataPath)
    .withColumn("hour_gap", F.expr("case when res_utility>=1.0 then 1 else 0 end"))
    // .drop("click_cnt").join(clickCountDF, Seq("svc_mgmt_num", "feature_ym", "send_hournum_cd"), "left").na.fill(Map("click_cnt"->0.0))
    .persist(StorageLevel.MEMORY_AND_DISK_SER)

// println(s"Test data loaded and cached: ${testDFRev.count()} records")

// 피처 컬럼 분석
println("Analyzing feature columns...")
val noFeatureCols = Array("click_yn","hour_gap","chnl_typ","cmpgn_typ")
val tokenCols = trainDFRev.columns.filter(x => x.endsWith("_token")).distinct
val continuousCols = (trainDFRev.columns.filter(x => numericColNameList.map(x.endsWith(_)).reduceOption(_ || _).getOrElse(false)).distinct.filter(x => !tokenCols.contains(x) && !noFeatureCols.contains(x))).distinct
val categoryCols = (trainDFRev.columns.filter(x => categoryColNameList.map(x.endsWith(_)).reduceOption(_ || _).getOrElse(false)).distinct.filter(x => !tokenCols.contains(x) && !noFeatureCols.contains(x) && !continuousCols.contains(x))).distinct
val vectorCols = trainDFRev.columns.filter(x => x.endsWith("_vec"))

println(s"Feature column summary:")
println(s"  - Token columns: ${tokenCols.length}")
println(s"  - Continuous columns: ${continuousCols.length}")
println(s"  - Category columns: ${categoryCols.length}")
println(s"  - Vector columns: ${vectorCols.length}")
println("=" * 80)


// =============================================================================
// Paragraph 19: Prediction Dataset Preparation for Production
// =============================================================================

val predDT = "20260101"
val predFeatureYM = getPreviousMonths(predDT.take(6), 2)(0)
val predSendYM = predDT.take(6)

val prdSuffix = "%"

println("=" * 80)
println("[작업 흐름 5] Prediction Dataset Preparation for Production")
println("=" * 80)
println(s"Prediction configuration:")
println(s"  - Prediction Date: $predDT")
println(s"  - Feature Month: $predFeatureYM")
println(s"  - Send Month: $predSendYM")
println(s"  - Suffix Pattern: $predSuffix")
println(s"  - Hour Range: ${startHour} ~ ${endHour}")
println("=" * 80)

// Suffix 조건 생성
val prdSuffixCond = predSuffix.map(c => s"svc_mgmt_num like '%${c}'").mkString(" or ")
val pivotColumns = hourRange.toList.map(h => f"$h, total_traffic_$h%02d").mkString(", ")

// Prediction용 앱 사용 데이터 로딩 최적화
println(s"Loading prediction XDR data for feature_ym=$predFeatureYM...")
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

println(s"Prediction XDR hourly data cached with traffic features: ${xdrDFPred.count()} records")

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


// =============================================================================
// Paragraph 20: Pipeline Parameters and Feature Column Settings
// =============================================================================

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


// =============================================================================
// Paragraph 21: Feature Engineering Pipeline Function Definition (makePipeline)
// =============================================================================

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


// =============================================================================
// Paragraph 22: Feature Engineering Pipeline Fitting and Transformation
// =============================================================================

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


// =============================================================================
// Paragraph 23: Transformer and Transformed Data Saving (Batch by Suffix)
// =============================================================================

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


// =============================================================================
// Paragraph 24: Pipeline Transformers and Transformed Data Loading
// =============================================================================

val sendMonth = "202512"
val featureMonth = "202511"
val period = 6
val sendYmList = getPreviousMonths(sendMonth, period+2)
val featureYmList = getPreviousMonths(featureMonth, period+2)


println("Loading Click transformer...")
val transformerClick = PipelineModel.load("aos/sto/transformPipelineXDRClick10")

println("Loading Gap transformer...")
val transformerGap = PipelineModel.load("aos/sto/transformPipelineXDRGap10")

println("Loading transformed training data...")
val transformedTrainDF = spark.read.parquet("aos/sto/transformedTrainDFXDR10")
    .persist(StorageLevel.MEMORY_AND_DISK_SER)

println("Transformed training data loaded and cached")

println("Loading transformed test data...")
val transformedTestDF = spark.read.parquet("aos/sto/transformedTestDFXDF10")
    .persist(StorageLevel.MEMORY_AND_DISK_SER)

println("Transformed test data loaded and cached")


// =============================================================================
// Paragraph 25: ML Model Definitions (GBT, FM, XGBoost, LightGBM)
// =============================================================================

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


// =============================================================================
// Paragraph 26: XGBoost Feature Interaction and Monotone Constraints
// =============================================================================

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


// =============================================================================
// Paragraph 27: Click Prediction Model Training
// =============================================================================

import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator

val modelClickforCV = gbtc

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
    // .persist(StorageLevel.MEMORY_AND_DISK_SER)

println("Click model training samples prepared and cached")

println("Training Click prediction model...")
val pipelineModelClick = pipelineMLClick.fit(trainSampleClick.select(indexedFeatureColClick, indexedLabelColClick))

trainSampleClick.unpersist()  // 학습 완료 후 메모리 해제
println("Click model training completed")


// =============================================================================
// Paragraph 28: Click-to-Action Gap Model Training
// =============================================================================

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
    // .persist(StorageLevel.MEMORY_AND_DISK_SER)

println("Gap model training samples prepared and cached")

println("Training Gap prediction model...")
val pipelineModelGap = pipelineMLGap.fit(trainSampleGap.select(indexedFeatureColGap, indexedLabelColGap))

trainSampleGap.unpersist()  // 학습 완료 후 메모리 해제
println("Gap model training completed")


// =============================================================================
// Paragraph 29: Prediction Model Saving
// =============================================================================

val modelVersion = 1

pipelineModelClick.write.overwrite().save(s"aos/sto/model/pipelineModelClick${modelVersion}")
pipelineModelGap.write.overwrite().save(s"aos/sto/model/pipelineModelGap${modelVersion}")


// =============================================================================
// Paragraph 30: Prediction Model Loading
// =============================================================================

val modelVersion = 1

val pipelineModelClick = PipelineModel.load(s"aos/sto/model/pipelineModelClick${modelVersion}")
val pipelineModelGap = PipelineModel.load(s"aos/sto/model/pipelineModelGap${modelVersion}")


// =============================================================================
// Paragraph 31: Response Utility Regression Model Training
// =============================================================================

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


// =============================================================================
// Paragraph 32: Model Prediction on Test Dataset
// =============================================================================

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


// =============================================================================
// Paragraph 33
// =============================================================================

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

val modelClickforCV = gbtc

val pipelineMLClick = new Pipeline().setStages(Array(modelClickforCV))

val negSampleRatioClick = 0.1

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
    .repartition(200)  // 학습을 위한 적절한 파티션 수
    // .persist(StorageLevel.MEMORY_AND_DISK_SER)

println("Click model training samples prepared and cached")

println("Training Click prediction model...")
val pipelineModelClick = pipelineMLClick.fit(trainSampleClick)

trainSampleClick.unpersist()  // 학습 완료 후 메모리 해제
println("Click model training completed")


println("Generating Click predictions...")
val predictionsClickDev = pipelineModelClick.transform(testDataForPred)
    .persist(StorageLevel.MEMORY_AND_DISK_SER)

println("Click predictions cached")


// =============================================================================
// Paragraph 34: Click Model Performance Evaluation (Precision@K per Hour & MAP)
// =============================================================================

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


// =============================================================================
// Paragraph 35: Gap Model Performance Evaluation (Precision, Recall, F1)
// =============================================================================

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


// =============================================================================
// Paragraph 36: Regression Model Performance Evaluation (RMSE)
// =============================================================================

import org.apache.spark.ml.evaluation.RegressionEvaluator

// val stages = pipelineModelReg.bestModel.asInstanceOf[PipelineModel].stages
val stages = pipelineModelReg.stages

stages.foreach{stage =>

    val modelName = stage.uid.split("_")(0)

    val evaluator = new RegressionEvaluator()
      .setLabelCol(indexedLabelColReg)
      .setPredictionCol(s"pred_${modelName}")
      .setMetricName("rmse")
    val rmse = evaluator.evaluate(predictionsDev.filter("click_yn>0"))
    println(s"######### $modelName 예측 결과 #########")
    println(f"Root Mean Squared Error (RMSE) : $rmse%.4f")
}


// =============================================================================
// Paragraph 37: Propensity Score Calculation and Persistence (Batch by Suffix)
// =============================================================================

import org.apache.spark.ml.linalg.Vector

val predDT = "20260101"
val predFeatureYM = getPreviousMonths(predDT.take(6), 2)(0)
val predSendYM = predDT.take(6)

val pivotColumns = hourRange.toList.map(h => f"$h, total_traffic_$h%02d").mkString(", ")

val suffix = z.input("suffix", "0").toString

(0 to 15).map(_.toHexString).foreach{suffix =>

// val suffix = "000"

    println(suffix)

    val prdSuffixCond = suffix.map(c => s"svc_mgmt_num like '%${c}'").mkString(" or ")

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
    .repartition(400, F.col("svc_mgmt_num"), F.col("feature_ym"), F.col("send_hournum_cd"))
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
    // .persist(StorageLevel.MEMORY_AND_DISK_SER)

    println(s"Prediction XDR hourly data cached with traffic features: ${xdrDFPred.count()} records")

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
        // .persist(StorageLevel.MEMORY_AND_DISK_SER)

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
        // .persist(StorageLevel.MEMORY_AND_DISK_SER)

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
        .repartition(400, F.col("svc_mgmt_num"), F.col("feature_ym"), F.col("send_hournum_cd"))
        .join(clickCountDF, Seq("svc_mgmt_num", "feature_ym", "send_hournum_cd"), "left")  // 1단계: 가장 작은 테이블
        .na.fill(Map("click_cnt" -> 0.0))
        .join(xdrDFPred, Seq("svc_mgmt_num", "feature_ym", "send_hournum_cd"), "left")  // 2단계: 세 번째로 작은 테이블
        .repartition(400, F.col("svc_mgmt_num"), F.col("feature_ym"))  // xdrPredDF 조인을 위한 파티셔닝
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
    // .distinct
    // .persist(StorageLevel.MEMORY_AND_DISK_SER)

    val transformedPredDF = transformerGap.transform(transformerClick.transform(
        predDFRev
        // .filter(s"svc_mgmt_num like '%${suffix}'")
    ))//.cache()

    val predictionsSVCClick = pipelineModelClick.transform(transformedPredDF)
    val predictionsSVCFinal = pipelineModelGap.transform(predictionsSVCClick)

    var predictedPropensityScoreDF = predictionsSVCFinal
    .withColumn("prob_click", F.expr(s"""aggregate(array(${pipelineModelClick.stages.map(m => s"vector_to_array(prob_${m.uid})[1]").mkString(",")}), 0D, (acc, x) -> acc + x)"""))
    .withColumn("prob_gap", F.expr(s"""aggregate(array(${pipelineModelGap.stages.map(m => s"vector_to_array(prob_${m.uid})[1]").mkString(",")}), 0D, (acc, x) -> acc + x)"""))
    // .withColumn("res_utility", F.expr(s"""aggregate(array(${pipelineModelReg.stages.map(m => s"pred_${m.uid}").mkString(",")}), 0D, (acc, x) -> acc + x)"""))
    .withColumn("propensity_score", F.expr("prob_click*prob_gap"))

    predictedPropensityScoreDF.selectExpr("default.decodekey(svc_mgmt_num, svc_mgmt_num) svc_mgmt_num", "send_ym","send_hournum_cd send_hour"
    ,"ROUND(prob_click, 4) prob_click"
    ,"ROUND(prob_gap, 4) prob_gap"
    // ,"ROUND(res_utility, 4) res_utility"
    ,"ROUND(propensity_score, 4) propensity_score")
    .withColumn("suffix", F.lit(suffix))
    // .repartition(10)
    .write.mode("overwrite").partitionBy("send_ym", "send_hour", "suffix").parquet("aos/sto/mms_score")
    // .sort("svc_mgmt_num","send_hour").show()

}