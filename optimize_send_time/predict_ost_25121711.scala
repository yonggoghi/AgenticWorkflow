// ===== paragraph_1764658338256_686533166 =====
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


// ===== paragraph_1764742922351_426209997 =====
import java.time.YearMonth
import java.time.format.DateTimeFormatter

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


// ===== paragraph_1764638859660_1599316492 =====

val df = spark.sql("""
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

""").cache()

df.createOrReplaceTempView("df_view")

// ===== paragraph_1764656596503_575732907 =====

df.write.mode("overwrite").partitionBy("send_ym").parquet("aos/sto/response")

// ===== paragraph_1764659911196_1763551717 =====

val resDF = spark.read.parquet("aos/sto/response").cache()
resDF.createOrReplaceTempView("res_df")

// ===== paragraph_1764638864655_898851625 =====

z.show(resDF.filter("click_hournum is not null"))


// ===== paragraph_1764742953919_436300403 =====
val sendMonth = "202512"
val featureMonth = "202511"
val period = 6
val sendYmList = getPreviousMonths(sendMonth, period+2)
val featureYmList = getPreviousMonths(featureMonth, period+2)


// ===== paragraph_1764641394585_598529380 =====
val upperHourGap = 1
val startHour = 10
val endHour = 17

val resDFFiltered = resDF
    .filter(s"""send_ym in (${sendYmList.mkString(",")})""")
    .filter(s"hour_gap is null or (hour_gap between 0 and 5)")
    .filter(s"send_hournum between $startHour and $endHour")
    // .filter("cmpgn_typ=='Sales'")
    .selectExpr("cmpgn_num","svc_mgmt_num","chnl_typ","cmpgn_typ","send_ym","send_dt","send_time","send_daynum","send_hournum","click_dt","click_time","click_daynum","click_hournum","case when hour_gap is null then 0 else 1 end click_yn","hour_gap")
    .withColumn("res_utility", F.expr(s"case when hour_gap is null then 0.0 else 1.0 / (1 + hour_gap) end"))
    .dropDuplicates()
    .cache()

// ===== paragraph_1764739202982_181479704 =====
val allFeaturesMMKT = spark.sql("describe wind_tmt.mmkt_svc_bas_f").select("col_name").collect().map(_.getString(0))
val sigFeaturesMMKT = spark.read.option("header", "true").csv("feature_importance/table=mmkt_bas/creation_dt=20230407").filter("rank<=100").select("col").collect().map(_ (0).toString()).map(_.trim)
val colListForMMKT = (Array("svc_mgmt_num", "strd_ym feature_ym", "mst_work_dt", "cust_birth_dt", "prcpln_last_chg_dt", "fee_prod_id", "eqp_mdl_cd", "eqp_acqr_dt", "equip_chg_cnt", "svc_scrb_dt", "chg_dt", "cust_age_cd", "sex_cd", "equip_chg_day", "last_equip_chg_dt", "prev_equip_chg_dt", "rten_pen_amt", "agrmt_brch_amt", "eqp_mfact_cd"
  , "allot_mth_cnt", "mbr_use_cnt", "mbr_use_amt", "tyr_mbr_use_cnt", "tyr_mbr_use_amt", "mth_cnsl_cnt", "dsat_cnsl_cnt", "simpl_ref_cnsl_cnt", "arpu", "bf_m1_arpu", "voc_arpu", "bf_m3_avg_arpu"
  , "tfmly_nh39_scrb_yn", "prcpln_chg_cnt", "email_inv_yn", "copn_use_psbl_cnt", "data_gift_send_yn", "data_gift_recv_yn", "equip_chg_mth_cnt", "dom_tot_pckt_cnt", "scrb_sale_chnl_cl_cd", "op_sale_chnl_cl_cd", "agrmt_dc_end_dt"
  , "svc_cd", "svc_st_cd", "pps_yn", "svc_use_typ_cd", "indv_corp_cl_cd", "frgnr_yn", "nm_cust_num", "wlf_dc_cd"
)
  ++ sigFeaturesMMKT).filter(c => allFeaturesMMKT.contains(c.trim.split(" ")(0).trim)).distinct

val mmktDFTemp = spark.sql(s"""select ${colListForMMKT.mkString(",")}, strd_ym from wind_tmt.mmkt_svc_bas_f a where strd_ym in (${featureYmList.mkString(",")})""")

val mmktDF = {
  mmktDFTemp
    .join(spark.sql("select prod_id fee_prod_id, prod_nm fee_prod_nm from wind.td_zprd_prod"), Seq("fee_prod_id"), "left")
    .filter("cust_birth_dt not like '9999%'")
    // .cache()
}


// ===== paragraph_1764739017819_1458690185 =====
val predictionDTSta = "20251101"
val predictionDTEnd = "20251201"


val resDFSelected = resDFFiltered
    .withColumn("feature_ym", F.date_format(F.add_months(F.unix_timestamp($"send_dt", "yyyyMMdd").cast(TimestampType), -1), "yyyyMM").cast(StringType))
    .selectExpr("cmpgn_num", "svc_mgmt_num", "chnl_typ", "cmpgn_typ", "send_ym", "send_dt", "feature_ym", "send_daynum", "send_hournum send_hournum_cd", "click_yn", "res_utility")
    .dropDuplicates("svc_mgmt_num", "chnl_typ", "cmpgn_typ", "send_ym", "send_hournum_cd", "click_yn")
    // .cache()

val resDFSelectedTr = resDFSelected.filter(s"send_dt<$predictionDTSta")
// .groupBy("svc_mgmt_num","send_ym","feature_ym").agg(F.sum("click_yn").alias("click_yn"))
// .withColumn("click_yn", F.expr("case when click_yn>0 then 1 else 0 end"))
//.cache()

var resDFSelectedTs = resDFSelected.filter(s"send_dt>=$predictionDTSta and send_dt<$predictionDTEnd")
    .selectExpr("cmpgn_num", "svc_mgmt_num", "chnl_typ", "cmpgn_typ", "send_ym", "send_dt", "feature_ym", "send_hournum_cd", "click_yn", "res_utility")
    // .cache()


// ===== paragraph_1764738582669_1614068999 =====
val samplingKeyCols = Array("chnl_typ","cmpgn_typ","send_daynum","send_hournum_cd","click_yn")
// val samplingKeyCols = Array("send_ym", "feature_ym", "click_yn")

val genSampleNumMulti = 3.0

val samplingRatioMapDF = {
  resDFSelectedTr
    .sample(0.3)
    .groupBy(samplingKeyCols.map(F.col(_)):_*).agg(F.count("*").alias("cnt"))
    .withColumn("min_cnt", F.min("cnt").over(Window.partitionBy(samplingKeyCols.filter(_!="click_yn").map(F.col(_)):_*)))
    .withColumn("ratio", F.col("min_cnt") / F.col("cnt"))
    .withColumn("sampling_col", F.expr(s"""concat_ws('-', ${samplingKeyCols.mkString(",")})"""))
    .selectExpr("sampling_col", s"least(1.0, ratio*${genSampleNumMulti}) ratio")
    .sort("sampling_col")
    // .cache()
}


// ===== paragraph_1764756027560_85739584 =====
var resDFSelectedTrBal = resDFSelectedTr
    .withColumn("sampling_col", F.expr(s"""concat_ws('-', ${samplingKeyCols.mkString(",")})"""))
    .join(F.broadcast(samplingRatioMapDF), "sampling_col")
    .withColumn("rand", F.rand())
    .filter("rand<=ratio")
    // .drop("sampling_col","rand","ratio")
    // .withColumn("feature_ym", F.date_format(F.add_months(F.unix_timestamp($"send_dt", "yyyyMMdd").cast(TimestampType), -1), "yyyyMM").cast(StringType))
    // .selectExpr("cmpgn_num", "svc_mgmt_num", "chnl_typ", "cmpgn_typ", "send_ym", "send_dt", "feature_ym", "send_hournum", "click_yn")
    // .cache()

// resDFSelectedTrBal.groupBy("click_yn").count().sort("click_yn").show()
// resDFSelectedTs.groupBy("click_yn").count().show()

// ===== paragraph_1764755002817_1620624445 =====

val trainDF = resDFSelectedTrBal.join(mmktDF,Seq("svc_mgmt_num","feature_ym"))
val testDF = resDFSelectedTs.join(mmktDF,Seq("svc_mgmt_num","feature_ym"))

// ===== paragraph_1764832142136_413314670 =====
val noFeatureCols = Array("click_yn")


val tokenCols = trainDF.columns.filter(x => x.endsWith("_token")).distinct
val continuousCols = (trainDF.columns.filter(x => numericColNameList.map(x.endsWith(_)).reduceOption(_ || _).getOrElse(false)).distinct.filter(x => !tokenCols.contains(x) && !noFeatureCols.contains(x))).distinct
val categoryCols = (trainDF.columns.filter(x => categoryColNameList.map(x.endsWith(_)).reduceOption(_ || _).getOrElse(false)).distinct.filter(x => !tokenCols.contains(x) && !noFeatureCols.contains(x) && !continuousCols.contains(x))).distinct
val vectorCols = trainDF.columns.filter(x => x.endsWith("_vec"))

val trainDFRev = trainDF.select(
        // (Array("cmpgn_num", "svc_mgmt_num", "chnl_typ", "cmpgn_typ", "send_ym", "send_dt", "feature_ym", "click_yn").map(F.col(_))
        (Array("svc_mgmt_num", "send_ym", "feature_ym", "click_yn", "res_utility").map(F.col(_))
        ++ tokenCols.map(cl => F.coalesce(F.col(cl), F.array(F.lit("#"))).alias(cl))
        ++ vectorCols.map(cl => F.col(cl).alias(cl))
        ++ categoryCols.map(cl => F.when(F.col(cl) === "", F.lit("UKV")).otherwise(F.coalesce(F.col(cl).cast("string"), F.lit("UKV"))).alias(cl))
        ++ continuousCols.map(cl => F.coalesce(F.col(cl).cast("float"), F.lit(Double.NaN)).alias(cl))
        ).distinct
        : _*)
        //.cache()

val testDFRev = testDF.select(
        (Array("cmpgn_num", "svc_mgmt_num", "chnl_typ", "cmpgn_typ", "send_ym", "send_dt", "feature_ym", "click_yn", "res_utility").map(F.col(_))
        ++ tokenCols.map(cl => F.coalesce(F.col(cl), F.array(F.lit("#"))).alias(cl))
        ++ vectorCols.map(cl => F.col(cl).alias(cl))
        ++ categoryCols.map(cl => F.when(F.col(cl) === "", F.lit("UKV")).otherwise(F.coalesce(F.col(cl).cast("string"), F.lit("UKV"))).alias(cl))
        ++ continuousCols.map(cl => F.coalesce(F.col(cl).cast("float"), F.lit(Double.NaN)).alias(cl))
        ).distinct
        : _*)
        //.cache()

// ===== paragraph_1765765120629_645290475 =====
val predDT = "20251201"
val predFeatureYM = getPreviousMonths(predDT.take(6), 2)(0)
val predSendYM = predDT.take(6)
val predDF = mmktDF.filter(s"strd_ym=='${predFeatureYM}'")
            .withColumn("feature_ym", F.col("strd_ym"))
            .withColumn("send_ym", F.expr(predSendYM))
            .withColumn("send_dt", F.expr(predDT))
            .withColumn("cmpgn_num", F.expr("'#'"))
            .withColumn("cmpgn_typ", F.expr("'#'"))
            .withColumn("chnl_typ", F.expr("'#'"))
            .withColumn("click_yn", F.expr("0"))
            .withColumn("res_utility", F.expr("0.0"))
            .withColumn("send_hournum_cd", F.explode(F.expr(s"array(${(startHour to endHour).toArray.mkString(",")})")))

val predDFRev = predDF.select(
        (Array("cmpgn_num", "svc_mgmt_num", "chnl_typ", "cmpgn_typ", "send_ym", "send_dt", "feature_ym", "click_yn", "res_utility").map(F.col(_))
        ++ tokenCols.map(cl => F.coalesce(F.col(cl), F.array(F.lit("#"))).alias(cl))
        ++ vectorCols.map(cl => F.col(cl).alias(cl))
        ++ categoryCols.map(cl => F.when(F.col(cl) === "", F.lit("UKV")).otherwise(F.coalesce(F.col(cl).cast("string"), F.lit("UKV"))).alias(cl))
        ++ continuousCols.map(cl => F.coalesce(F.col(cl).cast("float"), F.lit(Double.NaN)).alias(cl))
        ).distinct
        : _*)
        //.cache()
        

// ===== paragraph_1765330122144_909170709 =====

def makePipeline(
        labelCol: String, 
        indexedLabelCol: String, 
        indexedFeatureCol: String, 
        scaledFeatureCol: String, 
        selectedFeatureCol: String,
        tokenCols:Array[String], 
        vectorCols:Array[String], 
        continuousCols:Array[String], 
        categoryCols:Array[String],
        doNotHashingCateCols: Array[String],
        doNotHashingContCols: Array[String],
        useSelector:Boolean = false,
        useScaling:Boolean = false,
        tokenColsEmbStr:String = "#",
        featureHasherNumFeature:Int = 256,
        params:Map[String, Any]
    ) = {
        
    val minDF = params.get("minDF").getOrElse(1).asInstanceOf[Int]
    val minTF = params.get("minTF").getOrElse(5).asInstanceOf[Int]
    val embSize = params.get("embSize").getOrElse(10).asInstanceOf[Int]
    val vocabSize = params.get("vocabSize").getOrElse(30).asInstanceOf[Int]
    val numParts = params.get("numParts").getOrElse(10).asInstanceOf[Int]

    var featureListForAssembler = continuousCols ++ vectorCols
    
    import org.apache.spark.ml.{Pipeline, PipelineStage}
    val transformPipeline = new Pipeline().setStages(Array[PipelineStage]())//
    
    if (labelCol!=""){
        transformPipeline.setStages(Array(new StringIndexer().setInputCol(labelCol).setOutputCol(indexedLabelCol).setHandleInvalid("skip")))
    }
    
    val tokenColsEmb = tokenCols.filter(x => tokenColsEmbStr.split(",").map(x.contains(_)).reduceOption(_ || _).getOrElse(false))
    val tokenColsCnt = tokenCols.filter(!tokenColsEmb.contains(_))
    
    if (embSize > 0 && tokenColsEmb.size > 0) {
      val embEncoder = tokenColsEmb.map(c => new Word2Vec().setNumPartitions(numParts).setSeed(46).setVectorSize(embSize).setMinCount(minTF).setInputCol(c).setOutputCol(c + "_embvec"))
      transformPipeline.setStages(if(transformPipeline.getStages.isEmpty){embEncoder}else{transformPipeline.getStages++embEncoder})
      featureListForAssembler ++= tokenColsEmb.map(_ + "_embvec")
    }
    
    if (tokenColsCnt.size > 0) {
      val cntVerctorizer = tokenColsCnt.map(c => new HashingTF().setInputCol(c).setOutputCol(c + "_cntvec").setNumFeatures(vocabSize))
      transformPipeline.setStages(if(transformPipeline.getStages.isEmpty){cntVerctorizer
      }else{transformPipeline.getStages++cntVerctorizer
      })
      featureListForAssembler ++= tokenColsCnt.map(_ + "_cntvec")
      
    }
    
    if (featureHasherNumFeature > 0 && categoryCols.size > 0) {
        
      val featureHasher = new FeatureHasher().setNumFeatures(featureHasherNumFeature)
      .setInputCols((continuousCols++categoryCols)
      .filter(c => !doNotHashingContCols.contains(c))
      .filter(c => !doNotHashingCateCols.contains(c))
      ).setOutputCol("feature_hashed")
      
    
      transformPipeline.setStages(if(transformPipeline.getStages.isEmpty){Array(featureHasher)}else{transformPipeline.getStages++Array(featureHasher)})
      featureListForAssembler = featureListForAssembler.filter(!continuousCols.contains(_))
      featureListForAssembler ++= Array("feature_hashed")
    
      if (doNotHashingCateCols.size>0) {
          val catetoryIndexerList = doNotHashingCateCols.map(c => new StringIndexer().setInputCol(c).setOutputCol(c + "_index").setHandleInvalid("keep"))
          val encoder = new OneHotEncoder().setInputCols(doNotHashingCateCols.map(c => c + "_index")).setOutputCols(doNotHashingCateCols.map(c => c + "_enc")).setHandleInvalid("keep")
          transformPipeline.setStages(if(transformPipeline.getStages.isEmpty){catetoryIndexerList ++ Array(encoder)}else{transformPipeline.getStages++catetoryIndexerList ++ Array(encoder)})
          featureListForAssembler ++= doNotHashingCateCols.map(_ + "_enc")
      }
      
      if (doNotHashingContCols.size>0){
          featureListForAssembler ++= doNotHashingContCols
      }
      
    } else if (featureHasherNumFeature < 1 && categoryCols.size > 0) {
      val catetoryIndexerList = categoryCols.map(c => new StringIndexer().setInputCol(c).setOutputCol(c + "_index").setHandleInvalid("keep"))
      val encoder = new OneHotEncoder().setInputCols(categoryCols.map(c => c + "_index")).setOutputCols(categoryCols.map(c => c + "_enc")).setHandleInvalid("keep")
      transformPipeline.setStages(if(transformPipeline.getStages.isEmpty){catetoryIndexerList ++ Array(encoder)}else{transformPipeline.getStages++catetoryIndexerList ++ Array(encoder)})
      featureListForAssembler ++= categoryCols.map(_ + "_enc")
    }
    
    val assembler = new VectorAssembler().setInputCols(featureListForAssembler).setOutputCol(indexedFeatureCol).setHandleInvalid("keep")
    
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


// ===== paragraph_1764833771372_1110341451 =====

val tokenColsEmbStr = "#"
val featureHasherNumFeature = 128

var nodeNumber = 10
var coreNumber = 32
try {
  nodeNumber = spark.conf.get("spark.executor.instances").toInt
  coreNumber = spark.conf.get("spark.executor.cores").toInt
} catch {
  case ex: Exception => {}
}

val params:Map[String, Any] = Map("minDF"->1,"minTF"->5,"embSize"->10,"vocabSize"->30, "numParts"->nodeNumber)

val labelCol = "click_yn"
val indexedLabelColCls = "indexedLabelCls"
val indexedLabelColReg = "res_utility"
val indexedFeatureCol = "indexedFeatures"
val scaledFeatureCol = "scaledFeatures"
val selectedFeatureCol = "selectedFeatures"

// val tokenCols = tokenCols
// val vectorCols = vectorCols
// val continuousCols = continuousCols
// val categoryCols = categoryCols

val doNotHashingCateCols = Array[String]("send_hournum_cd")
val doNotHashingContCols = Array[String]()

val transformPipeline = makePipeline(
    labelCol, 
    indexedLabelColCls, 
    indexedFeatureCol, 
    scaledFeatureCol,
    selectedFeatureCol,
    tokenCols, 
    vectorCols, 
    continuousCols, 
    categoryCols,
    doNotHashingCateCols,
    doNotHashingContCols,
    params = params,
    useSelector = false,
    featureHasherNumFeature = featureHasherNumFeature
)

// val transformer = transformPipeline.fit(trainDFRev.sample(0.3))
// val transformedTrainDF = transformer.transform(trainDFRev).cache()
// val transformedTestDF = transformer.transform(testDFRev).cache()


// ===== paragraph_1765520460775_2098641576 =====
transformer.write.overwrite().save("aos/sto/transformPipeline")

transformedTrainDF.write.mode("overwrite").partitionBy("send_ym").parquet("aos/sto/transformedTrainDF")
transformedTestDF.write.mode("overwrite").partitionBy("send_ym").parquet("aos/sto/transformedTestDF")

// ===== paragraph_1765521446308_1651058139 =====
val transformer = PipelineModel.load("aos/sto/transformPipeline")
val transformedTrainDF = spark.read.parquet("aos/sto/transformedTrainDF").cache()
val transformedTestDF = spark.read.parquet("aos/sto/transformedTestDF").cache()

val transformedPredDF = transformer.transform(predDFRev).cache()


// ===== paragraph_1765445117259_700668975 =====
val labelIndexer: Option[StringIndexerModel] = transformer.stages.collectFirst {
  case sim: StringIndexerModel => sim
}

labelIndexer.get.labelsArray(0)

// ===== paragraph_1764836200898_700489598 =====
import org.apache.spark.ml.classification._
import org.apache.spark.ml.regression._

import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.StringIndexerModel

val labelIndexer: Option[StringIndexerModel] = transformer.stages.collectFirst {
  case sim: StringIndexerModel => sim
}

val featureColName = indexedFeatureCol//selectedFeatureCol

val gbtc = new GBTClassifier()
  .setLabelCol(indexedLabelColCls)
  .setFeaturesCol(featureColName)
  .setMaxIter(100)
//   .setMaxDepth(4)
  .setFeatureSubsetStrategy("auto")
//   .setWeightCol("sample_weight")
    .setPredictionCol("pred_gbtc")
  .setProbabilityCol("prob_gbtc")
  .setRawPredictionCol("pred_raw_gbtc")
  
val fmc = new FMClassifier()
    .setLabelCol(indexedLabelColCls)
    .setFeaturesCol(featureColName)
    .setStepSize(0.01)
    .setPredictionCol("pred_fmc")
    .setProbabilityCol("prob_fmc")
    .setRawPredictionCol("pred_raw_fmc")

val xgbParamC = Map(
  "eta" -> 0.01,
  "max_depth" -> 6,
  "objective" -> "binary:logistic",
  "num_round" -> 100,
  "num_workers" -> 10,
//   "num_early_stopping_rounds" -> 10,  // early stopping
//   "maximize_evaluation_metrics" -> true,  // loss 기준이면 false
  "eval_metric" -> "error",
//   "scale_pos_weight" -> 1.0
)

val xgbc = {
  new XGBoostClassifier(xgbParamC)
    .setFeaturesCol(featureColName)
    .setLabelCol(indexedLabelColCls)
    .setMissing(0)
    .setSeed(0)
    // .setWeightCol("sample_weight")
    .setProbabilityCol("prob_xgbc")
    .setPredictionCol("pred_xgbc")
      .setRawPredictionCol("pred_raw_xgbc")
    // .setThresholds(Array(0.4, 0.6))
    // .setEvalSets(Map("validation" -> valData))
}

val gbtr = new GBTRegressor()
  .setLabelCol(indexedLabelColReg)
  .setFeaturesCol(featureColName)
  .setMaxIter(100)
//   .setMaxDepth(4)
  .setFeatureSubsetStrategy("auto")
//   .setWeightCol("sample_weight")
    .setPredictionCol("pred_gbtr")

val fmr = new FMRegressor()
    .setLabelCol(indexedLabelColReg)
    .setFeaturesCol(featureColName)
    .setStepSize(0.01)
    .setPredictionCol("pred_fmr")

val xgbParamR = Map(
  "eta" -> 0.01,
  "max_depth" -> 6,
  "objective" -> "reg:squarederror",
  "num_round" -> 100,
  "num_workers" -> 10,
//   "num_early_stopping_rounds" -> 10,  // early stopping
//   "maximize_evaluation_metrics" -> true,  // loss 기준이면 false
  "eval_metric" -> "rmse",
//   "scale_pos_weight" -> 1.0
)

val xgbr = {
  new XGBoostRegressor(xgbParamR)
    .setFeaturesCol(featureColName)
    .setLabelCol(indexedLabelColReg)
    .setMissing(0)
    .setSeed(0)
    // .setWeightCol("sample_weight")
    .setPredictionCol("pred_xgbr")
    // .setThresholds(Array(0.4, 0.6))
    // .setEvalSets(Map("validation" -> valData))
}

val labelConverter = new IndexToString()
  .setInputCol("prediction_cls")
  .setOutputCol("predictedLabel")
  .setLabels(labelIndexer.get.labelsArray(0))

val Array(trainData, valData) = transformedTrainDF.randomSplit(Array(0.8, 0.2), seed = 42)


// ===== paragraph_1765841941095_360008212 =====
import org.apache.spark.ml.param.ParamMap
println(gbtc.explainParams())


// ===== paragraph_1765789893517_1550413688 =====

import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator

val modelforCV = gbtc

// val paramGridXGB = new ParamGridBuilder()
//         .addGrid(xgbc.maxDepth, Array(6))
//         .addGrid(xgbc.numRound, Array(100))
//         .addGrid(xgbc.eta, Array(0.1))
//         // .addGrid(xgbc.scalePosWeight, Array(3.0))
//         .build()

// val paramGridGBT = new ParamGridBuilder()
//         .addGrid(gbtc.maxDepth, Array(5))
//         .addGrid(gbtc.maxIter, Array(100))
//         .addGrid(gbtc.stepSize, Array(0.1))
//         .build()

// val paramGrid = if(modelforCV.uid.startsWith("xgb")){
//     paramGridXGB
// }else{
//     paramGridGBT
// }

// val pipelineMLCls = new CrossValidator()
//   .setEstimator(new Pipeline().setStages(Array(modelforCV)))
//   .setEvaluator(new BinaryClassificationEvaluator().setLabelCol(modelforCV.getLabelCol).setRawPredictionCol(modelforCV.getPredictionCol))
//   .setEstimatorParamMaps(paramGrid)
//   .setNumFolds(3)  // Use 3+ in practice
//   .setParallelism(6)  // Evaluate up to 2 parameter settings in parallel
  
val pipelineMLCls = new Pipeline().setStages(Array(modelforCV))

val pipelineModelCls = pipelineMLCls.fit(transformedTrainDF
    // .withColumn("sample_weight", F.expr("case when click_yn==0.0 then 1.0 else 1.0 end"))
    .stat.sampleBy(
            F.col("click_yn"),
            Map(
                0.0 -> 0.6,
                1.0 -> 1.0,
            ),
            42L
        )
)


// ===== paragraph_1765764610094_1504595267 =====
import org.apache.spark.ml.evaluation.RegressionEvaluator

val modelRegforCV = xgbr

// val paramGridXGB = new ParamGridBuilder()
//         .addGrid(xgbc.maxDepth, Array(6))
//         .addGrid(xgbc.numRound, Array(100))
//         .addGrid(xgbc.eta, Array(0.1))
//         // .addGrid(xgbc.scalePosWeight, Array(3.0))
//         .build()

// val paramGridGBT = new ParamGridBuilder()
//         .addGrid(gbtc.maxDepth, Array(5))
//         .addGrid(gbtc.maxIter, Array(100))
//         .addGrid(gbtc.stepSize, Array(0.1))
//         .build()

// val paramGrid = if(modelRegforCV.uid.startsWith("xgb")){
//     paramGridXGB
// }else{
//     paramGridGBT
// }

// val pipelineMLReg = new CrossValidator()
//   .setEstimator(new Pipeline().setStages(Array(modelRegforCV)))
//   .setEvaluator(new RegressionEvaluator().setLabelCol(modelRegforCV.getLabelCol).setPredictionCol(modelRegforCV.getPredictionCol))
//   .setEstimatorParamMaps(paramGrid)
//   .setNumFolds(3)  // Use 3+ in practice
//   .setParallelism(6)  // Evaluate up to 2 parameter settings in parallel
  
val pipelineMLReg = new Pipeline().setStages(Array(modelRegforCV))

val pipelineModelReg = pipelineMLReg.fit(
    transformedTrainDF
    .filter("click_yn>0")
// .withColumn("sample_weight", F.expr("case when click_yn==0.0 then 1.0 else 1.0 end"))
)

// ===== paragraph_1765345345715_612147457 =====

// Make predictions.
var predictionsDev = pipelineModelCls
.transform(transformedTestDF
.dropDuplicates("svc_mgmt_num", "chnl_typ", "cmpgn_typ", "send_ym", "send_hournum_cd", "click_yn")
)//.cache()

predictionsDev = pipelineModelReg.transform(predictionsDev).cache()

// ===== paragraph_1764838154931_1623772564 =====
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.functions._
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.tuning.CrossValidatorModel

import org.apache.spark.ml.linalg.Vector

spark.udf.register("vector_to_array", (v: Vector) => v.toArray)

val topK = 50000

// val stages = pipelineModelCls.bestModel.asInstanceOf[PipelineModel].stages
val stages = pipelineModelCls.stages

stages.foreach{stage => 
    
    val modelName = stage.uid.split("_")(0)
    
    predictionsDev
    .filter("cmpgn_typ=='Sales'")
    .groupBy("svc_mgmt_num", "send_ym", s"prob_$modelName").agg(F.sum(indexedLabelColCls).alias(indexedLabelColCls))

    // 1. prediction DataFrame에서 (prediction, indexedLabel) RDD[(Double, Double)] 생성
    val predictionAndLabels = labelConverter.transform(
    predictionsDev
    .filter("cmpgn_typ=='Sales'")
    .groupBy("svc_mgmt_num", "send_ym", s"prob_$modelName").agg(F.sum(indexedLabelColCls).alias(indexedLabelColCls))
    .withColumn(indexedLabelColCls, F.expr(s"case when $indexedLabelColCls>0 then cast(1.0 AS DOUBLE) else cast(0.0 AS DOUBLE) end"))
    .withColumn("prob", F.expr(s"vector_to_array(prob_$modelName)[1]"))
    .groupBy("svc_mgmt_num", "send_ym", indexedLabelColCls).agg(F.avg("prob").alias("prob"))
    .withColumn("rank", F.rank().over(Window.orderBy(F.desc("prob"))))
    .withColumn("prediction_cls", F.expr("case when prob>=0.5 then cast(1.0 AS DOUBLE) else cast(0.0 AS DOUBLE) end"))
    // .withColumn("prediction_cls", F.expr(s"case when rank<=${topK} then cast(1.0 AS DOUBLE) else cast(0.0 AS DOUBLE) end"))
    // .filter(f"rank<=${topK}")
    )
    .selectExpr("prediction_cls", s"cast($indexedLabelColCls as double)")
                                         .rdd
                                         .map(row => (row.getDouble(0), row.getDouble(1)))
    
    // 2. MulticlassMetrics 인스턴스 생성
    val metrics = new MulticlassMetrics(predictionAndLabels)
    
    // 3. 레이블별 지표 추출
    val labels = metrics.labels // 사용된 모든 고유 레이블 목록
    
    println(s"######### $modelName 예측 결과 #########")
    
    println("--- 레이블별 성능 지표 ---")
    labels.foreach { label =>
      val precision = metrics.precision(label) // 특정 레이블의 Precision
      val recall = metrics.recall(label)     // 특정 레이블의 Recall
      val f1 = metrics.fMeasure(label)       // 특정 레이블의 F1-Score
    
      println(f"Label $label (클래스): Precision = $precision%.4f, Recall = $recall%.4f, F1 = $f1%.4f")
    }
    
    // 4. (선택사항) 전체 가중 평균 지표 확인
    println(s"\nWeighted Precision (전체 평균): ${metrics.weightedPrecision}")
    println(s"Weighted Recall (전체 평균): ${metrics.weightedRecall}")
    println(s"Accuracy (전체 정확도): ${metrics.accuracy}")
    
    // 5. (선택사항) 혼동 행렬 출력
    println("\n--- Confusion Matrix (혼동 행렬) ---")
    println(metrics.confusionMatrix)
}

// ===== paragraph_1765786040626_1985577608 =====
import org.apache.spark.ml.evaluation.RegressionEvaluator

// val stages = pipelineModelReg.bestModel.asInstanceOf[PipelineModel].stages
val stages = pipelineModelReg.stages

stages.foreach{stage => 

    val modelName = stage.uid.split("_")(0)

    val evaluator = new RegressionEvaluator()
      .setLabelCol(indexedLabelColReg)
      .setPredictionCol(s"pred_${modelName}")
      .setMetricName("rmse")
    val rmse = evaluator.evaluate(predictionsDev)
    println(s"######### $modelName 예측 결과 #########")
    println(f"Root Mean Squared Error (RMSE) : $rmse%.4f")
}

// ===== paragraph_1765768974381_910321724 =====
val predictionsSVCCls = pipelineModelCls.transform(transformedPredDF.filter("svc_mgmt_num like '%00'").dropDuplicates("svc_mgmt_num", "chnl_typ", "cmpgn_typ", "send_ym", "send_hournum_cd", "click_yn"))
val predictionsSVCFinal = pipelineModelReg.transform(predictionsSVCCls)

import org.apache.spark.ml.linalg.Vector

spark.udf.register("vector_to_array", (v: Vector) => v.toArray)

var predictedPropensityScoreDF = predictionsSVCFinal
.withColumn("prob_click", F.expr(s"""aggregate(array(${pipelineModelCls.stages.map(m => s"vector_to_array(prob_${m.uid.split("_")(0)})[1]").mkString(",")}), 0D, (acc, x) -> acc + x)"""))
.withColumn("res_utility", F.expr(s"""aggregate(array(${pipelineModelReg.stages.map(m => s"pred_${m.uid.split("_")(0)}").mkString(",")}), 0D, (acc, x) -> acc + x)"""))
.withColumn("propensity_score", F.expr("prob_click*res_utility"))

predictedPropensityScoreDF.selectExpr("svc_mgmt_num","send_hournum_cd send_hour","ROUND(prob_click, 4) prob_click","ROUND(res_utility, 4) res_utility","ROUND(propensity_score, 4) propensity_score")
.write.mode("overwrite").partitionBy("send_hour").parquet("aos/sto/propensityScoreDF")
// .show()


// ===== paragraph_1765935603309_1543411151 =====
val pdf = spark.read.parquet("aos/sto/propensityScoreDF")
.select("svc_mgmt_num","send_hour","propensity_score").cache()

// pdf.sort("svc_mgmt_num")
// .show()

// ===== paragraph_1765935760016_917799410 =====
pdf.sort("svc_mgmt_num","send_hour")
.show()

// ===== paragraph_1764902404285_1154494058 =====

--- 2.0 레이블별 성능 지표 ---
Label 0.0 (클래스): Precision = 0.9878, Recall = 0.9182, F1 = 0.9517
Label 1.0 (클래스): Precision = 0.0330, Recall = 0.1972, F1 = 0.0565

--- 3.0 레이블별 성능 지표 ---
Label 0.0 (클래스): Precision = 0.9866, Recall = 0.9837, F1 = 0.9852
Label 1.0 (클래스): Precision = 0.0474, Recall = 0.0574, F1 = 0.0519

--- 4.0 레이블별 성능 지표 ---
Label 0.0 (클래스): Precision = 0.9863, Recall = 0.9942, F1 = 0.9902
Label 1.0 (클래스): Precision = 0.0552, Recall = 0.0241, F1 = 0.0336

