object PredictOst50BuildPredInputs {
  def run(): Unit = {
    import PredictOstConfig._
    initSpark()

    // [ZEPPELIN_PARAGRAPH] 50. Build inference (prediction) input DF
    import org.apache.spark.sql.types.{ArrayType, NumericType, StringType, TimestampType}
    import org.apache.spark.sql.functions._

    // [ZEPPELIN_PARAGRAPH] 51. Recompute feature columns from trainDFRev schema (for alignment)
    val trainSchemaDF = spark.read.parquet(trainDFRevPath).limit(1)

    val baseColsArr = Array("cmpgn_num", "svc_mgmt_num", "chnl_typ", "cmpgn_typ", "send_ym", "send_dt", "feature_ym", "hour_gap", "click_yn", "res_utility")
    val baseColsSet = baseColsArr.toSet

    val tokenCols = trainSchemaDF.schema.fields
      .collect { case f if f.name.endsWith("_token") && f.dataType.isInstanceOf[ArrayType] => f.name }
      .distinct

    val vectorCols = trainSchemaDF.columns.filter(_.endsWith("_vec")).distinct

    val continuousCols = trainSchemaDF.schema.fields
      .collect { case f if f.dataType.isInstanceOf[NumericType] && !tokenCols.contains(f.name) && !vectorCols.contains(f.name) && !baseColsSet.contains(f.name) => f.name }
      .distinct

    val categoryCols = trainSchemaDF.schema.fields
      .collect { case f if f.dataType == StringType && !tokenCols.contains(f.name) && !vectorCols.contains(f.name) && !baseColsSet.contains(f.name) => f.name }
      .distinct

    // [ZEPPELIN_PARAGRAPH] 52. Dates for inference
    val predFeatureYM = getPreviousMonths(predDT.take(6), 2)(0)
    val predSendYM = predDT.take(6)

    val suffixPredCsv = sys.env.getOrElse("SUFFIX_PRED", "%")
    val prdSuffixCond = buildSuffixLikeCond("svc_mgmt_num", suffixPredCsv)

    // [ZEPPELIN_PARAGRAPH] 53. XDR prediction tokens (hourly)
    val pivotColumns = hourRange.map(h => f"$h, total_traffic_$h%02d").mkString(", ")

    val xdrDFPred = spark.sql(
  s"""
     |SELECT
     |  svc_mgmt_num,
     |  ym AS feature_ym,
     |  COALESCE(rep_app_title, app_uid) AS app_nm,
     |  CAST(hour.send_hournum_cd AS STRING) AS send_hournum_cd,
     |  hour.traffic
     |FROM dprobe.mst_app_svc_app_monthly
     |LATERAL VIEW STACK(
     |  ${hourRange.size},
     |  $pivotColumns
     |) hour AS send_hournum_cd, traffic
     |WHERE hour.traffic > 1000
     |  AND ym = '$predFeatureYM'
     |  AND ($prdSuffixCond)
     |""".stripMargin
    )
      .groupBy("svc_mgmt_num", "feature_ym", "send_hournum_cd")
      .agg(collect_set("app_nm").alias("app_usage_token"))
      .cache()

    val xdrPredDF = xdrDFPred
  .groupBy("svc_mgmt_num", "feature_ym")
  .pivot("send_hournum_cd", hourRange.map(_.toString))
  .agg(first("app_usage_token"))
  .select(
    col("svc_mgmt_num") +: col("feature_ym") +:
      hourRange.map(h => coalesce(col(s"$h"), array(lit("#"))).alias(s"app_usage_${h}_token")): _*
    )

    // [ZEPPELIN_PARAGRAPH] 54. MMKT base features for inference month
    val allFeaturesMMKT = spark.sql("describe wind_tmt.mmkt_svc_bas_f").select("col_name").collect().map(_.getString(0))
    val sigFeaturesMMKT = spark.read
  .option("header", "true")
  .csv("feature_importance/table=mmkt_bas/creation_dt=20230407")
  .filter("rank<=100")
  .select("col")
  .collect()
  .map(_(0).toString())
  .map(_.trim)

    val colListForMMKT =
  (Array(
    "svc_mgmt_num",
    "strd_ym",
    "mst_work_dt",
    "cust_birth_dt",
    "prcpln_last_chg_dt",
    "fee_prod_id",
    "eqp_mdl_cd",
    "eqp_acqr_dt",
    "equip_chg_cnt",
    "svc_scrb_dt",
    "chg_dt",
    "cust_age_cd",
    "sex_cd",
    "equip_chg_day",
    "last_equip_chg_dt",
    "prev_equip_chg_dt",
    "rten_pen_amt",
    "agrmt_brch_amt",
    "eqp_mfact_cd",
    "allot_mth_cnt",
    "mbr_use_cnt",
    "mbr_use_amt",
    "tyr_mbr_use_cnt",
    "tyr_mbr_use_amt",
    "mth_cnsl_cnt",
    "dsat_cnsl_cnt",
    "simpl_ref_cnsl_cnt",
    "arpu",
    "bf_m1_arpu",
    "voc_arpu",
    "bf_m3_avg_arpu",
    "tfmly_nh39_scrb_yn",
    "prcpln_chg_cnt",
    "email_inv_yn",
    "copn_use_psbl_cnt",
    "data_gift_send_yn",
    "data_gift_recv_yn",
    "equip_chg_mth_cnt",
    "dom_tot_pckt_cnt",
    "scrb_sale_chnl_cl_cd",
    "op_sale_chnl_cl_cd",
    "agrmt_dc_end_dt",
    "svc_cd",
    "svc_st_cd",
    "pps_yn",
    "svc_use_typ_cd",
    "indv_corp_cl_cd",
    "frgnr_yn",
    "nm_cust_num",
    "wlf_dc_cd"
  ) ++ sigFeaturesMMKT)
    .filter(c => allFeaturesMMKT.contains(c.trim.split(" ")(0).trim))
    .distinct

    val mmktPredDF = spark.sql(
  s"""select ${colListForMMKT.mkString(",")}
     |from wind_tmt.mmkt_svc_bas_f a
     |where strd_ym = '$predFeatureYM'
     |""".stripMargin
    )
      .join(spark.sql("select prod_id fee_prod_id, prod_nm fee_prod_nm from wind.td_zprd_prod"), Seq("fee_prod_id"), "left")
      .filter("cust_birth_dt not like '9999%'")
      .filter(prdSuffixCond)

    // [ZEPPELIN_PARAGRAPH] 55. Click history count for inference month (n=3)
    val n = 3
    val resDF = spark.read.parquet(responsePath)
    val resWithFeatureYM = resDF.withColumn(
  "feature_ym",
  date_format(add_months(unix_timestamp(col("send_dt"), "yyyyMMdd").cast(TimestampType), -1), "yyyyMM").cast(StringType)
)
    val current = resWithFeatureYM.filter(col("feature_ym") === lit(predFeatureYM))
    val prevMonths = getPreviousMonths(predFeatureYM, n + 1).dropRight(1)
    val prevIn = prevMonths.map(m => s"'$m'").mkString(",")
    val previous = resWithFeatureYM.filter(s"feature_ym in ($prevIn)")

    val clickCountDF = current.as("current")
      .join(previous.as("previous"), col("current.svc_mgmt_num") === col("previous.svc_mgmt_num"), "left")
      .groupBy(col("current.svc_mgmt_num").as("svc_mgmt_num"), col("current.feature_ym").as("feature_ym"))
      .agg(sum(coalesce(col("previous.click_yn"), lit(0.0))).alias("click_cnt"))

    // [ZEPPELIN_PARAGRAPH] 56. Build predDF + predDFRev
    val sendHourArray = array(hourRange.map(h => lit(h.toString)): _*)

    val predDF = mmktPredDF
  .withColumn("feature_ym", col("strd_ym"))
  .withColumn("send_ym", lit(predSendYM))
  .withColumn("send_dt", lit(predDT))
  .withColumn("cmpgn_num", lit("#"))
  .withColumn("cmpgn_typ", lit("#"))
  .withColumn("chnl_typ", lit("#"))
  .withColumn("click_yn", lit(0))
  .withColumn("hour_gap", lit(0))
  .withColumn("res_utility", lit(0.0))
  .withColumn("send_hournum_cd", explode(sendHourArray))
  .join(xdrDFPred, Seq("svc_mgmt_num", "feature_ym", "send_hournum_cd"), "left")
  .join(xdrPredDF, Seq("svc_mgmt_num", "feature_ym"), "left")
  .join(clickCountDF, Seq("svc_mgmt_num", "feature_ym"), "left")
      .na.fill(Map("click_cnt" -> 0.0))

    val predDFRev = predDF
  .select(
    (baseColsArr.map(col) ++
      tokenCols.map(c => coalesce(col(c), array(lit("#"))).alias(c)) ++
      vectorCols.map(c => col(c).alias(c)) ++
      categoryCols.map(c => when(col(c) === "", lit("UKV")).otherwise(coalesce(col(c).cast("string"), lit("UKV"))).alias(c)) ++
      continuousCols.map(c => coalesce(col(c).cast("float"), lit(Double.NaN)).alias(c))
    ): _*
  )
  .distinct()
      .withColumn("suffix", expr("right(svc_mgmt_num, 1)"))

    predDFRev.createOrReplaceTempView("pred_df_rev")

    println(s"predDFRev ready. predDT=$predDT predFeatureYM=$predFeatureYM hourRange=$startHour-$endHour suffixPredCsv=$suffixPredCsv")
  }
}
