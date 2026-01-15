object PredictOst20BuildJoinedDatasets {
  def run(): Unit = {
    import PredictOstConfig._
    initSpark()

    // [ZEPPELIN_PARAGRAPH] 20. Build joined train/test datasets (feature join + sampling)
    // Output:
    // - `trainDFRevPath` (partitioned by send_ym, send_hournum_cd, suffix)
    // - `testDFRevPath`  (partitioned by send_ym, send_hournum_cd, suffix)

    import org.apache.spark.sql.DataFrame
    import org.apache.spark.sql.types.{NumericType, StringType, TimestampType}
    import org.apache.spark.sql.functions._

    // Ensure resDF is available
    val resDF = spark.read.parquet(responsePath).cache()
    resDF.createOrReplaceTempView("res_df")

    // [ZEPPELIN_PARAGRAPH] 21. Response filtering + label/utility
    val sendYmIn = sendYmList.map(ym => s"'$ym'").mkString(",")

    val resDFFiltered = resDF
  .filter(s"send_ym in ($sendYmIn)")
  .filter("hour_gap is null or (hour_gap between 0 and 5)")
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
    "case when hour_gap is null then 0 else 1 end click_yn",
    "hour_gap"
  )
  .withColumn("res_utility", expr("case when hour_gap is null then 0.0 else 1.0 / (1 + hour_gap) end"))
  .dropDuplicates()
      .cache()

    // [ZEPPELIN_PARAGRAPH] 22. MMKT base features loading
    val featureYmIn = featureYmList.map(ym => s"'$ym'").mkString(",")

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
    "strd_ym feature_ym",
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

    val mmktDFTemp = spark.sql(
  s"""select ${colListForMMKT.mkString(",")}, strd_ym
     |from wind_tmt.mmkt_svc_bas_f a
     |where strd_ym in ($featureYmIn)
     |""".stripMargin
)

    val mmktDF = mmktDFTemp
  .join(spark.sql("select prod_id fee_prod_id, prod_nm fee_prod_nm from wind.td_zprd_prod"), Seq("fee_prod_id"), "left")
  .filter("cust_birth_dt not like '9999%'")
  .checkpoint()

    // [ZEPPELIN_PARAGRAPH] 23. Train/Test split by date + send_hournum_cd
    val resDFSelected = resDFFiltered
  .withColumn(
    "feature_ym",
    date_format(add_months(unix_timestamp(col("send_dt"), "yyyyMMdd").cast(TimestampType), -1), "yyyyMM").cast(StringType)
  )
  .selectExpr(
    "cmpgn_num",
    "svc_mgmt_num",
    "chnl_typ",
    "cmpgn_typ",
    "send_ym",
    "send_dt",
    "feature_ym",
    "send_daynum",
    "cast(send_hournum as string) send_hournum_cd",
    "hour_gap",
    "click_yn",
    "res_utility"
  )
  .dropDuplicates("svc_mgmt_num", "chnl_typ", "cmpgn_typ", "send_ym", "send_hournum_cd", "click_yn")

    val resDFSelectedTr = resDFSelected.filter(s"send_dt<$predictionDTSta").checkpoint()
    val resDFSelectedTs = resDFSelected
  .filter(s"send_dt>=$predictionDTSta and send_dt<$predictionDTEnd")
  .selectExpr("cmpgn_num", "svc_mgmt_num", "chnl_typ", "cmpgn_typ", "send_ym", "send_dt", "feature_ym", "send_hournum_cd", "hour_gap", "click_yn", "res_utility")
  .checkpoint()

    // [ZEPPELIN_PARAGRAPH] 24. Sampling ratios (train balancing)
    val samplingKeyCols = Array("chnl_typ", "cmpgn_typ", "send_daynum", "send_hournum_cd", "click_yn")
    val genSampleNumMulti = 2.0

    val samplingRatioMapDF = resDFSelectedTr
  .sample(0.3)
  .groupBy(samplingKeyCols.map(col): _*)
  .agg(count(lit(1)).alias("cnt"))
  .withColumn("min_cnt", min(col("cnt")).over(Window.partitionBy(samplingKeyCols.filter(_ != "click_yn").map(col): _*)))
  .withColumn("ratio", col("min_cnt") / col("cnt"))
  .withColumn("sampling_col", expr(s"""concat_ws('-', ${samplingKeyCols.mkString(",")})"""))
  .selectExpr("sampling_col", s"least(1.0, ratio*$genSampleNumMulti) ratio")
  .sort("sampling_col")

    val resDFSelectedTrBal = resDFSelectedTr
  .withColumn("sampling_col", expr(s"""concat_ws('-', ${samplingKeyCols.mkString(",")})"""))
  .join(broadcast(samplingRatioMapDF), "sampling_col")
  .withColumn("rand", rand())
  .filter("rand<=ratio")
  .checkpoint()

    // [ZEPPELIN_PARAGRAPH] 25. XDR feature 만들기 (suffix filter)
    val userYmDF = resDFSelectedTrBal
  .select("svc_mgmt_num", "feature_ym")
  .union(resDFSelectedTs.select("svc_mgmt_num", "feature_ym"))
  .distinct()
  .filter(smnCond)

    userYmDF.createOrReplaceTempView("user_ym_df")

    val hourCols = hourRange.map(h => f"$h, a.total_traffic_$h%02d").mkString(", ")
    val xdrDF = spark.sql(
  s"""
     |SELECT
     |  a.svc_mgmt_num,
     |  a.ym AS feature_ym,
     |  COALESCE(a.rep_app_title, a.app_uid) AS app_nm,
     |  CAST(hour.send_hournum_cd AS STRING) AS send_hournum_cd,
     |  hour.traffic
     |FROM (
     |  SELECT * FROM dprobe.mst_app_svc_app_monthly
     |  WHERE (ym in ($featureYmIn))
     |    AND ($smnCond)
     |) a
     |INNER JOIN user_ym_df b
     |  ON a.svc_mgmt_num = b.svc_mgmt_num
     |  AND a.ym = b.feature_ym
     |LATERAL VIEW STACK(
     |  ${hourRange.size},
     |  $hourCols
     |) hour AS send_hournum_cd, traffic
     |WHERE hour.traffic > 1000
     |""".stripMargin
    )
      .groupBy("svc_mgmt_num", "feature_ym", "app_nm", "send_hournum_cd")
      .agg(sum("traffic").as("traffic"))
      .groupBy("svc_mgmt_num", "feature_ym", "send_hournum_cd")
      .agg(collect_list("app_nm").alias("app_usage_token"))

    val xdrDFMon = xdrDF
  .groupBy("svc_mgmt_num", "feature_ym")
  .pivot("send_hournum_cd", hourRange.map(_.toString))
  .agg(first("app_usage_token"))
  .select(
    col("svc_mgmt_num") +: col("feature_ym") +:
      hourRange.map(h => coalesce(col(s"$h"), array(lit("#"))).alias(s"app_usage_${h}_token")): _*
  )

    // [ZEPPELIN_PARAGRAPH] 26. Click history count (n=3 months)
    val n = 3
    val resWithFeatureYM = resDF
  .withColumn(
    "feature_ym",
    date_format(add_months(unix_timestamp(col("send_dt"), "yyyyMMdd").cast(TimestampType), -1), "yyyyMM").cast(StringType)
  )

    val clickCountDF = resWithFeatureYM.as("current")
  .join(
    resWithFeatureYM.as("previous"),
    col("current.svc_mgmt_num") === col("previous.svc_mgmt_num") &&
      col("previous.feature_ym") < col("current.feature_ym"),
    "left"
  )
  .where(
    months_between(to_date(col("current.feature_ym"), "yyyyMM"), to_date(col("previous.feature_ym"), "yyyyMM")) <= n &&
      months_between(to_date(col("current.feature_ym"), "yyyyMM"), to_date(col("previous.feature_ym"), "yyyyMM")) >= 0
  )
  .groupBy("current.svc_mgmt_num", "current.feature_ym")
  .agg(sum(coalesce(col("previous.click_yn"), lit(0.0))).alias("click_cnt"))
  .select(col("svc_mgmt_num"), col("feature_ym"), col("click_cnt"))
      .cache()

    // [ZEPPELIN_PARAGRAPH] 27. Join all features -> train/test
    val mmktDFFiltered = mmktDF.filter(smnCond)

    val trainDF = resDFSelectedTrBal
  .filter(smnCond)
  .join(mmktDFFiltered, Seq("svc_mgmt_num", "feature_ym"))
  .join(xdrDF, Seq("svc_mgmt_num", "feature_ym", "send_hournum_cd"))
  .join(xdrDFMon, Seq("svc_mgmt_num", "feature_ym"))
  .join(clickCountDF, Seq("svc_mgmt_num", "feature_ym"), "left")
  .na.fill(Map("click_cnt" -> 0.0))

    val testDF = resDFSelectedTs
  .filter(smnCond)
  .join(mmktDFFiltered, Seq("svc_mgmt_num", "feature_ym"))
  .join(xdrDF, Seq("svc_mgmt_num", "feature_ym", "send_hournum_cd"))
  .join(xdrDFMon, Seq("svc_mgmt_num", "feature_ym"))
  .join(clickCountDF, Seq("svc_mgmt_num", "feature_ym"), "left")
  .na.fill(Map("click_cnt" -> 0.0))

    // [ZEPPELIN_PARAGRAPH] 28. Make trainDFRev/testDFRev (schema-based feature typing)
    def buildRevDF(df: DataFrame): DataFrame = {
      val noFeatureCols = Set("click_yn", "hour_gap")
      val baseColsArr = Array("cmpgn_num", "svc_mgmt_num", "chnl_typ", "cmpgn_typ", "send_ym", "send_dt", "feature_ym", "click_yn", "res_utility")
      val baseColsSet = baseColsArr.toSet

      val tokenCols = df.columns.filter(_.endsWith("_token")).distinct
      val vectorCols = df.columns.filter(_.endsWith("_vec")).distinct

      val continuousCols = df.schema.fields
        .collect { case f if f.dataType.isInstanceOf[NumericType] && !tokenCols.contains(f.name) && !vectorCols.contains(f.name) && !baseColsSet.contains(f.name) && !noFeatureCols.contains(f.name) => f.name }
        .distinct

      val categoryCols = df.schema.fields
        .collect { case f if f.dataType == StringType && !tokenCols.contains(f.name) && !vectorCols.contains(f.name) && !baseColsSet.contains(f.name) && !noFeatureCols.contains(f.name) => f.name }
        .distinct

      df
        .select(
          (baseColsArr.map(col) ++
            tokenCols.map(c => coalesce(col(c), array(lit("#"))).alias(c)) ++
            vectorCols.map(c => col(c).alias(c)) ++
            categoryCols.map(c => when(col(c) === "", lit("UKV")).otherwise(coalesce(col(c).cast("string"), lit("UKV"))).alias(c)) ++
            continuousCols.map(c => coalesce(col(c).cast("float"), lit(Double.NaN)).alias(c))
          ).distinct.map(identity): _*
        )
        .withColumn("suffix", expr("right(svc_mgmt_num, 1)"))
    }

    val trainDFRev = buildRevDF(trainDF)
    val testDFRev = buildRevDF(testDF)

    // [ZEPPELIN_PARAGRAPH] 29. Save train/test rev datasets
    trainDFRev.write
      .mode("overwrite")
      .partitionBy("send_ym", "send_hournum_cd", "suffix")
      .parquet(trainDFRevPath)

    testDFRev.write
      .mode("overwrite")
      .partitionBy("send_ym", "send_hournum_cd", "suffix")
      .parquet(testDFRevPath)
  }
}
