object PredictOst10ResponseExtractSave {
  def run(): Unit = {
    import PredictOstConfig._
    initSpark()

    // [ZEPPELIN_PARAGRAPH] 10. Response Data Loading - Hive
    val responseDF = spark.sql("""
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

    responseDF.createOrReplaceTempView("response_df")

    // [ZEPPELIN_PARAGRAPH] 11. Response Data Saving
    responseDF.write.mode("overwrite").partitionBy("send_ym").parquet(responsePath)

    // [ZEPPELIN_PARAGRAPH] 12. Response Data Loading - HDFS (for next steps)
    val resDF = spark.read.parquet(responsePath).cache()
    resDF.createOrReplaceTempView("res_df")

    // [ZEPPELIN_PARAGRAPH] 13. Quick sanity check
    showN(resDF.filter("click_hournum is not null"), 20)
  }
}
