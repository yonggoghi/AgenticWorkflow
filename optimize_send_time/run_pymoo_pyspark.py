#!/usr/bin/env python3
"""
Optimize Send Time - PySpark 연동 예제
====================================

PySpark와 pymoo를 연동하여 대용량 데이터 처리

데이터 소스:
- 실제 데이터: aos/sto/propensityScoreDF (use_real_data=True)
- 샘플 데이터: 자동 생성 (use_real_data=False)

샘플링 옵션 (sample_suffix):
- "00": 1% 샘플 (끝자리 00인 사용자)
- "0": 10% 샘플 (끝자리 0인 사용자)
- None: 전체 데이터 (메모리 주의!)

사용 예:
    # 1% 샘플로 빠른 테스트
    python run_pymoo_pyspark.py  # (기본값: sample_suffix="00")
    
    # 전체 데이터 처리 (대용량)
    # sample_suffix=None으로 변경
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, desc, row_number, max as spark_max
from pyspark.sql.window import Window
import numpy as np
import time


def create_spark_session():
    """Spark 세션 생성 또는 기존 세션 재사용"""
    try:
        # 기존 세션이 있으면 재사용
        spark = SparkSession.getActiveSession()
        if spark is None:
            print("Creating new Spark session...")
            spark = SparkSession.builder \
                .appName("OptimizeSendTime-pymoo") \
                .config("spark.driver.memory", "8g") \
                .config("spark.executor.memory", "8g") \
                .config("spark.sql.shuffle.partitions", "200") \
                .getOrCreate()
        else:
            print("Reusing existing Spark session")
        return spark
    except Exception as e:
        print(f"Warning: Could not get active session: {e}")
        print("Creating new Spark session...")
        return SparkSession.builder \
            .appName("OptimizeSendTime-pymoo") \
            .config("spark.driver.memory", "8g") \
            .config("spark.executor.memory", "8g") \
            .config("spark.sql.shuffle.partitions", "200") \
            .getOrCreate()


def load_real_data_spark(spark, data_path: str = "aos/sto/propensityScoreDF", sample_suffix: str = None):
    """실제 데이터 로드"""
    
    print(f"Loading data from {data_path}...")
    
    # 데이터 로드
    df = spark.read.parquet(data_path)
    
    # 샘플링 (선택적)
    if sample_suffix:
        print(f"  Filtering users with suffix: {sample_suffix}")
        df = df.filter(f"svc_mgmt_num like '%{sample_suffix}'")
    
    total_records = df.count()
    unique_users = df.select('svc_mgmt_num').distinct().count()
    
    print(f"✓ Loaded {total_records:,} records")
    print(f"  Unique users: {unique_users:,}")
    print(f"  Avg choices per user: {total_records / unique_users:.1f}")
    
    return df


def generate_sample_data_spark(spark, n_users: int = 10000):
    """PySpark로 샘플 데이터 생성 (테스트용)"""
    
    print(f"Generating sample data for {n_users:,} users...")
    
    # Pandas로 생성 후 Spark로 변환 (더 빠름)
    import pandas as pd
    np.random.seed(42)
    
    hours = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    data = []
    
    for i in range(n_users):
        user_id = f"user_{i:06d}"
        n_choices = np.random.randint(3, 6)
        selected_hours = np.random.choice(hours, n_choices, replace=False)
        
        for hour in selected_hours:
            score = np.random.beta(2, 5)
            data.append((user_id, hour, float(score)))
    
    pdf = pd.DataFrame(data, columns=['svc_mgmt_num', 'send_hour', 'propensity_score'])
    df = spark.createDataFrame(pdf)
    
    print(f"✓ Generated {df.count():,} records")
    
    return df


def optimize_batch_pyspark(
    spark,
    df,
    capacity_per_hour: dict,
    batch_size: int = 10000
):
    """PySpark 배치 처리"""
    
    from optimize_ost_pymoo import SendTimeOptimizer
    
    print("\n" + "=" * 80)
    print("Batch Processing with PySpark")
    print("=" * 80)
    
    optimizer = SendTimeOptimizer(verbose=False)
    
    # 전체 사용자 수
    total_users = df.select("svc_mgmt_num").distinct().count()
    n_batches = int(np.ceil(total_users / batch_size))
    
    print(f"\nTotal users: {total_users:,}")
    print(f"Batch size: {batch_size:,}")
    print(f"Number of batches: {n_batches}")
    
    # 가치 기반 배치 분할
    user_priority = df.groupBy("svc_mgmt_num") \
        .agg(spark_max("propensity_score").alias("max_prob"))
    
    window = Window.orderBy(desc("max_prob"))
    all_users = user_priority \
        .withColumn("row_id", row_number().over(window)) \
        .withColumn("batch_id", ((col("row_id") - 1) / batch_size).cast("int")) \
        .select("svc_mgmt_num", "batch_id") \
        .cache()
    
    print("\nBatch distribution:")
    batch_counts = all_users.groupBy("batch_id").count() \
        .orderBy("batch_id") \
        .collect()
    
    for row in batch_counts:
        print(f"  Batch {row['batch_id']}: {row['count']:,} users")
    
    # 배치별 처리
    remaining_capacity = capacity_per_hour.copy()
    all_results = []
    total_assigned = 0
    
    start_time = time.time()
    
    for batch_id in range(n_batches):
        print(f"\n{'=' * 80}")
        print(f"Processing Batch {batch_id + 1}/{n_batches}")
        print(f"{'=' * 80}")
        
        # 용량이 남아있는 시간대만
        available_hours = [h for h, c in remaining_capacity.items() if c > 0]
        
        if not available_hours:
            print("⚠ No capacity left")
            break
        
        print(f"Available hours: {available_hours}")
        print(f"Remaining capacity: {sum(remaining_capacity.values()):,}")
        
        # 배치 데이터 추출
        batch_df = df.join(
            all_users.filter(col("batch_id") == batch_id),
            "svc_mgmt_num"
        ).filter(col("send_hour").isin(available_hours))
        
        batch_user_count = batch_df.select("svc_mgmt_num").distinct().count()
        print(f"Batch users: {batch_user_count:,}")
        
        if batch_user_count == 0:
            continue
        
        # Pandas로 변환하여 최적화
        batch_pdf = batch_df.select(
            "svc_mgmt_num", "send_hour", "propensity_score"
        ).toPandas()
        
        # Greedy 할당 (배치 처리에 적합)
        batch_start = time.time()
        result_pdf = optimizer.optimize_greedy(batch_pdf, remaining_capacity)
        batch_time = time.time() - batch_start
        
        assigned_count = len(result_pdf)
        total_assigned += assigned_count
        
        print(f"✓ Batch completed in {batch_time:.2f}s")
        print(f"  Assigned: {assigned_count:,} users")
        print(f"  Batch score: {result_pdf['score'].sum():,.2f}")
        
        # 용량 차감
        used_per_hour = result_pdf.groupby('assigned_hour').size().to_dict()
        for hour, count in used_per_hour.items():
            remaining_capacity[hour] = max(0, remaining_capacity[hour] - count)
        
        # Spark DataFrame으로 변환
        result_sdf = spark.createDataFrame(result_pdf)
        all_results.append(result_sdf)
        
        # 진행률
        progress = total_assigned / total_users * 100
        print(f"\nProgress: {total_assigned:,} / {total_users:,} ({progress:.1f}%)")
    
    total_time = time.time() - start_time
    
    # 결과 결합
    if not all_results:
        print("\n⚠ No results generated")
        return None
    
    from functools import reduce
    final_result = reduce(lambda a, b: a.union(b), all_results)
    
    print("\n" + "=" * 80)
    print("Final Results")
    print("=" * 80)
    print(f"Total time: {total_time:.2f}s")
    print(f"Total assigned: {total_assigned:,} / {total_users:,}")
    print(f"Coverage: {total_assigned / total_users * 100:.2f}%")
    
    # 시간대별 통계
    print("\nHour-wise allocation:")
    final_result.groupBy("assigned_hour") \
        .agg(
            {"svc_mgmt_num": "count", "score": "sum"}
        ) \
        .withColumnRenamed("count(svc_mgmt_num)", "count") \
        .withColumnRenamed("sum(score)", "total_score") \
        .orderBy("assigned_hour") \
        .show()
    
    all_users.unpersist()
    
    return final_result


def optimize_simple_pyspark(spark, df, capacity_per_hour: dict):
    """단순 PySpark 최적화 (전체 데이터 collect)"""
    
    from optimize_ost_pymoo import optimize_with_pyspark
    
    print("\n" + "=" * 80)
    print("Simple PySpark Optimization")
    print("=" * 80)
    
    print("\nCollecting data to driver...")
    start_time = time.time()
    
    # 최적화
    result = optimize_with_pyspark(
        df,
        capacity_per_hour,
        method="greedy"  # 'greedy', 'ga', 'hybrid'
    )
    
    elapsed_time = time.time() - start_time
    
    print(f"\n✓ Completed in {elapsed_time:.2f}s")
    
    # 결과 통계
    total_assigned = result.count()
    total_score = result.agg({"score": "sum"}).collect()[0][0]
    
    print(f"Total assigned: {total_assigned:,}")
    print(f"Total score: {total_score:,.2f}")
    
    return result


def main():
    """메인 실행"""
    
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║              Optimize Send Time - PySpark Integration                     ║
║                                                                           ║
║  Demonstrates large-scale optimization with PySpark + pymoo               ║
╚═══════════════════════════════════════════════════════════════════════════╝
""")
    
    # Spark 세션
    spark = create_spark_session()
    
    print(f"Spark version: {spark.version}")
    print(f"Spark master: {spark.sparkContext.master}")
    
    # 데이터 로드 옵션
    use_real_data = True  # False로 변경하면 샘플 데이터 사용
    sample_suffix = "0"  # 실제 데이터 샘플링 (예: "00" = 1%, "0" = 10%, None = 전체)
                         # Scala load_and_test.scala 기준: "0" (10% 샘플)
    
    if use_real_data:
        # 실제 데이터 로드
        df = load_real_data_spark(
            spark,
            data_path="aos/sto/propensityScoreDF",
            sample_suffix=sample_suffix
        )
    else:
        # 샘플 데이터 생성 (테스트용)
        n_users_sample = 50000
        df = generate_sample_data_spark(spark, n_users_sample)
    
    # 사용자 수 확인
    n_users = df.select("svc_mgmt_num").distinct().count()
    
    # 설정
    batch_size = max(10000, int(n_users * 0.1))  # 사용자 수의 10% 또는 최소 1만명
    hours = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    
    # 시간대별 용량 (load_and_test.scala 기준)
    # val capacityPerHour = (totalRecords * 0.2).toInt
    capacity_per_hour_value = int(n_users * 0.2)  # 사용자 수의 20%
    capacity_per_hour = {hour: capacity_per_hour_value for hour in hours}
    
    # 총 용량: n_users * 0.2 * 10시간 = n_users * 2.0 (200%)
    # → 모든 사용자가 할당될 수 있음
    
    # 또는 고정 용량:
    # capacity_per_hour_value = 10000
    # capacity_per_hour = {hour: capacity_per_hour_value for hour in hours}
    
    total_capacity = sum(capacity_per_hour.values())
    
    print(f"\nConfiguration (load_and_test.scala style):")
    print(f"  Users: {n_users:,}")
    print(f"  Batch size: {batch_size:,} (= users × 0.1 or min 10,000)")
    print(f"  Capacity per hour: {capacity_per_hour_value:,} (= users × 0.2)")
    print(f"  Total capacity: {total_capacity:,} (= users × 2.0)")
    print(f"  Capacity/Users ratio: {total_capacity / n_users:.2f}x")
    
    if n_users > total_capacity:
        print(f"\n⚠ Warning: Users ({n_users:,}) exceed total capacity ({total_capacity:,})")
        print(f"  Some users may not be assigned.")
    elif total_capacity >= n_users:
        print(f"\n✓ All users can be assigned (capacity is {total_capacity / n_users:.1f}x of users)")
    
    # 데이터 캐싱
    df.cache()
    df.count()  # 캐시 실행
    
    # 방법 선택
    use_batch = n_users > 10000
    
    if use_batch:
        print("\n→ Using batch processing (recommended for large data)")
        result = optimize_batch_pyspark(spark, df, capacity_per_hour, batch_size)
    else:
        print("\n→ Using simple processing (suitable for small data)")
        result = optimize_simple_pyspark(spark, df, capacity_per_hour)
    
    # 결과 저장
    if result:
        output_path = "aos/sto/allocation_result_pymoo"
        print(f"\nSaving results to: {output_path}")
        
        result.write.mode("overwrite").parquet(output_path)
        print(f"✓ Results saved")
        
        # 샘플 출력
        print("\nSample results:")
        result.show(20, truncate=False)
    
    # Spark 세션 종료하지 않음 (재사용을 위해)
    # spark.stop()  # 주석 처리: 다른 프로세스가 사용 중일 수 있음
    
    print("\n" + "=" * 80)
    print("PySpark optimization completed successfully!")
    print("=" * 80)
    print("\nNote: Spark session kept alive for reuse")


if __name__ == "__main__":
    main()
