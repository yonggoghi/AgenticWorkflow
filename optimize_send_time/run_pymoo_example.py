#!/usr/bin/env python3
"""
Optimize Send Time - pymoo 실행 예제
====================================

간단한 실행 예제로 각 알고리즘의 성능을 비교합니다.

데이터 소스:
- 실제 데이터: aos/sto/propensityScoreDF (use_real_data=True)
- 샘플 데이터: 자동 생성 (use_real_data=False)

샘플링 옵션 (sample_suffix):
- "00": 1% 샘플 (끝자리 00인 사용자)
- "0": 10% 샘플 (끝자리 0인 사용자)
- None: 전체 데이터
"""

import numpy as np
import pandas as pd
import time
from optimize_ost_pymoo import SendTimeOptimizer


def load_real_data_direct(data_path: str = "aos/sto/propensityScoreDF", sample_suffix: str = None) -> pd.DataFrame:
    """실제 데이터 로드 (Pandas 직접 사용 - PySpark 없이)"""
    import pyarrow.parquet as pq
    
    print(f"Loading data from {data_path}...")
    
    try:
        # Parquet 파일 직접 읽기
        table = pq.read_table(data_path)
        df_pandas = table.to_pandas()
        
        print(f"  Total records: {len(df_pandas):,}")
        
        # 샘플링 (선택적)
        if sample_suffix:
            print(f"  Filtering users with suffix: {sample_suffix}")
            df_pandas = df_pandas[df_pandas['svc_mgmt_num'].str.endswith(sample_suffix)]
        
        print(f"✓ Loaded {len(df_pandas):,} records")
        print(f"  Unique users: {df_pandas['svc_mgmt_num'].nunique():,}")
        print(f"  Avg choices per user: {len(df_pandas) / df_pandas['svc_mgmt_num'].nunique():.1f}")
        
        return df_pandas
        
    except ImportError:
        print("  ⚠ pyarrow not found. Falling back to PySpark method...")
        return load_real_data_pyspark(data_path, sample_suffix)
    except Exception as e:
        print(f"  ⚠ Error reading parquet directly: {e}")
        print("  Falling back to PySpark method...")
        return load_real_data_pyspark(data_path, sample_suffix)


def load_real_data(data_path: str = "aos/sto/propensityScoreDF", sample_suffix: str = None, 
                   use_pyspark: bool = False) -> pd.DataFrame:
    """
    실제 데이터 로드 (자동 선택)
    
    Args:
        data_path: 데이터 경로
        sample_suffix: 샘플링 suffix (예: "0" = 10%)
        use_pyspark: True면 PySpark 강제 사용, False면 Pandas 우선
    """
    if use_pyspark:
        return load_real_data_pyspark(data_path, sample_suffix)
    else:
        return load_real_data_direct(data_path, sample_suffix)


def load_real_data_pyspark(data_path: str = "aos/sto/propensityScoreDF", sample_suffix: str = None) -> pd.DataFrame:
    """실제 데이터 로드 (PySpark 사용)"""
    from pyspark.sql import SparkSession
    
    print(f"Loading data from {data_path}...")
    
    # Spark 세션 생성 또는 기존 세션 재사용
    try:
        # 기존 세션이 있으면 재사용
        spark = SparkSession.getActiveSession()
        if spark is None:
            spark = SparkSession.builder \
                .appName("OptimizeSendTime-LoadData") \
                .config("spark.driver.memory", "4g") \
                .getOrCreate()
            print("  Created new Spark session")
        else:
            print("  Reusing existing Spark session")
    except Exception as e:
        print(f"  Warning: Could not get active session: {e}")
        spark = SparkSession.builder \
            .appName("OptimizeSendTime-LoadData") \
            .config("spark.driver.memory", "4g") \
            .getOrCreate()
    
    # 데이터 로드
    df_spark = spark.read.parquet(data_path)
    
    # 샘플링 (선택적)
    if sample_suffix:
        print(f"  Filtering users with suffix: {sample_suffix}")
        df_spark = df_spark.filter(f"svc_mgmt_num like '%{sample_suffix}'")
    
    print(f"  Total records: {df_spark.count():,}")
    print(f"  Unique users: {df_spark.select('svc_mgmt_num').distinct().count():,}")
    
    # Pandas로 변환
    print("  Converting to Pandas...")
    df_pandas = df_spark.toPandas()
    
    # Spark 세션 종료하지 않음 (재사용을 위해)
    # spark.stop()  # 주석 처리: 다른 프로세스가 사용 중일 수 있음
    
    print(f"✓ Loaded {len(df_pandas):,} records")
    print(f"  Unique users: {df_pandas['svc_mgmt_num'].nunique():,}")
    print(f"  Avg choices per user: {len(df_pandas) / df_pandas['svc_mgmt_num'].nunique():.1f}")
    
    return df_pandas


def generate_sample_data(n_users: int = 1000, seed: int = 42) -> pd.DataFrame:
    """샘플 데이터 생성 (테스트용)"""
    np.random.seed(seed)
    
    hours = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    data = []
    
    print(f"Generating sample data for {n_users:,} users...")
    
    for i in range(n_users):
        user_id = f"user_{i:05d}"
        
        # 각 사용자는 3-5개 시간대 선택 가능
        n_choices = np.random.randint(3, 6)
        selected_hours = np.random.choice(hours, n_choices, replace=False)
        
        for hour in selected_hours:
            # Beta 분포로 현실적인 propensity score 생성
            score = np.random.beta(2, 5)
            data.append({
                'svc_mgmt_num': user_id,
                'send_hour': hour,
                'propensity_score': score
            })
    
    df = pd.DataFrame(data)
    
    print(f"✓ Generated {len(df):,} records")
    print(f"  Unique users: {df['svc_mgmt_num'].nunique():,}")
    print(f"  Avg choices per user: {len(df) / n_users:.1f}")
    
    return df


def compare_algorithms(df: pd.DataFrame, capacity_per_hour: dict):
    """알고리즘 비교"""
    
    print("\n" + "=" * 80)
    print("Algorithm Comparison")
    print("=" * 80)
    
    optimizer = SendTimeOptimizer(verbose=False)
    results = {}
    
    # 1. Greedy
    print("\n[1/3] Running Greedy Algorithm...")
    start = time.time()
    result_greedy = optimizer.optimize_greedy(df, capacity_per_hour)
    time_greedy = time.time() - start
    
    results['Greedy'] = {
        'result': result_greedy,
        'time': time_greedy,
        'score': result_greedy['score'].sum(),
        'assigned': len(result_greedy)
    }
    
    print(f"  ✓ Completed in {time_greedy:.2f}s")
    print(f"  Total score: {results['Greedy']['score']:,.2f}")
    print(f"  Assigned: {results['Greedy']['assigned']:,}")
    
    # 2. Genetic Algorithm (중소규모 데이터)
    # 임계값: 5,000명 (기본), 10,000명까지 조정 가능
    ga_threshold = 50000  # 필요시 증가 (예: 10000, 50000)
    if len(df['svc_mgmt_num'].unique()) <= ga_threshold:
        print("\n[2/3] Running Genetic Algorithm...")
        start = time.time()
        result_ga = optimizer.optimize_with_ga(
            df, capacity_per_hour,
            pop_size=50,
            n_gen=100
        )
        time_ga = time.time() - start
        
        results['GA'] = {
            'result': result_ga,
            'time': time_ga,
            'score': result_ga['score'].sum(),
            'assigned': len(result_ga)
        }
        
        print(f"  ✓ Completed in {time_ga:.2f}s")
        print(f"  Total score: {results['GA']['score']:,.2f}")
        print(f"  Assigned: {results['GA']['assigned']:,}")
    else:
        print(f"\n[2/3] Skipping GA (dataset too large: {len(df['svc_mgmt_num'].unique()):,} > {ga_threshold:,})")
        print(f"  Hint: Increase 'ga_threshold' to {len(df['svc_mgmt_num'].unique())} to run GA")
        results['GA'] = None
    
    # 3. Hybrid
    if len(df['svc_mgmt_num'].unique()) <= ga_threshold:
        print("\n[3/3] Running Hybrid Algorithm...")
        start = time.time()
        result_hybrid = optimizer.optimize_hybrid(
            df, capacity_per_hour,
            ga_pop_size=50,
            ga_n_gen=100
        )
        time_hybrid = time.time() - start
        
        results['Hybrid'] = {
            'result': result_hybrid,
            'time': time_hybrid,
            'score': result_hybrid['score'].sum(),
            'assigned': len(result_hybrid)
        }
        
        print(f"  ✓ Completed in {time_hybrid:.2f}s")
        print(f"  Total score: {results['Hybrid']['score']:,.2f}")
        print(f"  Assigned: {results['Hybrid']['assigned']:,}")
    else:
        print(f"\n[3/3] Skipping Hybrid (dataset too large: {len(df['svc_mgmt_num'].unique()):,} > {ga_threshold:,})")
        print(f"  Hint: Use smaller sample (e.g., suffix='00' for 1%) or PySpark batch version")
        results['Hybrid'] = None
    
    return results


def print_comparison(results: dict, total_users: int):
    """결과 비교 출력"""
    
    print("\n" + "=" * 80)
    print("Results Summary")
    print("=" * 80)
    
    # 테이블 헤더
    print(f"\n{'Algorithm':<15} {'Time':>10} {'Total Score':>15} {'Assigned':>12} {'Improvement':>12}")
    print("-" * 80)
    
    # Greedy를 기준으로
    baseline_score = results['Greedy']['score']
    
    for algo_name, result in results.items():
        if result is None:
            print(f"{algo_name:<15} {'N/A':>10} {'N/A':>15} {'N/A':>12} {'N/A':>12}")
            continue
        
        improvement = ((result['score'] - baseline_score) / baseline_score * 100) \
                      if baseline_score > 0 else 0
        
        print(f"{algo_name:<15} "
              f"{result['time']:>9.2f}s "
              f"{result['score']:>15,.2f} "
              f"{result['assigned']:>12,} "
              f"{improvement:>11.1f}%")
    
    print("-" * 80)
    print(f"Total users: {total_users:,}")
    
    # 추천
    print("\nRecommendation:")
    if results['Hybrid'] and results['Hybrid']['score'] > results['Greedy']['score'] * 1.05:
        print("  → Use HYBRID for best quality (5%+ improvement)")
    elif results['GA'] and results['GA']['score'] > results['Greedy']['score'] * 1.05:
        print("  → Use GA for better quality (5%+ improvement)")
    else:
        print("  → Use GREEDY for speed (negligible quality difference)")


def main():
    """메인 실행"""
    
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                  Optimize Send Time - pymoo Example                       ║
║                                                                           ║
║  This script demonstrates the Python/pymoo implementation                 ║
║  of the send time allocation optimizer.                                   ║
╚═══════════════════════════════════════════════════════════════════════════╝
""")
    
    # 설정
    hours = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    
    # 데이터 로드 옵션
    use_real_data = True  # False로 변경하면 샘플 데이터 사용
    sample_suffix = "0"  # 실제 데이터 샘플링 (예: "00" = 1%, "0" = 10%, None = 전체)
                         # Scala load_and_test.scala 기준: "0" (10% 샘플)
    
    if use_real_data:
        # 실제 데이터 로드
        df = load_real_data(
            data_path="aos/sto/propensityScoreDF",
            sample_suffix=sample_suffix
        )
    else:
        # 샘플 데이터 생성 (테스트용)
        n_users = 10000
        df = generate_sample_data(n_users)
    
    n_users = df['svc_mgmt_num'].nunique()
    
    # 시간대별 용량 (load_and_test.scala 기준)
    # val capacityPerHour = (totalRecords * 0.2).toInt
    capacity_per_hour_value = int(n_users * 0.2)  # 사용자 수의 20%
    capacity_per_hour = {hour: capacity_per_hour_value for hour in hours}
    
    # 총 용량: n_users * 0.2 * 10시간 = n_users * 2.0 (200%)
    # → 모든 사용자가 할당될 수 있음
    
    # 또는 고정 용량:
    # capacity_per_hour_value = 10000
    # capacity_per_hour = {hour: capacity_per_hour_value for hour in hours}
    
    # 또는 차등 용량:
    # capacity_per_hour = {
    #     9: 5000, 10: 8000, 11: 12000, 12: 10000, 13: 8000,
    #     14: 12000, 15: 15000, 16: 13000, 17: 10000, 18: 7000
    # }
    
    total_capacity = sum(capacity_per_hour.values())
    
    print(f"\nConfiguration (load_and_test.scala style):")
    print(f"  Users: {n_users:,}")
    print(f"  Hours: {hours}")
    print(f"  Capacity per hour: {capacity_per_hour_value:,} (= users × 0.2)")
    print(f"  Total capacity: {total_capacity:,} (= users × 2.0)")
    print(f"  Capacity/Users ratio: {total_capacity / n_users:.2f}x")
    
    if n_users > total_capacity:
        print(f"\n⚠ Warning: Users ({n_users:,}) exceed total capacity ({total_capacity:,})")
        print(f"  Some users may not be assigned.")
    elif total_capacity >= n_users:
        print(f"\n✓ All users can be assigned (capacity is {total_capacity / n_users:.1f}x of users)")
    
    # 데이터 통계
    print(f"\nData Statistics:")
    print(f"  Total records: {len(df):,}")
    hour_dist = df.groupby('send_hour').size()
    print(f"  Hour distribution:")
    for hour in sorted(hours):
        count = hour_dist.get(hour, 0)
        pct = count / len(df) * 100
        print(f"    Hour {hour:2d}: {count:>6,} ({pct:>5.1f}%)")
    
    # 알고리즘 비교
    results = compare_algorithms(df, capacity_per_hour)
    
    # 결과 요약
    print_comparison(results, n_users)
    
    # 시간대별 할당 상세 (Greedy)
    print("\n" + "=" * 80)
    print("Hour-wise Allocation Detail (Greedy)")
    print("=" * 80)
    
    greedy_result = results['Greedy']['result']
    hour_stats = greedy_result.groupby('assigned_hour').agg({
        'svc_mgmt_num': 'count',
        'score': ['sum', 'mean', 'min', 'max']
    }).round(4)
    
    print(hour_stats)
    
    # 결과 파일 저장
    output_file = "pymoo_allocation_result.csv"
    greedy_result.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to: {output_file}")
    
    print("\n" + "=" * 80)
    print("Benchmark completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
