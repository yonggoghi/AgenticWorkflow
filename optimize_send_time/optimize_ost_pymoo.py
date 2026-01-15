"""
Optimize Send Time - Python Implementation with pymoo
======================================================

Scala 버전을 pymoo를 이용한 Python으로 변환
- 유전 알고리즘 (GA) 기반 최적화
- 제약 조건 처리 (시간대별 용량)
- PySpark와 연동 가능

Requirements:
    pip install pymoo pandas numpy pyspark
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination
import time


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class AllocationResult:
    """할당 결과"""
    svc_mgmt_num: str
    assigned_hour: int
    score: float


class SendTimeAllocationProblem(ElementwiseProblem):
    """
    발송 시간 할당 최적화 문제 정의
    
    의사결정 변수: 각 사용자에게 할당할 시간대 인덱스
    목적 함수: propensity score 합 최대화 (음수로 변환하여 최소화)
    제약 조건: 시간대별 용량 제한
    """
    
    def __init__(
        self,
        user_data: Dict[str, Dict[int, float]],
        capacity_per_hour: Dict[int, int],
        users: List[str],
        hours: List[int]
    ):
        """
        Args:
            user_data: {user_id: {hour: propensity_score}}
            capacity_per_hour: {hour: capacity}
            users: 사용자 ID 리스트
            hours: 가능한 시간대 리스트
        """
        self.user_data = user_data
        self.capacity_per_hour = capacity_per_hour
        self.users = users
        self.hours = sorted(hours)
        self.n_users = len(users)
        self.n_hours = len(self.hours)
        
        # 각 사용자별 가능한 시간대 인덱스 매핑
        self.user_hour_mapping = {}
        self.hour_to_idx = {hour: idx for idx, hour in enumerate(self.hours)}
        
        for user in users:
            available_hours = list(user_data[user].keys())
            self.user_hour_mapping[user] = [
                self.hour_to_idx[h] for h in available_hours
            ]
        
        # 제약 조건 개수: 시간대별 용량 제약
        n_constraints = self.n_hours
        
        super().__init__(
            n_var=self.n_users,  # 각 사용자당 1개 변수 (시간대 인덱스)
            n_obj=1,  # 목적 함수: 총 점수 최대화
            n_constr=n_constraints,  # 시간대별 용량 제약
            xl=0,  # 하한: 시간대 인덱스 0
            xu=self.n_hours - 1  # 상한: 시간대 인덱스 n-1
        )
    
    def _evaluate(self, x, out, *args, **kwargs):
        """
        개별 솔루션 평가
        
        Args:
            x: 의사결정 변수 (각 사용자의 할당 시간대 인덱스)
            out: 출력 딕셔너리 (목적 함수값, 제약 조건 위반)
        """
        # 목적 함수: 총 propensity score (최대화 -> 음수로 변환)
        total_score = 0.0
        hour_usage = np.zeros(self.n_hours)
        
        for i, user in enumerate(self.users):
            hour_idx = int(x[i])
            hour = self.hours[hour_idx]
            
            # 해당 사용자가 이 시간대를 선택할 수 있는지 확인
            if hour in self.user_data[user]:
                score = self.user_data[user][hour]
                total_score += score
                hour_usage[hour_idx] += 1
            else:
                # 선택 불가능한 시간대면 페널티
                total_score -= 1000  # 큰 페널티
        
        # 목적 함수 (최소화 문제로 변환)
        out["F"] = -total_score
        
        # 제약 조건: 시간대별 용량 초과 여부
        # g(x) <= 0 형태로 변환
        constraints = []
        for hour_idx, hour in enumerate(self.hours):
            capacity = self.capacity_per_hour.get(hour, 0)
            violation = hour_usage[hour_idx] - capacity
            constraints.append(violation)
        
        out["G"] = np.array(constraints)


# ============================================================================
# Optimizer Class
# ============================================================================

class SendTimeOptimizer:
    """발송 시간 최적화 메인 클래스"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.num_formatter = lambda x: f"{x:,}"
    
    def optimize_with_ga(
        self,
        df: pd.DataFrame,
        capacity_per_hour: Dict[int, int],
        pop_size: int = 100,
        n_gen: int = 200,
        crossover_prob: float = 0.9,
        mutation_prob: float = 0.1
    ) -> pd.DataFrame:
        """
        유전 알고리즘으로 최적화
        
        Args:
            df: DataFrame with columns [svc_mgmt_num, send_hour, propensity_score]
            capacity_per_hour: {hour: capacity}
            pop_size: 개체군 크기
            n_gen: 세대 수
            crossover_prob: 교차 확률
            mutation_prob: 돌연변이 확률
        
        Returns:
            DataFrame with [svc_mgmt_num, assigned_hour, score]
        """
        print("=" * 80)
        print("Genetic Algorithm Optimization with pymoo")
        print("=" * 80)
        
        # 데이터 수집
        user_data = self._collect_user_data(df)
        users = sorted(user_data.keys())
        hours = sorted(df['send_hour'].unique())
        
        print(f"\n[INPUT INFO]")
        print(f"Users: {self.num_formatter(len(users))}")
        print(f"Hours: {len(hours)}")
        print(f"Total capacity: {self.num_formatter(sum(capacity_per_hour.values()))}")
        
        # 문제 정의
        problem = SendTimeAllocationProblem(
            user_data=user_data,
            capacity_per_hour=capacity_per_hour,
            users=users,
            hours=hours
        )
        
        # 알고리즘 설정
        algorithm = GA(
            pop_size=pop_size,
            sampling=IntegerRandomSampling(),
            crossover=TwoPointCrossover(prob=crossover_prob),
            mutation=PolynomialMutation(prob=mutation_prob),
            eliminate_duplicates=True
        )
        
        # 종료 조건
        termination = get_termination("n_gen", n_gen)
        
        print(f"\n[ALGORITHM PARAMETERS]")
        print(f"Population size: {pop_size}")
        print(f"Generations: {n_gen}")
        print(f"Crossover probability: {crossover_prob}")
        print(f"Mutation probability: {mutation_prob}")
        
        # 최적화 실행
        print("\nRunning optimization...")
        start_time = time.time()
        
        res = minimize(
            problem,
            algorithm,
            termination,
            seed=42,
            verbose=self.verbose
        )
        
        elapsed_time = time.time() - start_time
        
        print(f"\n✓ Optimization completed in {elapsed_time:.2f}s")
        print(f"Best objective value: {-res.F[0]:,.2f}")
        
        # 결과 추출
        best_solution = res.X
        results = self._extract_solution(
            best_solution, user_data, users, problem.hours
        )
        
        # 검증 및 통계
        self._validate_and_print_stats(results, users, hours, capacity_per_hour)
        
        return pd.DataFrame([
            {
                'svc_mgmt_num': r.svc_mgmt_num,
                'assigned_hour': r.assigned_hour,
                'score': r.score
            }
            for r in results
        ])
    
    def optimize_greedy(
        self,
        df: pd.DataFrame,
        capacity_per_hour: Dict[int, int]
    ) -> pd.DataFrame:
        """
        Greedy 알고리즘으로 빠른 할당
        
        Args:
            df: DataFrame with [svc_mgmt_num, send_hour, propensity_score]
            capacity_per_hour: {hour: capacity}
        
        Returns:
            DataFrame with [svc_mgmt_num, assigned_hour, score]
        """
        print("\n" + "=" * 80)
        print("Greedy Allocation Algorithm")
        print("=" * 80)
        
        user_data = self._collect_user_data(df)
        users = sorted(user_data.keys())
        hours = sorted(df['send_hour'].unique())
        
        print(f"\nUsers to assign: {self.num_formatter(len(users))}")
        
        # 사용자를 최고 점수 순으로 정렬
        user_best_scores = [
            (user, max(user_data[user].values()))
            for user in users
        ]
        user_best_scores.sort(key=lambda x: -x[1])
        
        # Greedy 할당
        hour_capacity = capacity_per_hour.copy()
        assignments = []
        
        for user, _ in user_best_scores:
            # 가능한 시간대를 점수 순으로 정렬
            choices = sorted(
                user_data[user].items(),
                key=lambda x: -x[1]
            )
            
            assigned = False
            for hour, score in choices:
                if hour_capacity.get(hour, 0) > 0:
                    assignments.append(
                        AllocationResult(user, hour, score)
                    )
                    hour_capacity[hour] -= 1
                    assigned = True
                    break
        
        print(f"Assigned: {self.num_formatter(len(assignments))} / {self.num_formatter(len(users))}")
        
        # 통계
        self._validate_and_print_stats(assignments, users, hours, capacity_per_hour)
        
        return pd.DataFrame([
            {
                'svc_mgmt_num': r.svc_mgmt_num,
                'assigned_hour': r.assigned_hour,
                'score': r.score
            }
            for r in assignments
        ])
    
    def optimize_hybrid(
        self,
        df: pd.DataFrame,
        capacity_per_hour: Dict[int, int],
        ga_pop_size: int = 100,
        ga_n_gen: int = 200
    ) -> pd.DataFrame:
        """
        Hybrid: GA + Greedy
        
        GA로 먼저 최적화하고, 미할당 사용자는 Greedy로 처리
        """
        print("=" * 80)
        print("Hybrid Allocation (GA + Greedy)")
        print("=" * 80)
        
        # GA 최적화
        try:
            ga_result = self.optimize_with_ga(
                df, capacity_per_hour, ga_pop_size, ga_n_gen
            )
            
            # 미할당 사용자 확인
            assigned_users = set(ga_result['svc_mgmt_num'])
            all_users = set(df['svc_mgmt_num'].unique())
            unassigned_users = all_users - assigned_users
            
            if not unassigned_users:
                print("✓ All users assigned by GA")
                return ga_result
            
            print(f"\n{self.num_formatter(len(unassigned_users))} users unassigned")
            print("Running Greedy for remainder...")
            
            # 남은 용량 계산
            used_capacity = ga_result.groupby('assigned_hour').size().to_dict()
            remaining_capacity = {
                hour: max(0, cap - used_capacity.get(hour, 0))
                for hour, cap in capacity_per_hour.items()
            }
            
            # Greedy로 나머지 할당
            unassigned_df = df[df['svc_mgmt_num'].isin(unassigned_users)]
            greedy_result = self.optimize_greedy(unassigned_df, remaining_capacity)
            
            # 결합
            combined = pd.concat([ga_result, greedy_result], ignore_index=True)
            print(f"\nTotal assigned: {self.num_formatter(len(combined))}")
            
            return combined
            
        except Exception as e:
            print(f"\nGA failed: {e}")
            print("Running full Greedy allocation...")
            return self.optimize_greedy(df, capacity_per_hour)
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    def _collect_user_data(self, df: pd.DataFrame) -> Dict[str, Dict[int, float]]:
        """DataFrame에서 사용자 데이터 수집"""
        user_data = {}
        for _, row in df.iterrows():
            user = row['svc_mgmt_num']
            hour = int(row['send_hour'])
            score = float(row['propensity_score'])
            
            if user not in user_data:
                user_data[user] = {}
            user_data[user][hour] = score
        
        return user_data
    
    def _extract_solution(
        self,
        solution: np.ndarray,
        user_data: Dict[str, Dict[int, float]],
        users: List[str],
        hours: List[int]
    ) -> List[AllocationResult]:
        """솔루션에서 결과 추출"""
        results = []
        
        for i, user in enumerate(users):
            hour_idx = int(solution[i])
            hour = hours[hour_idx]
            
            if hour in user_data[user]:
                score = user_data[user][hour]
                results.append(AllocationResult(user, hour, score))
        
        return results
    
    def _validate_and_print_stats(
        self,
        results: List[AllocationResult],
        users: List[str],
        hours: List[int],
        capacity_per_hour: Dict[int, int]
    ):
        """결과 검증 및 통계 출력"""
        print("\n" + "=" * 80)
        print("Allocation Statistics")
        print("=" * 80)
        
        total_assigned = len(results)
        total_score = sum(r.score for r in results)
        avg_score = total_score / total_assigned if total_assigned > 0 else 0
        
        print(f"\nTotal assigned: {self.num_formatter(total_assigned)} / {self.num_formatter(len(users))}")
        print(f"Total score: {total_score:,.2f}")
        print(f"Average score: {avg_score:.4f}")
        
        # 시간대별 할당
        hour_stats = {}
        for r in results:
            if r.assigned_hour not in hour_stats:
                hour_stats[r.assigned_hour] = []
            hour_stats[r.assigned_hour].append(r.score)
        
        print("\nHour-wise allocation:")
        print("-" * 60)
        print(f"{'Hour':<8} {'Count':>12} {'Total Score':>15} {'Avg Score':>12}")
        print("-" * 60)
        
        violation_detected = False
        for hour in sorted(hours):
            hour_results = hour_stats.get(hour, [])
            count = len(hour_results)
            capacity = capacity_per_hour.get(hour, 0)
            hour_total = sum(hour_results)
            hour_avg = hour_total / count if count > 0 else 0
            utilization = count / capacity * 100 if capacity > 0 else 0
            
            status = "✓" if count <= capacity else "✗ VIOLATION"
            
            print(f"{hour:<8} {self.num_formatter(count):>12} "
                  f"{hour_total:>15,.2f} {hour_avg:>12.4f} "
                  f"({utilization:>5.1f}%) {status}")
            
            if count > capacity:
                violation_detected = True
        
        print("-" * 60)
        
        if violation_detected:
            print("\n✗ WARNING: Some capacity constraints violated!")
        else:
            print("\n✓ All capacity constraints satisfied")


# ============================================================================
# PySpark Integration
# ============================================================================

def optimize_with_pyspark(
    spark_df,
    capacity_per_hour: Dict[int, int],
    method: str = "greedy",
    **kwargs
) -> 'pyspark.sql.DataFrame':
    """
    PySpark DataFrame을 사용한 최적화
    
    Args:
        spark_df: PySpark DataFrame with [svc_mgmt_num, send_hour, propensity_score]
        capacity_per_hour: {hour: capacity}
        method: 'greedy', 'ga', 'hybrid'
        **kwargs: 각 메서드별 파라미터
    
    Returns:
        PySpark DataFrame with results
    """
    from pyspark.sql import SparkSession
    
    # Pandas로 변환 (collect)
    pdf = spark_df.toPandas()
    
    # 최적화
    optimizer = SendTimeOptimizer()
    
    if method == "greedy":
        result_pdf = optimizer.optimize_greedy(pdf, capacity_per_hour)
    elif method == "ga":
        result_pdf = optimizer.optimize_with_ga(pdf, capacity_per_hour, **kwargs)
    elif method == "hybrid":
        result_pdf = optimizer.optimize_hybrid(pdf, capacity_per_hour, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Spark DataFrame으로 변환
    spark = SparkSession.builder.getOrCreate()
    result_sdf = spark.createDataFrame(result_pdf)
    
    return result_sdf


# ============================================================================
# Usage Examples
# ============================================================================

if __name__ == "__main__":
    # 샘플 데이터 생성
    np.random.seed(42)
    
    n_users = 1000
    hours = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    
    data = []
    for i in range(n_users):
        user_id = f"user_{i:05d}"
        # 각 사용자는 3-5개 시간대 선택 가능
        n_choices = np.random.randint(3, 6)
        selected_hours = np.random.choice(hours, n_choices, replace=False)
        
        for hour in selected_hours:
            score = np.random.beta(2, 5)  # 0~1 사이 점수
            data.append({
                'svc_mgmt_num': user_id,
                'send_hour': hour,
                'propensity_score': score
            })
    
    df = pd.DataFrame(data)
    
    # 용량 설정
    capacity_per_hour = {hour: 100 for hour in hours}
    
    print("Sample Data:")
    print(df.head(10))
    print(f"\nTotal records: {len(df)}")
    print(f"Unique users: {df['svc_mgmt_num'].nunique()}")
    
    # ========================================================================
    # 1. Greedy 알고리즘 (가장 빠름)
    # ========================================================================
    optimizer = SendTimeOptimizer()
    result_greedy = optimizer.optimize_greedy(df, capacity_per_hour)
    
    print("\n" + "=" * 80)
    print("Greedy Result:")
    print(result_greedy.groupby('assigned_hour').agg({
        'svc_mgmt_num': 'count',
        'score': ['sum', 'mean']
    }))
    
    # ========================================================================
    # 2. Genetic Algorithm (더 최적화된 결과)
    # ========================================================================
    result_ga = optimizer.optimize_with_ga(
        df,
        capacity_per_hour,
        pop_size=50,
        n_gen=100
    )
    
    print("\n" + "=" * 80)
    print("GA Result:")
    print(result_ga.groupby('assigned_hour').agg({
        'svc_mgmt_num': 'count',
        'score': ['sum', 'mean']
    }))
    
    # ========================================================================
    # 3. Hybrid (GA + Greedy) - 가장 안정적
    # ========================================================================
    result_hybrid = optimizer.optimize_hybrid(
        df,
        capacity_per_hour,
        ga_pop_size=50,
        ga_n_gen=100
    )
    
    print("\n" + "=" * 80)
    print("Hybrid Result:")
    print(result_hybrid.groupby('assigned_hour').agg({
        'svc_mgmt_num': 'count',
        'score': ['sum', 'mean']
    }))
    
    # 결과 비교
    print("\n" + "=" * 80)
    print("Result Comparison:")
    print("-" * 80)
    print(f"Greedy  - Total Score: {result_greedy['score'].sum():,.2f}")
    print(f"GA      - Total Score: {result_ga['score'].sum():,.2f}")
    print(f"Hybrid  - Total Score: {result_hybrid['score'].sum():,.2f}")
