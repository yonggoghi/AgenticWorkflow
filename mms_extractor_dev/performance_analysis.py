#!/usr/bin/env python3
"""
성능 테스트 결과 분석 및 시각화
==============================
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# 한글 폰트 설정
plt.rcParams['font.family'] = ['AppleGothic', 'Malgun Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_results():
    """성능 테스트 결과 로드"""
    # 가장 최근 결과 파일 찾기
    result_files = list(Path('.').glob('performance_test_results_*.json'))
    if not result_files:
        print("성능 테스트 결과 파일을 찾을 수 없습니다.")
        return None
    
    latest_file = max(result_files, key=lambda f: f.stat().st_mtime)
    print(f"로드할 결과 파일: {latest_file}")
    
    with open(latest_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_single_message_performance(results):
    """단일 메시지 처리 성능 분석"""
    single = results['single_message']
    
    print("=" * 60)
    print("📋 단일 메시지 처리 성능 분석")
    print("=" * 60)
    
    categories = ['순차\n(DAG 없음)', '순차\n(DAG 포함)', '병렬\n(DAG 포함)']
    times = [single['sequential_avg'], single['sequential_dag_avg'], single['parallel_dag_avg']]
    errors = [single['sequential_std'], single['sequential_dag_std'], single['parallel_dag_std']]
    
    # 막대 그래프 생성
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    bars = ax.bar(categories, times, yerr=errors, capsize=5, 
                  color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.8)
    
    # 값 표시
    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{time:.2f}초', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('처리 시간 (초)', fontsize=12)
    ax.set_title('단일 메시지 처리 성능 비교', fontsize=16, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)
    
    # 성능 향상 표시
    speedup = single['speedup_ratio']
    time_saved = single['sequential_dag_avg'] - single['parallel_dag_avg']
    
    ax.text(0.02, 0.98, f'🎯 병렬 처리 성능 향상: {speedup:.2f}x\n⏰ 시간 단축: {time_saved:.2f}초', 
            transform=ax.transAxes, va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
            fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('single_message_performance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return speedup, time_saved

def analyze_batch_performance(results):
    """배치 처리 성능 분석"""
    batch = results['batch_processing']
    
    print("=" * 60)
    print("📦 배치 처리 성능 분석 (10개 메시지)")
    print("=" * 60)
    
    workers = [1, 2, 4]
    no_dag_times = [batch[f'workers_{w}']['no_dag_time'] for w in workers]
    with_dag_times = [batch[f'workers_{w}']['with_dag_time'] for w in workers]
    
    # 성능 향상 계산
    baseline_no_dag = no_dag_times[0]
    baseline_with_dag = with_dag_times[0]
    
    no_dag_speedups = [baseline_no_dag / time for time in no_dag_times]
    with_dag_speedups = [baseline_with_dag / time for time in with_dag_times]
    
    # 두 개의 서브플롯 생성
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 처리 시간 그래프
    x = np.arange(len(workers))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, no_dag_times, width, label='DAG 없음', 
                    color='#3498db', alpha=0.8)
    bars2 = ax1.bar(x + width/2, with_dag_times, width, label='DAG 포함', 
                    color='#e74c3c', alpha=0.8)
    
    # 값 표시
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}초', ha='center', va='bottom', fontsize=10)
    
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{height:.1f}초', ha='center', va='bottom', fontsize=10)
    
    ax1.set_xlabel('워커 수', fontsize=12)
    ax1.set_ylabel('총 처리 시간 (초)', fontsize=12)
    ax1.set_title('워커 수별 배치 처리 시간', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(workers)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 성능 향상 그래프
    bars3 = ax2.bar(x - width/2, no_dag_speedups, width, label='DAG 없음', 
                    color='#2ecc71', alpha=0.8)
    bars4 = ax2.bar(x + width/2, with_dag_speedups, width, label='DAG 포함', 
                    color='#f39c12', alpha=0.8)
    
    # 값 표시
    for bar, speedup in zip(bars3, no_dag_speedups):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{speedup:.2f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    for bar, speedup in zip(bars4, with_dag_speedups):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{speedup:.2f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2.set_xlabel('워커 수', fontsize=12)
    ax2.set_ylabel('성능 향상 배수', fontsize=12)
    ax2.set_title('워커 수별 성능 향상', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(workers)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('batch_performance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return no_dag_speedups, with_dag_speedups

def analyze_dag_overhead(results):
    """DAG 추출 오버헤드 분석"""
    dag = results['dag_comparison']
    
    print("=" * 60)
    print("🎯 DAG 추출 성능 영향 분석")
    print("=" * 60)
    
    categories = ['DAG 없음', 'DAG 포함']
    times = [dag['no_dag_avg'], dag['with_dag_avg']]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 처리 시간 비교
    colors = ['#3498db', '#e74c3c']
    bars = ax1.bar(categories, times, color=colors, alpha=0.8)
    
    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{time:.2f}초', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax1.set_ylabel('평균 처리 시간 (초)', fontsize=12)
    ax1.set_title('DAG 추출 유무별 처리 시간', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # 오버헤드 정보
    overhead_ratio = dag['overhead_ratio']
    overhead_seconds = dag['overhead_seconds']
    
    ax1.text(0.02, 0.98, f'오버헤드: {overhead_ratio:.1f}x\n추가 시간: {overhead_seconds:.1f}초', 
            transform=ax1.transAxes, va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
            fontsize=11, fontweight='bold')
    
    # 파이 차트
    labels = ['기본 처리', 'DAG 오버헤드']
    sizes = [dag['no_dag_avg'], overhead_seconds]
    colors = ['#3498db', '#e74c3c']
    explode = (0, 0.1)
    
    ax2.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax2.set_title('처리 시간 구성 비율', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('dag_overhead_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return overhead_ratio, overhead_seconds

def generate_summary_report(results):
    """종합 성능 리포트 생성"""
    single = results['single_message']
    batch = results['batch_processing']
    dag = results['dag_comparison']
    
    print("\n" + "=" * 80)
    print("🚀 MMS 추출기 병렬 처리 성능 테스트 - 종합 분석 결과")
    print("=" * 80)
    
    print("\n📊 핵심 성과 지표:")
    print("-" * 50)
    
    # 단일 메시지 성능 향상
    speedup = single['speedup_ratio']
    time_saved = single['sequential_dag_avg'] - single['parallel_dag_avg']
    print(f"• 단일 메시지 병렬 처리 성능 향상: {speedup:.2f}x ({time_saved:.2f}초 단축)")
    
    # 배치 처리 성능 향상
    workers_4_speedup_no_dag = batch['workers_1']['no_dag_time'] / batch['workers_4']['no_dag_time']
    workers_4_speedup_with_dag = batch['workers_1']['with_dag_time'] / batch['workers_4']['with_dag_time']
    
    print(f"• 배치 처리 성능 향상 (4워커):")
    print(f"  - DAG 없음: {workers_4_speedup_no_dag:.2f}x")
    print(f"  - DAG 포함: {workers_4_speedup_with_dag:.2f}x")
    
    # 처리량 개선
    single_msg_time_no_dag = dag['no_dag_avg']
    batch_msg_time_no_dag = batch['workers_4']['no_dag_per_message']
    batch_improvement = single_msg_time_no_dag / batch_msg_time_no_dag
    
    print(f"• 메시지당 처리 시간 개선:")
    print(f"  - 단일 처리: {single_msg_time_no_dag:.2f}초/메시지")
    print(f"  - 배치 처리 (4워커): {batch_msg_time_no_dag:.2f}초/메시지")
    print(f"  - 개선율: {batch_improvement:.2f}x")
    
    print(f"\n🎯 DAG 추출 오버헤드:")
    print(f"• 추가 처리 시간: {dag['overhead_seconds']:.2f}초 ({dag['overhead_ratio']:.1f}x)")
    print(f"• 병렬 처리로 오버헤드 {((dag['overhead_ratio'] - single['speedup_ratio']) / dag['overhead_ratio'] * 100):.1f}% 감소")
    
    print(f"\n💡 최적 설정 권장사항:")
    print("-" * 50)
    print(f"• DAG 추출 없는 일반 처리: 4워커 사용 권장")
    print(f"• DAG 추출 포함 처리: 2-4워커 사용 권장 (리소스 대비 효율)")
    print(f"• 대량 처리 시: 50-100개 메시지 단위로 배치 처리")
    print(f"• 실시간 처리가 중요한 경우: DAG 추출 비활성화")
    
    # 시간당 처리량 계산
    hourly_single = 3600 / single_msg_time_no_dag
    hourly_batch = 3600 / batch_msg_time_no_dag
    
    print(f"\n📈 예상 처리량 (시간당):")
    print(f"• 단일 처리: {hourly_single:.0f}개 메시지/시간")
    print(f"• 배치 처리 (4워커): {hourly_batch:.0f}개 메시지/시간")
    print(f"• 처리량 증가: {(hourly_batch - hourly_single):.0f}개 메시지/시간 ({((hourly_batch / hourly_single - 1) * 100):.1f}% 향상)")

def main():
    """메인 함수"""
    results = load_results()
    if not results:
        return
    
    print("🔍 성능 테스트 결과 분석 시작...")
    
    # 1. 단일 메시지 성능 분석
    speedup, time_saved = analyze_single_message_performance(results)
    
    # 2. 배치 처리 성능 분석
    no_dag_speedups, with_dag_speedups = analyze_batch_performance(results)
    
    # 3. DAG 오버헤드 분석
    overhead_ratio, overhead_seconds = analyze_dag_overhead(results)
    
    # 4. 종합 리포트 생성
    generate_summary_report(results)
    
    print(f"\n✅ 성능 분석 완료!")
    print(f"📊 생성된 그래프:")
    print(f"  • single_message_performance.png")
    print(f"  • batch_performance.png")
    print(f"  • dag_overhead_analysis.png")

if __name__ == "__main__":
    main()
