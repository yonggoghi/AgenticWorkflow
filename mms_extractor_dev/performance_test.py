#!/usr/bin/env python3
"""
MMS 추출기 병렬 처리 성능 테스트
================================

이 스크립트는 MMS 추출기의 병렬 처리 성능을 측정하고 비교합니다.

테스트 항목:
1. 단일 메시지 처리 (순차 vs 병렬)
2. 배치 메시지 처리 (순차 vs 병렬)
3. DAG 추출 포함/미포함 성능 비교
4. 워커 수별 성능 변화
"""

import time
import json
import statistics
from pathlib import Path
import sys
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import pandas as pd

# 현재 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

from mms_extractor import MMSExtractor, process_message_with_dag, process_messages_batch, make_entity_dag

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 테스트용 샘플 메시지들
SAMPLE_MESSAGES = [
    """[SKT] ZEM폰 포켓몬에디션3 안내
    (광고)[SKT] 우리 아이 첫 번째 스마트폰, ZEM 키즈폰__#04 고객님, 안녕하세요!
    우리 아이 스마트폰 고민 중이셨다면, 자녀 스마트폰 관리 앱 ZEM이 설치된 SKT만의 안전한 키즈폰,
    ZEM폰 포켓몬에디션3으로 우리 아이 취향을 저격해 보세요!""",
    
    """[KT] 5G 프리미엄 요금제 혜택 안내
    안녕하세요! KT입니다. 새로운 5G 프리미엄 요금제로 더 빠르고 안정적인 통신 서비스를 경험해보세요.
    월 59,000원으로 데이터 무제한, 영상통화 무제한 혜택을 드립니다.""",
    
    """[LGU+] U+알뜰폰 가입 혜택
    LG유플러스 U+알뜰폰 가입하고 월 통신비 절약하세요!
    데이터 5GB + 통화 300분 월 25,000원 특가 혜택을 놓치지 마세요.""",
    
    """[현대카드] M포인트 적립 안내
    현대카드 사용 시 M포인트가 자동 적립됩니다.
    적립된 포인트로 다양한 혜택을 받아보세요. 포인트 조회는 앱에서 확인 가능합니다.""",
    
    """[삼성전자] 갤럭시 S24 출시 기념
    새로운 갤럭시 S24가 출시되었습니다!
    AI 카메라와 향상된 배터리로 더 스마트한 경험을 제공합니다."""
]

class PerformanceTester:
    """성능 테스트 클래스"""
    
    def __init__(self):
        """테스터 초기화"""
        logger.info("성능 테스터 초기화 중...")
        self.extractor = MMSExtractor(
            offer_info_data_src='local',
            product_info_extraction_mode='nlp',
            entity_extraction_mode='logic',
            llm_model='ax',
            extract_entity_dag=False
        )
        logger.info("추출기 초기화 완료")
        
        self.results = {
            'single_message': {},
            'batch_processing': {},
            'dag_comparison': {},
            'worker_scaling': {}
        }
    
    def test_single_message_performance(self, iterations: int = 3) -> Dict[str, Any]:
        """단일 메시지 처리 성능 테스트 (순차 vs 병렬)"""
        logger.info(f"단일 메시지 성능 테스트 시작 (반복: {iterations}회)")
        
        message = SAMPLE_MESSAGES[0]
        
        # 순차 처리 (DAG 없음)
        sequential_times = []
        for i in range(iterations):
            start_time = time.time()
            result = self.extractor.process_message(message)
            elapsed = time.time() - start_time
            sequential_times.append(elapsed)
            logger.info(f"순차 처리 {i+1}/{iterations}: {elapsed:.3f}초")
        
        # 순차 처리 (DAG 포함)
        sequential_dag_times = []
        for i in range(iterations):
            start_time = time.time()
            result = self.extractor.process_message(message)
            dag_result = make_entity_dag(message, self.extractor.llm_model)
            elapsed = time.time() - start_time
            sequential_dag_times.append(elapsed)
            logger.info(f"순차 처리 (DAG) {i+1}/{iterations}: {elapsed:.3f}초")
        
        # 병렬 처리 (DAG 포함)
        parallel_dag_times = []
        for i in range(iterations):
            start_time = time.time()
            result = process_message_with_dag(self.extractor, message, extract_dag=True)
            elapsed = time.time() - start_time
            parallel_dag_times.append(elapsed)
            logger.info(f"병렬 처리 (DAG) {i+1}/{iterations}: {elapsed:.3f}초")
        
        results = {
            'sequential_avg': statistics.mean(sequential_times),
            'sequential_std': statistics.stdev(sequential_times) if len(sequential_times) > 1 else 0,
            'sequential_dag_avg': statistics.mean(sequential_dag_times),
            'sequential_dag_std': statistics.stdev(sequential_dag_times) if len(sequential_dag_times) > 1 else 0,
            'parallel_dag_avg': statistics.mean(parallel_dag_times),
            'parallel_dag_std': statistics.stdev(parallel_dag_times) if len(parallel_dag_times) > 1 else 0,
            'speedup_ratio': statistics.mean(sequential_dag_times) / statistics.mean(parallel_dag_times)
        }
        
        self.results['single_message'] = results
        return results
    
    def test_batch_processing_performance(self, worker_counts: List[int] = [1, 2, 4, 8]) -> Dict[str, Any]:
        """배치 처리 성능 테스트 (워커 수별)"""
        logger.info(f"배치 처리 성능 테스트 시작 (워커 수: {worker_counts})")
        
        messages = SAMPLE_MESSAGES * 2  # 10개 메시지
        
        results = {}
        
        for worker_count in worker_counts:
            logger.info(f"워커 {worker_count}개로 배치 처리 테스트")
            
            # DAG 없음
            start_time = time.time()
            batch_results = process_messages_batch(
                self.extractor, 
                messages, 
                extract_dag=False,
                max_workers=worker_count
            )
            elapsed_no_dag = time.time() - start_time
            
            # DAG 포함
            start_time = time.time()
            batch_results_dag = process_messages_batch(
                self.extractor, 
                messages, 
                extract_dag=True,
                max_workers=worker_count
            )
            elapsed_with_dag = time.time() - start_time
            
            results[f'workers_{worker_count}'] = {
                'no_dag_time': elapsed_no_dag,
                'with_dag_time': elapsed_with_dag,
                'no_dag_per_message': elapsed_no_dag / len(messages),
                'with_dag_per_message': elapsed_with_dag / len(messages),
                'successful_count': len([r for r in batch_results if not r.get('error')])
            }
            
            logger.info(f"워커 {worker_count}개 - DAG 없음: {elapsed_no_dag:.3f}초, DAG 포함: {elapsed_with_dag:.3f}초")
        
        self.results['batch_processing'] = results
        return results
    
    def test_dag_extraction_impact(self, iterations: int = 3) -> Dict[str, Any]:
        """DAG 추출이 성능에 미치는 영향 테스트"""
        logger.info(f"DAG 추출 성능 영향 테스트 시작 (반복: {iterations}회)")
        
        message = SAMPLE_MESSAGES[0]
        
        # DAG 없이 처리
        no_dag_times = []
        for i in range(iterations):
            start_time = time.time()
            result = self.extractor.process_message(message)
            elapsed = time.time() - start_time
            no_dag_times.append(elapsed)
        
        # DAG 포함 병렬 처리
        with_dag_times = []
        for i in range(iterations):
            start_time = time.time()
            result = process_message_with_dag(self.extractor, message, extract_dag=True)
            elapsed = time.time() - start_time
            with_dag_times.append(elapsed)
        
        results = {
            'no_dag_avg': statistics.mean(no_dag_times),
            'with_dag_avg': statistics.mean(with_dag_times),
            'overhead_ratio': statistics.mean(with_dag_times) / statistics.mean(no_dag_times),
            'overhead_seconds': statistics.mean(with_dag_times) - statistics.mean(no_dag_times)
        }
        
        self.results['dag_comparison'] = results
        return results
    
    def generate_performance_report(self) -> str:
        """성능 테스트 결과 리포트 생성"""
        report = []
        report.append("=" * 80)
        report.append("🚀 MMS 추출기 병렬 처리 성능 테스트 결과")
        report.append("=" * 80)
        
        # 단일 메시지 처리 결과
        if 'single_message' in self.results:
            single = self.results['single_message']
            report.append("\n📋 1. 단일 메시지 처리 성능")
            report.append("-" * 50)
            report.append(f"순차 처리 (DAG 없음):     {single['sequential_avg']:.3f}초 (±{single['sequential_std']:.3f})")
            report.append(f"순차 처리 (DAG 포함):     {single['sequential_dag_avg']:.3f}초 (±{single['sequential_dag_std']:.3f})")
            report.append(f"병렬 처리 (DAG 포함):     {single['parallel_dag_avg']:.3f}초 (±{single['parallel_dag_std']:.3f})")
            report.append(f"🎯 병렬 처리 성능 향상:    {single['speedup_ratio']:.2f}x 빠름")
            report.append(f"⏰ 시간 단축:             {(single['sequential_dag_avg'] - single['parallel_dag_avg']):.3f}초")
        
        # 배치 처리 결과
        if 'batch_processing' in self.results:
            batch = self.results['batch_processing']
            report.append("\n📦 2. 배치 처리 성능 (10개 메시지)")
            report.append("-" * 50)
            
            for key, data in batch.items():
                worker_count = key.split('_')[1]
                report.append(f"워커 {worker_count}개:")
                report.append(f"  - DAG 없음:  {data['no_dag_time']:.3f}초 ({data['no_dag_per_message']:.3f}초/메시지)")
                report.append(f"  - DAG 포함:  {data['with_dag_time']:.3f}초 ({data['with_dag_per_message']:.3f}초/메시지)")
                report.append(f"  - 성공률:    {data['successful_count']}/10")
        
        # DAG 추출 영향
        if 'dag_comparison' in self.results:
            dag = self.results['dag_comparison']
            report.append("\n🎯 3. DAG 추출 성능 영향")
            report.append("-" * 50)
            report.append(f"DAG 없음:        {dag['no_dag_avg']:.3f}초")
            report.append(f"DAG 포함:        {dag['with_dag_avg']:.3f}초")
            report.append(f"오버헤드 비율:   {dag['overhead_ratio']:.2f}x")
            report.append(f"추가 시간:       {dag['overhead_seconds']:.3f}초")
        
        # 권장사항
        report.append("\n💡 성능 최적화 권장사항")
        report.append("-" * 50)
        report.append("• DAG 추출이 필요한 경우에만 활성화")
        report.append("• 배치 처리 시 적절한 워커 수 설정 (CPU 코어 수의 50-100%)")
        report.append("• 대량 처리 시 배치 크기를 50-100개로 제한")
        report.append("• 로컬 데이터 소스 사용 권장 (DB보다 빠름)")
        
        return "\n".join(report)
    
    def save_results_to_json(self, filename: str = None):
        """결과를 JSON 파일로 저장"""
        if filename is None:
            filename = f"performance_test_results_{int(time.time())}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=4, ensure_ascii=False)
        
        logger.info(f"성능 테스트 결과 저장: {filename}")
        return filename

def main():
    """메인 함수"""
    print("🚀 MMS 추출기 병렬 처리 성능 테스트 시작")
    print("=" * 60)
    
    tester = PerformanceTester()
    
    try:
        # 1. 단일 메시지 성능 테스트
        print("\n1️⃣ 단일 메시지 처리 성능 테스트...")
        single_results = tester.test_single_message_performance(iterations=3)
        
        # 2. DAG 추출 영향 테스트
        print("\n2️⃣ DAG 추출 성능 영향 테스트...")
        dag_results = tester.test_dag_extraction_impact(iterations=3)
        
        # 3. 배치 처리 성능 테스트
        print("\n3️⃣ 배치 처리 성능 테스트...")
        batch_results = tester.test_batch_processing_performance(worker_counts=[1, 2, 4])
        
        # 결과 리포트 생성 및 출력
        report = tester.generate_performance_report()
        print("\n" + report)
        
        # 결과를 JSON 파일로 저장
        json_file = tester.save_results_to_json()
        
        print(f"\n✅ 성능 테스트 완료!")
        print(f"📄 상세 결과: {json_file}")
        
        # 핵심 성과 요약
        if 'single_message' in tester.results:
            speedup = tester.results['single_message']['speedup_ratio']
            time_saved = tester.results['single_message']['sequential_dag_avg'] - tester.results['single_message']['parallel_dag_avg']
            print(f"\n🎯 핵심 성과:")
            print(f"   • 병렬 처리로 {speedup:.1f}x 성능 향상")
            print(f"   • 메시지당 {time_saved:.3f}초 시간 단축")
        
    except Exception as e:
        logger.error(f"성능 테스트 중 오류 발생: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
