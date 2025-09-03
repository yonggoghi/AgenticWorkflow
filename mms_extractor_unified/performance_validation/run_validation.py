#!/usr/bin/env python3
"""
성능 검증 통합 실행 스크립트
==========================

전체 모델 비교 실험을 실행하는 메인 스크립트입니다.
1. 모델 추출 실험 실행
2. 성능 평가 실행
3. 결과 리포트 생성
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime

# 상위 디렉토리를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'validation_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_extraction_experiment(batch_size: int, output_dir: str, min_message_length: int = 300):
    """모델 추출 실험 실행"""
    logger.info("=== 1단계: 모델 추출 실험 실행 ===")
    
    try:
        from model_comparison_experiment import ModelComparisonExperiment
        
        # 실험 객체 생성 (타임스탬프는 ModelComparisonExperiment에서 자동 추가됨)
        experiment = ModelComparisonExperiment(
            batch_size=batch_size,
            output_dir=output_dir,
            min_message_length=min_message_length
        )
        
        # MMS 데이터 로딩
        logger.info("MMS 데이터 로딩")
        messages_df = experiment.load_mms_data()
        
        # 모든 모델로 추출 실행
        logger.info("모든 모델로 추출 실행")
        experiment.run_extraction_for_all_models(messages_df)
        
        # 결과 저장
        logger.info("결과 저장")
        experiment.save_combined_results()
        
        logger.info("✅ 모델 추출 실험 완료")
        return str(experiment.output_dir)  # 실제 생성된 디렉토리 경로 반환
        
    except Exception as e:
        logger.error(f"❌ 모델 추출 실험 실패: {str(e)}")
        return None

def run_performance_evaluation(results_dir: str, similarity_threshold: float = 0.9, min_message_length: int = 300):
    """성능 평가 실행"""
    logger.info("=== 2단계: 성능 평가 실행 ===")
    
    try:
        from model_performance_evaluator import ModelPerformanceEvaluator
        
        # 평가기 객체 생성
        evaluator = ModelPerformanceEvaluator(results_dir=results_dir)
        
        # 추출 결과 로딩
        logger.info("추출 결과 로딩")
        evaluator.load_extraction_results()
        
        # 정답 데이터셋 생성
        logger.info("정답 데이터셋 생성")
        evaluator.generate_ground_truth_dataset(
            similarity_threshold=similarity_threshold,
            min_message_length=min_message_length
        )
        
        # 대상 모델 평가
        logger.info("대상 모델 평가")
        evaluator.evaluate_target_models()
        
        # 성능 리포트 생성
        logger.info("성능 리포트 생성")
        evaluator.generate_performance_report()
        
        logger.info("✅ 성능 평가 완료")
        return True
        
    except Exception as e:
        logger.error(f"❌ 성능 평가 실패: {str(e)}")
        return False

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="성능 검증 전체 실행")
    parser.add_argument('--batch-size', type=int, default=100, help='배치 크기 (기본값: 100)')
    parser.add_argument('--output-dir', type=str, default='results', help='결과 저장 디렉토리')
    parser.add_argument('--similarity-threshold', type=float, default=0.9, help='정답 생성 유사도 임계값 (기본값: 0.9)')
    parser.add_argument('--min-message-length', type=int, default=300, help='정답용 메시지 최소 길이 (기본값: 300)')
    parser.add_argument('--skip-extraction', action='store_true', help='추출 단계 건너뛰기 (이미 결과가 있는 경우)')
    parser.add_argument('--skip-evaluation', action='store_true', help='평가 단계 건너뛰기')
    
    args = parser.parse_args()
    
    logger.info("=== 성능 검증 전체 실행 시작 ===")
    logger.info(f"설정: 배치 크기={args.batch_size}, 출력 디렉토리={args.output_dir}")
    logger.info(f"유사도 임계값={args.similarity_threshold}, 최소 메시지 길이={args.min_message_length}자")
    
    success = True
    
    # 1단계: 모델 추출 실험
    actual_output_dir = args.output_dir
    if not args.skip_extraction:
        actual_output_dir = run_extraction_experiment(args.batch_size, args.output_dir, args.min_message_length)
        if not actual_output_dir:
            logger.error("추출 실험 실패로 인해 전체 실험을 중단합니다")
            return 1
        logger.info(f"실제 생성된 결과 디렉토리: {actual_output_dir}")
    else:
        logger.info("추출 단계 건너뛰기 (기존 결과 사용)")
    
    # 2단계: 성능 평가
    if not args.skip_evaluation:
        success = run_performance_evaluation(actual_output_dir, args.similarity_threshold, args.min_message_length)
        if not success:
            logger.error("성능 평가 실패")
            return 1
    else:
        logger.info("평가 단계 건너뛰기")
    
    logger.info("=== 성능 검증 전체 완료 ===")
    logger.info(f"결과는 {actual_output_dir} 디렉토리에서 확인할 수 있습니다")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
