#!/usr/bin/env python3
"""
모델 성능 평가 스크립트
=====================

모델 비교 실험 결과를 분석하여 성능을 평가합니다.
1. 정답 데이터셋 생성 (3개 모델 결과의 종합 유사도 90% 이상인 경우의 claude 결과)
2. gemma, ax 모델을 정답과 비교하여 성능 평가
"""

import os
import sys
import pandas as pd
import json
import pickle
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import numpy as np
from collections import defaultdict

# 상위 디렉토리를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 유사도 계산 함수들
from difflib import SequenceMatcher

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'model_evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# === 유사도 계산 함수들 (사용자 제공 코드) ===

def calculate_list_similarity(list1, list2):
    """Calculate Jaccard similarity between two lists"""
    if isinstance(list1, dict):
        list1 = [str(item) for item in list1.values()]
    if isinstance(list2, dict):
        list2 = [str(item) for item in list2.values()]
    # Ensure lists contain strings
    list1 = [str(item) for item in list1]
    list2 = [str(item) for item in list2]
    # Convert lists to sets for comparison
    set1 = set(sorted(set(list1)))
    set2 = set(sorted(set(list2)))
    # Calculate Jaccard similarity
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0

def calculate_text_similarity(text1, text2):
    """Calculate text similarity using SequenceMatcher"""
    return SequenceMatcher(None, str(text1), str(text2)).ratio()

def calculate_product_similarity(prod1, prod2):
    """Calculate similarity between product dictionaries with detailed structure"""
    if not isinstance(prod1, dict) or not isinstance(prod2, dict):
        return 0.0
    
    # Calculate similarity for each field
    item_name_message_sim = calculate_text_similarity(
        prod1.get('item_name_in_message', '#'),
        prod2.get('item_name_in_message', '&')
    )
    item_name_voca_sim = calculate_text_similarity(
        prod1.get('item_name_in_voca', '#'),
        prod2.get('item_name_in_voca', '&')
    )
    item_id_sim = calculate_text_similarity(
        prod1.get('item_id', '#'),
        prod2.get('item_id', '&')
    )
    domain_sim = calculate_text_similarity(
        prod1.get('domain', '#'),
        prod2.get('domain', '&')
    )
    name_sim = calculate_text_similarity(
        prod1.get('name', '#'),
        prod2.get('name', '&')
    )
    action_sim = calculate_text_similarity(
        prod1.get('action', '#'),
        prod2.get('action', '&')
    )
    
    # Weighted average - item_id and domain are more distinctive
    similarity = (
        item_name_message_sim +
        item_name_voca_sim +
        item_id_sim +
        domain_sim +
        name_sim +
        action_sim
    )/len(prod1.keys())
    return similarity

def calculate_channel_similarity(chan1, chan2):
    """Calculate similarity between channel dictionaries"""
    if not isinstance(chan1, dict) or not isinstance(chan2, dict):
        return 0.0
    type_sim = calculate_text_similarity(chan1.get('type', ''), chan2.get('type', ''))
    value_sim = calculate_text_similarity(chan1.get('value', ''), chan2.get('value', ''))
    action_sim = calculate_text_similarity(chan1.get('action', ''), chan2.get('action', ''))
    return (type_sim + value_sim + action_sim) / 3

def calculate_pgm_similarity(pgm1, pgm2):
    """Calculate similarity between program dictionaries"""
    if isinstance(pgm1, dict) and isinstance(pgm2, dict):
        pgm_nm_sim = calculate_text_similarity(pgm1.get('pgm_nm', ''), pgm2.get('pgm_nm', ''))
        pgm_id_sim = calculate_text_similarity(pgm1.get('pgm_id', ''), pgm2.get('pgm_id', ''))
        pgm_sim = pgm_nm_sim * 0.4 + pgm_id_sim * 0.6
    else:
        pgm_sim = 0.0
    return pgm_sim

def calculate_products_list_similarity(products1, products2):
    """Calculate similarity between two lists of product dictionaries"""
    if not products1 or not products2:
        return 0.0
    
    # For each product in list1, find best match in list2
    similarities = []
    for p1 in products1:
        best_match = 0.0
        for p2 in products2:
            similarity = calculate_product_similarity(p1, p2)
            best_match = max(best_match, similarity)
        similarities.append(best_match)
    
    # Also check reverse direction to handle different list sizes
    reverse_similarities = []
    for p2 in products2:
        best_match = 0.0
        for p1 in products1:
            similarity = calculate_product_similarity(p1, p2)
            best_match = max(best_match, similarity)
        reverse_similarities.append(best_match)
    
    # Take average of both directions
    forward_avg = sum(similarities) / len(similarities)
    reverse_avg = sum(reverse_similarities) / len(reverse_similarities)
    return (forward_avg + reverse_avg) / 2

def calculate_channels_list_similarity(channels1, channels2):
    """Calculate similarity between two lists of channel dictionaries"""
    if not channels1 or not channels2:
        return 0.0
    similarities = []
    for c1 in channels1:
        best_match = 0.0
        for c2 in channels2:
            similarity = calculate_channel_similarity(c1, c2)
            best_match = max(best_match, similarity)
        similarities.append(best_match)
    return sum(similarities) / len(similarities)

def calculate_pgms_list_similarity(pgms1, pgms2):
    """Calculate similarity between two lists of program dictionaries"""
    if not pgms1 or not pgms2:
        return 0.0
    if isinstance(pgms1, list) and isinstance(pgms2, list):
        pgm_sim = calculate_list_similarity(pgms1, pgms2)
        return pgm_sim
    
    # For each pgm in list1, find best match in list2
    similarities = []
    for p1 in pgms1:
        best_match = 0.0
        for p2 in pgms2:
            similarity = calculate_pgm_similarity(p1, p2)
            best_match = max(best_match, similarity)
        similarities.append(best_match)
    
    # Also check reverse direction
    reverse_similarities = []
    for p2 in pgms2:
        best_match = 0.0
        for p1 in pgms1:
            similarity = calculate_pgm_similarity(p1, p2)
            best_match = max(best_match, similarity)
        reverse_similarities.append(best_match)
    
    # Take average of both directions
    forward_avg = sum(similarities) / len(similarities)
    reverse_avg = sum(reverse_similarities) / len(reverse_similarities)
    return (forward_avg + reverse_avg) / 2

def calculate_dictionary_similarity(dict1, dict2):
    """
    Calculate similarity between two dictionaries with generalized structure:
    {
        'title': str,
        'purpose': [list of strings],
        'product': [list of product dicts],
        'channel': [list of channel dicts],
        'pgm': [list of program dicts]
    }
    """
    if not isinstance(dict1, dict) or not isinstance(dict2, dict):
        return {'overall_similarity': 0.0, 'error': 'Both inputs must be dictionaries'}
    
    # Calculate title similarity
    title_similarity = calculate_text_similarity(
        dict1.get('title', ''),
        dict2.get('title', '')
    )
    
    # Calculate purpose similarity (list of strings)
    purpose_similarity = calculate_list_similarity(
        dict1.get('purpose', []),
        dict2.get('purpose', [])
    )
    
    # Calculate product similarity (list of product dicts)
    product_similarity = calculate_products_list_similarity(
        dict1.get('product', []),
        dict2.get('product', [])
    )
    
    # Calculate channel similarity (list of channel dicts)
    channel_similarity = calculate_channels_list_similarity(
        dict1.get('channel', []),
        dict2.get('channel', [])
    )
    
    # Calculate pgm similarity (list of program dicts)
    pgm_similarity = calculate_pgms_list_similarity(
        dict1.get('pgm', []),
        dict2.get('pgm', [])
    )
    
    # Calculate overall similarity (weighted average)
    # Adjusted weights to reflect importance of each component
    overall_similarity = (
        title_similarity * 0.2 +
        purpose_similarity * 0.15 +
        product_similarity * 0.35 +
        channel_similarity * 0.15 +
        pgm_similarity * 0.15
    )
    
    return {
        'overall_similarity': overall_similarity,
        'title_similarity': title_similarity,
        'purpose_similarity': purpose_similarity,
        'product_similarity': product_similarity,
        'channel_similarity': channel_similarity,
        'pgm_similarity': pgm_similarity
    }

class ModelPerformanceEvaluator:
    """모델 성능 평가를 수행하는 클래스"""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.evaluation_dir = self.results_dir / "evaluation"
        self.results_dir.mkdir(exist_ok=True)  # 상위 디렉토리도 생성
        self.evaluation_dir.mkdir(exist_ok=True)
        
        # 결과 저장용
        self.extraction_results = {}
        self.ground_truth_data = []
        self.evaluation_results = {}
        
        # 평가 대상 모델들
        self.reference_models = ['gemini', 'gpt', 'claude']  # 정답 생성용
        self.target_models = ['gemma', 'ax']  # 평가 대상
        self.all_models = ['gemma', 'gemini', 'claude', 'ax', 'gpt']
    
    def load_extraction_results(self):
        """추출 결과 로딩"""
        logger.info("추출 결과 로딩 중...")
        
        # 피클 파일이 있으면 우선 사용
        pickle_file = self.results_dir / "combined_extraction_results.pkl"
        if pickle_file.exists():
            with open(pickle_file, 'rb') as f:
                self.extraction_results = pickle.load(f)
            logger.info("피클 파일에서 결과 로딩 완료")
        else:
            # JSON 파일 사용
            json_file = self.results_dir / "combined_extraction_results.json"
            if json_file.exists():
                with open(json_file, 'r', encoding='utf-8') as f:
                    self.extraction_results = json.load(f)
                logger.info("JSON 파일에서 결과 로딩 완료")
            else:
                raise FileNotFoundError("추출 결과 파일을 찾을 수 없습니다")
        
        # 데이터 검증
        for model in self.all_models:
            if model not in self.extraction_results:
                logger.warning(f"{model} 모델 결과가 없습니다")
            else:
                success_count = len([r for r in self.extraction_results[model] if r.get('success', False)])
                total_count = len(self.extraction_results[model])
                logger.info(f"{model}: {success_count}/{total_count} 성공")
    
    def generate_ground_truth_dataset(self, similarity_threshold: float = 0.9, min_message_length: int = 300):
        """정답 데이터셋 생성"""
        logger.info(f"정답 데이터셋 생성 시작 (유사도 임계값: {similarity_threshold}, 최소 메시지 길이: {min_message_length}자)")
        
        # 메시지 ID별로 결과 그룹화
        message_groups = defaultdict(dict)
        message_texts = {}  # 메시지 텍스트 저장
        
        for model in self.reference_models:
            if model in self.extraction_results:
                for result in self.extraction_results[model]:
                    if result.get('success', False):
                        msg_id = result['msg_id']
                        message_groups[msg_id][model] = result['json_objects']
                        message_texts[msg_id] = result['msg']  # 메시지 텍스트 저장
        
        # 3개 모델 모두 성공한 메시지만 고려
        valid_messages = []
        for msg_id, results in message_groups.items():
            if len(results) == len(self.reference_models):
                valid_messages.append(msg_id)
        
        logger.info(f"3개 참조 모델 모두 성공한 메시지: {len(valid_messages)}개")
        
        # 추가 필터링 적용
        filtered_messages = []
        
        for msg_id in valid_messages:
            msg_text = message_texts.get(msg_id, "")
            claude_result = message_groups[msg_id].get('claude', {})
            
            # 1. 메시지 길이 조건 확인 (최소 300자)
            if len(msg_text) < min_message_length:
                logger.debug(f"메시지 {msg_id}: 길이 부족 ({len(msg_text)}자 < {min_message_length}자)")
                continue
            
            # 2. 1st depth 태그들 값이 채워져 있는지 확인
            required_fields = ['title', 'purpose', 'product', 'channel', 'pgm']
            if not self._validate_first_depth_tags(claude_result, required_fields):
                logger.debug(f"메시지 {msg_id}: 1st depth 태그 값 부족")
                continue
            
            filtered_messages.append(msg_id)
        
        logger.info(f"추가 조건 적용 후 유효한 메시지: {len(filtered_messages)}개")
        
        # 유사도 90% 이상인 메시지들 찾기
        high_similarity_messages = []
        
        for msg_id in filtered_messages:
            results = message_groups[msg_id]
            
            # 3개 모델 간 유사도 계산
            similarities = []
            models = list(results.keys())
            
            for i in range(len(models)):
                for j in range(i+1, len(models)):
                    model1, model2 = models[i], models[j]
                    sim_result = calculate_dictionary_similarity(results[model1], results[model2])
                    similarities.append(sim_result['overall_similarity'])
            
            # 평균 유사도 계산
            avg_similarity = np.mean(similarities)
            
            if avg_similarity >= similarity_threshold:
                # claude 결과를 정답으로 사용
                ground_truth_record = {
                    'msg_id': msg_id,
                    'msg': message_texts[msg_id],
                    'msg_length': len(message_texts[msg_id]),
                    'ground_truth': results['claude'],
                    'avg_similarity': avg_similarity,
                    'similarities': dict(zip([f"{models[i]}_{models[j]}" for i in range(len(models)) for j in range(i+1, len(models))], similarities))
                }
                
                high_similarity_messages.append(ground_truth_record)
                self.ground_truth_data.append(ground_truth_record)
        
        logger.info(f"정답 데이터셋 생성 완료: {len(self.ground_truth_data)}개 메시지")
        
        # 정답 데이터셋 저장
        ground_truth_file = self.evaluation_dir / "ground_truth_dataset.json"
        with open(ground_truth_file, 'w', encoding='utf-8') as f:
            json.dump(self.ground_truth_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"정답 데이터셋 저장: {ground_truth_file}")
        
        return self.ground_truth_data
    
    def _validate_first_depth_tags(self, json_result: Dict, required_fields: List[str]) -> bool:
        """1st depth 태그들의 값이 유효하게 채워져 있는지 확인"""
        try:
            for field in required_fields:
                if field not in json_result:
                    return False
                
                value = json_result[field]
                
                # 값이 None이거나 빈 값인지 확인
                if value is None:
                    return False
                
                # 문자열인 경우 빈 문자열 또는 공백만 있는지 확인
                if isinstance(value, str):
                    if not value.strip():
                        return False
                
                # 리스트인 경우 빈 리스트이거나 모든 요소가 빈 값인지 확인
                elif isinstance(value, list):
                    if not value:  # 빈 리스트
                        return False
                    # 모든 요소가 유효한지 확인
                    for item in value:
                        if item is None:
                            return False
                        if isinstance(item, str) and not item.strip():
                            return False
                        elif isinstance(item, dict) and not item:  # 빈 딕셔너리
                            return False
                
                # 딕셔너리인 경우 빈 딕셔너리인지 확인
                elif isinstance(value, dict):
                    if not value:
                        return False
                
            return True
            
        except Exception as e:
            logger.debug(f"1st depth 태그 검증 중 오류: {e}")
            return False
    
    def evaluate_target_models(self):
        """대상 모델들을 정답과 비교하여 평가"""
        logger.info("대상 모델 성능 평가 시작")
        
        if not self.ground_truth_data:
            raise ValueError("정답 데이터셋이 생성되지 않았습니다")
        
        # 정답 데이터셋의 메시지 ID들
        ground_truth_msg_ids = set(record['msg_id'] for record in self.ground_truth_data)
        
        for model in self.target_models:
            logger.info(f"=== {model} 모델 평가 ===")
            
            if model not in self.extraction_results:
                logger.warning(f"{model} 모델 결과가 없습니다")
                continue
            
            model_evaluations = []
            
            for ground_truth_record in self.ground_truth_data:
                msg_id = ground_truth_record['msg_id']
                ground_truth = ground_truth_record['ground_truth']
                
                # 해당 메시지에 대한 모델 결과 찾기
                model_result = None
                for result in self.extraction_results[model]:
                    if result['msg_id'] == msg_id and result.get('success', False):
                        model_result = result['json_objects']
                        break
                
                if model_result is None:
                    logger.warning(f"{model} 모델 - 메시지 {msg_id} 결과 없음")
                    continue
                
                # 유사도 계산
                similarity_result = calculate_dictionary_similarity(ground_truth, model_result)
                
                evaluation_record = {
                    'msg_id': msg_id,
                    'model': model,
                    'similarity_result': similarity_result,
                    'ground_truth_avg_similarity': ground_truth_record['avg_similarity']
                }
                
                model_evaluations.append(evaluation_record)
            
            # 모델별 통계 계산
            if model_evaluations:
                overall_similarities = [eval_record['similarity_result']['overall_similarity'] for eval_record in model_evaluations]
                
                model_stats = {
                    'model': model,
                    'total_evaluated': len(model_evaluations),
                    'avg_overall_similarity': np.mean(overall_similarities),
                    'std_overall_similarity': np.std(overall_similarities),
                    'min_overall_similarity': np.min(overall_similarities),
                    'max_overall_similarity': np.max(overall_similarities),
                    'median_overall_similarity': np.median(overall_similarities),
                    'detailed_evaluations': model_evaluations
                }
                
                self.evaluation_results[model] = model_stats
                
                logger.info(f"{model} 평가 완료:")
                logger.info(f"  - 평가 메시지 수: {model_stats['total_evaluated']}")
                logger.info(f"  - 평균 유사도: {model_stats['avg_overall_similarity']:.4f}")
                logger.info(f"  - 표준편차: {model_stats['std_overall_similarity']:.4f}")
                logger.info(f"  - 최소/최대: {model_stats['min_overall_similarity']:.4f}/{model_stats['max_overall_similarity']:.4f}")
            else:
                logger.warning(f"{model} 모델에 대한 평가 데이터가 없습니다")
    
    def generate_performance_report(self):
        """성능 평가 리포트 생성"""
        logger.info("성능 평가 리포트 생성 중...")
        
        # 전체 평가 결과 저장
        evaluation_file = self.evaluation_dir / "model_evaluation_results.json"
        with open(evaluation_file, 'w', encoding='utf-8') as f:
            json.dump(self.evaluation_results, f, ensure_ascii=False, indent=2)
        
        # 요약 리포트 생성
        summary_report = {
            'evaluation_date': datetime.now().isoformat(),
            'ground_truth_count': len(self.ground_truth_data),
            'model_comparisons': {}
        }
        
        for model, stats in self.evaluation_results.items():
            summary_report['model_comparisons'][model] = {
                'total_evaluated': stats['total_evaluated'],
                'avg_overall_similarity': stats['avg_overall_similarity'],
                'std_overall_similarity': stats['std_overall_similarity'],
                'performance_grade': self._get_performance_grade(stats['avg_overall_similarity'])
            }
        
        # 성능 순위
        if len(self.evaluation_results) > 1:
            ranking = sorted(
                self.evaluation_results.items(),
                key=lambda x: x[1]['avg_overall_similarity'],
                reverse=True
            )
            summary_report['performance_ranking'] = [model for model, _ in ranking]
        
        # 요약 리포트 저장
        summary_file = self.evaluation_dir / "performance_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_report, f, ensure_ascii=False, indent=2)
        
        # 텍스트 리포트 생성
        self._generate_text_report(summary_report)
        
        logger.info(f"성능 평가 리포트 저장 완료: {self.evaluation_dir}")
        
        return summary_report
    
    def _get_performance_grade(self, similarity: float) -> str:
        """유사도에 따른 성능 등급 반환"""
        if similarity >= 0.9:
            return "A+ (Excellent)"
        elif similarity >= 0.8:
            return "A (Very Good)"
        elif similarity >= 0.7:
            return "B+ (Good)"
        elif similarity >= 0.6:
            return "B (Fair)"
        elif similarity >= 0.5:
            return "C+ (Below Average)"
        else:
            return "C (Poor)"
    
    def _generate_text_report(self, summary_report: Dict):
        """텍스트 형태의 리포트 생성"""
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("모델 성능 평가 리포트")
        report_lines.append("=" * 60)
        report_lines.append(f"평가 일시: {summary_report['evaluation_date']}")
        report_lines.append(f"정답 데이터셋 크기: {summary_report['ground_truth_count']}개 메시지")
        report_lines.append("")
        
        report_lines.append("평가 결과:")
        report_lines.append("-" * 40)
        
        for model, stats in summary_report['model_comparisons'].items():
            report_lines.append(f"📊 {model.upper()} 모델:")
            report_lines.append(f"   평가 메시지 수: {stats['total_evaluated']}")
            report_lines.append(f"   평균 유사도: {stats['avg_overall_similarity']:.4f}")
            report_lines.append(f"   표준편차: {stats['std_overall_similarity']:.4f}")
            report_lines.append(f"   성능 등급: {stats['performance_grade']}")
            report_lines.append("")
        
        if 'performance_ranking' in summary_report:
            report_lines.append("성능 순위:")
            report_lines.append("-" * 20)
            for i, model in enumerate(summary_report['performance_ranking'], 1):
                stats = summary_report['model_comparisons'][model]
                report_lines.append(f"{i}. {model.upper()}: {stats['avg_overall_similarity']:.4f}")
            report_lines.append("")
        
        # 상세 분석
        report_lines.append("상세 분석:")
        report_lines.append("-" * 20)
        
        for model, model_stats in self.evaluation_results.items():
            similarities = [eval_record['similarity_result'] for eval_record in model_stats['detailed_evaluations']]
            
            # 각 필드별 평균 유사도
            avg_title = np.mean([sim['title_similarity'] for sim in similarities])
            avg_purpose = np.mean([sim['purpose_similarity'] for sim in similarities])
            avg_product = np.mean([sim['product_similarity'] for sim in similarities])
            avg_channel = np.mean([sim['channel_similarity'] for sim in similarities])
            avg_pgm = np.mean([sim['pgm_similarity'] for sim in similarities])
            
            report_lines.append(f"{model.upper()} 모델 필드별 성능:")
            report_lines.append(f"  제목(title): {avg_title:.4f}")
            report_lines.append(f"  목적(purpose): {avg_purpose:.4f}")
            report_lines.append(f"  상품(product): {avg_product:.4f}")
            report_lines.append(f"  채널(channel): {avg_channel:.4f}")
            report_lines.append(f"  프로그램(pgm): {avg_pgm:.4f}")
            report_lines.append("")
        
        # 텍스트 리포트 저장
        text_report_file = self.evaluation_dir / "performance_report.txt"
        with open(text_report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        # 콘솔에도 출력
        print('\n'.join(report_lines))

def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="모델 성능 평가 실행")
    parser.add_argument('--results-dir', type=str, default='results', help='결과 디렉토리')
    parser.add_argument('--similarity-threshold', type=float, default=0.9, help='정답 생성 유사도 임계값')
    parser.add_argument('--min-message-length', type=int, default=300, help='정답용 메시지 최소 길이')
    
    args = parser.parse_args()
    
    logger.info("=== 모델 성능 평가 시작 ===")
    
    try:
        # 평가기 객체 생성
        evaluator = ModelPerformanceEvaluator(results_dir=args.results_dir)
        
        # 1. 추출 결과 로딩
        logger.info("1단계: 추출 결과 로딩")
        evaluator.load_extraction_results()
        
        # 2. 정답 데이터셋 생성
        logger.info("2단계: 정답 데이터셋 생성")
        evaluator.generate_ground_truth_dataset(
            similarity_threshold=args.similarity_threshold,
            min_message_length=args.min_message_length
        )
        
        # 3. 대상 모델 평가
        logger.info("3단계: 대상 모델 평가")
        evaluator.evaluate_target_models()
        
        # 4. 성능 리포트 생성
        logger.info("4단계: 성능 리포트 생성")
        evaluator.generate_performance_report()
        
        logger.info("=== 모델 성능 평가 완료 ===")
        
    except Exception as e:
        logger.error(f"평가 중 오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    main()
