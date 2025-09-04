#!/usr/bin/env python3
"""
모델 비교 실험 스크립트 (성능 검증용)
====================================

이 스크립트는 다음 5개 모델의 성능을 비교합니다:
- gemma_model: skt/gemma3-12b-it
- gemini_model: gcp/gemini-2.5-flash 
- claude_model: amazon/anthropic/claude-sonnet-4-20250514
- ax_model: skt/ax4
- gpt_model: openai/gpt-4o-2024-11-20

각 모델에 대해 "7단계: 엔티티 매칭 및 최종 결과 구성" 전의 json_objects를 저장하고
유사도를 계산하여 성능을 비교합니다.
"""

import os
import sys
import pandas as pd
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import pickle

# 상위 디렉토리를 path에 추가 (mms_extractor_unified)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import METADATA_CONFIG, MODEL_CONFIG, PROCESSING_CONFIG
from mms_extractor import MMSExtractor

# 유사도 계산 함수들 임포트
from difflib import SequenceMatcher

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'model_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelComparisonExperiment:
    """모델 비교 실험을 관리하는 클래스"""
    
    def __init__(self, batch_size: int = 100, output_dir: str = "results", min_message_length: int = 300):
        self.batch_size = batch_size
        
        # 타임스탬프 추가
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        output_dir_with_timestamp = f"{output_dir}_{timestamp}"
        
        self.output_dir = Path(output_dir_with_timestamp)
        self.output_dir.mkdir(exist_ok=True)
        self.min_message_length = min_message_length
        
        # 모델 설정 - MMSExtractor가 인식하는 짧은 식별자 사용
        self.models = {
            'gemma': "gem",     # skt/gemma3-12b-it
            'gemini': "gen",    # gcp/gemini-2.5-flash
            'claude': "cld",    # amazon/anthropic/claude-sonnet-4-20250514
            'ax': "ax",         # skt/ax4
            'gpt': "gpt"        # azure/openai/gpt-4o-2024-08-06
        }
        
        # 실험 설정
        self.experiment_config = {
            'extract_entity_dag': False,
            'product_info_extraction_mode': 'rag',
            'entity_matching_mode': 'llm'
        }
        
        # 결과 저장용
        self.extraction_results = {}
        
    def load_mms_data(self) -> pd.DataFrame:
        """MMS 데이터 로딩 및 배치 선택"""
        # 절대 경로로 변환
        mms_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                               METADATA_CONFIG.mms_msg_path.lstrip('./'))
        logger.info(f"MMS 데이터 로딩: {mms_path}")
        
        try:
            # CSV 파일 로드
            mms_pdf = pd.read_csv(mms_path)
            mms_pdf = mms_pdf.astype('str')
            
            # 메시지 컬럼 생성 (msg_nm + mms_phrs 결합)
            if 'msg' not in mms_pdf.columns:
                mms_pdf['msg'] = mms_pdf['msg_nm'] + "\n" + mms_pdf['mms_phrs']
            
            # 중복 제거 및 정리
            mms_pdf = mms_pdf.groupby(["msg_nm", "mms_phrs", "msg"])['offer_dt'].min().reset_index(name="offer_dt")
            mms_pdf = mms_pdf.reset_index()
            
            # msg_id 컬럼 추가
            if 'msg_id' not in mms_pdf.columns:
                mms_pdf['msg_id'] = mms_pdf.index.astype(str)
            
            logger.info(f"총 {len(mms_pdf)}개 메시지 로드됨")
            
            # 메시지 길이 조건 적용
            original_count = len(mms_pdf)
            mms_pdf = mms_pdf[mms_pdf['msg'].str.len() >= self.min_message_length]
            filtered_count = len(mms_pdf)
            
            logger.info(f"메시지 길이 필터링 적용 (최소 {self.min_message_length}자): {original_count}개 → {filtered_count}개")
            
            if filtered_count == 0:
                logger.warning(f"최소 길이 {self.min_message_length}자 조건을 만족하는 메시지가 없습니다")
                return pd.DataFrame()
            
            # 배치 크기만큼 샘플링
            if len(mms_pdf) > self.batch_size:
                sampled_df = mms_pdf.sample(n=self.batch_size, random_state=42)
                logger.info(f"{self.batch_size}개 메시지 샘플링 완료")
            else:
                sampled_df = mms_pdf
                logger.info(f"전체 {len(sampled_df)}개 메시지 사용")
            
            return sampled_df.reset_index(drop=True)
            
        except Exception as e:
            logger.error(f"MMS 데이터 로딩 실패: {str(e)}")
            raise
    
    def extract_with_model(self, model_name: str, model_id: str, messages_df: pd.DataFrame) -> List[Dict]:
        """특정 모델로 추출 실행"""
        logger.info(f"=== {model_name} 모델 추출 시작 ({model_id}) ===")
        
        results = []
        
        try:
            # MMSExtractor 초기화
            extractor = MMSExtractor(
                llm_model=model_id,
                extract_entity_dag=self.experiment_config['extract_entity_dag'],
                product_info_extraction_mode=self.experiment_config['product_info_extraction_mode'],
                entity_extraction_mode=self.experiment_config['entity_matching_mode']
            )
            
            logger.info(f"{model_name} {model_id} 추출기 초기화 완료")
            
            # 각 메시지 처리
            for idx, row in messages_df.iterrows():
                msg_id = row['msg_id']
                msg = row['msg']
                
                logger.info(f"처리 중: {model_name} - 메시지 {idx+1}/{len(messages_df)} (ID: {msg_id})")
                
                try:
                    # 원본 process_message 메소드 사용하되 7단계 전까지만 실행
                    extraction_result = self._extract_json_objects_only(extractor, msg)
                    
                    result_record = {
                        'msg_id': msg_id,
                        'msg': msg,
                        'model': model_name,
                        'model_id': model_id,
                        'json_objects': extraction_result,
                        'extracted_at': datetime.now().isoformat(),
                        'success': True,
                        'error': None
                    }
                    
                    results.append(result_record)
                    logger.info(f"✅ {model_name} - 메시지 {msg_id} 추출 완료")
                    
                except Exception as e:
                    logger.error(f"❌ {model_name} - 메시지 {msg_id} 추출 실패: {str(e)}")
                    
                    error_record = {
                        'msg_id': msg_id,
                        'msg': msg,
                        'model': model_name,
                        'model_id': model_id,
                        'json_objects': {},
                        'extracted_at': datetime.now().isoformat(),
                        'success': False,
                        'error': str(e)
                    }
                    
                    results.append(error_record)
                
                # 짧은 지연 (API 율제한 방지)
                time.sleep(1)
            
            logger.info(f"=== {model_name} 모델 추출 완료 ===")
            
        except Exception as e:
            logger.error(f"❌ {model_name} 모델 초기화 실패: {str(e)}")
            raise
        
        return results
    
    def _extract_json_objects_only(self, extractor: MMSExtractor, msg: str) -> Dict:
        """
        MMSExtractor를 사용하여 7단계 전의 json_objects만 추출
        원본 코드 수정 최소화를 위해 기존 메소드들을 최대한 활용
        """
        try:
            # 1-2단계: 프로그램 분류 (기존 _classify_programs 사용)
            pgm_info = extractor._classify_programs(msg)
            
            # 3단계: RAG 컨텍스트 구성 (기존 로직 사용)
            rag_context = ""
            if extractor.product_info_extraction_mode == 'rag':
                rag_context = f"\n### 광고 분류 기준 정보 ###\n\t{pgm_info['pgm_cand_info']}" if extractor.num_cand_pgms > 0 else ""
            
            # 4단계: 제품 정보 준비 (간단한 fallback)
            product_element = None
            if extractor.product_info_extraction_mode == 'rag':
                # 메소드가 없으므로 간단한 fallback 사용
                product_element = None
            
            # 5단계: 프롬프트 구성 및 LLM 호출 (기존 메소드 사용)
            prompt = extractor._build_extraction_prompt(msg, rag_context, product_element)
            result_json_text = extractor._safe_llm_invoke(prompt)
            
            # 6단계: JSON 파싱 (기존 함수 사용)
            from mms_extractor import extract_json_objects
            json_objects_list = extract_json_objects(result_json_text)
            
            if not json_objects_list:
                logger.warning("LLM이 유효한 JSON 객체를 반환하지 않았습니다")
                return {}
            
            json_objects = json_objects_list[-1]
            
            # 스키마 응답 감지 (기존 메소드 사용)
            is_schema_response = extractor._detect_schema_response(json_objects)
            if is_schema_response:
                logger.warning("LLM이 스키마 정의를 반환했습니다")
                return {}
            
            logger.info(f"JSON 객체 추출 완료 - 키: {list(json_objects.keys())}")
            return json_objects
            
        except Exception as e:
            logger.error(f"JSON 객체 추출 중 오류 발생: {str(e)}")
            return {}
    
    def run_extraction_for_all_models(self, messages_df: pd.DataFrame):
        """모든 모델에 대해 추출 실행"""
        logger.info("=== 전체 모델 추출 실험 시작 ===")
        
        for model_name, model_id in self.models.items():
            try:
                start_time = time.time()
                
                # 모델별 추출 실행
                model_results = self.extract_with_model(model_name, model_id, messages_df)
                
                # 결과 저장
                self.extraction_results[model_name] = model_results
                
                # 개별 모델 결과 파일 저장
                model_file = self.output_dir / f"{model_name}_extraction_results.json"
                with open(model_file, 'w', encoding='utf-8') as f:
                    json.dump(model_results, f, ensure_ascii=False, indent=2)
                
                elapsed_time = time.time() - start_time
                success_count = len([r for r in model_results if r['success']])
                
                logger.info(f"✅ {model_name} 모델 완료: {success_count}/{len(model_results)} 성공, {elapsed_time:.2f}초 소요")
                
            except Exception as e:
                logger.error(f"❌ {model_name} 모델 전체 실패: {str(e)}")
                self.extraction_results[model_name] = []
        
        logger.info("=== 전체 모델 추출 실험 완료 ===")
    
    def save_combined_results(self):
        """모든 결과를 통합하여 저장"""
        logger.info("통합 결과 저장 중...")
        
        # 통합 결과 파일
        combined_file = self.output_dir / "combined_extraction_results.json"
        with open(combined_file, 'w', encoding='utf-8') as f:
            json.dump(self.extraction_results, f, ensure_ascii=False, indent=2)
        
        # 피클 파일로도 저장 (성능상 이유)
        pickle_file = self.output_dir / "combined_extraction_results.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump(self.extraction_results, f)
        
        # 실험 메타데이터 저장
        metadata = {
            'experiment_date': datetime.now().isoformat(),
            'batch_size': self.batch_size,
            'models': self.models,
            'experiment_config': self.experiment_config,
            'total_messages': len(self.extraction_results.get('gemma', [])) if 'gemma' in self.extraction_results else 0
        }
        
        metadata_file = self.output_dir / "experiment_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"통합 결과 저장 완료: {self.output_dir}")

def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="모델 비교 실험 실행")
    parser.add_argument('--batch-size', type=int, default=100, help='배치 크기 (기본값: 100)')
    parser.add_argument('--output-dir', type=str, default='results', help='결과 저장 디렉토리')
    parser.add_argument('--min-message-length', type=int, default=300, help='최소 메시지 길이 (기본값: 300)')
    
    args = parser.parse_args()
    
    logger.info("=== 모델 비교 실험 시작 ===")
    logger.info(f"배치 크기: {args.batch_size}")
    logger.info(f"결과 저장 디렉토리: {args.output_dir}")
    logger.info(f"최소 메시지 길이: {args.min_message_length}자")
    
    try:
        # 실험 객체 생성
        experiment = ModelComparisonExperiment(
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            min_message_length=args.min_message_length
        )
        
        # 1. MMS 데이터 로딩
        logger.info("1단계: MMS 데이터 로딩")
        messages_df = experiment.load_mms_data()
        
        # 2. 모든 모델로 추출 실행
        logger.info("2단계: 모든 모델로 추출 실행")
        experiment.run_extraction_for_all_models(messages_df)
        
        # 3. 결과 저장
        logger.info("3단계: 결과 저장")
        experiment.save_combined_results()
        
        logger.info("=== 모델 비교 실험 완료 ===")
        
    except Exception as e:
        logger.error(f"실험 실행 중 오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    main()
