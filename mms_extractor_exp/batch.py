#!/usr/bin/env python3
"""
Batch Processing Script for MMS Message Extraction with Parallel Processing
==========================================================================

This script performs batch processing of MMS messages using the MMSExtractor with
support for parallel processing to improve performance. It reads messages from a CSV file,
processes them in batches using multiprocessing, and saves results.

Features:
- Batch processing with MMSExtractor
- Parallel processing support (multiprocessing)
- DAG extraction with parallel processing
- Performance monitoring and metrics
- Result storage with timestamps
- Automatic update of processing status
- Configurable worker count

Usage:
    # Basic batch processing (parallel)
    python batch.py --batch-size 10
    
    # Custom worker count
    python batch.py --batch-size 50 --max-workers 8 --output-file results_batch.csv
    
    # With DAG extraction (parallel processing of main + DAG)
    python batch.py --batch-size 20 --extract-entity-dag --max-workers 4
    
    # Sequential processing (disable multiprocessing)
    python batch.py --batch-size 10 --disable-multiprocessing
    
    # Full configuration
    python batch.py --batch-size 100 --max-workers 16 --extract-entity-dag \
                     --llm-model ax --entity-extraction-mode llm
"""

import os
import sys
import argparse
import pandas as pd
import json
from datetime import datetime
import logging
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.settings import METADATA_CONFIG
from mms_extractor import MMSExtractor, process_message_with_dag, process_messages_batch

# MongoDB 유틸리티 임포트 (선택적)
try:
    from mongodb_utils import save_to_mongodb, test_mongodb_connection
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    print("⚠️ MongoDB 유틸리티를 찾을 수 없습니다. --save-to-mongodb 옵션이 비활성화됩니다.")
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing
import time

# Configure logging - 통합 로그 파일 사용
from pathlib import Path

# 로그 디렉토리 생성
log_dir = Path(__file__).parent / 'logs'
log_dir.mkdir(exist_ok=True)

# 배치 처리 전용 로깅 설정
import logging.handlers

root_logger = logging.getLogger()
if not root_logger.handlers:
    # 배치 처리용 파일 핸들러 (회전 로그 - 20MB씩 최대 5개 파일, 장기 보존)
    batch_file_handler = logging.handlers.RotatingFileHandler(
        log_dir / 'batch_processing.log',
        maxBytes=20*1024*1024,  # 20MB (배치 로그는 많은 양)
        backupCount=5,          # 감사 목적으로 장기 보존
        encoding='utf-8'
    )
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    batch_file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logging.basicConfig(
        level=logging.INFO,
        handlers=[batch_file_handler, console_handler]
    )
logger = logging.getLogger(__name__)


def save_result_to_mongodb_if_enabled(message: str, result: dict, save_to_mongodb: bool, extractor_kwargs: dict = None, message_id: str = None):
    """MongoDB 저장이 활성화된 경우 결과를 저장하는 도우미 함수"""
    logger.debug(f"MongoDB 저장 함수 호출: save_to_mongodb={save_to_mongodb}, MONGODB_AVAILABLE={MONGODB_AVAILABLE}")
    
    if not save_to_mongodb:
        logger.debug("MongoDB 저장이 비활성화되어 있어 건너뜁니다.")
        return None
        
    if not MONGODB_AVAILABLE:
        logger.warning("MongoDB 저장이 요청되었지만 mongodb_utils를 찾을 수 없습니다.")
        return None
    
    try:
        logger.debug("MongoDB 저장 시작...")
        
        actual_prompts = result.get('prompts', {})
        
        # 디버깅: result 키들과 prompts 확인
        logger.debug(f"result 키들: {list(result.keys())}")
        logger.debug(f"actual_prompts: {actual_prompts}")
        logger.debug(f"actual_prompts 타입: {type(actual_prompts)}")
        
        # 프롬프트 정보 구성 (실제 사용된 프롬프트 우선 사용)
        prompts_data = {}
        
        if actual_prompts:
            logger.debug(f"프롬프트 정보 발견: {list(actual_prompts.keys())}")
            # result에 프롬프트 정보가 있는 경우 사용
            for key, prompt_content in actual_prompts.items():
                logger.debug(f"프롬프트 처리: {key}, 타입: {type(prompt_content)}")
                if isinstance(prompt_content, dict):
                    # 이미 구조화된 프롬프트 정보인 경우
                    prompts_data[key] = {
                        'title': prompt_content.get('title', f'{key.replace("_", " ").title()} (배치)'),
                        'description': prompt_content.get('description', f'배치 처리에서 사용된 {key} 프롬프트'),
                        'content': prompt_content.get('content', ''),
                        'length': prompt_content.get('length', len(prompt_content.get('content', '')))
                    }
                else:
                    # 문자열 프롬프트인 경우
                    prompts_data[key] = {
                        'title': f'{key.replace("_", " ").title()} (배치)',
                        'description': f'배치 처리에서 사용된 {key} 프롬프트',
                        'content': prompt_content if isinstance(prompt_content, str) else str(prompt_content),
                        'length': len(prompt_content) if isinstance(prompt_content, str) else len(str(prompt_content))
                    }
        else:
            logger.debug("프롬프트 정보가 없음 - 기본값 사용")
            # 프롬프트 정보가 없는 경우 기본값 사용 (배치 처리 특성 반영)
            prompts_data = {
                'batch_processing_info': {
                    'title': '배치 처리 정보',
                    'description': '배치 처리에서 사용된 설정 정보',
                    'content': f'배치 처리 모드로 실행됨. 설정: {extractor_kwargs or {}}',
                    'length': len(str(extractor_kwargs or {}))
                }
            }
        
        extraction_prompts = {
            'success': True,
            'prompts': prompts_data,
            'settings': extractor_kwargs or {}
        }

        # 추출 결과를 MongoDB 형식으로 구성
        extraction_result = {
            'success': not bool(result.get('error')),
            'result': result.get('extracted_result', {}),
            'metadata': {
                'processing_time_seconds': result.get('processing_time', 0),
                'processing_mode': 'batch',
                'model_used': extractor_kwargs.get('llm_model', 'unknown') if extractor_kwargs else 'unknown'
            }
        }

        raw_result = {
            'success': not bool(result.get('error')),
            'result': result.get('raw_result', {}),
            'metadata': {
                'processing_time_seconds': result.get('processing_time', 0),
                'processing_mode': 'single',
                'model_used': extractor_kwargs.get('llm_model', 'unknown') if extractor_kwargs else 'unknown'
            }
        }
        
        # MongoDB에 저장
        from mongodb_utils import save_to_mongodb as mongodb_save_to_mongodb
        saved_id = mongodb_save_to_mongodb(message, extraction_result, raw_result, extraction_prompts, 
                                         user_id="SKT1110566", message_id=message_id)
        
        if saved_id:
            logger.debug(f"MongoDB 저장 성공: {saved_id[:8]}...")
            return saved_id
        else:
            logger.warning("MongoDB 저장 실패")
            return None
            
    except Exception as e:
        logger.error(f"MongoDB 저장 중 오류 발생: {str(e)}")
        logger.error(f"오류 타입: {type(e)}")
        import traceback
        logger.error(f"스택 트레이스: {traceback.format_exc()}")
        return None


class BatchProcessor:
    """
    Batch processor for MMS message extraction
    """
    
    def __init__(self, result_file_path="./data/batch_results.csv", max_workers=None, enable_multiprocessing=True, save_to_mongodb=False):
        """
        Initialize batch processor
        
        Args:
            result_file_path: Path to store batch processing results
            max_workers: Maximum number of worker processes/threads (default: CPU count)
            enable_multiprocessing: Whether to use multiprocessing for batch processing
            save_to_mongodb: Whether to save results to MongoDB
        """
        self.result_file_path = result_file_path
        self.extractor = None
        self.mms_pdf = None
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.enable_multiprocessing = enable_multiprocessing
        self.save_to_mongodb = save_to_mongodb
        self.extract_entity_dag = False
        
    def initialize_extractor(self, **extractor_kwargs):
        """
        Initialize the MMS extractor with given parameters
        
        Args:
            **extractor_kwargs: Keyword arguments for MMSExtractor initialization
        """
        logger.info("Initializing MMS Extractor...")
        try:
            self.extract_entity_dag = extractor_kwargs.get('extract_entity_dag', False)
            self.extractor = MMSExtractor(**extractor_kwargs)
            logger.info(f"MMS Extractor initialized successfully (DAG 추출: {'ON' if self.extract_entity_dag else 'OFF'})")
        except Exception as e:
            logger.error(f"Failed to initialize MMS Extractor: {str(e)}")
            raise
    
    def load_mms_data(self):
        """
        Load MMS PDF dataframe and prepare it for processing
        """
        logger.info(f"Loading MMS data from: {METADATA_CONFIG.mms_msg_path}")
        
        try:
            # Load the CSV file
            self.mms_pdf = pd.read_csv(METADATA_CONFIG.mms_msg_path)
            self.mms_pdf = self.mms_pdf.astype('str')
            
            # Add msg_id column if it doesn't exist
            if 'msg_id' not in self.mms_pdf.columns:
                self.mms_pdf['msg_id'] = self.mms_pdf.index.astype(str)
            
            logger.info(f"Loaded {len(self.mms_pdf)} messages")
            
        except Exception as e:
            logger.error(f"Failed to load MMS data: {str(e)}")
            raise
    
    def get_processed_msg_ids(self):
        """
        Load processed message IDs from the result file
        
        Returns:
            set: Set of processed message IDs
        """
        processed_ids = set()
        
        if os.path.exists(self.result_file_path):
            try:
                results_df = pd.read_csv(self.result_file_path)
                if 'msg_id' in results_df.columns:
                    processed_ids = set(results_df['msg_id'].astype(str))
                    logger.info(f"Found {len(processed_ids)} previously processed messages")
                else:
                    logger.warning("No msg_id column found in results file")
            except Exception as e:
                logger.error(f"Failed to load processed message IDs: {str(e)}")
        else:
            logger.info("No existing results file found - all messages are unprocessed")
            
        return processed_ids
    
    def sample_unprocessed_messages(self, batch_size):
        """
        Sample M random unprocessed messages based on result file
        
        Args:
            batch_size: Number of messages to sample
            
        Returns:
            pandas.DataFrame: Sampled messages that haven't been processed yet
        """
        # Get processed message IDs from result file
        processed_ids = self.get_processed_msg_ids()
        
        # Filter out already processed messages
        unprocessed = self.mms_pdf[~self.mms_pdf['msg_id'].isin(processed_ids)].copy()
        
        if len(unprocessed) == 0:
            logger.warning("No unprocessed messages found")
            return pd.DataFrame()
        
        # Sample random messages
        sample_size = min(batch_size, len(unprocessed))
        sampled_messages = unprocessed.sample(n=sample_size, random_state=None)
        
        logger.info(f"Total messages: {len(self.mms_pdf)}")
        logger.info(f"Previously processed: {len(processed_ids)}")
        logger.info(f"Unprocessed messages: {len(unprocessed)}")
        logger.info(f"Sampled {len(sampled_messages)} messages for processing")
        return sampled_messages
    
    def process_messages(self, sampled_messages):
        """
        Process sampled messages using MMSExtractor with parallel processing support
        
        Args:
            sampled_messages: DataFrame with messages to process
            
        Returns:
            list: List of processing results
        """
        if self.extractor is None:
            raise ValueError("Extractor not initialized. Call initialize_extractor() first.")
        
        processing_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        messages_list = []
        
        # Convert DataFrame to list of messages
        for idx, row in sampled_messages.iterrows():
            msg = row.get('msg', '')
            msg_id = row.get('msg_id', str(idx))
            messages_list.append({'msg': msg, 'msg_id': msg_id})
        
        if self.enable_multiprocessing and len(messages_list) > 1:
            logger.info(f"🚀 병렬 처리 모드로 {len(messages_list)}개 메시지 처리 시작 (워커: {self.max_workers}개)")
            results = self._process_messages_parallel(messages_list, processing_time)
        else:
            logger.info(f"⚡ 순차 처리 모드로 {len(messages_list)}개 메시지 처리 시작")
            results = self._process_messages_sequential(messages_list, processing_time)
        
        return results
    
    def _process_messages_parallel(self, messages_list, processing_time):
        """
        Process messages in parallel using multiprocessing
        
        Args:
            messages_list: List of message dictionaries
            processing_time: Processing timestamp
            
        Returns:
            list: List of processing results
        """
        start_time = time.time()
        
        # 메시지 문자열만 추출
        message_texts = [msg['msg'] for msg in messages_list]
        
        try:
            # process_messages_batch 함수를 사용하여 병렬 처리
            batch_result = process_messages_batch(
                self.extractor,
                message_texts,
                extract_dag=self.extract_entity_dag,
                max_workers=self.max_workers
            )
            
            results = []
            for i, msg_info in enumerate(messages_list):
                msg_id = msg_info['msg_id']
                msg = msg_info['msg']
                
                if i < len(batch_result):
                    extraction_result = batch_result[i]

                    print("=" * 50 + " extraction_result " + "=" * 50)
                    print(extraction_result)
                    
                    # Create result record
                    result_record = {
                        'msg_id': msg_id,
                        'msg': msg,
                        'extraction_result': json.dumps(extraction_result, ensure_ascii=False),
                        'processed_at': processing_time,
                        'title': extraction_result.get('title', ''),
                        'purpose': json.dumps(extraction_result.get('purpose', []), ensure_ascii=False),
                        'product_count': len(extraction_result.get('product', [])),
                        'channel_count': len(extraction_result.get('channel', [])),
                        'pgm': json.dumps(extraction_result.get('pgm', []), ensure_ascii=False)
                    }
                    
                    # MongoDB 저장 (배치 병렬 처리)
                    if self.save_to_mongodb:
                        # LLM 모델 이름을 문자열로 변환
                        llm_model_name = getattr(self.extractor, 'llm_model', 'unknown')
                        if hasattr(llm_model_name, 'model_name'):
                            llm_model_name = llm_model_name.model_name
                        elif hasattr(llm_model_name, '__class__'):
                            llm_model_name = llm_model_name.__class__.__name__
                        
                        extractor_kwargs = {
                            'llm_model': str(llm_model_name),
                            'offer_info_data_src': getattr(self.extractor, 'offer_info_data_src', 'unknown'),
                            'entity_extraction_mode': getattr(self.extractor, 'entity_extraction_mode', 'unknown')
                        }
                        save_result_to_mongodb_if_enabled(msg, extraction_result, self.save_to_mongodb, extractor_kwargs, message_id=msg_id)
                    
                    # DAG 추출 결과 검증 및 로깅
                    if self.extract_entity_dag and 'entity_dag' in extraction_result:
                        dag_items = extraction_result['entity_dag']
                        if dag_items and len(dag_items) > 0:
                            logger.info(f"✅ 메시지 {msg_id} DAG 추출 성공 - {len(dag_items)}개 관계")
                        else:
                            logger.warning(f"⚠️ 메시지 {msg_id} DAG 추출 요청되었으나 결과가 비어있음")
                    
                    results.append(result_record)
                    logger.info(f"✅ 메시지 {msg_id} 병렬 처리 완료")
                else:
                    # 처리 실패한 경우
                    error_record = {
                        'msg_id': msg_id,
                        'msg': msg,
                        'extraction_result': json.dumps({'error': 'Processing failed'}, ensure_ascii=False),
                        'processed_at': processing_time,
                        'title': '',
                        'purpose': '[]',
                        'product_count': 0,
                        'channel_count': 0,
                        'pgm': '[]'
                    }
                    results.append(error_record)
                    logger.error(f"❌ 메시지 {msg_id} 병렬 처리 실패")
            
            elapsed_time = time.time() - start_time
            logger.info(f"🎯 병렬 처리 완료: {len(results)}개 메시지, {elapsed_time:.2f}초 소요")
            return results
            
        except Exception as e:
            logger.error(f"병렬 처리 실패, 순차 처리로 전환: {str(e)}")
            return self._process_messages_sequential(messages_list, processing_time)
    
    def _process_messages_sequential(self, messages_list, processing_time):
        """
        Process messages sequentially (fallback method)
        
        Args:
            messages_list: List of message dictionaries
            processing_time: Processing timestamp
            
        Returns:
            list: List of processing results
        """
        results = []
        start_time = time.time()
        
        for i, msg_info in enumerate(messages_list, 1):
            msg = msg_info['msg']
            msg_id = msg_info['msg_id']
            
            try:
                logger.info(f"처리 중 ({i}/{len(messages_list)}): {msg_id} - {msg[:50]}...")
                
                # DAG 추출이 활성화된 경우 병렬로 처리
                if self.extract_entity_dag:
                    extraction_result = process_message_with_dag(
                        self.extractor, 
                        msg, 
                        extract_dag=True
                    )
                else:
                    extraction_result = self.extractor.process_message(msg)
                
                # Create result record
                result_record = {
                    'msg_id': msg_id,
                    'msg': msg,
                    'extraction_result': json.dumps(extraction_result, ensure_ascii=False),
                    'processed_at': processing_time,
                    'title': extraction_result.get('title', ''),
                    'purpose': json.dumps(extraction_result.get('purpose', []), ensure_ascii=False),
                    'product_count': len(extraction_result.get('product', [])),
                    'channel_count': len(extraction_result.get('channel', [])),
                    'pgm': json.dumps(extraction_result.get('pgm', []), ensure_ascii=False)
                }
                
                # MongoDB 저장 (배치 순차 처리)
                if self.save_to_mongodb:
                    # LLM 모델 이름을 문자열로 변환
                    llm_model_name = getattr(self.extractor, 'llm_model', 'unknown')
                    if hasattr(llm_model_name, 'model_name'):
                        llm_model_name = llm_model_name.model_name
                    elif hasattr(llm_model_name, '__class__'):
                        llm_model_name = llm_model_name.__class__.__name__
                    
                    extractor_kwargs = {
                        'llm_model': str(llm_model_name),
                        'offer_info_data_src': getattr(self.extractor, 'offer_info_data_src', 'unknown'),
                        'entity_extraction_mode': getattr(self.extractor, 'entity_extraction_mode', 'unknown')
                    }
                    save_result_to_mongodb_if_enabled(msg, extraction_result, self.save_to_mongodb, extractor_kwargs, message_id=msg_id)
                
                results.append(result_record)
                
                # DAG 추출 결과 검증 및 로깅
                if self.extract_entity_dag and 'entity_dag' in extraction_result:
                    dag_items = extraction_result['entity_dag']
                    if dag_items and len(dag_items) > 0:
                        logger.info(f"✅ 메시지 {msg_id} DAG 추출 성공 - {len(dag_items)}개 관계")
                    else:
                        logger.warning(f"⚠️ 메시지 {msg_id} DAG 추출 요청되었으나 결과가 비어있음")
                
                logger.info(f"✅ 메시지 {msg_id} 순차 처리 완료")
                
            except Exception as e:
                logger.error(f"❌ 메시지 {msg_id} 처리 실패: {str(e)}")
                # Add error record
                error_record = {
                    'msg_id': msg_id,
                    'msg': msg,
                    'extraction_result': json.dumps({'error': str(e)}, ensure_ascii=False),
                    'processed_at': processing_time,
                    'title': '',
                    'purpose': '[]',
                    'product_count': 0,
                    'channel_count': 0,
                    'pgm': '[]'
                }
                results.append(error_record)
        
        elapsed_time = time.time() - start_time
        logger.info(f"⚡ 순차 처리 완료: {len(results)}개 메시지, {elapsed_time:.2f}초 소요")
        return results
    
    def save_results(self, results):
        """
        Save processing results to CSV file (append if exists, create if new)
        
        Args:
            results: List of result dictionaries
            
        Returns:
            str: Path to the results file
        """
        if not results:
            logger.warning("No results to save")
            return self.result_file_path
        
        results_df = pd.DataFrame(results)
        
        # Check if result file exists
        if os.path.exists(self.result_file_path):
            logger.info(f"Appending {len(results_df)} results to existing file: {self.result_file_path}")
            # Load existing results and append
            existing_df = pd.read_csv(self.result_file_path)
            combined_df = pd.concat([existing_df, results_df], ignore_index=True)
            combined_df.to_csv(self.result_file_path, index=False)
        else:
            logger.info(f"Creating new results file: {self.result_file_path}")
            results_df.to_csv(self.result_file_path, index=False)
        
        logger.info(f"Results saved to: {self.result_file_path}")
        return self.result_file_path
    
    def log_processing_summary(self, processed_msg_ids):
        """
        Log processing summary for processed message IDs
        
        Args:
            processed_msg_ids: List of message IDs that were processed
        """
        if processed_msg_ids:
            logger.info(f"Successfully processed {len(processed_msg_ids)} messages")
            logger.info(f"Processed message IDs: {processed_msg_ids[:10]}{'...' if len(processed_msg_ids) > 10 else ''}")
        else:
            logger.warning("No messages were successfully processed")
    
    def run_batch(self, batch_size, **extractor_kwargs):
        """
        Run complete batch processing pipeline with performance monitoring
        
        Args:
            batch_size: Number of messages to process
            **extractor_kwargs: Keyword arguments for MMSExtractor
            
        Returns:
            dict: Processing summary with performance metrics
        """
        overall_start_time = time.time()
        
        try:
            # Initialize components
            self.load_mms_data()
            self.initialize_extractor(**extractor_kwargs)
            
            # Sample and process messages
            sampled_messages = self.sample_unprocessed_messages(batch_size)
            
            if sampled_messages.empty:
                return {
                    'status': 'completed',
                    'processed_count': 0,
                    'message': 'No unprocessed messages found',
                    'processing_mode': 'N/A',
                    'max_workers': self.max_workers,
                    'dag_extraction': self.extract_entity_dag
                }
            
            # Process messages with timing
            processing_start_time = time.time()
            results = self.process_messages(sampled_messages)
            processing_time = time.time() - processing_start_time
            
            # Save results and log summary
            self.save_results(results)
            processed_msg_ids = [r['msg_id'] for r in results if 'error' not in r.get('extraction_result', '')]
            self.log_processing_summary(processed_msg_ids)
            
            total_time = time.time() - overall_start_time
            processing_mode = "병렬 처리" if (self.enable_multiprocessing and len(results) > 1) else "순차 처리"
            
            # Performance metrics
            avg_time_per_message = processing_time / len(results) if results else 0
            throughput = len(results) / processing_time if processing_time > 0 else 0
            
            logger.info("="*50)
            logger.info("🎯 배치 처리 성능 요약")
            logger.info("="*50)
            logger.info(f"처리 모드: {processing_mode}")
            logger.info(f"워커 수: {self.max_workers}")
            logger.info(f"DAG 추출: {'ON' if self.extract_entity_dag else 'OFF'}")
            logger.info(f"총 처리 시간: {total_time:.2f}초")
            logger.info(f"메시지 처리 시간: {processing_time:.2f}초")
            logger.info(f"메시지당 평균 시간: {avg_time_per_message:.2f}초")
            logger.info(f"처리량: {throughput:.2f} 메시지/초")
            logger.info("="*50)
            
            return {
                'status': 'completed',
                'processed_count': len(results),
                'successful_count': len(processed_msg_ids),
                'failed_count': len(results) - len(processed_msg_ids),
                'results_file': self.result_file_path,
                'processing_mode': processing_mode,
                'max_workers': self.max_workers,
                'dag_extraction': self.extract_entity_dag,
                'total_time_seconds': round(total_time, 2),
                'processing_time_seconds': round(processing_time, 2),
                'avg_time_per_message': round(avg_time_per_message, 2),
                'throughput_messages_per_second': round(throughput, 2)
            }
            
        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e),
                'processed_count': 0,
                'processing_mode': 'N/A',
                'max_workers': self.max_workers,
                'dag_extraction': self.extract_entity_dag
            }


def main():
    """
    Main function for command-line execution
    """
    parser = argparse.ArgumentParser(description='Batch MMS Message Processing')
    
    # Batch processing arguments
    parser.add_argument('--batch-size', '-b', type=int, default=10,
                       help='Number of messages to process (default: 10)')
    parser.add_argument('--output-file', '-o', type=str, default='./data/batch_results.csv',
                       help='Output CSV file for results (default: batch_results.csv)')
    
    # Parallel processing arguments
    parser.add_argument('--max-workers', '-w', type=int, default=None,
                       help='Maximum number of worker processes (default: CPU count)')
    parser.add_argument('--disable-multiprocessing', action='store_true', default=False,
                       help='Disable multiprocessing and use sequential processing')
    
    # MMSExtractor arguments
    parser.add_argument('--offer-data-source', choices=['local', 'db'], default='local',
                       help='Data source for offer information (default: local)')
    parser.add_argument('--product-info-extraction-mode', choices=['nlp', 'llm', 'rag'], default='llm',
                       help='Product information extraction mode (default: llm)')
    parser.add_argument('--entity-extraction-mode', choices=['logic', 'llm'], default='llm',
                       help='Entity extraction mode (default: llm)')
    parser.add_argument('--llm-model', choices=['gem', 'ax', 'cld', 'gen', 'gpt'], default='ax',
                       help='LLM model to use (default: ax)')
    parser.add_argument('--extract-entity-dag', action='store_true', default=False, 
                       help='엔티티 DAG 추출 활성화 - 메시지에서 엔티티 간 관계를 그래프로 추출하고 시각화 (default: False)')
    
    # MongoDB arguments
    parser.add_argument('--save-to-mongodb', action='store_true', default=False,
                       help='추출 결과를 MongoDB에 저장 (mongodb_utils.py 필요)')
    parser.add_argument('--test-mongodb', action='store_true', default=False,
                       help='MongoDB 연결 테스트만 수행하고 종료')

    args = parser.parse_args()
    
    # MongoDB 연결 테스트만 수행하는 경우
    if args.test_mongodb:
        if not MONGODB_AVAILABLE:
            print("❌ MongoDB 유틸리티를 찾을 수 없습니다.")
            print("mongodb_utils.py 파일과 pymongo 패키지를 확인하세요.")
            sys.exit(1)
        
        print("🔌 MongoDB 연결 테스트 중...")
        if test_mongodb_connection():
            print("✅ MongoDB 연결 성공!")
            sys.exit(0)
        else:
            print("❌ MongoDB 연결 실패!")
            print("MongoDB 서버가 실행 중인지 확인하세요.")
            sys.exit(1)
    
    # 추출기 설정 준비
    # extract_entity_dag: True인 경우 각 메시지마다 DAG 추출 및 이미지 생성 수행
    extractor_kwargs = {
        'offer_info_data_src': args.offer_data_source,
        'product_info_extraction_mode': args.product_info_extraction_mode,
        'entity_extraction_mode': args.entity_extraction_mode,
        'llm_model': args.llm_model,
        'extract_entity_dag': args.extract_entity_dag  # DAG 추출 여부
    }
    
    # 병렬 처리 설정
    max_workers = args.max_workers or multiprocessing.cpu_count()
    enable_multiprocessing = not args.disable_multiprocessing
    
    logger.info("="*50)
    logger.info("🚀 Starting Batch MMS Processing")
    logger.info("="*50)
    logger.info(f"배치 크기: {args.batch_size}")
    logger.info(f"출력 파일: {args.output_file}")
    logger.info(f"병렬 처리: {'ON' if enable_multiprocessing else 'OFF'}")
    logger.info(f"최대 워커 수: {max_workers}")
    logger.info(f"추출기 설정: {extractor_kwargs}")
    if args.extract_entity_dag:
        logger.info("🎯 DAG 추출 모드 활성화됨")
    if args.save_to_mongodb:
        logger.info("📄 MongoDB 저장 모드 활성화됨")
    logger.info("="*50)
    
    # Run batch processing
    processor = BatchProcessor(
        result_file_path=args.output_file,
        max_workers=max_workers,
        enable_multiprocessing=enable_multiprocessing,
        save_to_mongodb=args.save_to_mongodb
    )
    summary = processor.run_batch(args.batch_size, **extractor_kwargs)
    
    # Print summary
    logger.info("="*50)
    logger.info("Batch Processing Summary")
    logger.info("="*50)
    for key, value in summary.items():
        logger.info(f"{key}: {value}")
    logger.info("="*50)
    
    return summary


if __name__ == '__main__':
    main()