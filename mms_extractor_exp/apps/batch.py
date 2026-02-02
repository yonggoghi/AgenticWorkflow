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
# Add parent directory to path to allow imports from core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import METADATA_CONFIG
from core.mms_extractor import MMSExtractor, process_message_worker, process_messages_batch, save_result_to_mongodb_if_enabled

# MongoDB 유틸리티는 필요할 때 동적으로 임포트
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
# from utils.mongodb_utils import save_to_mongodb # (필요시 사용)
import multiprocessing
import time

# Configure logging - 통합 로그 파일 사용
from pathlib import Path

# 로그 디렉토리 생성
log_dir = Path(__file__).parent.parent / 'logs'
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


# save_result_to_mongodb_if_enabled 함수는 mms_extractor.py에서 임포트됨


class BatchProcessor:
    """
    Batch processor for MMS message extraction
    """
    
    def __init__(self, result_file_path="./results/batch_results.csv", max_workers=None, enable_multiprocessing=True, save_to_mongodb=False, save_results_enabled=False):
        """
        Initialize batch processor
        
        Args:
            result_file_path: Path to store batch processing results
            max_workers: Maximum number of worker processes/threads (default: CPU count)
            enable_multiprocessing: Whether to use multiprocessing for batch processing
            save_to_mongodb: Whether to save results to MongoDB
            save_results_enabled: Whether to save results to CSV file
        """
        self.result_file_path = result_file_path
        self.extractor = None
        self.mms_pdf = None
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.enable_multiprocessing = enable_multiprocessing
        self.save_to_mongodb = save_to_mongodb
        self.save_results_enabled = save_results_enabled
        self.extract_entity_dag = True
        
    def initialize_extractor(self, **extractor_kwargs):
        """
        Initialize the MMS extractor with given parameters
        
        Args:
            **extractor_kwargs: Keyword arguments for MMSExtractor initialization
        """
        logger.info("Initializing MMS Extractor...")
        try:
            self.extract_entity_dag = extractor_kwargs.get('extract_entity_dag', True)
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
            
            # Handle different column name formats
            # If 'mms_phrs' exists but 'msg' doesn't, rename it
            if 'mms_phrs' in self.mms_pdf.columns and 'msg' not in self.mms_pdf.columns:
                self.mms_pdf['msg'] = self.mms_pdf['mms_phrs']
                logger.info("Renamed 'mms_phrs' column to 'msg'")
            elif 'msg' not in self.mms_pdf.columns:
                raise ValueError("CSV file must have either 'msg' or 'mms_phrs' column")
            
            # Filter out rows with empty or NaN messages before converting to string
            original_count = len(self.mms_pdf)
            self.mms_pdf = self.mms_pdf[self.mms_pdf['msg'].notna() & (self.mms_pdf['msg'].astype(str).str.strip() != '')]
            filtered_count = original_count - len(self.mms_pdf)
            if filtered_count > 0:
                logger.info(f"Filtered out {filtered_count} empty messages")
            
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
            # message_id: DataFrame에 message_id 컬럼이 있으면 사용, 없으면 msg_id 사용, 둘 다 없으면 idx 사용
            # row는 Series이므로 .get() 사용, 없으면 None 반환
            message_id = row.get('message_id')
            if message_id is None or pd.isna(message_id):
                message_id = row.get('msg_id')
            if message_id is None or pd.isna(message_id):
                message_id = str(idx)
            else:
                message_id = str(message_id)
            
            # Skip empty messages (safety check)
            if not msg or msg.strip() == '' or msg == 'nan':
                logger.warning(f"Skipping empty message with ID: {message_id}")
                continue
                
            messages_list.append({'msg': msg, 'message_id': message_id})
        
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
                messages_list,
                extract_dag=self.extract_entity_dag,
                max_workers=self.max_workers
            )
            
            results = []
            for i, msg_info in enumerate(messages_list):
                message_id = msg_info['message_id']
                msg = msg_info['msg']
                
                if i < len(batch_result):
                    extraction_result = batch_result[i]

                    # 디버깅용 출력 (필요시에만 활성화)
                    # print("=" * 50 + " extraction_result " + "=" * 50)
                    # print(f"Type: {type(extraction_result)}")
                    # print(f"Content: {extraction_result}")
                    # print("=" * 50)
                    
                    # Create result record
                    result_record = {
                        'message_id': message_id,
                        'message': msg,
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
                        
                        # MongoDB 저장을 위한 args 딕셔너리 구성
                        args_data = {
                            'save_to_mongodb': self.save_to_mongodb,
                            'llm_model': str(llm_model_name),
                            'offer_data_source': getattr(self.extractor, 'offer_info_data_src', 'unknown'),
                            'product_info_extraction_mode': getattr(self.extractor, 'product_info_extraction_mode', 'unknown'),
                            'entity_matching_mode': getattr(self.extractor, 'entity_extraction_mode', 'unknown'),
                            'extract_entity_dag': getattr(self.extractor, 'extract_entity_dag', False),
                            'user_id': 'BATCH_USER',
                            'processing_mode': 'batch'
                        }
                        save_result_to_mongodb_if_enabled(msg, extraction_result, args_data, self.extractor)
                    
                    # DAG 추출 결과 검증 및 로깅
                    if self.extract_entity_dag and 'entity_dag' in extraction_result:
                        dag_items = extraction_result['entity_dag']
                        if dag_items and len(dag_items) > 0:
                            logger.info(f"✅ 메시지 {message_id} DAG 추출 성공 - {len(dag_items)}개 관계")
                        else:
                            logger.warning(f"⚠️ 메시지 {message_id} DAG 추출 요청되었으나 결과가 비어있음")
                    
                    results.append(result_record)
                    
                    # 성공/실패 여부 확인 및 로깅
                    is_error = self._is_error_result(result_record['extraction_result'])
                    logger.debug(f"메시지 {message_id} 에러 판단 결과: {is_error}")
                    
                    if is_error:
                        logger.error(f"❌ 메시지 {message_id} 처리 실패")
                    else:
                        logger.info(f"✅ 메시지 {message_id} 처리 성공")
                else:
                    # 처리 실패한 경우 (배치 결과에 해당 메시지가 없는 경우)
                    error_record = {
                        'message_id': message_id,
                        'message': msg,
                        'extraction_result': json.dumps({'error': 'Processing failed - no result returned'}, ensure_ascii=False),
                        'processed_at': processing_time,
                        'title': '',
                        'purpose': '[]',
                        'product_count': 0,
                        'channel_count': 0,
                        'pgm': '[]'
                    }
                    results.append(error_record)
                    logger.error(f"❌ 메시지 {message_id} 병렬 처리 실패 - 배치 결과 없음")
            
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
            message_id = msg_info['message_id']
            
            try:
                logger.info(f"처리 중 ({i}/{len(messages_list)}): {message_id} - {msg[:50]}...")
                
                # DAG 추출이 활성화된 경우 병렬로 처리
                if self.extract_entity_dag:
                    extraction_result = process_message_worker(
                        self.extractor, 
                        msg, 
                        extract_dag=True,
                        message_id=message_id
                    )
                else:
                    extraction_result = self.extractor.process_message(msg, message_id=message_id)
                
                # Create result record
                result_record = {
                    'message_id': message_id,
                    'message': msg,
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
                    
                    # MongoDB 저장을 위한 args 딕셔너리 구성
                    args_data = {
                        'save_to_mongodb': self.save_to_mongodb,
                        'llm_model': str(llm_model_name),
                        'offer_data_source': getattr(self.extractor, 'offer_info_data_src', 'unknown'),
                        'product_info_extraction_mode': getattr(self.extractor, 'product_info_extraction_mode', 'unknown'),
                        'entity_matching_mode': getattr(self.extractor, 'entity_extraction_mode', 'unknown'),
                        'extract_entity_dag': getattr(self.extractor, 'extract_entity_dag', True),
                        'user_id': 'BATCH_USER',
                        'processing_mode': 'batch'
                    }
                    save_result_to_mongodb_if_enabled(msg, extraction_result, args_data, self.extractor)
                
                results.append(result_record)
                
                # DAG 추출 결과 검증 및 로깅
                if self.extract_entity_dag and 'entity_dag' in extraction_result:
                    dag_items = extraction_result['entity_dag']
                    if dag_items and len(dag_items) > 0:
                        logger.info(f"✅ 메시지 {message_id} DAG 추출 성공 - {len(dag_items)}개 관계")
                    else:
                        logger.warning(f"⚠️ 메시지 {message_id} DAG 추출 요청되었으나 결과가 비어있음")
                
                # 성공/실패 여부 확인 및 로깅
                is_error = self._is_error_result(result_record['extraction_result'])
                logger.debug(f"메시지 {message_id} 에러 판단 결과: {is_error}")
                
                if is_error:
                    logger.error(f"❌ 메시지 {message_id} 처리 실패")
                else:
                    logger.info(f"✅ 메시지 {message_id} 처리 성공")
                
            except Exception as e:
                logger.error(f"❌ 메시지 {message_id} 처리 실패: {str(e)}")
                # Add error record
                error_record = {
                    'message_id': message_id,
                    'message': msg,
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
        
        # results 디렉토리 생성
        result_dir = os.path.dirname(self.result_file_path)
        if result_dir:
            os.makedirs(result_dir, exist_ok=True)
        
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
    
    def _is_error_result(self, extraction_result_str):
        """
        Check if extraction result contains an error
        
        Args:
            extraction_result_str: JSON string of extraction result
            
        Returns:
            bool: True if result contains error, False otherwise
        """
        try:
            if isinstance(extraction_result_str, str):
                result_dict = json.loads(extraction_result_str)
                # process_messages_batch는 error 키가 있어도 실제 에러가 아닐 수 있음
                # error 키가 있고 실제 에러 메시지가 있는 경우만 실패로 간주
                if 'error' in result_dict:
                    error_msg = result_dict['error']
                    # 빈 문자열이거나 None인 경우는 성공으로 간주
                    if error_msg and str(error_msg).strip():
                        logger.debug(f"실제 에러 발견: {error_msg}")
                        return True
                    else:
                        logger.debug(f"error 키는 있지만 빈 값: '{error_msg}' - 성공으로 간주")
                        return False
                return False
            elif isinstance(extraction_result_str, dict):
                if 'error' in extraction_result_str:
                    error_msg = extraction_result_str['error']
                    if error_msg and str(error_msg).strip():
                        return True
                    else:
                        return False
                return False
            else:
                logger.debug(f"알 수 없는 타입: {type(extraction_result_str)}")
                return False
        except (json.JSONDecodeError, TypeError) as e:
            logger.debug(f"JSON 파싱 실패: {e}, 원본: {extraction_result_str}")
            return False
    
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
            
            # Save results and log summary (옵션이 활성화된 경우만)
            if self.save_results_enabled:
                self.save_results(results)
            else:
                logger.info("💾 배치 결과 CSV 파일 저장 생략 (--save-results 옵션으로 활성화 가능)")
            
            processed_msg_ids = [r['message_id'] for r in results if not self._is_error_result(r.get('extraction_result', ''))]
            self.log_processing_summary(processed_msg_ids)
            
            total_time = time.time() - overall_start_time
            processing_mode = "병렬 처리" if (self.enable_multiprocessing and len(results) > 1) else "순차 처리"
            
            # Performance metrics
            avg_time_per_message = processing_time / len(results) if results else 0
            throughput = len(results) / processing_time if processing_time > 0 else 0
            
            success_rate = len(processed_msg_ids) / len(results) * 100 if results else 0
            
            logger.info("="*50)
            logger.info("🎯 배치 처리 성능 요약")
            logger.info("="*50)
            logger.info(f"처리 모드: {processing_mode}")
            logger.info(f"워커 수: {self.max_workers}")
            logger.info(f"DAG 추출: {'ON' if self.extract_entity_dag else 'OFF'}")
            logger.info(f"총 처리 메시지: {len(results)}개")
            logger.info(f"성공: {len(processed_msg_ids)}개")
            logger.info(f"실패: {len(results) - len(processed_msg_ids)}개")
            logger.info(f"성공률: {success_rate:.2f}%")
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
                'throughput_messages_per_second': round(throughput, 2),
                'success_rate': round(len(processed_msg_ids) / len(results) * 100, 2) if results else 0
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
    parser.add_argument('--output-file', '-o', type=str, default='./results/batch_results.csv',
                       help='Output CSV file for results (default: results/batch_results.csv)')
    parser.add_argument('--save-results', action='store_true', default=False,
                       help='배치 처리 결과를 CSV 파일로 저장 (results/ 디렉토리에 저장)')
    
    # Parallel processing arguments
    parser.add_argument('--max-workers', '-w', type=int, default=None,
                       help='Maximum number of worker processes (default: CPU count)')
    parser.add_argument('--disable-multiprocessing', action='store_true', default=False,
                       help='Disable multiprocessing and use sequential processing')
    
    # MMSExtractor arguments
    parser.add_argument('--offer-data-source', choices=['local', 'db'], default='db',
                       help='데이터 소스 (local: CSV 파일, db: 데이터베이스)')
    parser.add_argument('--product-info-extraction-mode', choices=['nlp', 'llm', 'rag'], default='llm',
                       help='Product information extraction mode (default: llm)')
    parser.add_argument('--entity-extraction-mode', choices=['logic', 'llm'], default='llm',
                       help='Entity extraction mode (default: llm)')
    parser.add_argument('--llm-model', choices=['gem', 'ax', 'cld', 'gen', 'gpt'], default='ax',
                       help='메인 프롬프트에 사용할 LLM 모델 (default: ax)')
    parser.add_argument('--entity-llm-model', choices=['gem', 'ax', 'cld', 'gen', 'gpt'], default='ax',
                       help='엔티티 추출에 사용할 LLM 모델 (default: ax)')
    parser.add_argument('--entity-extraction-context-mode', choices=['dag', 'pairing', 'none', 'ont'], default='dag',
                       help='엔티티 추출 컨텍스트 모드 (dag: DAG 컨텍스트, pairing: PAIRING 컨텍스트, none: 컨텍스트 없음, ont: 온톨로지 기반 추출, default: dag)')
    parser.add_argument('--extract-entity-dag', action='store_true', default=False, 
                       help='엔티티 DAG 추출 활성화 - 메시지에서 엔티티 간 관계를 그래프로 추출하고 시각화 (default: False)')
    
    # Logging arguments
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO',
                       help='로그 레벨 설정 (DEBUG: 상세, INFO: 일반, WARNING: 경고, ERROR: 오류만)')
    
    # MongoDB arguments
    parser.add_argument('--save-to-mongodb', action='store_true', default=True,
                       help='추출 결과를 MongoDB에 저장 (utils/mongodb_utils.py 필요)')
    parser.add_argument('--test-mongodb', action='store_true', default=False,
                       help='MongoDB 연결 테스트만 수행하고 종료')

    args = parser.parse_args()
    
    # 로그 레벨 설정 - 루트 로거와 모든 핸들러에 적용
    log_level = getattr(logging, args.log_level)
    root_logger.setLevel(log_level)
    for handler in root_logger.handlers:
        handler.setLevel(log_level)
    
    logger.info(f"로그 레벨 설정: {args.log_level}")
    
    # MongoDB 연결 테스트만 수행하는 경우
    if args.test_mongodb:
        try:
            from utils.mongodb_utils import test_mongodb_connection
        except ImportError:
            print("❌ MongoDB 유틸리티를 찾을 수 없습니다.")
            print("utils/mongodb_utils.py 파일과 pymongo 패키지를 확인하세요.")
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
        'entity_llm_model': args.entity_llm_model,
        'extract_entity_dag': args.extract_entity_dag,  # DAG 추출 여부
        'entity_extraction_context_mode': args.entity_extraction_context_mode  # 컨텍스트 모드
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
        save_to_mongodb=args.save_to_mongodb,
        save_results_enabled=args.save_results
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