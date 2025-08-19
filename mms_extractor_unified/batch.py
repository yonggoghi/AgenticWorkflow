#!/usr/bin/env python3
"""
Batch Processing Script for MMS Message Extraction
=================================================

This script performs batch processing of MMS messages using the MMSExtractor.
It reads messages from a CSV file, processes them in batches, and saves results.

Features:
- Batch processing with MMSExtractor
- Result storage with timestamps
- Automatic update of processing status

Usage:
    python batch.py --batch-size 10
    python batch.py --batch-size 50 --output-file results_batch.csv
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
from mms_extractor import MMSExtractor

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


class BatchProcessor:
    """
    Batch processor for MMS message extraction
    """
    
    def __init__(self, result_file_path="./data/batch_results.csv"):
        """
        Initialize batch processor
        
        Args:
            result_file_path: Path to store batch processing results
        """
        self.result_file_path = result_file_path
        self.extractor = None
        self.mms_pdf = None
        
    def initialize_extractor(self, **extractor_kwargs):
        """
        Initialize the MMS extractor with given parameters
        
        Args:
            **extractor_kwargs: Keyword arguments for MMSExtractor initialization
        """
        logger.info("Initializing MMS Extractor...")
        try:
            self.extractor = MMSExtractor(**extractor_kwargs)
            logger.info("MMS Extractor initialized successfully")
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
        Process sampled messages using MMSExtractor
        
        Args:
            sampled_messages: DataFrame with messages to process
            
        Returns:
            list: List of processing results
        """
        if self.extractor is None:
            raise ValueError("Extractor not initialized. Call initialize_extractor() first.")
        
        results = []
        processing_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        logger.info(f"Processing {len(sampled_messages)} messages...")
        
        for idx, row in sampled_messages.iterrows():
            try:
                msg = row.get('msg', '')
                msg_id = row.get('msg_id', str(idx))
                
                logger.info(f"Processing message {msg_id}: {msg[:50]}...")
                
                # Process the message using MMSExtractor
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
                
                results.append(result_record)
                logger.info(f"Successfully processed message {msg_id}")
                
            except Exception as e:
                logger.error(f"Failed to process message {msg_id}: {str(e)}")
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
        
        logger.info(f"Completed processing {len(results)} messages")
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
        Run complete batch processing pipeline
        
        Args:
            batch_size: Number of messages to process
            **extractor_kwargs: Keyword arguments for MMSExtractor
            
        Returns:
            dict: Processing summary
        """
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
                    'message': 'No unprocessed messages found'
                }
            
            results = self.process_messages(sampled_messages)
            
            # Save results and log summary
            self.save_results(results)
            processed_msg_ids = [r['msg_id'] for r in results if 'error' not in r.get('extraction_result', '')]
            self.log_processing_summary(processed_msg_ids)
            
            return {
                'status': 'completed',
                'processed_count': len(results),
                'successful_count': len(processed_msg_ids),
                'failed_count': len(results) - len(processed_msg_ids),
                'results_file': self.result_file_path
            }
            
        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e),
                'processed_count': 0
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
    
    # MMSExtractor arguments
    parser.add_argument('--offer-data-source', choices=['local', 'db'], default='local',
                       help='Data source for offer information (default: local)')
    parser.add_argument('--product-info-extraction-mode', choices=['nlp', 'llm', 'rag'], default='llm',
                       help='Product information extraction mode (default: nlp)')
    parser.add_argument('--entity-extraction-mode', choices=['logic', 'llm'], default='llm',
                       help='Entity extraction mode (default: llm)')
    parser.add_argument('--llm-model', choices=['gem', 'ax', 'cld', 'gen', 'gpt'], default='ax',
                       help='LLM model to use (default: ax)')
    parser.add_argument('--extract-entity-dag', action='store_true', default=False, help='Entity DAG extraction (default: False)')

    args = parser.parse_args()
    
    # Prepare extractor arguments
    extractor_kwargs = {
        'offer_info_data_src': args.offer_data_source,
        'product_info_extraction_mode': args.product_info_extraction_mode,
        'entity_extraction_mode': args.entity_extraction_mode,
        'llm_model': args.llm_model,
        'extract_entity_dag': args.extract_entity_dag
    }
    
    logger.info("="*50)
    logger.info("Starting Batch MMS Processing")
    logger.info("="*50)
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Output file: {args.output_file}")
    logger.info(f"Extractor config: {extractor_kwargs}")
    logger.info("="*50)
    
    # Run batch processing
    processor = BatchProcessor(result_file_path=args.output_file)
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