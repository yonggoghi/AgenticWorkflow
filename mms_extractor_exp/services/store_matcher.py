"""
MMS Extractor - Store Matcher Service
====================================
"""

import logging
from typing import List, Dict
import pandas as pd
from utils import (
    safe_execute,
    parallel_fuzzy_similarity,
    combined_sequence_similarity,
    preprocess_text
)

# Config imports
try:
    from config.settings import PROCESSING_CONFIG
except ImportError:
    logging.warning("Config file not found. Using defaults.")
    class PROCESSING_CONFIG:
        n_jobs = 4
        batch_size = 100
        similarity_threshold_for_store = 0.2

logger = logging.getLogger(__name__)

class StoreMatcher:
    """Service for matching store/agency names to organization database"""

    def __init__(self, org_pdf: pd.DataFrame):
        self.org_pdf = org_pdf

    def match_store(self, store_name: str) -> List[Dict]:
        """Match store name to organization info"""
        try:
            org_pdf_cand = safe_execute(
                parallel_fuzzy_similarity,
                [preprocess_text(store_name.lower())],
                self.org_pdf['item_nm'].unique(),
                threshold=0.5,
                text_col_nm='org_nm_in_msg',
                item_col_nm='item_nm',
                n_jobs=getattr(PROCESSING_CONFIG, 'n_jobs', 4),
                batch_size=getattr(PROCESSING_CONFIG, 'batch_size', 100),
                default_return=pd.DataFrame()
            )

            if org_pdf_cand.empty:
                return []

            org_pdf_cand = org_pdf_cand.drop('org_nm_in_msg', axis=1)
            org_pdf_cand = self.org_pdf.merge(org_pdf_cand, on=['item_nm'])
            org_pdf_cand['sim'] = org_pdf_cand.apply(
                lambda x: combined_sequence_similarity(store_name, x['item_nm'])[0], axis=1
            ).round(5)
            
            similarity_threshold = getattr(PROCESSING_CONFIG, 'similarity_threshold_for_store', 0.2)
            org_pdf_tmp = org_pdf_cand.query(
                "sim >= @similarity_threshold", engine='python'
            ).sort_values('sim', ascending=False)
            
            if org_pdf_tmp.empty:
                org_pdf_tmp = org_pdf_cand.query("sim >= @similarity_threshold").sort_values('sim', ascending=False)
            
            if not org_pdf_tmp.empty:
                org_pdf_tmp['rank'] = org_pdf_tmp['sim'].rank(method='dense', ascending=False)
                org_pdf_tmp = org_pdf_tmp.rename(columns={'item_id':'org_cd','item_nm':'org_nm'})
                org_info = org_pdf_tmp.query("rank == 1").groupby('org_nm')['org_cd'].apply(list).reset_index(name='org_cd').to_dict('records')
                return org_info
            else:
                return []
                
        except Exception as e:
            logger.error(f"Store matching failed: {e}")
            return []
