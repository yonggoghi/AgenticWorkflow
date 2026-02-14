"""
Program Classifier Service - 프로그램 분류 서비스
==============================================

📋 개요: 임베딩 기반 프로그램 카테고리 분류
🔗 사용: ProgramClassificationStep에서 호출
⚙️ 방식: Cosine similarity로 상위 N개 후보 선택
"""

import logging
import re
from typing import Dict, Any
import pandas as pd
import torch

logger = logging.getLogger(__name__)

class ProgramClassifier:
    """
    프로그램 분류 서비스
    
    책임: 메시지를 사전 정의된 프로그램 카테고리로 분류
    방법: 임베딩 코사인 유사도 계산
    출력: 상위 N개 후보 프로그램 정보
    """

    def __init__(self, emb_model, pgm_pdf: pd.DataFrame, clue_embeddings: torch.Tensor, num_cand_pgms: int = 20):
        self.emb_model = emb_model
        self.pgm_pdf = pgm_pdf
        self.clue_embeddings = clue_embeddings
        self.num_cand_pgms = num_cand_pgms

    def classify(self, mms_msg: str) -> Dict[str, Any]:
        """Classify message into program categories"""
        try:
            if self.emb_model is None or self.clue_embeddings.numel() == 0:
                return {"pgm_cand_info": "", "similarities": []}
            
            mms_embedding = self.emb_model.encode([mms_msg.lower()], convert_to_tensor=True, show_progress_bar=False)
            similarities = torch.nn.functional.cosine_similarity(mms_embedding, self.clue_embeddings, dim=1).cpu().numpy()
            
            pgm_pdf_tmp = self.pgm_pdf.copy()
            pgm_pdf_tmp['sim'] = similarities
            pgm_pdf_tmp = pgm_pdf_tmp.sort_values('sim', ascending=False)
            
            pgm_cand_info = "\n\t".join(
                pgm_pdf_tmp.iloc[:self.num_cand_pgms][['pgm_nm','clue_tag']].apply(
                    lambda x: re.sub(r'\[.*?\]', '', x['pgm_nm']) + " : " + x['clue_tag'], axis=1
                ).to_list()
            )
            
            return {
                "pgm_cand_info": pgm_cand_info,
                "similarities": similarities,
                "pgm_pdf_tmp": pgm_pdf_tmp
            }
            
        except Exception as e:
            logger.error(f"Program classification failed: {e}")
            return {"pgm_cand_info": "", "similarities": [], "pgm_pdf_tmp": pd.DataFrame()}
