"""
Classification tools for MMS Agent
Independent implementation using mms_agent.core
"""

import sys
import os
from typing import Dict, Any
from langchain.tools import tool
import re
import logging

logger = logging.getLogger(__name__)

# Import independent core
from ..core import ExtractorBase

# Singleton instance
_extractor = ExtractorBase()


@tool
def classify_program(message: str, top_k: int = 5) -> Dict[str, Any]:
    """
    임베딩 유사도 기반으로 메시지를 프로그램으로 분류합니다.
    
    Args:
        message: 분류할 MMS 메시지
        top_k: 반환할 상위 프로그램 수 (기본 5)
    
    Returns:
        {
            "programs": [{"pgm_nm": "...", "pgm_id": "...", "similarity": 0.95}],
            "context": "프로그램명 : 키워드\\n..."
        }
    """
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
        import torch
        
        # Check availability
        if _extractor.emb_model is None:
            return {"programs": [], "context": "", "error": "Embedding model not available"}
        
        if _extractor.clue_embeddings is None or _extractor.clue_embeddings.numel() == 0:
            return {"programs": [], "context": "", "error": "Clue embeddings not available"}
        
        # Message embedding
        mms_embedding = _extractor.emb_model.encode(
            [message.lower()],
            convert_to_tensor=True,
            show_progress_bar=False
        )
        
        # Calculate similarities
        similarities = torch.nn.functional.cosine_similarity(
            mms_embedding,
            _extractor.clue_embeddings,
            dim=1
        ).cpu().numpy()
        
        # Get top programs
        if _extractor.pgm_pdf.empty:
            return {"programs": [], "context": ""}
        
        pgm_pdf_tmp = _extractor.pgm_pdf.copy()
        pgm_pdf_tmp['sim'] = similarities
        pgm_pdf_tmp = pgm_pdf_tmp.sort_values('sim', ascending=False)
        
        # Build program list
        top_programs = pgm_pdf_tmp.head(top_k)
        programs = []
        for _, row in top_programs.iterrows():
            programs.append({
                "pgm_nm": row['pgm_nm'],
                "pgm_id": row.get('pgm_id', ''),
                "similarity": float(row['sim'])
            })
        
        # Build context string
        context = "\n\t".join(
            top_programs[['pgm_nm', 'clue_tag']].apply(
                lambda x: re.sub(r'\[.*?\]', '', x['pgm_nm']) + " : " + x['clue_tag'],
                axis=1
            ).tolist()
        )
        
        return {
            "programs": programs,
            "context": context
        }
        
    except ImportError:
        logger.error("torch not installed")
        return {
            "programs": [],
            "context": "",
            "error": "torch not installed"
        }
    except Exception as e:
        logger.error(f"classify_program failed: {e}")
        return {
            "programs": [],
            "context": "",
            "error": str(e)
        }


if __name__ == "__main__":
    # Test tool
    print("Testing classify_program...")
    result = classify_program.invoke({
        "message": "5GX 프라임 요금제 가입하고 아이폰 받으세요",
        "top_k": 3
    })
    print(result)
