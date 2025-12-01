"""
Matching tools for MMS Agent
Independent implementation using mms_agent.core
"""

import sys
import os
from typing import List, Dict
from langchain.tools import tool
import logging

logger = logging.getLogger(__name__)

# Import independent core
from ..core import ExtractorBase

# Singleton instance
_extractor = ExtractorBase()


@tool
def match_store_info(store_name: str) -> str:
    """
    대리점명으로 조직 정보를 검색합니다.
    
    Args:
        store_name: 대리점 이름
    
    Returns:
        JSON 문자열: [{"org_nm": "조직명", "org_cd": ["코드1"]}, ...]
    """
    import json
    from rapidfuzz import process, fuzz
    
    try:
        if _extractor.org_pdf.empty:
            return json.dumps([])
        
        # Fuzzy search
        choices = _extractor.org_pdf['item_nm'].unique().tolist()
        matches = process.extract(
            store_name,
            choices,
            scorer=fuzz.WRatio,
            limit=10,
            score_cutoff=50  # 50% threshold
        )
        
        if not matches:
            return json.dumps([])
        
        # Get top matches
        result = []
        seen = set()
        for name, score, idx in matches[:5]:  # Top 5
            if name not in seen:
                # Find org codes
                matched_orgs = _extractor.org_pdf[_extractor.org_pdf['item_nm'] == name]
                if not matched_orgs.empty:
                    org_codes = matched_orgs['item_id'].tolist()
                    result.append({
                        "org_nm": name,
                        "org_cd": org_codes
                    })
                    seen.add(name)
        
        return json.dumps(result, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"match_store_info failed: {e}")
        return json.dumps({"error": str(e)})


if __name__ == "__main__":
    # Test tool
    print("Testing match_store_info...")
    result = match_store_info.invoke({"store_name": "티원대리점"})
    print(result)
