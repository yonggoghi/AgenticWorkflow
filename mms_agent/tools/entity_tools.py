"""
Entity extraction tools for MMS Agent
Independent implementation using mms_agent.core
"""

import sys
import os
from typing import Dict, Any, List
from langchain.tools import tool
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Import independent core
from ..core import ExtractorBase

# Singleton instance
_extractor = ExtractorBase()


@tool
def search_entities_kiwi(message: str) -> Dict[str, Any]:
    """
    Kiwi 형태소 분석을 통해 메시지에서 엔티티 후보를 추출합니다.
    
    Args:
        message: 분석할 MMS 메시지
    
    Returns:
        {
            "entities": ["엔티티1", "엔티티2", ...],
            "candidate_items": ["후보1", "후보2", ...],
            "extra_item_count": 10
        }
    """
    try:
        # Call core method
        entities_from_kiwi, extra_item_pdf = _extractor.extract_entities_from_kiwi(message)
        
        # Extract candidate items
        if not extra_item_pdf.empty and 'item_nm' in extra_item_pdf.columns:
            cand_item_list = extra_item_pdf['item_nm'].unique().tolist()
        else:
            cand_item_list = []
        
        return {
            "entities": entities_from_kiwi if entities_from_kiwi else [],
            "candidate_items": cand_item_list,
            "extra_item_count": len(extra_item_pdf)
        }
    except Exception as e:
        logger.error(f"search_entities_kiwi failed: {e}")
        return {
            "entities": [],
            "candidate_items": [],
            "extra_item_count": 0,
            "error": str(e)
        }


@tool
def search_entities_fuzzy(
    entities: str,
    threshold: float = 0.5
) -> str:
    """
    후보 엔티티 리스트를 Fuzzy matching으로 상품 DB와 매칭합니다.
    
    Args:
        entities: 후보 엔티티 리스트 (콤마로 구분된 문자열)
        threshold: 유사도 임계값 (기본 0.5)
    
    Returns:
        JSON 문자열: 매칭된 상품 정보 리스트
    """
    import json
    from rapidfuzz import process, fuzz
    
    try:
        # Parse entities
        entity_list = [e.strip() for e in entities.split(',') if e.strip()]
        
        if not entity_list:
            return json.dumps([])
        
        if _extractor.item_pdf.empty:
            return json.dumps([])
        
        # Fuzzy matching
        result = []
        for entity in entity_list:
            # Search in item_nm (item_nm_alias is mostly nan)
            choices = _extractor.item_pdf['item_nm'].dropna().astype(str).tolist()
            if not choices:
                continue
                
            matches = process.extract(
                entity,
                choices,
                scorer=fuzz.WRatio,
                limit=5,
                score_cutoff=threshold * 100
            )
            
            for name, score, idx in matches:
                # Find the actual row
                matched_rows = _extractor.item_pdf[_extractor.item_pdf['item_nm'].astype(str) == name]
                for _, row in matched_rows.head(1).iterrows():  # Take first match
                    result.append({
                        "item_nm": row.get('item_nm', ''),
                        "item_id": row.get('item_id', ''),
                        "similarity": score / 100.0
                    })
        
        return json.dumps(result, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"search_entities_fuzzy failed: {e}")
        return json.dumps({"error": str(e)})


@tool
def validate_entities(
    entities_json: str,
    message: str
) -> str:
    """
    추출된 엔티티를 별칭 규칙과 필터로 검증합니다.
    
    Args:
        entities_json: 엔티티 리스트 (JSON 문자열)
        message: 원본 메시지 (검증용)
    
    Returns:
        JSON 문자열: 검증된 엔티티 리스트
    """
    import json
    
    try:
        entities = json.loads(entities_json)
        
        if not entities:
            return json.dumps([])
        
        # Simple validation: filter by stop words
        filtered = [
            e for e in entities
            if e.get('item_nm', '') not in _extractor.stop_item_names
        ]
        
        return json.dumps(filtered, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"validate_entities failed: {e}")
        return json.dumps({"error": str(e)})


if __name__ == "__main__":
    # Test tools
    print("Testing search_entities_kiwi...")
    result = search_entities_kiwi.invoke({"message": "아이폰 15 구매하고 넷플릭스 받으세요"})
    print(result)
    
    print("\nTesting search_entities_fuzzy...")
    result = search_entities_fuzzy.invoke({"entities": "아이폰,넷플릭스", "threshold": 0.5})
    print(result)
