"""
Product Schema Transformer - Schema transformation utilities
============================================================

This service handles transformation of product data between different schemas,
particularly converting from item_name_in_msg-centric to item_nm-centric format.
"""

import logging
import pandas as pd
from typing import List, Dict, Any
from utils import select_most_comprehensive

logger = logging.getLogger(__name__)


class ProductSchemaTransformer:
    """
    Transforms product data between different schema formats
    
    Main transformation: From item_name_in_msg-centric to item_nm-centric
    
    Before (LLM output):
        {
            "item_name_in_msg": "갤럭시 S24",
            "item_in_voca": "갤럭시S24 5G",
            "expected_action": "구매"
        }
    
    After (Final schema):
        {
            "item_nm": "갤럭시S24 5G",
            "item_id": ["PROD123", "PROD456"],
            "item_name_in_msg": ["갤럭시 S24"],
            "expected_action": ["구매"]
        }
    """
    
    def __init__(self):
        """Initialize the transformer"""
        pass
    
    def transform_to_item_schema(
        self,
        df: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """
        데이터프레임을 Item Centric 스키마로 변환
        
        Args:
            df: DataFrame with columns: 
                         item_nm, item_id, item_name_in_msg, expected_action (optional)
        
        Returns:
            List of dictionaries with item_nm as key and aggregated fields
        
        Example:
            >>> df = pd.DataFrame([
            ...     {"item_nm": "갤럭시S24", "item_id": "P1", "item_name_in_msg": "갤럭시"},
            ...     {"item_nm": "갤럭시S24", "item_id": "P2", "item_name_in_msg": "S24"}
            ... ])
            >>> transformer.transform_to_item_schema(df)
            [
                {
                    "item_nm": "갤럭시S24",
                    "item_id": ["P1", "P2"],
                    "item_name_in_msg": ["갤럭시", "S24"]
                }
            ]
        """
        if df.empty:
            logger.debug("Empty products DataFrame, returning empty list")
            return []
        
        logger.debug(f"Transforming {len(df)} product rows to item-centric schema")
        
        result = []
        
        # Group by item_nm
        grouped = df.groupby('item_nm')
        
        for item_nm, group in grouped:
            item_dict = self._aggregate_group(item_nm, group)
            result.append(item_dict)
        
        logger.debug(f"Transformed to {len(result)} unique items")
        return result
    
    def _aggregate_group(
        self,
        item_nm: str,
        group: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Aggregate fields for a group of products with same item_nm
        
        Args:
            item_nm: The item name (key)
            group: DataFrame group for this item_nm
        
        Returns:
            Dictionary with aggregated fields
        """
        # Get unique item_ids
        item_ids = list(group['item_id'].unique())
        
        # Get unique item_name_in_msg and select most comprehensive
        item_names_in_msg_raw = list(group['item_name_in_msg'].unique())
        item_names_in_msg = select_most_comprehensive(item_names_in_msg_raw)
        
        # Base item dictionary
        item_dict = {
            'item_nm': item_nm,
            'item_id': item_ids,
            'item_name_in_msg': item_names_in_msg
        }
        
        # Add expected_action if it exists
        if 'expected_action' in group.columns:
            expected_actions = [
                action for action in group['expected_action'].dropna().unique()
                if action and str(action).strip()
            ]
            if expected_actions:
                item_dict['expected_action'] = expected_actions
        
        return item_dict
    
    def transform_llm_products_to_schema(
        self,
        product_list: List[Dict],
        matched_entities_df: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """
        LLM 추출 결과(List[Dict])를 최종 스키마로 변환
        
        Args:
            product_list: List of products from LLM (item_name_in_msg-centric)
            matched_entities_df: DataFrame with matched entity information
                                 Columns: item_nm, item_id, item_name_in_msg, similarity
        
        Returns:
            List of products in item_nm-centric schema
        """
        if not product_list or matched_entities_df.empty:
            logger.debug("No products or matched entities, returning empty list")
            return []
        
        logger.debug(f"Transforming {len(product_list)} LLM products with {len(matched_entities_df)} matched entities")
        
        # Create a mapping DataFrame combining LLM output with matched entities
        transformation_rows = []
        
        for llm_product in product_list:
            item_name_in_msg = llm_product.get('item_name_in_msg') or llm_product.get('name')
            expected_action = llm_product.get('expected_action') or llm_product.get('action')
            
            if not item_name_in_msg:
                logger.warning(f"Product missing item_name_in_msg: {llm_product}")
                continue
            
            # Find matching entities for this item_name_in_msg
            matches = matched_entities_df[
                matched_entities_df['item_name_in_msg'] == item_name_in_msg
            ]
            
            if matches.empty:
                logger.debug(f"No matched entities for '{item_name_in_msg}', skipping")
                continue
            
            # Add rows for each matched entity
            for _, match in matches.iterrows():
                transformation_rows.append({
                    'item_nm': match['item_nm'],
                    'item_id': match['item_id'],
                    'item_name_in_msg': item_name_in_msg,
                    'expected_action': expected_action
                })
        
        if not transformation_rows:
            logger.warning("No transformation rows created")
            return []
        
        # Convert to DataFrame and transform
        transformation_df = pd.DataFrame(transformation_rows)
        return self.transform_to_item_schema(transformation_df)
