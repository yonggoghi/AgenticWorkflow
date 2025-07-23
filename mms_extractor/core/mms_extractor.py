"""
Main MMS extractor class for information extraction from MMS messages.
"""
import re
import json
import torch
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import logging

from ..models.language_models import LLMManager, EmbeddingManager
from ..core.entity_extractor import KiwiEntityExtractor, extract_entities_by_logic, extract_entities_by_llm
from ..utils.text_processing import extract_json_objects, convert_df_to_json_list
from ..utils.similarity import combined_sequence_similarity, parallel_fuzzy_similarity, parallel_seq_similarity
from ..config.settings import (
    DATA_CONFIG, PROCESSING_CONFIG, EXTRACTION_SCHEMA, 
    MODEL_CONFIG, get_device
)


logger = logging.getLogger(__name__)


class MMSExtractor:
    """Main class for extracting structured information from MMS messages."""
    
    def __init__(self, data_manager: Optional['DataManager'] = None, model_name: str = 'gemma_3'):
        """
        Initialize MMS extractor.
        
        Args:
            data_manager: Optional data manager instance
            model_name: LLM model to use for extraction
        """
        self.data_manager = data_manager
        self.model_name = model_name
        self.llm_manager = LLMManager()
        self.embedding_manager = EmbeddingManager(
            model_name=MODEL_CONFIG.embedding_model,
            device=get_device()
        )
        self.entity_extractor = None
        
        # Data storage
        self.item_df = None
        self.pgm_df = None
        self.org_df = None
        self.clue_embeddings = None
        self.stop_words = []
        
        logger.info("Initialized MMS extractor")
    
    def load_data(self):
        """Load all required data."""
        if self.data_manager:
            logger.info("Loading data using data manager")
            try:
                self.item_df = self.data_manager.get_item_data()
                self.pgm_df = self.data_manager.get_program_data()
                self.org_df = self.data_manager.get_organization_data()
                self.stop_words = self.data_manager.get_stop_words()
                self.clue_embeddings = self.data_manager.get_clue_embeddings()
                
                # Initialize entity extractor
                entity_vocab = self.data_manager.get_entity_vocab()
                self.entity_extractor = KiwiEntityExtractor(entity_vocab, self.stop_words)
                
                logger.info("Data loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load data: {e}")
                raise
        else:
            raise ValueError("Data manager is required to load data")
    
    def extract_program_candidates(self, message: str) -> str:
        """
        Extract program candidates using embedding similarity.
        
        Args:
            message: Input MMS message
            
        Returns:
            RAG context string with program candidates
        """
        if self.clue_embeddings is None or self.pgm_df is None:
            return ""
        
        try:
            # Get message embedding
            message_embedding = self.embedding_manager.encode([message.lower()], convert_to_tensor=True)
            
            # Calculate similarities
            similarities = torch.nn.functional.cosine_similarity(
                message_embedding,
                self.clue_embeddings,
                dim=1
            ).cpu().numpy()
            
            # Get top candidates
            pgm_df_tmp = self.pgm_df.copy()
            pgm_df_tmp['sim'] = similarities
            pgm_df_tmp = pgm_df_tmp.sort_values('sim', ascending=False)
            
            # Format context
            pgm_cand_info = "\n\t".join(
                pgm_df_tmp.iloc[:PROCESSING_CONFIG.num_candidate_programs]
                [['pgm_nm', 'clue_tag']]
                .apply(lambda x: re.sub(r'\[.*?\]', '', str(x['pgm_nm'])) + " : " + str(x['clue_tag']), axis=1)
                .to_list()
            )
            
            return f"\n### 광고 분류 기준 정보 ###\n\t{pgm_cand_info}" if pgm_cand_info else ""
        except Exception as e:
            logger.warning(f"Failed to extract program candidates: {e}")
            return ""
    
    def create_extraction_prompt(self, message: str, rag_context: str, 
                               candidate_items: List[str] = None, 
                               product_elements: List[Dict[str, Any]] = None) -> str:
        """
        Create prompt for information extraction.
        
        Args:
            message: Input MMS message
            rag_context: RAG context information
            candidate_items: List of candidate items
            product_elements: Pre-filled product elements for NLP mode
            
        Returns:
            Formatted extraction prompt
        """
        # Get schema based on extraction mode
        schema = EXTRACTION_SCHEMA.get_product_schema(product_elements)
        
        schema_prompt = f"""
아래와 같은 스키마로 결과를 제공해 주세요.

{schema}
"""
        
        # Handle different extraction modes
        extraction_mode = PROCESSING_CONFIG.product_info_extraction_mode
        
        # Add candidate items to RAG context for RAG mode
        if extraction_mode == 'rag' and candidate_items and len(candidate_items) > 0:
            rag_context += f"\n\n### 후보 상품 이름 목록 ###\n\t{candidate_items}"
        
        # Get extraction guide based on mode
        prd_ext_guide = PROCESSING_CONFIG.get_extraction_guide(candidate_items)
        
        # Get chain of thought based on mode
        chain_of_thought = PROCESSING_CONFIG.chain_of_thought
        
        prompt = f"""
아래 광고 메시지에서 광고 목적과 상품 이름을 추출해 주세요.

### 광고 메시지 ###
{message}

### 추출 작업 순서 ###
{chain_of_thought}

### 추출 작업 가이드 ###
* 광고 목적에 대리점 방문이 포함되어 있으면 대리점 채널 정보를 제공해라.
{prd_ext_guide}

{schema_prompt}

{rag_context}
"""
        
        return prompt
    
    def extract_llm_information(self, message: str, candidate_items: List[str] = None,
                               extra_item_df: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Extract information using LLM with different extraction modes.
        
        Args:
            message: Input MMS message
            candidate_items: List of candidate items
            extra_item_df: DataFrame with extracted items for NLP mode
            
        Returns:
            Extracted information as dictionary
        """
        try:
            # Get program candidates
            rag_context = self.extract_program_candidates(message)
            
            # Handle different extraction modes
            extraction_mode = PROCESSING_CONFIG.product_info_extraction_mode
            product_elements = None
            
            if extraction_mode == 'nlp' and extra_item_df is not None and not extra_item_df.empty:
                # Create product elements from NLP extraction
                product_df = extra_item_df.rename(columns={'item_nm': 'name'})
                product_df = product_df[~product_df['name'].isin(self.stop_words)][['name']].copy()
                product_df['action'] = '고객에게 기대하는 행동: [구매, 가입, 사용, 방문, 참여, 코드입력, 쿠폰다운로드, 기타] 중에서 선택'
                product_df['position'] = '광고 상품의 분류. [main, sub, etc] 중에서 선택'
                product_elements = product_df.to_dict(orient='records') if not product_df.empty else None
            
            # Create extraction prompt
            prompt = self.create_extraction_prompt(message, rag_context, candidate_items, product_elements)
            
            # Generate response
            result_json_text = self.llm_manager.generate(prompt, self.model_name)
            
            # Extract JSON objects
            json_objects = extract_json_objects(result_json_text)
            if not json_objects:
                logger.warning("No JSON objects extracted from LLM response")
                return {}
            
            return json_objects[0]
        except Exception as e:
            logger.error(f"Failed to extract LLM information: {e}")
            return {}
    
    def match_products_to_vocabulary(self, llm_products: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Match LLM-extracted products to vocabulary using similarity.
        
        Args:
            llm_products: List of product dictionaries from LLM
            
        Returns:
            DataFrame with matched products
        """
        if not llm_products or self.item_df is None:
            return pd.DataFrame()
        
        try:
            # Extract product names
            product_names = []
            if isinstance(llm_products, dict) and 'items' in llm_products:
                product_names = [str(item.get('name', '')) for item in llm_products['items']]
            elif isinstance(llm_products, list):
                product_names = [str(item.get('name', '')) for item in llm_products]
            
            # Filter out empty names
            product_names = [name for name in product_names if name.strip()]
            
            if not product_names:
                return pd.DataFrame()
            
            # Get item aliases safely
            item_aliases = [str(alias) for alias in self.item_df['item_nm_alias'].unique() if pd.notna(alias)]
            
            # Calculate similarities
            similarities_fuzzy = parallel_fuzzy_similarity(
                product_names,
                item_aliases,
                threshold=0.8,
                text_col_nm='item_name_in_msg',
                item_col_nm='item_nm_alias',
                n_jobs=PROCESSING_CONFIG.n_jobs,
                batch_size=30
            )
            
            if similarities_fuzzy.empty:
                return pd.DataFrame()
            
            # Calculate sequence similarities
            similarities_seq = parallel_seq_similarity(
                sent_item_pdf=similarities_fuzzy,
                text_col_nm='item_name_in_msg',
                item_col_nm='item_nm_alias',
                n_jobs=PROCESSING_CONFIG.n_jobs,
                batch_size=PROCESSING_CONFIG.batch_size,
                normalization_value='min'
            )
            
            return similarities_seq
        except Exception as e:
            logger.warning(f"Failed to match products to vocabulary: {e}")
            return pd.DataFrame()
    
    def match_organizations(self, org_name: str) -> List[Dict[str, Any]]:
        """
        Match organization name to organization database.
        
        Args:
            org_name: Organization name to match
            
        Returns:
            List of matched organizations
        """
        if not org_name or self.org_df is None:
            return []
        
        try:
            # Get organization aliases safely
            org_aliases = [str(alias) for alias in self.org_df['org_abbr_nm'].unique() if pd.notna(alias)]
            
            # Use fuzzy similarity for organization matching
            org_similarities = parallel_fuzzy_similarity(
                [org_name.lower()],
                org_aliases,
                threshold=0.5,
                text_col_nm='org_nm_in_msg',
                item_col_nm='org_abbr_nm',
                n_jobs=PROCESSING_CONFIG.n_jobs,
                batch_size=100
            )
            
            if org_similarities.empty:
                return []
            
            # Drop the input column for merging
            org_similarities = org_similarities.drop('org_nm_in_msg', axis=1)
            
            # Merge with organization data
            org_pdf_cand = self.org_df.merge(org_similarities, on=['org_abbr_nm'])
            org_pdf_cand['sim'] = org_pdf_cand['sim'].round(5)
            
            # Filter by organization code starting with 'D' and similarity threshold
            d_code_mask = org_pdf_cand['org_cd'].str.startswith('D', na=False)
            sim_mask = org_pdf_cand['sim'] >= 0.7
            org_pdf_tmp = org_pdf_cand[d_code_mask & sim_mask].sort_values('sim', ascending=False)

            if org_pdf_tmp.empty:
                org_pdf_tmp = org_pdf_cand[sim_mask].sort_values('sim', ascending=False)
            
            if org_pdf_tmp.empty:
                return []
            
            # Calculate sequence similarity
            org_pdf_tmp['sim'] = org_pdf_tmp.apply(
                lambda x: combined_sequence_similarity(org_name, str(x['org_nm']))[0], axis=1
            )
            org_pdf_tmp['rank'] = org_pdf_tmp['sim'].rank(method='dense', ascending=False)
            org_pdf_tmp['org_cd'] = org_pdf_tmp.apply(
                lambda x: str(x['org_cd']) + str(x['sub_org_cd']), axis=1
            )
            
            return (org_pdf_tmp[org_pdf_tmp['rank'] == 1]
                    .groupby('org_nm')['org_cd']
                    .apply(list)
                    .reset_index(name='org_cd')
                    .to_dict('records'))
        except Exception as e:
            logger.warning(f"Failed to match organizations: {e}")
            return []
    
    def process_channels(self, channels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process channel information and add store info.
        
        Args:
            channels: List of channel dictionaries
            
        Returns:
            Processed channels with store information
        """
        processed_channels = []
        
        for channel in channels:
            try:
                channel_copy = channel.copy()
                
                if channel.get('type') == '대리점':
                    org_matches = self.match_organizations(channel.get('value', ''))
                    channel_copy['store_info'] = org_matches
                else:
                    channel_copy['store_info'] = []
                
                processed_channels.append(channel_copy)
            except Exception as e:
                logger.warning(f"Failed to process channel {channel}: {e}")
                # Add channel without store info
                channel_copy = channel.copy()
                channel_copy['store_info'] = []
                processed_channels.append(channel_copy)
        
        return processed_channels
    
    def process_programs(self, pgm_names: List[str]) -> List[Dict[str, str]]:
        """
        Process program names and match to program database.
        
        Args:
            pgm_names: List of program names
            
        Returns:
            List of matched programs with IDs
        """
        if not pgm_names or self.pgm_df is None:
            return []
        
        try:
            # Match program names
            pgm_names_str = ' '.join([str(name) for name in pgm_names])
            
            matched_programs = []
            for _, row in self.pgm_df.iterrows():
                pgm_nm_clean = re.sub(r'\[.*?\]', '', str(row['pgm_nm']))
                if pgm_nm_clean.strip() in pgm_names_str:
                    matched_programs.append({
                        'pgm_nm': str(row['pgm_nm']),
                        'pgm_id': str(row['pgm_id'])
                    })
            
            return matched_programs
        except Exception as e:
            logger.warning(f"Failed to process programs: {e}")
            return []
    
    def extract(self, message: str) -> Dict[str, Any]:
        """
        Extract structured information from MMS message.
        
        Args:
            message: Input MMS message
            
        Returns:
            Extracted structured information
        """
        logger.info("Starting MMS information extraction")
        
        if not self.entity_extractor:
            raise ValueError("Entity extractor not initialized. Call load_data() first.")
        
        try:
            # Extract entities using Kiwi
            candidate_items, extra_item_df = self.entity_extractor.extract_entities_from_message(
                message, self.item_df
            )
            
            logger.info(f"Extracted entities: {candidate_items}")
            
            # Extract information using LLM
            llm_result = self.extract_llm_information(message, candidate_items, extra_item_df)
            
            if not llm_result:
                logger.warning("No information extracted from LLM")
                return {
                    "title": "",
                    "purpose": [],
                    "product": [],
                    "channel": [],
                    "pgm": []
                }
            
            # Apply entity extraction mode for product matching
            entity_extraction_mode = PROCESSING_CONFIG.entity_extraction_mode
            
            if entity_extraction_mode == 'logic':
                # Logic-based entity extraction
                llm_products = llm_result.get('product', [])
                if isinstance(llm_products, dict) and 'items' in llm_products:
                    product_names = [str(item.get('name', '')) for item in llm_products['items']]
                elif isinstance(llm_products, list):
                    product_names = [str(item.get('name', '')) for item in llm_products]
                else:
                    product_names = []
                
                # Filter out empty names
                product_names = [name for name in product_names if name.strip()]
                
                if product_names:
                    similarities_fuzzy = extract_entities_by_logic(product_names, self.item_df, self.stop_words)
                else:
                    similarities_fuzzy = pd.DataFrame(columns=['item_name_in_msg', 'item_nm_alias', 'sim'])
                    
            elif entity_extraction_mode == 'llm':
                # LLM-based entity extraction
                similarities_fuzzy = extract_entities_by_llm(
                    self.llm_manager.get_model(self.model_name), 
                    message, 
                    self.item_df, 
                    self.stop_words
                )
            else:
                # Default to original product matching logic
                similarities_fuzzy = self.match_products_to_vocabulary(llm_result.get('product', []))
            
            # Build final result
            final_result = {
                "title": str(llm_result.get('title', '')),
                "purpose": llm_result.get('purpose', []),
                "product": [],
                "channel": [],
                "pgm": []
            }
            
            # Process products based on similarity results
            if not similarities_fuzzy.empty:
                # Apply filtering thresholds based on entity extraction mode
                if entity_extraction_mode in ['logic', 'llm']:
                    # Use higher threshold for logic/llm modes
                    high_sim_items = similarities_fuzzy.query('sim >= 1.5')['item_nm_alias'].unique()
                    
                    # Filter similarities_fuzzy for conditions
                    filtered_similarities = similarities_fuzzy[
                        (similarities_fuzzy['item_nm_alias'].isin(high_sim_items)) &
                        (~similarities_fuzzy['item_nm_alias'].str.contains('test', case=False)) &
                        (~similarities_fuzzy['item_name_in_msg'].isin(self.stop_words))
                    ]
                    
                    if not filtered_similarities.empty:
                        # Merge with item data and convert to final format
                        product_matches = self.item_df.merge(filtered_similarities, on=['item_nm_alias'])
                        final_result['product'] = convert_df_to_json_list(product_matches)
                else:
                    # Use original logic for backward compatibility
                    sim_mask = similarities_fuzzy['sim'] >= 0.8
                    test_mask = ~similarities_fuzzy['item_nm_alias'].str.contains('test', case=False, na=False)
                    stop_mask = ~similarities_fuzzy['item_nm_alias'].isin(self.stop_words)
                    msg_stop_mask = ~similarities_fuzzy['item_name_in_msg'].isin(self.stop_words)
                    
                    filtered_similarities = similarities_fuzzy[sim_mask & test_mask & stop_mask & msg_stop_mask]
                    
                    if not filtered_similarities.empty:
                        product_matches = self.item_df.merge(filtered_similarities, on=['item_nm_alias'])
                        final_result['product'] = convert_df_to_json_list(product_matches)
            
            # Use LLM products as fallback if no matches found
            if not final_result['product']:
                llm_products = llm_result.get('product', [])
                if isinstance(llm_products, dict) and 'items' in llm_products:
                    product_list = llm_products['items']
                else:
                    product_list = llm_products if isinstance(llm_products, list) else []
                
                final_result['product'] = [
                    {
                        'item_name_in_msg': str(product.get('name', '')),
                        'item_in_voca': [{'item_name_in_voca': str(product.get('name', '')), 'item_id': ['#']}]
                    }
                    for product in product_list
                    if product.get('name') and str(product.get('name', '')).strip() not in self.stop_words
                ]
            
            # Process channels
            channels = llm_result.get('channel', [])
            if isinstance(channels, dict) and 'items' in channels:
                channels = channels['items']
            elif not isinstance(channels, list):
                channels = []
            
            final_result['channel'] = self.process_channels(channels)
            
            # Process programs
            pgm_names = llm_result.get('pgm', [])
            if not isinstance(pgm_names, list):
                pgm_names = []
            
            final_result['pgm'] = self.process_programs(pgm_names)
            
            logger.info("MMS information extraction completed")
            
            return final_result
            
        except Exception as e:
            logger.error(f"Error during MMS extraction: {e}")
            # Return empty structure on error
            return {
                "title": "",
                "purpose": [],
                "product": [],
                "channel": [],
                "pgm": []
            } 