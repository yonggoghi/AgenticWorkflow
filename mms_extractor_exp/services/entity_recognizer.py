"""
MMS Extractor - Entity Recognizer Service
========================================

This service handles entity extraction and matching logic, decoupled from the main MMSExtractor.
"""

import logging
import traceback
import re
from typing import List, Tuple, Dict, Optional, Any
import pandas as pd
from langchain_core.prompts import PromptTemplate
from joblib import Parallel, delayed

# Utility imports
from utils import (
    log_performance,
    validate_text_input,
    safe_execute,
    parallel_fuzzy_similarity,
    parallel_seq_similarity,
    filter_text_by_exc_patterns,
    filter_specific_terms,
    extract_ngram_candidates,
    convert_df_to_json_list,
    select_most_comprehensive
)

# Prompt imports
from prompts import (
    HYBRID_DAG_EXTRACTION_PROMPT,
    SIMPLE_ENTITY_EXTRACTION_PROMPT
)

# Config imports
try:
    from config.settings import PROCESSING_CONFIG
except ImportError:
    logging.warning("Config file not found. Using defaults.")
    class PROCESSING_CONFIG:
        fuzzy_threshold = 0.5
        n_jobs = 4
        batch_size = 100
        similarity_threshold = 0.2
        combined_similarity_threshold = 0.2
        high_similarity_threshold = 1.0

logger = logging.getLogger(__name__)


class EntityRecognizer:
    """
    Service for extracting entities from text using Kiwi (NLP) and LLMs,
    and matching them against a product database.
    """

    def __init__(self, kiwi, item_pdf_all: pd.DataFrame, stop_item_names: List[str], 
                 llm_model, alias_pdf_raw: pd.DataFrame = None, entity_extraction_mode: str = 'llm'):
        """
        Initialize the EntityRecognizer service.

        Args:
            kiwi: Initialized Kiwi instance
            item_pdf_all: DataFrame containing all item information
            stop_item_names: List of stop words/items to ignore
            llm_model: Initialized LLM model instance
            alias_pdf_raw: DataFrame containing alias rules (optional)
            entity_extraction_mode: Mode of entity extraction ('llm', 'nlp', 'logic')
        """
        self.kiwi = kiwi
        self.item_pdf_all = item_pdf_all
        self.stop_item_names = stop_item_names
        self.llm_model = llm_model
        self.alias_pdf_raw = alias_pdf_raw
        self.entity_extraction_mode = entity_extraction_mode
        
        # Exclusion patterns for Kiwi
        self.exc_tag_patterns = [
            ['SN', 'NNB'], ['W_SERIAL'], ['JKO'], ['W_URL'], ['W_EMAIL'],
            ['XSV', 'EC'], ['VV', 'EC'], ['VCP', 'ETM'], ['XSA', 'ETM'],
            ['VV', 'ETN'], ['SSO'], ['SSC'], ['SW'], ['SF'], ['SP'], 
            ['SS'], ['SE'], ['SO'], ['SB'], ['SH'], ['W_HASHTAG']
        ]

    @log_performance
    def extract_entities_from_kiwi(self, mms_msg: str) -> Tuple[List[str], List[str], pd.DataFrame]:
        """Extract entities using Kiwi morphological analyzer"""
        try:
            logger.info("=== Kiwi Entity Extraction Started ===")
            mms_msg = validate_text_input(mms_msg)
            logger.info(f"Message length: {len(mms_msg)} chars")
            
            if self.item_pdf_all.empty:
                logger.error("Item data is empty! Cannot extract entities.")
                return [], [], pd.DataFrame()
            
            if 'item_nm_alias' not in self.item_pdf_all.columns:
                logger.error("item_nm_alias column missing! Cannot extract entities.")
                return [], [], pd.DataFrame()
            
            unique_aliases = self.item_pdf_all['item_nm_alias'].unique()
            logger.info(f"Number of aliases to match: {len(unique_aliases)}")
            
            # Sentence splitting
            sentences = sum(self.kiwi.split_into_sents(
                re.split(r"_+", mms_msg), return_tokens=True, return_sub_sents=True
            ), [])
            
            sentences_all = []
            for sent in sentences:
                if sent.subs:
                    sentences_all.extend(sent.subs)
                else:
                    sentences_all.append(sent)
            
            # Filter sentences
            sentence_list = [
                filter_text_by_exc_patterns(sent, self.exc_tag_patterns) 
                for sent in sentences_all
            ]
            
            # Tokenize and extract NNPs
            result_msg = self.kiwi.tokenize(mms_msg, normalize_coda=True, z_coda=False, split_complex=False)
            
            entities_from_kiwi = [
                token.form for token in result_msg 
                if token.tag == 'NNP' and 
                   token.form not in self.stop_item_names + ['-'] and 
                   len(token.form) >= 2 and 
                   not token.form.lower() in self.stop_item_names
            ]
            entities_from_kiwi = [e for e in filter_specific_terms(entities_from_kiwi) if e in unique_aliases]
            
            logger.info(f"Entities from Kiwi (filtered): {list(set(entities_from_kiwi))}")

            # Fuzzy matching
            logger.info("Starting fuzzy matching...")
            similarities_fuzzy = safe_execute(
                parallel_fuzzy_similarity,
                sentence_list, 
                unique_aliases,
                threshold=getattr(PROCESSING_CONFIG, 'fuzzy_threshold', 0.5),
                text_col_nm='sent', 
                item_col_nm='item_nm_alias',
                n_jobs=getattr(PROCESSING_CONFIG, 'n_jobs', 4),
                batch_size=30,
                default_return=pd.DataFrame()
            )
            
            if similarities_fuzzy.empty:
                logger.warning("Fuzzy matching result empty. Using Kiwi results only.")
                cand_item_list = list(entities_from_kiwi) if entities_from_kiwi else []
                
                if cand_item_list:
                    extra_item_pdf = self.item_pdf_all.query("item_nm_alias in @cand_item_list")[
                        ['item_nm','item_nm_alias','item_id']
                    ].groupby(["item_nm"])['item_id'].apply(list).reset_index()
                else:
                    extra_item_pdf = pd.DataFrame()
                
                return entities_from_kiwi, cand_item_list, extra_item_pdf

            # Sequence similarity
            logger.info("Starting sequence similarity calculation...")
            similarities_seq = safe_execute(
                parallel_seq_similarity,
                sent_item_pdf=similarities_fuzzy,
                text_col_nm='sent',
                item_col_nm='item_nm_alias',
                n_jobs=getattr(PROCESSING_CONFIG, 'n_jobs', 4),
                batch_size=getattr(PROCESSING_CONFIG, 'batch_size', 100),
                default_return=pd.DataFrame()
            )
            
            # Filter by threshold
            similarity_threshold = getattr(PROCESSING_CONFIG, 'similarity_threshold', 0.2)
            cand_items = similarities_seq.query(
                "sim >= @similarity_threshold and "
                "item_nm_alias.str.contains('', case=False) and "
                "item_nm_alias not in @self.stop_item_names"
            )
            
            # Add Kiwi entities
            entities_from_kiwi_pdf = self.item_pdf_all.query("item_nm_alias in @entities_from_kiwi")[
                ['item_nm','item_nm_alias']
            ]
            entities_from_kiwi_pdf['sim'] = 1.0

            # Merge results
            cand_item_pdf = pd.concat([cand_items, entities_from_kiwi_pdf])
            
            if not cand_item_pdf.empty:
                cand_item_array = cand_item_pdf.sort_values('sim', ascending=False).groupby([
                    "item_nm_alias"
                ])['sim'].max().reset_index(name='final_sim').sort_values(
                    'final_sim', ascending=False
                ).query("final_sim >= 0.2")['item_nm_alias'].unique()
                
                cand_item_list = list(cand_item_array) if hasattr(cand_item_array, '__iter__') else []
                
                if cand_item_list:
                    extra_item_pdf = self.item_pdf_all.query("item_nm_alias in @cand_item_list")[
                        ['item_nm','item_nm_alias','item_id']
                    ].groupby(["item_nm"])['item_id'].apply(list).reset_index()
                else:
                    extra_item_pdf = pd.DataFrame()
            else:
                cand_item_list = []
                extra_item_pdf = pd.DataFrame()

            return entities_from_kiwi, cand_item_list, extra_item_pdf
            
        except Exception as e:
            logger.error(f"Kiwi entity extraction failed: {e}")
            logger.error(f"Details: {traceback.format_exc()}")
            return [], [], pd.DataFrame()

    def extract_entities_by_logic(self, cand_entities: List[str], threshold_for_fuzzy: float = 0.5) -> pd.DataFrame:
        """Logic-based entity extraction (Fuzzy + Sequence)"""
        try:
            if not cand_entities:
                return pd.DataFrame()
            
            similarities_fuzzy = safe_execute(
                parallel_fuzzy_similarity,
                cand_entities,
                self.item_pdf_all['item_nm_alias'].unique(),
                threshold=threshold_for_fuzzy,
                text_col_nm='item_name_in_msg',
                item_col_nm='item_nm_alias',
                n_jobs=getattr(PROCESSING_CONFIG, 'n_jobs', 4),
                batch_size=30,
                default_return=pd.DataFrame()
            )
            
            if similarities_fuzzy.empty:
                return pd.DataFrame()
            
            cand_entities_sim = self._calculate_combined_similarity(similarities_fuzzy)
            return cand_entities_sim
            
        except Exception as e:
            logger.error(f"Logic-based extraction failed: {e}")
            return pd.DataFrame()

    def _calculate_combined_similarity(self, similarities_fuzzy: pd.DataFrame) -> pd.DataFrame:
        """Calculate combined similarity (s1 + s2)"""
        try:
            sim_s1 = safe_execute(
                parallel_seq_similarity,
                sent_item_pdf=similarities_fuzzy,
                text_col_nm='item_name_in_msg',
                item_col_nm='item_nm_alias',
                n_jobs=getattr(PROCESSING_CONFIG, 'n_jobs', 4),
                batch_size=30,
                normalizaton_value='s1',
                default_return=pd.DataFrame()
            ).rename(columns={'sim': 'sim_s1'})
            
            sim_s2 = safe_execute(
                parallel_seq_similarity,
                sent_item_pdf=similarities_fuzzy,
                text_col_nm='item_name_in_msg',
                item_col_nm='item_nm_alias',
                n_jobs=getattr(PROCESSING_CONFIG, 'n_jobs', 4),
                batch_size=30,
                normalizaton_value='s2',
                default_return=pd.DataFrame()
            ).rename(columns={'sim': 'sim_s2'})
            
            if not sim_s1.empty and not sim_s2.empty:
                combined = sim_s1.merge(sim_s2, on=['item_name_in_msg', 'item_nm_alias'])
                filtered = combined.query("(sim_s1>=@PROCESSING_CONFIG.combined_similarity_threshold and sim_s2>=@PROCESSING_CONFIG.combined_similarity_threshold)")
                
                if filtered.empty:
                    return pd.DataFrame()
                    
                combined = filtered.groupby(['item_name_in_msg', 'item_nm_alias']).agg({
                    'sim_s1': 'sum',
                    'sim_s2': 'sum'
                }).reset_index()
                combined['sim'] = combined['sim_s1'] + combined['sim_s2']
                return combined
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Combined similarity calculation failed: {e}")
            return pd.DataFrame()

    def _parse_entity_response(self, response: str) -> List[str]:
        """Parse entities from LLM response"""
        try:
            lines = response.split('\n')
            for line in lines:
                line_stripped = line.strip()
                line_upper = line_stripped.upper()
                
                if line_upper.startswith('REASON:'):
                    continue
                
                if line_upper.startswith('ENTITY:'):
                    entity_part = line_stripped[line_upper.find('ENTITY:') + 7:].strip()
                    
                    if not entity_part or entity_part.lower() in ['none', 'empty', '없음', 'null']:
                        return []
                    
                    if len(entity_part) > 200:
                        continue
                    
                    entities = [e.strip() for e in entity_part.split(',') if e.strip()]
                    return [e for e in entities if len(e) <= 100 and not (e.startswith('"') and not e.endswith('"'))]
            
            entity_pattern = r'ENTITY:\s*([^\n]*?)(?:\n|$)'
            entity_matches = list(re.finditer(entity_pattern, response, re.IGNORECASE))
            
            if entity_matches:
                last_match = entity_matches[-1]
                entity_text = last_match.group(1).strip()
                if entity_text and entity_text.lower() not in ['none', 'empty', '없음', 'null']:
                    if len(entity_text) <= 200:
                        return [e.strip() for e in entity_text.split(',') if e.strip() and len(e.strip()) <= 100]
            
            for line in reversed(lines):
                line_stripped = line.strip()
                if not line_stripped or line_stripped.upper().startswith('REASON:') or len(line_stripped) > 200:
                    continue
                
                if ',' in line_stripped:
                    entities = [e.strip() for e in line_stripped.split(',') if e.strip() and len(e.strip()) <= 100]
                    if entities and all(len(e) <= 100 for e in entities):
                        return entities
                elif len(line_stripped) <= 100:
                    return [line_stripped]
            
            return []
            
        except Exception as e:
            logger.error(f"Entity parsing failed: {e}")
            return []

    def _calculate_optimal_batch_size(self, msg_text: str, base_size: int = 50) -> int:
        """Calculate optimal batch size based on message length"""
        msg_length = len(msg_text)
        if msg_length < 500:
            return min(base_size * 2, 100)
        elif msg_length < 1000:
            return base_size
        else:
            return max(base_size // 2, 25)

    @log_performance
    def extract_entities_by_llm(self, msg_text: str, rank_limit: int = 50, llm_models: List = None, external_cand_entities: List[str] = []) -> pd.DataFrame:
        """LLM-based entity extraction with multi-model support"""
        try:
            logger.info("=== LLM Entity Extraction Started ===")
            msg_text = validate_text_input(msg_text)
            
            if llm_models is None:
                llm_models = [self.llm_model]
            
            # Internal function for parallel execution
            def get_entities_and_dag_by_llm(args_dict):
                llm_model, prompt = args_dict['llm_model'], args_dict['prompt']
                extract_dag = args_dict.get('extract_dag', True)
                model_name = getattr(llm_model, 'model_name', 'Unknown')
                
                try:
                    zero_shot_prompt = PromptTemplate(input_variables=["prompt"], template="{prompt}")
                    chain = zero_shot_prompt | llm_model
                    response = chain.invoke({"prompt": prompt}).content
                    
                    cand_entity_list_raw = self._parse_entity_response(response)
                    cand_entity_list = [e for e in cand_entity_list_raw if e not in self.stop_item_names and len(e) >= 2]
                    
                    dag_text = ""
                    if extract_dag:
                        dag_match = re.search(r'DAG:\s*(.*)', response, re.DOTALL | re.IGNORECASE)
                        if dag_match:
                            dag_text = dag_match.group(1).strip()
                    
                    return {"entities": cand_entity_list, "dag_text": dag_text}
                except Exception as e:
                    logger.error(f"LLM extraction failed for {model_name}: {e}")
                    return {"entities": [], "dag_text": ""}

            def get_entities_only_by_llm(args_dict):
                result = get_entities_and_dag_by_llm(args_dict)
                return result['entities']

            # 1. First Stage: Extract entities and DAG
            batches = []
            for llm_model in llm_models:
                prompt = f"{HYBRID_DAG_EXTRACTION_PROMPT}\n\n## message:\n{msg_text}"
                batches.append({"prompt": prompt, "llm_model": llm_model, "extract_dag": True})
            
            n_jobs = min(len(batches), 3)
            with Parallel(n_jobs=n_jobs, backend='threading') as parallel:
                batch_results_dicts = parallel(delayed(get_entities_and_dag_by_llm)(args) for args in batches)
            
            all_entities = []
            all_dags = []
            for result_dict in batch_results_dicts:
                all_entities.extend(result_dict['entities'])
                if result_dict['dag_text']:
                    all_dags.append(result_dict['dag_text'])
            
            combined_dag_context = "\n".join(all_dags)
            
            if external_cand_entities:
                all_entities.extend(external_cand_entities)
            
            cand_entity_list = list(set(all_entities))
            
            # N-gram expansion
            cand_entity_list = list(set(sum([[c['text'] for c in extract_ngram_candidates(cand_entity, min_n=2, max_n=len(cand_entity.split())) if c['start_idx']<=0] if len(cand_entity.split())>=4 else [cand_entity] for cand_entity in cand_entity_list], [])))
            
            if not cand_entity_list:
                return pd.DataFrame()
            
            # Match with products
            cand_entities_sim = self._match_entities_with_products(cand_entity_list, rank_limit)
            
            if cand_entities_sim.empty:
                return pd.DataFrame()
            
            # 2. Second Stage: Filtering
            entities_in_message = cand_entities_sim['item_name_in_msg'].unique()
            cand_entities_voca_all = cand_entities_sim['item_nm_alias'].unique()
            optimal_batch_size = self._calculate_optimal_batch_size(msg_text, base_size=10)
            
            second_stage_llm = llm_models[0] if llm_models else self.llm_model
            
            batches = []
            for i in range(0, len(cand_entities_voca_all), optimal_batch_size):
                cand_entities_voca = cand_entities_voca_all[i:i+optimal_batch_size]
                prompt = f"""
                {SIMPLE_ENTITY_EXTRACTION_PROMPT}
                
                ## message:                
                {msg_text}

                ## DAG Context (User Action Paths):
                {combined_dag_context}

                ## entities in message:
                {entities_in_message}

                ## candidate entities in vocabulary:
                {cand_entities_voca}
                """
                batches.append({"prompt": prompt, "llm_model": second_stage_llm, "extract_dag": False})
            
            n_jobs = min(len(batches), 3)
            with Parallel(n_jobs=n_jobs, backend='threading') as parallel:
                batch_results = parallel(delayed(get_entities_only_by_llm)(args) for args in batches)
            
            cand_entity_list = list(set(sum(batch_results, [])))
            
            cand_entities_sim = cand_entities_sim.query("item_nm_alias in @cand_entity_list")
            
            return cand_entities_sim
            
        except Exception as e:
            logger.error(f"LLM entity extraction failed: {e}")
            logger.error(traceback.format_exc())
            return pd.DataFrame()

    def _match_entities_with_products(self, cand_entity_list: List[str], rank_limit: int) -> pd.DataFrame:
        """Match candidate entities with product database"""
        try:
            similarities_fuzzy = parallel_fuzzy_similarity(
                cand_entity_list,
                self.item_pdf_all['item_nm_alias'].unique(),
                threshold=0.6,
                text_col_nm='item_name_in_msg',
                item_col_nm='item_nm_alias',
                n_jobs=6,
                batch_size=30
            )
            
            if similarities_fuzzy.empty:
                return pd.DataFrame()
            
            similarities_fuzzy = similarities_fuzzy[
                ~similarities_fuzzy['item_nm_alias'].isin(self.stop_item_names)
            ]
            
            sim_s1 = parallel_seq_similarity(
                sent_item_pdf=similarities_fuzzy,
                text_col_nm='item_name_in_msg',
                item_col_nm='item_nm_alias',
                n_jobs=6,
                batch_size=30,
                normalizaton_value='s1'
            ).rename(columns={'sim': 'sim_s1'})
            
            sim_s2 = parallel_seq_similarity(
                sent_item_pdf=similarities_fuzzy,
                text_col_nm='item_name_in_msg',
                item_col_nm='item_nm_alias',
                n_jobs=6,
                batch_size=30,
                normalizaton_value='s2'
            ).rename(columns={'sim': 'sim_s2'})
            
            cand_entities_sim = sim_s1.merge(sim_s2, on=['item_name_in_msg', 'item_nm_alias'])
            
            if cand_entities_sim.empty:
                return pd.DataFrame()
            
            cand_entities_sim = cand_entities_sim.query("(sim_s1>=@PROCESSING_CONFIG.combined_similarity_threshold and sim_s2>=@PROCESSING_CONFIG.combined_similarity_threshold)")
            
            cand_entities_sim = cand_entities_sim.groupby(['item_name_in_msg', 'item_nm_alias'])[['sim_s1', 'sim_s2']].apply(
                lambda x: x['sim_s1'].sum() + x['sim_s2'].sum()
            ).reset_index(name='sim')
            
            cand_entities_sim = cand_entities_sim.query("sim >= @PROCESSING_CONFIG.high_similarity_threshold").copy()
            
            if cand_entities_sim.empty:
                return pd.DataFrame()
            
            cand_entities_sim["rank"] = cand_entities_sim.groupby('item_name_in_msg')['sim'].rank(
                method='dense', ascending=False
            )
            cand_entities_sim = cand_entities_sim.query(f"rank <= {rank_limit}").sort_values(
                ['item_name_in_msg', 'rank'], ascending=[True, True]
            )
            
            if 'item_dmn_nm' in self.item_pdf_all.columns:
                cand_entities_sim = cand_entities_sim.merge(
                    self.item_pdf_all[['item_nm_alias', 'item_dmn_nm']].drop_duplicates(),
                    on='item_nm_alias',
                    how='left'
                )
            
            return cand_entities_sim
            
        except Exception as e:
            logger.error(f"Entity-product matching failed: {e}")
            return pd.DataFrame()

    def map_products_with_similarity(self, similarities_fuzzy: pd.DataFrame, json_objects: Dict = None) -> List[Dict]:
        """Map products based on similarity results"""
        try:
            logger.info("🔍 [map_products_with_similarity] Started")
            logger.info(f"   - Input similarities_fuzzy shape: {similarities_fuzzy.shape}")
            
            # Filter high similarity items
            high_sim_threshold = getattr(PROCESSING_CONFIG, 'high_similarity_threshold', 1.0)
            
            high_sim_items = similarities_fuzzy.query('sim >= @high_sim_threshold')['item_nm_alias'].unique()
            
            before_filter = len(similarities_fuzzy)
            filtered_similarities = similarities_fuzzy[
                (similarities_fuzzy['item_nm_alias'].isin(high_sim_items)) &
                (~similarities_fuzzy['item_nm_alias'].str.contains('test', case=False)) &
                (~similarities_fuzzy['item_name_in_msg'].isin(self.stop_item_names))
            ]
            after_filter = len(filtered_similarities)
            logger.info(f"   - Filtering: {before_filter} -> {after_filter}")
            
            if filtered_similarities.empty:
                logger.warning("   ⚠️ filtered_similarities is empty -> returning empty list")
                return []
            
            # Merge with product info
            merged_items = self.item_pdf_all.merge(filtered_similarities, on=['item_nm_alias'])
            
            if merged_items.empty:
                logger.warning("   ⚠️ merged_items is empty -> returning empty list")
                return []
            
            product_tag = convert_df_to_json_list(merged_items)
            logger.info(f"   ✅ product_tag count: {len(product_tag)}")
            
            # Add expected_action to each product
            if json_objects:
                action_mapping = self._create_action_mapping(json_objects)
                
                for product in product_tag:
                    # New schema: item_name_in_msg is a list
                    item_names_in_msg = product.get('item_name_in_msg', [])
                    found_actions = []
                    for item_name in item_names_in_msg:
                        if item_name in action_mapping:
                            found_actions.append(action_mapping[item_name])
                    product['expected_action'] = list(dict.fromkeys(found_actions)) if found_actions else ['기타']
            
            return product_tag
            
        except Exception as e:
            logger.error(f"❌ [map_products_with_similarity] Failed: {e}")
            logger.error(f"   Details: {traceback.format_exc()}")
            return []

    def _create_action_mapping(self, json_objects: Dict) -> Dict[str, str]:
        """Create product name to action mapping from LLM response"""
        try:
            action_mapping = {}
            product_data = json_objects.get('product', [])
            
            if isinstance(product_data, list):
                for item in product_data:
                    if isinstance(item, dict) and 'name' in item and 'action' in item:
                        action_mapping[item['name']] = item['action']
            elif isinstance(product_data, dict):
                if 'items' in product_data:
                    items = product_data.get('items', [])
                    for item in items:
                        if isinstance(item, dict) and 'name' in item and 'action' in item:
                            action_mapping[item['name']] = item['action']
                elif 'type' in product_data and product_data.get('type') == 'array':
                    logger.debug("Schema definition detected, skipping action mapping")
                else:
                    if 'name' in product_data and 'action' in product_data:
                        action_mapping[product_data['name']] = product_data['action']
            
            return action_mapping
            
        except Exception as e:
            logger.error(f"Action mapping creation failed: {e}")
            return {}
