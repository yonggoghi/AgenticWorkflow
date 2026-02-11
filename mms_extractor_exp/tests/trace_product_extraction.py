#!/usr/bin/env python3
"""
Product Extraction Tracing Tool
================================

A detailed tracing tool to understand why specific product results are generated
from an MMS message. This helps debug product extraction by tracking inputs and
outputs at each workflow step.

ENHANCED: Detailed tracing for entity extraction LLM calls and similarity scoring
to debug differences between context modes (DAG vs ONT vs PAIRING).

Usage:
    # Basic tracing with text output
    python tests/trace_product_extraction.py --message "아이폰 17 구매하세요"

    # JSON output
    python tests/trace_product_extraction.py --message "아이폰 17 구매하세요" --output-format json

    # With specific context mode (dag/ont/pairing/none)
    python tests/trace_product_extraction.py --message "메시지" --context-mode ont

    # Save to file
    python tests/trace_product_extraction.py --message "아이폰 17" --output-file trace.txt
"""

import argparse
import copy
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.product_trace_report import (
    StepTrace,
    TraceResult,
    generate_text_report,
    generate_json_report,
    generate_markdown_report
)

# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # Suppress most logs during tracing
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _safe_copy(obj: Any) -> Any:
    """Safely copy an object, handling non-copyable types."""
    if obj is None:
        return None
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, list):
        return [_safe_copy(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _safe_copy(v) for k, v in obj.items()}
    try:
        return copy.deepcopy(obj)
    except Exception:
        return str(obj)


def _dataframe_to_serializable(df: pd.DataFrame, max_rows: int = 50) -> List[Dict]:
    """Convert DataFrame to serializable list of dicts."""
    if df is None or df.empty:
        return []
    return df.head(max_rows).to_dict(orient='records')


class EntityExtractionTraceData:
    """
    Container for detailed entity extraction trace data.
    Captures LLM prompts, responses, and similarity calculations in Step 7.
    """
    def __init__(self):
        # First stage: Initial entity extraction
        self.first_stage_prompt: str = ""
        self.first_stage_response: str = ""
        self.first_stage_entities: List[str] = []
        self.context_text: str = ""  # DAG/ONT context extracted

        # For ONT mode: additional metadata
        self.ont_entity_types: Dict[str, str] = {}
        self.ont_relationships: List[Dict] = []

        # N-gram expansion
        self.entities_after_ngram: List[str] = []

        # Similarity matching - DETAILED FILTERING TRACE
        self.similarities_fuzzy_raw: Optional[pd.DataFrame] = None  # Initial fuzzy match
        self.similarities_with_seq: Optional[pd.DataFrame] = None   # After sequence similarity
        self.similarities_before_combined_filter: Optional[pd.DataFrame] = None  # Before combined_similarity_threshold
        self.similarities_after_combined_filter: Optional[pd.DataFrame] = None   # After combined_similarity_threshold
        self.similarities_before_high_sim_filter: Optional[pd.DataFrame] = None  # Before high_similarity_threshold
        self.similarities_after_high_sim_filter: Optional[pd.DataFrame] = None   # After high_similarity_threshold
        self.similarities_filtered: Optional[pd.DataFrame] = None   # After threshold filtering

        # Result builder filtering trace
        self.merged_with_alias_pdf: Optional[pd.DataFrame] = None  # After merge with alias_pdf_raw
        self.filtered_by_substring: Optional[pd.DataFrame] = None  # After substring matching filter
        self.map_products_input: Optional[pd.DataFrame] = None     # Input to map_products_to_entities
        self.high_sim_items_list: List[str] = []                   # Items passing high_sim_threshold
        self.filtered_similarities_final: Optional[pd.DataFrame] = None  # After final filtering
        self.merged_with_item_pdf: Optional[pd.DataFrame] = None   # After merge with item_pdf_all

        # Threshold values used
        self.thresholds_used: Dict[str, float] = {}

        # Second stage: LLM filtering - CRITICAL FOR DAG vs ONT DIFFERENCE
        self.second_stage_context_mode: str = ""
        self.second_stage_context_text: str = ""  # The context passed to 2nd stage (DAG/ONT context)
        self.second_stage_entities_in_message: List[str] = []  # entities_in_message
        self.second_stage_candidate_aliases: List[str] = []  # cand_entities_voca_all
        self.second_stage_prompts: List[str] = []  # Full prompts sent to 2nd stage LLM
        self.second_stage_responses: List[str] = []  # Raw responses from 2nd stage LLM
        self.second_stage_confirmed_entities: List[str] = []  # Entities confirmed by 2nd stage
        self.similarities_before_2nd_stage: Optional[pd.DataFrame] = None  # Before 2nd stage filter
        self.similarities_after_2nd_stage: Optional[pd.DataFrame] = None   # After 2nd stage filter

        # Sub-step timing for performance analysis
        self.substep_timings: Dict[str, float] = {}  # substep_name -> duration_seconds

        # Final result
        self.final_similarities: Optional[pd.DataFrame] = None
        self.mapped_products: List[Dict] = []


class ProductExtractionTracer:
    """
    Traces product extraction through all workflow steps.
    Captures inputs and outputs with focus on product-related data.

    ENHANCED: Deep tracing for entity extraction LLM calls and similarity scoring.
    """

    def __init__(self, extractor_kwargs: Optional[Dict] = None):
        """
        Initialize with MMSExtractor configuration.

        Args:
            extractor_kwargs: Optional kwargs to pass to MMSExtractor
        """
        self.extractor_kwargs = extractor_kwargs or {}
        self.extractor = None
        self._trace_data = {}
        self._entity_trace = EntityExtractionTraceData()

    def _initialize_extractor(self):
        """Initialize the MMSExtractor if not already done."""
        if self.extractor is None:
            from core.mms_extractor import MMSExtractor
            print("Initializing MMSExtractor... (this may take a moment)")
            self.extractor = MMSExtractor(**self.extractor_kwargs)
            print("MMSExtractor initialized.")

    def _get_extractor_config(self) -> Dict[str, Any]:
        """Extract relevant configuration from the extractor."""
        if self.extractor is None:
            return {}

        return {
            'llm_model': self.extractor.llm_model_name,
            'entity_extraction_mode': self.extractor.entity_extraction_mode,
            'product_info_extraction_mode': self.extractor.product_info_extraction_mode,
            'offer_info_data_src': self.extractor.offer_info_data_src,
            'extract_entity_dag': self.extractor.extract_entity_dag,
            'entity_extraction_context_mode': getattr(self.extractor, 'entity_extraction_context_mode', 'dag'),
            'num_cand_pgms': self.extractor.num_cand_pgms,
            'item_data_shape': str(self.extractor.item_pdf_all.shape),
            'pgm_data_shape': str(self.extractor.pgm_pdf.shape),
        }

    def _patch_entity_recognizer_for_tracing(self):
        """
        Patch the entity recognizer to capture detailed LLM calls and similarity data.
        This is the key enhancement to trace WHY products differ between context modes.
        ENHANCED: Now captures 2nd stage LLM filtering details.
        """
        recognizer = self.extractor.entity_recognizer
        original_extract_entities_with_llm = recognizer.extract_entities_with_llm
        trace_data = self._entity_trace

        def traced_extract_entities_with_llm(msg_text, rank_limit=50, llm_models=None,
                                             external_cand_entities=[], context_mode='dag',
                                             pre_extracted=None):
            """Wrapped version that captures all intermediate data including 2nd stage filtering."""
            from prompts import (
                SIMPLE_ENTITY_EXTRACTION_PROMPT,
                HYBRID_DAG_EXTRACTION_PROMPT,
                HYBRID_PAIRING_EXTRACTION_PROMPT,
                ONTOLOGY_PROMPT,
                TYPED_ENTITY_EXTRACTION_PROMPT,
                build_context_based_entity_extraction_prompt
            )
            from utils import extract_ngram_candidates, validate_text_input
            from joblib import Parallel, delayed
            import re

            # Capture context mode
            trace_data.second_stage_context_mode = context_mode

            # Determine which prompt is used
            if context_mode == 'dag':
                first_stage_prompt_template = HYBRID_DAG_EXTRACTION_PROMPT
                context_keyword = 'DAG'
            elif context_mode == 'pairing':
                first_stage_prompt_template = HYBRID_PAIRING_EXTRACTION_PROMPT
                context_keyword = 'PAIRING'
            elif context_mode == 'ont':
                first_stage_prompt_template = ONTOLOGY_PROMPT
                context_keyword = 'ONT'
            elif context_mode == 'typed':
                first_stage_prompt_template = TYPED_ENTITY_EXTRACTION_PROMPT
                context_keyword = 'TYPED'
            else:
                first_stage_prompt_template = SIMPLE_ENTITY_EXTRACTION_PROMPT
                context_keyword = None

            # Build full prompt
            full_prompt = f"{first_stage_prompt_template}\n\n## message:\n{msg_text}"
            trace_data.first_stage_prompt = full_prompt

            # We need to intercept the 2nd stage filtering
            # To do this, we'll re-implement the key parts of extract_entities_with_llm

            msg_text = validate_text_input(msg_text)
            if llm_models is None:
                llm_models = [recognizer.llm_model]

            # Internal function for getting entities
            def get_entities_and_context_by_llm(args_dict):
                llm_model, prompt = args_dict['llm_model'], args_dict['prompt']
                extract_context = args_dict.get('extract_context', True)
                context_kw = args_dict.get('context_keyword', None)
                is_ontology_mode = args_dict.get('is_ontology_mode', False)
                is_typed_mode = args_dict.get('is_typed_mode', False)

                try:
                    response = llm_model.invoke(prompt).content

                    if is_ontology_mode:
                        parsed = recognizer._parse_ontology_response(response)
                        cand_entity_list = [e for e in parsed['entities']
                                          if e not in recognizer.stop_item_names and len(e) >= 2]
                        entity_types = parsed.get('entity_types', {})
                        relationships = parsed.get('relationships', [])
                        dag_text = parsed['dag_text']

                        # Filter by entity type - keep only product/service-relevant types
                        from services.entity_recognizer import ONT_PRODUCT_RELEVANT_TYPES
                        removed = [f"{e}({entity_types.get(e, '?')})" for e in cand_entity_list
                                   if entity_types.get(e, 'Unknown') not in ONT_PRODUCT_RELEVANT_TYPES]
                        cand_entity_list = [e for e in cand_entity_list
                                           if entity_types.get(e, 'Unknown') in ONT_PRODUCT_RELEVANT_TYPES]
                        if removed:
                            logger.info(f"ONT type filter removed {len(removed)}: {removed}")

                        entity_type_str = ", ".join([f"{k}({v})" for k, v in entity_types.items()]) if entity_types else ""
                        rel_lines = []
                        for rel in relationships:
                            src = rel.get('source', '')
                            tgt = rel.get('target', '')
                            rel_type = rel.get('type', '')
                            if src and tgt and rel_type:
                                rel_lines.append(f"  - {src} -[{rel_type}]-> {tgt}")
                        relationships_str = "\n".join(rel_lines) if rel_lines else ""

                        context_parts = []
                        if entity_type_str:
                            context_parts.append(f"Entities: {entity_type_str}")
                        if relationships_str:
                            context_parts.append(f"Relationships:\n{relationships_str}")
                        if dag_text:
                            context_parts.append(f"DAG: {dag_text}")
                        context_text = "\n".join(context_parts)

                        return {
                            "entities": cand_entity_list,
                            "context_text": context_text,
                            "entity_types": entity_types,
                            "relationships": relationships,
                            "response": response
                        }

                    # Typed mode: JSON parsing
                    if is_typed_mode:
                        import json as _json
                        json_str = response.strip()
                        if json_str.startswith('```'):
                            json_str = re.sub(r'^```(?:json)?\n?', '', json_str)
                            json_str = re.sub(r'\n?```$', '', json_str)
                        try:
                            data = _json.loads(json_str)
                            entities_raw = data.get('entities', [])
                        except _json.JSONDecodeError:
                            entities_raw = []

                        cand_entity_list = [
                            e.get('name', '') for e in entities_raw
                            if e.get('name') and e['name'] not in recognizer.stop_item_names and len(e['name']) >= 2
                        ]
                        type_pairs = [
                            f"{e['name']}({e['type']})" for e in entities_raw
                            if e.get('name') and e.get('type')
                        ]
                        context_text = ", ".join(type_pairs)
                        return {"entities": cand_entity_list, "context_text": context_text, "response": response}

                    cand_entity_list_raw = recognizer._parse_entity_response(response)
                    cand_entity_list = [e for e in cand_entity_list_raw
                                       if e not in recognizer.stop_item_names and len(e) >= 2]

                    context_text = ""
                    if extract_context and context_kw:
                        context_match = re.search(rf'{context_kw}:\s*(.*)', response, re.DOTALL | re.IGNORECASE)
                        if context_match:
                            context_text = context_match.group(1).strip()

                    return {"entities": cand_entity_list, "context_text": context_text, "response": response}
                except Exception as e:
                    logger.error(f"LLM extraction failed: {e}")
                    return {"entities": [], "context_text": "", "response": ""}

            def get_entities_only_by_llm(args_dict):
                result = get_entities_and_context_by_llm(args_dict)
                # Capture 2nd stage response
                if 'response' in result:
                    trace_data.second_stage_responses.append(result.get('response', ''))
                return result['entities']

            # Initialize LLM sub-step timings
            llm_timings = {}

            # --- Pre-extracted entities: skip Stage 1 entirely (langextract) ---
            if pre_extracted:
                logger.info("=== Traced: Using pre-extracted entities (Stage 1 skipped) ===")
                from services.entity_recognizer import normalize_entity_name

                cand_entity_list = list(pre_extracted['entities'])
                combined_context = pre_extracted.get('context_text', '')
                context_keyword = 'TYPED'

                # Capture trace data
                trace_data.first_stage_prompt = "(pre-extracted by langextract)"
                trace_data.first_stage_response = f"langextract entities: {cand_entity_list}"
                trace_data.first_stage_entities = list(cand_entity_list)
                trace_data.second_stage_context_text = combined_context

                if external_cand_entities:
                    cand_entity_list = list(set(cand_entity_list + external_cand_entities))

                # Normalize + N-gram expansion
                t0 = time.time()
                cand_entity_list = list(set(
                    normalize_entity_name(e) for e in cand_entity_list if normalize_entity_name(e)
                ))
                cand_entity_list = list(set(sum([
                    [c['text'] for c in extract_ngram_candidates(cand_entity, min_n=2, max_n=len(cand_entity.split()))
                     if c['start_idx']<=0] if len(cand_entity.split())>=4 else [cand_entity]
                    for cand_entity in cand_entity_list
                ], [])))
                llm_timings['normalization_ngram'] = time.time() - t0

                if not cand_entity_list:
                    trace_data.substep_timings.update({f"llm_{k}": v for k, v in llm_timings.items()})
                    return pd.DataFrame()

                # Match with products
                t0 = time.time()
                cand_entities_sim = recognizer._match_entities_with_products(cand_entity_list, rank_limit)
                llm_timings['product_matching'] = time.time() - t0

                if cand_entities_sim.empty:
                    trace_data.substep_timings.update({f"llm_{k}": v for k, v in llm_timings.items()})
                    return pd.DataFrame()

                # *** CAPTURE BEFORE 2ND STAGE FILTERING ***
                trace_data.similarities_before_2nd_stage = cand_entities_sim.copy()

                # Stage 2: vocabulary filtering (same as standard path)
                t0 = time.time()
                entities_in_message = cand_entities_sim['item_name_in_msg'].unique()
                cand_entities_voca_all = cand_entities_sim['item_nm_alias'].unique()

                trace_data.second_stage_entities_in_message = list(entities_in_message)
                trace_data.second_stage_candidate_aliases = list(cand_entities_voca_all)

                optimal_batch_size = recognizer._calculate_optimal_batch_size(msg_text, base_size=10)
                second_stage_llm = llm_models[0] if llm_models else recognizer.llm_model

                batches = []
                for i in range(0, len(cand_entities_voca_all), optimal_batch_size):
                    cand_entities_voca = cand_entities_voca_all[i:i+optimal_batch_size]

                    context_section = f"\n## TYPED Context (Entity Types):\n{combined_context}\n" if combined_context else ""
                    second_stage_prompt = build_context_based_entity_extraction_prompt(context_keyword)

                    prompt = f"""
                    {second_stage_prompt}

                    ## message:
                    {msg_text}

                    {context_section}

                    ## entities in message:
                    {', '.join(entities_in_message)}

                    ## candidate entities in vocabulary:
                    {', '.join(cand_entities_voca)}
                    """
                    batches.append({
                        "prompt": prompt,
                        "llm_model": second_stage_llm,
                        "extract_context": False,
                        "context_keyword": None
                    })
                    trace_data.second_stage_prompts.append(prompt)

                n_jobs = min(len(batches), 3)
                with Parallel(n_jobs=n_jobs, backend='threading') as parallel:
                    batch_results = parallel(delayed(get_entities_only_by_llm)(args) for args in batches)

                cand_entity_list_2nd = list(set(sum(batch_results, [])))
                llm_timings['stage2_llm_call'] = time.time() - t0

                trace_data.second_stage_confirmed_entities = cand_entity_list_2nd

                cand_entities_sim = cand_entities_sim.query("item_nm_alias in @cand_entity_list_2nd")
                trace_data.similarities_after_2nd_stage = cand_entities_sim.copy() if not cand_entities_sim.empty else None

                for k, v in llm_timings.items():
                    trace_data.substep_timings[f"llm_{k}"] = v

                return cand_entities_sim

            # 1. First Stage
            t0 = time.time()
            batches = []
            for llm_model in llm_models:
                prompt = f"{first_stage_prompt_template}\n\n## message:\n{msg_text}"
                batches.append({
                    "prompt": prompt,
                    "llm_model": llm_model,
                    "extract_context": (context_mode != 'none'),
                    "context_keyword": context_keyword,
                    "is_ontology_mode": (context_mode == 'ont'),
                    "is_typed_mode": (context_mode == 'typed')
                })

            n_jobs = min(len(batches), 3)
            with Parallel(n_jobs=n_jobs, backend='threading') as parallel:
                batch_results_dicts = parallel(delayed(get_entities_and_context_by_llm)(args) for args in batches)
            llm_timings['stage1_llm_call'] = time.time() - t0

            all_entities = []
            all_contexts = []
            all_entity_types = {}
            all_relationships = []

            for result_dict in batch_results_dicts:
                all_entities.extend(result_dict['entities'])
                if result_dict['context_text']:
                    all_contexts.append(result_dict['context_text'])
                trace_data.first_stage_response = result_dict.get('response', '')
                if context_mode == 'ont':
                    if 'entity_types' in result_dict:
                        all_entity_types.update(result_dict.get('entity_types', {}))
                    if 'relationships' in result_dict:
                        all_relationships.extend(result_dict.get('relationships', []))

            combined_context = "\n".join(all_contexts)
            trace_data.second_stage_context_text = combined_context
            trace_data.first_stage_entities = list(set(all_entities))

            if external_cand_entities:
                all_entities.extend(external_cand_entities)

            cand_entity_list = list(set(all_entities))

            # Normalize entity names (strip parenthetical specs, collapse spaces)
            t0 = time.time()
            from services.entity_recognizer import normalize_entity_name
            cand_entity_list = list(set(
                normalize_entity_name(e) for e in cand_entity_list if normalize_entity_name(e)
            ))

            # N-gram expansion
            cand_entity_list = list(set(sum([
                [c['text'] for c in extract_ngram_candidates(cand_entity, min_n=2, max_n=len(cand_entity.split()))
                 if c['start_idx']<=0] if len(cand_entity.split())>=4 else [cand_entity]
                for cand_entity in cand_entity_list
            ], [])))
            llm_timings['normalization_ngram'] = time.time() - t0

            if not cand_entity_list:
                if context_mode == 'ont':
                    return {
                        'similarities_df': pd.DataFrame(),
                        'ont_metadata': {
                            'dag_text': combined_context,
                            'entity_types': all_entity_types,
                            'relationships': all_relationships
                        }
                    }
                return pd.DataFrame()

            # Match with products (this calls _match_entities_with_products which we also patch)
            t0 = time.time()
            cand_entities_sim = recognizer._match_entities_with_products(cand_entity_list, rank_limit)
            llm_timings['product_matching'] = time.time() - t0

            if cand_entities_sim.empty:
                if context_mode == 'ont':
                    return {
                        'similarities_df': pd.DataFrame(),
                        'ont_metadata': {
                            'dag_text': combined_context,
                            'entity_types': all_entity_types,
                            'relationships': all_relationships
                        }
                    }
                return pd.DataFrame()

            # *** CAPTURE BEFORE 2ND STAGE FILTERING ***
            trace_data.similarities_before_2nd_stage = cand_entities_sim.copy()

            # 2. Second Stage: Filtering - THIS IS THE KEY PART
            t0 = time.time()
            entities_in_message = cand_entities_sim['item_name_in_msg'].unique()
            cand_entities_voca_all = cand_entities_sim['item_nm_alias'].unique()

            # Capture 2nd stage inputs
            trace_data.second_stage_entities_in_message = list(entities_in_message)
            trace_data.second_stage_candidate_aliases = list(cand_entities_voca_all)

            optimal_batch_size = recognizer._calculate_optimal_batch_size(msg_text, base_size=10)
            second_stage_llm = llm_models[0] if llm_models else recognizer.llm_model

            batches = []
            for i in range(0, len(cand_entities_voca_all), optimal_batch_size):
                cand_entities_voca = cand_entities_voca_all[i:i+optimal_batch_size]

                # Build context section based on mode
                if context_mode == 'none' or not combined_context:
                    context_section = ""
                else:
                    if context_keyword == 'TYPED':
                        context_label = "TYPED Context (Entity Types)"
                    elif context_keyword:
                        context_label = f"{context_keyword} Context (User Action Paths)"
                    else:
                        context_label = "Context"
                    context_section = f"\n## {context_label}:\n{combined_context}\n"

                second_stage_prompt = build_context_based_entity_extraction_prompt(context_keyword)

                prompt = f"""
                {second_stage_prompt}

                ## message:
                {msg_text}

                {context_section}

                ## entities in message:
                {', '.join(entities_in_message)}

                ## candidate entities in vocabulary:
                {', '.join(cand_entities_voca)}
                """
                batches.append({
                    "prompt": prompt,
                    "llm_model": second_stage_llm,
                    "extract_context": False,
                    "context_keyword": None
                })
                # Capture 2nd stage prompts
                trace_data.second_stage_prompts.append(prompt)

            n_jobs = min(len(batches), 3)
            with Parallel(n_jobs=n_jobs, backend='threading') as parallel:
                batch_results = parallel(delayed(get_entities_only_by_llm)(args) for args in batches)

            cand_entity_list = list(set(sum(batch_results, [])))
            llm_timings['stage2_llm_call'] = time.time() - t0

            # Capture 2nd stage confirmed entities
            trace_data.second_stage_confirmed_entities = cand_entity_list

            # Apply 2nd stage filter
            cand_entities_sim = cand_entities_sim.query("item_nm_alias in @cand_entity_list")

            # *** CAPTURE AFTER 2ND STAGE FILTERING ***
            trace_data.similarities_after_2nd_stage = cand_entities_sim.copy() if not cand_entities_sim.empty else None

            # Merge LLM sub-step timings into trace_data (prefix with 'llm_' to distinguish)
            for k, v in llm_timings.items():
                trace_data.substep_timings[f"llm_{k}"] = v

            # ONT mode: return with metadata
            if context_mode == 'ont':
                return {
                    'similarities_df': cand_entities_sim,
                    'ont_metadata': {
                        'dag_text': combined_context,
                        'entity_types': all_entity_types,
                        'relationships': all_relationships
                    }
                }

            return cand_entities_sim

        # Apply patch for extract_entities_with_llm
        recognizer.extract_entities_with_llm = traced_extract_entities_with_llm

        # Also patch extract_entities_hybrid for sub-step timing
        original_extract_entities_hybrid = recognizer.extract_entities_hybrid

        def traced_extract_entities_hybrid(mms_msg):
            """Wrapped version that captures sub-step timing."""
            from utils import (validate_text_input, filter_text_by_exc_patterns, filter_specific_terms,
                              safe_execute, parallel_fuzzy_similarity, parallel_seq_similarity)
            from services.entity_recognizer import PROCESSING_CONFIG
            import re

            timings = {}
            trace_data.substep_timings = timings

            try:
                mms_msg = validate_text_input(mms_msg)

                if recognizer.item_pdf_all.empty or 'item_nm_alias' not in recognizer.item_pdf_all.columns:
                    return [], [], pd.DataFrame()

                unique_aliases = recognizer.item_pdf_all['item_nm_alias'].unique()

                # Sub-step 1: Kiwi sentence splitting
                t0 = time.time()
                sentences = sum(recognizer.kiwi.split_into_sents(
                    re.split(r"_+", mms_msg), return_tokens=True, return_sub_sents=True
                ), [])
                sentences_all = []
                for sent in sentences:
                    if sent.subs:
                        sentences_all.extend(sent.subs)
                    else:
                        sentences_all.append(sent)
                sentence_list = [
                    filter_text_by_exc_patterns(sent, recognizer.exc_tag_patterns)
                    for sent in sentences_all
                ]
                timings['kiwi_sentence_split'] = time.time() - t0

                # Sub-step 2: Kiwi tokenization + NNP extraction
                t0 = time.time()
                result_msg = recognizer.kiwi.tokenize(mms_msg, normalize_coda=True, z_coda=False, split_complex=False)
                entities_from_kiwi = [
                    token.form for token in result_msg
                    if token.tag == 'NNP' and
                       token.form not in recognizer.stop_item_names + ['-'] and
                       len(token.form) >= 2 and
                       not token.form.lower() in recognizer.stop_item_names
                ]
                entities_from_kiwi = [e for e in filter_specific_terms(entities_from_kiwi) if e in unique_aliases]
                timings['kiwi_tokenization_nnp'] = time.time() - t0

                # Sub-step 3: Fuzzy matching
                t0 = time.time()
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
                timings['fuzzy_matching'] = time.time() - t0

                if similarities_fuzzy.empty:
                    cand_item_list = list(entities_from_kiwi) if entities_from_kiwi else []
                    if cand_item_list:
                        extra_item_pdf = recognizer.item_pdf_all.query("item_nm_alias in @cand_item_list")[
                            ['item_nm','item_nm_alias','item_id']
                        ].groupby(["item_nm"])['item_id'].apply(list).reset_index()
                    else:
                        extra_item_pdf = pd.DataFrame()
                    return entities_from_kiwi, cand_item_list, extra_item_pdf

                # Sub-step 4: Sequence similarity
                t0 = time.time()
                similarities_seq = safe_execute(
                    parallel_seq_similarity,
                    sent_item_pdf=similarities_fuzzy,
                    text_col_nm='sent',
                    item_col_nm='item_nm_alias',
                    n_jobs=getattr(PROCESSING_CONFIG, 'n_jobs', 4),
                    batch_size=getattr(PROCESSING_CONFIG, 'batch_size', 100),
                    default_return=pd.DataFrame()
                )
                timings['sequence_similarity'] = time.time() - t0

                # Sub-step 5: Threshold filtering + merge
                t0 = time.time()
                similarity_threshold = getattr(PROCESSING_CONFIG, 'similarity_threshold', 0.2)
                cand_items = similarities_seq.query(
                    "sim >= @similarity_threshold and "
                    "item_nm_alias.str.contains('', case=False) and "
                    "item_nm_alias not in @recognizer.stop_item_names"
                )

                entities_from_kiwi_pdf = recognizer.item_pdf_all.query("item_nm_alias in @entities_from_kiwi")[
                    ['item_nm','item_nm_alias']
                ]
                entities_from_kiwi_pdf['sim'] = 1.0

                cand_item_pdf = pd.concat([cand_items, entities_from_kiwi_pdf])

                if not cand_item_pdf.empty:
                    cand_item_array = cand_item_pdf.sort_values('sim', ascending=False).groupby([
                        "item_nm_alias"
                    ])['sim'].max().reset_index(name='final_sim').sort_values(
                        'final_sim', ascending=False
                    ).query("final_sim >= 0.2")['item_nm_alias'].unique()

                    cand_item_list = list(cand_item_array) if hasattr(cand_item_array, '__iter__') else []

                    if cand_item_list:
                        extra_item_pdf = recognizer.item_pdf_all.query("item_nm_alias in @cand_item_list")[
                            ['item_nm','item_nm_alias','item_id']
                        ].groupby(["item_nm"])['item_id'].apply(list).reset_index()
                    else:
                        extra_item_pdf = pd.DataFrame()
                else:
                    cand_item_list = []
                    extra_item_pdf = pd.DataFrame()
                timings['threshold_filter_merge'] = time.time() - t0

                return entities_from_kiwi, cand_item_list, extra_item_pdf

            except Exception as e:
                logger.error(f"Traced extract_entities_hybrid failed: {e}")
                import traceback as tb
                logger.error(tb.format_exc())
                return [], [], pd.DataFrame()

        recognizer.extract_entities_hybrid = traced_extract_entities_hybrid

    def _patch_result_builder_for_tracing(self):
        """
        Patch the result builder to capture similarity DataFrames during product mapping.
        ENHANCED: Captures detailed filtering steps with thresholds.
        """
        builder = self.extractor.result_builder
        trace_data = self._entity_trace
        recognizer = self.extractor.entity_recognizer

        # Patch _match_entities_with_products for detailed filtering trace
        original_match_entities = recognizer._match_entities_with_products

        def traced_match_entities_with_products(cand_entity_list, rank_limit):
            """Capture detailed filtering steps with thresholds."""
            from utils import parallel_fuzzy_similarity, parallel_seq_similarity
            from config.settings import PROCESSING_CONFIG

            # Capture input
            trace_data.entities_after_ngram = list(cand_entity_list)

            # Capture thresholds
            trace_data.thresholds_used = {
                'entity_llm_fuzzy_threshold': getattr(PROCESSING_CONFIG, 'entity_llm_fuzzy_threshold', 0.6),
                'combined_similarity_threshold': getattr(PROCESSING_CONFIG, 'combined_similarity_threshold', 0.2),
                'high_similarity_threshold': getattr(PROCESSING_CONFIG, 'high_similarity_threshold', 1.0),
            }

            try:
                # Step 1: Fuzzy similarity
                similarities_fuzzy = parallel_fuzzy_similarity(
                    cand_entity_list,
                    recognizer.item_pdf_all['item_nm_alias'].unique(),
                    threshold=trace_data.thresholds_used['entity_llm_fuzzy_threshold'],
                    text_col_nm='item_name_in_msg',
                    item_col_nm='item_nm_alias',
                    n_jobs=6,
                    batch_size=30
                )
                trace_data.similarities_fuzzy_raw = similarities_fuzzy.copy() if not similarities_fuzzy.empty else None

                if similarities_fuzzy.empty:
                    return pd.DataFrame()

                # Filter stop items
                similarities_fuzzy = similarities_fuzzy[
                    ~similarities_fuzzy['item_nm_alias'].isin(recognizer.stop_item_names)
                ]

                # Step 2: Sequence similarity
                sim_s1 = parallel_seq_similarity(
                    sent_item_pdf=similarities_fuzzy,
                    text_col_nm='item_name_in_msg',
                    item_col_nm='item_nm_alias',
                    n_jobs=6,
                    batch_size=30,
                    normalization_value='s1'
                ).rename(columns={'sim': 'sim_s1'})

                sim_s2 = parallel_seq_similarity(
                    sent_item_pdf=similarities_fuzzy,
                    text_col_nm='item_name_in_msg',
                    item_col_nm='item_nm_alias',
                    n_jobs=6,
                    batch_size=30,
                    normalization_value='s2'
                ).rename(columns={'sim': 'sim_s2'})

                cand_entities_sim = sim_s1.merge(sim_s2, on=['item_name_in_msg', 'item_nm_alias'])
                trace_data.similarities_before_combined_filter = cand_entities_sim.copy() if not cand_entities_sim.empty else None

                if cand_entities_sim.empty:
                    return pd.DataFrame()

                # Step 3: Combined similarity threshold filter
                combined_threshold = trace_data.thresholds_used['combined_similarity_threshold']
                cand_entities_sim = cand_entities_sim.query(f"(sim_s1>={combined_threshold} and sim_s2>={combined_threshold})")
                trace_data.similarities_after_combined_filter = cand_entities_sim.copy() if not cand_entities_sim.empty else None

                if cand_entities_sim.empty:
                    return pd.DataFrame()

                # Step 4: Aggregate and compute combined score
                cand_entities_sim = cand_entities_sim.groupby(['item_name_in_msg', 'item_nm_alias'])[['sim_s1', 'sim_s2']].apply(
                    lambda x: x['sim_s1'].sum() + x['sim_s2'].sum()
                ).reset_index(name='sim')
                trace_data.similarities_before_high_sim_filter = cand_entities_sim.copy() if not cand_entities_sim.empty else None

                # Step 5: High similarity threshold filter
                high_sim_threshold = trace_data.thresholds_used['high_similarity_threshold']
                cand_entities_sim = cand_entities_sim.query(f"sim >= {high_sim_threshold}").copy()
                trace_data.similarities_after_high_sim_filter = cand_entities_sim.copy() if not cand_entities_sim.empty else None

                if cand_entities_sim.empty:
                    return pd.DataFrame()

                # Step 6: Ranking
                cand_entities_sim["rank"] = cand_entities_sim.groupby('item_name_in_msg')['sim'].rank(
                    method='dense', ascending=False
                )
                cand_entities_sim = cand_entities_sim.query(f"rank <= {rank_limit}").sort_values(
                    ['item_name_in_msg', 'rank'], ascending=[True, True]
                )

                if 'item_dmn_nm' in recognizer.item_pdf_all.columns:
                    cand_entities_sim = cand_entities_sim.merge(
                        recognizer.item_pdf_all[['item_nm_alias', 'item_dmn_nm']].drop_duplicates(),
                        on='item_nm_alias',
                        how='left'
                    )

                trace_data.final_similarities = cand_entities_sim.copy()
                return cand_entities_sim

            except Exception as e:
                logger.error(f"Traced match entities failed: {e}")
                return pd.DataFrame()

        recognizer._match_entities_with_products = traced_match_entities_with_products

        # Patch map_products_to_entities for detailed filtering trace
        original_map_products = recognizer.map_products_to_entities

        def traced_map_products_to_entities(similarities_fuzzy, json_objects=None):
            """Capture detailed filtering in map_products_to_entities."""
            from config.settings import PROCESSING_CONFIG

            trace_data.map_products_input = similarities_fuzzy.copy() if not similarities_fuzzy.empty else None

            try:
                high_sim_threshold = getattr(PROCESSING_CONFIG, 'high_similarity_threshold', 1.0)
                trace_data.thresholds_used['map_products_high_sim_threshold'] = high_sim_threshold

                # Get high similarity items
                high_sim_items = similarities_fuzzy.query('sim >= @high_sim_threshold')['item_nm_alias'].unique()
                trace_data.high_sim_items_list = list(high_sim_items)

                # Apply filtering
                filtered_similarities = similarities_fuzzy[
                    (similarities_fuzzy['item_nm_alias'].isin(high_sim_items)) &
                    (~similarities_fuzzy['item_nm_alias'].str.contains('test', case=False)) &
                    (~similarities_fuzzy['item_name_in_msg'].isin(recognizer.stop_item_names))
                ]
                trace_data.filtered_similarities_final = filtered_similarities.copy() if not filtered_similarities.empty else None

                if filtered_similarities.empty:
                    return []

                # Merge with item_pdf_all
                merged_items = recognizer.item_pdf_all.merge(filtered_similarities, on=['item_nm_alias'])
                trace_data.merged_with_item_pdf = merged_items.copy() if not merged_items.empty else None

            except Exception as e:
                logger.error(f"Traced map_products filtering capture failed: {e}")

            # Call original
            return original_map_products(similarities_fuzzy, json_objects)

        recognizer.map_products_to_entities = traced_map_products_to_entities

        # Patch assemble_result to capture alias_pdf_raw merge data from EntityMatchingStep
        original_assemble = builder.assemble_result

        def traced_assemble_result(json_objects, matched_products, msg, pgm_info, message_id='#'):
            """Capture merge with alias_pdf_raw and substring filtering."""
            from utils import replace_special_chars_with_space

            result = original_assemble(json_objects, matched_products, msg, pgm_info, message_id)

            # Try to capture alias_pdf_raw merge (if we have final_similarities)
            if trace_data.final_similarities is not None and not trace_data.final_similarities.empty:
                try:
                    # alias_pdf_raw is now on EntityMatchingStep, find it from the workflow
                    alias_pdf_raw = self.extractor.alias_pdf_raw
                    if alias_pdf_raw is not None and not alias_pdf_raw.empty:
                        merged_df = trace_data.final_similarities.merge(
                            alias_pdf_raw[['alias_1', 'type']].drop_duplicates(),
                            left_on='item_name_in_msg',
                            right_on='alias_1',
                            how='left'
                        )
                        trace_data.merged_with_alias_pdf = merged_df.copy()

                        # Apply substring filter
                        filtered_df = merged_df[merged_df.apply(
                            lambda x: (
                                replace_special_chars_with_space(str(x['item_nm_alias'])) in replace_special_chars_with_space(str(x['item_name_in_msg'])) or
                                replace_special_chars_with_space(str(x['item_name_in_msg'])) in replace_special_chars_with_space(str(x['item_nm_alias']))
                            ) if x.get('type') != 'expansion' else True,
                            axis=1
                        )]
                        trace_data.filtered_by_substring = filtered_df.copy()
                except Exception as e:
                    logger.debug(f"Could not capture alias_pdf_raw merge: {e}")

            return result

        builder.assemble_result = traced_assemble_result

    def trace_message(self, message: str, message_id: str = "#") -> TraceResult:
        """
        Run extraction and capture all intermediate data.

        Args:
            message: MMS message to process
            message_id: Optional message identifier

        Returns:
            TraceResult with per-step input/output snapshots
        """
        self._initialize_extractor()

        # Reset trace data
        self._entity_trace = EntityExtractionTraceData()

        # Apply patches for detailed tracing
        self._patch_entity_recognizer_for_tracing()
        self._patch_result_builder_for_tracing()

        # Prepare trace result
        trace = TraceResult(
            message=message,
            message_id=message_id,
            timestamp=datetime.now(),
            total_duration=0.0,
            extractor_config=self._get_extractor_config()
        )

        # Import workflow components
        from core.workflow_core import WorkflowState, WorkflowEngine
        from core.mms_workflow_steps import (
            InputValidationStep,
            EntityExtractionStep,
            ProgramClassificationStep,
            ContextPreparationStep,
            LLMExtractionStep,
            ResponseParsingStep,
            EntityMatchingStep,
            ResultConstructionStep,
            ValidationStep,
            DAGExtractionStep
        )
        from utils import PromptManager

        # Clear any stored prompts from previous runs
        PromptManager.clear_stored_prompts()

        # Create initial state
        initial_state = WorkflowState(
            mms_msg=message,
            extractor=self.extractor,
            message_id=message_id
        )

        # Define steps to trace
        steps = self.extractor.workflow_engine.steps

        # Run steps one by one with tracing
        state = initial_state
        total_start = time.time()

        for i, step in enumerate(steps, 1):
            step_name = step.name()
            step_start = time.time()

            # Capture input state (product-focused)
            input_data = self._capture_state_before_step(state, step_name)

            # Check should_execute (conditional step execution)
            if not step.should_execute(state):
                step_duration = time.time() - step_start
                step_trace = StepTrace(
                    step_name=step_name,
                    step_number=i,
                    duration_seconds=step_duration,
                    status="skipped",
                    input_data=input_data,
                    output_data={"skipped": True, "reason": "should_execute returned False"},
                )
                trace.step_traces.append(step_trace)
                state.add_history(step_name, step_duration, "skipped")
                continue

            # Execute step
            try:
                state = step.execute(state)
                status = "success" if not state.has_error() else "failed"
            except Exception as e:
                status = "failed"
                logger.error(f"Step {step_name} failed: {e}")
                import traceback
                traceback.print_exc()

            step_duration = time.time() - step_start

            # Capture output state (product-focused)
            output_data = self._capture_state_after_step(state, step_name)

            # Capture product-specific changes
            product_changes = self._capture_product_changes(state, step_name)

            # Capture sub-step timings for EntityExtractionStep and ResultConstructionStep
            substep_timings = {}
            if step_name in ("EntityExtractionStep", "EntityMatchingStep", "ResultConstructionStep") and self._entity_trace.substep_timings:
                substep_timings = dict(self._entity_trace.substep_timings)
                # Clear for next step so timings don't bleed across steps
                self._entity_trace.substep_timings = {}

            # Create step trace
            step_trace = StepTrace(
                step_name=step_name,
                step_number=i,
                duration_seconds=step_duration,
                status=status,
                input_data=input_data,
                output_data=output_data,
                product_changes=product_changes,
                substep_timings=substep_timings
            )
            trace.step_traces.append(step_trace)

            # Capture LLM prompt and response for Step 5
            if step_name == "LLMExtractionStep":
                stored_prompts = PromptManager.get_stored_prompts_from_thread()
                if 'main_extraction' in stored_prompts:
                    trace.llm_prompt = stored_prompts['main_extraction'].get('content', '')
                trace.llm_response = state.get("result_json_text", "")

            # Capture similarity scores for EntityMatchingStep
            if step_name == "EntityMatchingStep":
                trace.similarity_scores = self._entity_trace.final_similarities

            if state.has_error():
                break

        trace.total_duration = time.time() - total_start

        # Capture final products
        final_result = state.get("final_result", {})
        trace.final_products = final_result.get('product', [])

        return trace

    def _capture_state_before_step(self, state: 'WorkflowState', step_name: str) -> Dict[str, Any]:
        """Capture relevant input data before a step executes."""
        captured = {}

        if step_name == "InputValidationStep":
            captured['mms_msg'] = state.get("mms_msg", "")[:500]

        elif step_name == "EntityExtractionStep":
            captured['msg'] = state.get("msg", "")[:500]
            captured['context_mode'] = getattr(self.extractor, 'entity_extraction_context_mode', 'dag')

        elif step_name == "ProgramClassificationStep":
            captured['msg'] = state.get("msg", "")[:200]

        elif step_name == "ContextPreparationStep":
            pgm_info = state.get("pgm_info", {})
            captured['pgm_info_keys'] = list(pgm_info.keys()) if pgm_info else []
            cand_item_list = state.get("cand_item_list", [])
            captured['cand_item_list'] = cand_item_list[:20] if isinstance(cand_item_list, list) else []
            extra_item_pdf = state.get("extra_item_pdf", pd.DataFrame())
            if isinstance(extra_item_pdf, pd.DataFrame) and not extra_item_pdf.empty:
                captured['extra_item_pdf_shape'] = extra_item_pdf.shape

        elif step_name == "LLMExtractionStep":
            captured['msg'] = state.get("msg", "")[:200]
            captured['rag_context_length'] = len(state.get("rag_context", ""))
            captured['has_product_element'] = state.get("product_element") is not None

        elif step_name == "ResponseParsingStep":
            captured['result_json_text_length'] = len(state.get("result_json_text", ""))
            captured['result_json_text_preview'] = state.get("result_json_text", "")[:500]

        elif step_name == "EntityMatchingStep":
            json_objects = state.get("json_objects", {})
            captured['json_objects_product'] = json_objects.get('product', [])
            captured['entities_from_kiwi'] = state.get("entities_from_kiwi", [])
            captured['entity_extraction_mode'] = self.extractor.entity_extraction_mode
            captured['context_mode'] = getattr(self.extractor, 'entity_extraction_context_mode', 'dag')

        elif step_name == "ResultConstructionStep":
            captured['matched_products_count'] = len(state.matched_products)
            captured['matched_products'] = state.matched_products[:10]

        elif step_name == "ValidationStep":
            final_result = state.get("final_result", {})
            captured['pre_validation_product_count'] = len(final_result.get('product', []))

        elif step_name == "DAGExtractionStep":
            captured['has_final_result'] = state.get("final_result") is not None

        return captured

    def _capture_state_after_step(self, state: 'WorkflowState', step_name: str) -> Dict[str, Any]:
        """Capture relevant output data after a step executes."""
        captured = {}

        if step_name == "InputValidationStep":
            captured['msg'] = state.get("msg", "")[:500]
            captured['is_fallback'] = state.get("is_fallback", False)

        elif step_name == "EntityExtractionStep":
            captured['entities_from_kiwi'] = state.get("entities_from_kiwi", [])
            cand_item_list = state.get("cand_item_list", [])
            captured['cand_item_list'] = cand_item_list[:30] if isinstance(cand_item_list, list) else []
            captured['cand_item_count'] = len(cand_item_list) if isinstance(cand_item_list, list) else 0
            extra_item_pdf = state.get("extra_item_pdf", pd.DataFrame())
            if isinstance(extra_item_pdf, pd.DataFrame) and not extra_item_pdf.empty:
                captured['extra_item_pdf_shape'] = extra_item_pdf.shape
                captured['extra_item_pdf_sample'] = _dataframe_to_serializable(extra_item_pdf, 5)

            # ONT mode specific data
            ont_result = state.get("ont_extraction_result")
            if ont_result:
                captured['ont_entity_types'] = ont_result.get('entity_types', {})
                captured['ont_relationships'] = ont_result.get('relationships', [])
                captured['ont_dag_text'] = ont_result.get('dag_text', '')[:500]

        elif step_name == "ProgramClassificationStep":
            pgm_info = state.get("pgm_info", {})
            captured['pgm_info_keys'] = list(pgm_info.keys()) if pgm_info else []
            captured['pgm_cand_info'] = pgm_info.get('pgm_cand_info', '')[:300]

        elif step_name == "ContextPreparationStep":
            captured['rag_context_length'] = len(state.get("rag_context", ""))
            captured['rag_context_preview'] = state.get("rag_context", "")[:500]
            captured['has_product_element'] = state.get("product_element") is not None
            product_element = state.get("product_element")
            if product_element:
                captured['product_element_count'] = len(product_element)
                captured['product_element_sample'] = product_element[:3]

        elif step_name == "LLMExtractionStep":
            captured['result_json_text_length'] = len(state.get("result_json_text", ""))
            captured['result_json_text_preview'] = state.get("result_json_text", "")[:500]

        elif step_name == "ResponseParsingStep":
            json_objects = state.get("json_objects", {})
            captured['json_objects_keys'] = list(json_objects.keys())
            captured['json_objects_product'] = json_objects.get('product', [])
            captured['is_fallback'] = state.get("is_fallback", False)

        elif step_name == "EntityMatchingStep":
            captured['matched_products'] = state.matched_products
            captured['matched_products_count'] = len(state.matched_products)

            # ENHANCED: Capture entity extraction trace data
            if self._entity_trace.first_stage_prompt:
                captured['entity_extraction_prompt'] = self._entity_trace.first_stage_prompt[:1000]
            if self._entity_trace.entities_after_ngram:
                captured['candidate_entities_for_matching'] = self._entity_trace.entities_after_ngram[:50]
            if self._entity_trace.final_similarities is not None and not self._entity_trace.final_similarities.empty:
                captured['similarity_scores_count'] = len(self._entity_trace.final_similarities)
                captured['similarity_scores_sample'] = _dataframe_to_serializable(
                    self._entity_trace.final_similarities, 20
                )

        elif step_name == "ResultConstructionStep":
            final_result = state.get("final_result", {})
            captured['final_result_product'] = final_result.get('product', [])
            captured['final_result_product_count'] = len(final_result.get('product', []))
            captured['offer_type'] = final_result.get('offer', {}).get('type', 'N/A')

        elif step_name == "ValidationStep":
            final_result = state.get("final_result", {})
            captured['validated_product_count'] = len(final_result.get('product', []))
            captured['has_errors'] = state.has_error()

        elif step_name == "DAGExtractionStep":
            final_result = state.get("final_result", {})
            captured['entity_dag'] = final_result.get('entity_dag', [])
            captured['entity_dag_count'] = len(final_result.get('entity_dag', []))

        return captured

    def _capture_product_changes(self, state: 'WorkflowState', step_name: str) -> Dict[str, Any]:
        """Capture product-specific changes and details."""
        changes = {}

        if step_name == "EntityExtractionStep":
            entities = state.get("entities_from_kiwi", [])
            cand_list = state.get("cand_item_list", [])
            changes['nnp_tokens_extracted'] = len(entities)
            changes['candidate_products_matched'] = len(cand_list) if isinstance(cand_list, list) else 0

            # ONT mode extra data
            ont_result = state.get("ont_extraction_result")
            if ont_result:
                changes['ont_mode_detected'] = True
                changes['ont_entities_count'] = len(ont_result.get('entity_types', {}))
                changes['ont_relationships_count'] = len(ont_result.get('relationships', []))

        elif step_name == "ContextPreparationStep":
            product_element = state.get("product_element")
            if product_element:
                changes['nlp_mode_products'] = len(product_element)

        elif step_name == "ResponseParsingStep":
            json_objects = state.get("json_objects", {})
            products = json_objects.get('product', [])
            changes['llm_extracted_products'] = len(products) if isinstance(products, list) else 0
            if products and isinstance(products, list):
                changes['llm_product_names'] = [p.get('name', '') for p in products if isinstance(p, dict)]
                changes['llm_product_actions'] = [p.get('action', '') for p in products if isinstance(p, dict)]

        elif step_name == "EntityMatchingStep":
            # This is the key step where product matching happens
            products = state.matched_products
            changes['matched_product_count'] = len(products)

            if products:
                changes['product_details'] = []
                for p in products:
                    changes['product_details'].append({
                        'item_nm': p.get('item_nm', ''),
                        'item_id': p.get('item_id', []),
                        'item_name_in_msg': p.get('item_name_in_msg', []),
                        'expected_action': p.get('expected_action', [])
                    })

            # ENHANCED: Show similarity matching results
            if self._entity_trace.final_similarities is not None and not self._entity_trace.final_similarities.empty:
                df = self._entity_trace.final_similarities
                changes['similarity_matching'] = {
                    'total_matches': len(df),
                    'unique_entities_in_msg': df['item_name_in_msg'].nunique() if 'item_name_in_msg' in df.columns else 0,
                    'unique_aliases_matched': df['item_nm_alias'].nunique() if 'item_nm_alias' in df.columns else 0,
                }
                # Show top matches
                if 'sim' in df.columns:
                    top_matches = df.nlargest(10, 'sim')[['item_name_in_msg', 'item_nm_alias', 'sim']].to_dict('records')
                    changes['top_similarity_matches'] = top_matches

        return changes

    def generate_report(self, trace: TraceResult, output_format: str = "text") -> str:
        """
        Generate human-readable trace report.

        Args:
            trace: TraceResult to format
            output_format: 'text', 'json', or 'markdown'

        Returns:
            Formatted report string
        """
        if output_format == "json":
            return generate_json_report(trace)
        elif output_format == "markdown":
            return generate_markdown_report(trace)
        else:
            return self._generate_enhanced_text_report(trace)

    def _generate_enhanced_text_report(self, trace: TraceResult) -> str:
        """Generate enhanced text report with entity extraction details."""
        lines = []

        # Header
        lines.append("=" * 80)
        lines.append("PRODUCT EXTRACTION TRACE REPORT (ENHANCED)")
        lines.append("=" * 80)
        lines.append(f"Message ID: {trace.message_id}")
        lines.append(f"Message: \"{trace.message[:100]}...\"" if len(trace.message) > 100 else f"Message: \"{trace.message}\"")
        lines.append(f"Timestamp: {trace.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Total Duration: {trace.total_duration:.2f} seconds")
        lines.append("")

        # Configuration (CRITICAL for understanding differences)
        lines.append("-" * 80)
        lines.append("EXTRACTOR CONFIGURATION (KEY FOR DEBUGGING)")
        lines.append("-" * 80)
        for key, value in trace.extractor_config.items():
            if key == 'entity_extraction_context_mode':
                lines.append(f"  >>> {key}: {value} <<<  (THIS AFFECTS ENTITY EXTRACTION)")
            else:
                lines.append(f"  {key}: {value}")
        lines.append("")

        # Step-by-step traces
        for step_trace in trace.step_traces:
            status_icon = {"success": "OK", "skipped": "SKIP", "failed": "FAIL"}.get(step_trace.status, "FAIL")
            lines.append("-" * 80)
            lines.append(f"STEP {step_trace.step_number}: {step_trace.step_name} "
                        f"({step_trace.duration_seconds:.2f}s) [{status_icon}]")
            lines.append("-" * 80)

            # Sub-step timings (if available)
            if step_trace.substep_timings:
                lines.append("SUB-STEP TIMINGS:")
                total_substep = sum(step_trace.substep_timings.values())
                for substep_name, substep_duration in step_trace.substep_timings.items():
                    pct = (substep_duration / step_trace.duration_seconds * 100) if step_trace.duration_seconds > 0 else 0
                    lines.append(f"  {substep_name}: {substep_duration:.3f}s ({pct:.1f}%)")
                overhead = step_trace.duration_seconds - total_substep
                if overhead > 0.01:
                    pct = (overhead / step_trace.duration_seconds * 100) if step_trace.duration_seconds > 0 else 0
                    lines.append(f"  (overhead/other): {overhead:.3f}s ({pct:.1f}%)")
                lines.append("")

            # Input
            if step_trace.input_data:
                lines.append("INPUT:")
                for key, value in step_trace.input_data.items():
                    if isinstance(value, pd.DataFrame):
                        lines.append(f"  {key}: DataFrame(shape={value.shape})")
                    elif isinstance(value, list) and len(str(value)) > 200:
                        lines.append(f"  {key}: [{len(value)} items]")
                    elif isinstance(value, dict) and len(str(value)) > 200:
                        lines.append(f"  {key}: {{...{len(value)} keys...}}")
                    else:
                        lines.append(f"  {key}: {value}")
                lines.append("")

            # Output
            if step_trace.output_data:
                lines.append("OUTPUT:")
                for key, value in step_trace.output_data.items():
                    if isinstance(value, pd.DataFrame):
                        lines.append(f"  {key}: DataFrame(shape={value.shape})")
                    elif key == 'similarity_scores_sample' and isinstance(value, list):
                        lines.append(f"  {key}:")
                        for item in value[:5]:
                            lines.append(f"    - {item}")
                        if len(value) > 5:
                            lines.append(f"    ... ({len(value) - 5} more)")
                    elif key == 'entity_extraction_prompt':
                        lines.append(f"  {key}: (length={len(value)})")
                        lines.append(f"    Preview: {value[:300]}...")
                    elif key in ['ont_entity_types', 'ont_relationships']:
                        lines.append(f"  {key}: {json.dumps(value, ensure_ascii=False, indent=4)[:500]}")
                    elif isinstance(value, list) and len(str(value)) > 300:
                        lines.append(f"  {key}: [{len(value)} items]")
                        if value:
                            lines.append(f"    First 3: {value[:3]}")
                    elif isinstance(value, str) and len(value) > 300:
                        lines.append(f"  {key}: {value[:300]}...")
                    else:
                        lines.append(f"  {key}: {value}")
                lines.append("")

            # Product changes (IMPORTANT)
            if step_trace.product_changes:
                lines.append("PRODUCT FOCUS:")
                for key, value in step_trace.product_changes.items():
                    if key == 'top_similarity_matches':
                        lines.append(f"  {key}:")
                        for match in value:
                            lines.append(f"    - {match}")
                    elif key == 'similarity_matching':
                        lines.append(f"  {key}:")
                        for k, v in value.items():
                            lines.append(f"    - {k}: {v}")
                    elif key == 'product_details':
                        lines.append(f"  {key}:")
                        for detail in value:
                            lines.append(f"    - item_nm: {detail.get('item_nm')}")
                            lines.append(f"      item_id: {detail.get('item_id')}")
                            lines.append(f"      item_name_in_msg: {detail.get('item_name_in_msg')}")
                            lines.append(f"      expected_action: {detail.get('expected_action')}")
                    elif isinstance(value, dict):
                        lines.append(f"  {key}: {json.dumps(value, ensure_ascii=False)}")
                    else:
                        lines.append(f"  {key}: {value}")
                lines.append("")

        # Similarity Scores Detail
        if trace.similarity_scores is not None and not trace.similarity_scores.empty:
            lines.append("=" * 80)
            lines.append("SIMILARITY SCORES DETAIL (Step 7)")
            lines.append("=" * 80)
            lines.append(f"Total rows: {len(trace.similarity_scores)}")
            lines.append(f"Columns: {list(trace.similarity_scores.columns)}")
            lines.append("")
            lines.append("Sample (top 15 by similarity):")
            if 'sim' in trace.similarity_scores.columns:
                sample_df = trace.similarity_scores.nlargest(15, 'sim')
            else:
                sample_df = trace.similarity_scores.head(15)
            lines.append(sample_df.to_string())
            lines.append("")

        # ENHANCED: Detailed Filtering Analysis
        lines.append("=" * 80)
        lines.append("DETAILED FILTERING ANALYSIS (WHY PRODUCTS ARE FILTERED)")
        lines.append("=" * 80)
        lines.append("")

        # Show thresholds used
        if self._entity_trace.thresholds_used:
            lines.append("-" * 60)
            lines.append("THRESHOLDS USED:")
            lines.append("-" * 60)
            for threshold_name, threshold_value in self._entity_trace.thresholds_used.items():
                lines.append(f"  {threshold_name}: {threshold_value}")
            lines.append("")

        # Filter Stage 1: Fuzzy similarity
        if self._entity_trace.similarities_fuzzy_raw is not None:
            lines.append("-" * 60)
            lines.append("STAGE 1: FUZZY SIMILARITY (entity_llm_fuzzy_threshold)")
            lines.append("-" * 60)
            df = self._entity_trace.similarities_fuzzy_raw
            lines.append(f"  Rows after fuzzy matching: {len(df)}")
            lines.append(f"  Unique item_name_in_msg: {df['item_name_in_msg'].nunique()}")
            lines.append(f"  Unique item_nm_alias: {df['item_nm_alias'].nunique()}")
            if 'sim' in df.columns:
                lines.append(f"  Similarity range: {df['sim'].min():.4f} - {df['sim'].max():.4f}")
            lines.append("")

        # Filter Stage 2: Before combined threshold
        if self._entity_trace.similarities_before_combined_filter is not None:
            lines.append("-" * 60)
            lines.append("STAGE 2: SEQUENCE SIMILARITY (before combined_similarity_threshold)")
            lines.append("-" * 60)
            df = self._entity_trace.similarities_before_combined_filter
            lines.append(f"  Rows: {len(df)}")
            if 'sim_s1' in df.columns and 'sim_s2' in df.columns:
                lines.append(f"  sim_s1 range: {df['sim_s1'].min():.4f} - {df['sim_s1'].max():.4f}")
                lines.append(f"  sim_s2 range: {df['sim_s2'].min():.4f} - {df['sim_s2'].max():.4f}")
            lines.append("")

        # Filter Stage 3: After combined threshold
        if self._entity_trace.similarities_after_combined_filter is not None:
            lines.append("-" * 60)
            combined_thresh = self._entity_trace.thresholds_used.get('combined_similarity_threshold', 0.2)
            lines.append(f"STAGE 3: AFTER COMBINED THRESHOLD (sim_s1>={combined_thresh} AND sim_s2>={combined_thresh})")
            lines.append("-" * 60)
            df = self._entity_trace.similarities_after_combined_filter
            lines.append(f"  Rows remaining: {len(df)}")

            # Show what was filtered out
            if self._entity_trace.similarities_before_combined_filter is not None:
                before_df = self._entity_trace.similarities_before_combined_filter
                filtered_out = before_df[
                    ~before_df.apply(lambda x: (x['item_name_in_msg'], x['item_nm_alias']) in
                        set(zip(df['item_name_in_msg'], df['item_nm_alias'])), axis=1)
                ]
                if not filtered_out.empty:
                    lines.append(f"  Filtered OUT at this stage: {len(filtered_out)} rows")
                    lines.append("  Sample of filtered items:")
                    for _, row in filtered_out.head(5).iterrows():
                        lines.append(f"    - {row['item_name_in_msg']} -> {row['item_nm_alias']} "
                                   f"(s1={row['sim_s1']:.4f}, s2={row['sim_s2']:.4f})")
            lines.append("")

        # Filter Stage 4: Before high similarity threshold
        if self._entity_trace.similarities_before_high_sim_filter is not None:
            lines.append("-" * 60)
            lines.append("STAGE 4: COMBINED SCORE (sim = sim_s1 + sim_s2)")
            lines.append("-" * 60)
            df = self._entity_trace.similarities_before_high_sim_filter
            lines.append(f"  Rows: {len(df)}")
            if 'sim' in df.columns:
                lines.append(f"  Combined sim range: {df['sim'].min():.4f} - {df['sim'].max():.4f}")
                lines.append("  All items with combined scores:")
                for _, row in df.sort_values('sim', ascending=False).iterrows():
                    lines.append(f"    - {row['item_name_in_msg']} -> {row['item_nm_alias']}: sim={row['sim']:.6f}")
            lines.append("")

        # Filter Stage 5: After high similarity threshold (CRITICAL)
        if self._entity_trace.similarities_after_high_sim_filter is not None:
            lines.append("-" * 60)
            high_thresh = self._entity_trace.thresholds_used.get('high_similarity_threshold', 1.0)
            lines.append(f"STAGE 5: AFTER HIGH SIMILARITY THRESHOLD (sim>={high_thresh}) *** CRITICAL ***")
            lines.append("-" * 60)
            df = self._entity_trace.similarities_after_high_sim_filter
            lines.append(f"  Rows remaining: {len(df)}")

            # Show what was filtered out at this critical stage
            if self._entity_trace.similarities_before_high_sim_filter is not None:
                before_df = self._entity_trace.similarities_before_high_sim_filter
                # Items that passed
                passed_items = set(zip(df['item_name_in_msg'], df['item_nm_alias']))
                filtered_out = before_df[
                    ~before_df.apply(lambda x: (x['item_name_in_msg'], x['item_nm_alias']) in passed_items, axis=1)
                ]
                if not filtered_out.empty:
                    lines.append(f"")
                    lines.append(f"  *** FILTERED OUT AT HIGH_SIMILARITY_THRESHOLD ({high_thresh}): ***")
                    lines.append(f"  {len(filtered_out)} items filtered out:")
                    for _, row in filtered_out.sort_values('sim', ascending=False).iterrows():
                        status = "PASSED" if row['sim'] >= high_thresh else f"FAILED (need {high_thresh}, got {row['sim']:.6f})"
                        lines.append(f"    - {row['item_name_in_msg']} -> {row['item_nm_alias']}: sim={row['sim']:.6f} -> {status}")
                lines.append("")

                if not df.empty:
                    lines.append(f"  *** ITEMS THAT PASSED (sim>={high_thresh}): ***")
                    for _, row in df.sort_values('sim', ascending=False).iterrows():
                        lines.append(f"    - {row['item_name_in_msg']} -> {row['item_nm_alias']}: sim={row['sim']:.6f}")
            lines.append("")

        # 2nd Stage LLM Filtering (CRITICAL for DAG vs ONT difference)
        if self._entity_trace.second_stage_context_mode:
            lines.append("-" * 60)
            lines.append("STAGE 5.5: 2nd STAGE LLM FILTERING *** CRITICAL FOR DAG vs ONT ***")
            lines.append("-" * 60)
            lines.append(f"  Context Mode: {self._entity_trace.second_stage_context_mode}")
            lines.append("")

            # Show entities in message
            if self._entity_trace.second_stage_entities_in_message:
                lines.append("  Entities in Message (sent to 2nd stage):")
                for ent in self._entity_trace.second_stage_entities_in_message:
                    lines.append(f"    - {ent}")
                lines.append("")

            # Show candidate aliases
            if self._entity_trace.second_stage_candidate_aliases:
                lines.append(f"  Candidate Aliases (vocab sent to 2nd stage): {len(self._entity_trace.second_stage_candidate_aliases)}")
                for alias in self._entity_trace.second_stage_candidate_aliases[:20]:
                    lines.append(f"    - {alias}")
                if len(self._entity_trace.second_stage_candidate_aliases) > 20:
                    lines.append(f"    ... ({len(self._entity_trace.second_stage_candidate_aliases) - 20} more)")
                lines.append("")

            # Show context text (truncated)
            if self._entity_trace.second_stage_context_text:
                lines.append("  Context Text (passed to 2nd stage LLM):")
                context_lines = self._entity_trace.second_stage_context_text.split('\n')
                for ctx_line in context_lines[:30]:
                    lines.append(f"    {ctx_line}")
                if len(context_lines) > 30:
                    lines.append(f"    ... ({len(context_lines) - 30} more lines)")
                lines.append("")

            # Show 2nd stage prompts (truncated)
            if self._entity_trace.second_stage_prompts:
                lines.append(f"  2nd Stage Prompts: {len(self._entity_trace.second_stage_prompts)} calls")
                for i, prompt in enumerate(self._entity_trace.second_stage_prompts[:2], 1):
                    lines.append(f"  --- Prompt {i} (truncated) ---")
                    prompt_preview = prompt[:1500] if len(prompt) > 1500 else prompt
                    for p_line in prompt_preview.split('\n'):
                        lines.append(f"    {p_line}")
                    if len(prompt) > 1500:
                        lines.append(f"    ... ({len(prompt) - 1500} more chars)")
                if len(self._entity_trace.second_stage_prompts) > 2:
                    lines.append(f"  ... ({len(self._entity_trace.second_stage_prompts) - 2} more prompts)")
                lines.append("")

            # Show 2nd stage responses
            if self._entity_trace.second_stage_responses:
                lines.append(f"  2nd Stage Responses: {len(self._entity_trace.second_stage_responses)} responses")
                for i, resp in enumerate(self._entity_trace.second_stage_responses, 1):
                    lines.append(f"  --- Response {i} ---")
                    lines.append(f"    {resp}")
                lines.append("")

            # Show confirmed entities
            if self._entity_trace.second_stage_confirmed_entities:
                lines.append(f"  *** CONFIRMED BY 2nd STAGE LLM: {len(self._entity_trace.second_stage_confirmed_entities)} entities ***")
                for ent in self._entity_trace.second_stage_confirmed_entities:
                    lines.append(f"    - {ent}")
                lines.append("")

            # Show before/after 2nd stage filtering comparison
            before_2nd = self._entity_trace.similarities_before_2nd_stage
            after_2nd = self._entity_trace.similarities_after_2nd_stage
            if before_2nd is not None and after_2nd is not None:
                lines.append("  *** 2nd STAGE FILTERING EFFECT: ***")
                lines.append(f"    Before 2nd stage: {len(before_2nd)} rows")
                lines.append(f"    After 2nd stage:  {len(after_2nd)} rows")
                lines.append(f"    Filtered out:     {len(before_2nd) - len(after_2nd)} rows")
                lines.append("")

                # Show what was filtered out
                if len(before_2nd) > len(after_2nd):
                    after_aliases = set(after_2nd['item_nm_alias'].unique()) if 'item_nm_alias' in after_2nd.columns else set()
                    filtered_out = before_2nd[~before_2nd['item_nm_alias'].isin(after_aliases)]
                    if not filtered_out.empty:
                        lines.append("    *** FILTERED OUT BY 2nd STAGE LLM: ***")
                        for _, row in filtered_out.sort_values('sim', ascending=False).iterrows():
                            lines.append(f"      - {row['item_name_in_msg']} -> {row['item_nm_alias']}: sim={row['sim']:.6f}")
                        lines.append("")

                # Show what passed
                if not after_2nd.empty and 'item_nm_alias' in after_2nd.columns:
                    lines.append("    *** PASSED 2nd STAGE LLM FILTER: ***")
                    for _, row in after_2nd.sort_values('sim', ascending=False).iterrows():
                        lines.append(f"      - {row['item_name_in_msg']} -> {row['item_nm_alias']}: sim={row['sim']:.6f}")
            lines.append("")

        # map_products_to_entities filtering
        if self._entity_trace.map_products_input is not None:
            lines.append("-" * 60)
            lines.append("STAGE 6: map_products_to_entities INPUT")
            lines.append("-" * 60)
            df = self._entity_trace.map_products_input
            lines.append(f"  Input rows: {len(df)}")
            lines.append(f"  Unique aliases: {df['item_nm_alias'].nunique()}")
            lines.append("")

        if self._entity_trace.high_sim_items_list:
            lines.append("-" * 60)
            lines.append("STAGE 7: HIGH SIMILARITY ITEMS (for final filtering)")
            lines.append("-" * 60)
            lines.append(f"  Items passing high_sim_threshold: {len(self._entity_trace.high_sim_items_list)}")
            for item in self._entity_trace.high_sim_items_list[:20]:
                lines.append(f"    - {item}")
            if len(self._entity_trace.high_sim_items_list) > 20:
                lines.append(f"    ... ({len(self._entity_trace.high_sim_items_list) - 20} more)")
            lines.append("")

        if self._entity_trace.filtered_similarities_final is not None:
            lines.append("-" * 60)
            lines.append("STAGE 8: AFTER FINAL FILTERING (stop_items, 'test' removed)")
            lines.append("-" * 60)
            df = self._entity_trace.filtered_similarities_final
            lines.append(f"  Rows remaining: {len(df)}")
            lines.append("")

        if self._entity_trace.merged_with_item_pdf is not None:
            lines.append("-" * 60)
            lines.append("STAGE 9: AFTER MERGE WITH item_pdf_all")
            lines.append("-" * 60)
            df = self._entity_trace.merged_with_item_pdf
            lines.append(f"  Rows after merge: {len(df)}")
            if 'item_nm' in df.columns:
                lines.append(f"  Unique item_nm: {df['item_nm'].nunique()}")
                lines.append("  Final product names:")
                for item_nm in df['item_nm'].unique()[:10]:
                    lines.append(f"    - {item_nm}")
            lines.append("")

        # Alias PDF merge analysis (result_builder)
        if self._entity_trace.merged_with_alias_pdf is not None:
            lines.append("-" * 60)
            lines.append("STAGE 10: MERGE WITH alias_pdf_raw (result_builder)")
            lines.append("-" * 60)
            df = self._entity_trace.merged_with_alias_pdf
            lines.append(f"  Rows after merge: {len(df)}")
            if 'type' in df.columns:
                lines.append(f"  Type distribution: {df['type'].value_counts().to_dict()}")
            lines.append("")

        if self._entity_trace.filtered_by_substring is not None:
            lines.append("-" * 60)
            lines.append("STAGE 11: AFTER SUBSTRING FILTER (non-expansion types)")
            lines.append("-" * 60)
            df = self._entity_trace.filtered_by_substring
            lines.append(f"  Rows remaining: {len(df)}")

            # Show what was filtered
            if self._entity_trace.merged_with_alias_pdf is not None:
                before_df = self._entity_trace.merged_with_alias_pdf
                if len(before_df) > len(df):
                    lines.append(f"  Filtered out: {len(before_df) - len(df)} rows")
            lines.append("")

        lines.append("=" * 80)

        # LLM Response (Step 5)
        if trace.llm_response:
            lines.append("=" * 80)
            lines.append("LLM RESPONSE (Step 5 - Main Extraction)")
            lines.append("=" * 80)
            lines.append(trace.llm_response[:2000])
            if len(trace.llm_response) > 2000:
                lines.append(f"... ({len(trace.llm_response) - 2000} more chars)")
            lines.append("")

        # Final Products
        lines.append("=" * 80)
        lines.append(f"FINAL PRODUCTS: {len(trace.final_products)}")
        lines.append("=" * 80)

        for i, product in enumerate(trace.final_products, 1):
            lines.append(f"{i}. {product.get('item_nm', 'Unknown')}")
            lines.append(f"   - item_id: {product.get('item_id', [])}")
            lines.append(f"   - item_name_in_msg: {product.get('item_name_in_msg', [])}")
            lines.append(f"   - expected_action: {product.get('expected_action', [])}")

        if not trace.final_products:
            lines.append("  (No products extracted)")

        lines.append("")
        lines.append("=" * 80)

        return "\n".join(lines)

    def analyze_with_llm(self, trace: TraceResult, llm_model: str = "cld") -> str:
        """
        Optional: Send trace to LLM for analysis (Option B).
        """
        self._initialize_extractor()

        summary = {
            'message': trace.message,
            'message_id': trace.message_id,
            'total_duration': trace.total_duration,
            'config': trace.extractor_config,
            'steps': []
        }

        for step_trace in trace.step_traces:
            summary['steps'].append({
                'step': f"{step_trace.step_number}. {step_trace.step_name}",
                'status': step_trace.status,
                'duration': step_trace.duration_seconds,
                'product_changes': step_trace.product_changes
            })

        summary['final_products'] = trace.final_products

        analysis_prompt = f"""
You are analyzing a product extraction trace from an MMS message processing system.

## Trace Summary
```json
{json.dumps(summary, ensure_ascii=False, indent=2)}
```

## LLM Prompt Used (Step 5)
{trace.llm_prompt[:2000] if trace.llm_prompt else "(Not captured)"}

## LLM Response (Step 5)
{trace.llm_response[:2000] if trace.llm_response else "(Not captured)"}

## Analysis Questions
1. Why were these specific products extracted from the message?
2. At which step did the key product identification happen?
3. Were there any products that might have been missed or incorrectly identified?
4. What was the similarity matching process and were the thresholds appropriate?
5. Are there any issues or improvements you would suggest?

Please provide a detailed analysis in Korean, focusing on the product extraction logic.
"""

        from utils.llm_factory import LLMFactory
        factory = LLMFactory()
        models = factory.create_models([llm_model])

        if not models:
            return "Error: Could not initialize LLM for analysis"

        try:
            response = models[0].invoke(analysis_prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            return f"Error during LLM analysis: {e}"


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Product Extraction Tracing Tool (Enhanced)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic tracing
  python tests/trace_product_extraction.py --message "아이폰 17 구매하세요"

  # With DAG context mode
  python tests/trace_product_extraction.py --message "메시지" --context-mode dag

  # With ONT context mode (compare with DAG)
  python tests/trace_product_extraction.py --message "메시지" --context-mode ont

  # JSON output for detailed analysis
  python tests/trace_product_extraction.py --message "메시지" --output-format json

  # Save to file
  python tests/trace_product_extraction.py --message "메시지" --output-file trace.txt
        """
    )

    parser.add_argument("--message", "-m", type=str, required=True, help="MMS message to trace")
    parser.add_argument("--message-id", type=str, default="#", help="Message identifier")
    parser.add_argument("--output-format", "-f", choices=["text", "json", "markdown"], default="text")
    parser.add_argument("--output-file", "-o", type=str, help="Save output to file")
    parser.add_argument("--save-to", type=str, default="outputs/", help="Directory to save trace results (default: outputs/)")
    parser.add_argument("--analyze-with-llm", action="store_true", help="Perform LLM analysis")
    parser.add_argument("--analysis-model", type=str, default="cld", help="LLM model for analysis")
    parser.add_argument("--llm-model", type=str, default="ax", help="LLM model for extraction")
    parser.add_argument("--entity-mode", type=str, choices=["llm", "logic", "nlp"], default="llm")
    parser.add_argument("--data-source", type=str, choices=["local", "db"], default="local")
    parser.add_argument("--extract-dag", action="store_true", help="Enable DAG extraction")
    parser.add_argument("--context-mode", type=str, choices=["dag", "pairing", "none", "ont", "typed"], default="dag",
                       help="Entity extraction context mode (CRITICAL: dag vs ont vs typed produce different results)")
    parser.add_argument("--skip-entity-extraction", action="store_true",
                       help="Skip Kiwi + fuzzy matching entity pre-extraction (Step 2)")
    parser.add_argument("--no-external-candidates", action="store_true",
                       help="Disable external candidate injection in Step 7 matching")

    args = parser.parse_args()

    extractor_kwargs = {
        'llm_model': args.llm_model,
        'entity_extraction_mode': args.entity_mode,
        'offer_info_data_src': args.data_source,
        'extract_entity_dag': args.extract_dag,
        'entity_extraction_context_mode': args.context_mode,
        'skip_entity_extraction': args.skip_entity_extraction,
        'use_external_candidates': not args.no_external_candidates,
    }

    print("=" * 60)
    print("Product Extraction Tracing Tool (Enhanced)")
    print("=" * 60)
    print(f"Message: \"{args.message[:50]}...\"" if len(args.message) > 50 else f"Message: \"{args.message}\"")
    print(f"Message ID: {args.message_id}")
    print(f"Context Mode: {args.context_mode} (CRITICAL FOR ENTITY EXTRACTION)")
    print(f"Output Format: {args.output_format}")
    print("=" * 60)

    tracer = ProductExtractionTracer(extractor_kwargs)

    print("\nTracing extraction process...")
    trace = tracer.trace_message(args.message, args.message_id)
    print(f"Trace complete. Duration: {trace.total_duration:.2f}s")

    report = tracer.generate_report(trace, args.output_format)

    if args.output_file:
        # Insert timestamp before extension: trace.txt -> trace_20260209_123456.txt
        base, ext = os.path.splitext(args.output_file)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{base}_{timestamp}{ext}"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\nReport saved to: {output_path}")
    elif args.save_to:
        os.makedirs(args.save_to, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = {'text': 'txt', 'json': 'json', 'markdown': 'md'}[args.output_format]
        filepath = os.path.join(args.save_to, f"trace_{args.context_mode}_{args.llm_model}_{args.message_id}_{timestamp}.{ext}")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\nReport saved to: {filepath}")
    else:
        print("\n" + report)

    if args.analyze_with_llm:
        print("\n" + "=" * 60)
        print("LLM Analysis")
        print("=" * 60)
        analysis = tracer.analyze_with_llm(trace, args.analysis_model)
        print(analysis)


if __name__ == "__main__":
    main()
