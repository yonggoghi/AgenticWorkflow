"""
MMS Extractor - Entity Recognizer Service
==========================================

📋 개요
-------
이 서비스는 MMS 메시지에서 상품/서비스 엔티티를 추출하고 데이터베이스와 매칭하는
핵심 로직을 담당합니다. MMSExtractor로부터 분리되어 독립적으로 테스트 및 재사용 가능합니다.

🔗 의존성
---------
**사용하는 모듈:**
- `utils`: 유사도 계산 (parallel_fuzzy_similarity, parallel_seq_similarity)
- `prompts`: 엔티티 추출 프롬프트 템플릿
- `config.settings`: 임계값 및 처리 설정 (PROCESSING_CONFIG)
- `Kiwi`: 한국어 형태소 분석
- `LangChain`: LLM 모델 인터페이스

**사용되는 곳:**
- `core.mms_workflow_steps.EntityExtractionStep`: 워크플로우 단계에서 사용
- `core.mms_extractor`: MMSExtractor 초기화 시 생성

🏗️ 엔티티 추출 모드 비교
------------------------

| 모드 | 방법 | 속도 | 정확도 | 사용 시나리오 |
|------|------|------|--------|--------------|
| **Kiwi** | 형태소 분석 + Fuzzy/Sequence 매칭 | 빠름 | 중간 | 명확한 상품명, 빠른 처리 필요 |
| **Logic** | Fuzzy + Sequence 유사도 조합 | 중간 | 중간 | 후보 엔티티 목록이 있을 때 |
| **LLM** | 2단계 LLM 추출 + 필터링 | 느림 | 높음 | 복잡한 문맥, 높은 정확도 필요 |

### LLM 모드 상세 흐름
```
1단계: 초기 추출
  ├─ DAG/PAIRING/SIMPLE 프롬프트 선택 (context_mode)
  ├─ 멀티모델 병렬 실행 (최대 3개)
  └─ 엔티티 + 컨텍스트 추출

2단계: 정제 및 필터링
  ├─ N-gram 확장
  ├─ 상품 DB와 Fuzzy/Sequence 매칭
  ├─ 배치 단위로 LLM 재검증
  └─ 최종 엔티티 목록 반환
```

🏗️ 주요 컴포넌트
----------------
- **EntityRecognizer**: 엔티티 추출 및 매칭 서비스 클래스
  - `extract_entities_hybrid()`: 하이브리드 추출 (Kiwi + Fuzzy + Sequence)
  - `extract_entities_with_fuzzy_matching()`: 퍼지 매칭 기반 추출
  - `extract_entities_with_llm()`: LLM 기반 추출 (2단계)
  - `map_products_with_similarity()`: 유사도 기반 상품 매핑

💡 사용 예시
-----------
```python
from services.entity_recognizer import EntityRecognizer

# 초기화
recognizer = EntityRecognizer(
    kiwi=kiwi_instance,
    item_pdf_all=product_dataframe,
    stop_item_names=['광고', '이벤트'],
    llm_model=llm_instance,
    entity_extraction_mode='llm'
)

# Kiwi 기반 추출
entities, candidates, extra_df = recognizer.extract_entities_hybrid(
    "아이폰 17 구매 시 캐시백 제공"
)

# LLM 기반 추출 (DAG 컨텍스트 모드)
similarity_df = recognizer.extract_entities_with_llm(
    msg_text="아이폰 17 구매 시 캐시백 제공",
    rank_limit=50,
    llm_models=[llm1, llm2],
    context_mode='dag'
)

# 상품 매핑
products = recognizer.map_products_with_similarity(
    similarities_fuzzy=similarity_df,
    json_objects=llm_response
)
```

📊 유사도 계산 알고리즘
---------------------
**Fuzzy Similarity**: RapidFuzz 기반 문자열 유사도
- 임계값: `PROCESSING_CONFIG.fuzzy_threshold` (기본 0.5)
- 용도: 초기 후보 필터링

**Sequence Similarity**: 시퀀스 매칭 (s1, s2)
- s1: 정규화 방식 1
- s2: 정규화 방식 2
- Combined: s1 + s2 (임계값: 1.0)

**최종 점수**: 
```
final_sim = sim_s1 + sim_s2
필터: final_sim >= high_similarity_threshold (기본 1.0)
```

📝 참고사항
----------
- LLM 모드는 context_mode에 따라 다른 프롬프트 사용
  - 'dag': HYBRID_DAG_EXTRACTION_PROMPT (사용자 행동 경로)
  - 'pairing': HYBRID_PAIRING_EXTRACTION_PROMPT (혜택 매핑)
  - 'ont': ONTOLOGY_PROMPT (팔란티어 온톨로지 기반 JSON 추출)
  - 'none': SIMPLE_ENTITY_EXTRACTION_PROMPT (단순 추출)
- 병렬 처리로 성능 최적화 (joblib Parallel)
- 메시지 길이에 따라 배치 크기 자동 조정
- Stop words 필터링으로 노이즈 제거

"""


import logging
import traceback
import re
import json
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
    SIMPLE_ENTITY_EXTRACTION_PROMPT,
    HYBRID_DAG_EXTRACTION_PROMPT,
    CONTEXT_BASED_ENTITY_EXTRACTION_PROMPT,
    build_context_based_entity_extraction_prompt,
    HYBRID_PAIRING_EXTRACTION_PROMPT,
    ONTOLOGY_PROMPT
)

# Config imports
try:
    from config.settings import PROCESSING_CONFIG
except ImportError:
    logging.warning("Config file not found. Using defaults.")
    class PROCESSING_CONFIG:        # 임계값 설정 (config에서 로드)
        # If config.settings is not found, these are the default values.
        # The original instruction's intent to import PROCESSING_CONFIG inside this block
        # would cause an infinite ImportError loop.
        # Assuming the intent is to define default entity-specific thresholds
        # if the main config is not available.
        fuzzy_threshold = 0.5
        n_jobs = 4
        batch_size = 100
        similarity_threshold = 0.2
        combined_similarity_threshold = 0.2
        high_similarity_threshold = 1.0
        entity_fuzzy_threshold = 0.5
        entity_similarity_threshold = 0.2
        entity_combined_similarity_threshold = 0.2
        entity_high_similarity_threshold = 1.0
        entity_llm_fuzzy_threshold = 0.6  # LLM-based entity extraction threshold

logger = logging.getLogger(__name__)


class EntityRecognizer:
    """
    엔티티 추출 및 매칭 서비스 클래스
    
    책임:
        - MMS 메시지에서 상품/서비스 엔티티 추출
        - 추출된 엔티티를 상품 데이터베이스와 매칭
        - 다양한 추출 모드 지원 (Kiwi, Logic, LLM)
        - 유사도 기반 후보 필터링 및 랭킹
    
    협력 객체:
        - **Kiwi**: 한국어 형태소 분석 (NNP 태그 추출)
        - **LLM Model**: 컨텍스트 기반 엔티티 추출
        - **ItemDataLoader**: 상품 데이터 제공 (item_pdf_all)
        - **Parallel (joblib)**: 병렬 처리로 성능 최적화
    
    데이터 흐름:
        ```
        MMS 메시지
            ↓
        [Kiwi/Logic/LLM 추출]
            ↓
        후보 엔티티 목록
            ↓
        [Fuzzy + Sequence 유사도 계산]
            ↓
        유사도 DataFrame
            ↓
        [임계값 필터링 + 랭킹]
            ↓
        최종 매칭 결과
        ```
    
    Attributes:
        kiwi: Kiwi 형태소 분석기 인스턴스
        item_pdf_all (pd.DataFrame): 전체 상품 정보 (item_nm, item_nm_alias, item_id 등)
        stop_item_names (List[str]): 제외할 불용어 목록
        llm_model: LangChain LLM 모델 인스턴스
        alias_pdf_raw (pd.DataFrame): 별칭 규칙 (선택사항)
        entity_extraction_mode (str): 추출 모드 ('llm', 'nlp', 'logic')
        exc_tag_patterns (List): Kiwi 제외 태그 패턴
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
    def extract_entities_hybrid(self, mms_msg: str) -> Tuple[List[str], List[str], pd.DataFrame]:
        """하이브리드 엔티티 추출 (Kiwi 형태소 분석 + Fuzzy Matching + Sequence Similarity)"""
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

    def extract_entities_with_fuzzy_matching(self, cand_entities: List[str], threshold_for_fuzzy: float = 0.5) -> pd.DataFrame:
        """퍼지 유사도 + 시퀀스 유사도 기반 엔티티 추출"""
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
                normalization_value='s1',
                default_return=pd.DataFrame()
            ).rename(columns={'sim': 'sim_s1'})
            
            sim_s2 = safe_execute(
                parallel_seq_similarity,
                sent_item_pdf=similarities_fuzzy,
                text_col_nm='item_name_in_msg',
                item_col_nm='item_nm_alias',
                n_jobs=getattr(PROCESSING_CONFIG, 'n_jobs', 4),
                batch_size=30,
                normalization_value='s2',
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

    def _parse_ontology_response(self, response: str) -> dict:
        """
        Parse ontology JSON response from LLM.

        Args:
            response: LLM response (expected JSON format)

        Returns:
            dict with keys:
              - 'entities': List[str] - entity IDs
              - 'entity_types': Dict[str, str] - {id: type} mapping
              - 'relationships': List[dict] - [{source, target, type}, ...]
              - 'dag_text': str - user_action_path.dag
              - 'raw_json': dict - original parsed JSON
        """
        try:
            # JSON 추출 (코드 블록 처리)
            json_str = response.strip()
            if json_str.startswith('```'):
                json_str = re.sub(r'^```(?:json)?\n?', '', json_str)
                json_str = re.sub(r'\n?```$', '', json_str)

            data = json.loads(json_str)

            # entities 추출
            entities = [e.get('id', '') for e in data.get('entities', []) if e.get('id')]

            # entity_types 매핑
            entity_types = {e.get('id'): e.get('type', 'Unknown')
                           for e in data.get('entities', []) if e.get('id')}

            # relationships 추출
            relationships = data.get('relationships', [])

            # DAG 텍스트 추출
            dag_text = data.get('user_action_path', {}).get('dag', '')

            return {
                'entities': entities,
                'entity_types': entity_types,
                'relationships': relationships,
                'dag_text': dag_text,
                'raw_json': data
            }
        except json.JSONDecodeError as e:
            logger.warning(f"Ontology JSON parsing failed: {e}")
            # Fallback: 기존 파싱 로직 사용
            return {
                'entities': self._parse_entity_response(response),
                'entity_types': {},
                'relationships': [],
                'dag_text': '',
                'raw_json': {}
            }

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
    def extract_entities_with_llm(self, msg_text: str, rank_limit: int = 50, llm_models: List = None, 
                                external_cand_entities: List[str] = [], context_mode: str = 'dag') -> pd.DataFrame:
        """
        LLM-based entity extraction with multi-model support and configurable context mode.

        Args:
            msg_text: Message text to extract entities from
            rank_limit: Maximum rank for entity candidates
            llm_models: List of LLM models to use (defaults to self.llm_model)
            external_cand_entities: External candidate entities to include
            context_mode: Context extraction mode - 'dag', 'pairing', 'ont', or 'none' (default: 'dag')
                - 'dag': HYBRID_DAG_EXTRACTION_PROMPT (사용자 행동 경로)
                - 'pairing': HYBRID_PAIRING_EXTRACTION_PROMPT (혜택 매핑)
                - 'ont': ONTOLOGY_PROMPT (팔란티어 온톨로지 기반 JSON 추출)
                - 'none': SIMPLE_ENTITY_EXTRACTION_PROMPT (단순 추출)

        Returns:
            DataFrame with extracted entities and similarity scores
        """

        try:
            logger.info("=== LLM Entity Extraction Started ===")
            logger.info(f"Context mode: {context_mode}")
            msg_text = validate_text_input(msg_text)
            
            if llm_models is None:
                llm_models = [self.llm_model]
            
            # Validate context_mode
            if context_mode not in ['dag', 'pairing', 'none', 'ont']:
                logger.warning(f"Invalid context_mode '{context_mode}', defaulting to 'dag'")
                context_mode = 'dag'
            
            # Select prompt based on context_mode
            if context_mode == 'dag':
                first_stage_prompt = HYBRID_DAG_EXTRACTION_PROMPT
                context_keyword = 'DAG'
            elif context_mode == 'pairing':
                first_stage_prompt = HYBRID_PAIRING_EXTRACTION_PROMPT
                context_keyword = 'PAIRING'
            elif context_mode == 'ont':
                first_stage_prompt = ONTOLOGY_PROMPT
                context_keyword = 'ONT'
            else:  # 'none'
                first_stage_prompt = SIMPLE_ENTITY_EXTRACTION_PROMPT
                context_keyword = None
            
            # Internal function for parallel execution
            def get_entities_and_context_by_llm(args_dict):
                llm_model, prompt = args_dict['llm_model'], args_dict['prompt']
                extract_context = args_dict.get('extract_context', True)
                context_kw = args_dict.get('context_keyword', None)
                is_ontology_mode = args_dict.get('is_ontology_mode', False)
                model_name = getattr(llm_model, 'model_name', 'Unknown')

                try:
                    # Log the prompt being sent to LLM
                    prompt_res_log_list = []
                    prompt_res_log_list.append(f"[{model_name}] Sending prompt to LLM:")
                    prompt_res_log_list.append("="*100)
                    prompt_res_log_list.append(prompt)
                    prompt_res_log_list.append("="*100)

                    response = llm_model.invoke(f"""

                    {prompt}

                    """).content

                    # Log the response received from LLM
                    prompt_res_log_list.append(f"[{model_name}] Received response from LLM:")
                    prompt_res_log_list.append("-"*100)
                    prompt_res_log_list.append(response)
                    prompt_res_log_list.append("-"*100)

                    logger.debug("\n".join(prompt_res_log_list))

                    # Ontology mode: use JSON parsing
                    if is_ontology_mode:
                        parsed = self._parse_ontology_response(response)
                        cand_entity_list = [e for e in parsed['entities']
                                          if e not in self.stop_item_names and len(e) >= 2]

                        # Build rich context with Entity Types, Relationships, and DAG
                        entity_types = parsed.get('entity_types', {})
                        relationships = parsed.get('relationships', [])
                        dag_text = parsed['dag_text']

                        # Format entity types: Name(Type), ...
                        entity_type_str = ", ".join([f"{k}({v})" for k, v in entity_types.items()]) if entity_types else ""

                        # Format relationships: Source -[TYPE]-> Target
                        rel_lines = []
                        for rel in relationships:
                            src = rel.get('source', '')
                            tgt = rel.get('target', '')
                            rel_type = rel.get('type', '')
                            if src and tgt and rel_type:
                                rel_lines.append(f"  - {src} -[{rel_type}]-> {tgt}")
                        relationships_str = "\n".join(rel_lines) if rel_lines else ""

                        # Combine all parts into context_text
                        context_parts = []
                        if entity_type_str:
                            context_parts.append(f"Entities: {entity_type_str}")
                        if relationships_str:
                            context_parts.append(f"Relationships:\n{relationships_str}")
                        if dag_text:
                            context_parts.append(f"DAG: {dag_text}")
                        context_text = "\n".join(context_parts)

                        logger.info(f"[{model_name}] Extracted {len(cand_entity_list)} entities (ONT mode): {cand_entity_list}")
                        logger.info(f"[{model_name}] Entity types: {entity_types}")
                        logger.info(f"[{model_name}] Relationships: {len(relationships)} found")
                        return {
                            "entities": cand_entity_list,
                            "context_text": context_text,
                            "entity_types": entity_types,
                            "relationships": relationships
                        }

                    # Standard mode: use regex parsing
                    cand_entity_list_raw = self._parse_entity_response(response)
                    cand_entity_list = [e for e in cand_entity_list_raw if e not in self.stop_item_names and len(e) >= 2]

                    logger.info(f"[{model_name}] Extracted {len(cand_entity_list)} entities: {cand_entity_list}")

                    context_text = ""
                    if extract_context and context_kw:
                        context_match = re.search(rf'{context_kw}:\s*(.*)', response, re.DOTALL | re.IGNORECASE)
                        if context_match:
                            context_text = context_match.group(1).strip()
                            logger.info(f"[{model_name}] Extracted {context_kw} text ({len(context_text)} chars)")
                        else:
                            logger.debug(f"[{model_name}] No {context_kw} found in response")

                    return {"entities": cand_entity_list, "context_text": context_text}
                except Exception as e:
                    logger.error(f"LLM extraction failed for {model_name}: {e}")
                    return {"entities": [], "context_text": ""}

            def get_entities_only_by_llm(args_dict):
                result = get_entities_and_context_by_llm(args_dict)
                return result['entities']

            # 1. First Stage: Extract entities and context
            batches = []
            for llm_model in llm_models:
                prompt = f"{first_stage_prompt}\n\n## message:\n{msg_text}"
                batches.append({
                    "prompt": prompt,
                    "llm_model": llm_model,
                    "extract_context": (context_mode != 'none'),
                    "context_keyword": context_keyword,
                    "is_ontology_mode": (context_mode == 'ont')
                })
            
            n_jobs = min(len(batches), 3)
            with Parallel(n_jobs=n_jobs, backend='threading') as parallel:
                batch_results_dicts = parallel(delayed(get_entities_and_context_by_llm)(args) for args in batches)
            
            all_entities = []
            all_contexts = []
            all_entity_types = {}
            all_relationships = []

            for result_dict in batch_results_dicts:
                all_entities.extend(result_dict['entities'])
                if result_dict['context_text']:
                    all_contexts.append(result_dict['context_text'])
                # ONT 모드에서 entity_types와 relationships 수집
                if context_mode == 'ont':
                    if 'entity_types' in result_dict:
                        all_entity_types.update(result_dict.get('entity_types', {}))
                    if 'relationships' in result_dict:
                        all_relationships.extend(result_dict.get('relationships', []))

            combined_context = "\n".join(all_contexts)
            
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
                
                # Build context section based on mode
                if context_mode == 'none' or not combined_context:
                    context_section = ""
                else:
                    context_label = f"{context_keyword} Context (User Action Paths)" if context_keyword else "Context"
                    context_section = f"\n## {context_label}:\n{combined_context}\n"
                
                # Build dynamic prompt based on context_keyword
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
            
            n_jobs = min(len(batches), 3)
            with Parallel(n_jobs=n_jobs, backend='threading') as parallel:
                batch_results = parallel(delayed(get_entities_only_by_llm)(args) for args in batches)
            
            cand_entity_list = list(set(sum(batch_results, [])))

            cand_entities_sim = cand_entities_sim.query("item_nm_alias in @cand_entity_list")

            # ONT 모드일 경우 메타데이터 포함하여 반환
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

        except Exception as e:
            logger.error(f"LLM entity extraction failed: {e}")
            logger.error(traceback.format_exc())
            if context_mode == 'ont':
                return {
                    'similarities_df': pd.DataFrame(),
                    'ont_metadata': None
                }
            return pd.DataFrame()

    def _match_entities_with_products(self, cand_entity_list: List[str], rank_limit: int) -> pd.DataFrame:
        """Match candidate entities with product database"""
        try:
            # print(cand_entity_list)
            # LLM 기반 엔티티 추출을 위한 Fuzzy 유사도 계산
            similarities_fuzzy = parallel_fuzzy_similarity(
                cand_entity_list,
                self.item_pdf_all['item_nm_alias'].unique(),
                threshold=getattr(PROCESSING_CONFIG, 'entity_llm_fuzzy_threshold', 0.6),
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


            # print(cand_entities_sim['item_nm_alias'].unique())
            
            return cand_entities_sim
            
        except Exception as e:
            logger.error(f"Entity-product matching failed: {e}")
            return pd.DataFrame()

    def map_products_to_entities(self, similarities_fuzzy: pd.DataFrame, json_objects: Dict = None) -> List[Dict]:
        """유사도 결과를 기반으로 상품을 엔티티에 매핑"""
        try:
            logger.info("🔍 [map_products_to_entities] Started")
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
