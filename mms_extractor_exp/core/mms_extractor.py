# %%
"""
MMS 추출기 (MMS Extractor) - AI 기반 광고 텍스트 분석 시스템
================================================================

📋 개요
-------
이 모듈은 MMS(멀티미디어 메시지) 광고 텍스트에서 구조화된 정보를 자동으로 추출하는
AI 기반 시스템입니다. LLM(Large Language Model)을 활용하여 비정형 텍스트에서
상품명, 채널 정보, 광고 목적, 엔티티 관계 등을 정확하게 식별하고 추출합니다.

🎯 핵심 기능
-----------
1. **엔티티 추출**: 상품명, 브랜드명, 서비스명 등 핵심 엔티티 식별
2. **채널 분석**: URL, 전화번호, 앱 링크 등 고객 접점 채널 추출
3. **목적 분류**: 광고의 주요 목적 및 액션 타입 분석
4. **프로그램 매칭**: 사전 정의된 프로그램 카테고리와의 유사도 기반 분류
5. **DAG 생성**: 엔티티 간 관계를 방향성 그래프로 시각화

🔧 주요 개선사항
--------------
- **모듈화 설계**: 대형 메소드를 기능별 모듈로 분리하여 유지보수성 향상
- **프롬프트 외부화**: 하드코딩된 프롬프트를 외부 모듈로 분리하여 관리 용이성 증대
- **예외 처리 강화**: LLM 호출 실패, 네트워크 오류 등에 대한 robust한 에러 복구
- **성능 모니터링**: 상세한 로깅 및 실행 시간 추적으로 성능 최적화 지원
- **데이터 검증**: 추출 결과의 품질 보장을 위한 다층 검증 시스템
- **하이브리드 데이터 소스**: CSV 파일과 Oracle DB를 모두 지원하는 유연한 데이터 로딩

🏗️ 아키텍처
-----------
- **MMSExtractor**: 메인 추출 엔진 클래스
- **DataManager**: 데이터 로딩 및 관리 담당
- **LLMProcessor**: LLM 호출 및 응답 처리
- **EntityMatcher**: 엔티티 매칭 및 유사도 계산
- **PromptModule**: 외부화된 프롬프트 관리

⚙️ 설정 및 환경
--------------
- Python 3.8+
- LangChain, OpenAI, Anthropic API 지원
- Oracle Database 연동 (선택사항)
- GPU 가속 (CUDA 지원 시)

"""

from concurrent.futures import ThreadPoolExecutor
import time
import logging
import warnings
from functools import wraps
from typing import List, Tuple, Union, Dict, Any, Optional
from abc import ABC, abstractmethod
import traceback
import json
import re
import ast
import glob
import os
import copy
import pandas as pd
import numpy as np
from langchain_core.prompts import PromptTemplate

# joblib과 multiprocessing 경고 억제
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")
warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing")
warnings.filterwarnings("ignore", message=".*resource_tracker.*")
warnings.filterwarnings("ignore", message=".*leaked.*")
import torch
from sentence_transformers import SentenceTransformer
from difflib import SequenceMatcher
import difflib
from dotenv import load_dotenv
import cx_Oracle
from contextlib import contextmanager

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from rapidfuzz import fuzz, process
from kiwipiepy import Kiwi
from joblib import Parallel, delayed
from .entity_dag_extractor import DAGParser, extract_dag

# 프롬프트 모듈 임포트
from prompts import (
    build_extraction_prompt,
    enhance_prompt_for_retry,
    get_fallback_result,
    build_entity_extraction_prompt,
    DEFAULT_ENTITY_EXTRACTION_PROMPT,
    DETAILED_ENTITY_EXTRACTION_PROMPT,
    CONTEXT_BASED_ENTITY_EXTRACTION_PROMPT,
    build_context_based_entity_extraction_prompt,
    HYBRID_DAG_EXTRACTION_PROMPT,
    SIMPLE_ENTITY_EXTRACTION_PROMPT
    )



# Helpers 모듈 임포트
from utils import PromptManager
from utils.llm_factory import LLMFactory

# Workflow 모듈 임포트
from .workflow_core import WorkflowEngine, WorkflowState
from .mms_workflow_steps import (
    InputValidationStep,
    EntityExtractionStep,
    ProgramClassificationStep,
    ContextPreparationStep,
    LLMExtractionStep,
    ResponseParsingStep,
    EntityContextExtractionStep,
    VocabularyFilteringStep,
    ResultConstructionStep,
    ValidationStep,
    DAGExtractionStep
)
from services.entity_recognizer import EntityRecognizer
from services.program_classifier import ProgramClassifier
from services.store_matcher import StoreMatcher
from services.result_builder import ResultBuilder
from core.mms_extractor_data import MMSExtractorDataMixin


# 유틸리티 함수 모듈 임포트
from utils import (
    select_most_comprehensive,
    log_performance,
    safe_execute,
    validate_text_input,
    safe_check_empty,
    dataframe_to_markdown_prompt,
    extract_json_objects,
    preprocess_text,
    calculate_fuzzy_similarity,
    calculate_fuzzy_similarity_batch,
    parallel_fuzzy_similarity,
    longest_common_subsequence_ratio,
    sequence_matcher_similarity,
    substring_aware_similarity,
    token_sequence_similarity,
    combined_sequence_similarity,
    calculate_seq_similarity,
    parallel_seq_similarity,
    load_sentence_transformer,
    Token,
)

# Database utilities 임포트
from utils.db_utils import (
    get_database_connection,
    database_connection,
    load_program_from_database,
    load_org_from_database
)

from utils import (
    Sentence,
    filter_text_by_exc_patterns,
    filter_specific_terms,
    convert_df_to_json_list,
    create_dag_diagram,
    sha256_hash,
    replace_special_chars_with_space,
    extract_ngram_candidates
)

# 설정 및 의존성 임포트 (원본 코드에서 가져옴)
try:
    from config.settings import API_CONFIG, MODEL_CONFIG, PROCESSING_CONFIG, METADATA_CONFIG, EMBEDDING_CONFIG
except ImportError:
    logging.warning("설정 파일을 찾을 수 없습니다. 기본값을 사용합니다.")
    # 기본 설정값들을 여기에 정의할 수 있습니다.

# 로깅 설정 - api.py에서 실행될 때는 해당 설정을 사용하고, 직접 실행될 때만 기본 설정 적용
logger = logging.getLogger(__name__)

# 직접 실행될 때만 로깅 설정 (api.py에서 임포트될 때는 api.py의 설정 사용)
if __name__ == '__main__':
    import sys
    from pathlib import Path
    
    # MongoDB 유틸리티는 필요할 때 동적으로 임포트
    
    # 로그 디렉토리 생성
    log_dir = Path(__file__).parent / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'mms_extractor.log'),
            logging.StreamHandler()
        ]
    )

# pandas 출력 설정
pd.set_option('display.max_colwidth', 500)
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# ===== 추상 클래스 및 전략 패턴 =====

class EntityExtractionStrategy(ABC):
    """엔티티 추출 전략 추상 클래스"""
    
    @abstractmethod
    def extract(self, text: str, **kwargs) -> pd.DataFrame:
        """엔티티 추출 메소드"""
        pass

class DataLoader(ABC):
    """데이터 로더 추상 클래스"""
    
    @abstractmethod
    def load_data(self) -> Dict[str, Any]:
        """데이터 로드 메소드"""
        pass

# ===== 개선된 MMSExtractor 클래스 =====

class MMSExtractor(MMSExtractorDataMixin):
    """
    MMS 광고 텍스트 AI 분석 시스템 - 메인 추출 엔진
    ================================================================
    
    🎨 개요
    -------
    이 클래스는 MMS 광고 텍스트에서 구조화된 정보를 추출하는 핵심 엔진입니다.
    LLM(Large Language Model), 임베딩 모델, NLP 기법을 조합하여
    비정형 텍스트에서 정형화된 데이터를 추출합니다.
    
    🔧 주요 기능
    -----------
    1. **다단계 엔티티 추출**: Kiwi NLP + 임베딩 유사도 + LLM 기반 추출
    2. **지능형 프로그램 분류**: 사전 정의된 카테고리와의 유사도 매칭
    3. **RAG 기반 컨텍스트 증강**: 관련 데이터를 활용한 정확도 향상
    4. **다중 LLM 지원**: OpenAI, Anthropic, Gemini, AX 등
    5. **DAG 생성**: 엔티티 간 관계를 방향성 그래프로 시각화
    
    🏗️ 아키텍처 (Workflow 기반)
    --------------------------
    ```
    MMSExtractor
        ├─ WorkflowEngine (9 Steps)
        │   ├─ InputValidationStep
        │   ├─ EntityExtractionStep → EntityRecognizer
        │   ├─ ProgramClassificationStep → ProgramClassifier
        │   ├─ ContextPreparationStep
        │   ├─ LLMExtractionStep → LLM Model
        │   ├─ ResponseParsingStep
        │   ├─ ResultConstructionStep → ResultBuilder
        │   ├─ ValidationStep
        │   └─ DAGExtractionStep (선택적)
        │
        ├─ Services
        │   ├─ EntityRecognizer (엔티티 추출/매칭)
        │   ├─ ItemDataLoader (데이터 로딩)
        │   ├─ ProgramClassifier (프로그램 분류)
        │   ├─ StoreMatcher (매장 매칭)
        │   └─ ResultBuilder (결과 구성)
        │
        └─ Data Components
            ├─ item_pdf_all (상품 데이터)
            ├─ pgm_pdf (프로그램 데이터)
            ├─ org_pdf (조직 데이터)
            └─ embeddings (임베딩 캐시)
    ```
    
    📊 성능 특징
    -----------
    - **정확도**: 85%+ (수동 검증 기준)
    - **처리 속도**: 평균 30초/메시지
    - **확장성**: 모듈화된 설계로 새로운 기능 추가 용이
    - **안정성**: 강화된 예외 처리 및 재시도 메커니즘
    
    ⚙️ 주요 개선사항
    --------------
    - **워크플로우 엔진 도입**: 9단계 처리 파이프라인으로 구조화
    - **서비스 분리**: EntityRecognizer, ResultBuilder 등 독립 서비스화
    - **프롬프트 외부화**: prompts 모듈로 분리하여 관리 효율성 증대
    - **다층 예외 처리**: LLM API 실패, 네트워크 오류 등에 대한 robust한 에러 복구
    - **상세 로깅**: 성능 모니터링, 디버깅, 감사 로그를 위한 포괄적 로깅 시스템
    - **데이터 검증**: 입력/출력 데이터 품질 보장을 위한 다단계 검증
    - **하이브리드 데이터 소스**: CSV 파일과 Oracle DB를 모두 지원하는 유연한 데이터 로딩
    
    🤝 협력 객체
    -----------
    - **WorkflowEngine**: 9단계 처리 파이프라인 실행
    - **EntityRecognizer**: Kiwi + LLM 기반 엔티티 추출
    - **ItemDataLoader**: 상품 데이터 로딩 및 전처리
    - **ProgramClassifier**: 임베딩 기반 프로그램 분류
    - **StoreMatcher**: 매장 정보 매칭
    - **ResultBuilder**: 최종 결과 구성 및 스키마 변환
    - **LLMFactory**: LLM 모델 생성 및 관리
    
    📝 사용 예시
    -----------
    ```python
    # 1. 기본 초기화
    extractor = MMSExtractor(
        llm_model='ax',
        entity_extraction_mode='llm',
        extract_entity_dag=True
    )
    
    # 2. 단일 메시지 처리
    result = extractor.process_message("샘플 MMS 텍스트")
    
    # 3. 결과 활용
    products = result['ext_result']['product']
    channels = result['ext_result']['channel']
    entity_dag = result['ext_result'].get('entity_dag', [])
    
    # 4. 배치 처리
    messages = ["메시지1", "메시지2", "메시지3"]
    results = [extractor.process_message(msg) for msg in messages]
    
    # 5. 런타임 설정 변경
    extractor.llm_model_name = 'gpt'
    extractor.entity_extraction_mode = 'logic'
    extractor._initialize_llm()  # LLM 재초기화
    ```
    
    💼 의존성
    ---------
    - **LangChain**: LLM 인터페이스
    - **SentenceTransformers**: 임베딩 모델
    - **KiwiPiePy**: 한국어 형태소 분석
    - **cx_Oracle**: 데이터베이스 연동 (선택적)
    - **pandas**: 데이터 처리
    - **torch**: 딥러닝 프레임워크
    
    📌 주요 속성
    -----------
    - `workflow_engine`: WorkflowEngine 인스턴스
    - `entity_recognizer`: EntityRecognizer 서비스
    - `program_classifier`: ProgramClassifier 서비스
    - `result_builder`: ResultBuilder 서비스
    - `llm_model`: 활성 LLM 모델
    - `item_pdf_all`: 전체 상품 데이터
    - `extract_entity_dag`: DAG 추출 활성화 여부
    """
    
    def __init__(self, model_path=None, data_dir=None, product_info_extraction_mode=None,
                 entity_extraction_mode=None, offer_info_data_src='db', llm_model='ax',
                 entity_llm_model='ax', extract_entity_dag=False, entity_extraction_context_mode='dag',
                 skip_entity_extraction=False, use_external_candidates=False,
                 extraction_engine='default',
                 num_cand_pgms=None, num_select_pgms=None):
        """
        MMSExtractor 초기화 메소드
        
        시스템에 필요한 모든 구성 요소들을 초기화합니다:
        - LLM 모델 설정 및 연결
        - 임베딩 모델 로드
        - NLP 도구 (Kiwi) 초기화
        - 데이터 소스 로드 (CSV/DB)
        - 각종 설정 매개변수 구성
        
        Args:
            model_path (str, optional): 임베딩 모델 경로. 기본값: 'jhgan/ko-sroberta-multitask'
            data_dir (str, optional): 데이터 디렉토리 경로. 기본값: './data/'
            product_info_extraction_mode (str, optional): 상품 정보 추출 모드 ('nlp' 또는 'llm')
            entity_extraction_mode (str, optional): 엔티티 추출 모드 ('nlp', 'llm', 'hybrid')
            offer_info_data_src (str, optional): 데이터 소스 타입 ('local' 또는 'db')
            llm_model (str, optional): 사용할 LLM 모델. 기본값: 'ax'
            entity_llm_model (str, optional): 엔티티 추출용 LLM 모델. 기본값: 'ax'
            extract_entity_dag (bool, optional): DAG 추출 여부. 기본값: False
            entity_extraction_context_mode (str, optional): 엔티티 추출 컨텍스트 모드 ('dag', 'pairing', 'none', 'ont', 'kg'). 기본값: 'dag'
            
        Raises:
            Exception: 초기화 과정에서 발생하는 모든 오류
            
        Example:
            >>> extractor = MMSExtractor(
            ...     llm_model='gpt-4',
            ...     entity_extraction_mode='hybrid',
            ...     extract_entity_dag=True
            ... )
        """
        logger.info("🚀 MMSExtractor 초기화 시작")
        
        try:
            # 1단계: 기본 설정 매개변수 구성
            logger.info("⚙️ 기본 설정 적용 중...")
            self._set_default_config(model_path, data_dir, product_info_extraction_mode,
                                   entity_extraction_mode, offer_info_data_src, llm_model, entity_llm_model,
                                   extract_entity_dag, entity_extraction_context_mode,
                                   skip_entity_extraction, use_external_candidates,
                                   extraction_engine, num_cand_pgms, num_select_pgms)
            
            # 2단계: 환경변수 로드 (API 키 등)
            logger.info("🔑 환경변수 로드 중...")
            load_dotenv()
            
            # 3단계: 주요 구성 요소들 순차 초기화
            logger.info("💻 디바이스 설정 중...")
            self._initialize_device()
            
            logger.info("🤖 LLM 모델 초기화 중...")
            self._initialize_llm()
            
            logger.info("🧠 임베딩 모델 로드 중...")
            self._initialize_embedding_model()
            
            logger.info("📝 NLP 도구 (Kiwi) 초기화 중...")
            self._initialize_kiwi()
            
            logger.info("📁 데이터 로드 중...")
            self._load_data()
            
            # Initialize LLM Factory
            logger.info("🏭 LLM Factory 초기화 중...")
            self.llm_factory = LLMFactory()
            logger.info("✅ LLM Factory 초기화 완료")
            
            # Initialize Services
            logger.info("🛠️ 서비스 초기화 중...")
            self.entity_recognizer = EntityRecognizer(
                self.kiwi, 
                self.item_pdf_all, 
                self.stop_item_names, 
                self.llm_model, 
                self.alias_pdf_raw,
                self.entity_extraction_mode
            )
            self.program_classifier = ProgramClassifier(
                self.emb_model, 
                self.pgm_pdf, 
                self.clue_embeddings,
                self.num_cand_pgms
            )
            self.store_matcher = StoreMatcher(self.org_pdf)
            self.result_builder = ResultBuilder(
                self.store_matcher,
                self.stop_item_names,
                self.num_cand_pgms,
            )
            logger.info("✅ 서비스 초기화 완료")

            # Workflow 엔진 초기화
            logger.info("⚙️ Workflow 엔진 초기화 중...")
            self.workflow_engine = WorkflowEngine("MMS Extraction Workflow")
            self.workflow_engine.add_step(InputValidationStep())
            self.workflow_engine.add_step(EntityExtractionStep(
                self.entity_recognizer,
                skip_entity_extraction=self.skip_entity_extraction,
            ))
            self.workflow_engine.add_step(ProgramClassificationStep(self.program_classifier))
            self.workflow_engine.add_step(ContextPreparationStep())
            self.workflow_engine.add_step(LLMExtractionStep())
            self.workflow_engine.add_step(ResponseParsingStep())
            # Step 7: Entity + Context Extraction
            self.workflow_engine.add_step(EntityContextExtractionStep(
                entity_recognizer=self.entity_recognizer,
                llm_factory=self.llm_factory,
                llm_model=self.entity_llm_model_name,
                entity_extraction_context_mode=self.entity_extraction_context_mode,
                use_external_candidates=self.use_external_candidates,
                extraction_engine=self.extraction_engine,
                stop_item_names=self.stop_item_names,
                entity_extraction_mode=self.entity_extraction_mode,
            ))
            # Step 8: Vocabulary Filtering
            self.workflow_engine.add_step(VocabularyFilteringStep(
                entity_recognizer=self.entity_recognizer,
                alias_pdf_raw=self.alias_pdf_raw,
                stop_item_names=self.stop_item_names,
                entity_extraction_mode=self.entity_extraction_mode,
                llm_factory=self.llm_factory,
                llm_model=self.entity_llm_model_name,
                entity_extraction_context_mode=self.entity_extraction_context_mode,
            ))
            self.workflow_engine.add_step(ResultConstructionStep(self.result_builder))
            self.workflow_engine.add_step(ValidationStep())
            
            # DAG 추출 단계는 플래그가 활성화된 경우만 등록
            if self.extract_entity_dag:
                self.workflow_engine.add_step(DAGExtractionStep())
                logger.info("🎯 DAG 추출 단계 등록됨")
            
            logger.info(f"✅ Workflow 엔진 초기화 완료 ({len(self.workflow_engine.steps)}개 단계)")
            
            logger.info("✅ MMSExtractor 초기화 완료")

            
        except Exception as e:
            logger.error(f"❌ MMSExtractor 초기화 실패: {e}")
            logger.error(traceback.format_exc())
            raise

    def _set_default_config(self, model_path, data_dir, product_info_extraction_mode,
                          entity_extraction_mode, offer_info_data_src, llm_model, entity_llm_model,
                          extract_entity_dag, entity_extraction_context_mode,
                          skip_entity_extraction, use_external_candidates,
                          extraction_engine, num_cand_pgms=None, num_select_pgms=None):
        """기본 설정값 적용"""
        self.data_dir = data_dir if data_dir is not None else './data/'
        self.model_path = model_path if model_path is not None else getattr(EMBEDDING_CONFIG, 'ko_sbert_model_path', 'jhgan/ko-sroberta-multitask')
        self.offer_info_data_src = offer_info_data_src
        self.product_info_extraction_mode = product_info_extraction_mode if product_info_extraction_mode is not None else getattr(PROCESSING_CONFIG, 'product_info_extraction_mode', 'nlp')
        self.entity_extraction_mode = entity_extraction_mode if entity_extraction_mode is not None else getattr(PROCESSING_CONFIG, 'entity_extraction_mode', 'llm')
        self.llm_model_name = llm_model
        self.entity_llm_model_name = entity_llm_model
        self.num_cand_pgms = num_cand_pgms if num_cand_pgms is not None else getattr(PROCESSING_CONFIG, 'num_candidate_programs', 20)
        self.num_select_pgms = num_select_pgms if num_select_pgms is not None else getattr(PROCESSING_CONFIG, 'num_select_programs', 1)
        self.extract_entity_dag = extract_entity_dag
        self.entity_extraction_context_mode = entity_extraction_context_mode
        self.skip_entity_extraction = skip_entity_extraction
        self.use_external_candidates = use_external_candidates
        self.extraction_engine = extraction_engine

        # DAG 추출 설정 로깅
        # extract_entity_dag: 엔티티 간 관계를 DAG(Directed Acyclic Graph)로 추출
        # True인 경우 추가적으로 LLM을 사용하여 엔티티 관계를 분석하고
        # NetworkX + Graphviz를 통해 시각적 다이어그램을 생성
        if self.extract_entity_dag:
            logger.info("🎯 DAG 추출 모드 활성화됨")
        else:
            logger.info("📋 표준 추출 모드 (DAG 비활성화)")

    @log_performance
    def _initialize_device(self):
        """사용할 디바이스 초기화"""
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        logger.info(f"Using device: {self.device}")

    @log_performance
    def _initialize_llm(self):
        """LLM 모델 초기화"""
        try:
            # 모델 설정 매핑
            model_mapping = {
                "gemma": getattr(MODEL_CONFIG, 'gemma_model', 'gemma-7b'),
                "gem": getattr(MODEL_CONFIG, 'gemma_model', 'gemma-7b'),  # 'gem'은 'gemma'의 줄임말
                "ax": getattr(MODEL_CONFIG, 'ax_model', 'ax-4'),
                "claude": getattr(MODEL_CONFIG, 'claude_model', 'claude-4'),
                "cld": getattr(MODEL_CONFIG, 'claude_model', 'claude-4'),  # 'cld'는 'claude'의 줄임말
                "gemini": getattr(MODEL_CONFIG, 'gemini_model', 'gemini-pro'),
                "gen": getattr(MODEL_CONFIG, 'gemini_model', 'gemini-pro'),  # 'gen'은 'gemini'의 줄임말
                "gpt": getattr(MODEL_CONFIG, 'gpt_model', 'gpt-4')
            }
            
            model_name = model_mapping.get(self.llm_model_name, getattr(MODEL_CONFIG, 'llm_model', 'gemini-pro'))
            
            # LLM 모델별 일관성 설정
            model_kwargs = {
                "temperature": 0.0,  # 완전 결정적 출력을 위해 0.0 고정
                "openai_api_key": getattr(API_CONFIG, 'llm_api_key', os.getenv('OPENAI_API_KEY')),
                "openai_api_base": getattr(API_CONFIG, 'llm_api_url', None),
                "model": model_name,
                "max_tokens": getattr(MODEL_CONFIG, 'llm_max_tokens', 4000)
            }
            
            # GPT 모델의 경우 시드 설정으로 일관성 강화
            if 'gpt' in model_name.lower():
                model_kwargs["seed"] = 42  # 고정 시드로 일관성 보장
                
            self.llm_model = ChatOpenAI(**model_kwargs)
            
            logger.info(f"LLM 초기화 완료: {self.llm_model_name} ({model_name})")
            
        except Exception as e:
            logger.error(f"LLM 초기화 실패: {e}")
            raise

    @log_performance
    def _initialize_embedding_model(self):
        """임베딩 모델 초기화"""
        # 임베딩 비활성화 옵션 확인
        if MODEL_CONFIG.disable_embedding:
            logger.info("임베딩 모델 비활성화 모드 (DISABLE_EMBEDDING=true)")
            self.emb_model = None
            return
            
        try:
            self.emb_model = load_sentence_transformer(self.model_path, self.device)
        except Exception as e:
            logger.error(f"임베딩 모델 초기화 실패: {e}")
            # 기본 모델로 fallback
            logger.info("기본 모델로 fallback 시도")
            try:
                self.emb_model = load_sentence_transformer('jhgan/ko-sroberta-multitask', self.device)
            except Exception as e2:
                logger.error(f"Fallback 모델도 실패: {e2}")
                logger.warning("임베딩 모델 없이 동작 모드로 전환")
                self.emb_model = None

    def _initialize_multiple_llm_models(self, model_names: List[str]) -> List:
        """
        복수의 LLM 모델을 초기화하는 헬퍼 메서드 (LLMFactory로 위임)
        
        Note:
            이 메서드는 하위 호환성을 위해 유지되며, 내부적으로 LLMFactory를 사용합니다.
        
        Args:
            model_names (List[str]): 초기화할 모델명 리스트 (예: ['ax', 'gpt', 'gen'])
            
        Returns:
            List: 초기화된 LLM 모델 객체 리스트
        """
        return self.llm_factory.create_models(model_names)

    @log_performance
    def _initialize_kiwi(self):
        """Kiwi 형태소 분석기 초기화"""
        try:
            self.kiwi = Kiwi()
            
            # 제외할 품사 태그 패턴들
            self.exc_tag_patterns = [
                ['SN', 'NNB'], ['W_SERIAL'], ['JKO'], ['W_URL'], ['W_EMAIL'],
                ['XSV', 'EC'], ['VV', 'EC'], ['VCP', 'ETM'], ['XSA', 'ETM'],
                ['VV', 'ETN'], ['SSO'], ['SSC'], ['SW'], ['SF'], ['SP'], 
                ['SS'], ['SE'], ['SO'], ['SB'], ['SH'], ['W_HASHTAG']
            ]
            logger.info("Kiwi 형태소 분석기 초기화 완료")
            
        except Exception as e:
            logger.error(f"Kiwi 초기화 실패: {e}")
            raise

    @log_performance
    def _load_data(self):
        """필요한 데이터 파일들 로드"""
        try:
            logger.info("=" * 60)
            logger.info("📊 데이터 로딩 시작")
            logger.info("=" * 60)
            logger.info(f"데이터 소스 모드: {self.offer_info_data_src}")
            
            # 상품 정보 로드 및 준비 (별칭 규칙 적용 포함)
            logger.info("1️⃣ 상품 정보 로드 및 준비 중...")
            self._load_item_data()
            logger.info(f"상품 정보 최종 데이터 크기: {self.item_pdf_all.shape}")
            logger.info(f"상품 정보 컬럼들: {list(self.item_pdf_all.columns)}")
            
            # 정지어 로드
            logger.info("2️⃣ 정지어 로드 중...")
            self._load_stopwords()
            logger.info(f"로드된 정지어 수: {len(self.stop_item_names)}개")
            
            # Kiwi에 상품명 등록
            logger.info("3️⃣ Kiwi에 상품명 등록 중...")
            self._register_items_in_kiwi()
            
            # 프로그램 분류 정보 로드
            logger.info("4️⃣ 프로그램 분류 정보 로드 중...")
            self._load_program_data()
            logger.info(f"프로그램 분류 정보 로드 후 데이터 크기: {self.pgm_pdf.shape}")
            
            # 조직 정보 로드
            logger.info("5️⃣ 조직 정보 로드 중...")
            self._load_organization_data()
            logger.info(f"조직 정보 로드 후 데이터 크기: {self.org_pdf.shape}")
            
            # 최종 데이터 상태 요약
            logger.info("=" * 60)
            logger.info("📋 데이터 로딩 완료 - 최종 상태 요약")
            logger.info("=" * 60)
            logger.info(f"✅ 상품 데이터: {self.item_pdf_all.shape}")
            logger.info(f"✅ 프로그램 데이터: {self.pgm_pdf.shape}")
            logger.info(f"✅ 조직 데이터: {self.org_pdf.shape}")
            logger.info(f"✅ 정지어: {len(self.stop_item_names)}개")
            
            # 데이터 소스별 상태 비교를 위한 추가 정보
            if hasattr(self, 'item_pdf_all') and not self.item_pdf_all.empty:
                logger.info("=== 상품 데이터 상세 정보 ===")
                if 'item_nm' in self.item_pdf_all.columns:
                    unique_items = self.item_pdf_all['item_nm'].nunique()
                    logger.info(f"고유 상품명 수: {unique_items}개")
                if 'item_nm_alias' in self.item_pdf_all.columns:
                    unique_aliases = self.item_pdf_all['item_nm_alias'].nunique()
                    logger.info(f"고유 별칭 수: {unique_aliases}개")
                if 'item_id' in self.item_pdf_all.columns:
                    unique_ids = self.item_pdf_all['item_id'].nunique()
                    logger.info(f"고유 상품ID 수: {unique_ids}개")
            
        except Exception as e:
            logger.error(f"데이터 로딩 실패: {e}")
            logger.error(f"오류 상세: {traceback.format_exc()}")
            raise

    def _load_item_data(self):
        """
        상품 정보 로드 (ItemDataLoader로 위임)
        
        기존 197줄의 복잡한 로직을 ItemDataLoader 서비스로 분리하여
        재사용성과 테스트 용이성을 향상시켰습니다.
        """
        try:
            from services.item_data_loader import ItemDataLoader
            from utils.db_utils import load_item_from_database
            
            # ItemDataLoader 인스턴스 생성
            loader = ItemDataLoader(
                data_source=self.offer_info_data_src,
                db_loader=load_item_from_database if self.offer_info_data_src == 'db' else None
            )
            
            # 전체 파이프라인 실행
            self.item_pdf_all, _ = loader.load_and_prepare_items()
            
            # 별칭 규칙 원본 저장 (다른 컴포넌트에서 사용)
            self.alias_pdf_raw = loader.alias_pdf_raw
            
            logger.info(f"✅ ItemDataLoader를 통한 상품 정보 준비 완료: {self.item_pdf_all.shape}")
            
        except Exception as e:
            logger.error(f"❌ 상품 데이터 로드 실패: {e}")
            import traceback
            logger.error(f"오류 상세: {traceback.format_exc()}")
            raise RuntimeError(
                f"상품 데이터 로드 중 오류 발생. "
                f"데이터 소스({self.offer_info_data_src})를 확인하세요. "
                f"시스템을 초기화할 수 없습니다. 원인: {e}"
            ) from e

    # Database methods moved to utils/db_utils.py

    def _load_stopwords(self):
        """정지어 목록 로드"""
        try:
            self.stop_item_names = pd.read_csv(getattr(METADATA_CONFIG, 'stop_items_path', './data/stop_words.csv'))['stop_words'].to_list()
            logger.info(f"정지어 로드 완료: {len(self.stop_item_names)}개")
        except Exception as e:
            logger.error(f"❌ 정지어 로드 실패: {e}")
            logger.error(f"오류 상세: {traceback.format_exc()}")
            raise RuntimeError(
                f"정지어 데이터 로드 중 오류 발생. "
                f"파일 경로를 확인하세요. "
                f"시스템을 초기화할 수 없습니다. 원인: {e}"
            ) from e

    def _register_items_in_kiwi(self):
        """Kiwi에 상품명들을 고유명사로 등록"""
        try:
            logger.info("=== Kiwi에 상품명 등록 시작 ===")
            
            # 상품명 별칭 데이터 확인
            if 'item_nm_alias' not in self.item_pdf_all.columns:
                logger.error("item_nm_alias 컬럼이 존재하지 않습니다!")
                return
            
            unique_aliases = self.item_pdf_all['item_nm_alias'].unique()
            logger.info(f"등록할 고유 별칭 수: {len(unique_aliases)}개")
            
            # null이 아닌 유효한 별칭들만 필터링
            valid_aliases = [w for w in unique_aliases if isinstance(w, str) and len(w.strip()) > 0]
            logger.info(f"유효한 별칭 수: {len(valid_aliases)}개")
            
            if len(valid_aliases) > 0:
                sample_aliases = valid_aliases[:5]
                logger.info(f"등록할 별칭 샘플: {sample_aliases}")
            
            registered_count = 0
            failed_count = 0
            
            for w in valid_aliases:
                try:
                    self.kiwi.add_user_word(w, "NNP")
                    registered_count += 1
                except Exception as reg_error:
                    failed_count += 1
                    if failed_count <= 5:  # 처음 5개 실패만 로깅
                        logger.warning(f"Kiwi 등록 실패 - '{w}': {reg_error}")
            
            logger.info(f"Kiwi에 상품명 등록 완료: {registered_count}개 성공, {failed_count}개 실패")
            
        except Exception as e:
            logger.error(f"Kiwi 상품명 등록 실패: {e}")
            logger.error(f"오류 상세: {traceback.format_exc()}")

    def _load_program_data(self):
        """프로그램 분류 정보 로드 및 임베딩 생성"""
        try:
            logger.info("프로그램 분류 정보 로딩 시작...")
            
            if self.offer_info_data_src == "local":
                # 로컬 CSV 파일에서 로드
                self.pgm_pdf = pd.read_csv(getattr(METADATA_CONFIG, 'pgm_info_path', './data/program_info.csv'))
                logger.info(f"로컬 파일에서 프로그램 정보 로드: {len(self.pgm_pdf)}개")
            elif self.offer_info_data_src == "db":
                # 데이터베이스에서 로드
                self.pgm_pdf = load_program_from_database()
                logger.info(f"데이터베이스에서 프로그램 정보 로드: {len(self.pgm_pdf)}개")
            
            # clue_tag 보강: 원본 데이터(CSV/DB)를 수정하지 않고 런타임에 추가
            if not self.pgm_pdf.empty:
                self._enhance_pgm_clue_tags()

            # 프로그램 분류를 위한 임베딩 생성
            if not self.pgm_pdf.empty:
                logger.info("프로그램 분류 임베딩 생성 시작...")
                clue_texts = self.pgm_pdf[["pgm_nm","clue_tag"]].apply(
                    lambda x: preprocess_text(x['pgm_nm'].lower()) + " " + x['clue_tag'].lower(), axis=1
                ).tolist()

                if self.emb_model is not None:
                    self.clue_embeddings = self.emb_model.encode(
                        clue_texts, convert_to_tensor=True, show_progress_bar=False
                    )
                else:
                    logger.warning("임베딩 모델이 없어 빈 tensor 사용")
                    self.clue_embeddings = torch.empty((0, 768))

                logger.info(f"프로그램 분류 임베딩 생성 완료: {len(self.pgm_pdf)}개 프로그램")
            else:
                logger.warning("프로그램 데이터가 비어있어 임베딩을 생성할 수 없습니다")
                self.clue_embeddings = torch.tensor([])
            
        except Exception as e:
            logger.error(f"❌ 프로그램 데이터 로드 실패: {e}")
            logger.error(f"오류 상세: {traceback.format_exc()}")
            raise RuntimeError(
                f"프로그램 데이터 로드 중 오류 발생. "
                f"데이터 소스({self.offer_info_data_src})를 확인하세요. "
                f"시스템을 초기화할 수 없습니다. 원인: {e}"
            ) from e

    def _load_organization_data(self):
        """조직/매장 정보 로드"""
        try:
            logger.info(f"=== 조직 정보 로드 시작 (모드: {self.offer_info_data_src}) ===")
            
            if self.offer_info_data_src == "local":
                # 로컬 CSV 파일에서 로드
                logger.info("로컬 CSV 파일에서 조직 정보 로드 중...")
                csv_path = getattr(METADATA_CONFIG, 'org_info_path', './data/org_info_all_250605.csv')
                logger.info(f"CSV 파일 경로: {csv_path}")
                
                org_pdf_raw = pd.read_csv(csv_path)
                logger.info(f"로컬 CSV에서 로드된 원본 조직 데이터 크기: {org_pdf_raw.shape}")
                logger.info(f"로컬 CSV 원본 컬럼들: {list(org_pdf_raw.columns)}")
                
                # ITEM_DMN='R' 조건으로 필터링
                if 'ITEM_DMN' in org_pdf_raw.columns:
                    self.org_pdf = org_pdf_raw.query("ITEM_DMN=='R'").copy()
                elif 'item_dmn' in org_pdf_raw.columns:
                    self.org_pdf = org_pdf_raw.query("item_dmn=='R'").copy()
                else:
                    logger.warning("ITEM_DMN/item_dmn 컬럼을 찾을 수 없어 전체 데이터를 사용합니다.")
                    self.org_pdf = org_pdf_raw.copy()
                
                # 컬럼명을 소문자로 리네임
                self.org_pdf = self.org_pdf.rename(columns={c: c.lower() for c in self.org_pdf.columns})
                
                logger.info(f"로컬 모드: ITEM_DMN='R' 필터링 후 데이터 크기: {self.org_pdf.shape}")
                
            elif self.offer_info_data_src == "db":
                # 데이터베이스에서 로드
                logger.info("데이터베이스에서 조직 정보 로드 중...")
                self._load_org_from_database()
            
            # 데이터 샘플 확인
            if not self.org_pdf.empty:
                sample_orgs = self.org_pdf.head(3).to_dict('records')
                logger.info(f"조직 데이터 샘플 (3개 행): {sample_orgs}")
            
            logger.info(f"=== 조직 정보 로드 최종 완료: {len(self.org_pdf)}개 조직 ===")
            logger.info(f"최종 조직 데이터 스키마: {list(self.org_pdf.columns)}")
            
            # 조직 데이터 최종 검증
            if not self.org_pdf.empty:
                critical_org_columns = ['item_nm', 'item_id']
                missing_org_columns = [col for col in critical_org_columns if col not in self.org_pdf.columns]
                if missing_org_columns:
                    logger.error(f"조직 데이터에서 중요 컬럼이 누락되었습니다: {missing_org_columns}")
                    logger.error("이로 인해 조직/매장 추출 기능이 정상 동작하지 않을 수 있습니다.")
                else:
                    logger.info("모든 중요 조직 컬럼이 정상적으로 로드되었습니다.")
            else:
                logger.warning("조직 데이터가 비어있습니다. 조직/매장 추출이 동작하지 않을 수 있습니다.")
            
        except Exception as e:
            logger.error(f"❌ 조직 데이터 로드 실패: {e}")
            logger.error(f"오류 상세: {traceback.format_exc()}")
            raise RuntimeError(
                f"조직 데이터 로드 중 오류 발생. "
                f"데이터 소스({self.offer_info_data_src})를 확인하세요. "
                f"시스템을 초기화할 수 없습니다. 원인: {e}"
            ) from e

    def _load_org_from_database(self):
        """데이터베이스에서 조직 정보 로드 (ITEM_DMN='R')"""
        self.org_pdf = load_org_from_database()

    def _store_prompt_for_preview(self, prompt: str, prompt_type: str):
        """프롬프트를 미리보기용으로 저장 (PromptManager 사용)"""
        PromptManager.store_prompt_for_preview(prompt, prompt_type)


    def _safe_llm_invoke(self, prompt: str, max_retries: int = 3) -> str:
        """안전한 LLM 호출 메소드"""
        for attempt in range(max_retries):
            try:
                # LLM 호출
                response = self.llm_model.invoke(prompt)
                result_text = response.content if hasattr(response, 'content') else str(response)
                
                # 스키마 응답 감지
                json_objects_list = extract_json_objects(result_text)
                if json_objects_list:
                    json_objects = json_objects_list[-1]
                    if self._is_schema_response(json_objects):
                        logger.warning(f"시도 {attempt + 1}: LLM이 스키마를 반환했습니다. 재시도합니다.")
                        
                        # 스키마 응답인 경우 더 강한 지시사항으로 재시도
                        if attempt < max_retries - 1:
                            enhanced_prompt = self._enhance_prompt_for_retry(prompt)
                            response = self.llm_model.invoke(enhanced_prompt)
                            result_text = response.content if hasattr(response, 'content') else str(response)
                
                return result_text
                
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"LLM 호출 최종 실패: {e}")
                    return self._fallback_extraction(prompt)
                else:
                    logger.warning(f"LLM 호출 재시도 {attempt + 1}/{max_retries}: {e}")
                    time.sleep(2 ** attempt)  # 지수 백오프
        
        return ""

    def _enhance_prompt_for_retry(self, original_prompt: str) -> str:
        """스키마 응답 방지를 위한 프롬프트 강화"""
        return enhance_prompt_for_retry(original_prompt)

    def _fallback_extraction(self, prompt: str) -> str:
        """LLM 실패 시 fallback 추출 로직"""
        logger.info("Fallback 추출 로직 실행")
        
        # 외부 프롬프트 모듈에서 fallback 결과 가져오기
        fallback_result = get_fallback_result()
        
        return json.dumps(fallback_result, ensure_ascii=False)


    # _extract_entities and _classify_programs removed (moved to services)

    def _build_extraction_prompt(self, msg: str, rag_context: str, product_element: Optional[List[Dict]]) -> str:
        """추출용 프롬프트 구성 - 외부 프롬프트 모듈 사용"""

        # 외부 프롬프트 모듈의 함수 사용
        prompt = build_extraction_prompt(
            message=msg,
            rag_context=rag_context,
            product_element=product_element,
            product_info_extraction_mode=self.product_info_extraction_mode,
            num_select_pgms=self.num_select_pgms,
        )
        
        # 디버깅을 위한 프롬프트 로깅 (LLM 모드에서만)
        if self.product_info_extraction_mode == 'llm':
            logger.debug(f"LLM 모드 프롬프트 길이: {len(prompt)} 문자")
            logger.debug(f"후보 상품 목록 포함 여부: {'참고용 후보 상품 이름 목록' in rag_context}")
            
        return prompt

    # _extract_channels removed (moved to ResultBuilder)

    # _match_store_info removed (moved to services)

    def _validate_extraction_result(self, result: Dict) -> Dict:
        """추출 결과 검증 및 정리"""
        try:
            # 필수 필드 확인
            required_fields = ['title', 'purpose', 'sales_script', 'product', 'channel', 'offer']
            for field in required_fields:
                if field not in result:
                    logger.warning(f"필수 필드 누락: {field}")
                    if field == 'title':
                        result[field] = "광고 메시지"
                    elif field == 'sales_script':
                        result[field] = ""
                    elif field == 'offer':
                        result[field] = {"type": "product", "value": []}
                    else:
                        result[field] = []

            # 채널 정보 검증
            validated_channels = []
            for channel in result.get('channel', []):
                if isinstance(channel, dict) and channel.get('value'):
                    validated_channels.append(channel)
            
            result['channel'] = validated_channels
            
            # offer 정보 검증
            if not isinstance(result.get('offer'), dict):
                logger.warning("offer 필드가 딕셔너리가 아님, 기본값으로 설정")
                result['offer'] = {"type": "product", "value": []}
            elif 'type' not in result['offer'] or 'value' not in result['offer']:
                logger.warning("offer 필드에 type 또는 value가 없음, 기본값으로 설정")
                result['offer'] = {"type": "product", "value": result.get('product', [])}

            return result
            
        except Exception as e:
            logger.error(f"결과 검증 실패: {e}")
            return result

    @log_performance
    def process_message(self, message: str, message_id: str = '#') -> Dict[str, Any]:
        """
        MMS 메시지 전체 처리 (Workflow 기반)
        
        Args:
            message: 처리할 MMS 메시지 텍스트
        
        Returns:
            dict: 추출된 정보가 담긴 JSON 구조
        """
        try:
            # 초기 상태 생성 (typed dataclass)
            initial_state = WorkflowState(
                mms_msg=message,
                extractor=self,
                message_id=message_id # message_id 추가
            )
            
            # Workflow 실행
            final_state = self.workflow_engine.run(initial_state)
            
            # Fallback 처리
            if final_state.get("is_fallback"):
                logger.warning("Workflow에서 Fallback 결과 반환")
                return self._create_fallback_result(message, message_id) # message_id 전달
            
            # 결과 추출
            final_result = final_state.get("final_result", {})
            raw_result = final_state.get("raw_result", {})

            # message_id를 결과에 포함
            final_result['message_id'] = message_id
            raw_result['message_id'] = message_id

            # Step 7 엔티티 추출 결과 (ent_result)
            ent_result = {}
            extracted_entities = final_state.get("extracted_entities")
            if extracted_entities:
                ent_result = {
                    'entities': extracted_entities.get('entities', []),
                    'context_text': extracted_entities.get('context_text', ''),
                    'entity_roles': extracted_entities.get('entity_roles', {}),
                    'message_id': message_id,
                }
            kg_metadata = final_state.get("kg_metadata")
            if kg_metadata:
                ent_result['kg_metadata'] = kg_metadata

            # 프롬프트 정보 가져오기
            actual_prompts = PromptManager.get_stored_prompts_from_thread()

            return {
                "ext_result": final_result,
                "raw_result": raw_result,
                "ent_result": ent_result,
                "prompts": actual_prompts
            }
            
        except Exception as e:
            logger.error(f"메시지 처리 실패: {e}")
            logger.error(traceback.format_exc())
            return self._create_fallback_result(message, message_id) # message_id 전달
    
    @log_performance
    def extract_json_objects_only(self, mms_msg: str) -> Dict[str, Any]:
        """
        메시지에서 7단계(엔티티 매칭 및 최종 결과 구성) 전의 json_objects만 추출
        
        Args:
            mms_msg: 처리할 MMS 메시지
            
        Returns:
            Dict: LLM이 생성한 json_objects (엔티티 매칭 전)
        """
        try:
            msg = mms_msg.strip()
            logger.info(f"JSON 객체 추출 시작 - 메시지 길이: {len(msg)}자")
            
            # 1-4단계: 기존 프로세스
            pgm_info = self._classify_program(msg)
            
            # RAG 컨텍스트 준비 (product_info_extraction_mode가 'rag'인 경우)
            # TODO: This method is outdated and needs refactoring to use the workflow-based approach
            # The _prepare_rag_context method no longer exists - it was moved to ContextPreparationStep
            rag_context = ""
            # if self.product_info_extraction_mode == 'rag':
            #     rag_context = self._prepare_rag_context(msg)  # This function doesn't exist
            
            # 5단계: 프롬프트 구성 및 LLM 호출
            # Note: _build_extraction_prompt expects (msg, rag_context, product_element)
            prompt = self._build_extraction_prompt(msg, rag_context, None)
            result_json_text = self._safe_llm_invoke(prompt)
            
            # 6단계: JSON 파싱
            json_objects_list = extract_json_objects(result_json_text)
            
            if not json_objects_list:
                logger.warning("LLM이 유효한 JSON 객체를 반환하지 않았습니다")
                return {}
            
            json_objects = json_objects_list[-1]
            
            # 스키마 응답 감지
            is_schema_response = self._is_schema_response(json_objects)
            if is_schema_response:
                logger.warning("LLM이 스키마 정의를 반환했습니다")
                return {}
            
            logger.info(f"JSON 객체 추출 완료 - 키: {list(json_objects.keys())}")
            return json_objects
            
        except Exception as e:
            logger.error(f"JSON 객체 추출 중 오류 발생: {e}")
            return {}
    
    def _classify_program(self, mms_msg: str) -> Dict[str, Any]:
        """프로그램 분류 수행 (ProgramClassifier.classify와 동일한 로직)"""
        try:
            if self.emb_model is None or self.clue_embeddings.numel() == 0:
                return {"pgm_cand_info": "", "similarities": []}
            
            # 메시지 임베딩 및 프로그램 분류 유사도 계산
            mms_embedding = self.emb_model.encode([mms_msg.lower()], convert_to_tensor=True, show_progress_bar=False)
            similarities = torch.nn.functional.cosine_similarity(mms_embedding, self.clue_embeddings, dim=1).cpu().numpy()
            
            # 상위 후보 프로그램들 선별
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
            logger.error(f"프로그램 분류 실패: {e}")
            return {"pgm_cand_info": "", "similarities": [], "pgm_pdf_tmp": pd.DataFrame()}

    def _is_schema_response(self, json_objects: Dict) -> bool:
        """LLM 응답이 스키마 정의인지 확인"""
        try:
            # purpose 필드가 스키마 구조인지 확인
            purpose = json_objects.get('purpose', {})
            if isinstance(purpose, dict) and 'type' in purpose and purpose.get('type') == 'array':
                logger.warning("purpose 필드가 스키마 구조로 감지됨")
                return True
            
            # product 필드가 스키마 구조인지 확인  
            product = json_objects.get('product', {})
            if isinstance(product, dict) and 'type' in product and product.get('type') == 'array':
                logger.warning("product 필드가 스키마 구조로 감지됨")
                return True
            
            # channel 필드가 스키마 구조인지 확인
            channel = json_objects.get('channel', {})
            if isinstance(channel, dict) and 'type' in channel and channel.get('type') == 'array':
                logger.warning("channel 필드가 스키마 구조로 감지됨")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"스키마 응답 감지 중 오류: {e}")
            return False

    def convert_df_to_json_list(self, df: pd.DataFrame) -> List[Dict]:
        """
        DataFrame을 특정 JSON 구조로 변환
        새로운 스키마: item_nm 기준으로 그룹화하고 모든 item_name_in_msg를 배열로 수집
        
        Schema:
        {
            "item_nm": "상품명",
            "item_id": ["ID1", "ID2"],
            "item_name_in_msg": ["메시지내표현1", "메시지내표현2"]
        }
        """
        result = []
        # item_nm 기준으로 그룹화
        grouped = df.groupby('item_nm')
        for item_nm, group in grouped:
            # 메인 아이템 딕셔너리 생성
            item_name_in_msg_raw = list(group['item_name_in_msg'].unique())
            item_dict = {
                'item_nm': item_nm,
                'item_id': list(group['item_id'].unique()),
                'item_name_in_msg': select_most_comprehensive(item_name_in_msg_raw)
            }
            result.append(item_dict)
        return result

    def _create_fallback_result(self, msg: str, message_id: str = '#') -> Dict[str, Any]:
        """처리 실패 시 기본 결과 생성"""
        fallback = {
            "message_id": message_id,
            "title": "광고 메시지",
            "purpose": ["정보 제공"],
            "sales_script": "",
            "product": [],
            "channel": [],
            "pgm": [],
            "offer": {"type": "product", "value": []},
            "entity_dag": []
        }
        return {
            "ext_result": fallback,
            "raw_result": fallback.copy(),
            "ent_result": {"message_id": message_id},
            "prompts": {}
        }

    # _build_final_result and _map_program_classification removed (moved to ResultBuilder)



def process_message_worker(extractor, message: str, extract_dag: bool = False, message_id: str = '#') -> Dict[str, Any]:
    """
    단일 메시지를 처리하는 워커 함수 (멀티프로세스용)
    
    Args:
        extractor: MMSExtractor 인스턴스
        message: 처리할 메시지
        extract_dag: DAG 추출 여부
        message_id: 메시지 ID (선택 사항)
    
    Returns:
        dict: 처리 결과 (프롬프트 정보 포함)
    """
    try:
        # 스레드 로컬 프롬프트 저장소 초기화 (배치 처리 시 스레드 재사용 문제 방지)
        PromptManager.clear_stored_prompts()
        
        logger.info(f"워커 프로세스에서 메시지 처리 시작: {message[:50]}...")

        # 1. 메인 추출
        result = extractor.process_message(message, message_id) # message_id 전달
        dag_list = []
        
        if extract_dag:
            # Check if DAG was already extracted by the workflow
            existing_dag = result.get('ext_result', {}).get('entity_dag')
            
            if existing_dag and len(existing_dag) > 0:
                logger.info("✅ 워크플로우에서 이미 추출된 DAG가 존재합니다. 중복 추출을 건너뜁니다.")
                dag_list = existing_dag
            else:
                # 순차적 처리로 변경 (프롬프트 캡처를 위해)
                # 멀티스레드를 사용하면 스레드 로컬 저장소가 분리되어 프롬프트 캡처가 안됨
                logger.info("순차적 처리로 메인 추출 및 DAG 추출 수행")
                
                # 2. DAG 추출
                dag_result = make_entity_dag(message, extractor.llm_model, message_id=message_id)
                dag_list = sorted([d for d in dag_result['dag_section'].split('\n') if d!=''])

        extracted_result = result.get('ext_result', {})
        extracted_result['entity_dag'] = dag_list
        result['ext_result'] = extracted_result

        raw_result = result.get('raw_result', {})
        raw_result['entity_dag'] = dag_list
        result['raw_result'] = raw_result

        # ent_result는 process_message에서 이미 설정됨
        result['error'] = ""

        logger.info(f"워커 프로세스에서 메시지 처리 완료")
        return result

    except Exception as e:
        logger.error(f"워커 프로세스에서 메시지 처리 실패: {e}")
        return {
            "ent_result": {"message_id": message_id},
            "ext_result": {
                "message_id": message_id, # message_id 추가
                "title": "처리 실패",
                "purpose": ["오류"],
                "sales_script": "",
                "product": [],
                "channel": [],
                "pgm": [],
                "offer": {"type": "product", "value": []},
                "entity_dag": []
            },
            "raw_result": {
                "message_id": message_id, # message_id 추가
            },
            "prompts": {},
            "error": str(e)
        }

def process_messages_batch(extractor, messages: List[Union[str, Dict[str, Any]]], extract_dag: bool = False, max_workers: int = None) -> List[Dict[str, Any]]:
    """
    여러 메시지를 배치로 처리하는 함수
    
    Args:
        extractor: MMSExtractor 인스턴스
        messages: 처리할 메시지 리스트 (문자열 또는 {'message': str, 'message_id': str} 딕셔너리)
        extract_dag: DAG 추출 여부
        max_workers: 최대 워커 수 (None이면 CPU 코어 수)
    
    Returns:
        list: 처리 결과 리스트
    """
    from concurrent.futures import as_completed
    
    if max_workers is None:
        max_workers = min(len(messages), os.cpu_count())
    
    logger.info(f"배치 처리 시작: {len(messages)}개 메시지, {max_workers}개 워커")
    
    start_time = time.time()
    results = [None] * len(messages)  # 결과를 원래 순서대로 저장하기 위한 리스트
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 모든 메시지에 대해 작업 제출
        future_to_index = {}
        for idx, item in enumerate(messages):
            if isinstance(item, dict):
                # 딕셔너리인 경우 (batch.py, cli.py JSONL 등에서 전달)
                # 키 이름이 다양할 수 있으므로 확인 (msg/message, msg_id/message_id)
                msg = item.get('message') or item.get('msg') or ''
                msg_id = item.get('message_id') or item.get('msg_id') or f"batch_{idx}"
            else:
                # 문자열인 경우
                msg = str(item)
                msg_id = f"batch_{idx}"
                
            future = executor.submit(process_message_worker, extractor, msg, extract_dag, msg_id)
            future_to_index[future] = (idx, msg_id)
        
        # 완료된 작업들 수집
        completed = 0
        for future in as_completed(future_to_index):
            idx, msg_id = future_to_index[future]
            try:
                result = future.result()
                results[idx] = result
                completed += 1
                logger.info(f"배치 처리 진행률: {completed}/{len(messages)} ({(completed/len(messages)*100):.1f}%)")
            except Exception as e:
                logger.error(f"배치 처리 중 오류 발생 (메시지 {msg_id}): {e}")
                results[idx] = {
                    "ext_result": {
                        "message_id": msg_id,
                        "title": "처리 실패",
                        "purpose": ["오류"],
                        "sales_script": "",
                        "product": [],
                        "channel": [],
                        "pgm": [],
                        "offer": {"type": "product", "value": []},
                        "entity_dag": []
                    },
                    "raw_result": {
                        "message_id": msg_id,
                    },
                    "prompts": {},
                    "error": str(e)
                }
    
    elapsed_time = time.time() - start_time
    logger.info(f"배치 처리 완료: {len(messages)}개 메시지, {elapsed_time:.2f}초")
    logger.info(f"평균 처리 시간: {elapsed_time/len(messages):.2f}초/메시지")
    
    return results

def make_entity_dag(msg: str, llm_model, save_dag_image=True, message_id: str = '#'):

    # 메시지에서 엔티티 간의 관계를 방향성 있는 그래프로 추출
    # 예: (고객:가입) -[하면]-> (혜택:수령) -[통해]-> (만족도:향상)
    extract_dag_result = {}
    logger.info("=" * 30 + " DAG 추출 시작 " + "=" * 30)
    try:
        dag_start_time = time.time()
        # DAG 추출 함수 호출 (entity_dag_extractor.py)
        extract_dag_result = extract_dag(DAGParser(), msg, llm_model)
        dag_raw = extract_dag_result['dag_raw']      # LLM 원본 응답
        dag_section = extract_dag_result['dag_section']  # 파싱된 DAG 텍스트
        dag = extract_dag_result['dag']             # NetworkX 그래프 객체
        
        # 시각적 다이어그램 생성 (utils.py)
        dag_filename = ""
        if save_dag_image:
            dag_filename = f'dag_{message_id}_{sha256_hash(msg)}'
            create_dag_diagram(dag, filename=dag_filename)
            logger.info(f"✅ DAG 추출 완료: {dag_filename}")

        extract_dag_result['dag_filename'] = dag_filename
        
        dag_processing_time = time.time() - dag_start_time
        
        logger.info(f"🕒 DAG 처리 시간: {dag_processing_time:.3f}초")
        logger.info(f"📏 DAG 섹션 길이: {len(dag_section)}자")
        if dag_section:
            logger.info(f"📄 DAG 내용 미리보기: {dag_section[:200]}...")
        else:
            logger.warning("⚠️ DAG 섹션이 비어있습니다")
            
    except Exception as e:
        logger.error(f"❌ DAG 추출 중 오류 발생: {e}")
        dag_section = ""

    return extract_dag_result


def get_stored_prompts_from_thread():
    """현재 스레드에서 저장된 프롬프트 정보를 가져오는 함수 (PromptManager 사용)"""
    return PromptManager.get_stored_prompts_from_thread()

def save_result_to_mongodb_if_enabled(message: str, result: dict, args_or_data, extractor=None):
    """MongoDB 저장이 활성화된 경우 결과를 저장하는 도우미 함수
    
    Args:
        message: 처리할 메시지
        result: 처리 결과 (extracted_result, raw_result 포함)
        args_or_data: argparse.Namespace 객체 또는 딕셔너리
        extractor: MMSExtractor 인스턴스 (선택적)
    
    Returns:
        str: 저장된 문서 ID, 실패 시 None
    """
    # args_or_data가 딕셔너리인 경우 Namespace로 변환
    if isinstance(args_or_data, dict):
        import argparse
        args = argparse.Namespace(**args_or_data)
    else:
        args = args_or_data
    
    # save_to_mongodb 속성이 없거나 False인 경우
    if not getattr(args, 'save_to_mongodb', False):
        return None
        
    try:
        # MongoDB 임포트 시도
        from utils.mongodb_utils import save_to_mongodb
        
        # 스레드 로컬 저장소에서 프롬프트 정보 가져오기
        stored_prompts = result.get('prompts', get_stored_prompts_from_thread()) 
        
        # 프롬프트 정보 구성
        prompts_data = {}
        for key, prompt_data in stored_prompts.items():
            prompts_data[key] = {
                'title': prompt_data.get('title', f'{key} 프롬프트'),
                'description': prompt_data.get('description', f'{key} 처리를 위한 프롬프트'),
                'content': prompt_data.get('content', ''),
                'length': len(prompt_data.get('content', ''))
            }
        
        # 저장된 프롬프트가 없는 경우 기본값 사용
        if not prompts_data:
            prompts_data = {
                'main_extraction_prompt': {
                    'title': '메인 정보 추출 프롬프트',
                    'description': 'MMS 메시지에서 기본 정보 추출',
                    'content': '실제 프롬프트 내용이 저장되지 않았습니다.',
                    'length': 0
                }
            }
        
        extraction_prompts = {
            'success': True,
            'prompts': prompts_data,
            'settings': {
                'llm_model': getattr(args, 'llm_model', 'unknown'),
                'offer_data_source': getattr(args, 'offer_data_source', getattr(args, 'offer_info_data_src', 'unknown')),
                'product_info_extraction_mode': getattr(args, 'product_info_extraction_mode', 'unknown'),
                'entity_matching_mode': getattr(args, 'entity_matching_mode', getattr(args, 'entity_extraction_mode', 'unknown')),
                'extract_entity_dag': getattr(args, 'extract_entity_dag', False)
            }
        }
        
        # 추출 결과를 MongoDB 형식으로 구성
        extraction_result = {
            'success': not bool(result.get('error')),
            'result': result.get('ext_result', result.get('result', {})),
            'metadata': {
                'processing_time_seconds': result.get('processing_time', 0),
                'processing_mode': getattr(args, 'processing_mode', 'single'),
                'model_used': getattr(args, 'llm_model', 'unknown')
            }
        }

        raw_result_data = {
            'success': not bool(result.get('error')),
            'result': result.get('raw_result', {}),
            'metadata': {
                'processing_time_seconds': result.get('processing_time', 0),
                'processing_mode': getattr(args, 'processing_mode', 'single'),
                'model_used': getattr(args, 'llm_model', 'unknown')
            }
        }
        
        # ent_result (Step 7 엔티티 추출 결과)
        ent_result = result.get('ent_result', {})

        # MongoDB에 저장
        user_id = getattr(args, 'user_id', 'DEFAULT_USER')
        # result에서 message_id 추출 (ext_result 또는 raw_result에서)
        message_id = result.get('ext_result', {}).get('message_id') or result.get('raw_result', {}).get('message_id') or '#'
        saved_id = save_to_mongodb(message, extraction_result, raw_result_data, extraction_prompts,
                         user_id=user_id, message_id=message_id, ent_result=ent_result)
        
        if saved_id:
            print(f"📄 결과가 MongoDB에 저장되었습니다. (ID: {saved_id[:8]}...)")
            return saved_id
        else:
            print("⚠️ MongoDB 저장에 실패했습니다.")
            return None
            
    except ImportError:
        print("❌ MongoDB 저장이 요청되었지만 mongodb_utils를 찾을 수 없습니다.")
        return None
    except Exception as e:
        print(f"❌ MongoDB 저장 중 오류 발생: {str(e)}")
        return None

# CLI interface moved to cli.py
# To run from command line: python cli.py --help
# %%
