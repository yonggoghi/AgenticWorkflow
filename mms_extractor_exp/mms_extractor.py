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

📊 성능 지표
-----------
- 평균 처리 시간: ~30초/메시지
- 정확도: 85%+ (수동 검증 기준)
- 처리량: ~120 메시지/시간 (단일 프로세스)

작성자: MMS 분석팀
최종 수정: 2024-09
버전: 2.0.0
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
from langchain.prompts import PromptTemplate

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
from entity_dag_extractor import DAGParser, extract_dag

# 프롬프트 모듈 임포트
from prompts import (
    build_extraction_prompt,
    enhance_prompt_for_retry,
    get_fallback_result,
    build_entity_extraction_prompt,
    DEFAULT_ENTITY_EXTRACTION_PROMPT,
    DETAILED_ENTITY_EXTRACTION_PROMPT,
    SIMPLE_ENTITY_EXTRACTION_PROMPT
    )

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
    fuzzy_similarities,
    get_fuzzy_similarities,
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

class MMSExtractor:
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
    3. **RAG 기반 컬텍스트 증강**: 관련 데이터를 활용한 정확도 향상
    4. **다중 LLM 지원**: OpenAI, Anthropic 등 다양한 모델 지원
    5. **DAG 생성**: 엔티티 간 관계를 방향성 그래프로 시각화
    
    📊 성능 특징
    -----------
    - **정확도**: 85%+ (수동 검증 기준)
    - **처리 속도**: 평균 30초/메시지
    - **확장성**: 모듈화된 설계로 새로운 기능 추가 용이
    - **안정성**: 강화된 예외 처리 및 재시도 메커니즘
    
    ⚙️ 주요 개선사항
    --------------
    - **아키텍처 모듈화**: 대형 메소드를 기능별 모듈로 분리하여 유지보수성 향상
    - **프롬프트 외부화**: 하드코딩된 프롬프트를 별도 모듈로 분리하여 관리 효율성 증대
    - **다층 예외 처리**: LLM API 실패, 네트워크 오류 등에 대한 robust한 에러 복구
    - **상세 로깅**: 성능 모니터링, 디버깅, 감사 로그를 위한 포괄적 로깅 시스템
    - **데이터 검증**: 입력/출력 데이터 품질 보장을 위한 다단계 검증
    - **하이브리드 데이터 소스**: CSV 파일과 Oracle DB를 모두 지원하는 유연한 데이터 로딩
    
    📝 사용 예시
    -----------
    ```python
    # 기본 초기화
    extractor = MMSExtractor(
        llm_model='ax',
        entity_extraction_mode='llm',
        extract_entity_dag=True
    )
    
    # 메시지 처리
    result = extractor.process_message("샘플 MMS 텍스트")
    
    # 결과 활용
    products = result['product']
    channels = result['channel']
    entity_dag = result.get('entity_dag', [])
    ```
    
    💼 의존성
    ---------
    - LangChain (LLM 인터페이스)
    - SentenceTransformers (임베딩)
    - KiwiPiePy (NLP)
    - cx_Oracle (데이터베이스 연동)
    """
    
    def __init__(self, model_path=None, data_dir=None, product_info_extraction_mode=None, 
                 entity_extraction_mode=None, offer_info_data_src='local', llm_model='ax', extract_entity_dag=False):
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
            extract_entity_dag (bool, optional): DAG 추출 여부. 기본값: False
            
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
                                   entity_extraction_mode, offer_info_data_src, llm_model, extract_entity_dag)
            
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
            
            logger.info("✅ MMSExtractor 초기화 완료")
            
        except Exception as e:
            logger.error(f"❌ MMSExtractor 초기화 실패: {e}")
            logger.error(traceback.format_exc())
            raise

    def _set_default_config(self, model_path, data_dir, product_info_extraction_mode, 
                          entity_extraction_mode, offer_info_data_src, llm_model, extract_entity_dag):
        """기본 설정값 적용"""
        self.data_dir = data_dir if data_dir is not None else './data/'
        self.model_path = model_path if model_path is not None else getattr(EMBEDDING_CONFIG, 'ko_sbert_model_path', 'jhgan/ko-sroberta-multitask')
        self.offer_info_data_src = offer_info_data_src
        self.product_info_extraction_mode = product_info_extraction_mode if product_info_extraction_mode is not None else getattr(PROCESSING_CONFIG, 'product_info_extraction_mode', 'nlp')
        self.entity_extraction_mode = entity_extraction_mode if entity_extraction_mode is not None else getattr(PROCESSING_CONFIG, 'entity_extraction_mode', 'llm')
        self.llm_model_name = llm_model
        self.num_cand_pgms = getattr(PROCESSING_CONFIG, 'num_candidate_programs', 5)
        self.extract_entity_dag = extract_entity_dag
        
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
        복수의 LLM 모델을 초기화하는 헬퍼 메서드
        
        Args:
            model_names (List[str]): 초기화할 모델명 리스트 (예: ['ax', 'gpt', 'gen'])
            
        Returns:
            List: 초기화된 LLM 모델 객체 리스트
        """
        llm_models = []
        
        # 모델명 매핑 (기존 LLM 초기화 로직과 동일)
        model_mapping = {
            "cld": getattr(MODEL_CONFIG, 'anthropic_model', 'amazon/anthropic/claude-sonnet-4-20250514'),
            "ax": getattr(MODEL_CONFIG, 'ax_model', 'skt/ax4'),
            "gpt": getattr(MODEL_CONFIG, 'gpt_model', 'azure/openai/gpt-4o-2024-08-06'),
            "gen": getattr(MODEL_CONFIG, 'gemini_model', 'gcp/gemini-2.5-flash')
        }
        
        for model_name in model_names:
            try:
                actual_model_name = model_mapping.get(model_name, model_name)
                
                # 모델별 설정 (기존 로직과 동일)
                model_kwargs = {
                    "temperature": 0.0,
                    "openai_api_key": getattr(API_CONFIG, 'llm_api_key', os.getenv('OPENAI_API_KEY')),
                    "openai_api_base": getattr(API_CONFIG, 'llm_api_url', None),
                    "model": actual_model_name,
                    "max_tokens": getattr(MODEL_CONFIG, 'llm_max_tokens', 4000)
                }
                
                # GPT 모델의 경우 시드 설정
                if 'gpt' in actual_model_name.lower():
                    model_kwargs["seed"] = 42
                
                llm_model = ChatOpenAI(**model_kwargs)
                llm_models.append(llm_model)
                logger.info(f"✅ LLM 모델 초기화 완료: {model_name} ({actual_model_name})")
                
            except Exception as e:
                logger.error(f"❌ LLM 모델 초기화 실패: {model_name} - {e}")
                continue
        
        return llm_models

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
            self._load_and_prepare_item_data()
            logger.info(f"상품 정보 최종 데이터 크기: {self.item_pdf_all.shape}")
            logger.info(f"상품 정보 컬럼들: {list(self.item_pdf_all.columns)}")
            
            # 정지어 로드
            logger.info("2️⃣ 정지어 로드 중...")
            self._load_stop_words()
            logger.info(f"로드된 정지어 수: {len(self.stop_item_names)}개")
            
            # Kiwi에 상품명 등록
            logger.info("3️⃣ Kiwi에 상품명 등록 중...")
            self._register_items_to_kiwi()
            
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

    def _load_and_prepare_item_data(self):
        """상품 정보 로드 및 준비 (ipynb 코드 기준으로 통합)"""
        try:
            logger.info(f"=== 상품 정보 로드 및 준비 시작 (모드: {self.offer_info_data_src}) ===")
            
            # ===== 1단계: 데이터 소스에서 원본 데이터 로드 =====
            if self.offer_info_data_src == "local":
                logger.info("📁 로컬 CSV 파일에서 로드")
                csv_path = getattr(METADATA_CONFIG, 'offer_data_path', './data/items.csv')
                item_pdf_raw = pd.read_csv(csv_path)
            elif self.offer_info_data_src == "db":
                logger.info("🗄️ 데이터베이스에서 로드")
                with self._database_connection() as conn:
                    sql = "SELECT * FROM TCAM_RC_OFER_MST"
                    item_pdf_raw = pd.read_sql(sql, conn)
            
            logger.info(f"원본 데이터 크기: {item_pdf_raw.shape}")
            
            # ===== 2단계: 공통 전처리 (데이터 소스 무관) =====
            # ITEM_DESC를 str로 변환
            item_pdf_raw['ITEM_DESC'] = item_pdf_raw['ITEM_DESC'].astype('str')
            
            # 단말기인 경우 설명을 상품명으로 사용
            item_pdf_raw['ITEM_NM'] = item_pdf_raw.apply(
                lambda x: x['ITEM_DESC'] if x['ITEM_DMN']=='E' else x['ITEM_NM'], axis=1
            )
            
            # 컬럼명을 소문자로 변환
            item_pdf_all = item_pdf_raw.rename(columns={c: c.lower() for c in item_pdf_raw.columns})
            logger.info(f"컬럼명 소문자 변환 완료")
            
            # 추가 컬럼 생성
            item_pdf_all['item_ctg'] = None
            item_pdf_all['item_emb_vec'] = None
            item_pdf_all['ofer_cd'] = item_pdf_all['item_id']
            item_pdf_all['oper_dt_hms'] = '20250101000000'
            
            # 제외할 도메인 코드 필터링
            excluded_domains = getattr(PROCESSING_CONFIG, 'excluded_domain_codes_for_items', [])
            if excluded_domains:
                before_filter = len(item_pdf_all)
                item_pdf_all = item_pdf_all.query("item_dmn not in @excluded_domains")
                logger.info(f"도메인 필터링: {before_filter} -> {len(item_pdf_all)}")
            
            # ===== 3단계: 별칭 규칙 로드 및 처리 (데이터 소스 무관) =====
            logger.info("🔗 별칭 규칙 로드 중...")
            self.alias_pdf_raw = pd.read_csv(getattr(METADATA_CONFIG, 'alias_rules_path', './data/alias_rules.csv'))
            alias_pdf = self.alias_pdf_raw.copy()
            alias_pdf['alias_1'] = alias_pdf['alias_1'].str.split("&&")
            alias_pdf['alias_2'] = alias_pdf['alias_2'].str.split("&&")
            alias_pdf = alias_pdf.explode('alias_1')
            alias_pdf = alias_pdf.explode('alias_2')
            
            # build 타입 별칭 확장
            alias_list_ext = alias_pdf.query("type=='build'")[['alias_1','category','direction','type']].to_dict('records')
            for alias in alias_list_ext:
                adf = item_pdf_all.query(
                    "item_nm.str.contains(@alias['alias_1']) and item_dmn==@alias['category']"
                )[['item_nm','item_desc','item_dmn']].rename(
                    columns={'item_nm':'alias_2','item_desc':'description','item_dmn':'category'}
                ).drop_duplicates()
                adf['alias_1'] = alias['alias_1']
                adf['direction'] = alias['direction']
                adf['type'] = alias['type']
                adf = adf[alias_pdf.columns]
                alias_pdf = pd.concat([alias_pdf.query(f"alias_1!='{alias['alias_1']}'"), adf])
            
            alias_pdf = alias_pdf.drop_duplicates()
            
            # 양방향(B) 별칭 추가
            alias_pdf = pd.concat([
                alias_pdf, 
                alias_pdf.query("direction=='B'").rename(
                    columns={'alias_1':'alias_2', 'alias_2':'alias_1'}
                )[alias_pdf.columns]
            ])
            
            alias_rule_set = list(zip(alias_pdf['alias_1'], alias_pdf['alias_2'], alias_pdf['type']))
            logger.info(f"별칭 규칙 수: {len(alias_rule_set)}개")
            
            # ===== 4단계: 별칭 규칙 연쇄 적용 (병렬 처리) =====
            def apply_alias_rule_cascade_parallel(args_dict):
                """별칭 규칙을 연쇄적으로 적용"""
                item_nm = args_dict['item_nm']
                max_depth = args_dict['max_depth']
                
                processed = set()
                result_dict = {item_nm: '#' * len(item_nm)}
                to_process = [(item_nm, 0, frozenset())]
                
                while to_process:
                    current_item, depth, path_applied_rules = to_process.pop(0)
                    
                    if depth >= max_depth or current_item in processed:
                        continue
                    
                    processed.add(current_item)
                    
                    for r in alias_rule_set:
                        alias_from, alias_to, alias_type = r[0], r[1], r[2]
                        rule_key = (alias_from, alias_to, alias_type)
                        
                        if rule_key in path_applied_rules:
                            continue
                        
                        # 타입에 따른 매칭
                        if alias_type == 'exact':
                            matched = (current_item == alias_from)
                        else:
                            matched = (alias_from in current_item)
                        
                        if matched:
                            new_item = alias_to.strip() if alias_type == 'exact' else current_item.replace(alias_from.strip(), alias_to.strip())
                            
                            if new_item not in result_dict:
                                result_dict[new_item] = alias_from.strip()
                                to_process.append((new_item, depth + 1, path_applied_rules | {rule_key}))
                
                item_nm_list = [{'item_nm': k, 'item_nm_alias': v} for k, v in result_dict.items()]
                adf = pd.DataFrame(item_nm_list)
                selected_alias = select_most_comprehensive(adf['item_nm_alias'].tolist())
                result_aliases = list(adf.query("item_nm_alias in @selected_alias")['item_nm'].unique())
                
                if item_nm not in result_aliases:
                    result_aliases.append(item_nm)
                
                return {'item_nm': item_nm, 'item_nm_alias': result_aliases}
            
            def parallel_alias_rule_cascade(texts, max_depth=5, n_jobs=None):
                """병렬 별칭 규칙 적용"""
                if n_jobs is None:
                    n_jobs = min(os.cpu_count()-1, 4)
                
                batches = [{"item_nm": text, "max_depth": max_depth} for text in texts]
                with Parallel(n_jobs=n_jobs, backend='threading') as parallel:
                    batch_results = parallel(delayed(apply_alias_rule_cascade_parallel)(args) for args in batches)
                
                return pd.DataFrame(batch_results)
            
            logger.info("🔄 별칭 규칙 연쇄 적용 중...")
            item_alias_pdf = parallel_alias_rule_cascade(item_pdf_all['item_nm'], max_depth=3)
            
            # 별칭 병합 및 explode
            item_pdf_all = item_pdf_all.merge(item_alias_pdf, on='item_nm', how='left')
            before_explode = len(item_pdf_all)
            item_pdf_all = item_pdf_all.explode('item_nm_alias').drop_duplicates()
            logger.info(f"별칭 explode: {before_explode} -> {len(item_pdf_all)}")
            
            # ===== 5단계: 사용자 정의 엔티티 추가 =====
            user_defined_entity = ['AIA Vitality', '부스트 파크 건대입구', 'Boost Park 건대입구']
            item_pdf_ext = pd.DataFrame([{
                'item_nm': e, 'item_id': e, 'item_desc': e, 'item_dmn': 'user_defined',
                'start_dt': 20250101, 'end_dt': 99991231, 'rank': 1, 'item_nm_alias': e
            } for e in user_defined_entity])
            item_pdf_all = pd.concat([item_pdf_all, item_pdf_ext])
            
            # ===== 6단계: item_dmn_nm 컬럼 추가 =====
            item_dmn_map = pd.DataFrame([
                {"item_dmn": 'P', 'item_dmn_nm': '요금제 및 관련 상품'},
                {"item_dmn": 'E', 'item_dmn_nm': '단말기'},
                {"item_dmn": 'S', 'item_dmn_nm': '구독 상품'},
                {"item_dmn": 'C', 'item_dmn_nm': '쿠폰'},
                {"item_dmn": 'X', 'item_dmn_nm': '가상 상품'}
            ])
            item_pdf_all = item_pdf_all.merge(item_dmn_map, on='item_dmn', how='left')
            item_pdf_all['item_dmn_nm'] = item_pdf_all['item_dmn_nm'].fillna('기타')
            
            # ===== 7단계: TEST 필터링 =====
            before_test = len(item_pdf_all)
            item_pdf_all = item_pdf_all.query("not item_nm_alias.str.contains('TEST', case=False, na=False)")
            logger.info(f"TEST 필터링: {before_test} -> {len(item_pdf_all)}")
            
            self.item_pdf_all = item_pdf_all
            
            # 최종 확인
            logger.info(f"=== 상품 정보 준비 완료 ===")
            logger.info(f"최종 데이터 크기: {self.item_pdf_all.shape}")
            logger.info(f"최종 컬럼들: {list(self.item_pdf_all.columns)}")
            
            # 중요 컬럼 확인
            critical_columns = ['item_nm', 'item_id', 'item_nm_alias']
            missing_columns = [col for col in critical_columns if col not in self.item_pdf_all.columns]
            if missing_columns:
                logger.error(f"중요 컬럼 누락: {missing_columns}")
            else:
                logger.info("✅ 모든 중요 컬럼 존재")
            
            # 샘플 데이터 확인
            if not self.item_pdf_all.empty:
                logger.info(f"상품명 샘플: {self.item_pdf_all['item_nm'].dropna().head(3).tolist()}")
                logger.info(f"별칭 샘플: {self.item_pdf_all['item_nm_alias'].dropna().head(3).tolist()}")
            
        except Exception as e:
            logger.error(f"상품 정보 로드 및 준비 실패: {e}")
            logger.error(f"오류 상세: {traceback.format_exc()}")
            # 빈 DataFrame으로 fallback
            self.item_pdf_all = pd.DataFrame(columns=['item_nm', 'item_id', 'item_desc', 'item_dmn', 'item_nm_alias'])
            logger.warning("빈 DataFrame으로 fallback 설정됨")

    def _get_database_connection(self):
        """Oracle 데이터베이스 연결 생성"""
        try:
            logger.info("=== 데이터베이스 연결 시도 중 ===")
            
            username = os.getenv("DB_USERNAME")
            password = os.getenv("DB_PASSWORD")
            host = os.getenv("DB_HOST")
            port = os.getenv("DB_PORT")
            service_name = os.getenv("DB_NAME")
            
            # 연결 정보 로깅 (비밀번호는 마스킹)
            logger.info(f"DB 연결 정보:")
            logger.info(f"  - 사용자명: {username if username else '[비어있음]'}")
            logger.info(f"  - 비밀번호: {'*' * len(password) if password else '[비어있음]'}")
            logger.info(f"  - 호스트: {host if host else '[비어있음]'}")
            logger.info(f"  - 포트: {port if port else '[비어있음]'}")
            logger.info(f"  - 서비스명: {service_name if service_name else '[비어있음]'}")
            
            # 환경 변수 확인
            missing_vars = []
            if not username: missing_vars.append('DB_USERNAME')
            if not password: missing_vars.append('DB_PASSWORD')
            if not host: missing_vars.append('DB_HOST')
            if not port: missing_vars.append('DB_PORT')
            if not service_name: missing_vars.append('DB_NAME')
            
            if missing_vars:
                logger.error(f"누락된 환경 변수: {missing_vars}")
                logger.error("필요한 환경 변수들을 .env 파일에 설정해주세요.")
                raise ValueError(f"데이터베이스 연결 정보가 불완전합니다. 누락: {missing_vars}")
            
            # DSN 생성 및 로깅
            logger.info(f"DSN 생성 중: {host}:{port}/{service_name}")
            dsn = cx_Oracle.makedsn(host, port, service_name=service_name)
            logger.info(f"DSN 생성 성공: {dsn}")
            
            # 데이터베이스 연결 시도
            logger.info("데이터베이스 연결 시도 중...")
            conn = cx_Oracle.connect(user=username, password=password, dsn=dsn, encoding="UTF-8")
            logger.info("데이터베이스 연결 성공!")
            
            # LOB 데이터 처리를 위한 outputtypehandler 설정
            def output_type_handler(cursor, name, default_type, size, precision, scale):
                if default_type == cx_Oracle.CLOB:
                    return cursor.var(cx_Oracle.LONG_STRING, arraysize=cursor.arraysize)
                elif default_type == cx_Oracle.BLOB:
                    return cursor.var(cx_Oracle.LONG_BINARY, arraysize=cursor.arraysize)
            
            conn.outputtypehandler = output_type_handler
            
            # 연결 정보 확인
            logger.info(f"연결된 DB 버전: {conn.version}")
            
            return conn
            
        except cx_Oracle.DatabaseError as db_error:
            error_obj, = db_error.args
            logger.error(f"Oracle 데이터베이스 오류:")
            logger.error(f"  - 오류 코드: {error_obj.code}")
            logger.error(f"  - 오류 메시지: {error_obj.message}")
            logger.error(f"  - 전체 오류: {db_error}")
            raise
        except ImportError as import_error:
            logger.error(f"cx_Oracle 모듈 임포트 오류: {import_error}")
            logger.error("코맨드: pip install cx_Oracle")
            raise
        except Exception as e:
            logger.error(f"데이터베이스 연결 실패: {e}")
            logger.error(f"오류 타입: {type(e).__name__}")
            logger.error(f"오류 상세: {traceback.format_exc()}")
            raise

    @contextmanager
    def _database_connection(self):
        """데이터베이스 연결 context manager"""
        conn = None
        start_time = time.time()
        try:
            logger.info("데이터베이스 연결 context manager 시작")
            conn = self._get_database_connection()
            connection_time = time.time() - start_time
            logger.info(f"데이터베이스 연결 완료 ({connection_time:.2f}초)")
            yield conn
        except Exception as e:
            logger.error(f"데이터베이스 작업 중 오류: {e}")
            logger.error(f"오류 타입: {type(e).__name__}")
            logger.error(f"오류 상세: {traceback.format_exc()}")
            raise
        finally:
            if conn:
                try:
                    conn.close()
                    total_time = time.time() - start_time
                    logger.info(f"데이터베이스 연결 정상 종료 (총 소요시간: {total_time:.2f}초)")
                except Exception as close_error:
                    logger.warning(f"연결 종료 중 오류: {close_error}")
            else:
                logger.warning("데이터베이스 연결이 생성되지 않았습니다.")

    def _load_program_from_database(self):
        """데이터베이스에서 프로그램 분류 정보 로드"""
        try:
            logger.info("=== 데이터베이스에서 프로그램 분류 정보 로드 시작 ===")
            
            with self._database_connection() as conn:
                # 프로그램 분류 정보 쿼리
                sql = """SELECT CMPGN_PGM_NUM pgm_id, CMPGN_PGM_NM pgm_nm, RMK clue_tag 
                         FROM TCAM_CMPGN_PGM_INFO
                         WHERE DEL_YN = 'N' 
                         AND APRV_OP_RSLT_CD = 'APPR'
                         AND EXPS_YN = 'Y'
                         AND CMPGN_PGM_NUM like '2025%' 
                         AND RMK is not null"""
                
                logger.info(f"실행할 SQL: {sql}")
                
                self.pgm_pdf = pd.read_sql(sql, conn)
                logger.info(f"DB에서 로드된 프로그램 데이터 크기: {self.pgm_pdf.shape}")
                logger.info(f"DB에서 로드된 프로그램 컬럼들: {list(self.pgm_pdf.columns)}")
                
                # 컬럼명 소문자 변환
                original_columns = list(self.pgm_pdf.columns)
                self.pgm_pdf = self.pgm_pdf.rename(columns={c:c.lower() for c in self.pgm_pdf.columns})
                logger.info(f"프로그램 컬럼명 변환: {dict(zip(original_columns, self.pgm_pdf.columns))}")
                
                # LOB 데이터가 있는 경우를 대비해 데이터 강제 로드
                if not self.pgm_pdf.empty:
                    try:
                        # DataFrame의 모든 데이터를 메모리로 강제 로드
                        _ = self.pgm_pdf.values  # 모든 데이터 접근하여 LOB 로드 유도
                        
                        # 프로그램 데이터 샘플 확인
                        if 'pgm_nm' in self.pgm_pdf.columns:
                            sample_pgms = self.pgm_pdf['pgm_nm'].dropna().head(3).tolist()
                            logger.info(f"프로그램명 샘플: {sample_pgms}")
                        
                        if 'clue_tag' in self.pgm_pdf.columns:
                            sample_clues = self.pgm_pdf['clue_tag'].dropna().head(3).tolist()
                            logger.info(f"클루 태그 샘플: {sample_clues}")
                            
                        logger.info(f"데이터베이스에서 프로그램 분류 정보 로드 완료: {len(self.pgm_pdf)}개")
                    except Exception as load_error:
                        logger.error(f"프로그램 데이터 강제 로드 중 오류: {load_error}")
                        raise
                else:
                    logger.warning("로드된 프로그램 데이터가 비어있습니다!")
            
        except Exception as e:
            logger.error(f"프로그램 분류 정보 데이터베이스 로드 실패: {e}")
            logger.error(f"오류 상세: {traceback.format_exc()}")
            # 빈 데이터로 fallback
            self.pgm_pdf = pd.DataFrame(columns=['pgm_nm', 'clue_tag', 'pgm_id'])
            raise

    def _load_stop_words(self):
        """정지어 목록 로드"""
        try:
            self.stop_item_names = pd.read_csv(getattr(METADATA_CONFIG, 'stop_items_path', './data/stop_words.csv'))['stop_words'].to_list()
            logger.info(f"정지어 로드 완료: {len(self.stop_item_names)}개")
        except Exception as e:
            logger.warning(f"정지어 로드 실패: {e}")
            self.stop_item_names = []

    def _register_items_to_kiwi(self):
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
                self._load_program_from_database()
                logger.info(f"데이터베이스에서 프로그램 정보 로드: {len(self.pgm_pdf)}개")
            
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
            logger.error(f"프로그램 데이터 로드 실패: {e}")
            # 빈 데이터로 fallback
            self.pgm_pdf = pd.DataFrame(columns=['pgm_nm', 'clue_tag', 'pgm_id'])
            self.clue_embeddings = torch.tensor([])

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
            logger.error(f"조직 정보 로드 실패: {e}")
            logger.error(f"오류 상세: {traceback.format_exc()}")
            # 빈 DataFrame으로 fallback (조직 데이터에 필요한 컬럼들 포함)
            self.org_pdf = pd.DataFrame(columns=['item_nm', 'item_id', 'item_desc', 'item_dmn'])
            logger.warning("빈 조직 DataFrame으로 fallback 설정됨")
            logger.warning("이로 인해 조직/매장 추출 기능이 비활성화됩니다.")

    def _load_org_from_database(self):
        """데이터베이스에서 조직 정보 로드 (ITEM_DMN='R')"""
        try:
            logger.info("데이터베이스 연결 시도 중...")
            
            with self._database_connection() as conn:
                sql = "SELECT * FROM TCAM_RC_OFER_MST WHERE ITEM_DMN='R'"
                logger.info(f"실행할 SQL: {sql}")
                
                self.org_pdf = pd.read_sql(sql, conn)
                logger.info(f"DB에서 로드된 조직 데이터 크기: {self.org_pdf.shape}")
                logger.info(f"DB 조직 데이터 컬럼들: {list(self.org_pdf.columns)}")
                
                # 컬럼명 매핑 및 소문자 변환
                original_columns = list(self.org_pdf.columns)
                logger.info(f"DB 조직 데이터 원본 컬럼들: {original_columns}")
                
                # 조직 데이터를 위한 컬럼 매핑 (동일한 테이블이지만 사용 목적이 다름)
                column_mapping = {c: c.lower() for c in self.org_pdf.columns}
                
                # 조직 데이터는 item 테이블과 동일한 스키마를 사용하므로 컬럼명 그대로 사용
                # ITEM_NM -> item_nm, ITEM_ID -> item_id, ITEM_DESC -> item_desc 등
                
                self.org_pdf = self.org_pdf.rename(columns=column_mapping)
                logger.info(f"DB 모드 조직 컬럼명 매핑 완료: {dict(zip(original_columns, self.org_pdf.columns))}")
                logger.info(f"DB 모드 조직 최종 컬럼들: {list(self.org_pdf.columns)}")
                
                # 데이터 샘플 확인 및 컬럼 존재 여부 검증
                if not self.org_pdf.empty:
                    logger.info(f"DB 모드 조직 데이터 최종 크기: {self.org_pdf.shape}")
                    
                    # 필수 컬럼 존재 여부 확인
                    required_columns = ['item_nm', 'item_id']
                    missing_columns = [col for col in required_columns if col not in self.org_pdf.columns]
                    if missing_columns:
                        logger.error(f"DB 모드 조직 데이터에서 필수 컬럼 누락: {missing_columns}")
                        logger.error(f"사용 가능한 컬럼들: {list(self.org_pdf.columns)}")
                    else:
                        logger.info("모든 필수 조직 컬럼이 존재합니다.")
                    
                    # 샘플 데이터 확인
                    if 'item_nm' in self.org_pdf.columns:
                        sample_orgs = self.org_pdf['item_nm'].dropna().head(5).tolist()
                        logger.info(f"DB 모드 조직명 샘플: {sample_orgs}")
                    else:
                        logger.error("item_nm 컬럼이 없어 샘플을 표시할 수 없습니다.")
                        # 전체 데이터 샘플 표시
                        sample_data = self.org_pdf.head(3).to_dict('records')
                        logger.info(f"DB 모드 조직 데이터 샘플: {sample_data}")
                else:
                    logger.warning("DB에서 로드된 조직 데이터가 비어있습니다!")
                
                logger.info(f"DB에서 조직 데이터 로드 성공: {len(self.org_pdf)}개 조직")
                
        except Exception as e:
            logger.error(f"DB에서 조직 데이터 로드 실패: {e}")
            logger.error(f"DB 조직 로드 오류 상세: {traceback.format_exc()}")
            
            # 빈 DataFrame으로 fallback (조직 데이터에 필요한 컬럼들 포함)
            self.org_pdf = pd.DataFrame(columns=['item_nm', 'item_id', 'item_desc', 'item_dmn'])
            logger.warning("조직 데이터 DB 로드 실패로 빈 DataFrame 사용")
            
            raise

    def _store_prompt_for_preview(self, prompt: str, prompt_type: str):
        """프롬프트를 미리보기용으로 저장"""
        import threading
        current_thread = threading.current_thread()
        
        if not hasattr(current_thread, 'stored_prompts'):
            current_thread.stored_prompts = {}
        
        # 프롬프트 타입별 제목과 설명 매핑
        prompt_info = {
            "main_extraction": {
                'title': '메인 정보 추출 프롬프트',
                'description': '광고 메시지에서 제목, 목적, 상품, 채널, 프로그램 정보를 추출하는 프롬프트'
            },
            "entity_extraction": {
                'title': '엔티티 추출 프롬프트', 
                'description': '메시지에서 상품/서비스 엔티티를 추출하는 프롬프트'
            }
        }
        
        info = prompt_info.get(prompt_type, {
            'title': f'{prompt_type} 프롬프트',
            'description': f'{prompt_type} 처리를 위한 프롬프트'
        })
        
        prompt_key = f'{prompt_type}_prompt'
        prompt_data = {
            'title': info['title'],
            'description': info['description'],
            'content': prompt,
            'length': len(prompt)
        }
        
        current_thread.stored_prompts[prompt_key] = prompt_data
        
        # 디버깅 로그 추가
        prompt_length = len(prompt)
        logger.info(f"📝 프롬프트 저장됨: {prompt_key}")
        logger.info(f"📝 프롬프트 길이: {prompt_length:,} 문자")
        
        # 프롬프트가 매우 긴 경우 경고
        if prompt_length > 20000:
            logger.warning(f"⚠️ 매우 긴 프롬프트가 저장됨: {prompt_length:,} 문자")
            logger.warning("이는 UI 표시 성능에 영향을 줄 수 있습니다.")
            
            # 프롬프트 내용 분석 (엔티티 추출 프롬프트인 경우)
            if 'entity' in prompt_key.lower():
                entity_section_start = prompt.find("## Candidate entities:")
                if entity_section_start > 0:
                    entity_section = prompt[entity_section_start:]
                    entity_lines = entity_section.split('\n')
                    entity_count = len([line for line in entity_lines if line.strip().startswith('-')])
                    logger.warning(f"🔍 후보 엔티티 개수: {entity_count}개")
        
        logger.info(f"📝 현재 저장된 프롬프트 수: {len(current_thread.stored_prompts)}")
        logger.info(f"📝 저장된 프롬프트 키들: {list(current_thread.stored_prompts.keys())}")

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
                    if self._detect_schema_response(json_objects):
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

    @log_performance
    def extract_entities_from_kiwi(self, mms_msg: str) -> Tuple[List[str], pd.DataFrame]:
        """Kiwi 형태소 분석기를 사용한 엔티티 추출"""
        try:
            logger.info("=== Kiwi 기반 엔티티 추출 시작 ===")
            mms_msg = validate_text_input(mms_msg)
            logger.info(f"처리할 메시지 길이: {len(mms_msg)} 문자")
            
            # 상품 데이터 상태 확인
            if self.item_pdf_all.empty:
                logger.error("상품 데이터가 비어있습니다! 엔티티 추출 불가")
                return [], pd.DataFrame()
            
            if 'item_nm_alias' not in self.item_pdf_all.columns:
                logger.error("item_nm_alias 컬럼이 없습니다! 엔티티 추출 불가")
                return [], pd.DataFrame()
            
            unique_aliases = self.item_pdf_all['item_nm_alias'].unique()
            logger.info(f"매칭할 상품 별칭 수: {len(unique_aliases)}개")
            
            # 문장 분할 및 하위 문장 처리
            sentences = sum(self.kiwi.split_into_sents(
                re.split(r"_+", mms_msg), return_tokens=True, return_sub_sents=True
            ), [])
            
            sentences_all = []
            for sent in sentences:
                if sent.subs:
                    sentences_all.extend(sent.subs)
                else:
                    sentences_all.append(sent)
            
            logger.info(f"분할된 문장 수: {len(sentences_all)}개")
            
            # 제외 패턴을 적용하여 문장 필터링
            sentence_list = [
                filter_text_by_exc_patterns(sent, self.exc_tag_patterns) 
                for sent in sentences_all
            ]
            
            logger.info(f"필터링된 문장들: {sentence_list[:3]}...")  # 처음 3개만 로깅

            # 형태소 분석을 통한 고유명사 추출
            result_msg = self.kiwi.tokenize(mms_msg, normalize_coda=True, z_coda=False, split_complex=False)
            all_tokens = [(token.form, token.tag) for token in result_msg]
            logger.info(f"전체 토큰 수: {len(all_tokens)}개")
            
            # NNP 태그 토큰들만 추출
            nnp_tokens = [token.form for token in result_msg if token.tag == 'NNP']
            logger.info(f"NNP 태그 토큰들: {nnp_tokens}")
            
            entities_from_kiwi = [
                token.form for token in result_msg 
                if token.tag == 'NNP' and 
                   token.form not in self.stop_item_names + ['-'] and 
                   len(token.form) >= 2 and 
                   not token.form.lower() in self.stop_item_names
            ]
            entities_from_kiwi = [e for e in filter_specific_terms(entities_from_kiwi) if e in unique_aliases]
            
            logger.info(f"필터링 후 Kiwi 추출 엔티티: {list(set(entities_from_kiwi))}")

            # 퍼지 매칭을 통한 유사 상품명 찾기
            logger.info("퍼지 매칭 시작...")
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
            
            logger.info(f"퍼지 매칭 결과 크기: {similarities_fuzzy.shape if not similarities_fuzzy.empty else '비어있음'}")
            
            if similarities_fuzzy.empty:
                logger.warning("퍼지 매칭 결과가 비어있습니다. Kiwi 결과만 사용합니다.")
                # 퍼지 매칭 결과가 없으면 Kiwi 결과만 사용
                cand_item_list = list(entities_from_kiwi) if entities_from_kiwi else []
                logger.info(f"Kiwi 기반 후보 아이템: {cand_item_list}")
                
                if cand_item_list:
                    extra_item_pdf = self.item_pdf_all.query("item_nm_alias in @cand_item_list")[
                        ['item_nm','item_nm_alias','item_id']
                    ].groupby(["item_nm"])['item_id'].apply(list).reset_index()
                    logger.info(f"매칭된 상품 정보: {extra_item_pdf.shape}")
                else:
                    extra_item_pdf = pd.DataFrame()
                    logger.warning("후보 아이템이 없습니다!")
                
                return cand_item_list, extra_item_pdf
            else:
                logger.info(f"퍼지 매칭 성공: {len(similarities_fuzzy)}개 결과")
                if not similarities_fuzzy.empty:
                    sample_fuzzy = similarities_fuzzy.head(3)[['sent', 'item_nm_alias', 'sim']].to_dict('records')
                    logger.info(f"퍼지 매칭 샘플: {sample_fuzzy}")

            # 시퀀스 유사도를 통한 정밀 매칭
            logger.info("시퀀스 유사도 계산 시작...")
            similarities_seq = safe_execute(
                parallel_seq_similarity,
                sent_item_pdf=similarities_fuzzy,
                text_col_nm='sent',
                item_col_nm='item_nm_alias',
                n_jobs=getattr(PROCESSING_CONFIG, 'n_jobs', 4),
                batch_size=getattr(PROCESSING_CONFIG, 'batch_size', 100),
                default_return=pd.DataFrame()
            )
            
            logger.info(f"시퀀스 유사도 결과 크기: {similarities_seq.shape if not similarities_seq.empty else '비어있음'}")
            if not similarities_seq.empty:
                sample_seq = similarities_seq.head(3)[['sent', 'item_nm_alias', 'sim']].to_dict('records')
                logger.info(f"시퀀스 유사도 샘플: {sample_seq}")
            
            # 임계값 이상의 후보 아이템들 필터링
            similarity_threshold = getattr(PROCESSING_CONFIG, 'similarity_threshold', 0.2)
            logger.info(f"사용할 유사도 임계값: {similarity_threshold}")
            
            cand_items = similarities_seq.query(
                "sim >= @similarity_threshold and "
                "item_nm_alias.str.contains('', case=False) and "
                "item_nm_alias not in @self.stop_item_names"
            )
            logger.info(f"임계값 필터링 후 후보 아이템 수: {len(cand_items)}개")
            
            # Kiwi에서 추출한 엔티티들 추가
            entities_from_kiwi_pdf = self.item_pdf_all.query("item_nm_alias in @entities_from_kiwi")[
                ['item_nm','item_nm_alias']
            ]
            entities_from_kiwi_pdf['sim'] = 1.0
            logger.info(f"Kiwi 엔티티 매칭 결과: {len(entities_from_kiwi_pdf)}개")

            # 결과 통합 및 최종 후보 리스트 생성
            cand_item_pdf = pd.concat([cand_items, entities_from_kiwi_pdf])
            logger.info(f"통합된 후보 아이템 수: {len(cand_item_pdf)}개")
            
            if not cand_item_pdf.empty:
                cand_item_array = cand_item_pdf.sort_values('sim', ascending=False).groupby([
                    "item_nm_alias"
                ])['sim'].max().reset_index(name='final_sim').sort_values(
                    'final_sim', ascending=False
                ).query("final_sim >= 0.2")['item_nm_alias'].unique()
                
                # numpy 배열을 리스트로 변환하여 안전성 보장
                cand_item_list = list(cand_item_array) if hasattr(cand_item_array, '__iter__') else []
                
                logger.info(f"최종 후보 아이템 리스트: {cand_item_list}")
                
                if cand_item_list:  # 리스트가 비어있지 않은 경우에만 쿼리 실행
                    extra_item_pdf = self.item_pdf_all.query("item_nm_alias in @cand_item_list")[
                        ['item_nm','item_nm_alias','item_id']
                    ].groupby(["item_nm"])['item_id'].apply(list).reset_index()
                else:
                    extra_item_pdf = pd.DataFrame()
                
                logger.info(f"최종 상품 정보 DataFrame 크기: {extra_item_pdf.shape}")
                if not extra_item_pdf.empty:
                    sample_final = extra_item_pdf.head(3).to_dict('records')
                    logger.info(f"최종 상품 정보 샘플: {sample_final}")
            else:
                logger.warning("통합된 후보 아이템이 없습니다!")
                cand_item_list = []
                extra_item_pdf = pd.DataFrame()

            return entities_from_kiwi, cand_item_list, extra_item_pdf
            
        except Exception as e:
            logger.error(f"Kiwi 엔티티 추출 실패: {e}")
            logger.error(f"오류 상세: {traceback.format_exc()}")
            # 안전한 기본값 반환 - 빈 리스트와 빈 DataFrame
            return [], [], pd.DataFrame()

    def extract_entities_by_logic(self, cand_entities: List[str], threshold_for_fuzzy: float = 0.5) -> pd.DataFrame:
        """로직 기반 엔티티 추출"""
        try:
            if not cand_entities:
                return pd.DataFrame()
            
            # 퍼지 유사도 계산
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
            
            # 시퀀스 유사도 계산
            cand_entities_sim = self._calculate_combined_similarity(similarities_fuzzy)
            
            return cand_entities_sim
            
        except Exception as e:
            logger.error(f"로직 기반 엔티티 추출 실패: {e}")
            return pd.DataFrame()

    def _calculate_combined_similarity(self, similarities_fuzzy: pd.DataFrame, weights: dict = None) -> pd.DataFrame:
        """s1, s2 정규화 방식으로 각각 계산 후 합산"""
        try:
            # s1 정규화
            sim_s1 = safe_execute(
                parallel_seq_similarity,
                sent_item_pdf=similarities_fuzzy,
                text_col_nm='item_name_in_msg',
                item_col_nm='item_nm_alias',
                n_jobs=getattr(PROCESSING_CONFIG, 'n_jobs', 4),
                batch_size=30,
                normalizaton_value='s1',
                # weights=weights,
                default_return=pd.DataFrame()
            ).rename(columns={'sim': 'sim_s1'})
            
            # s2 정규화
            sim_s2 = safe_execute(
                parallel_seq_similarity,
                sent_item_pdf=similarities_fuzzy,
                text_col_nm='item_name_in_msg',
                item_col_nm='item_nm_alias',
                n_jobs=getattr(PROCESSING_CONFIG, 'n_jobs', 4),
                batch_size=30,
                normalizaton_value='s2',
                # weights=weights,
                default_return=pd.DataFrame()
            ).rename(columns={'sim': 'sim_s2'})
            
            # 결과 합치기
            if not sim_s1.empty and not sim_s2.empty:
                try:
                    # ipynb와 동일하게 merge 후 쿼리 조건 적용
                    combined = sim_s1.merge(sim_s2, on=['item_name_in_msg', 'item_nm_alias'])
                    # ipynb와 동일한 필터링 조건: (sim_s1>=0.4 and sim_s2>=0.4) or (sim_s1>=1.9 and sim_s2>=0.3) or (sim_s1>=0.3 and sim_s2>=0.9)
                    filtered = combined.query("(sim_s1>=0.4 and sim_s2>=0.4) or (sim_s1>=1.9 and sim_s2>=0.3) or (sim_s1>=0.3 and sim_s2>=0.9)")
                    # sim_s1과 sim_s2를 각각 합산한 후 더하기 (ipynb와 동일)
                    combined = filtered.groupby(['item_name_in_msg', 'item_nm_alias']).agg({
                        'sim_s1': 'sum',
                        'sim_s2': 'sum'
                    }).reset_index()
                    combined['sim'] = combined['sim_s1'] + combined['sim_s2']
                except Exception as e:
                    logger.error(f"결합 유사도 계산 실패: {e}")
                    return pd.DataFrame()
                return combined
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"결합 유사도 계산 실패: {e}")
            return pd.DataFrame()

    @log_performance
    def extract_entities_by_llm(self, msg_text: str, rank_limit: int = 200, llm_models: List = None, external_cand_entities: List[str] = []) -> pd.DataFrame:
        """
        LLM 기반 엔티티 추출 (복수 모델 병렬 처리 지원)
        
        Args:
            msg_text (str): 분석할 메시지 텍스트
            rank_limit (int): 결과에서 반환할 최대 순위
            llm_models (List, optional): 사용할 LLM 모델 리스트. None이면 기본 모델 사용
            
        Returns:
            pd.DataFrame: 추출된 엔티티와 유사도 정보
        """
        try:
            logger.info("=" * 80)
            logger.info("🔍 [LLM 엔티티 추출] 함수 시작")
            logger.info(f"📝 입력 파라미터:")
            logger.info(f"   - rank_limit: {rank_limit}")
            logger.info(f"   - external_cand_entities 제공 여부: {external_cand_entities is not None}")
            if external_cand_entities is not None:
                logger.info(f"   - external_cand_entities 개수: {len(external_cand_entities)}")
            
            msg_text = validate_text_input(msg_text)
            logger.info(f"📄 메시지 텍스트 길이: {len(msg_text):,} 문자")
            logger.info(f"📄 메시지 텍스트 미리보기: {msg_text[:100]}..." if len(msg_text) > 100 else f"📄 메시지 텍스트: {msg_text}")
            
            # LLM 모델이 지정되지 않은 경우 기본 모델 사용
            if llm_models is None:
                llm_models = [self.llm_model]
                logger.info(f"🤖 LLM 모델 자동 선택: 기본 모델 사용 (1개)")
            else:
                logger.info(f"🤖 LLM 모델 지정됨: {len(llm_models)}개 모델 사용")
            
            for idx, model in enumerate(llm_models):
                model_name = getattr(model, 'model_name', 'Unknown')
                logger.info(f"   [{idx+1}] 모델: {model_name}")
            
            def get_entities_by_llm(args_dict):
                """단일 LLM으로 엔티티 추출하는 내부 함수"""
                llm_model, msg_text = args_dict['llm_model'], args_dict['msg_text']
                model_name = getattr(llm_model, 'model_name', 'Unknown')
                
                try:
                    logger.info(f"   ⚙️  [{model_name}] 엔티티 추출 시작")
                    
                    # 프롬프트 구성 - 기존 로직과 동일
                    base_prompt = getattr(PROCESSING_CONFIG, 'entity_extraction_prompt', None)
                    if base_prompt is None:
                        base_prompt = DETAILED_ENTITY_EXTRACTION_PROMPT
                        logger.info(f"   📋 [{model_name}] 엔티티 추출에 prompts 디렉토리의 DETAILED_ENTITY_EXTRACTION_PROMPT 사용")
                    else:
                        logger.info(f"   📋 [{model_name}] 엔티티 추출에 settings.py의 entity_extraction_prompt 사용")
                    
                    # 베이스 프롬프트 길이 확인
                    base_prompt_length = len(base_prompt)
                    msg_length = len(msg_text)
                    logger.info(f"   📏 [{model_name}] 베이스 프롬프트 길이: {base_prompt_length:,} 문자")
                    logger.info(f"   📏 [{model_name}] 메시지 길이: {msg_length:,} 문자")
                    
                    # 프롬프트 내용 로깅 (전체)
                    logger.info(f"   📝 [{model_name}] 베이스 프롬프트 내용 (전체):")
                    logger.info(f"   {'-' * 75}")
                    for line in base_prompt.split('\n'):
                        logger.info(f"   {line}")
                    logger.info(f"   {'-' * 75}")
                    
                    # PromptTemplate 사용 (langchain 방식)
                    zero_shot_prompt = PromptTemplate(
                        input_variables=["entity_extraction_prompt", "msg", "cand_entities"],
                        template="""
                        {entity_extraction_prompt}
                        
                        ## message:                
                        {msg}
                        """
                    )
                    
                    # 최종 프롬프트 생성 (실제로 LLM에 전달되는 프롬프트)
                    final_prompt_for_llm = zero_shot_prompt.format(
                        entity_extraction_prompt=base_prompt,
                        msg=msg_text
                    )
                    final_prompt_length = len(final_prompt_for_llm)
                    logger.info(f"   📏 [{model_name}] 최종 프롬프트 길이: {final_prompt_length:,} 문자")
                    logger.info(f"   📝 [{model_name}] 최종 프롬프트 내용 (전체):")
                    logger.info(f"   {'-' * 75}")
                    for line in final_prompt_for_llm.split('\n'):
                        logger.info(f"   {line}")
                    logger.info(f"   {'-' * 75}")

                    logger.info(f"   🚀 [{model_name}] LLM 호출 시작...")
                    chain = zero_shot_prompt | llm_model
                    cand_entities = chain.invoke({
                        "entity_extraction_prompt": base_prompt, 
                        "msg": msg_text, 
                    }).content
                    logger.info(f"   ✅ [{model_name}] LLM 호출 완료")
                    logger.info(f"   📥 [{model_name}] LLM 응답 길이: {len(cand_entities):,} 문자")
                    logger.info(f"   📥 [{model_name}] LLM 응답 미리보기: {cand_entities[:200]}..." if len(cand_entities) > 200 else f"   📥 [{model_name}] LLM 응답: {cand_entities}")

                    # LLM 응답 파싱 및 정리
                    logger.info(f"   🔧 [{model_name}] 엔티티 파싱 시작...")
                    cand_entity_list_raw = [e.strip() for e in cand_entities.split(',') if e.strip()]
                    logger.info(f"   📊 [{model_name}] 콤마로 분할 후 엔티티 수: {len(cand_entity_list_raw)}개")
                    
                    before_filter = len(cand_entity_list_raw)
                    cand_entity_list = [e for e in cand_entity_list_raw if e not in self.stop_item_names and len(e) >= 2]
                    after_filter = len(cand_entity_list)
                    filtered_count = before_filter - after_filter
                    
                    logger.info(f"   🎯 [{model_name}] 필터링 결과:")
                    logger.info(f"      - 필터링 전: {before_filter}개")
                    logger.info(f"      - 필터링 후: {after_filter}개 (제거: {filtered_count}개)")
                    logger.info(f"      - 최종 엔티티: {cand_entity_list[:10]}..." if len(cand_entity_list) > 10 else f"      - 최종 엔티티: {cand_entity_list}")

                    return cand_entity_list
                    
                except Exception as e:
                    logger.error(f"   ❌ [{model_name}] LLM 모델에서 엔티티 추출 실패: {e}")
                    logger.error(f"   ❌ [{model_name}] 오류 상세: {traceback.format_exc()}")
                    return []
            
            # 프롬프트 미리보기 저장 (디버깅용) - 복수 모델이어도 프롬프트는 동일하므로 항상 저장
            logger.info("📋 프롬프트 미리보기 저장 중...")
            base_prompt = getattr(PROCESSING_CONFIG, 'entity_extraction_prompt', None)
            if base_prompt is None:
                base_prompt = DETAILED_ENTITY_EXTRACTION_PROMPT
            preview_prompt = build_entity_extraction_prompt(msg_text, base_prompt)
            
            # 최종 프롬프트 길이 확인
            final_prompt_length = len(preview_prompt)
            logger.info(f"📏 최종 엔티티 추출 프롬프트 길이: {final_prompt_length:,} 문자")
            logger.info(f"📝 프롬프트 미리보기 내용 (전체):")
            logger.info("-" * 80)
            for line in preview_prompt.split('\n'):
                logger.info(f"   {line}")
            logger.info("-" * 80)
            
            self._store_prompt_for_preview(preview_prompt, "entity_extraction")
            logger.info("✅ 프롬프트 미리보기 저장 완료")

            
            logger.info("🔄 LLM 직접 추출 모드")
            # 병렬 처리를 위한 배치 구성 (단일/복수 모델 모두 동일하게 처리)
            batches = []
            for llm_model in llm_models:
                batches.append({
                    "msg_text": msg_text, 
                    "llm_model": llm_model, 
                })
            
            logger.info(f"🔄 {len(llm_models)}개 LLM 모델로 엔티티 추출 시작")
            logger.info(f"🔄 병렬 작업 수: {len(batches)}개 배치")
            
            # 병렬 작업 실행
            n_jobs = min(3, len(llm_models))  # 최대 3개 작업으로 제한
            logger.info(f"⚙️  병렬 처리 설정: {n_jobs}개 워커 (threading 백엔드)")
            
            with Parallel(n_jobs=n_jobs, backend='threading') as parallel:
                batch_results = parallel(delayed(get_entities_by_llm)(args) for args in batches)
            
            logger.info(f"✅ 모든 LLM 모델 처리 완료")
            logger.info(f"📊 모델별 결과:")
            for idx, (model, result) in enumerate(zip(llm_models, batch_results)):
                model_name = getattr(model, 'model_name', 'Unknown')
                logger.info(f"   [{idx+1}] {model_name}: {len(result)}개 엔티티 추출")
            
            # 모든 결과를 합치고 중복 제거
            all_entities = sum(batch_results, [])
            if external_cand_entities is not None and len(external_cand_entities)>0:
                all_entities = list(set(all_entities+external_cand_entities))
            logger.info(f"📊 병합 전 총 엔티티 수: {len(all_entities)}개")
            cand_entity_list = list(set(all_entities))
            cand_entity_list = list(set(sum([[c['text'] for c in extract_ngram_candidates(cand_entity, min_n=2, max_n=len(cand_entity.split())) if c['start_idx']<=0] if len(cand_entity.split())>=4 else [cand_entity] for cand_entity in cand_entity_list], [])))
            logger.info(f"📊 중복 제거 후 엔티티 수: {len(cand_entity_list)}개")
            logger.info(f"✅ LLM 추출 완료: {cand_entity_list[:20]}..." if len(cand_entity_list) > 20 else f"✅ LLM 추출 완료: {cand_entity_list}")

            if not cand_entity_list:
                logger.warning("⚠️  LLM 추출에서 유효한 엔티티를 찾지 못함")
                logger.info("=" * 80)
                return pd.DataFrame()
            
            # cand_entity_list = select_most_comprehensive(cand_entity_list)
            logger.info("🔍 엔티티-상품 매칭 시작...")
            logger.info(f"   입력 엔티티 수: {len(cand_entity_list)}개")
            cand_entities_sim = self._match_entities_with_products(cand_entity_list, rank_limit)
            logger.info(f"   매칭 결과: {len(cand_entities_sim)}개 행")
            
            if cand_entities_sim.empty:
                logger.warning("⚠️  엔티티-상품 매칭 결과가 비어있음")
                logger.info("=" * 80)
                return pd.DataFrame()
            
            # [단계 1] 매칭 완료 직후 item_name_in_msg 로깅
            logger.info(f"   [단계 1] 매칭 완료 직후 item_name_in_msg:")
            logger.info(f"      - 고유 개수: {cand_entities_sim['item_name_in_msg'].nunique()}개")
            logger.info(f"      - 전체 개수: {len(cand_entities_sim)}개")
            item_name_list_1 = list(cand_entities_sim['item_name_in_msg'].unique())
            logger.info(f"      - 고유 item_name_in_msg 목록: {item_name_list_1}")
            
            logger.info(f"   매칭된 고유 item_name_in_msg 수: {cand_entities_sim['item_name_in_msg'].nunique()}개")
            logger.info(f"   매칭된 고유 item_nm_alias 수: {cand_entities_sim['item_nm_alias'].nunique()}개")

            # 후보 엔티티들과 상품 DB 매칭
            logger.info("🔍 2단계 LLM 필터링 시작...")
            logger.info(f"   입력 메시지 엔티티 수: {len(cand_entities_sim['item_name_in_msg'].unique())}개")
            logger.info(f"   후보 상품 별칭 수: {len(cand_entities_sim['item_nm_alias'].unique())}개")
            
            # [단계 2] 2단계 LLM 필터링 시작 전 item_name_in_msg 로깅
            logger.info(f"   [단계 2] 2단계 LLM 필터링 시작 전 item_name_in_msg:")
            logger.info(f"      - 고유 개수: {cand_entities_sim['item_name_in_msg'].nunique()}개")
            logger.info(f"      - 전체 개수: {len(cand_entities_sim)}개")
            item_name_list_2 = list(cand_entities_sim['item_name_in_msg'].unique())
            logger.info(f"      - 고유 item_name_in_msg 목록: {item_name_list_2}")
            
            # SIMPLE_ENTITY_EXTRACTION_PROMPT 로깅
            simple_prompt_length = len(SIMPLE_ENTITY_EXTRACTION_PROMPT)
            logger.info(f"   📏 SIMPLE_ENTITY_EXTRACTION_PROMPT 길이: {simple_prompt_length:,} 문자")
            logger.info(f"   📝 SIMPLE_ENTITY_EXTRACTION_PROMPT 내용 (전체):")
            logger.info(f"   {'-' * 75}")
            for line in SIMPLE_ENTITY_EXTRACTION_PROMPT.split('\n'):
                logger.info(f"   {line}")
            logger.info(f"   {'-' * 75}")
            
            zero_shot_prompt = PromptTemplate(
            input_variables=["msg","entities_msg","cand_entities_voca"],
            template="""
            {entity_extraction_prompt}
            
            ## message:                
            {msg}

            ## entities in message:
            {entities_msg}

            ## candidate entities in vocabulary:
            {cand_entities_voca}

            """
            )
            
            # 2단계 최종 프롬프트 생성
            entities_msg_list = list(cand_entities_sim['item_name_in_msg'].unique())
            cand_entities_voca_list = list(cand_entities_sim['item_nm_alias'].unique())
            
            logger.info(f"   📝 입력 엔티티 리스트 (처음 20개): {entities_msg_list[:20]}..." if len(entities_msg_list) > 20 else f"   📝 입력 엔티티 리스트: {entities_msg_list}")
            logger.info(f"   📝 후보 상품 별칭 리스트 (처음 20개): {cand_entities_voca_list[:20]}..." if len(cand_entities_voca_list) > 20 else f"   📝 후보 상품 별칭 리스트: {cand_entities_voca_list}")
            
            final_prompt_2nd = zero_shot_prompt.format(
                entity_extraction_prompt=SIMPLE_ENTITY_EXTRACTION_PROMPT,
                msg=msg_text,
                entities_msg=entities_msg_list,
                cand_entities_voca=cand_entities_voca_list
            )
            final_prompt_2nd_length = len(final_prompt_2nd)
            logger.info(f"   📏 2단계 최종 프롬프트 길이: {final_prompt_2nd_length:,} 문자")
            logger.info(f"   📝 2단계 최종 프롬프트 내용 (전체):")
            logger.info(f"   {'-' * 75}")
            for line in final_prompt_2nd.split('\n'):
                logger.info(f"   {line}")
            logger.info(f"   {'-' * 75}")
                        
            logger.info("🚀 2단계 LLM 호출 시작...")
            chain = zero_shot_prompt | self.llm_model
            cand_entities = chain.invoke({"entity_extraction_prompt": SIMPLE_ENTITY_EXTRACTION_PROMPT, "msg": msg_text, "entities_msg":cand_entities_sim['item_name_in_msg'].unique(), "cand_entities_voca":cand_entities_sim['item_nm_alias'].unique()}).content
            logger.info("✅ 2단계 LLM 호출 완료")
            logger.info(f"📥 2단계 LLM 응답 길이: {len(cand_entities):,} 문자")
            logger.info(f"📥 2단계 LLM 응답: {cand_entities}")

            logger.info("🔧 2단계 엔티티 파싱 시작...")
            cand_entity_list = [e.strip() for e in cand_entities.split("\n")[-1].replace("ENTITY: ","").split(',') if e.strip()]
            logger.info(f"   파싱 직후 엔티티 수: {len(cand_entity_list)}개")
            
            before_filter = len(cand_entity_list)
            cand_entity_list = [e for e in cand_entity_list if e not in self.stop_item_names and len(e)>=2]
            after_filter = len(cand_entity_list)
            
            logger.info(f"   필터링 결과:")
            logger.info(f"      - 필터링 전: {before_filter}개")
            logger.info(f"      - 필터링 후: {after_filter}개 (제거: {before_filter - after_filter}개)")
            logger.info(f"   최종 선택된 엔티티: {cand_entity_list}")

            logger.info(f"🔍 최종 엔티티로 필터링 중...")
            logger.info(f"   필터링 전 행 수: {len(cand_entities_sim)}개")
            
            # [단계 3] 최종 필터링 전 item_name_in_msg 로깅
            logger.info(f"   [단계 3] 최종 필터링 전 item_name_in_msg:")
            logger.info(f"      - 고유 개수: {cand_entities_sim['item_name_in_msg'].nunique()}개")
            logger.info(f"      - 전체 개수: {len(cand_entities_sim)}개")
            item_name_list_3 = list(cand_entities_sim['item_name_in_msg'].unique())
            logger.info(f"      - 고유 item_name_in_msg 목록: {item_name_list_3}")
            
            cand_entities_sim = cand_entities_sim.query("item_nm_alias in @cand_entity_list")
            logger.info(f"   필터링 후 행 수: {len(cand_entities_sim)}개")
            
            # [단계 4] 최종 필터링 후 item_name_in_msg 로깅
            logger.info(f"   [단계 4] 최종 필터링 후 item_name_in_msg:")
            logger.info(f"      - 고유 개수: {cand_entities_sim['item_name_in_msg'].nunique()}개")
            logger.info(f"      - 전체 개수: {len(cand_entities_sim)}개")
            if not cand_entities_sim.empty:
                item_name_list_4 = list(cand_entities_sim['item_name_in_msg'].unique())
                logger.info(f"      - 고유 item_name_in_msg 목록: {item_name_list_4}")
            else:
                logger.info(f"      - 고유 item_name_in_msg 목록: [] (비어있음)")
            
            logger.info("=" * 80)
            logger.info("✅ [LLM 엔티티 추출] 함수 완료")
            logger.info(f"📊 최종 결과: {len(cand_entities_sim)}개 행 반환")
            logger.info("=" * 80)

            return cand_entities_sim
            
        except Exception as e:
            logger.error("=" * 80)
            logger.error("❌ [LLM 엔티티 추출] 함수 실패")
            logger.error(f"오류 메시지: {e}")
            logger.error(f"오류 상세: {traceback.format_exc()}")
            logger.error("=" * 80)
            return pd.DataFrame()

    def _match_entities_with_products(self, cand_entity_list: List[str], rank_limit: int) -> pd.DataFrame:
        """후보 엔티티들을 상품 DB와 매칭 (ipynb 로직과 동일)"""
        try:
            logger.info("   🔍 [매칭] 퍼지 유사도 매칭 시작...")
            logger.info(f"   📝 입력 엔티티 수: {len(cand_entity_list)}개")
            logger.info(f"   📝 상품 DB 별칭 수: {len(self.item_pdf_all['item_nm_alias'].unique()):,}개")
            logger.info(f"   ⚙️  퍼지 유사도 임계값: 0.6")
            logger.info(f"   ⚙️  n_jobs: 6, batch_size: 30")
            
            # 퍼지 유사도 매칭 (ipynb와 동일하게 직접 호출)
            similarities_fuzzy = parallel_fuzzy_similarity(
                cand_entity_list,
                self.item_pdf_all['item_nm_alias'].unique(),
                threshold=0.6,
                text_col_nm='item_name_in_msg',
                item_col_nm='item_nm_alias',
                n_jobs=6,
                batch_size=30
            )
            
            logger.info(f"   ✅ 퍼지 유사도 매칭 완료: {len(similarities_fuzzy)}개 행")
            
            if similarities_fuzzy.empty:
                logger.warning("   ⚠️  퍼지 유사도 매칭 결과가 비어있음")
                return pd.DataFrame()
            
            logger.info(f"   📊 퍼지 매칭 고유 엔티티 수: {similarities_fuzzy['item_name_in_msg'].nunique()}개")
            logger.info(f"   📊 퍼지 매칭 고유 별칭 수: {similarities_fuzzy['item_nm_alias'].nunique()}개")
            
            # 정지어 필터링
            logger.info("   🔍 [매칭] 정지어 필터링...")
            before_stopwords = len(similarities_fuzzy)
            similarities_fuzzy = similarities_fuzzy[
                ~similarities_fuzzy['item_nm_alias'].isin(self.stop_item_names)
            ]
            after_stopwords = len(similarities_fuzzy)
            logger.info(f"   📊 정지어 필터링 결과: {before_stopwords}개 → {after_stopwords}개 (제거: {before_stopwords - after_stopwords}개)")

            # 시퀀스 유사도 매칭 (ipynb와 동일하게 두 번 호출)
            logger.info("   🔍 [매칭] 시퀀스 유사도 계산 시작 (s1, s2 각각)...")
            logger.info(f"   ⚙️  ipynb와 동일하게 weights=None, n_jobs=6, batch_size=30 사용")
            
            # s1 정규화
            sim_s1 = parallel_seq_similarity(
                sent_item_pdf=similarities_fuzzy,
                text_col_nm='item_name_in_msg',
                item_col_nm='item_nm_alias',
                n_jobs=6,
                batch_size=30,
                # weights=None,  # ipynb와 동일하게 weights 없음
                normalizaton_value='s1'
            ).rename(columns={'sim': 'sim_s1'})
            
            # s2 정규화
            sim_s2 = parallel_seq_similarity(
                sent_item_pdf=similarities_fuzzy,
                text_col_nm='item_name_in_msg',
                item_col_nm='item_nm_alias',
                n_jobs=6,
                batch_size=30,
                # weights=None,  # ipynb와 동일하게 weights 없음
                normalizaton_value='s2'
            ).rename(columns={'sim': 'sim_s2'})
            
            logger.info(f"   ✅ 시퀀스 유사도 계산 완료: sim_s1={len(sim_s1)}개, sim_s2={len(sim_s2)}개")
            
            # merge로 합치기 (ipynb와 동일)
            logger.info("   🔍 [매칭] sim_s1과 sim_s2 병합 중...")
            cand_entities_sim = sim_s1.merge(sim_s2, on=['item_name_in_msg', 'item_nm_alias'])
            logger.info(f"   ✅ 병합 완료: {len(cand_entities_sim)}개 행")
            
            if cand_entities_sim.empty:
                logger.warning("   ⚠️  시퀀스 유사도 계산 결과가 비어있음")
                return pd.DataFrame()
            
            logger.info(f"   📊 유사도 통계:")
            logger.info(f"      - sim_s1 최소: {cand_entities_sim['sim_s1'].min():.4f}")
            logger.info(f"      - sim_s1 최대: {cand_entities_sim['sim_s1'].max():.4f}")
            logger.info(f"      - sim_s2 최소: {cand_entities_sim['sim_s2'].min():.4f}")
            logger.info(f"      - sim_s2 최대: {cand_entities_sim['sim_s2'].max():.4f}")
            
            # ipynb와 동일한 필터링 조건 적용
            logger.info(f"   🔍 [매칭] 쿼리 조건 필터링...")
            logger.info(f"   ⚙️  조건: (sim_s1>=0.4 and sim_s2>=0.4) or (sim_s1>=1.9 and sim_s2>=0.3) or (sim_s1>=0.3 and sim_s2>=0.9)")
            before_query = len(cand_entities_sim)
            cand_entities_sim = cand_entities_sim.query("(sim_s1>=0.4 and sim_s2>=0.4) or (sim_s1>=1.9 and sim_s2>=0.3) or (sim_s1>=0.3 and sim_s2>=0.9)")
            after_query = len(cand_entities_sim)
            logger.info(f"   📊 쿼리 필터링 결과: {before_query}개 → {after_query}개 (제거: {before_query - after_query}개)")

            # ipynb와 동일하게 groupby로 합산
            logger.info(f"   🔍 [매칭] sim_s1과 sim_s2 합산 중...")
            cand_entities_sim = cand_entities_sim.groupby(['item_name_in_msg', 'item_nm_alias'])[['sim_s1', 'sim_s2']].apply(
                lambda x: x['sim_s1'].sum() + x['sim_s2'].sum()
            ).to_frame('sim').reset_index()
            logger.info(f"   ✅ 합산 완료: {len(cand_entities_sim)}개 행")
            
            # ipynb와 동일하게 sim>=1.1 필터링
            logger.info(f"   🔍 [매칭] 유사도 필터링 (임계값: sim>=1.0)...")
            before_sim_filter = len(cand_entities_sim)
            cand_entities_sim = cand_entities_sim.query("sim >= 1.0").copy()
            after_sim_filter = len(cand_entities_sim)
            logger.info(f"   📊 유사도 필터링 결과: {before_sim_filter}개 → {after_sim_filter}개 (제거: {before_sim_filter - after_sim_filter}개)")
            
            logger.info(f"   📊 합산 sim 통계:")
            logger.info(f"      - 최소: {cand_entities_sim['sim'].min():.4f}")
            logger.info(f"      - 최대: {cand_entities_sim['sim'].max():.4f}")
            logger.info(f"      - 평균: {cand_entities_sim['sim'].mean():.4f}")
            logger.info(f"      - 중앙값: {cand_entities_sim['sim'].median():.4f}")

            # 순위 매기기 및 결과 제한
            logger.info(f"   🔍 [매칭] 순위 매기기 및 결과 제한 (rank_limit: {rank_limit})...")
            cand_entities_sim["rank"] = cand_entities_sim.groupby('item_name_in_msg')['sim'].rank(
                method='dense', ascending=False
            )
            before_rank_limit = len(cand_entities_sim)
            cand_entities_sim = cand_entities_sim.query(f"rank <= {rank_limit}").sort_values(
                ['item_name_in_msg', 'rank'], ascending=[True, True]
            )
            after_rank_limit = len(cand_entities_sim)
            logger.info(f"   📊 순위 제한 결과: {before_rank_limit}개 → {after_rank_limit}개 (제거: {before_rank_limit - after_rank_limit}개)")
            
            # ipynb와 동일하게 rank 제한 후 item_dmn_nm 병합
            logger.info(f"   🔍 [매칭] item_dmn_nm 병합 중...")
            if 'item_dmn_nm' in self.item_pdf_all.columns:
                cand_entities_sim = cand_entities_sim.merge(
                    self.item_pdf_all[['item_nm_alias', 'item_dmn_nm']].drop_duplicates(),
                    on='item_nm_alias',
                    how='left'
                )
                logger.info(f"   ✅ item_dmn_nm 병합 완료")
            else:
                logger.warning(f"   ⚠️  item_dmn_nm 컬럼이 없어 병합을 건너뜁니다.")
                logger.warning(f"   ⚠️  item_pdf_all 컬럼 목록: {list(self.item_pdf_all.columns)}")
            
            logger.info(f"   ✅ [매칭] 최종 결과: {len(cand_entities_sim)}개 행, {cand_entities_sim['item_name_in_msg'].nunique()}개 고유 엔티티")

            return cand_entities_sim
            
        except Exception as e:
            logger.error(f"   ❌ [매칭] 엔티티-상품 매칭 실패: {e}")
            logger.error(f"   ❌ [매칭] 오류 상세: {traceback.format_exc()}")
            return pd.DataFrame()

    def _extract_entities(self, mms_msg: str) -> Tuple[List[str], List[str], pd.DataFrame]:
        """엔티티 추출 (Kiwi 또는 LLM 방식)"""
        try:
            if self.entity_extraction_mode == 'logic':
                # Kiwi 기반 추출
                return self.extract_entities_from_kiwi(mms_msg)
            else:
                # LLM 기반 추출을 위해 먼저 Kiwi로 기본 추출
                entities_from_kiwi, cand_item_list, extra_item_pdf = self.extract_entities_from_kiwi(mms_msg)
                return entities_from_kiwi, cand_item_list, extra_item_pdf
                
        except Exception as e:
            logger.error(f"엔티티 추출 실패: {e}")
            logger.error(f"오류 상세: {traceback.format_exc()}")
            # 안전한 기본값 반환
            return [], [], pd.DataFrame()

    def _classify_programs(self, mms_msg: str) -> Dict[str, Any]:
        """프로그램 분류"""
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

    def _build_extraction_prompt(self, msg: str, rag_context: str, product_element: Optional[List[Dict]]) -> str:
        """추출용 프롬프트 구성 - 외부 프롬프트 모듈 사용"""
        
        # 외부 프롬프트 모듈의 함수 사용
        prompt = build_extraction_prompt(
            message=msg,
            rag_context=rag_context,
            product_element=product_element,
            product_info_extraction_mode=self.product_info_extraction_mode
        )
        
        # 디버깅을 위한 프롬프트 로깅 (LLM 모드에서만)
        if self.product_info_extraction_mode == 'llm':
            logger.debug(f"LLM 모드 프롬프트 길이: {len(prompt)} 문자")
            logger.debug(f"후보 상품 목록 포함 여부: {'참고용 후보 상품 이름 목록' in rag_context}")
            
        return prompt

    def _extract_channels(self, json_objects: Dict, msg: str) -> List[Dict]:
        """채널 정보 추출 및 매칭"""
        try:
            channel_tag = []
            channel_items = json_objects.get('channel', [])
            if isinstance(channel_items, dict):
                channel_items = channel_items.get('items', [])

            for d in channel_items:
                if d.get('type') == '대리점' and d.get('value'):
                    # 대리점명으로 조직 정보 검색
                    store_info = self._match_store_info(d['value'])
                    d['store_info'] = store_info
                else:
                    d['store_info'] = []
                channel_tag.append(d)

            return channel_tag
            
        except Exception as e:
            logger.error(f"채널 정보 추출 실패: {e}")
            return []

    def _match_store_info(self, store_name: str) -> List[Dict]:
        """대리점 정보 매칭"""
        try:
            # 대리점명으로 조직 정보 검색
            org_pdf_cand = safe_execute(
                parallel_fuzzy_similarity,
                [preprocess_text(store_name.lower())],
                self.org_pdf['item_nm'].unique(),
                threshold=0.5,
                text_col_nm='org_nm_in_msg',
                item_col_nm='item_nm',
                n_jobs=getattr(PROCESSING_CONFIG, 'n_jobs', 4),
                batch_size=getattr(PROCESSING_CONFIG, 'batch_size', 100),
                default_return=pd.DataFrame()
            )

            if org_pdf_cand.empty:
                return []

            org_pdf_cand = org_pdf_cand.drop('org_nm_in_msg', axis=1)
            org_pdf_cand = self.org_pdf.merge(org_pdf_cand, on=['item_nm'])
            org_pdf_cand['sim'] = org_pdf_cand.apply(
                lambda x: combined_sequence_similarity(store_name, x['item_nm'])[0], axis=1
            ).round(5)
            
            # 대리점 코드('D'로 시작) 우선 검색
            similarity_threshold = getattr(PROCESSING_CONFIG, 'similarity_threshold_for_store', 0.2)
            org_pdf_tmp = org_pdf_cand.query(
                "sim >= @similarity_threshold", engine='python'
            ).sort_values('sim', ascending=False)
            
            if org_pdf_tmp.empty:
                # 대리점이 없으면 전체에서 검색
                org_pdf_tmp = org_pdf_cand.query("sim >= @similarity_threshold").sort_values('sim', ascending=False)
            
            if not org_pdf_tmp.empty:
                # 최고 순위 조직들의 정보 추출
                org_pdf_tmp['rank'] = org_pdf_tmp['sim'].rank(method='dense', ascending=False)
                org_pdf_tmp = org_pdf_tmp.rename(columns={'item_id':'org_cd','item_nm':'org_nm'})
                org_info = org_pdf_tmp.query("rank == 1").groupby('org_nm')['org_cd'].apply(list).reset_index(name='org_cd').to_dict('records')
                return org_info
            else:
                return []
                
        except Exception as e:
            logger.error(f"대리점 정보 매칭 실패: {e}")
            return []

    def _validate_extraction_result(self, result: Dict) -> Dict:
        """추출 결과 검증 및 정리"""
        try:
            # 필수 필드 확인
            required_fields = ['title', 'purpose', 'product', 'channel']
            for field in required_fields:
                if field not in result:
                    logger.warning(f"필수 필드 누락: {field}")
                    result[field] = [] if field != 'title' else "광고 메시지"

            # 상품명 길이 검증
            validated_products = []
            for product in result.get('product', []):
                if isinstance(product, dict):
                    item_name = product.get('item_name_in_msg', product.get('name', ''))
                    if len(item_name) >= 2 and item_name not in self.stop_item_names:
                        validated_products.append(product)
                    else:
                        logger.warning(f"의심스러운 상품명 제외: {item_name}")
            
            result['product'] = validated_products

            # 채널 정보 검증
            validated_channels = []
            for channel in result.get('channel', []):
                if isinstance(channel, dict) and channel.get('value'):
                    validated_channels.append(channel)
            
            result['channel'] = validated_channels

            return result
            
        except Exception as e:
            logger.error(f"결과 검증 실패: {e}")
            return result

    @log_performance
    def process_message(self, mms_msg: str) -> Dict[str, Any]:
        """
        MMS 메시지 전체 처리 (메인 처리 함수)
        
        Args:
            mms_msg: 처리할 MMS 메시지 텍스트
        
        Returns:
            dict: 추출된 정보가 담긴 JSON 구조
        """
        try:
            logger.info("=" * 60)
            logger.info("🚀 MMS 메시지 처리 시작")
            logger.info("=" * 60)
            logger.info(f"메시지 내용: {mms_msg[:200]}...")
            logger.info(f"메시지 길이: {len(mms_msg)} 문자")
            
            # 현재 설정 상태 로깅
            logger.info("=== 현재 추출기 설정 ===")
            logger.info(f"데이터 소스: {self.offer_info_data_src}")
            logger.info(f"상품 정보 추출 모드: {self.product_info_extraction_mode}")
            logger.info(f"엔티티 추출 모드: {self.entity_extraction_mode}")
            logger.info(f"LLM 모델: {self.llm_model_name}")
            logger.info(f"상품 데이터 크기: {self.item_pdf_all.shape}")
            logger.info(f"프로그램 데이터 크기: {self.pgm_pdf.shape}")
            
            # 입력 검증
            msg = validate_text_input(mms_msg)
            
            # 1단계: 엔티티 추출
            logger.info("=" * 30 + " 1단계: 엔티티 추출 " + "=" * 30)
            
            # DB 모드 특별 진단
            if self.offer_info_data_src == "db":
                logger.info("🔍 DB 모드 특별 진단 시작")
                logger.info(f"상품 데이터 상태: {self.item_pdf_all.shape}")
                
                # 필수 컬럼 존재 여부 확인
                required_columns = ['item_nm', 'item_id', 'item_nm_alias']
                missing_columns = [col for col in required_columns if col not in self.item_pdf_all.columns]
                if missing_columns:
                    logger.error(f"🚨 DB 모드에서 필수 컬럼 누락: {missing_columns}")
                
                # 데이터 품질 확인
                if 'item_nm_alias' in self.item_pdf_all.columns:
                    null_aliases = self.item_pdf_all['item_nm_alias'].isnull().sum()
                    total_aliases = len(self.item_pdf_all)
                    logger.info(f"DB 모드 별칭 데이터 품질: {total_aliases - null_aliases}/{total_aliases} 유효")
            
            entities_from_kiwi, cand_item_list, extra_item_pdf = self._extract_entities(msg)
            logger.info(f"추출된 Kiwi 엔티티: {entities_from_kiwi}")
            logger.info(f"추출된 후보 엔티티: {cand_item_list}")
            logger.info(f"매칭된 상품 정보: {extra_item_pdf.shape}")
            
            # DB 모드에서 엔티티 추출 결과 특별 분석
            if self.offer_info_data_src == "db":
                logger.info("🔍 DB 모드 엔티티 추출 결과 분석")
                # cand_item_list가 numpy 배열일 수 있으므로 안전한 검사 사용
                if safe_check_empty(cand_item_list):
                    logger.error("🚨 DB 모드에서 후보 엔티티가 전혀 추출되지 않았습니다!")
                    logger.error("가능한 원인:")
                    logger.error("1. 상품 데이터베이스에 해당 상품이 없음")
                    logger.error("2. 별칭 규칙 적용 실패")
                    logger.error("3. 유사도 임계값이 너무 높음")
                    logger.error("4. Kiwi 형태소 분석 실패")
            
            # 2단계: 프로그램 분류
            logger.info("=" * 30 + " 2단계: 프로그램 분류 " + "=" * 30)
            pgm_info = self._classify_programs(msg)
            logger.info(f"프로그램 분류 결과 키: {list(pgm_info.keys())}")
            
            # 3단계: RAG 컨텍스트 구성
            logger.info("=" * 30 + " 3단계: RAG 컨텍스트 구성 " + "=" * 30)
            rag_context = f"\n### 광고 분류 기준 정보 ###\n\t{pgm_info['pgm_cand_info']}" if self.num_cand_pgms > 0 else ""
            logger.info(f"프로그램 분류 컨텍스트 길이: {len(rag_context)} 문자")
            
            # 4단계: 제품 정보 준비 (모드별 처리)
            logger.info("=" * 30 + " 4단계: 제품 정보 준비 " + "=" * 30)
            product_element = None
            
            # cand_item_list가 비어있지 않은지 안전하게 검사
            if not safe_check_empty(cand_item_list):
                logger.info(f"후보 아이템 리스트 크기: {len(cand_item_list)}개")
                logger.info(f"후보 아이템 리스트: {cand_item_list}")
                
                # extra_item_pdf 상태 확인
                logger.info(f"extra_item_pdf 크기: {extra_item_pdf.shape}")
                if not extra_item_pdf.empty:
                    logger.info(f"extra_item_pdf 컬럼들: {list(extra_item_pdf.columns)}")
                    logger.info(f"extra_item_pdf 샘플: {extra_item_pdf.head(2).to_dict('records')}")
                
                if self.product_info_extraction_mode == 'rag':
                    rag_context += f"\n\n### 후보 상품 이름 목록 ###\n\t{cand_item_list}"
                    logger.info("RAG 모드: 후보 상품 목록을 RAG 컨텍스트에 추가")
                elif self.product_info_extraction_mode == 'llm':
                    # LLM 모드에도 후보 목록 제공하여 일관성 향상
                    rag_context += f"\n\n### 참고용 후보 상품 이름 목록 ###\n\t{cand_item_list}"
                    logger.info("LLM 모드: 참고용 후보 상품 목록을 RAG 컨텍스트에 추가")
                elif self.product_info_extraction_mode == 'nlp':
                    if not extra_item_pdf.empty and 'item_nm' in extra_item_pdf.columns:
                        product_df = extra_item_pdf.rename(columns={'item_nm': 'name'}).query(
                            "not name in @self.stop_item_names"
                        )[['name']]
                        product_df['action'] = '기타'
                        product_element = product_df.to_dict(orient='records') if product_df.shape[0] > 0 else None
                        logger.info(f"NLP 모드: 제품 요소 준비 완료 - {len(product_element) if product_element else 0}개")
                        if product_element:
                            logger.info(f"NLP 모드 제품 요소 샘플: {product_element[:2]}")
                    else:
                        logger.warning("NLP 모드: extra_item_pdf가 비어있거나 item_nm 컬럼이 없습니다!")
            else:
                logger.warning("후보 아이템이 없습니다!")
                logger.warning("이는 다음 중 하나의 문제일 수 있습니다:")
                logger.warning("1. 상품 데이터 로딩 실패")
                logger.warning("2. 엔티티 추출 실패") 
                logger.warning("3. 유사도 매칭 임계값 문제")

            # 5단계: LLM 프롬프트 구성 및 실행
            logger.info("=" * 30 + " 5단계: LLM 호출 " + "=" * 30)
            prompt = self._build_extraction_prompt(msg, rag_context, product_element)
            logger.info(f"구성된 프롬프트 길이: {len(prompt)} 문자")
            logger.info(f"RAG 컨텍스트 포함 여부: {'후보 상품' in rag_context}")
            
            # 프롬프트 저장 (디버깅/미리보기용)
            self._store_prompt_for_preview(prompt, "main_extraction")
            
            result_json_text = self._safe_llm_invoke(prompt)
            logger.info(f"LLM 응답 길이: {len(result_json_text)} 문자")
            logger.info(f"LLM 응답 내용 (처음 500자): {result_json_text[:500]}...")
            
            # 6단계: JSON 파싱
            logger.info("=" * 30 + " 6단계: JSON 파싱 " + "=" * 30)
            json_objects_list = extract_json_objects(result_json_text)
            logger.info(f"추출된 JSON 객체 수: {len(json_objects_list)}개")
            
            if not json_objects_list:
                logger.warning("LLM이 유효한 JSON 객체를 반환하지 않았습니다")
                logger.warning(f"LLM 원본 응답: {result_json_text}")
                return self._create_fallback_result(msg)
            
            json_objects = json_objects_list[-1]
            logger.info(f"파싱된 JSON 객체 키: {list(json_objects.keys())}")
            logger.info(f"파싱된 JSON 내용: {json_objects}")
            
            # 스키마 응답 감지 및 처리
            is_schema_response = self._detect_schema_response(json_objects)
            if is_schema_response:
                logger.error("🚨 LLM이 스키마 정의를 반환했습니다! 실제 데이터가 아닙니다.")
                logger.error("재시도 또는 fallback 결과를 사용합니다.")
                return self._create_fallback_result(msg)

            raw_result = copy.deepcopy(json_objects)
            
            # 7단계: 엔티티 매칭 및 최종 결과 구성
            logger.info("=" * 30 + " 7단계: 최종 결과 구성 " + "=" * 30)
            final_result = self._build_final_result(json_objects, msg, pgm_info, entities_from_kiwi)
            
            # 8단계: 결과 검증
            logger.info("=" * 30 + " 8단계: 결과 검증 " + "=" * 30)
            final_result = self._validate_extraction_result(final_result)

            # # DAG 추출 프로세스 (선택적)
            # # 메시지에서 엔티티 간의 관계를 방향성 있는 그래프로 추출
            # # 예: (고객:가입) -[하면]-> (혜택:수령) -[통해]-> (만족도:향상)
            # dag_section = ""
            # if self.extract_entity_dag:
            #     logger.info("=" * 30 + " DAG 추출 시작 " + "=" * 30)
            #     try:
            #         dag_start_time = time.time()
            #         # DAG 추출 함수 호출 (entity_dag_extractor.py)
            #         extract_dag_result = extract_dag(DAGParser(), msg, self.llm_model)
            #         dag_raw = extract_dag_result['dag_raw']      # LLM 원본 응답
            #         dag_section = extract_dag_result['dag_section']  # 파싱된 DAG 텍스트
            #         dag = extract_dag_result['dag']             # NetworkX 그래프 객체
                    
            #         # 시각적 다이어그램 생성 (utils.py)
            #         dag_filename = f'dag_{sha256_hash(msg)}'
            #         create_dag_diagram(dag, filename=dag_filename)
            #         dag_processing_time = time.time() - dag_start_time
                    
            #         logger.info(f"✅ DAG 추출 완료: {dag_filename}")
            #         logger.info(f"🕒 DAG 처리 시간: {dag_processing_time:.3f}초")
            #         logger.info(f"📏 DAG 섹션 길이: {len(dag_section)}자")
            #         if dag_section:
            #             logger.info(f"📄 DAG 내용 미리보기: {dag_section[:200]}...")
            #         else:
            #             logger.warning("⚠️ DAG 섹션이 비어있습니다")
                        
            #     except Exception as e:
            #         logger.error(f"❌ DAG 추출 중 오류 발생: {e}")
            #         dag_section = ""

            # # 최종 결과에 DAG 정보 추가 (비어있을 수도 있음)
            # final_result['entity_dag'] = sorted([d for d in dag_section.split('\n') if d!=''])
            
            # 최종 결과 요약 로깅
            logger.info("=" * 60)
            logger.info("✅ 메시지 처리 완료 - 최종 결과 요약")
            logger.info("=" * 60)
            logger.info(f"제목: {final_result.get('title', 'N/A')}")
            logger.info(f"목적: {final_result.get('purpose', [])}")
            logger.info(f"상품 수: {len(final_result.get('product', []))}개")
            logger.info(f"채널 수: {len(final_result.get('channel', []))}개")
            logger.info(f"프로그램 수: {len(final_result.get('pgm', []))}개")

            actual_prompts = get_stored_prompts_from_thread()

            return {"extracted_result": final_result, "raw_result": raw_result, "prompts": actual_prompts}
            
        except Exception as e:
            logger.error(f"메시지 처리 실패: {e}")
            logger.error(traceback.format_exc())
            return self._create_fallback_result(mms_msg)
    
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
            pgm_info = self._prepare_program_classification(msg)
            
            # RAG 컨텍스트 준비 (product_info_extraction_mode가 'rag'인 경우)
            rag_context = ""
            if self.product_info_extraction_mode == 'rag':
                rag_context = self._prepare_rag_context(msg)
            
            # 5단계: 프롬프트 구성 및 LLM 호출
            prompt = self._build_extraction_prompt(msg, pgm_info, rag_context)
            result_json_text = self._safe_llm_invoke(prompt)
            
            # 6단계: JSON 파싱
            json_objects_list = extract_json_objects(result_json_text)
            
            if not json_objects_list:
                logger.warning("LLM이 유효한 JSON 객체를 반환하지 않았습니다")
                return {}
            
            json_objects = json_objects_list[-1]
            
            # 스키마 응답 감지
            is_schema_response = self._detect_schema_response(json_objects)
            if is_schema_response:
                logger.warning("LLM이 스키마 정의를 반환했습니다")
                return {}
            
            logger.info(f"JSON 객체 추출 완료 - 키: {list(json_objects.keys())}")
            return json_objects
            
        except Exception as e:
            logger.error(f"JSON 객체 추출 중 오류 발생: {e}")
            return {}
    
    def _prepare_program_classification(self, mms_msg: str) -> Dict[str, Any]:
        """프로그램 분류 준비 (_classify_programs 메소드와 동일)"""
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

    def _detect_schema_response(self, json_objects: Dict) -> bool:
        """LLM이 스키마 정의를 반환했는지 감지"""
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

    def _create_fallback_result(self, msg: str) -> Dict[str, Any]:
        """처리 실패 시 기본 결과 생성"""
        return {
            "title": "광고 메시지",
            "purpose": ["정보 제공"],
            "product": [],
            "channel": [],
            "pgm": []
        }

    def _build_final_result(self, json_objects: Dict, msg: str, pgm_info: Dict, entities_from_kiwi: List[str]) -> Dict[str, Any]:
        """최종 결과 구성"""
        try:
            final_result = json_objects.copy()
            
            # 상품 정보에서 엔티티 추출
            product_items = json_objects.get('product', [])
            if isinstance(product_items, dict):
                product_items = product_items.get('items', [])

            primary_llm_extracted_entities = [x.get('name', '') for x in product_items]
            logger.info(f"Primary LLM 추출 엔티티: {primary_llm_extracted_entities}")
            logger.info(f"Kiwi 엔티티: {entities_from_kiwi}")

            # 엔티티 매칭 모드에 따른 처리
            if self.entity_extraction_mode == 'logic':
                # 로직 기반: 퍼지 + 시퀀스 유사도
                cand_entities = list(set(entities_from_kiwi+[item.get('name', '') for item in product_items if item.get('name')]))
                similarities_fuzzy = self.extract_entities_by_logic(cand_entities)
            else:
                # LLM 기반: LLM을 통한 엔티티 추출 (기본 모델들: ax=ax, cld=claude)
                default_llm_models = self._initialize_multiple_llm_models(['ax','gen'])
                similarities_fuzzy = self.extract_entities_by_llm(msg, llm_models=default_llm_models, external_cand_entities=entities_from_kiwi)

            # similarities_fuzzy = similarities_fuzzy[similarities_fuzzy.apply(lambda x: (x['item_nm_alias'].replace(' ', '').lower() in x['item_name_in_msg'].replace(' ', '').lower() or x['item_name_in_msg'].replace(' ', '').lower() in x['item_nm_alias'].replace(' ', '').lower()) , axis=1)]
            merged_df = similarities_fuzzy.merge(
                self.alias_pdf_raw[['alias_1','type']].drop_duplicates(), 
                left_on='item_name_in_msg', 
                right_on='alias_1', 
                how='left'
            )

            filtered_df = merged_df[merged_df.apply(
                lambda x: (
                    replace_special_chars_with_space(x['item_nm_alias']) in replace_special_chars_with_space(x['item_name_in_msg']) or 
                    replace_special_chars_with_space(x['item_name_in_msg']) in replace_special_chars_with_space(x['item_nm_alias'])
                ) if x['type'] != 'expansion' else True, 
                axis=1
            )]

            # similarities_fuzzy = filtered_df[similarities_fuzzy.columns]

            # 상품 정보 매핑
            if not similarities_fuzzy.empty:
                final_result['product'] = self._map_products_with_similarity(similarities_fuzzy, json_objects)
            else:
                # 유사도 결과가 없으면 LLM 결과 그대로 사용 (새 스키마 + expected_action 리스트)
                final_result['product'] = [
                    {
                        'item_nm': d.get('name', ''), 
                        'item_id': ['#'],
                        'item_name_in_msg': [d.get('name', '')],
                        'expected_action': [d.get('action', '기타')]
                    } 
                    for d in product_items 
                    if d.get('name') and d['name'] not in self.stop_item_names
                ]

            # 프로그램 분류 정보 매핑
            final_result['pgm'] = self._map_program_classification(json_objects, pgm_info)
            
            # 채널 정보 처리
            final_result['channel'] = self._extract_channels(json_objects, msg)

            return final_result
            
        except Exception as e:
            logger.error(f"최종 결과 구성 실패: {e}")
            return json_objects

    def _map_products_with_similarity(self, similarities_fuzzy: pd.DataFrame, json_objects: Dict = None) -> List[Dict]:
        """유사도를 기반으로 상품 정보 매핑"""
        try:
            # 높은 유사도 아이템들 필터링
            high_sim_threshold = getattr(PROCESSING_CONFIG, 'high_similarity_threshold', 1.5)
            high_sim_items = similarities_fuzzy.query('sim >= @high_sim_threshold')['item_nm_alias'].unique()
            filtered_similarities = similarities_fuzzy[
                (similarities_fuzzy['item_nm_alias'].isin(high_sim_items)) &
                (~similarities_fuzzy['item_nm_alias'].str.contains('test', case=False)) &
                (~similarities_fuzzy['item_name_in_msg'].isin(self.stop_item_names))
            ]
            
            # 상품 정보와 매핑하여 최종 결과 생성 (새 스키마 + expected_action)
            product_tag = self.convert_df_to_json_list(
                self.item_pdf_all.merge(filtered_similarities, on=['item_nm_alias'])
            )
            
            # Add expected_action to each product
            if json_objects:
                action_mapping = self._create_action_mapping(json_objects)
                for product in product_tag:
                    item_names_in_msg = product.get('item_name_in_msg', [])
                    # 배열의 각 항목에 대해 모든 action 찾기 (리스트로 수집, 중복 제거)
                    found_actions = []
                    for item_name in item_names_in_msg:
                        if item_name in action_mapping:
                            found_actions.append(action_mapping[item_name])
                    # 중복 제거 (순서 유지)
                    product['expected_action'] = list(dict.fromkeys(found_actions)) if found_actions else ['기타']
            
            return product_tag
            
        except Exception as e:
            logger.error(f"상품 정보 매핑 실패: {e}")
            return []

    def _create_action_mapping(self, json_objects: Dict) -> Dict[str, str]:
        """LLM 응답에서 상품명-액션 매핑 생성"""
        try:
            action_mapping = {}
            product_data = json_objects.get('product', [])
            
            if isinstance(product_data, list):
                # 정상적인 배열 구조
                for item in product_data:
                    if isinstance(item, dict) and 'name' in item and 'action' in item:
                        action_mapping[item['name']] = item['action']
            elif isinstance(product_data, dict):
                # 스키마 구조 또는 기타 딕셔너리 구조 처리
                if 'items' in product_data:
                    # 스키마 구조: {"items": [...]}
                    items = product_data.get('items', [])
                    for item in items:
                        if isinstance(item, dict) and 'name' in item and 'action' in item:
                            action_mapping[item['name']] = item['action']
                elif 'type' in product_data and product_data.get('type') == 'array':
                    # 스키마 정의 구조는 건너뛰기
                    logger.debug("스키마 정의 구조 감지됨, 액션 매핑 건너뛰기")
                else:
                    # 기타 딕셔너리 구조 처리
                    if 'name' in product_data and 'action' in product_data:
                        action_mapping[product_data['name']] = product_data['action']
            
            logger.debug(f"생성된 액션 매핑: {action_mapping}")
            return action_mapping
            
        except Exception as e:
            logger.error(f"액션 매핑 생성 실패: {e}")
            return {}

    def _map_program_classification(self, json_objects: Dict, pgm_info: Dict) -> List[Dict]:
        """프로그램 분류 정보 매핑"""
        try:
            if (self.num_cand_pgms > 0 and 
                'pgm' in json_objects and 
                isinstance(json_objects['pgm'], list) and
                not pgm_info.get('pgm_pdf_tmp', pd.DataFrame()).empty):
                
                pgm_json = pgm_info['pgm_pdf_tmp'][
                    pgm_info['pgm_pdf_tmp']['pgm_nm'].apply(
                        lambda x: re.sub(r'\[.*?\]', '', x) in ' '.join(json_objects['pgm'])
                    )
                ][['pgm_nm', 'pgm_id']].to_dict('records')
                
                return pgm_json
            
            return []
            
        except Exception as e:
            logger.error(f"프로그램 분류 매핑 실패: {e}")
            return []

def process_message_with_dag(extractor, message: str, extract_dag: bool = False) -> Dict[str, Any]:
    """
    단일 메시지를 처리하는 워커 함수 (멀티프로세스용)
    
    Args:
        extractor: MMSExtractor 인스턴스
        message: 처리할 메시지
        extract_dag: DAG 추출 여부
    
    Returns:
        dict: 처리 결과 (프롬프트 정보 포함)
    """
    try:
        logger.info(f"워커 프로세스에서 메시지 처리 시작: {message[:50]}...")

        # 1. 메인 추출
        result = extractor.process_message(message)
        dag_list = []
        
        if extract_dag:
            # 순차적 처리로 변경 (프롬프트 캡처를 위해)
            # 멀티스레드를 사용하면 스레드 로컬 저장소가 분리되어 프롬프트 캡처가 안됨
            logger.info("순차적 처리로 메인 추출 및 DAG 추출 수행")
            
            # 2. DAG 추출
            dag_result = make_entity_dag(message, extractor.llm_model)
            dag_list = sorted([d for d in dag_result['dag_section'].split('\n') if d!=''])

        extracted_result = result.get('extracted_result', {})
        extracted_result['entity_dag'] = dag_list
        result['extracted_result'] = extracted_result

        raw_result = result.get('raw_result', {})
        raw_result['entity_dag'] = dag_list
        result['raw_result'] = raw_result

        result['error'] = ""
        
        logger.info(f"워커 프로세스에서 메시지 처리 완료")
        return result
        
    except Exception as e:
        logger.error(f"워커 프로세스에서 메시지 처리 실패: {e}")
        return {
            "title": "처리 실패",
            "purpose": ["오류"],
            "product": [],
            "channel": [],
            "pgm": [],
            "entity_dag": [],
            "error": str(e)
        }

def process_messages_batch(extractor, messages: List[str], extract_dag: bool = False, max_workers: int = None) -> List[Dict[str, Any]]:
    """
    여러 메시지를 배치로 처리하는 함수
    
    Args:
        extractor: MMSExtractor 인스턴스
        messages: 처리할 메시지 리스트
        extract_dag: DAG 추출 여부
        max_workers: 최대 워커 수 (None이면 CPU 코어 수)
    
    Returns:
        list: 처리 결과 리스트
    """
    if max_workers is None:
        max_workers = min(len(messages), os.cpu_count())
    
    logger.info(f"배치 처리 시작: {len(messages)}개 메시지, {max_workers}개 워커")
    
    start_time = time.time()
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 모든 메시지에 대해 작업 제출
        future_to_message = {
            executor.submit(process_message_with_dag, extractor, msg, extract_dag): msg 
            for msg in messages
        }
        
        # 완료된 작업들 수집
        for i, future in enumerate(future_to_message):
            try:
                result = future.result()
                results.append(result)
                logger.info(f"배치 처리 진행률: {i+1}/{len(messages)} ({((i+1)/len(messages)*100):.1f}%)")
            except Exception as e:
                logger.error(f"배치 처리 중 오류 발생: {e}")
                results.append({
                    "title": "처리 실패",
                    "purpose": ["오류"],
                    "product": [],
                    "channel": [],
                    "pgm": [],
                    "entity_dag": [],
                    "error": str(e)
                })
    
    elapsed_time = time.time() - start_time
    logger.info(f"배치 처리 완료: {len(messages)}개 메시지, {elapsed_time:.2f}초")
    logger.info(f"평균 처리 시간: {elapsed_time/len(messages):.2f}초/메시지")
    
    return results

def make_entity_dag(msg: str, llm_model, save_dag_image=True):

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
            dag_filename = f'dag_{sha256_hash(msg)}'
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
    """현재 스레드에서 저장된 프롬프트 정보를 가져오는 함수"""
    import threading
    current_thread = threading.current_thread()
    
    if hasattr(current_thread, 'stored_prompts'):
        return current_thread.stored_prompts
    else:
        return {}

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
        from mongodb_utils import save_to_mongodb
        
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
            'result': result.get('extracted_result', result.get('result', {})),
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
        
        # MongoDB에 저장
        user_id = getattr(args, 'user_id', 'DEFAULT_USER')
        saved_id = save_to_mongodb(message, extraction_result, raw_result_data, extraction_prompts, 
                                 user_id=user_id, message_id=None)
        
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

            
    except Exception as e:
        print(f"❌ MongoDB 저장 중 오류 발생: {str(e)}")
        return None

def main():
    """
    커맨드라인에서 실행할 때의 메인 함수
    다양한 옵션을 통해 추출기 설정을 변경할 수 있습니다.
    
    사용법:
    # 단일 메시지 처리 (멀티스레드)
    python mms_extractor.py --message "광고 메시지" --extract-entity-dag
    
    # 배치 처리 (멀티프로세스)
    python mms_extractor.py --batch-file messages.txt --max-workers 4 --extract-entity-dag
    
    # 데이터베이스 모드로 배치 처리
    python mms_extractor.py --batch-file messages.txt --offer-data-source db --max-workers 8
    
    # MongoDB에 결과 저장
    python mms_extractor.py --message "광고 메시지" --save-to-mongodb --extract-entity-dag
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='MMS 광고 텍스트 추출기 - 개선된 버전')
    parser.add_argument('--message', type=str, help='테스트할 메시지')
    parser.add_argument('--batch-file', type=str, help='배치 처리할 메시지가 담긴 파일 경로 (한 줄에 하나씩)')
    parser.add_argument('--max-workers', type=int, help='배치 처리 시 최대 워커 수 (기본값: CPU 코어 수)')
    parser.add_argument('--offer-data-source', choices=['local', 'db'], default='local',
                       help='데이터 소스 (local: CSV 파일, db: 데이터베이스)')
    parser.add_argument('--product-info-extraction-mode', choices=['nlp', 'llm', 'rag'], default='llm',
                       help='상품 정보 추출 모드 (nlp: 형태소분석, llm: LLM 기반, rag: 검색증강생성)')
    parser.add_argument('--entity-matching-mode', choices=['logic', 'llm'], default='llm',
                       help='엔티티 매칭 모드 (logic: 로직 기반, llm: LLM 기반)')
    parser.add_argument('--llm-model', choices=['gem', 'ax', 'cld', 'gen', 'gpt'], default='gen',
                       help='사용할 LLM 모델 (gem: Gemma, ax: ax, cld: Claude, gen: Gemini, gpt: GPT)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO',
                       help='로그 레벨 설정')
    parser.add_argument('--extract-entity-dag', action='store_true', default=False, help='Entity DAG extraction (default: False)')
    parser.add_argument('--save-to-mongodb', action='store_true', default=True, 
                       help='추출 결과를 MongoDB에 저장 (mongodb_utils.py 필요)')
    parser.add_argument('--test-mongodb', action='store_true', default=False,
                       help='MongoDB 연결 테스트만 수행하고 종료')

    args = parser.parse_args()
    
    # 로그 레벨 설정
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # MongoDB 연결 테스트만 수행하는 경우
    if args.test_mongodb:
        try:
            from mongodb_utils import test_mongodb_connection
        except ImportError:
            print("❌ MongoDB 유틸리티를 찾을 수 없습니다.")
            print("mongodb_utils.py 파일과 pymongo 패키지를 확인하세요.")
            exit(1)
        
        print("🔌 MongoDB 연결 테스트 중...")
        if test_mongodb_connection():
            print("✅ MongoDB 연결 성공!")
            exit(0)
        else:
            print("❌ MongoDB 연결 실패!")
            print("MongoDB 서버가 실행 중인지 확인하세요.")
            exit(1)
    
    try:
                # 추출기 초기화
        logger.info("MMS 추출기 초기화 중...")
        extractor = MMSExtractor(
            offer_info_data_src=args.offer_data_source,
            product_info_extraction_mode=args.product_info_extraction_mode,
            entity_extraction_mode=args.entity_matching_mode,
            llm_model=args.llm_model,
            extract_entity_dag=args.extract_entity_dag
        )
        
        # 배치 처리 또는 단일 메시지 처리
        if args.batch_file:
            # 배치 파일에서 메시지들 로드
            logger.info(f"배치 파일에서 메시지 로드: {args.batch_file}")
            try:
                with open(args.batch_file, 'r', encoding='utf-8') as f:
                    messages = [line.strip() for line in f if line.strip()]
                
                logger.info(f"로드된 메시지 수: {len(messages)}개")
                
                # 배치 처리 실행
                results = process_messages_batch(
                    extractor, 
                    messages, 
                    extract_dag=args.extract_entity_dag,
                    max_workers=args.max_workers
                )
                
                # MongoDB 저장 (배치 처리)
                if args.save_to_mongodb:
                    print("\n📄 MongoDB 저장 중...")
                    args.processing_mode = 'batch'
                    saved_count = 0
                    for i, result in enumerate(results):
                        if i < len(messages):  # 메시지가 있는 경우만
                            saved_id = save_result_to_mongodb_if_enabled(messages[i], result, args, extractor)
                            if saved_id:
                                saved_count += 1
                    print(f"📄 MongoDB 저장 완료: {saved_count}/{len(results)}개")
                
                # 배치 결과 출력
                print("\n" + "="*50)
                print("🎯 배치 처리 결과")
                print("="*50)
                
                for i, result in enumerate(results):
                    print(f"\n--- 메시지 {i+1} ---")
                    print(f"제목: {result.get('title', 'N/A')}")
                    print(f"상품: {len(result.get('product', []))}개")
                    if result.get('error'):
                        print(f"오류: {result['error']}")
                
                # 전체 배치 통계
                successful = len([r for r in results if not r.get('error')])
                failed = len(results) - successful
                print(f"\n📊 배치 처리 통계")
                print(f"✅ 성공: {successful}개")
                print(f"❌ 실패: {failed}개")
                print(f"📈 성공률: {(successful/len(results)*100):.1f}%")
                
                # 결과를 JSON 파일로 저장
                output_file = f"batch_results_{int(time.time())}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=4, ensure_ascii=False)
                print(f"💾 결과 저장: {output_file}")
                
            except FileNotFoundError:
                logger.error(f"배치 파일을 찾을 수 없습니다: {args.batch_file}")
                exit(1)
            except Exception as e:
                logger.error(f"배치 파일 처리 실패: {e}")
                exit(1)
        
        else:
            # 단일 메시지 처리
            test_message = args.message if args.message else """
  message: '[SKT] Netflix 광고형 스탠다드 구독료 변경 안내__고객님, 안녕하세요._2025년 12월 1일(월)부터 Netflix 광고형 스탠다드 구독료가 변경됩니다.__요금제 혜택으로 Netflix 광고형 스탠다드를 추가 요금 없이 이용 중인 경우, 별도 안내 전까지 기존 구독료로 동일하게 즐기실 수 있습니다.__아직 가입하지 않으셨다면, 아래 URL을 통해 가입 가능합니다.__▶ 가입하기: https://m.sktuniverse.co.kr/product/detail?prdId=PR00000501__■ 변경 내용_- 대상: Netflix 광고형 스탠다드_- 변경일: 2025년 12월 1일(월)_- 내용: 월 구독료 변경(5,500원 → 7,000원)_* 2025년 11월 30일(일)까지 Netflix 광고형 스탠다드와 할인 대상 요금제 모두 가입 시, 별도 안내 전까지 기존 구독료로 계속 이용 가능__■ 유의 사항_- 구독료 변경 후에도 <T 우주 Netflix> 광고형 스탠다드 할인 요금제 혜택은 기존과 동일(5,500원 할인)하게 유지됩니다._* 대상 요금제: 5GX 프라임(넷플릭스), 0 청년 89(넷플릭스), 다이렉트5G 62(넷플릭스), 0 청년 다이렉트 62(넷플릭스)_- 2025년 12월 1일(월)부터 할인 대상 요금제 또는 Netflix 광고형 스탠다드 상품 신규가입 시 변경된 구독료로 결제됩니다._ - Wavve와 결합된 <T 우주패스 Netflix>에 가입한 경우, 2025년 12월 1일(월)부터 가격이 인상됩니다.__■ 문의: SKT 고객센터(114)__SKT와 함께해 주셔서 감사합니다.',


"""
            
            # 단일 메시지 처리 (멀티스레드)
            logger.info("단일 메시지 처리 시작 (멀티스레드)")
            result = process_message_with_dag(extractor, test_message, args.extract_entity_dag)
                    
            # MongoDB 저장 (단일 메시지)
            if args.save_to_mongodb:
                print("\n📄 MongoDB 저장 중...")
                args.processing_mode = 'single'
                saved_id = save_result_to_mongodb_if_enabled(test_message, result, args, extractor)
                if saved_id:
                    print("📄 MongoDB 저장 완료!")

            
            extracted_result = result.get('extracted_result', {})
        
            print("\n" + "="*50)
            print("🎯 최종 추출된 정보")
            print("="*50)
            print(json.dumps(extracted_result, indent=4, ensure_ascii=False))

            # 성능 요약 정보 출력
            print("\n" + "="*50)
            print("📊 처리 완료")
            print("="*50)
            print(f"✅ 제목: {extracted_result.get('title', 'N/A')}")
            print(f"✅ 목적: {len(extracted_result.get('purpose', []))}개")
            print(f"✅ 상품: {len(extracted_result.get('product', []))}개")
            print(f"✅ 채널: {len(extracted_result.get('channel', []))}개")
            print(f"✅ 프로그램: {len(extracted_result.get('pgm', []))}개")
            if extracted_result.get('error'):
                print(f"❌ 오류: {extracted_result['error']}")
        
    except Exception as e:
        logger.error(f"실행 실패: {e}")
        logger.error(traceback.format_exc())
        exit(1)


if __name__ == '__main__':
    main()
# %%
