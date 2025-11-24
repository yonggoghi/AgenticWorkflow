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
    SIMPLE_ENTITY_EXTRACTION_PROMPT,
    HYBRID_DAG_EXTRACTION_PROMPT
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

# Mixin 클래스 임포트
from mms_extractor_data import MMSExtractorDataMixin
from mms_extractor_entity import MMSExtractorEntityMixin

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

# ===== 개선된 MMSExtractor 클래스 =====


class MMSExtractor(MMSExtractorDataMixin, MMSExtractorEntityMixin):
    """
    MMS 광고 텍스트 AI 분석 시스템 - 메인 추출 엔진
    ================================================================
    
    🎨 개요
    -------
    이 클래스는 MMS 광고 텍스트에서 구조화된 정보를 추출하는 핵심 엔진입니다.
    LLM(Large Language Model), 임베딩 모델, NLP 기법을 조합하여
    비정형 텍스트에서 정형화된 데이터를 추출합니다.
    
    🏗️ 아키텍처
    -----------
    이 클래스는 Mixin 패턴을 사용하여 기능별로 모듈화되어 있습니다:
    - **MMSExtractorDataMixin**: 데이터 로딩 및 초기화 기능
    - **MMSExtractorEntityMixin**: 엔티티 추출 및 매칭 기능
    - **MMSExtractor**: 핵심 추출 로직 및 통합
    
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

    def _initialize_device(self):
        """사용할 디바이스 초기화"""
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        logger.info(f"Using device: {self.device}")

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

    def _extract_channels(self, json_objects: Dict, msg: str, offer_object: Dict) -> tuple[List[Dict], Dict]:
        """채널 정보 추출 및 매칭 (offer_object도 함께 반환)"""
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
                    
                    # offer_object를 org 타입으로 변경
                    if store_info:
                        offer_object['type'] = 'org'
                        org_tmp = [
                            {
                                'item_nm': o['org_nm'], 
                                'item_id': o['org_cd'], 
                                'item_name_in_msg': d['value'], 
                                'expected_action': ['방문']
                            } 
                            for o in store_info
                        ]
                        offer_object['value'] = org_tmp
                else:
                    d['store_info'] = []
                channel_tag.append(d)

            return channel_tag, offer_object
            
        except Exception as e:
            logger.error(f"채널 정보 추출 실패: {e}")
            return [], offer_object

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
            sales_script = final_result.get('sales_script', '')
            if sales_script:
                logger.info(f"판매 스크립트: {sales_script[:100]}..." if len(sales_script) > 100 else f"판매 스크립트: {sales_script}")
            logger.info(f"상품 수: {len(final_result.get('product', []))}개")
            logger.info(f"채널 수: {len(final_result.get('channel', []))}개")
            logger.info(f"프로그램 수: {len(final_result.get('pgm', []))}개")
            offer_info = final_result.get('offer', {})
            logger.info(f"오퍼 타입: {offer_info.get('type', 'N/A')}")
            logger.info(f"오퍼 항목 수: {len(offer_info.get('value', []))}개")

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
            "sales_script": "",
            "product": [],
            "channel": [],
            "pgm": [],
            "offer": {"type": "product", "value": []},
            "entity_dag": []
        }

    def _build_final_result(self, json_objects: Dict, msg: str, pgm_info: Dict, entities_from_kiwi: List[str]) -> Dict[str, Any]:
        """최종 결과 구성"""
        try:
            logger.info("=" * 80)
            logger.info("🔍 [PRODUCT DEBUG] _build_final_result 시작")
            logger.info("=" * 80)
            
            final_result = json_objects.copy()
            
            # offer_object 초기화
            offer_object = {}
            
            # 상품 정보에서 엔티티 추출
            logger.info("📋 [STEP 1] product_items 추출")
            product_items = json_objects.get('product', [])
            logger.info(f"   - 원본 product 타입: {type(product_items)}")
            logger.info(f"   - 원본 product 내용: {product_items}")
            
            if isinstance(product_items, dict):
                logger.info("   - product가 dict 타입 → 'items' 키로 접근")
                product_items = product_items.get('items', [])
                logger.info(f"   - items 추출 후: {product_items}")
            
            logger.info(f"   ✅ 최종 product_items 개수: {len(product_items)}개")
            logger.info(f"   ✅ 최종 product_items 내용: {product_items}")

            primary_llm_extracted_entities = [x.get('name', '') for x in product_items]
            logger.info(f"📋 [STEP 2] LLM 추출 엔티티: {primary_llm_extracted_entities}")
            logger.info(f"📋 [STEP 2] Kiwi 엔티티: {entities_from_kiwi}")
            logger.info(f"📋 [STEP 2] entity_extraction_mode: {self.entity_extraction_mode}")

            # 엔티티 매칭 모드에 따른 처리
            if self.entity_extraction_mode == 'logic':
                logger.info("🔍 [STEP 3] 로직 기반 엔티티 매칭 시작")
                # 로직 기반: 퍼지 + 시퀀스 유사도
                cand_entities = list(set(entities_from_kiwi+[item.get('name', '') for item in product_items if item.get('name')]))
                logger.info(f"   - cand_entities: {cand_entities}")
                similarities_fuzzy = self.extract_entities_by_logic(cand_entities)
                logger.info(f"   ✅ similarities_fuzzy 결과 크기: {similarities_fuzzy.shape if not similarities_fuzzy.empty else '비어있음'}")
            else:
                logger.info("🔍 [STEP 3] LLM 기반 엔티티 매칭 시작")
                # LLM 기반: LLM을 통한 엔티티 추출 (기본 모델들: ax=ax, cld=claude)
                default_llm_models = self._initialize_multiple_llm_models(['gen','ax'])
                logger.info(f"   - 초기화된 LLM 모델 수: {len(default_llm_models)}개")
                similarities_fuzzy = self.extract_entities_by_llm(msg, llm_models=default_llm_models, external_cand_entities=entities_from_kiwi)
                logger.info(f"   ✅ similarities_fuzzy 결과 크기: {similarities_fuzzy.shape if not similarities_fuzzy.empty else '비어있음'}")
            
            if not similarities_fuzzy.empty:
                logger.info(f"   📊 similarities_fuzzy 샘플 (처음 3개):")
                logger.info(f"{similarities_fuzzy.head(3).to_dict('records')}")
            else:
                logger.warning("   ⚠️ similarities_fuzzy가 비어있습니다!")

            if not similarities_fuzzy.empty:
                logger.info("🔍 [STEP 4] alias_pdf_raw와 merge 시작")
                logger.info(f"   - alias_pdf_raw 크기: {self.alias_pdf_raw.shape}")
                merged_df = similarities_fuzzy.merge(
                    self.alias_pdf_raw[['alias_1','type']].drop_duplicates(), 
                    left_on='item_name_in_msg', 
                    right_on='alias_1', 
                    how='left'
                )
                logger.info(f"   ✅ merged_df 크기: {merged_df.shape if not merged_df.empty else '비어있음'}")
                if not merged_df.empty:
                    logger.info(f"   📊 merged_df 샘플 (처음 3개):")
                    logger.info(f"{merged_df.head(3).to_dict('records')}")

                logger.info("🔍 [STEP 5] filtered_df 생성 (expansion 타입 필터링)")
                filtered_df = merged_df[merged_df.apply(
                    lambda x: (
                        replace_special_chars_with_space(x['item_nm_alias']) in replace_special_chars_with_space(x['item_name_in_msg']) or 
                        replace_special_chars_with_space(x['item_name_in_msg']) in replace_special_chars_with_space(x['item_nm_alias'])
                    ) if x['type'] != 'expansion' else True, 
                    axis=1
                )]
                logger.info(f"   ✅ filtered_df 크기: {filtered_df.shape if not filtered_df.empty else '비어있음'}")
                if not filtered_df.empty:
                    logger.info(f"   📊 filtered_df 샘플 (처음 3개):")
                    logger.info(f"{filtered_df.head(3).to_dict('records')}")

                # similarities_fuzzy = filtered_df[similarities_fuzzy.columns]

            # 상품 정보 매핑
            logger.info("🔍 [STEP 6] 상품 정보 매핑 시작")
            logger.info(f"   - similarities_fuzzy.empty: {similarities_fuzzy.empty}")
            
            if not similarities_fuzzy.empty:
                logger.info("   ✅ similarities_fuzzy가 비어있지 않음 → _map_products_with_similarity 호출")
                final_result['product'] = self._map_products_with_similarity(similarities_fuzzy, json_objects)
                logger.info(f"   ✅ 최종 product 개수: {len(final_result['product'])}개")
                logger.info(f"   ✅ 최종 product 내용: {final_result['product']}")
            else:
                logger.warning("   ⚠️ similarities_fuzzy가 비어있음 → LLM 결과 그대로 사용 (else 브랜치)")
                logger.info(f"   - product_items 개수: {len(product_items)}개")
                logger.info(f"   - stop_item_names 개수: {len(self.stop_item_names)}개")
                
                # 유사도 결과가 없으면 LLM 결과 그대로 사용 (새 스키마 + expected_action 리스트)
                filtered_product_items = [
                    d for d in product_items 
                    if d.get('name') and d['name'] not in self.stop_item_names
                ]
                logger.info(f"   - 필터링 후 product_items 개수: {len(filtered_product_items)}개")
                logger.info(f"   - 필터링 후 product_items: {filtered_product_items}")
                
                final_result['product'] = [
                    {
                        'item_nm': d.get('name', ''), 
                        'item_id': ['#'],
                        'item_name_in_msg': [d.get('name', '')],
                        'expected_action': [d.get('action', '기타')]
                    } 
                    for d in filtered_product_items
                ]
                logger.info(f"   ✅ 최종 product 개수: {len(final_result['product'])}개")
                logger.info(f"   ✅ 최종 product 내용: {final_result['product']}")

            # offer_object에 product 타입으로 설정
            offer_object['type'] = 'product'
            offer_object['value'] = final_result['product']
            logger.info(f"🏷️  [STEP 7] offer_object 초기화: type=product, value 개수={len(offer_object['value'])}개")

            # 프로그램 분류 정보 매핑
            final_result['pgm'] = self._map_program_classification(json_objects, pgm_info)
            
            # 채널 정보 처리 (offer_object도 함께 전달 및 반환)
            logger.info("🔍 [STEP 8] 채널 정보 처리 및 offer_object 업데이트")
            final_result['channel'], offer_object = self._extract_channels(json_objects, msg, offer_object)
            logger.info(f"   ✅ 최종 channel 개수: {len(final_result['channel'])}개")
            logger.info(f"   ✅ 최종 offer_object type: {offer_object.get('type', 'N/A')}")
            logger.info(f"   ✅ 최종 offer_object value 개수: {len(offer_object.get('value', []))}개")
            
            # offer 필드 추가
            final_result['offer'] = offer_object
            logger.info(f"✅ [STEP 9] final_result에 offer 필드 추가 완료")
            
            # entity_dag 초기화 (빈 배열)
            final_result['entity_dag'] = []
            
            logger.info("=" * 80)
            logger.info("✅ [PRODUCT DEBUG] _build_final_result 완료")
            logger.info(f"   최종 final_result['product'] 개수: {len(final_result.get('product', []))}개")
            logger.info("=" * 80)

            return final_result
            
        except Exception as e:
            logger.error(f"최종 결과 구성 실패: {e}")
            return json_objects

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
            "extracted_result": {
                "title": "처리 실패",
                "purpose": ["오류"],
                "sales_script": "",
                "product": [],
                "channel": [],
                "pgm": [],
                "offer": {"type": "product", "value": []},
                "entity_dag": []
            },
            "raw_result": {},
            "prompts": {},
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
                    "extracted_result": {
                        "title": "처리 실패",
                        "purpose": ["오류"],
                        "sales_script": "",
                        "product": [],
                        "channel": [],
                        "pgm": [],
                        "offer": {"type": "product", "value": []},
                        "entity_dag": []
                    },
                    "raw_result": {},
                    "prompts": {},
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
                    extracted = result.get('extracted_result', {})
                    print(f"\n--- 메시지 {i+1} ---")
                    print(f"제목: {extracted.get('title', 'N/A')}")
                    sales_script = extracted.get('sales_script', '')
                    if sales_script:
                        print(f"판매 스크립트: {sales_script[:80]}..." if len(sales_script) > 80 else f"판매 스크립트: {sales_script}")
                    print(f"상품: {len(extracted.get('product', []))}개")
                    print(f"채널: {len(extracted.get('channel', []))}개")
                    print(f"프로그램: {len(extracted.get('pgm', []))}개")
                    offer_info = extracted.get('offer', {})
                    print(f"오퍼 타입: {offer_info.get('type', 'N/A')}")
                    print(f"오퍼 항목: {len(offer_info.get('value', []))}개")
                    if result.get('error'):
                        print(f"오류: {result['error']}")
                
                # 전체 배치 통계
                successful = len([r for r in results if not r.get('error') and r.get('extracted_result')])
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
  message: '(광고)[SKT] iPhone 신제품 구매 혜택 안내 __#04 고객님, 안녕하세요._SK텔레콤에서 iPhone 신제품 구매하면, 최대 22만 원 캐시백 이벤트에 참여하실 수 있습니다.__현대카드로 애플 페이도 더 편리하게 이용해 보세요.__▶ 현대카드 바로 가기: https://t-mms.kr/ais/#74_ _애플 페이 티머니 충전 쿠폰 96만 원, 샌프란시스코 왕복 항공권, 애플 액세서리 팩까지!_Lucky 1717 이벤트 응모하고 경품 당첨의 행운을 누려 보세요.__▶ 이벤트 자세히 보기: https://t-mms.kr/aiN/#74_ _■ 문의: SKT 고객센터(1558, 무료)__SKT와 함께해 주셔서 감사합니다.__무료 수신거부 1504',


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
            sales_script = extracted_result.get('sales_script', '')
            if sales_script:
                print(f"✅ 판매 스크립트: {sales_script[:100]}..." if len(sales_script) > 100 else f"✅ 판매 스크립트: {sales_script}")
            print(f"✅ 상품: {len(extracted_result.get('product', []))}개")
            print(f"✅ 채널: {len(extracted_result.get('channel', []))}개")
            print(f"✅ 프로그램: {len(extracted_result.get('pgm', []))}개")
            offer_info = extracted_result.get('offer', {})
            print(f"✅ 오퍼 타입: {offer_info.get('type', 'N/A')}")
            print(f"✅ 오퍼 항목: {len(offer_info.get('value', []))}개")
            if extracted_result.get('error'):
                print(f"❌ 오류: {extracted_result['error']}")
        
    except Exception as e:
        logger.error(f"실행 실패: {e}")
        logger.error(traceback.format_exc())
        exit(1)


if __name__ == '__main__':
    main()
# %%
