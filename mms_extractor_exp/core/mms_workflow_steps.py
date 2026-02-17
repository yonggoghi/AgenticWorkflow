"""
MMS Workflow Steps - MMS 추출기 워크플로우 단계 구현
===================================================

📋 개요
-------
이 모듈은 MMS 메시지 처리의 각 단계를 독립적인 클래스로 구현합니다.
각 단계는 `WorkflowStep`을 상속받아 `execute` 메서드를 구현하며,
`WorkflowState`를 통해 데이터를 주고받습니다.

🔗 의존성
---------
**사용하는 모듈:**
- `workflow_core`: WorkflowStep, WorkflowState 기반 클래스
- `services.*`: 각 단계에서 사용하는 서비스 (EntityRecognizer, ProgramClassifier 등)
- `utils`: 검증, 파싱, 프롬프트 관리 유틸리티

**사용되는 곳:**
- `core.mms_extractor`: MMSExtractor 초기화 시 워크플로우 엔진에 단계 등록

🏗️ 워크플로우 단계 순서
-----------------------

```mermaid
graph TB
    A[InputValidationStep] -->|msg| B[EntityExtractionStep]
    B -->|entities_from_kiwi, cand_item_list| C[ProgramClassificationStep]
    C -->|pgm_info| D[ContextPreparationStep]
    D -->|rag_context, product_element| E[LLMExtractionStep]
    E -->|result_json_text| F[ResponseParsingStep]
    F -->|json_objects, raw_result| G[EntityContextExtractionStep]
    G -->|extracted_entities| H[VocabularyFilteringStep]
    H -->|matched_products| I[ResultConstructionStep]
    I -->|final_result| J[ValidationStep]
    J -->|validated final_result| K{extract_entity_dag?}
    K -->|Yes| L[DAGExtractionStep]
    K -->|No| M[End]
    L -->|entity_dag| M

    style A fill:#e1f5ff
    style E fill:#ffe1e1
    style G fill:#ffd
    style H fill:#ffd
    style I fill:#fff4e1
    style L fill:#e1ffe1
    style M fill:#d4edda
```

📊 각 단계별 역할
----------------

### 1. InputValidationStep
**목적**: 입력 메시지 검증 및 설정 로깅
**입력**: mms_msg (원본 메시지)
**출력**: msg (검증된 메시지)
**주요 작업**:
- 메시지 유효성 검사
- 추출기 설정 상태 로깅
- 메시지 길이 및 내용 확인

### 2. EntityExtractionStep
**목적**: Kiwi 기반 엔티티 추출
**입력**: msg
**출력**: entities_from_kiwi, cand_item_list, extra_item_pdf
**주요 작업**:
- Kiwi 형태소 분석 (NNP 태그 추출)
- 임베딩 유사도 매칭
- DB 모드 진단 및 결과 분석

### 3. ProgramClassificationStep
**목적**: 프로그램 카테고리 분류
**입력**: msg
**출력**: pgm_info (프로그램 분류 정보)
**주요 작업**:
- 임베딩 기반 프로그램 유사도 계산
- 상위 N개 후보 프로그램 선택

### 4. ContextPreparationStep
**목적**: RAG 컨텍스트 및 제품 정보 준비
**입력**: pgm_info, cand_item_list, extra_item_pdf
**출력**: rag_context, product_element
**주요 작업**:
- RAG 컨텍스트 구성 (프로그램 정보 포함)
- 모드별 제품 정보 준비 (nlp/llm/rag)
- NLP 모드: 제품 요소 직접 생성

### 5. LLMExtractionStep
**목적**: LLM 호출 및 정보 추출
**입력**: msg, rag_context, product_element
**출력**: result_json_text (LLM 응답)
**주요 작업**:
- 프롬프트 구성 (build_extraction_prompt)
- LLM 호출 (safe_llm_invoke)
- 프롬프트 저장 (디버깅용)

### 6. ResponseParsingStep
**목적**: LLM 응답 JSON 파싱
**입력**: result_json_text
**출력**: json_objects, raw_result
**주요 작업**:
- JSON 파싱 (extract_json_objects)
- 스키마 응답 감지 (detect_schema_response)
- raw_result 생성

### 7. EntityContextExtractionStep
**목적**: 엔티티 및 컨텍스트 추출 (Stage 1)
**입력**: msg, entities_from_kiwi, json_objects
**출력**: extracted_entities (entities, context_text, entity_types, relationships)
**주요 작업**:
- LangExtract 또는 Default 엔진으로 엔티티 추출
- logic 모드 시 스킵

### 8. VocabularyFilteringStep
**목적**: 어휘 매칭 및 필터링 (Stage 2)
**입력**: extracted_entities, msg, json_objects
**출력**: matched_products
**주요 작업**:
- Fuzzy/Sequence 유사도 기반 상품 DB 매칭
- LLM 기반 vocabulary 필터링

### 9. ResultConstructionStep
**목적**: 최종 결과 구성
**입력**: matched_products, json_objects, msg, pgm_info
**출력**: final_result
**주요 작업**:
- ResultBuilder.assemble_result() 호출
- 채널 정보 추출 및 보강
- 프로그램 매핑
- offer 객체 생성

### 10. ValidationStep
**목적**: 결과 검증 및 요약
**입력**: final_result
**출력**: validated final_result
**주요 작업**:
- 결과 유효성 검증 (validate_extraction_result)
- 최종 결과 요약 로깅

### 11. DAGExtractionStep (선택적)
**목적**: 엔티티 간 관계 그래프 생성
**입력**: msg, extract_entity_dag 플래그
**출력**: entity_dag (DAG 리스트)
**주요 작업**:
- LLM 기반 DAG 추출 (extract_dag) — 모든 모드에서 fresh LLM call
- NetworkX 그래프 생성
- Graphviz 다이어그램 생성

💡 사용 예시
-----------
```python
from core.workflow_core import WorkflowEngine, WorkflowState
from core.mms_workflow_steps import (
    InputValidationStep,
    EntityExtractionStep,
    # ... 기타 단계들
)

# 워크플로우 엔진 초기화
engine = WorkflowEngine("MMS Extraction")

# 단계 등록
engine.add_step(InputValidationStep())
engine.add_step(EntityExtractionStep(entity_recognizer))
engine.add_step(ProgramClassificationStep(program_classifier))
# ... 기타 단계들

# 초기 상태 설정
state = WorkflowState()
state.set("mms_msg", "샘플 MMS 메시지")
state.set("extractor", extractor_instance)

# 워크플로우 실행
final_state = engine.execute(state)

# 결과 확인
if final_state.has_error():
    errors = final_state.get_errors()
else:
    result = final_state.get("final_result")
```

📝 참고사항
----------
- 각 단계는 독립적으로 테스트 가능
- 에러 발생 시 `state.add_error()`로 기록
- 에러가 있으면 후속 단계는 자동으로 스킵
- DAGExtractionStep은 `extract_entity_dag=True`일 때만 실행
- 모든 단계는 WorkflowState를 통해 데이터 공유

"""

import logging
import copy
import traceback
from typing import Any, Dict, List
import pandas as pd
from .workflow_core import WorkflowStep, WorkflowState
from utils import (
    validate_text_input,
    safe_check_empty,
    extract_json_objects,
    replace_special_chars_with_space
)
from utils import PromptManager, validate_extraction_result, detect_schema_response


logger = logging.getLogger(__name__)


class InputValidationStep(WorkflowStep):
    """
    입력 메시지 검증 단계 (Step 1/9)
    
    책임:
        - 원본 MMS 메시지 유효성 검사
        - 텍스트 정규화 및 전처리
        - 추출기 설정 상태 로깅
    
    데이터 흐름:
        입력: mms_msg (원본 메시지), extractor (추출기 인스턴스)
        출력: msg (전처리된 메시지)
    
    에러 처리:
        - 검증 실패 시 is_fallback=True 설정
        - 에러 메시지를 state에 기록
    """
    
    def execute(self, state: WorkflowState) -> WorkflowState:
        mms_msg = state.get("mms_msg")
        extractor = state.get("extractor")
        
        self._log_message_info(mms_msg)
        self._log_extractor_config(extractor)
        
        try:
            msg = validate_text_input(mms_msg)
            state.set("msg", msg)
        except Exception as e:
            logger.error(f"입력 검증 실패: {e}")
            state.add_error(f"입력 검증 실패: {e}")
            state.set("is_fallback", True)
        
        return state
    
    def _log_message_info(self, mms_msg: str):
        """메시지 정보 로깅"""
        logger.info(f"메시지 내용: {mms_msg[:200]}...")
        logger.info(f"메시지 길이: {len(mms_msg)} 문자")
    
    def _log_extractor_config(self, extractor):
        """추출기 설정 로깅"""
        logger.info("=== 현재 추출기 설정 ===")
        logger.info(f"데이터 소스: {extractor.offer_info_data_src}")
        logger.info(f"상품 정보 추출 모드: {extractor.product_info_extraction_mode}")
        logger.info(f"엔티티 추출 모드: {extractor.entity_extraction_mode}")
        logger.info(f"LLM 모델: {extractor.llm_model_name}")
        logger.info(f"상품 데이터 크기: {extractor.item_pdf_all.shape}")
        logger.info(f"프로그램 데이터 크기: {extractor.pgm_pdf.shape}")


class EntityExtractionStep(WorkflowStep):
    """
    엔티티 추출 단계 (Step 2)

    책임:
        - Kiwi 형태소 분석을 통한 NNP 태그 추출
        - 임베딩 기반 유사도 매칭
        - 후보 상품 목록 생성
        - DB 모드 진단 및 데이터 품질 검증

    협력 객체:
        - EntityRecognizer: 엔티티 추출 및 매칭 수행

    데이터 흐름:
        입력: msg (검증된 메시지)
        출력: entities_from_kiwi (Kiwi 추출 엔티티)
              cand_item_list (후보 상품 리스트)
              extra_item_pdf (매칭된 상품 정보)

    특이사항:
        - skip_entity_extraction=True이면 should_execute() → False로 스킵
        - DB 모드에서는 별칭 데이터 품질 진단 수행
        - 후보 엔티티가 없으면 경고 로그 출력
    """

    def __init__(self, entity_recognizer, skip_entity_extraction: bool = False):
        self.entity_recognizer = entity_recognizer
        self.skip_entity_extraction = skip_entity_extraction

    def should_execute(self, state: WorkflowState) -> bool:
        if self.skip_entity_extraction:
            return False
        return not state.has_error()

    def execute(self, state: WorkflowState) -> WorkflowState:
        if state.has_error():
            return state

        msg = state.get("msg")
        extractor = state.get("extractor")

        # DB 모드 진단
        if extractor.offer_info_data_src == "db":
            self._diagnose_db_mode(extractor)

        # 엔티티 추출
        entities_from_kiwi, cand_item_list, extra_item_pdf = self.entity_recognizer.extract_entities_hybrid(msg)

        self._log_extraction_results(entities_from_kiwi, cand_item_list, extra_item_pdf)

        # DB 모드 결과 분석
        if extractor.offer_info_data_src == "db":
            self._analyze_db_results(cand_item_list)

        # ONT 모드일 경우 LLM 기반 추출 및 메타데이터 저장
        if hasattr(extractor, 'entity_extraction_context_mode') and extractor.entity_extraction_context_mode == 'ont':
            logger.info("🔍 ONT 모드 감지: LLM 기반 엔티티 추출 수행")
            try:
                ont_result = self.entity_recognizer.extract_entities_with_llm(
                    msg_text=msg,
                    rank_limit=50,
                    llm_models=[extractor.llm_model],
                    external_cand_entities=cand_item_list,
                    context_mode='ont'
                )

                # ONT 결과에서 메타데이터 추출 및 저장
                if isinstance(ont_result, dict) and 'ont_metadata' in ont_result:
                    ont_metadata = ont_result.get('ont_metadata')
                    if ont_metadata:
                        state.set("ont_extraction_result", ont_metadata)
                        logger.info(f"✅ ONT 메타데이터 저장: entity_types={len(ont_metadata.get('entity_types', {}))}, "
                                   f"relationships={len(ont_metadata.get('relationships', []))}")
            except Exception as e:
                logger.warning(f"ONT 모드 추출 실패 (무시): {e}")

        state.set("entities_from_kiwi", entities_from_kiwi)
        state.set("cand_item_list", cand_item_list)
        state.set("extra_item_pdf", extra_item_pdf)

        return state
    
    def _diagnose_db_mode(self, extractor):
        """DB 모드 진단"""
        logger.info("🔍 DB 모드 특별 진단 시작")
        logger.info(f"상품 데이터 상태: {extractor.item_pdf_all.shape}")
        
        required_columns = ['item_nm', 'item_id', 'item_nm_alias']
        missing_columns = [col for col in required_columns if col not in extractor.item_pdf_all.columns]
        if missing_columns:
            logger.error(f"🚨 DB 모드에서 필수 컬럼 누락: {missing_columns}")
        
        if 'item_nm_alias' in extractor.item_pdf_all.columns:
            null_aliases = extractor.item_pdf_all['item_nm_alias'].isnull().sum()
            total_aliases = len(extractor.item_pdf_all)
            logger.info(f"DB 모드 별칭 데이터 품질: {total_aliases - null_aliases}/{total_aliases} 유효")
    
    def _log_extraction_results(self, entities_from_kiwi, cand_item_list, extra_item_pdf):
        """추출 결과 로깅"""
        logger.info(f"추출된 Kiwi 엔티티: {entities_from_kiwi}")
        logger.info(f"추출된 후보 엔티티: {cand_item_list}")
        logger.info(f"매칭된 상품 정보: {extra_item_pdf.shape}")
    
    def _analyze_db_results(self, cand_item_list):
        """DB 모드 결과 분석"""
        logger.info("🔍 DB 모드 엔티티 추출 결과 분석")
        if safe_check_empty(cand_item_list):
            logger.error("🚨 DB 모드에서 후보 엔티티가 전혀 추출되지 않았습니다!")
            logger.error("가능한 원인:")
            logger.error("1. 상품 데이터베이스에 해당 상품이 없음")
            logger.error("2. 별칭 규칙 적용 실패")
            logger.error("3. 유사도 임계값이 너무 높음")
            logger.error("4. Kiwi 형태소 분석 실패")


class ProgramClassificationStep(WorkflowStep):
    """
    프로그램 분류 단계 (Step 3/9)
    
    책임:
        - 메시지를 사전 정의된 프로그램 카테고리로 분류
        - 임베딩 기반 유사도 계산
        - 상위 N개 후보 프로그램 선택
    
    협력 객체:
        - ProgramClassifier: 프로그램 분류 수행
    
    데이터 흐름:
        입력: msg (검증된 메시지)
        출력: pgm_info (프로그램 분류 정보)
              - pgm_cand_info: 후보 프로그램 정보
              - pgm_pdf_tmp: 후보 프로그램 DataFrame
    """
    
    def __init__(self, program_classifier):
        self.program_classifier = program_classifier

    def execute(self, state: WorkflowState) -> WorkflowState:
        if state.has_error():
            return state

        msg = state.get("msg")
        extractor = state.get("extractor")

        pgm_info = self.program_classifier.classify(msg)
        logger.info(f"프로그램 분류 결과 키: {list(pgm_info.keys())}")

        # 후보 프로그램 상세 로깅
        if pgm_info.get("pgm_cand_info"):
            logger.info(f"후보 프로그램 목록:\n\t{pgm_info['pgm_cand_info']}")
        else:
            logger.info("후보 프로그램 없음 (빈 결과)")

        # 유사도 상위 N개 로깅
        pgm_pdf_tmp = pgm_info.get("pgm_pdf_tmp")
        if pgm_pdf_tmp is not None and not pgm_pdf_tmp.empty:
            num_cand = self.program_classifier.num_cand_pgms
            top_rows = pgm_pdf_tmp.head(num_cand)
            for _, row in top_rows.iterrows():
                logger.info(f"  pgm: {row.get('pgm_nm', 'N/A')} | sim: {row.get('sim', 0):.4f}")

        state.set("pgm_info", pgm_info)

        return state


class ContextPreparationStep(WorkflowStep):
    """
    RAG 컨텍스트 및 제품 정보 준비 단계 (Step 4/9)
    
    책임:
        - RAG 컨텍스트 구성 (프로그램 분류 정보 포함)
        - 모드별 제품 정보 준비 (nlp/llm/rag)
        - NLP 모드: 제품 요소 직접 생성
    
    데이터 흐름:
        입력: pgm_info, cand_item_list, extra_item_pdf
        출력: rag_context (RAG 컨텍스트 문자열)
              product_element (NLP 모드 제품 요소, 선택적)
    
    모드별 동작:
        - rag: 후보 상품 목록을 RAG 컨텍스트에 추가
        - llm: 참고용 후보 상품 목록 추가
        - nlp: product_element 직접 생성 (name, action)
    """
    
    def execute(self, state: WorkflowState) -> WorkflowState:
        if state.has_error():
            return state
        
        extractor = state.get("extractor")
        pgm_info = state.get("pgm_info")
        cand_item_list = state.get("cand_item_list")
        extra_item_pdf = state.get("extra_item_pdf")
        
        # RAG 컨텍스트 구성
        rag_context = self._build_ad_classification_rag_context(extractor, pgm_info)
        
        # 제품 정보 준비
        product_element = None
        
        if not safe_check_empty(cand_item_list):
            self._log_candidate_items(cand_item_list, extra_item_pdf)
            rag_context, product_element = self._build_product_rag_context(
                extractor, rag_context, cand_item_list, extra_item_pdf
            )
        else:
            self._log_no_candidates()
        
        state.set("rag_context", rag_context)
        state.set("product_element", product_element)
        
        return state
    
    def _build_ad_classification_rag_context(self, extractor, pgm_info) -> str:
        """광고 분류용 RAG 컨텍스트 구성"""
        if extractor.num_cand_pgms > 0 and pgm_info.get('pgm_cand_info'):
            num_select = extractor.num_select_pgms
            rag_context = (
                f"\n### 광고 분류 기준 정보 (pgm 후보 목록) ###\n"
                f"For the pgm field, select up to {num_select} from the following list. Copy the name EXACTLY.\n"
                f"\t{pgm_info['pgm_cand_info']}"
            )
        else:
            rag_context = ""
        logger.info(f"프로그램 분류 컨텍스트 길이: {len(rag_context)} 문자")
        return rag_context
    
    def _log_candidate_items(self, cand_item_list, extra_item_pdf):
        """후보 아이템 로깅"""
        logger.info(f"후보 아이템 리스트 크기: {len(cand_item_list)}개")
        logger.info(f"후보 아이템 리스트: {cand_item_list}")
        logger.info(f"extra_item_pdf 크기: {extra_item_pdf.shape}")
        if not extra_item_pdf.empty:
            logger.info(f"extra_item_pdf 컬럼들: {list(extra_item_pdf.columns)}")
            logger.info(f"extra_item_pdf 샘플: {extra_item_pdf.head(2).to_dict('records')}")
    
    def _build_product_rag_context(self, extractor, rag_context, cand_item_list, extra_item_pdf):
        """제품 정보용 RAG 컨텍스트 구성 (모드별)"""
        product_element = None
        
        if extractor.product_info_extraction_mode == 'rag':
            rag_context += f"\n\n### 후보 상품 이름 목록 ###\n\t{cand_item_list}"
            logger.info("RAG 모드: 후보 상품 목록을 RAG 컨텍스트에 추가")
        elif extractor.product_info_extraction_mode == 'llm':
            rag_context += f"\n\n### 참고용 후보 상품 이름 목록 ###\n\t{cand_item_list}"
            logger.info("LLM 모드: 참고용 후보 상품 목록을 RAG 컨텍스트에 추가")
        elif extractor.product_info_extraction_mode == 'nlp':
            product_element = self._build_nlp_product_element(extractor, extra_item_pdf)
        
        return rag_context, product_element
    
    def _build_nlp_product_element(self, extractor, extra_item_pdf):
        """NLP 모드 제품 요소 구성"""
        if not extra_item_pdf.empty and 'item_nm' in extra_item_pdf.columns:
            product_df = extra_item_pdf.rename(columns={'item_nm': 'name'}).query(
                "not name in @extractor.stop_item_names"
            )[['name']]
            product_df['action'] = '기타'
            product_element = product_df.to_dict(orient='records') if product_df.shape[0] > 0 else None
            logger.info(f"NLP 모드: 제품 요소 준비 완료 - {len(product_element) if product_element else 0}개")
            if product_element:
                logger.info(f"NLP 모드 제품 요소 샘플: {product_element[:2]}")
            return product_element
        else:
            logger.warning("NLP 모드: extra_item_pdf가 비어있거나 item_nm 컬럼이 없습니다!")
            return None
    
    def _log_no_candidates(self):
        """후보 아이템 없음 경고"""
        logger.warning("후보 아이템이 없습니다!")
        logger.warning("이는 다음 중 하나의 문제일 수 있습니다:")
        logger.warning("1. 상품 데이터 로딩 실패")
        logger.warning("2. 엔티티 추출 실패")
        logger.warning("3. 유사도 매칭 임계값 문제")


class LLMExtractionStep(WorkflowStep):
    """
    LLM 호출 및 추출 단계 (Step 5/9)
    
    책임:
        - 최종 프롬프트 구성 (메시지 + RAG 컨텍스트 + 제품 정보)
        - LLM 호출 및 응답 수신
        - 프롬프트 저장 (디버깅 및 검토용)
    
    데이터 흐름:
        입력: msg, rag_context, product_element
        출력: result_json_text (LLM JSON 응답)
    
    주요 작업:
        1. build_extraction_prompt()로 프롬프트 생성
        2. PromptManager.store_prompt_for_preview()로 저장
        3. safe_llm_invoke()로 LLM 호출
    
    특이사항:
        - 프롬프트는 스레드 로컬 저장소에 캐시됨
        - LLM 호출 실패 시 재시도 메커니즘 적용
    """
    
    def execute(self, state: WorkflowState) -> WorkflowState:
        if state.has_error():
            return state
        
        msg = state.get("msg")
        extractor = state.get("extractor")
        rag_context = state.get("rag_context")
        product_element = state.get("product_element")
        
        # 프롬프트 구성
        prompt = extractor._build_extraction_prompt(msg, rag_context, product_element)
        logger.info(f"구성된 프롬프트 길이: {len(prompt)} 문자")
        logger.info(f"RAG 컨텍스트 포함 여부: {'후보 상품' in rag_context}")
        
        # 프롬프트 저장 (helpers 모듈 사용)
        PromptManager.store_prompt_for_preview(prompt, "main_extraction")
        
        # LLM 호출
        result_json_text = extractor._safe_llm_invoke(prompt)
        logger.info(f"LLM 응답 길이: {len(result_json_text)} 문자")
        logger.info(f"LLM 응답 내용 (처음 500자): {result_json_text[:500]}...")
        
        state.set("result_json_text", result_json_text)
        
        return state


class ResponseParsingStep(WorkflowStep):
    """
    LLM 응답 JSON 파싱 단계 (Step 6/9)
    
    책임:
        - LLM 응답에서 JSON 객체 추출
        - 스키마 응답 감지 및 필터링
        - raw_result 생성
    
    데이터 흐름:
        입력: result_json_text (LLM 응답)
        출력: json_objects (파싱된 JSON)
              raw_result (원본 결과, message_id 포함)
    
    에러 처리:
        - JSON 파싱 실패: is_fallback=True 설정
        - 스키마 응답 감지: is_fallback=True 설정
    
    특이사항:
        - 여러 JSON 객체가 있으면 마지막 것 사용
        - detect_schema_response()로 스키마 정의 필터링
    """
    
    def execute(self, state: WorkflowState) -> WorkflowState:
        if state.has_error():
            return state
        
        result_json_text = state.get("result_json_text")
        # extractor = state.get("extractor") # No longer needed for detect_schema_response
        # msg = state.get("msg") # No longer needed
        
        # JSON 파싱
        json_objects_list = extract_json_objects(result_json_text)
        logger.info(f"추출된 JSON 객체 수: {len(json_objects_list)}개")
        
        if not json_objects_list:
            logger.warning("LLM이 유효한 JSON 객체를 반환하지 않았습니다")
            logger.warning(f"LLM 원본 응답: {result_json_text}")
            state.add_error("JSON 파싱 실패")
            state.set("is_fallback", True)
            return state
        
        json_objects = json_objects_list[-1]
        logger.info(f"파싱된 JSON 객체 키: {list(json_objects.keys())}")
        logger.info(f"파싱된 JSON 내용: {json_objects}")
        
        # 스키마 응답 감지 (helpers 모듈 사용)
        if detect_schema_response(json_objects):
            logger.error("🚨 LLM이 스키마 정의를 반환했습니다! 실제 데이터가 아닙니다.")
            logger.error("재시도 또는 fallback 결과를 사용합니다.")
            state.add_error("스키마 응답 감지")
            state.set("is_fallback", True)
            return state
        
        raw_result = copy.deepcopy(json_objects)
        
        # message_id 추가
        message_id = state.get("message_id", "#")
        raw_result['message_id'] = message_id
        
        state.set("json_objects", json_objects)
        state.set("raw_result", raw_result)
        
        return state


class EntityContextExtractionStep(WorkflowStep):
    """
    엔티티 + 컨텍스트 추출 단계 (Step 7)

    책임:
        - 메시지에서 엔티티와 컨텍스트 추출 (Stage 1)
        - 두 가지 추출 방식 지원:
          1. langextract: Google langextract 기반 6-type 분류
          2. default: entity_recognizer._extract_entities_stage1() 호출
        - 추출 결과를 state.extracted_entities에 저장

    데이터 흐름:
        입력: msg, entities_from_kiwi, json_objects
        출력: extracted_entities (entities, context_text, entity_types, relationships)
    """

    def __init__(self, entity_recognizer,
                 llm_factory=None, llm_model: str = 'ax',
                 entity_extraction_context_mode: str = 'dag',
                 use_external_candidates: bool = False,
                 extraction_engine: str = 'default',
                 stop_item_names: List[str] = None,
                 entity_extraction_mode: str = 'llm'):
        self.entity_recognizer = entity_recognizer
        self.llm_factory = llm_factory
        self.llm_model = llm_model
        self.entity_extraction_context_mode = entity_extraction_context_mode
        self.use_external_candidates = use_external_candidates
        self.extraction_engine = extraction_engine
        self.stop_item_names = stop_item_names or []
        self.entity_extraction_mode = entity_extraction_mode

    def should_execute(self, state: WorkflowState) -> bool:
        """Skip if there's an error or in logic mode (logic mode doesn't use Stage 1 output)"""
        if state.has_error():
            return False
        # Skip in logic mode - VocabularyFilteringStep doesn't use extracted_entities in logic mode
        if self.entity_extraction_mode == 'logic':
            return False
        return True

    def execute(self, state: WorkflowState) -> WorkflowState:
        msg = state.msg
        entities_from_kiwi = state.entities_from_kiwi
        json_objects = state.json_objects

        # Extract product items from json_objects
        product_items = json_objects.get('product', [])
        if isinstance(product_items, dict):
            product_items = product_items.get('items', [])

        primary_llm_extracted_entities = [x.get('name', '') for x in product_items]
        logger.debug(f"LLM 추출 엔티티: {primary_llm_extracted_entities}")
        logger.debug(f"Kiwi 엔티티: {entities_from_kiwi}")

        # Build external candidate list
        if self.use_external_candidates:
            external_cand = list(set(entities_from_kiwi + primary_llm_extracted_entities))
        else:
            external_cand = []
            logger.info("외부 후보 엔티티 비활성화 (use_external_candidates=False)")

        # Stage 1: Entity + Context Extraction
        if self.extraction_engine == 'langextract':
            # Method A: LangExtract-based extraction
            try:
                from core.lx_extractor import extract_mms_entities
                logger.info("🔗 langextract 엔진으로 Stage 1 엔티티 추출 시작...")
                doc = extract_mms_entities(msg, model_id=self.llm_model)
                entities = []
                type_pairs = []
                for ext in (doc.extractions or []):
                    name = ext.extraction_text
                    if ext.extraction_class in ('Channel', 'Purpose'):
                        continue
                    if name not in self.stop_item_names and len(name) >= 2:
                        entities.append(name)
                        type_pairs.append(f"{name}({ext.extraction_class})")

                logger.info(f"✅ langextract Stage 1 완료: {len(entities)}개 엔티티 추출")
                logger.info(f"   엔티티: {entities}")

                # Post-extraction role classification
                entity_roles = {}
                if entities and self.llm_factory:
                    llm_models = self.llm_factory.create_models([self.llm_model])
                    if llm_models:
                        entity_roles = self.entity_recognizer.classify_entity_roles(
                            msg, entities, llm_models[0]
                        )
                        if entity_roles:
                            logger.info(f"📋 LANGEXTRACT 역할 분류: {entity_roles}")
                            # Filter out prerequisite/context entities
                            prereq_removed = [e for e in entities if entity_roles.get(e) in ('prerequisite', 'context')]
                            entities = [e for e in entities if entity_roles.get(e) not in ('prerequisite', 'context')]
                            if prereq_removed:
                                logger.info(f"🔽 Role filter removed {len(prereq_removed)}: {prereq_removed}")
                            # Update type_pairs to include roles
                            type_pairs = [f"{tp}:{entity_roles.get(tp.split('(')[0], '?')})"
                                         if tp.endswith(')') else tp for tp in type_pairs]

                state.extracted_entities = {
                    'entities': entities,
                    'context_text': ", ".join(type_pairs),
                    'entity_types': {},
                    'relationships': []
                }
                if entity_roles:
                    state.extracted_entities['entity_roles'] = entity_roles
                    state.kg_metadata = {'entity_roles': entity_roles}
                    prereqs = [e for e, r in entity_roles.items() if r == 'prerequisite']
                    offers = [e for e, r in entity_roles.items() if r in ('offer', 'benefit')]
                    logger.info(f"📋 LANGEXTRACT 역할 분류: prerequisite={prereqs}, offer/benefit={offers}")

                logger.info(f"   최종 엔티티: {entities}")
                logger.info(f"   컨텍스트: {state.extracted_entities['context_text']}")
            except Exception as e:
                logger.error(f"❌ langextract 추출 실패, 기본 모드로 폴백: {e}")
                state.extracted_entities = None
        else:
            # Method B: Standard LLM-based extraction
            if self.llm_factory:
                llm_models = self.llm_factory.create_models([self.llm_model])
            else:
                logger.warning("llm_factory가 설정되지 않았습니다. 빈 리스트를 사용합니다.")
                llm_models = []

            try:
                logger.info(f"🔍 entity_recognizer로 Stage 1 추출 시작 (context_mode={self.entity_extraction_context_mode})...")
                stage1_result = self.entity_recognizer._extract_entities_stage1(
                    msg_text=msg,
                    context_mode=self.entity_extraction_context_mode,
                    llm_models=llm_models,
                    external_cand_entities=external_cand
                )
                state.extracted_entities = stage1_result
                logger.info(f"✅ Stage 1 완료: {len(stage1_result.get('entities', []))}개 엔티티 추출")
                logger.info(f"   엔티티: {stage1_result.get('entities', [])}")

                # KG/DAG 모드: kg_metadata를 state에 저장 (역할 분류 기반 필터링에 사용)
                if stage1_result.get('kg_metadata'):
                    state.kg_metadata = stage1_result['kg_metadata']
                    entity_roles = stage1_result.get('entity_roles', {})
                    prereqs = [e for e, r in entity_roles.items() if r == 'prerequisite']
                    offers = [e for e, r in entity_roles.items() if r in ('offer', 'benefit')]
                    mode_label = self.entity_extraction_context_mode.upper()
                    logger.info(f"📋 {mode_label} 역할 분류: prerequisite={prereqs}, offer/benefit={offers}")

            except Exception as e:
                logger.error(f"❌ Stage 1 추출 실패: {e}")
                state.extracted_entities = None

        return state


class VocabularyFilteringStep(WorkflowStep):
    """
    어휘 기반 필터링 단계 (Step 8)

    책임:
        - Stage 1에서 추출한 엔티티를 상품 DB와 매칭
        - logic 모드: fuzzy + sequence 유사도 매칭
        - llm 모드: LLM 기반 어휘 필터링
        - alias 타입 필터링 (non-expansion)
        - 매칭 결과를 state.matched_products에 저장

    데이터 흐름:
        입력: extracted_entities, msg, json_objects
        출력: matched_products
    """

    def __init__(self, entity_recognizer, alias_pdf_raw: pd.DataFrame,
                 stop_item_names: List[str], entity_extraction_mode: str,
                 llm_factory=None, llm_model: str = 'ax',
                 entity_extraction_context_mode: str = 'dag'):
        self.entity_recognizer = entity_recognizer
        self.alias_pdf_raw = alias_pdf_raw
        self.stop_item_names = stop_item_names
        self.entity_extraction_mode = entity_extraction_mode
        self.llm_factory = llm_factory
        self.llm_model = llm_model
        self.entity_extraction_context_mode = entity_extraction_context_mode

    def should_execute(self, state: WorkflowState) -> bool:
        """Skip if error, fallback, or no entities"""
        if state.has_error():
            return False
        if state.is_fallback:
            return False

        # Check if we have extracted entities from Stage 1
        extracted_entities = state.extracted_entities
        if extracted_entities and len(extracted_entities.get('entities', [])) > 0:
            return True

        # Fallback: check if we have product items or kiwi entities
        json_objects = state.json_objects
        product_items = json_objects.get('product', [])
        if isinstance(product_items, dict):
            product_items = product_items.get('items', [])
        has_entities = len(product_items) > 0 or len(state.entities_from_kiwi) > 0
        return has_entities

    def execute(self, state: WorkflowState) -> WorkflowState:
        msg = state.msg
        json_objects = state.json_objects
        extracted_entities = state.extracted_entities

        # Get product items for fallback logic
        product_items = json_objects.get('product', [])
        if isinstance(product_items, dict):
            product_items = product_items.get('items', [])

        # Stage 2: Vocabulary Filtering
        if self.entity_extraction_mode == 'logic':
            # Logic mode: fuzzy matching
            entities_from_kiwi = state.entities_from_kiwi
            cand_entities = list(set(
                entities_from_kiwi + [item.get('name', '') for item in product_items if item.get('name')]
            ))
            logger.debug(f"로직 모드 cand_entities: {cand_entities}")
            similarities_fuzzy = self.entity_recognizer.extract_entities_with_fuzzy_matching(cand_entities)
        else:
            # LLM mode: vocabulary filtering
            if self.llm_factory:
                llm_models = self.llm_factory.create_models([self.llm_model])
            else:
                logger.warning("llm_factory가 설정되지 않았습니다. 빈 리스트를 사용합니다.")
                llm_models = []

            if extracted_entities:
                # Use extracted entities from Stage 1
                entities = extracted_entities.get('entities', [])
                context_text = extracted_entities.get('context_text', '')
                logger.info(f"🔍 Stage 2 시작: {len(entities)}개 엔티티 필터링 (context_mode={self.entity_extraction_context_mode})")

                stage2_result = self.entity_recognizer._filter_with_vocabulary(
                    entities=entities,
                    context_text=context_text,
                    context_mode=self.entity_extraction_context_mode,
                    msg_text=msg,
                    rank_limit=100,
                    llm_model=llm_models[0] if llm_models else None
                )

                # ONT 모드는 dict를 반환, 그 외는 DataFrame
                if isinstance(stage2_result, dict):
                    similarities_fuzzy = stage2_result.get('similarities_df', pd.DataFrame())
                else:
                    similarities_fuzzy = stage2_result
                logger.info(f"✅ Stage 2 완료: {similarities_fuzzy.shape[0] if not similarities_fuzzy.empty else 0}개 엔티티 필터링됨")
            else:
                # Fallback: no extracted entities, use wrapper
                logger.warning("extracted_entities가 없습니다. wrapper를 사용합니다.")
                llm_result = self.entity_recognizer.extract_entities_with_llm(
                    msg,
                    llm_models=llm_models,
                    rank_limit=100,
                    external_cand_entities=[],
                    context_mode=self.entity_extraction_context_mode,
                    pre_extracted=None,
                )

                if isinstance(llm_result, dict):
                    similarities_fuzzy = llm_result.get('similarities_df', pd.DataFrame())
                else:
                    similarities_fuzzy = llm_result

        logger.info(f"similarities_fuzzy 크기: {similarities_fuzzy.shape if not similarities_fuzzy.empty else '비어있음'}")

        # Alias type filtering
        if not similarities_fuzzy.empty:
            merged_df = similarities_fuzzy.merge(
                self.alias_pdf_raw[['alias_1', 'type']].drop_duplicates(),
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
            logger.debug(f"alias 필터링 후 크기: {filtered_df.shape}")

        # Map products to entities
        if not similarities_fuzzy.empty:
            matched_products = self.entity_recognizer.map_products_to_entities(similarities_fuzzy, json_objects)
            logger.info(f"매칭된 상품 수: {len(matched_products)}개")
        else:
            # Fallback: use LLM results directly with item_id='#'
            filtered_product_items = [
                d for d in product_items
                if d.get('name') and d['name'] not in self.stop_item_names
            ]
            matched_products = [
                {
                    'item_nm': d.get('name', ''),
                    'item_id': ['#'],
                    'item_name_in_msg': [d.get('name', '')],
                    'expected_action': [d.get('action', '기타')]
                }
                for d in filtered_product_items
            ]
            logger.info(f"폴백 상품 수 (item_id=#): {len(matched_products)}개")

        # Role-based filter: remove prerequisite/context entities from matched_products (KG and DAG modes)
        if hasattr(state, 'kg_metadata') and state.kg_metadata:
            entity_roles = state.kg_metadata.get('entity_roles', {})
            if entity_roles:
                excluded = []
                filtered = []
                for p in matched_products:
                    # Check item_nm AND item_name_in_msg against KG roles
                    # Use both exact match and substring containment
                    check_names = p.get('item_name_in_msg', []) + [p.get('item_nm', '')]
                    role = None
                    for name in check_names:
                        if not name:
                            continue
                        # Exact match
                        if name in entity_roles:
                            role = entity_roles[name]
                            break
                        # Substring: KG entity name contained in product name or vice versa
                        for kg_entity, kg_role in entity_roles.items():
                            if kg_entity in name or name in kg_entity:
                                role = kg_role
                                break
                        if role:
                            break
                    if role in ('prerequisite', 'context'):
                        excluded.append(f"{p.get('item_nm', '')}({role})")
                    else:
                        filtered.append(p)
                if excluded:
                    matched_products = filtered
                    logger.info(f"Role filter removed {len(excluded)} from matched_products: {excluded}")

                # Offer passthrough: add offer-role entities not in matched_products
                existing_names = set()
                for p in matched_products:
                    existing_names.update(p.get('item_name_in_msg', []))
                    if p.get('item_nm'):
                        existing_names.add(p['item_nm'])

                added = []
                # Build action map from LLM product_items
                action_map = {item.get('name', ''): item.get('action', '기타') for item in product_items}

                for kg_entity, kg_role in entity_roles.items():
                    if kg_role != 'offer':
                        continue
                    if kg_entity in self.stop_item_names:
                        continue
                    # Skip if already in matched_products (exact or substring)
                    already_matched = any(
                        kg_entity in ex or ex in kg_entity for ex in existing_names
                    )
                    if already_matched:
                        continue
                    # Find action from LLM product_items (exact or substring match)
                    action = '기타'
                    for pname, paction in action_map.items():
                        if pname == kg_entity or pname in kg_entity or kg_entity in pname:
                            action = paction
                            break
                    matched_products.append({
                        'item_nm': kg_entity,
                        'item_id': ['#'],
                        'item_name_in_msg': [kg_entity],
                        'expected_action': [action]
                    })
                    added.append(kg_entity)

                if added:
                    logger.info(f"Offer passthrough: added {len(added)} unmatched offer products: {added}")

        state.matched_products = matched_products
        return state


class ResultConstructionStep(WorkflowStep):
    """
    최종 결과 구성 단계 (Step 9)

    책임:
        - matched_products를 final_result에 반영
        - offer 객체 생성 (product/org 타입)
        - 채널 정보 추출 및 매장 정보 매칭
        - 프로그램 분류 매핑
        - entity_dag 초기화, message_id 추가

    협력 객체:
        - ResultBuilder: 결과 조립 로직 수행

    데이터 흐름:
        입력: json_objects, matched_products, msg, pgm_info, message_id
        출력: final_result (최종 추출 결과)
    """

    def __init__(self, result_builder):
        self.result_builder = result_builder

    def execute(self, state: WorkflowState) -> WorkflowState:
        if state.has_error():
            return state

        json_objects = state.json_objects
        msg = state.msg
        pgm_info = state.pgm_info
        matched_products = state.matched_products
        message_id = state.message_id

        final_result = self.result_builder.assemble_result(
            json_objects, matched_products, msg, pgm_info, message_id
        )

        state.final_result = final_result
        return state


class ValidationStep(WorkflowStep):
    """
    결과 검증 단계 (Step 10)

    책임:
        - 최종 결과 유효성 검증
        - 필수 필드 존재 여부 확인
        - 결과 요약 로깅
    
    데이터 흐름:
        입력: final_result
        출력: validated final_result
    
    검증 항목:
        - 필수 필드: title, purpose, product, channel, pgm, offer
        - 데이터 타입 검증
        - 빈 값 처리
    
    로깅 정보:
        - 제목, 목적, 판매 스크립트
        - 상품/채널/프로그램 개수
        - offer 타입 및 항목 수
    """
    
    def execute(self, state: WorkflowState) -> WorkflowState:
        if state.has_error():
            return state
        
        final_result = state.get("final_result")
        # extractor = state.get("extractor") # No longer needed for validate_extraction_result
        
        # 결과 검증 (helpers 모듈 사용)
        validated_result = validate_extraction_result(final_result)
        
        # 최종 결과 요약 로깅
        self._log_final_summary(validated_result)
        
        state.set("final_result", validated_result)
        
        return state
    
    def _log_final_summary(self, result: Dict[str, Any]):
        """최종 결과 요약 로깅"""
        logger.info("=== 최종 결과 요약 ===")
        logger.info(f"제목: {result.get('title', 'N/A')}")
        logger.info(f"목적: {result.get('purpose', [])}")
        
        sales_script = result.get('sales_script', '')
        if sales_script:
            preview = sales_script[:100] + "..." if len(sales_script) > 100 else sales_script
            logger.info(f"판매 스크립트: {preview}")
        
        logger.info(f"상품 수: {len(result.get('product', []))}개")
        logger.info(f"채널 수: {len(result.get('channel', []))}개")
        logger.info(f"프로그램 수: {len(result.get('pgm', []))}개")
        
        offer_info = result.get('offer', {})
        logger.info(f"오퍼 타입: {offer_info.get('type', 'N/A')}")
        logger.info(f"오퍼 항목 수: {len(offer_info.get('value', []))}개")


class DAGExtractionStep(WorkflowStep):
    """
    DAG 추출 단계 (Step 11, 선택적)

    책임:
        - LLM 기반 엔티티 간 관계 분석
        - DAG(Directed Acyclic Graph) 생성
        - NetworkX 그래프 구조 생성
        - Graphviz 다이어그램 이미지 생성
    
    협력 객체:
        - DAGParser: DAG 텍스트 파싱
        - extract_dag: LLM 기반 DAG 추출
    
    데이터 흐름:
        입력: msg, extract_entity_dag 플래그, message_id
        출력: entity_dag (DAG 엣지 리스트)
    
    출력 형식:
        entity_dag: [
            "(상품A:구매) -[획득]-> (혜택B:제공)",
            "(이벤트C:참여) -[응모]-> (혜택B:제공)"
        ]
    
    특이사항:
        - extract_entity_dag=False이면 빈 배열 반환
        - DAG 다이어그램은 ./dag_images/ 디렉토리에 저장
        - 실패 시에도 빈 배열로 처리 (에러 전파 안 함)
    """
    
    def __init__(self, dag_parser=None):
        """
        Args:
            dag_parser: DAGParser 인스턴스 (선택사항, None이면 자동 생성)
        """
        from .entity_dag_extractor import DAGParser
        self.dag_parser = dag_parser or DAGParser()
    
    def execute(self, state: WorkflowState) -> WorkflowState:
        """
        DAG 추출 실행

        Args:
            state: 현재 워크플로우 상태

        Returns:
            업데이트된 워크플로우 상태 (entity_dag 필드 추가)
        """
        # extract_entity_dag 플래그 확인
        extractor = state.get("extractor")
        if not extractor.extract_entity_dag:
            logger.info("DAG 추출이 비활성화되어 있습니다")
            # 비활성화된 경우 빈 배열로 설정
            final_result = state.get("final_result", {})
            final_result['entity_dag'] = []
            state.set("final_result", final_result)

            raw_result = state.get("raw_result", {})
            raw_result['entity_dag'] = []
            state.set("raw_result", raw_result)
            return state

        msg = state.get("msg")
        message_id = state.get("message_id", "#")

        # KG 모드: kg_metadata가 있으면 LLM 호출 없이 KG→DAG 변환
        kg_metadata = state.kg_metadata if hasattr(state, 'kg_metadata') else None

        if kg_metadata and (kg_metadata.get('relationships') or kg_metadata.get('dag_text')):
            logger.info("🔗 KG → DAG 변환 시작 (LLM 호출 없음)...")
            try:
                dag_result = self._convert_kg_to_dag(kg_metadata, msg)
            except Exception as e:
                logger.warning(f"KG→DAG 변환 실패, fresh LLM call로 폴백: {e}")
                kg_metadata = None  # fall through to LLM path

        if not kg_metadata or not (kg_metadata.get('relationships') or kg_metadata.get('dag_text')):
            logger.info("🔗 DAG 추출 시작 (fresh LLM call)...")

        try:
            if not kg_metadata or not (kg_metadata.get('relationships') or kg_metadata.get('dag_text')):
                from .entity_dag_extractor import extract_dag

                dag_result = extract_dag(
                    self.dag_parser,
                    msg,
                    extractor.llm_model,
                    prompt_mode='cot'
                )

            # DAG 섹션을 리스트로 변환 (빈 줄 제거 및 정렬)
            dag_list = sorted([
                d.strip() for d in dag_result['dag_section'].split('\n')
                if d.strip()
            ])

            logger.info(f"✅ DAG 추출 완료: {len(dag_list)}개 엣지")

            # final_result에 entity_dag 추가
            final_result = state.get("final_result", {})
            final_result['entity_dag'] = dag_list
            state.set("final_result", final_result)

            # raw_result에도 추가
            raw_result = state.get("raw_result", {})
            raw_result['entity_dag'] = dag_list
            state.set("raw_result", raw_result)

            # DAG 다이어그램 생성 (선택적)
            if dag_result['dag'].number_of_nodes() > 0:
                try:
                    from utils import create_dag_diagram, sha256_hash
                    dag_filename = f'dag_{message_id}_{sha256_hash(msg)}'
                    create_dag_diagram(dag_result['dag'], filename=dag_filename)
                    logger.info(f"📊 DAG 다이어그램 저장: {dag_filename}.png")
                except Exception as e:
                    logger.warning(f"DAG 다이어그램 생성 실패 (무시): {e}")

        except Exception as e:
            logger.error(f"❌ DAG 추출 실패: {e}")
            logger.error(f"상세 오류: {traceback.format_exc()}")

            # 실패 시 빈 배열로 설정
            final_result = state.get("final_result", {})
            final_result['entity_dag'] = []
            state.set("final_result", final_result)

            raw_result = state.get("raw_result", {})
            raw_result['entity_dag'] = []
            state.set("raw_result", raw_result)

        return state

    def _convert_kg_to_dag(self, kg_metadata: dict, msg: str) -> dict:
        """
        KG 메타데이터에서 DAG 변환 (LLM 호출 없음).

        Args:
            kg_metadata: Step 7에서 생성된 KG 메타데이터
                {
                    'dag_text': str,
                    'entity_types': dict,
                    'entity_roles': dict,
                    'relationships': list
                }
            msg: 원본 메시지

        Returns:
            dict: extract_dag()와 동일한 형식
                {
                    'dag_section': str,
                    'dag': nx.DiGraph,
                    'dag_raw': str
                }
        """
        import networkx as nx

        dag_text = kg_metadata.get('dag_text', '')

        # 방법 1: dag_text가 있으면 직접 파싱
        if dag_text:
            dag_graph = self.dag_parser.parse_dag(dag_text)
            logger.info(f"✅ KG→DAG 변환 완료 (dag_text 파싱): {dag_graph.number_of_nodes()} 노드, {dag_graph.number_of_edges()} 엣지")
            return {
                'dag_section': dag_text,
                'dag': dag_graph,
                'dag_raw': dag_text
            }

        # 방법 2: relationships에서 그래프 생성
        from .entity_dag_extractor import build_dag_from_ontology
        dag_graph = build_dag_from_ontology(kg_metadata)

        # graph → dag_text 변환
        dag_lines = []
        for u, v, data in dag_graph.edges(data=True):
            relation = data.get('relation', '')
            dag_lines.append(f"({u}) -[{relation}]-> ({v})")

        dag_section = '\n'.join(dag_lines)
        logger.info(f"✅ KG→DAG 변환 완료 (relationships 변환): {dag_graph.number_of_nodes()} 노드, {dag_graph.number_of_edges()} 엣지")

        return {
            'dag_section': dag_section,
            'dag': dag_graph,
            'dag_raw': dag_section
        }

    def _execute_from_ont(self, state: WorkflowState, ont_result: dict, msg: str, message_id: str) -> WorkflowState:
        """
        ⚠️ DEPRECATED: No longer called since commit a4e1ef0.
        DAGExtractionStep now always makes fresh LLM call for consistent behavior.
        This method is kept for potential future use or rollback.

        ONT 결과에서 DAG 생성 (LLM 호출 없음)

        Args:
            state: 워크플로우 상태
            ont_result: ONT 모드에서 추출된 메타데이터
            msg: 원본 메시지
            message_id: 메시지 ID

        Returns:
            업데이트된 워크플로우 상태
        """
        from .entity_dag_extractor import build_dag_from_ontology

        try:
            # 1. relationships에서 DAG 리스트 생성
            # 형식: (entity value:entity type) -[relationship]-> (entity value:entity type)
            entity_types = ont_result.get('entity_types', {})
            relationships = ont_result.get('relationships', [])
            dag_lines = []

            for rel in relationships:
                if not isinstance(rel, dict):
                    continue
                src = rel.get('source', '')
                tgt = rel.get('target', '')
                rel_type = rel.get('type', '')

                if src and tgt and rel_type:
                    src_type = entity_types.get(src, 'Unknown')
                    tgt_type = entity_types.get(tgt, 'Unknown')
                    # DAG 모드와 동일한 형식: (entity:type) -[relation]-> (entity:type)
                    dag_line = f"({src}:{src_type}) -[{rel_type}]-> ({tgt}:{tgt_type})"
                    dag_lines.append(dag_line)

            dag_list = sorted([d for d in dag_lines if d])

            # 2. NetworkX 그래프 생성
            dag = build_dag_from_ontology(ont_result)

            logger.info(f"✅ ONT 기반 DAG 생성 완료: {len(dag_list)}개 엣지, {dag.number_of_nodes()} 노드")

            # 3. 결과 저장
            final_result = state.get("final_result", {})
            final_result['entity_dag'] = dag_list
            state.set("final_result", final_result)

            raw_result = state.get("raw_result", {})
            raw_result['entity_dag'] = dag_list
            state.set("raw_result", raw_result)

            # 4. 이미지 생성
            if dag.number_of_nodes() > 0:
                try:
                    from utils import create_dag_diagram, sha256_hash
                    dag_filename = f'dag_{message_id}_{sha256_hash(msg)}'
                    create_dag_diagram(dag, filename=dag_filename)
                    logger.info(f"📊 DAG 다이어그램 저장 (ONT): {dag_filename}.png")
                except Exception as e:
                    logger.warning(f"DAG 다이어그램 생성 실패 (무시): {e}")

        except Exception as e:
            logger.error(f"❌ ONT 기반 DAG 생성 실패: {e}")
            logger.error(f"상세 오류: {traceback.format_exc()}")

            # 실패 시 빈 배열로 설정
            final_result = state.get("final_result", {})
            final_result['entity_dag'] = []
            state.set("final_result", final_result)

            raw_result = state.get("raw_result", {})
            raw_result['entity_dag'] = []
            state.set("raw_result", raw_result)

        return state

