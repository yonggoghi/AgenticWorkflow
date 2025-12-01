"""
MMS Workflow Steps - MMS 추출기 워크플로우 단계 구현
===================================================

이 모듈은 MMS 메시지 처리의 각 단계를 독립적인 클래스로 구현합니다.
각 단계는 WorkflowStep을 상속받아 execute 메서드를 구현합니다.
"""

import logging
import copy
from typing import Any, Dict
from .workflow_core import WorkflowStep, WorkflowState
from utils import (
    validate_text_input,
    safe_check_empty,
    extract_json_objects
)
from utils import PromptManager, validate_extraction_result, detect_schema_response


logger = logging.getLogger(__name__)


class InputValidationStep(WorkflowStep):
    """
    입력 메시지 검증 단계
    
    - 메시지 유효성 검사
    - 추출기 설정 상태 로깅
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
    엔티티 추출 단계
    
    - Kiwi 형태소 분석
    - 임베딩 유사도 매칭
    - DB 모드 진단
    """
    
    def __init__(self, entity_recognizer):
        self.entity_recognizer = entity_recognizer

    def execute(self, state: WorkflowState) -> WorkflowState:
        if state.has_error():
            return state
        
        msg = state.get("msg")
        extractor = state.get("extractor")
        
        # DB 모드 진단
        if extractor.offer_info_data_src == "db":
            self._diagnose_db_mode(extractor)
        
        # 엔티티 추출
        entities_from_kiwi, cand_item_list, extra_item_pdf = self.entity_recognizer.extract_entities_from_kiwi(msg)
        
        self._log_extraction_results(entities_from_kiwi, cand_item_list, extra_item_pdf)
        
        # DB 모드 결과 분석
        if extractor.offer_info_data_src == "db":
            self._analyze_db_results(cand_item_list)
        
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
    프로그램 분류 단계
    
    - 메시지를 프로그램 카테고리로 분류
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
        
        state.set("pgm_info", pgm_info)
        
        return state


class ContextPreparationStep(WorkflowStep):
    """
    RAG 컨텍스트 및 제품 정보 준비 단계
    
    - RAG 컨텍스트 구성
    - 제품 정보 준비 (모드별: nlp/llm/rag)
    """
    
    def execute(self, state: WorkflowState) -> WorkflowState:
        if state.has_error():
            return state
        
        extractor = state.get("extractor")
        pgm_info = state.get("pgm_info")
        cand_item_list = state.get("cand_item_list")
        extra_item_pdf = state.get("extra_item_pdf")
        
        # RAG 컨텍스트 구성
        rag_context = self._build_rag_context(extractor, pgm_info)
        
        # 제품 정보 준비
        product_element = None
        
        if not safe_check_empty(cand_item_list):
            self._log_candidate_items(cand_item_list, extra_item_pdf)
            rag_context, product_element = self._prepare_product_info(
                extractor, rag_context, cand_item_list, extra_item_pdf
            )
        else:
            self._log_no_candidates()
        
        state.set("rag_context", rag_context)
        state.set("product_element", product_element)
        
        return state
    
    def _build_rag_context(self, extractor, pgm_info) -> str:
        """RAG 컨텍스트 구성"""
        rag_context = f"\n### 광고 분류 기준 정보 ###\n\t{pgm_info['pgm_cand_info']}" if extractor.num_cand_pgms > 0 else ""
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
    
    def _prepare_product_info(self, extractor, rag_context, cand_item_list, extra_item_pdf):
        """제품 정보 준비 (모드별)"""
        product_element = None
        
        if extractor.product_info_extraction_mode == 'rag':
            rag_context += f"\n\n### 후보 상품 이름 목록 ###\n\t{cand_item_list}"
            logger.info("RAG 모드: 후보 상품 목록을 RAG 컨텍스트에 추가")
        elif extractor.product_info_extraction_mode == 'llm':
            rag_context += f"\n\n### 참고용 후보 상품 이름 목록 ###\n\t{cand_item_list}"
            logger.info("LLM 모드: 참고용 후보 상품 목록을 RAG 컨텍스트에 추가")
        elif extractor.product_info_extraction_mode == 'nlp':
            product_element = self._prepare_nlp_product_element(extractor, extra_item_pdf)
        
        return rag_context, product_element
    
    def _prepare_nlp_product_element(self, extractor, extra_item_pdf):
        """NLP 모드 제품 요소 준비"""
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
    LLM 호출 및 추출 단계
    
    - 프롬프트 구성
    - LLM 호출
    - 프롬프트 저장 (디버깅용)
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
    LLM 응답 JSON 파싱 단계
    
    - JSON 파싱
    - 스키마 응답 감지 (helpers 모듈 사용)
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
        
        state.set("json_objects", json_objects)
        state.set("raw_result", raw_result)
        
        return state


class ResultConstructionStep(WorkflowStep):
    """
    최종 결과 구성 단계
    
    - 엔티티 매칭
    - 채널 추출
    - 프로그램 매핑
    """
    
    def __init__(self, result_builder):
        self.result_builder = result_builder

    def execute(self, state: WorkflowState) -> WorkflowState:
        if state.has_error():
            return state
        
        json_objects = state.get("json_objects")
        msg = state.get("msg")
        pgm_info = state.get("pgm_info")
        entities_from_kiwi = state.get("entities_from_kiwi")
        extractor = state.get("extractor")
        
        # 최종 결과 구성
        final_result = self.result_builder.build_final_result(json_objects, msg, pgm_info, entities_from_kiwi)
        
        state.set("final_result", final_result)
        
        return state


class ValidationStep(WorkflowStep):
    """
    결과 검증 단계
    
    - 결과 유효성 검증 (helpers 모듈 사용)
    - 최종 결과 요약 로깅
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
    DAG 추출 단계 (선택적)
    
    현재는 process_message_with_dag에서 별도로 처리됨
    향후 workflow에 통합 가능
    """
    
    def execute(self, state: WorkflowState) -> WorkflowState:
        # DAG 추출은 process_message_with_dag에서 별도로 처리되므로
        # 여기서는 스킵
        # 필요시 나중에 구현 가능
        return state
