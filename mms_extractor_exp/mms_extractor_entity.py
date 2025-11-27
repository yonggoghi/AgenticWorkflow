# %%
"""
MMS Extractor - 엔티티 추출 및 매칭 모듈
========================================

이 모듈은 MMSExtractor의 엔티티 추출 및 매칭 기능을 담당합니다.
Mixin 패턴을 사용하여 MMSExtractor 클래스에 통합됩니다.

주요 기능:
- Kiwi 기반 엔티티 추출
- LLM 기반 엔티티 추출
- 엔티티-상품 매칭
- 유사도 계산 및 필터링
"""

import logging
import traceback
import re
from typing import List, Tuple, Dict
import pandas as pd
from langchain_core.prompts import PromptTemplate
from joblib import Parallel, delayed

# 유틸리티 함수 임포트
from utils import (
    log_performance,
    validate_text_input,
    safe_execute,
    parallel_fuzzy_similarity,
    parallel_seq_similarity,
    filter_text_by_exc_patterns,
    filter_specific_terms,
    extract_ngram_candidates
)

# 프롬프트 임포트
from prompts import (
    HYBRID_DAG_EXTRACTION_PROMPT,
    SIMPLE_ENTITY_EXTRACTION_PROMPT
)

# 설정 임포트
try:
    from config.settings import PROCESSING_CONFIG
except ImportError:
    logging.warning("설정 파일을 찾을 수 없습니다. 기본값을 사용합니다.")

logger = logging.getLogger(__name__)


class MMSExtractorEntityMixin:
    """
    MMS Extractor 엔티티 추출 및 매칭 Mixin
    
    이 클래스는 MMSExtractor의 엔티티 추출 및 매칭 기능을 제공합니다.
    """
    
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
            
            logger.info(f"필터링된 문장들: {sentence_list[:3]}...")

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
                
                if cand_item_list:
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
                default_return=pd.DataFrame()
            ).rename(columns={'sim': 'sim_s2'})
            
            # 결과 합치기
            if not sim_s1.empty and not sim_s2.empty:
                try:
                    combined = sim_s1.merge(sim_s2, on=['item_name_in_msg', 'item_nm_alias'])
                    filtered = combined.query("(sim_s1>=@PROCESSING_CONFIG.combined_similarity_threshold and sim_s2>=@PROCESSING_CONFIG.combined_similarity_threshold)")
                    if filtered.empty:
                        logger.warning("결합 유사도 계산 결과가 비어있음")
                        return pd.DataFrame()
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
                logger.warning("결합 유사도 계산 결과가 비어있음")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"결합 유사도 계산 실패: {e}")
            return pd.DataFrame()

    def _parse_entity_response(self, response: str) -> List[str]:
        """
        LLM 응답에서 엔티티를 견고하게 파싱
        
        여러 전략을 사용하여 다양한 LLM 응답 형식을 처리
        """
        try:
            # Strategy 1: ENTITY: 라인을 찾아서 정확하게 추출
            lines = response.split('\n')
            for line in lines:
                line_stripped = line.strip()
                line_upper = line_stripped.upper()
                
                if line_upper.startswith('REASON:'):
                    continue
                
                if line_upper.startswith('ENTITY:'):
                    entity_part = line_stripped[line_upper.find('ENTITY:') + 7:].strip()
                    
                    if not entity_part or entity_part.lower() in ['none', 'empty', '없음', 'null']:
                        logger.debug("ENTITY 섹션이 비어있음 (정상)")
                        return []
                    
                    if len(entity_part) > 200:
                        logger.warning(f"ENTITY 값이 너무 김 ({len(entity_part)}자) - 설명 문장으로 판단")
                        continue
                    
                    entities = [e.strip() for e in entity_part.split(',') if e.strip()]
                    
                    valid_entities = []
                    for entity in entities:
                        if len(entity) > 100:
                            logger.debug(f"엔티티가 너무 김 ({len(entity)}자): {entity[:50]}...")
                            continue
                        if entity.startswith('"') and not entity.endswith('"'):
                            logger.debug(f"불완전한 따옴표 구조: {entity[:50]}...")
                            continue
                        valid_entities.append(entity)
                    
                    if valid_entities:
                        logger.debug(f"파싱된 엔티티: {valid_entities}")
                        return valid_entities
            
            # Strategy 2: ENTITY: 패턴을 정규식으로 찾기
            entity_pattern = r'ENTITY:\s*([^\n]*?)(?:\n|$)'
            entity_matches = list(re.finditer(entity_pattern, response, re.IGNORECASE))
            
            if entity_matches:
                last_match = entity_matches[-1]
                entity_text = last_match.group(1).strip()
                
                if entity_text and entity_text.lower() not in ['none', 'empty', '없음', 'null']:
                    if len(entity_text) <= 200:
                        entities = [e.strip() for e in entity_text.split(',') 
                                   if e.strip() and len(e.strip()) <= 100]
                        if entities:
                            logger.debug(f"정규식으로 파싱된 엔티티: {entities}")
                            return entities
            
            # Strategy 3: ENTITY: 키워드 없이 엔티티만 반환된 경우
            for line in reversed(lines):
                line_stripped = line.strip()
                if not line_stripped:
                    continue
                
                if line_stripped.upper().startswith('REASON:'):
                    continue
                
                if len(line_stripped) > 200:
                    continue
                
                if ',' in line_stripped:
                    entities = [e.strip() for e in line_stripped.split(',') 
                               if e.strip() and len(e.strip()) <= 100]
                    if entities:
                        if all(len(e) <= 100 for e in entities):
                            logger.debug(f"키워드 없이 파싱된 엔티티: {entities}")
                            return entities
                elif len(line_stripped) <= 100:
                    logger.debug(f"단일 엔티티: [{line_stripped}]")
                    return [line_stripped]
            
            # Strategy 4: 빈 리스트 반환
            logger.debug(f"엔티티를 찾을 수 없음. 응답: {response[:100]}...")
            return []
            
        except Exception as e:
            logger.error(f"엔티티 응답 파싱 실패: {e}")
            return []
    
    def _calculate_optimal_batch_size(self, msg_text: str, base_size: int = 50) -> int:
        """메시지 길이에 따라 동적으로 배치 크기 계산"""
        msg_length = len(msg_text)
        
        if msg_length < 500:
            return min(base_size * 2, 100)
        elif msg_length < 1000:
            return base_size
        else:
            return max(base_size // 2, 25)
    
    @log_performance
    def extract_entities_by_llm(self, msg_text: str, rank_limit: int = 50, llm_models: List = None, external_cand_entities: List[str] = []) -> pd.DataFrame:
        """
        LLM 기반 엔티티 추출 (복수 모델 병렬 처리 지원)
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
            
            # LLM 모델이 지정되지 않은 경우 기본 모델 사용
            if llm_models is None:
                llm_models = [self.llm_model]
                logger.info(f"🤖 LLM 모델 자동 선택: 기본 모델 사용 (1개)")
            else:
                logger.info(f"🤖 LLM 모델 지정됨: {len(llm_models)}개 모델 사용")
            
            for idx, model in enumerate(llm_models):
                model_name = getattr(model, 'model_name', 'Unknown')
                logger.info(f"   [{idx+1}] 모델: {model_name}")
            
            def get_entities_and_dag_by_llm(args_dict):
                """단일 LLM으로 엔티티와 DAG 추출하는 내부 함수"""
                llm_model, prompt = args_dict['llm_model'], args_dict['prompt']
                extract_dag = args_dict.get('extract_dag', True)  # 기본값은 True (하위 호환성)
                model_name = getattr(llm_model, 'model_name', 'Unknown')
                
                try:
                    logger.info(f"🔄 [{model_name}] LLM 호출 시작")
                    logger.info(f"   📏 프롬프트 길이: {len(prompt):,} 문자")
                    logger.info(f"   📄 프롬프트 미리보기 (처음 200자): {prompt[:200]}...")
                    
                    # PromptTemplate 사용
                    zero_shot_prompt = PromptTemplate(
                        input_variables=["prompt"],
                        template="{prompt}"
                    )
                    
                    logger.info(f"   🚀 [{model_name}] LLM API 호출 중...")
                    # LLM 호출
                    chain = zero_shot_prompt | llm_model
                    response = chain.invoke({"prompt": prompt}).content
                    
                    logger.info(f"   ✅ [{model_name}] LLM 응답 수신 완료")
                    logger.info(f"   📏 응답 길이: {len(response):,} 문자")
                    logger.info(f"   📄 응답 미리보기 (처음 300자): {response[:300]}...")
                    
                    # 견고한 응답 파싱 사용
                    logger.info(f"   🔍 [{model_name}] 엔티티 파싱 시작...")
                    cand_entity_list_raw = self._parse_entity_response(response)
                    logger.info(f"   📊 [{model_name}] 파싱된 원본 엔티티 수: {len(cand_entity_list_raw)}개")
                    if cand_entity_list_raw:
                        logger.info(f"   📝 [{model_name}] 원본 엔티티: {cand_entity_list_raw}")
                    
                    # 정지어 및 길이 필터링
                    cand_entity_list = [e for e in cand_entity_list_raw if e not in self.stop_item_names and len(e) >= 2]
                    logger.info(f"   📊 [{model_name}] 필터링 후 엔티티 수: {len(cand_entity_list)}개")
                    if cand_entity_list:
                        logger.info(f"   📝 [{model_name}] 필터링된 엔티티: {cand_entity_list}")
                    else:
                        logger.warning(f"   ⚠️ [{model_name}] 필터링 후 유효한 엔티티가 없습니다!")
                    
                    # DAG 섹션 추출 (extract_dag가 True일 때만)
                    dag_text = ""
                    if extract_dag:
                        logger.info(f"   🔍 [{model_name}] DAG 섹션 추출 시작...")
                        dag_match = re.search(r'DAG:\s*(.*)', response, re.DOTALL | re.IGNORECASE)
                        if dag_match:
                            dag_text = dag_match.group(1).strip()
                            logger.info(f"   ✅ [{model_name}] DAG 추출 성공")
                            logger.info(f"   📏 DAG 텍스트 길이: {len(dag_text):,} 문자")
                            logger.info(f"   📄 DAG 미리보기 (처음 200자): {dag_text[:200]}...")
                        else:
                            logger.warning(f"   ⚠️ [{model_name}] DAG 섹션을 찾을 수 없습니다")
                            logger.info(f"   💡 응답에 'DAG:' 키워드가 포함되어 있는지 확인: {'DAG:' in response.upper()}")
                    
                    logger.info(f"   ✅ [{model_name}] 처리 완료 - 엔티티: {len(cand_entity_list)}개, DAG: {'있음' if dag_text else '없음'}")
                    return {"entities": cand_entity_list, "dag_text": dag_text}
                    
                except Exception as e:
                    logger.error(f"   ❌ [{model_name}] LLM 모델에서 엔티티 추출 실패: {e}")
                    logger.error(f"   ❌ [{model_name}] 오류 타입: {type(e).__name__}")
                    logger.error(f"   ❌ [{model_name}] 오류 상세: {traceback.format_exc()}")
                    return {"entities": [], "dag_text": ""}
            
            def get_entities_only_by_llm(args_dict):
                """get_entities_and_dag_by_llm의 래퍼 (엔티티 리스트만 반환)"""
                result = get_entities_and_dag_by_llm(args_dict)
                return result['entities']
            
            # 프롬프트 미리보기 저장
            logger.info("📋 프롬프트 미리보기 저장 중...")
            preview_prompt = f"""
            {HYBRID_DAG_EXTRACTION_PROMPT}

            ## message:                
            {msg_text}
            """
            self._store_prompt_for_preview(preview_prompt, "entity_extraction")
            logger.info("✅ 프롬프트 미리보기 저장 완료")

            
            logger.info("🔄 1단계 LLM 추출 - 메시지에서 직접 엔티티 및 DAG 추출")
            # 1단계: 각 LLM 모델로 메시지에서 엔티티 추출
            batches = []
            for llm_model in llm_models:
                prompt = f"""
                {HYBRID_DAG_EXTRACTION_PROMPT}

                ## message:                
                {msg_text}
                """
                batches.append({"prompt": prompt, "llm_model": llm_model, "extract_dag": True})  # 1단계는 DAG 추출 필요
            
            logger.info(f"🔄 {len(llm_models)}개 LLM 모델로 1단계 엔티티 추출 시작")
            
            # 병렬 작업 실행
            n_jobs = min(len(batches), 3)
            logger.info(f"⚙️  병렬 처리 설정: {n_jobs}개 워커 (threading 백엔드)")
            
            with Parallel(n_jobs=n_jobs, backend='threading') as parallel:
                batch_results_dicts = parallel(delayed(get_entities_and_dag_by_llm)(args) for args in batches)
            
            logger.info(f"✅ 모든 LLM 모델 처리 완료")
            
            # 결과 분리 및 수집
            all_entities = []
            all_dags = []
            
            for idx, (model, result_dict) in enumerate(zip(llm_models, batch_results_dicts)):
                model_name = getattr(model, 'model_name', 'Unknown')
                entities = result_dict['entities']
                dag_text = result_dict['dag_text']
                
                logger.info(f"   [{idx+1}] {model_name}: {len(entities)}개 엔티티 추출")
                all_entities.extend(entities)
                if dag_text:
                    all_dags.append(dag_text)
            
            # DAG 컨텍스트 병합
            combined_dag_context = "\n".join(all_dags)
            if combined_dag_context:
                logger.info(f"   📝 캡처된 DAG 컨텍스트 길이: {len(combined_dag_context)}자")
            
            # 외부 엔티티 추가 및 중복 제거
            if external_cand_entities is not None and len(external_cand_entities)>0:
                all_entities.extend(external_cand_entities)
            
            logger.info(f"📊 병합 전 총 엔티티 수: {len(all_entities)}개")
            cand_entity_list = list(set(all_entities))
            
            # N-gram 확장
            cand_entity_list = list(set(sum([[c['text'] for c in extract_ngram_candidates(cand_entity, min_n=2, max_n=len(cand_entity.split())) if c['start_idx']<=0] if len(cand_entity.split())>=4 else [cand_entity] for cand_entity in cand_entity_list], [])))
            
            logger.info(f"📊 중복 제거 및 확장 후 엔티티 수: {len(cand_entity_list)}개")
            logger.info(f"✅ LLM 추출 완료: {cand_entity_list[:20]}..." if len(cand_entity_list) > 20 else f"✅ LLM 추출 완료: {cand_entity_list}")

            if not cand_entity_list:
                logger.warning("⚠️  LLM 추출에서 유효한 엔티티를 찾지 못함")
                logger.info("=" * 80)
                return pd.DataFrame()
            
            logger.info("🔍 엔티티-상품 매칭 시작...")
            logger.info(f"   입력 엔티티 수: {len(cand_entity_list)}개")
            cand_entities_sim = self._match_entities_with_products(cand_entity_list, rank_limit)
            logger.info(f"   매칭 결과: {len(cand_entities_sim)}개 행")
            
            if cand_entities_sim.empty:
                logger.warning("⚠️  엔티티-상품 매칭 결과가 비어있음")
                logger.info("=" * 80)
                return pd.DataFrame()
            
            logger.info(f"   매칭된 고유 item_name_in_msg 수: {cand_entities_sim['item_name_in_msg'].nunique()}개")
            logger.info(f"   매칭된 고유 item_nm_alias 수: {cand_entities_sim['item_nm_alias'].nunique()}개")

            # 후보 엔티티들과 상품 DB 매칭
            logger.info("🔍 2단계 LLM 필터링 시작 (동적 배치 크기 + DAG 컨텍스트 사용)...")
            logger.info(f"   입력 메시지 엔티티 수: {len(cand_entities_sim['item_name_in_msg'].unique())}개")
            logger.info(f"   후보 상품 별칭 수: {len(cand_entities_sim['item_nm_alias'].unique())}개")
            
            # entities_in_message 추출
            entities_in_message = cand_entities_sim['item_name_in_msg'].unique()
            
            # 2단계: 동적 배치 크기 계산
            optimal_batch_size = self._calculate_optimal_batch_size(msg_text, base_size=10)
            logger.info(f"   📏 메시지 길이 기반 최적 배치 크기: {optimal_batch_size}개")
            
            # cand_entities_voca_all을 동적 배치 크기로 분할해서 병렬 처리
            cand_entities_voca_all = cand_entities_sim['item_nm_alias'].unique()
            logger.info(f"   총 후보 상품 별칭: {len(cand_entities_voca_all)}개")
            
            # 2단계 필터링에는 첫 번째 모델 사용
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
                batches.append({"prompt": prompt, "llm_model": second_stage_llm, "extract_dag": False})  # 2단계는 DAG 추출 불필요
            
            logger.info(f"🔄 2단계 LLM 필터링: {len(batches)}개 배치로 분할 (배치당 ~{optimal_batch_size}개)")
            
            # 병렬 작업 실행
            n_jobs = min(len(batches), 3)
            logger.info(f"⚙️  병렬 처리 설정: {n_jobs}개 워커 (threading 백엔드)")
            
            with Parallel(n_jobs=n_jobs, backend='threading') as parallel:
                batch_results = parallel(delayed(get_entities_only_by_llm)(args) for args in batches)
            
            # 모든 배치 결과를 합치고 중복 제거
            logger.info(f"📊 배치별 결과 요약:")
            for idx, batch_result in enumerate(batch_results):
                logger.info(f"   배치 {idx+1}: {len(batch_result)}개 엔티티")
            
            cand_entity_list = list(set(sum(batch_results, [])))
            
            logger.info(f"✅ 2단계 LLM 필터링 완료")
            logger.info(f"📊 최종 선택된 엔티티 수: {len(cand_entity_list)}개")
            logger.info(f"📊 최종 선택된 엔티티: {cand_entity_list}")

            logger.info(f"🔍 최종 엔티티로 필터링 중...")
            logger.info(f"   필터링 전 행 수: {len(cand_entities_sim)}개")
            
            cand_entities_sim = cand_entities_sim.query("item_nm_alias in @cand_entity_list")
            logger.info(f"   필터링 후 행 수: {len(cand_entities_sim)}개")
            
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
        """후보 엔티티들을 상품 DB와 매칭"""
        try:
            logger.info("   🔍 [매칭] 퍼지 유사도 매칭 시작...")
            logger.info(f"   📝 입력 엔티티 수: {len(cand_entity_list)}개")
            logger.info(f"   📝 상품 DB 별칭 수: {len(self.item_pdf_all['item_nm_alias'].unique()):,}개")
            
            # 퍼지 유사도 매칭
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
            
            # 정지어 필터링
            before_stopwords = len(similarities_fuzzy)
            similarities_fuzzy = similarities_fuzzy[
                ~similarities_fuzzy['item_nm_alias'].isin(self.stop_item_names)
            ]
            after_stopwords = len(similarities_fuzzy)
            logger.info(f"   📊 정지어 필터링 결과: {before_stopwords}개 → {after_stopwords}개")

            # 시퀀스 유사도 매칭
            logger.info("   🔍 [매칭] 시퀀스 유사도 계산 시작 (s1, s2 각각)...")
            
            # s1 정규화
            sim_s1 = parallel_seq_similarity(
                sent_item_pdf=similarities_fuzzy,
                text_col_nm='item_name_in_msg',
                item_col_nm='item_nm_alias',
                n_jobs=6,
                batch_size=30,
                normalizaton_value='s1'
            ).rename(columns={'sim': 'sim_s1'})
            
            # s2 정규화
            sim_s2 = parallel_seq_similarity(
                sent_item_pdf=similarities_fuzzy,
                text_col_nm='item_name_in_msg',
                item_col_nm='item_nm_alias',
                n_jobs=6,
                batch_size=30,
                normalizaton_value='s2'
            ).rename(columns={'sim': 'sim_s2'})
            
            logger.info(f"   ✅ 시퀀스 유사도 계산 완료: sim_s1={len(sim_s1)}개, sim_s2={len(sim_s2)}개")
            
            # merge로 합치기
            cand_entities_sim = sim_s1.merge(sim_s2, on=['item_name_in_msg', 'item_nm_alias'])
            logger.info(f"   ✅ 병합 완료: {len(cand_entities_sim)}개 행")
            
            if cand_entities_sim.empty:
                logger.warning("   ⚠️  시퀀스 유사도 계산 결과가 비어있음")
                return pd.DataFrame()
            
            # 필터링 조건 적용
            before_query = len(cand_entities_sim)
            cand_entities_sim = cand_entities_sim.query("(sim_s1>=@PROCESSING_CONFIG.combined_similarity_threshold and sim_s2>=@PROCESSING_CONFIG.combined_similarity_threshold)")
            after_query = len(cand_entities_sim)
            logger.info(f"   📊 쿼리 필터링 결과: {before_query}개 → {after_query}개")

            # groupby로 합산
            cand_entities_sim = cand_entities_sim.groupby(['item_name_in_msg', 'item_nm_alias'])[['sim_s1', 'sim_s2']].apply(
                lambda x: x['sim_s1'].sum() + x['sim_s2'].sum()
            )
            if cand_entities_sim.empty:
                logger.warning("합산 결과가 비어있음")
                return pd.DataFrame()
            
            cand_entities_sim = cand_entities_sim.reset_index(name='sim')
            logger.info(f"   ✅ 합산 완료: {len(cand_entities_sim)}개 행")
            
            # sim>=1.0 필터링
            before_sim_filter = len(cand_entities_sim)
            cand_entities_sim = cand_entities_sim.query("sim >= @PROCESSING_CONFIG.high_similarity_threshold").copy()
            if cand_entities_sim.empty:
                logger.warning("필터링 결과가 비어있음")
                return pd.DataFrame()
            after_sim_filter = len(cand_entities_sim)
            logger.info(f"   📊 유사도 필터링 결과: {before_sim_filter}개 → {after_sim_filter}개")

            # 순위 매기기 및 결과 제한
            cand_entities_sim["rank"] = cand_entities_sim.groupby('item_name_in_msg')['sim'].rank(
                method='dense', ascending=False
            )
            before_rank_limit = len(cand_entities_sim)
            cand_entities_sim = cand_entities_sim.query(f"rank <= {rank_limit}").sort_values(
                ['item_name_in_msg', 'rank'], ascending=[True, True]
            )
            after_rank_limit = len(cand_entities_sim)
            logger.info(f"   📊 순위 제한 결과: {before_rank_limit}개 → {after_rank_limit}개")
            
            # item_dmn_nm 병합
            if 'item_dmn_nm' in self.item_pdf_all.columns:
                cand_entities_sim = cand_entities_sim.merge(
                    self.item_pdf_all[['item_nm_alias', 'item_dmn_nm']].drop_duplicates(),
                    on='item_nm_alias',
                    how='left'
                )
                logger.info(f"   ✅ item_dmn_nm 병합 완료")
            
            logger.info(f"   ✅ [매칭] 최종 결과: {len(cand_entities_sim)}개 행")

            return cand_entities_sim
            
        except Exception as e:
            logger.error(f"   ❌ [매칭] 엔티티-상품 매칭 실패: {e}")
            logger.error(f"   ❌ [매칭] 오류 상세: {traceback.format_exc()}")
            return pd.DataFrame()

    def _map_products_with_similarity(self, similarities_fuzzy: pd.DataFrame, json_objects: Dict = None) -> List[Dict]:
        """유사도를 기반으로 상품 정보 매핑"""
        try:
            logger.info("🔍 [_map_products_with_similarity] 시작")
            logger.info(f"   - 입력 similarities_fuzzy 크기: {similarities_fuzzy.shape}")
            
            # 높은 유사도 아이템들 필터링
            high_sim_threshold = getattr(PROCESSING_CONFIG, 'high_similarity_threshold', 1.0)
            logger.info(f"   - high_sim_threshold: {high_sim_threshold}")
            
            high_sim_items = similarities_fuzzy.query('sim >= @high_sim_threshold')['item_nm_alias'].unique()
            logger.info(f"   - high_sim_items 개수: {len(high_sim_items)}개")
            
            before_filter = len(similarities_fuzzy)
            filtered_similarities = similarities_fuzzy[
                (similarities_fuzzy['item_nm_alias'].isin(high_sim_items)) &
                (~similarities_fuzzy['item_nm_alias'].str.contains('test', case=False)) &
                (~similarities_fuzzy['item_name_in_msg'].isin(self.stop_item_names))
            ]
            after_filter = len(filtered_similarities)
            logger.info(f"   - 필터링: {before_filter}개 → {after_filter}개")
            
            if filtered_similarities.empty:
                logger.warning("   ⚠️ filtered_similarities가 비어있음 → 빈 배열 반환")
                return []
            
            # 상품 정보와 매핑하여 최종 결과 생성
            merged_items = self.item_pdf_all.merge(filtered_similarities, on=['item_nm_alias'])
            logger.info(f"   - merged_items 크기: {merged_items.shape}")
            
            if merged_items.empty:
                logger.warning("   ⚠️ merged_items가 비어있음 → 빈 배열 반환")
                return []
            
            product_tag = self.convert_df_to_json_list(merged_items)
            logger.info(f"   ✅ product_tag 개수: {len(product_tag)}개")
            
            # Add expected_action to each product
            if json_objects:
                logger.info("   🔍 expected_action 추가 시작")
                action_mapping = self._create_action_mapping(json_objects)
                
                for product in product_tag:
                    item_names_in_msg = product.get('item_name_in_msg', [])
                    found_actions = []
                    for item_name in item_names_in_msg:
                        if item_name in action_mapping:
                            found_actions.append(action_mapping[item_name])
                    product['expected_action'] = list(dict.fromkeys(found_actions)) if found_actions else ['기타']
                
                logger.info(f"   ✅ expected_action 추가 완료")
            
            logger.info(f"✅ [_map_products_with_similarity] 완료 - 반환: {len(product_tag)}개")
            return product_tag
            
        except Exception as e:
            logger.error(f"❌ [_map_products_with_similarity] 실패: {e}")
            logger.error(f"   오류 상세: {traceback.format_exc()}")
            return []

    def _create_action_mapping(self, json_objects: Dict) -> Dict[str, str]:
        """LLM 응답에서 상품명-액션 매핑 생성"""
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
                    logger.debug("스키마 정의 구조 감지됨, 액션 매핑 건너뛰기")
                else:
                    if 'name' in product_data and 'action' in product_data:
                        action_mapping[product_data['name']] = product_data['action']
            
            logger.debug(f"생성된 액션 매핑: {action_mapping}")
            return action_mapping
            
        except Exception as e:
            logger.error(f"액션 매핑 생성 실패: {e}")
            return {}
