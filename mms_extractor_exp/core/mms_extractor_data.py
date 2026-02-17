# %%
"""
MMS Extractor - 데이터 로딩 및 초기화 모듈
===========================================

이 모듈은 MMSExtractor의 데이터 로딩 및 초기화 기능을 담당합니다.
Mixin 패턴을 사용하여 MMSExtractor 클래스에 통합됩니다.

주요 기능:
- LLM 모델 초기화
- 임베딩 모델 초기화  
- Kiwi 형태소 분석기 초기화
- 데이터 파일 로드 (상품, 프로그램, 조직 정보)
- 데이터베이스 연결 관리
"""

import time
import logging
import traceback
from typing import List
import os
import pandas as pd
import torch
from contextlib import contextmanager
import cx_Oracle
from kiwipiepy import Kiwi
from langchain_openai import ChatOpenAI
from joblib import Parallel, delayed
from dotenv import load_dotenv

# 유틸리티 함수 임포트
from utils import (
    log_performance,
    load_sentence_transformer,
    preprocess_text,
    select_most_comprehensive
)

# 설정 임포트
try:
    from config.settings import API_CONFIG, MODEL_CONFIG, PROCESSING_CONFIG, METADATA_CONFIG, EMBEDDING_CONFIG
except ImportError:
    logging.warning("설정 파일을 찾을 수 없습니다. 기본값을 사용합니다.")

logger = logging.getLogger(__name__)


class MMSExtractorDataMixin:
    """
    MMS Extractor 데이터 로딩 및 초기화 Mixin
    
    이 클래스는 MMSExtractor의 초기화 및 데이터 로딩 기능을 제공합니다.
    """
    
    def _set_default_config(self, model_path, data_dir, product_info_extraction_mode, 
                          entity_extraction_mode, offer_info_data_src, llm_model):
        """기본 설정값 적용"""
        self.data_dir = data_dir if data_dir is not None else './data/'
        self.model_path = model_path if model_path is not None else getattr(EMBEDDING_CONFIG, 'ko_sbert_model_path', 'jhgan/ko-sroberta-multitask')
        self.offer_info_data_src = offer_info_data_src
        self.product_info_extraction_mode = product_info_extraction_mode if product_info_extraction_mode is not None else getattr(PROCESSING_CONFIG, 'product_info_extraction_mode', 'nlp')
        self.entity_extraction_mode = entity_extraction_mode if entity_extraction_mode is not None else getattr(PROCESSING_CONFIG, 'entity_extraction_mode', 'llm')
        self.llm_model_name = llm_model
        self.num_cand_pgms = getattr(PROCESSING_CONFIG, 'num_candidate_programs', 20)
        self.num_select_pgms = getattr(PROCESSING_CONFIG, 'num_select_programs', 1)

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
                "gem": getattr(MODEL_CONFIG, 'gemma_model', 'gemma-7b'),
                "ax": getattr(MODEL_CONFIG, 'ax_model', 'ax-4'),
                "claude": getattr(MODEL_CONFIG, 'claude_model', 'claude-4'),
                "cld": getattr(MODEL_CONFIG, 'claude_model', 'claude-4'),
                "gemini": getattr(MODEL_CONFIG, 'gemini_model', 'gemini-pro'),
                "gen": getattr(MODEL_CONFIG, 'gemini_model', 'gemini-pro'),
                "gpt": getattr(MODEL_CONFIG, 'gpt_model', 'gpt-4')
            }
            
            model_name = model_mapping.get(self.llm_model_name, getattr(MODEL_CONFIG, 'llm_model', 'gemini-pro'))
            
            # LLM 모델별 일관성 설정
            model_kwargs = {
                "temperature": 0.0,
                "openai_api_key": getattr(API_CONFIG, 'llm_api_key', os.getenv('OPENAI_API_KEY')),
                "openai_api_base": getattr(API_CONFIG, 'llm_api_url', None),
                "model": model_name,
                "max_tokens": getattr(MODEL_CONFIG, 'llm_max_tokens', 4000)
            }
            
            # GPT 모델의 경우 시드 설정으로 일관성 강화
            if 'gpt' in model_name.lower():
                model_kwargs["seed"] = 42
                
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
            model_names (List[str]): 초기화할 모델명 리스트
            
        Returns:
            List: 초기화된 LLM 모델 객체 리스트
        """
        llm_models = []
        
        # 모델명 매핑
        model_mapping = {
            "cld": getattr(MODEL_CONFIG, 'anthropic_model', 'amazon/anthropic/claude-sonnet-4-20250514'),
            "ax": getattr(MODEL_CONFIG, 'ax_model', 'skt/ax4'),
            "gpt": getattr(MODEL_CONFIG, 'gpt_model', 'azure/openai/gpt-4o-2024-08-06'),
            "gen": getattr(MODEL_CONFIG, 'gemini_model', 'gcp/gemini-2.5-flash')
        }
        
        for model_name in model_names:
            try:
                actual_model_name = model_mapping.get(model_name, model_name)
                
                # 모델별 설정
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
            
            # 상품 정보 로드 및 준비
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
            logger.info(f"=== 상품 정보 로드 및 준비 시작 (모드: {self.offer_info_data_src}) ===")
            
            # 1단계: 데이터 소스에서 원본 데이터 로드
            if self.offer_info_data_src == "local":
                logger.info("📁 로컬 CSV 파일에서 로드")
                csv_path = getattr(METADATA_CONFIG, 'offer_data_path', './data/items.csv')
                item_pdf_raw = pd.read_csv(csv_path)
            elif self.offer_info_data_src == "db":
                logger.info("🗄️ 데이터베이스에서 로드")
                # Import DATABASE_CONFIG
                from config.settings import DATABASE_CONFIG
                with self._database_connection() as conn:
                    sql = DATABASE_CONFIG.get_offer_table_query()
                    item_pdf_raw = pd.read_sql(sql, conn)
            
            logger.info(f"원본 데이터 크기: {item_pdf_raw.shape}")
            
            # 2단계: 공통 전처리
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
            
            # 3단계: 별칭 규칙 로드 및 처리
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
            
            # 4단계: 별칭 규칙 연쇄 적용 (병렬 처리)
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
            
            # 5단계: 사용자 정의 엔티티 추가
            user_defined_entity = ['AIA Vitality', '부스트 파크 건대입구', 'Boost Park 건대입구']
            item_pdf_ext = pd.DataFrame([{
                'item_nm': e, 'item_id': e, 'item_desc': e, 'item_dmn': 'user_defined',
                'start_dt': 20250101, 'end_dt': 99991231, 'rank': 1, 'item_nm_alias': e
            } for e in user_defined_entity])
            item_pdf_all = pd.concat([item_pdf_all, item_pdf_ext])
            
            # 6단계: item_dmn_nm 컬럼 추가
            item_dmn_map = pd.DataFrame([
                {"item_dmn": 'P', 'item_dmn_nm': '요금제 및 관련 상품'},
                {"item_dmn": 'E', 'item_dmn_nm': '단말기'},
                {"item_dmn": 'S', 'item_dmn_nm': '구독 상품'},
                {"item_dmn": 'C', 'item_dmn_nm': '쿠폰'},
                {"item_dmn": 'X', 'item_dmn_nm': '가상 상품'}
            ])
            item_pdf_all = item_pdf_all.merge(item_dmn_map, on='item_dmn', how='left')
            item_pdf_all['item_dmn_nm'] = item_pdf_all['item_dmn_nm'].fillna('기타')
            
            # 7단계: TEST 필터링
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
            
            # Import DATABASE_CONFIG
            from config.settings import DATABASE_CONFIG
            
            with self._database_connection() as conn:
                # 프로그램 분류 정보 쿼리
                where_clause = """DEL_YN = 'N' 
                         AND APRV_OP_RSLT_CD = 'APPR'
                         AND EXPS_YN = 'Y'
                         AND CMPGN_PGM_NUM like '2025%' 
                         AND RMK is not null"""
                sql = DATABASE_CONFIG.get_program_table_query(where_clause)
                
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
                        _ = self.pgm_pdf.values
                        
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

    def _load_stopwords(self):
        """정지어 목록 로드"""
        try:
            self.stop_item_names = pd.read_csv(getattr(METADATA_CONFIG, 'stop_items_path', './data/stop_words.csv'))['stop_words'].to_list()
            logger.info(f"정지어 로드 완료: {len(self.stop_item_names)}개")
        except Exception as e:
            logger.warning(f"정지어 로드 실패: {e}")
            self.stop_item_names = []

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

    # pgm별 clue_tag 보강 규칙: 원본 데이터를 수정하지 않고 런타임에 추가
    PGM_CLUE_TAG_ENHANCEMENTS = {
        '매장오픈안내 및 방문유도': '직영점, 이벤트, 행사, 기념, 개통, 할인, 혜택, 안내드립니다, 주소, 연락처, 상담',
    }

    def _enhance_pgm_clue_tags(self):
        """pgm clue_tag를 런타임에 보강 (원본 CSV/DB 데이터 변경 없이)"""
        enhanced_count = 0
        for pgm_nm, extra_tags in self.PGM_CLUE_TAG_ENHANCEMENTS.items():
            mask = self.pgm_pdf['pgm_nm'] == pgm_nm
            if mask.any():
                original = self.pgm_pdf.loc[mask, 'clue_tag'].iloc[0]
                self.pgm_pdf.loc[mask, 'clue_tag'] = original + ', ' + extra_tags
                enhanced_count += 1
                logger.info(f"clue_tag 보강: {pgm_nm} (+{extra_tags})")
        if enhanced_count:
            logger.info(f"clue_tag 보강 완료: {enhanced_count}개 프로그램")

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
            # 빈 DataFrame으로 fallback
            self.org_pdf = pd.DataFrame(columns=['item_nm', 'item_id', 'item_desc', 'item_dmn'])
            logger.warning("빈 조직 DataFrame으로 fallback 설정됨")
            logger.warning("이로 인해 조직/매장 추출 기능이 비활성화됩니다.")

    def _load_org_from_database(self):
        """데이터베이스에서 조직 정보 로드 (ITEM_DMN='R')"""
        try:
            logger.info("데이터베이스 연결 시도 중...")
            
            # Import DATABASE_CONFIG
            from config.settings import DATABASE_CONFIG
            
            with self._database_connection() as conn:
                sql = DATABASE_CONFIG.get_offer_table_query("ITEM_DMN='R'")
                logger.info(f"실행할 SQL: {sql}")
                
                self.org_pdf = pd.read_sql(sql, conn)
                logger.info(f"DB에서 로드된 조직 데이터 크기: {self.org_pdf.shape}")
                logger.info(f"DB 조직 데이터 컬럼들: {list(self.org_pdf.columns)}")
                
                # 컬럼명 매핑 및 소문자 변환
                original_columns = list(self.org_pdf.columns)
                logger.info(f"DB 조직 데이터 원본 컬럼들: {original_columns}")
                
                # 조직 데이터를 위한 컬럼 매핑
                column_mapping = {c: c.lower() for c in self.org_pdf.columns}
                
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
            
            # 빈 DataFrame으로 fallback
            self.org_pdf = pd.DataFrame(columns=['item_nm', 'item_id', 'item_desc', 'item_dmn'])
            logger.warning("조직 데이터 DB 로드 실패로 빈 DataFrame 사용")
            
            raise
