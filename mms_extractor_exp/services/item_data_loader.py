"""
Item Data Loader Service
=========================

상품 데이터 로딩 및 전처리를 전담하는 서비스 클래스.
기존 MMSExtractor의 _load_and_prepare_item_data 메서드(197줄)를 
모듈화하여 재사용성과 테스트 용이성을 향상시킵니다.
"""

import logging
import os
import pandas as pd
from typing import List, Tuple
from joblib import Parallel, delayed
from utils import select_most_comprehensive

logger = logging.getLogger(__name__)


class ItemDataLoader:
    """
    상품 데이터 로딩 및 전처리 서비스
    
    책임:
    - 데이터 소스(CSV/DB)에서 원시 데이터 로드
    - 별칭 규칙 로드 및 적용
    - 데이터 필터링 및 정제
    - 메타데이터 컬럼 추가
    """
    
    def __init__(self, data_source='local', db_loader=None):
        """
        Args:
            data_source: 데이터 소스 ('local' 또는 'db')
            db_loader: DB 로더 함수 (data_source='db'일 때 필요)
        """
        self.data_source = data_source
        self.db_loader = db_loader
        self.alias_pdf_raw = None  # 별칭 규칙 원본 (외부에서 접근 가능)
        
    def load_raw_data(self, offer_data_path: str = None) -> pd.DataFrame:
        """
        데이터 소스에서 원시 상품 데이터 로드
        
        Args:
            offer_data_path: CSV 파일 경로 (local 모드일 때)
            
        Returns:
            pd.DataFrame: 원시 상품 데이터
        """
        logger.info(f"📊 상품 정보 로드 시작 (소스: {self.data_source})")
        
        if self.data_source == 'db':
            if not self.db_loader:
                raise ValueError("DB 모드에서는 db_loader가 필요합니다")
            logger.info("데이터베이스에서 로드 중...")
            item_pdf_raw = self.db_loader()
        else:
            if not offer_data_path:
                from config.settings import METADATA_CONFIG
                offer_data_path = getattr(METADATA_CONFIG, 'offer_data_path', './data/offer_master_data.csv')
            logger.info(f"CSV 파일에서 로드 중: {offer_data_path}")
            item_pdf_raw = pd.read_csv(offer_data_path, encoding='cp949')
        
        logger.info(f"원시 데이터 로드 완료: {item_pdf_raw.shape}")
        return item_pdf_raw
    
    def normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        컬럼명 정규화 및 기본 전처리
        
        Args:
            df: 원시 데이터프레임
            
        Returns:
            pd.DataFrame: 정규화된 데이터프레임
        """
        logger.info("🔧 컬럼 정규화 중...")
        
        # ITEM_NM 처리: 단말기(E)는 ITEM_DESC 사용
        if 'ITEM_DMN' in df.columns and 'ITEM_DESC' in df.columns:
            df['ITEM_NM'] = df.apply(
                lambda x: x['ITEM_DESC'] if x['ITEM_DMN']=='E' else x['ITEM_NM'], 
                axis=1
            )
        
        # 컬럼명을 소문자로 변환
        df = df.rename(columns={c: c.lower() for c in df.columns})
        logger.info("컬럼명 소문자 변환 완료")
        
        # 추가 컬럼 생성
        df['item_ctg'] = None
        df['item_emb_vec'] = None
        df['ofer_cd'] = df['item_id']
        df['oper_dt_hms'] = '20250101000000'
        
        return df
    
    def filter_by_domain(self, df: pd.DataFrame, excluded_domains: List[str] = None) -> pd.DataFrame:
        """
        도메인 코드로 데이터 필터링
        
        Args:
            df: 데이터프레임
            excluded_domains: 제외할 도메인 코드 리스트
            
        Returns:
            pd.DataFrame: 필터링된 데이터프레임
        """
        if not excluded_domains:
            from config.settings import PROCESSING_CONFIG
            excluded_domains = getattr(PROCESSING_CONFIG, 'excluded_domain_codes_for_items', [])
        
        if excluded_domains and 'item_dmn' in df.columns:
            before_filter = len(df)
            df = df.query("item_dmn not in @excluded_domains")
            logger.info(f"도메인 필터링: {before_filter} -> {len(df)}")
        
        return df
    
    def load_alias_rules(self, alias_rules_path: str = None) -> pd.DataFrame:
        """
        별칭 규칙 로드 및 전처리
        
        Args:
            alias_rules_path: 별칭 규칙 CSV 파일 경로
            
        Returns:
            pd.DataFrame: 전처리된 별칭 규칙
        """
        logger.info("🔗 별칭 규칙 로드 중...")
        
        if not alias_rules_path:
            from config.settings import METADATA_CONFIG
            alias_rules_path = getattr(METADATA_CONFIG, 'alias_rules_path', './data/alias_rules.csv')
        
        # 원본 저장 (외부 접근용)
        self.alias_pdf_raw = pd.read_csv(alias_rules_path)
        
        # 별칭 규칙 전처리
        alias_pdf = self.alias_pdf_raw.copy()
        alias_pdf['alias_1'] = alias_pdf['alias_1'].str.split("&&")
        alias_pdf['alias_2'] = alias_pdf['alias_2'].str.split("&&")
        alias_pdf = alias_pdf.explode('alias_1')
        alias_pdf = alias_pdf.explode('alias_2')
        
        return alias_pdf
    
    def expand_build_type_aliases(self, alias_pdf: pd.DataFrame, item_df: pd.DataFrame) -> pd.DataFrame:
        """
        'build' 타입 별칭 확장
        
        Args:
            alias_pdf: 별칭 규칙 데이터프레임
            item_df: 상품 데이터프레임
            
        Returns:
            pd.DataFrame: 확장된 별칭 규칙
        """
        logger.info("🔨 Build 타입 별칭 확장 중...")
        
        alias_list_ext = alias_pdf.query("type=='build'")[
            ['alias_1','category','direction','type']
        ].to_dict('records')
        
        for alias in alias_list_ext:
            adf = item_df.query(
                "item_nm.str.contains(@alias['alias_1']) and item_dmn==@alias['category']"
            )[['item_nm','item_desc','item_dmn']].rename(
                columns={'item_nm':'alias_2','item_desc':'description','item_dmn':'category'}
            ).drop_duplicates()
            
            adf['alias_1'] = alias['alias_1']
            adf['direction'] = alias['direction']
            adf['type'] = alias['type']
            adf = adf[alias_pdf.columns]
            
            alias_pdf = pd.concat([
                alias_pdf.query(f"alias_1!='{alias['alias_1']}'"), 
                adf
            ])
        
        alias_pdf = alias_pdf.drop_duplicates()
        return alias_pdf
    
    def add_bidirectional_aliases(self, alias_pdf: pd.DataFrame) -> pd.DataFrame:
        """
        양방향(B) 별칭 추가
        
        Args:
            alias_pdf: 별칭 규칙 데이터프레임
            
        Returns:
            pd.DataFrame: 양방향 별칭이 추가된 데이터프레임
        """
        logger.info("↔️ 양방향 별칭 추가 중...")
        
        bidirectional = alias_pdf.query("direction=='B'").rename(
            columns={'alias_1':'alias_2', 'alias_2':'alias_1'}
        )[alias_pdf.columns]
        
        alias_pdf = pd.concat([alias_pdf, bidirectional])
        return alias_pdf
    
    def apply_alias_cascade(self, item_df: pd.DataFrame, alias_pdf: pd.DataFrame, 
                           max_depth: int = 3, n_jobs: int = None) -> pd.DataFrame:
        """
        별칭 규칙 연쇄 적용 (병렬 처리)
        
        Args:
            item_df: 상품 데이터프레임
            alias_pdf: 별칭 규칙 데이터프레임
            max_depth: 최대 연쇄 깊이
            n_jobs: 병렬 작업 수
            
        Returns:
            pd.DataFrame: 별칭이 적용된 상품 데이터프레임
        """
        logger.info("🔄 별칭 규칙 연쇄 적용 중...")
        
        alias_rule_set = list(zip(
            alias_pdf['alias_1'], 
            alias_pdf['alias_2'], 
            alias_pdf['type']
        ))
        logger.info(f"별칭 규칙 수: {len(alias_rule_set)}개")
        
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
        
        # 별칭 적용
        item_alias_pdf = parallel_alias_rule_cascade(item_df['item_nm'], max_depth=max_depth, n_jobs=n_jobs)
        
        # 별칭 병합 및 explode
        item_df = item_df.merge(item_alias_pdf, on='item_nm', how='left')
        before_explode = len(item_df)
        item_df = item_df.explode('item_nm_alias').drop_duplicates()
        logger.info(f"별칭 explode: {before_explode} -> {len(item_df)}")
        
        return item_df
    
    def add_user_defined_entities(self, item_df: pd.DataFrame, 
                                  user_entities: List[str] = None) -> pd.DataFrame:
        """
        사용자 정의 엔티티 추가
        
        Args:
            item_df: 상품 데이터프레임
            user_entities: 사용자 정의 엔티티 리스트
            
        Returns:
            pd.DataFrame: 사용자 엔티티가 추가된 데이터프레임
        """
        if not user_entities:
            from config.settings import PROCESSING_CONFIG
            user_entities = getattr(PROCESSING_CONFIG, 'user_defined_entities', [])
        
        if user_entities:
            logger.info(f"👤 사용자 정의 엔티티 추가: {len(user_entities)}개")
            item_pdf_ext = pd.DataFrame([{
                'item_nm': e, 'item_id': e, 'item_desc': e, 'item_dmn': 'user_defined',
                'start_dt': 20250101, 'end_dt': 99991231, 'rank': 1, 'item_nm_alias': e
            } for e in user_entities])
            item_df = pd.concat([item_df, item_pdf_ext])
        
        return item_df
    
    def add_domain_name_column(self, item_df: pd.DataFrame) -> pd.DataFrame:
        """
        item_dmn_nm 컬럼 추가 (도메인 코드 -> 도메인명)
        
        Args:
            item_df: 상품 데이터프레임
            
        Returns:
            pd.DataFrame: 도메인명이 추가된 데이터프레임
        """
        logger.info("🏷️ 도메인명 컬럼 추가 중...")
        
        item_dmn_map = pd.DataFrame([
            {"item_dmn": 'P', 'item_dmn_nm': '요금제 및 관련 상품'},
            {"item_dmn": 'E', 'item_dmn_nm': '단말기'},
            {"item_dmn": 'S', 'item_dmn_nm': '구독 상품'},
            {"item_dmn": 'C', 'item_dmn_nm': '쿠폰'},
            {"item_dmn": 'X', 'item_dmn_nm': '가상 상품'}
        ])
        
        item_df = item_df.merge(item_dmn_map, on='item_dmn', how='left')
        item_df['item_dmn_nm'] = item_df['item_dmn_nm'].fillna('기타')
        
        return item_df
    
    def filter_test_items(self, item_df: pd.DataFrame) -> pd.DataFrame:
        """
        TEST 문자열이 포함된 항목 필터링
        
        Args:
            item_df: 상품 데이터프레임
            
        Returns:
            pd.DataFrame: TEST 항목이 제거된 데이터프레임
        """
        logger.info("🧹 TEST 항목 필터링 중...")
        
        before_test = len(item_df)
        item_df = item_df.query("not item_nm_alias.str.contains('TEST', case=False, na=False)")
        logger.info(f"TEST 필터링: {before_test} -> {len(item_df)}")
        
        return item_df
    
    def prepare_item_data(self, offer_data_path: str = None, 
                         alias_rules_path: str = None,
                         excluded_domains: List[str] = None,
                         user_entities: List[str] = None) -> pd.DataFrame:
        """
        전체 파이프라인 실행: 데이터 로드부터 최종 전처리까지
        
        Args:
            offer_data_path: 상품 데이터 CSV 경로
            alias_rules_path: 별칭 규칙 CSV 경로
            excluded_domains: 제외할 도메인 코드 리스트
            user_entities: 사용자 정의 엔티티 리스트
            
        Returns:
            pd.DataFrame: 최종 전처리된 상품 데이터
        """
        try:
            logger.info("=" * 50)
            logger.info("🚀 상품 데이터 준비 파이프라인 시작")
            logger.info("=" * 50)
            
            # 1. 원시 데이터 로드
            raw_data = self.load_raw_data(offer_data_path)
            
            # 2. 컬럼 정규화
            normalized_data = self.normalize_columns(raw_data)
            
            # 3. 도메인 필터링
            filtered_data = self.filter_by_domain(normalized_data, excluded_domains)
            
            # 4. 별칭 규칙 로드 및 처리
            alias_pdf = self.load_alias_rules(alias_rules_path)
            alias_pdf = self.expand_build_type_aliases(alias_pdf, filtered_data)
            alias_pdf = self.add_bidirectional_aliases(alias_pdf)
            
            # 5. 별칭 연쇄 적용
            with_aliases = self.apply_alias_cascade(filtered_data, alias_pdf)
            
            # 6. 사용자 정의 엔티티 추가
            with_user_entities = self.add_user_defined_entities(with_aliases, user_entities)
            
            # 7. 도메인명 컬럼 추가
            with_domain_names = self.add_domain_name_column(with_user_entities)
            
            # 8. TEST 항목 필터링
            final_data = self.filter_test_items(with_domain_names)
            
            # 최종 확인
            logger.info("=" * 50)
            logger.info("✅ 상품 정보 준비 완료")
            logger.info(f"최종 데이터 크기: {final_data.shape}")
            logger.info(f"최종 컬럼들: {list(final_data.columns)}")
            
            # 중요 컬럼 확인
            critical_columns = ['item_nm', 'item_id', 'item_nm_alias']
            missing_columns = [col for col in critical_columns if col not in final_data.columns]
            if missing_columns:
                logger.error(f"중요 컬럼 누락: {missing_columns}")
            else:
                logger.info("✅ 모든 중요 컬럼 존재")
            
            # 샘플 데이터 확인
            if not final_data.empty:
                logger.info(f"상품명 샘플: {final_data['item_nm'].dropna().head(3).tolist()}")
                logger.info(f"별칭 샘플: {final_data['item_nm_alias'].dropna().head(3).tolist()}")
            
            logger.info("=" * 50)
            
            return final_data
            
        except Exception as e:
            logger.error(f"상품 정보 로드 및 준비 실패: {e}")
            import traceback
            logger.error(f"오류 상세: {traceback.format_exc()}")
            # 빈 DataFrame으로 fallback
            return pd.DataFrame(columns=['item_nm', 'item_id', 'item_desc', 'item_dmn', 'item_nm_alias'])
