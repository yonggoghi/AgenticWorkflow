#!/usr/bin/env python
"""
DB Connection Diagnostic Script
================================

리눅스 환경에서 --offer-data-source db 실행 시 발생하는 문제를 진단합니다.

주요 체크 항목:
1. DB 연결 상태
2. 오퍼 데이터 (item) 로드 확인
3. 프로그램 데이터 로드 확인
4. 조직 데이터 로드 확인
5. 데이터 컬럼 및 샘플 확인

사용법:
    python tests/diagnose_db_connection.py
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import traceback
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_environment_variables():
    """환경 변수 확인"""
    logger.info("=" * 80)
    logger.info("1. 환경 변수 확인")
    logger.info("=" * 80)
    
    required_vars = ['DB_USERNAME', 'DB_PASSWORD', 'DB_HOST', 'DB_PORT', 'DB_NAME']
    missing_vars = []
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            if 'PASSWORD' in var:
                logger.info(f"✅ {var}: {'*' * len(value)}")
            else:
                logger.info(f"✅ {var}: {value}")
        else:
            logger.error(f"❌ {var}: [비어있음]")
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"\n⚠️  누락된 환경 변수: {missing_vars}")
        logger.error("   .env 파일을 확인하세요.")
        return False
    else:
        logger.info("\n✅ 모든 환경 변수가 설정되어 있습니다.")
        return True


def check_cx_oracle():
    """cx_Oracle 모듈 확인"""
    logger.info("\n" + "=" * 80)
    logger.info("2. cx_Oracle 모듈 확인")
    logger.info("=" * 80)
    
    try:
        import cx_Oracle
        logger.info(f"✅ cx_Oracle 버전: {cx_Oracle.version}")
        
        # Oracle Client 버전 확인
        try:
            client_version = cx_Oracle.clientversion()
            logger.info(f"✅ Oracle Client 버전: {'.'.join(map(str, client_version))}")
        except Exception as e:
            logger.warning(f"⚠️  Oracle Client 버전 확인 실패: {e}")
        
        return True
    except ImportError as e:
        logger.error(f"❌ cx_Oracle 모듈을 찾을 수 없습니다: {e}")
        logger.error("   설치 명령: pip install cx_Oracle")
        return False


def check_database_connection():
    """데이터베이스 연결 확인"""
    logger.info("\n" + "=" * 80)
    logger.info("3. 데이터베이스 연결 확인")
    logger.info("=" * 80)
    
    try:
        from utils.db_utils import get_database_connection
        
        logger.info("데이터베이스 연결 시도 중...")
        conn = get_database_connection()
        
        if conn:
            logger.info(f"✅ 데이터베이스 연결 성공!")
            logger.info(f"   DB 버전: {conn.version}")
            
            # 간단한 쿼리 테스트
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM DUAL")
            result = cursor.fetchone()
            cursor.close()
            
            if result:
                logger.info(f"✅ 쿼리 테스트 성공: {result}")
            
            conn.close()
            return True
        else:
            logger.error("❌ 데이터베이스 연결 실패")
            return False
            
    except Exception as e:
        logger.error(f"❌ 데이터베이스 연결 오류: {e}")
        logger.error(f"   상세 오류:\n{traceback.format_exc()}")
        return False


def check_offer_data_loading():
    """오퍼 데이터 (item) 로드 확인"""
    logger.info("\n" + "=" * 80)
    logger.info("4. 오퍼 데이터 (Item) 로드 확인")
    logger.info("=" * 80)
    
    try:
        from core.mms_extractor_data import ItemDataLoader
        
        logger.info("ItemDataLoader 초기화 중...")
        loader = ItemDataLoader(offer_info_data_src='db')
        
        logger.info("오퍼 데이터 로드 중...")
        item_pdf_all = loader.load_item_data()
        
        if item_pdf_all is not None and not item_pdf_all.empty:
            logger.info(f"✅ 오퍼 데이터 로드 성공!")
            logger.info(f"   데이터 크기: {item_pdf_all.shape}")
            logger.info(f"   컬럼 목록: {list(item_pdf_all.columns)}")
            
            # 필수 컬럼 확인
            required_columns = ['item_nm', 'item_nm_alias', 'item_id']
            missing_columns = [col for col in required_columns if col not in item_pdf_all.columns]
            
            if missing_columns:
                logger.error(f"❌ 필수 컬럼 누락: {missing_columns}")
                return False
            else:
                logger.info(f"✅ 모든 필수 컬럼 존재: {required_columns}")
            
            # 샘플 데이터 확인
            logger.info("\n   샘플 데이터 (처음 5개):")
            sample_df = item_pdf_all.head(5)
            for idx, row in sample_df.iterrows():
                logger.info(f"   [{idx}] item_nm: {row.get('item_nm', 'N/A')}, "
                          f"item_nm_alias: {row.get('item_nm_alias', 'N/A')}, "
                          f"item_id: {row.get('item_id', 'N/A')}")
            
            # item_nm_alias 유니크 값 확인
            if 'item_nm_alias' in item_pdf_all.columns:
                unique_aliases = item_pdf_all['item_nm_alias'].unique()
                logger.info(f"\n   item_nm_alias 유니크 값 개수: {len(unique_aliases)}")
                logger.info(f"   item_nm_alias 샘플: {list(unique_aliases[:10])}")
            
            return True
        else:
            logger.error("❌ 오퍼 데이터가 비어있습니다!")
            logger.error("   이것이 'Item data is empty! Cannot extract entities.' 오류의 원인입니다.")
            return False
            
    except Exception as e:
        logger.error(f"❌ 오퍼 데이터 로드 오류: {e}")
        logger.error(f"   상세 오류:\n{traceback.format_exc()}")
        return False


def check_program_data_loading():
    """프로그램 데이터 로드 확인"""
    logger.info("\n" + "=" * 80)
    logger.info("5. 프로그램 데이터 로드 확인")
    logger.info("=" * 80)
    
    try:
        from utils.db_utils import load_program_from_database
        
        logger.info("프로그램 데이터 로드 중...")
        pgm_pdf = load_program_from_database()
        
        if pgm_pdf is not None and not pgm_pdf.empty:
            logger.info(f"✅ 프로그램 데이터 로드 성공!")
            logger.info(f"   데이터 크기: {pgm_pdf.shape}")
            logger.info(f"   컬럼 목록: {list(pgm_pdf.columns)}")
            
            # 샘플 데이터
            if 'pgm_nm' in pgm_pdf.columns:
                sample_pgms = pgm_pdf['pgm_nm'].dropna().head(5).tolist()
                logger.info(f"   프로그램명 샘플: {sample_pgms}")
            
            return True
        else:
            logger.warning("⚠️  프로그램 데이터가 비어있습니다.")
            return False
            
    except Exception as e:
        logger.error(f"❌ 프로그램 데이터 로드 오류: {e}")
        logger.error(f"   상세 오류:\n{traceback.format_exc()}")
        return False


def check_org_data_loading():
    """조직 데이터 로드 확인"""
    logger.info("\n" + "=" * 80)
    logger.info("6. 조직 데이터 로드 확인")
    logger.info("=" * 80)
    
    try:
        from utils.db_utils import load_org_from_database
        
        logger.info("조직 데이터 로드 중...")
        org_pdf = load_org_from_database()
        
        if org_pdf is not None and not org_pdf.empty:
            logger.info(f"✅ 조직 데이터 로드 성공!")
            logger.info(f"   데이터 크기: {org_pdf.shape}")
            logger.info(f"   컬럼 목록: {list(org_pdf.columns)}")
            
            # 샘플 데이터
            if 'item_nm' in org_pdf.columns:
                sample_orgs = org_pdf['item_nm'].dropna().head(5).tolist()
                logger.info(f"   조직명 샘플: {sample_orgs}")
            
            return True
        else:
            logger.warning("⚠️  조직 데이터가 비어있습니다.")
            return False
            
    except Exception as e:
        logger.error(f"❌ 조직 데이터 로드 오류: {e}")
        logger.error(f"   상세 오류:\n{traceback.format_exc()}")
        return False


def check_database_config():
    """데이터베이스 설정 확인"""
    logger.info("\n" + "=" * 80)
    logger.info("7. 데이터베이스 설정 확인")
    logger.info("=" * 80)
    
    try:
        from config.settings import DATABASE_CONFIG
        
        logger.info("✅ DATABASE_CONFIG 로드 성공")
        
        # 오퍼 테이블 쿼리 확인
        offer_query = DATABASE_CONFIG.get_offer_table_query("1=1")
        logger.info(f"\n   오퍼 테이블 쿼리 샘플:")
        logger.info(f"   {offer_query[:200]}...")
        
        # 프로그램 테이블 쿼리 확인
        program_query = DATABASE_CONFIG.get_program_table_query("1=1")
        logger.info(f"\n   프로그램 테이블 쿼리 샘플:")
        logger.info(f"   {program_query[:200]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 데이터베이스 설정 확인 오류: {e}")
        logger.error(f"   상세 오류:\n{traceback.format_exc()}")
        return False


def main():
    """메인 진단 함수"""
    logger.info("\n" + "=" * 80)
    logger.info("DB 연결 진단 스크립트 시작")
    logger.info("=" * 80)
    
    results = {}
    
    # 1. 환경 변수 확인
    results['env_vars'] = check_environment_variables()
    
    # 2. cx_Oracle 모듈 확인
    results['cx_oracle'] = check_cx_oracle()
    
    # 3. 데이터베이스 연결 확인
    if results['env_vars'] and results['cx_oracle']:
        results['db_connection'] = check_database_connection()
    else:
        logger.warning("\n⚠️  환경 변수 또는 cx_Oracle 모듈 문제로 DB 연결 테스트를 건너뜁니다.")
        results['db_connection'] = False
    
    # 4. 데이터베이스 설정 확인
    results['db_config'] = check_database_config()
    
    # 5. 오퍼 데이터 로드 확인
    if results['db_connection']:
        results['offer_data'] = check_offer_data_loading()
    else:
        logger.warning("\n⚠️  DB 연결 실패로 오퍼 데이터 로드 테스트를 건너뜁니다.")
        results['offer_data'] = False
    
    # 6. 프로그램 데이터 로드 확인
    if results['db_connection']:
        results['program_data'] = check_program_data_loading()
    else:
        logger.warning("\n⚠️  DB 연결 실패로 프로그램 데이터 로드 테스트를 건너뜁니다.")
        results['program_data'] = False
    
    # 7. 조직 데이터 로드 확인
    if results['db_connection']:
        results['org_data'] = check_org_data_loading()
    else:
        logger.warning("\n⚠️  DB 연결 실패로 조직 데이터 로드 테스트를 건너뜁니다.")
        results['org_data'] = False
    
    # 최종 결과 요약
    logger.info("\n" + "=" * 80)
    logger.info("진단 결과 요약")
    logger.info("=" * 80)
    
    for check_name, result in results.items():
        status = "✅ 통과" if result else "❌ 실패"
        logger.info(f"{check_name:20s}: {status}")
    
    # 문제 해결 가이드
    if not results.get('offer_data', False):
        logger.info("\n" + "=" * 80)
        logger.info("🔧 문제 해결 가이드")
        logger.info("=" * 80)
        logger.info("\n'Item data is empty!' 오류의 가능한 원인:")
        logger.info("1. DB 쿼리가 빈 결과를 반환하는 경우")
        logger.info("   - WHERE 조건이 너무 엄격한지 확인")
        logger.info("   - 테이블에 실제 데이터가 있는지 확인")
        logger.info("\n2. 컬럼 매핑 문제")
        logger.info("   - DB 컬럼명과 코드에서 기대하는 컬럼명이 일치하는지 확인")
        logger.info("   - 대소문자 변환이 올바르게 되는지 확인")
        logger.info("\n3. LOB 데이터 처리 문제")
        logger.info("   - CLOB/BLOB 컬럼이 제대로 로드되는지 확인")
        logger.info("   - outputtypehandler가 올바르게 설정되었는지 확인")
        logger.info("\n4. 네트워크 또는 권한 문제")
        logger.info("   - DB 사용자가 테이블에 대한 SELECT 권한이 있는지 확인")
        logger.info("   - 방화벽이나 네트워크 설정 확인")
    
    logger.info("\n" + "=" * 80)
    logger.info("진단 완료")
    logger.info("=" * 80)
    
    # 종료 코드 반환
    if all(results.values()):
        logger.info("\n✅ 모든 진단 항목 통과!")
        return 0
    else:
        logger.error("\n❌ 일부 진단 항목 실패. 위의 오류 메시지를 확인하세요.")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
