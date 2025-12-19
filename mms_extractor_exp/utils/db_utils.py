"""
Database Utilities - Oracle DB 연결 및 데이터 로딩
=================================================

Oracle 데이터베이스 연결 및 데이터 로딩 유틸리티 함수들을 제공합니다.
"""

import os
import logging
import traceback
import time
from contextlib import contextmanager
import pandas as pd

try:
    import cx_Oracle
except ImportError:
    cx_Oracle = None
    logging.warning("cx_Oracle 모듈을 찾을 수 없습니다. DB 기능이 비활성화됩니다.")

logger = logging.getLogger(__name__)


def get_database_connection():
    """
    Oracle 데이터베이스 연결 생성
    
    Returns:
        cx_Oracle.Connection: 데이터베이스 연결 객체
        
    Raises:
        ValueError: 환경 변수가 누락된 경우
        cx_Oracle.DatabaseError: DB 연결 실패 시
    """
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
def database_connection():
    """
    데이터베이스 연결 context manager
    
    Yields:
        cx_Oracle.Connection: 데이터베이스 연결 객체
    """
    conn = None
    start_time = time.time()
    try:
        logger.info("데이터베이스 연결 context manager 시작")
        conn = get_database_connection()
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


def load_program_from_database() -> pd.DataFrame:
    """
    데이터베이스에서 프로그램 분류 정보 로드
    
    Returns:
        pd.DataFrame: 프로그램 분류 정보 (pgm_id, pgm_nm, clue_tag)
        
    Raises:
        Exception: DB 로드 실패 시
    """
    try:
        logger.info("=== 데이터베이스에서 프로그램 분류 정보 로드 시작 ===")
        
        # Import DATABASE_CONFIG
        from config.settings import DATABASE_CONFIG
        
        with database_connection() as conn:
            # 프로그램 분류 정보 쿼리
            where_clause = """DEL_YN = 'N' 
                     AND APRV_OP_RSLT_CD = 'APPR'
                     AND EXPS_YN = 'Y'
                     AND CMPGN_PGM_NUM like '2025%' 
                     AND RMK is not null"""
            sql = DATABASE_CONFIG.get_program_table_query(where_clause)
            
            logger.info(f"실행할 SQL: {sql}")
            
            pgm_pdf = pd.read_sql(sql, conn)
            logger.info(f"DB에서 로드된 프로그램 데이터 크기: {pgm_pdf.shape}")
            logger.info(f"DB에서 로드된 프로그램 컬럼들: {list(pgm_pdf.columns)}")
            
            # 컬럼명 소문자 변환
            original_columns = list(pgm_pdf.columns)
            pgm_pdf = pgm_pdf.rename(columns={c:c.lower() for c in pgm_pdf.columns})
            logger.info(f"프로그램 컬럼명 변환: {dict(zip(original_columns, pgm_pdf.columns))}")
            
            # LOB 데이터가 있는 경우를 대비해 데이터 강제 로드
            if not pgm_pdf.empty:
                try:
                    # DataFrame의 모든 데이터를 메모리로 강제 로드
                    _ = pgm_pdf.values  # 모든 데이터 접근하여 LOB 로드 유도
                    
                    # 프로그램 데이터 샘플 확인
                    if 'pgm_nm' in pgm_pdf.columns:
                        sample_pgms = pgm_pdf['pgm_nm'].dropna().head(3).tolist()
                        logger.info(f"프로그램명 샘플: {sample_pgms}")
                    
                    if 'clue_tag' in pgm_pdf.columns:
                        sample_clues = pgm_pdf['clue_tag'].dropna().head(3).tolist()
                        logger.info(f"클루 태그 샘플: {sample_clues}")
                        
                    logger.info(f"데이터베이스에서 프로그램 분류 정보 로드 완료: {len(pgm_pdf)}개")
                except Exception as load_error:
                    logger.error(f"프로그램 데이터 강제 로드 중 오류: {load_error}")
                    raise
            else:
                logger.warning("로드된 프로그램 데이터가 비어있습니다!")
            
            return pgm_pdf
        
    except Exception as e:
        logger.error(f"프로그램 분류 정보 데이터베이스 로드 실패: {e}")
        logger.error(f"오류 상세: {traceback.format_exc()}")
        # 빈 데이터로 fallback
        return pd.DataFrame(columns=['pgm_nm', 'clue_tag', 'pgm_id'])


def load_org_from_database() -> pd.DataFrame:
    """
    데이터베이스에서 조직 정보 로드 (ITEM_DMN='R')
    
    Returns:
        pd.DataFrame: 조직 정보 (item_nm, item_id, item_desc, item_dmn)
        
    Raises:
        Exception: DB 로드 실패 시
    """
    try:
        logger.info("데이터베이스 연결 시도 중...")
        
        # Import DATABASE_CONFIG
        from config.settings import DATABASE_CONFIG
        
        with database_connection() as conn:
            sql = DATABASE_CONFIG.get_offer_table_query("ITEM_DMN='R'")
            logger.info(f"실행할 SQL: {sql}")
            
            org_pdf = pd.read_sql(sql, conn)
            logger.info(f"DB에서 로드된 조직 데이터 크기: {org_pdf.shape}")
            logger.info(f"DB 조직 데이터 컬럼들: {list(org_pdf.columns)}")
            
            # 컬럼명 매핑 및 소문자 변환
            original_columns = list(org_pdf.columns)
            logger.info(f"DB 조직 데이터 원본 컬럼들: {original_columns}")
            
            # 조직 데이터를 위한 컬럼 매핑 (동일한 테이블이지만 사용 목적이 다름)
            column_mapping = {c: c.lower() for c in org_pdf.columns}
            
            # 조직 데이터는 item 테이블과 동일한 스키마를 사용하므로 컬럼명 그대로 사용
            # ITEM_NM -> item_nm, ITEM_ID -> item_id, ITEM_DESC -> item_desc 등
            
            org_pdf = org_pdf.rename(columns=column_mapping)
            logger.info(f"DB 모드 조직 컬럼명 매핑 완료: {dict(zip(original_columns, org_pdf.columns))}")
            logger.info(f"DB 모드 조직 최종 컬럼들: {list(org_pdf.columns)}")
            
            # 데이터 샘플 확인 및 컬럼 존재 여부 검증
            if not org_pdf.empty:
                logger.info(f"DB 모드 조직 데이터 최종 크기: {org_pdf.shape}")
                
                # 필수 컬럼 존재 여부 확인
                required_columns = ['item_nm', 'item_id']
                missing_columns = [col for col in required_columns if col not in org_pdf.columns]
                if missing_columns:
                    logger.error(f"DB 모드 조직 데이터에서 필수 컬럼 누락: {missing_columns}")
                    logger.error(f"사용 가능한 컬럼들: {list(org_pdf.columns)}")
                else:
                    logger.info("모든 필수 조직 컬럼이 존재합니다.")
                
                # 샘플 데이터 확인
                if 'item_nm' in org_pdf.columns:
                    sample_orgs = org_pdf['item_nm'].dropna().head(5).tolist()
                    logger.info(f"DB 모드 조직명 샘플: {sample_orgs}")
                else:
                    logger.error("item_nm 컬럼이 없어 샘플을 표시할 수 없습니다.")
                    # 전체 데이터 샘플 표시
                    sample_data = org_pdf.head(3).to_dict('records')
                    logger.info(f"DB 모드 조직 데이터 샘플: {sample_data}")
            else:
                logger.warning("DB에서 로드된 조직 데이터가 비어있습니다!")
            
            logger.info(f"DB에서 조직 데이터 로드 성공: {len(org_pdf)}개 조직")
            
            return org_pdf
        
    except Exception as e:
        logger.error(f"DB에서 조직 데이터 로드 실패: {e}")
        logger.error(f"DB 조직 로드 오류 상세: {traceback.format_exc()}")
        
        # 빈 DataFrame으로 fallback (조직 데이터에 필요한 컬럼들 포함)
        return pd.DataFrame(columns=['item_nm', 'item_id', 'item_desc', 'item_dmn'])
