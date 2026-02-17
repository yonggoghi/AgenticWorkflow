#!/usr/bin/env python
"""
CLI Interface for MMS Extractor
================================

Command-line interface for the MMS Extractor system.
"""

import argparse
import sys
import os
# Add parent directory to path to allow imports from core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import logging
import logging.handlers
import time
import traceback
from pathlib import Path
from core.mms_extractor import MMSExtractor, process_message_worker, process_messages_batch, save_result_to_mongodb_if_enabled

# Configure logging with console and file handlers
log_dir = Path(__file__).parent.parent / 'logs'
log_dir.mkdir(exist_ok=True)

root_logger = logging.getLogger()
if not root_logger.handlers:
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / 'cli.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=3,
        encoding='utf-8'
    )
    
    # Console handler for terminal output
    console_handler = logging.StreamHandler()
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

logger = logging.getLogger(__name__)


def main():
    """
    커맨드라인에서 실행할 때의 메인 함수
    다양한 옵션을 통해 추출기 설정을 변경할 수 있습니다.
    
    사용법:
    # 단일 메시지 처리 (멀티스레드)
    python cli.py --message "광고 메시지" --extract-entity-dag
    
    # 배치 처리 (멀티프로세스)
    python cli.py --batch-file messages.txt --max-workers 4 --extract-entity-dag
    
    # 데이터베이스 모드로 배치 처리
    python cli.py --batch-file messages.txt --offer-data-source db --max-workers 8
    
    # MongoDB에 결과 저장
    python cli.py --message "광고 메시지" --save-to-mongodb --extract-entity-dag
    """
    
    parser = argparse.ArgumentParser(description='MMS 광고 텍스트 추출기 - 개선된 버전')
    parser.add_argument('--message', type=str, help='테스트할 메시지')
    parser.add_argument('--batch-file', type=str, help='배치 처리할 메시지가 담긴 파일 경로 (한 줄에 하나씩)')
    parser.add_argument('--max-workers', type=int, help='배치 처리 시 최대 워커 수 (기본값: CPU 코어 수)')
    parser.add_argument('--offer-data-source', choices=['local', 'db'], default='db',
                       help='데이터 소스 (local: CSV 파일, db: 데이터베이스)')
    parser.add_argument('--product-info-extraction-mode', choices=['nlp', 'llm', 'rag'], default='llm',
                       help='Main prompt에서 상품 정보 추출 모드 (nlp: 형태소분석, llm: LLM 기반, rag: 검색증강생성)')
    parser.add_argument('--entity-matching-mode', choices=['logic', 'llm'], default='llm',
                       help='엔티티 매칭 모드 (logic: 로직 기반, llm: LLM 기반)')
    parser.add_argument('--llm-model', choices=['gem', 'ax', 'cld', 'gen', 'gpt', 'opus'], default='ax',
                       help='메인 프롬프트에 사용할 LLM 모델 (gem: Gemma, ax: ax, cld: Claude, gen: Gemini, gpt: GPT, opus: Claude Opus)')
    parser.add_argument('--entity-llm-model', choices=['gem', 'ax', 'cld', 'gen', 'gpt', 'opus'], default='ax',
                       help='엔티티 추출에 사용할 LLM 모델 (gem: Gemma, ax: ax, cld: Claude, gen: Gemini, gpt: GPT, opus: Claude Opus)')
    parser.add_argument('--entity-extraction-context-mode', choices=['dag', 'pairing', 'none', 'ont', 'typed', 'kg'], default='dag',
                       help='엔티티 추출 컨텍스트 모드 (dag: DAG 컨텍스트, pairing: PAIRING 컨텍스트, none: 컨텍스트 없음, ont: 온톨로지 기반 추출, typed: 6-type 엔티티 추출, kg: Knowledge Graph 기반 역할 분류)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO',
                       help='로그 레벨 설정')
    parser.add_argument('--message-id', type=str, default='#',
                       help='메시지 식별자 (기본값: #)')
    parser.add_argument('--skip-entity-extraction', action='store_true', default=False,
                       help='Kiwi + fuzzy matching 기반 엔티티 사전추출 스킵 (Step 2)')
    parser.add_argument('--no-external-candidates', action='store_true', default=True,
                       help='Step 7 매칭 시 외부 후보 엔티티(Kiwi+LLM) 주입 비활성화 (기본: True)')
    parser.add_argument('--extract-entity-dag', action='store_true', default=False, help='Entity DAG extraction (default: False)')
    parser.add_argument('--save-to-mongodb', action='store_true', default=True, 
                       help='추출 결과를 MongoDB에 저장 (utils/mongodb_utils.py 필요)')
    parser.add_argument('--save-batch-results', action='store_true', default=False,
                       help='배치 처리 결과를 JSON 파일로 저장 (results/ 디렉토리에 저장)')
    parser.add_argument('--test-mongodb', action='store_true', default=False,
                       help='MongoDB 연결 테스트만 수행하고 종료')
    parser.add_argument('--extraction-engine', choices=['default', 'langextract'], default='default',
                       help='추출 엔진 선택 (default: 11-step pipeline, langextract: Google langextract 기반)')
    parser.add_argument('--num-cand-pgms', type=int, default=None,
                       help='프로그램 후보 개수 (기본값: config의 num_candidate_programs=15)')
    parser.add_argument('--num-select-pgms', type=int, default=None,
                       help='LLM이 최종 선정할 프로그램 수 (기본값: config의 num_select_programs=1)')

    args = parser.parse_args()
    
    # 로그 레벨 설정 - 루트 로거와 모든 핸들러에 적용
    log_level = getattr(logging, args.log_level)
    root_logger.setLevel(log_level)
    for handler in root_logger.handlers:
        handler.setLevel(log_level)

    
    # MongoDB 연결 테스트만 수행하는 경우
    if args.test_mongodb:
        try:
            from utils.mongodb_utils import test_mongodb_connection
        except ImportError:
            print("❌ MongoDB 유틸리티를 찾을 수 없습니다.")
            print("utils/mongodb_utils.py 파일과 pymongo 패키지를 확인하세요.")
            exit(1)
        
        print("🔌 MongoDB 연결 테스트 중...")
        if test_mongodb_connection():
            print("✅ MongoDB 연결 성공!")
            exit(0)
        else:
            print("❌ MongoDB 연결 실패!")
            print("MongoDB 서버가 실행 중인지 확인하세요.")
            exit(1)
    
    # When using langextract engine, force entity_extraction_context_mode to 'typed'
    entity_extraction_context_mode = args.entity_extraction_context_mode
    if args.extraction_engine == 'langextract':
        entity_extraction_context_mode = 'typed'
        logger.info("langextract 엔진 선택: entity_extraction_context_mode를 'typed'로 강제 설정")

    try:
                # 추출기 초기화
        logger.info("MMS 추출기 초기화 중...")
        extractor = MMSExtractor(
            offer_info_data_src=args.offer_data_source,
            product_info_extraction_mode=args.product_info_extraction_mode,
            entity_extraction_mode=args.entity_matching_mode,
            llm_model=args.llm_model,
            entity_llm_model=args.entity_llm_model,
            extract_entity_dag=args.extract_entity_dag,
            entity_extraction_context_mode=entity_extraction_context_mode,
            skip_entity_extraction=args.skip_entity_extraction,
            use_external_candidates=not args.no_external_candidates,
            extraction_engine=args.extraction_engine,
            num_cand_pgms=args.num_cand_pgms,
            num_select_pgms=args.num_select_pgms,
        )
        
        # 배치 처리 또는 단일 메시지 처리
        if args.batch_file:
            # 배치 파일에서 메시지들 로드
            logger.info(f"배치 파일에서 메시지 로드: {args.batch_file}")
            try:
                with open(args.batch_file, 'r', encoding='utf-8') as f:
                    lines = [line.strip() for line in f if line.strip()]
                
                # JSON Lines 형식인지 확인 (첫 줄이 JSON인지 체크)
                messages = []
                message_ids = []
                is_jsonl = False
                
                if lines and lines[0].startswith('{'):
                    # JSON Lines 형식
                    is_jsonl = True
                    for idx, line in enumerate(lines):
                        try:
                            data = json.loads(line)
                            if isinstance(data, dict):
                                messages.append(data.get('message', ''))
                                message_ids.append(data.get('message_id', f'batch_{idx}'))
                            else:
                                messages.append(str(data))
                                message_ids.append(f'batch_{idx}')
                        except json.JSONDecodeError:
                            # JSON 파싱 실패 시 일반 텍스트로 처리
                            messages.append(line)
                            message_ids.append(f'batch_{idx}')
                else:
                    # 일반 텍스트 형식
                    messages = lines
                    message_ids = [f'batch_{idx}' for idx in range(len(messages))]
                
                logger.info(f"로드된 메시지 수: {len(messages)}개 (형식: {'JSON Lines' if is_jsonl else '일반 텍스트'})")
                
                # 배치 처리 실행 (message_id와 함께)
                results = []
                for message, message_id in zip(messages, message_ids):
                    if args.extract_entity_dag:
                        result = process_message_worker(extractor, message, args.extract_entity_dag, message_id)
                    else:
                        result = extractor.process_message(message, message_id=message_id)
                    results.append(result)
                
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
                    extracted = result.get('ext_result', {})
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
                successful = len([r for r in results if not r.get('error') and r.get('ext_result')])
                failed = len(results) - successful
                print(f"\n📊 배치 처리 통계")
                print(f"✅ 성공: {successful}개")
                print(f"❌ 실패: {failed}개")
                print(f"📈 성공률: {(successful/len(results)*100):.1f}%")
                
                # 결과를 JSON 파일로 저장 (옵션이 활성화된 경우만)
                if args.save_batch_results:
                    # results 디렉토리 생성
                    results_dir = Path(__file__).parent.parent / 'results'
                    results_dir.mkdir(exist_ok=True)

                    # 원본 메시지를 결과에 추가
                    results_with_raw = []
                    for i, result in enumerate(results):
                        result_with_raw = result.copy()
                        if i < len(messages):
                            result_with_raw['raw_message'] = messages[i]
                        results_with_raw.append(result_with_raw)

                    output_file = results_dir / f"batch_results_{int(time.time())}.json"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(results_with_raw, f, indent=4, ensure_ascii=False)
                    print(f"💾 결과 저장: {output_file}")
                else:
                    logger.info("💾 배치 결과 JSON 파일 저장 생략 (--save-batch-results 옵션으로 활성화 가능)")
                
            except FileNotFoundError:
                logger.error(f"배치 파일을 찾을 수 없습니다: {args.batch_file}")
                exit(1)
            except Exception as e:
                logger.error(f"배치 파일 처리 실패: {e}")
                exit(1)
        
        else:
            # 단일 메시지 처리
            test_message = args.message if args.message else """
[SK텔레콤] 공식인증대리점 혜택 안내드립니다.	(광고)[SKT] 공식인증대리점 혜택 안내__고객님, 안녕하세요._SK텔레콤 공식인증대리점에서 상담받고 다양한 혜택을 누려 보세요.__■ 공식인증대리점 혜택_- T끼리 온가족할인, 선택약정으로 통신 요금 최대 55% 할인_- 갤럭시 폴더블/퀀텀, 아이폰 등 기기 할인 상담__■ T 멤버십 고객 감사제 안내_- 2025년 12월까지 매달 Big 3 제휴사 릴레이 할인(10일 단위)__궁금한 점이 있으면 가까운 T 월드 매장에 방문하거나 전화로 문의해 주세요.__▶ 가까운 매장 찾기: https://tworldfriends.co.kr/h/B11109__■ 문의: SKT 고객센터(1558, 무료)__SKT와 함께해 주셔서 감사합니다.__무료 수신거부 1504

"""
            
            if args.extract_entity_dag:
                logger.info("DAG 추출과 함께 병렬 처리 시작")
                result = process_message_worker(extractor, test_message, args.extract_entity_dag, args.message_id)
            else:
                result = extractor.process_message(test_message, args.message_id)
            if args.save_to_mongodb:
                print("\n📄 MongoDB 저장 중...")
                args.processing_mode = 'single'
                saved_id = save_result_to_mongodb_if_enabled(test_message, result, args, extractor)
                if saved_id:
                    print("📄 MongoDB 저장 완료!")

            
            extracted_result = result.get('ext_result', {})
        
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
