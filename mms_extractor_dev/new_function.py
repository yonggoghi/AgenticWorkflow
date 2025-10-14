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
        stored_prompts = get_stored_prompts_from_thread()
        
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
