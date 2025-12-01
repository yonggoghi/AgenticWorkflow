"""
Prompt Manager - 프롬프트 저장 및 관리
====================================

프롬프트를 스레드 로컬 저장소에 저장하고 검색하는 기능을 제공합니다.
"""

import logging
import threading
from typing import Dict, Any

logger = logging.getLogger(__name__)


class PromptManager:
    """프롬프트 관리 클래스"""
    
    # 프롬프트 타입별 메타데이터
    PROMPT_INFO = {
        "main_extraction": {
            'title': '메인 정보 추출 프롬프트',
            'description': '광고 메시지에서 제목, 목적, 상품, 채널, 프로그램 정보를 추출하는 프롬프트'
        },
        "entity_extraction": {
            'title': '엔티티 추출 프롬프트',
            'description': '메시지에서 상품/서비스 엔티티를 추출하는 프롬프트'
        }
    }
    
    @staticmethod
    def store_prompt_for_preview(prompt: str, prompt_type: str) -> None:
        """
        프롬프트를 미리보기용으로 저장
        
        Args:
            prompt: 저장할 프롬프트 텍스트
            prompt_type: 프롬프트 타입 (main_extraction, entity_extraction 등)
        """
        current_thread = threading.current_thread()
        
        if not hasattr(current_thread, 'stored_prompts'):
            current_thread.stored_prompts = {}
        
        # 프롬프트 메타데이터 가져오기
        info = PromptManager.PROMPT_INFO.get(prompt_type, {
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
        
        # 로깅
        logger.info(f"📝 프롬프트 저장됨: {prompt_key}")
        logger.info(f"📝 프롬프트 길이: {len(prompt):,} 문자")
    
    @staticmethod
    def get_stored_prompts_from_thread() -> Dict[str, Any]:
        """
        현재 스레드에서 저장된 프롬프트 정보를 가져옴
        
        Returns:
            저장된 프롬프트 딕셔너리 (없으면 빈 딕셔너리)
        """
        current_thread = threading.current_thread()
        
        if hasattr(current_thread, 'stored_prompts'):
            return current_thread.stored_prompts
        else:
            return {}
    
    @staticmethod
    def clear_stored_prompts() -> None:
        """현재 스레드의 저장된 프롬프트 초기화"""
        current_thread = threading.current_thread()
        
        if hasattr(current_thread, 'stored_prompts'):
            current_thread.stored_prompts = {}
            logger.debug("프롬프트 저장소 초기화됨")
