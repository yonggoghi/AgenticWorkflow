"""
Workflow Core - 워크플로우 프레임워크 핵심 컴포넌트
=================================================

이 모듈은 MMS Extractor의 처리 로직을 구조화된 워크플로우로 실행하기 위한
핵심 인프라를 제공합니다.

주요 클래스:
- WorkflowState: 단계 간 데이터 전달 및 상태 관리
- WorkflowStep: 워크플로우 단계의 추상 베이스 클래스
- WorkflowEngine: 워크플로우 실행 오케스트레이터
"""

import logging
import time
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class WorkflowState:
    """
    워크플로우 상태 관리 클래스 (Typed Dataclass)
    
    워크플로우의 각 단계 간에 데이터를 전달하고 관리합니다.
    Type-safe 필드로 IDE 지원 및 런타임 타입 체크 향상.
    """
    
    # Input fields (set at initialization)
    mms_msg: str
    extractor: Any  # MMSExtractor instance (avoid circular import)
    message_id: str = "#"  # 메시지 식별자 (기본값: '#')
    
    # Processing fields (set during workflow)
    msg: str = ""  # Validated/trimmed message
    entities_from_kiwi: List[str] = field(default_factory=list)
    cand_item_list: pd.DataFrame = field(default_factory=pd.DataFrame)
    extra_item_pdf: pd.DataFrame = field(default_factory=pd.DataFrame)
    pgm_info: Dict[str, Any] = field(default_factory=dict)
    rag_context: str = ""
    product_element: Optional[Any] = None
    result_json_text: str = ""
    json_objects: Dict[str, Any] = field(default_factory=dict)
    raw_result: Dict[str, Any] = field(default_factory=dict)
    final_result: Dict[str, Any] = field(default_factory=dict)
    
    # Control flags
    is_fallback: bool = False
    
    # Internal tracking
    _history: List[Dict[str, Any]] = field(default_factory=list, repr=False)
    _errors: List[str] = field(default_factory=list, repr=False)
    
    # Backward compatibility methods (for gradual migration)
    def get(self, key: str, default: Any = None) -> Any:
        """상태에서 값 가져오기 (backward compatible)"""
        return getattr(self, key, default)
    
    def set(self, key: str, value: Any) -> None:
        """상태에 값 저장 (backward compatible)"""
        setattr(self, key, value)
    
    def has_error(self) -> bool:
        """에러 발생 여부 확인"""
        return len(self._errors) > 0
    
    def add_error(self, error: str) -> None:
        """에러 추가"""
        self._errors.append(error)
    
    def get_errors(self) -> List[str]:
        """모든 에러 반환"""
        return self._errors
    
    def add_history(self, step_name: str, duration: float, status: str) -> None:
        """실행 히스토리 추가"""
        self._history.append({
            "step": step_name,
            "duration": duration,
            "status": status
        })
    
    def get_history(self) -> List[Dict[str, Any]]:
        """실행 히스토리 반환"""
        return self._history


class WorkflowStep(ABC):
    """
    워크플로우 단계 추상 베이스 클래스
    
    모든 워크플로우 단계는 이 클래스를 상속받아 execute 메서드를 구현해야 합니다.
    """
    
    @abstractmethod
    def execute(self, state: WorkflowState) -> WorkflowState:
        """
        단계 실행 메서드
        
        Args:
            state: 현재 워크플로우 상태
            
        Returns:
            업데이트된 워크플로우 상태
        """
        pass
    
    def name(self) -> str:
        """단계 이름 반환 (로깅용)"""
        return self.__class__.__name__


class WorkflowEngine:
    """
    워크플로우 실행 엔진
    
    등록된 단계들을 순차적으로 실행하고 상태를 관리합니다.
    """
    
    def __init__(self, name: str = "Workflow"):
        """
        Args:
            name: 워크플로우 이름 (로깅용)
        """
        self.name = name
        self.steps: List[WorkflowStep] = []
    
    def add_step(self, step: WorkflowStep) -> None:
        """
        워크플로우 단계 추가
        
        Args:
            step: 추가할 워크플로우 단계
        """
        self.steps.append(step)
        logger.debug(f"Added step: {step.name()} to {self.name}")
    
    def run(self, initial_state: WorkflowState) -> WorkflowState:
        """
        워크플로우 실행
        
        Args:
            initial_state: 초기 상태
            
        Returns:
            최종 상태
        """
        logger.info(f"{'='*60}")
        logger.info(f"🚀 {self.name} 시작")
        logger.info(f"{'='*60}")
        
        state = initial_state
        total_start_time = time.time()
        
        for i, step in enumerate(self.steps, 1):
            step_name = step.name()
            logger.info(f"\n{'='*30} {i}/{len(self.steps)}: {step_name} {'='*30}")
            
            step_start_time = time.time()
            
            try:
                state = step.execute(state)
                step_duration = time.time() - step_start_time
                
                state.add_history(step_name, step_duration, "success")
                logger.info(f"✅ {step_name} 완료 ({step_duration:.2f}초)")
                
                # 에러가 발생한 경우 조기 종료
                if state.has_error():
                    logger.warning(f"⚠️ {step_name}에서 에러 발생, 워크플로우 중단")
                    break
                    
            except Exception as e:
                step_duration = time.time() - step_start_time
                error_msg = f"{step_name} 실패: {str(e)}"
                
                state.add_error(error_msg)
                state.add_history(step_name, step_duration, "failed")
                
                logger.error(f"❌ {error_msg}")
                logger.exception(e)
                
                # 치명적 에러인 경우 중단
                break
        
        total_duration = time.time() - total_start_time
        
        logger.info(f"\n{'='*60}")
        logger.info(f"✅ {self.name} 완료 (총 {total_duration:.2f}초)")
        logger.info(f"{'='*60}")
        
        # 실행 요약
        history = state.get_history()
        if history:
            logger.info("\n📊 실행 요약:")
            for entry in history:
                status_icon = "✅" if entry["status"] == "success" else "❌"
                logger.info(f"  {status_icon} {entry['step']}: {entry['duration']:.2f}초")
        
        if state.has_error():
            logger.error(f"\n⚠️ 에러 발생: {state.get_errors()}")
        
        return state
