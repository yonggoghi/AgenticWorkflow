"""
Workflow Core - 워크플로우 프레임워크 핵심 컴포넌트
=================================================

📋 개요
-------
이 모듈은 MMS Extractor의 처리 로직을 구조화된 워크플로우로 실행하기 위한
핵심 인프라를 제공합니다. 각 처리 단계를 독립적인 Step으로 분리하여
유지보수성, 테스트 용이성, 확장성을 향상시킵니다.

🔗 의존성
---------
**사용하는 모듈:**
- `logging`: 워크플로우 실행 로깅
- `time`: 단계별 실행 시간 측정
- `pandas`: 데이터 전달 (WorkflowState 내부)
- `abc`: 추상 베이스 클래스 정의
- `dataclasses`: 타입 안전 상태 관리

**사용되는 곳:**
- `core.mms_workflow_steps`: 각 워크플로우 단계 구현
- `core.mms_extractor`: MMSExtractor.process_message()에서 워크플로우 실행

🏗️ 아키텍처
-----------
```mermaid
graph TB
    subgraph "Workflow Core"
        WE[WorkflowEngine]
        WS[WorkflowState]
        WSt[WorkflowStep Abstract]
    end
    
    subgraph "Workflow Steps"
        S1[InputValidationStep]
        S2[EntityExtractionStep]
        S3[ProgramClassificationStep]
        S4[ContextPreparationStep]
        S5[LLMExtractionStep]
        S6[ResponseParsingStep]
        S7[ResultConstructionStep]
        S8[ValidationStep]
    end
    
    subgraph "External Services"
        ER[EntityRecognizer]
        PC[ProgramClassifier]
        RB[ResultBuilder]
    end
    
    WE -->|manages| WS
    WE -->|executes| S1
    S1 -->|updates| WS
    S1 --> S2
    S2 -->|uses| ER
    S2 --> S3
    S3 -->|uses| PC
    S3 --> S4
    S4 --> S5
    S5 --> S6
    S6 --> S7
    S7 -->|uses| RB
    S7 --> S8
    
    WSt -.implements.- S1
    WSt -.implements.- S2
    WSt -.implements.- S3
    WSt -.implements.- S4
    WSt -.implements.- S5
    WSt -.implements.- S6
    WSt -.implements.- S7
    WSt -.implements.- S8
```

🏗️ 주요 컴포넌트
----------------
- **WorkflowState**: 단계 간 데이터 전달 및 상태 관리 (Typed Dataclass)
- **WorkflowStep**: 워크플로우 단계의 추상 베이스 클래스
- **WorkflowEngine**: 워크플로우 실행 오케스트레이터

💡 사용 예시
-----------
```python
from core.workflow_core import WorkflowEngine, WorkflowState
from core.mms_workflow_steps import InputValidationStep, EntityExtractionStep

# 1. 워크플로우 엔진 생성
engine = WorkflowEngine(name="MMS Processing")

# 2. 단계 추가
engine.add_step(InputValidationStep())
engine.add_step(EntityExtractionStep(entity_recognizer))

# 3. 초기 상태 생성
initial_state = WorkflowState(
    mms_msg="광고 메시지 텍스트",
    extractor=extractor_instance,
    message_id="MSG001"
)

# 4. 워크플로우 실행
final_state = engine.run(initial_state)

# 5. 결과 확인
if not final_state.has_error():
    print(f"추출 결과: {final_state.final_result}")
else:
    print(f"에러 발생: {final_state.get_errors()}")
```

📝 데이터 흐름
------------
```
입력 메시지 (mms_msg)
    ↓
WorkflowState 초기화
    ↓
[Step 1] InputValidation → state.msg 설정
    ↓
[Step 2] EntityExtraction → state.entities_from_kiwi, state.cand_item_list 설정
    ↓
[Step 3] ProgramClassification → state.pgm_info 설정
    ↓
[Step 4] ContextPreparation → state.rag_context, state.product_element 설정
    ↓
[Step 5] LLMExtraction → state.result_json_text 설정
    ↓
[Step 6] ResponseParsing → state.json_objects 설정
    ↓
[Step 7] ResultConstruction → state.raw_result, state.final_result 설정
    ↓
[Step 8] Validation → 최종 검증
    ↓
최종 결과 (final_result)
```

📝 참고사항
----------
- WorkflowState는 dataclass로 구현되어 타입 안전성 제공
- 각 Step은 독립적으로 테스트 가능
- 에러 발생 시 자동으로 워크플로우 중단
- 실행 히스토리를 통해 성능 분석 가능
- Backward compatibility를 위해 get/set 메서드 제공

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
    
    데이터 흐름:
        각 필드는 특정 워크플로우 단계에서 설정되며, 이후 단계에서 읽기 전용으로 사용됩니다.
    
    Attributes:
        mms_msg: 원본 MMS 메시지 (초기화 시 설정)
        extractor: MMSExtractor 인스턴스 (초기화 시 설정)
        message_id: 메시지 식별자 (초기화 시 설정, 기본값: '#')
        msg: 검증/정제된 메시지 (InputValidationStep에서 설정)
        entities_from_kiwi: Kiwi로 추출한 엔티티 목록 (EntityExtractionStep에서 설정)
        cand_item_list: 후보 상품 목록 DataFrame (EntityExtractionStep에서 설정)
        extra_item_pdf: 추가 상품 정보 DataFrame (EntityExtractionStep에서 설정)
        pgm_info: 프로그램 분류 정보 (ProgramClassificationStep에서 설정)
        rag_context: RAG 컨텍스트 문자열 (ContextPreparationStep에서 설정)
        product_element: 제품 요소 (ContextPreparationStep에서 설정)
        result_json_text: LLM 응답 JSON 텍스트 (LLMExtractionStep에서 설정)
        json_objects: 파싱된 JSON 객체 (ResponseParsingStep에서 설정)
        raw_result: 원시 추출 결과 (ResultConstructionStep에서 설정)
        final_result: 최종 추출 결과 (ResultConstructionStep에서 설정)
        is_fallback: 폴백 모드 여부 (LLMExtractionStep 또는 ResponseParsingStep에서 설정)
    """
    
    # Input fields (set at initialization)
    mms_msg: str  # 원본 MMS 메시지 텍스트
    extractor: Any  # MMSExtractor instance (avoid circular import)
    message_id: str = "#"  # 메시지 식별자 (기본값: '#')
    
    # Processing fields (set during workflow)
    msg: str = ""  # Validated/trimmed message (set by InputValidationStep)
    entities_from_kiwi: List[str] = field(default_factory=list)  # Kiwi 추출 엔티티 (set by EntityExtractionStep)
    cand_item_list: pd.DataFrame = field(default_factory=pd.DataFrame)  # 후보 상품 목록 (set by EntityExtractionStep)
    extra_item_pdf: pd.DataFrame = field(default_factory=pd.DataFrame)  # 추가 상품 정보 (set by EntityExtractionStep)
    pgm_info: Dict[str, Any] = field(default_factory=dict)  # 프로그램 분류 정보 (set by ProgramClassificationStep)
    rag_context: str = ""  # RAG 컨텍스트 (set by ContextPreparationStep)
    product_element: Optional[Any] = None  # 제품 요소 (set by ContextPreparationStep)
    result_json_text: str = ""  # LLM 응답 JSON (set by LLMExtractionStep)
    json_objects: Dict[str, Any] = field(default_factory=dict)  # 파싱된 JSON (set by ResponseParsingStep)
    raw_result: Dict[str, Any] = field(default_factory=dict)  # 원시 결과 (set by ResultConstructionStep)
    final_result: Dict[str, Any] = field(default_factory=dict)  # 최종 결과 (set by ResultConstructionStep)
    
    # Entity extraction and matching fields
    extracted_entities: Optional[Dict[str, Any]] = None  # Entity + Context (set by EntityContextExtractionStep)
    matched_products: List[Dict[str, Any]] = field(default_factory=list)  # 매칭된 상품 목록 (set by VocabularyFilteringStep)
    kg_metadata: Optional[Dict[str, Any]] = None  # KG 전체 메타데이터 (set by EntityContextExtractionStep, KG 모드)

    # Control flags
    is_fallback: bool = False  # 폴백 모드 여부 (set by LLMExtractionStep or ResponseParsingStep)
    
    # Internal tracking
    _history: List[Dict[str, Any]] = field(default_factory=list, repr=False)  # 실행 히스토리
    _errors: List[str] = field(default_factory=list, repr=False)  # 에러 목록
    
    # Backward compatibility methods (for gradual migration)
    def get(self, key: str, default: Any = None) -> Any:
        """
        상태에서 값 가져오기 (backward compatible)
        
        Args:
            key: 필드 이름
            default: 기본값
            
        Returns:
            필드 값 또는 기본값
        """
        return getattr(self, key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        상태에 값 저장 (backward compatible)
        
        Args:
            key: 필드 이름
            value: 저장할 값
        """
        setattr(self, key, value)
    
    def has_error(self) -> bool:
        """
        에러 발생 여부 확인
        
        Returns:
            에러가 있으면 True, 없으면 False
        """
        return len(self._errors) > 0
    
    def add_error(self, error: str) -> None:
        """
        에러 추가
        
        Args:
            error: 에러 메시지
        """
        self._errors.append(error)
    
    def get_errors(self) -> List[str]:
        """
        모든 에러 반환
        
        Returns:
            에러 메시지 리스트
        """
        return self._errors
    
    def add_history(self, step_name: str, duration: float, status: str) -> None:
        """
        실행 히스토리 추가
        
        Args:
            step_name: 단계 이름
            duration: 실행 시간 (초)
            status: 실행 상태 ('success' 또는 'failed')
        """
        self._history.append({
            "step": step_name,
            "duration": duration,
            "status": status
        })
    
    def get_history(self) -> List[Dict[str, Any]]:
        """
        실행 히스토리 반환
        
        Returns:
            히스토리 딕셔너리 리스트
        """
        return self._history


class WorkflowStep(ABC):
    """
    워크플로우 단계 추상 베이스 클래스
    
    모든 워크플로우 단계는 이 클래스를 상속받아 execute 메서드를 구현해야 합니다.
    """
    
    def should_execute(self, state: WorkflowState) -> bool:
        """
        단계 실행 여부 결정

        Args:
            state: 현재 워크플로우 상태

        Returns:
            True이면 실행, False이면 스킵
        """
        return True

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

            # 조건부 실행: should_execute()가 False이면 스킵
            if not step.should_execute(state):
                step_duration = time.time() - step_start_time
                state.add_history(step_name, step_duration, "skipped")
                logger.info(f"⏭️ {step_name} 스킵됨 (should_execute=False)")
                continue

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
                status_icon = {"success": "✅", "skipped": "⏭️", "failed": "❌"}.get(entry["status"], "❓")
                logger.info(f"  {status_icon} {entry['step']}: {entry['duration']:.2f}초")
        
        if state.has_error():
            logger.error(f"\n⚠️ 에러 발생: {state.get_errors()}")
        
        return state
