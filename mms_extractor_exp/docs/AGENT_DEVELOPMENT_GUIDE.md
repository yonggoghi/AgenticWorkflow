# MMS Extractor - Agent 개발 가이드

## 📋 목차

1. [프로젝트 개요](#프로젝트-개요)
2. [아키텍처 분석](#아키텍처-분석)
3. [Agent 작업 적합성 평가](#agent-작업-적합성-평가)
4. [핵심 컴포넌트 가이드](#핵심-컴포넌트-가이드)
5. [일반적인 개선 작업 패턴](#일반적인-개선-작업-패턴)
6. [개선 권장사항](#개선-권장사항)

---

## 프로젝트 개요

### 목적
MMS 광고 메시지에서 구조화된 정보를 추출하는 AI 기반 시스템

### 핵심 기능
- **엔티티 추출**: 상품, 채널, 프로그램, 매장 정보 추출
- **LLM 기반 분석**: 다중 LLM 모델 지원 (Gemini, Claude, GPT-4, AX)
- **DAG 추출**: 엔티티 간 관계 그래프 생성
- **Workflow 기반**: 모듈화된 8단계 처리 파이프라인

---

## 아키텍처 분석

### 디렉토리 구조

```
mms_extractor_exp/
├── core/                    # 핵심 로직
│   ├── mms_extractor.py    # 메인 추출기 (MMSExtractor 클래스)
│   ├── workflow_core.py    # Workflow 프레임워크
│   ├── mms_workflow_steps.py  # 9개 Workflow 단계 구현
│   └── entity_dag_extractor.py  # DAG 추출 로직
│
├── services/                # 비즈니스 로직 서비스
│   ├── entity_recognizer.py   # 엔티티 인식
│   ├── result_builder.py      # 결과 구성
│   ├── program_classifier.py  # 프로그램 분류
│   ├── store_matcher.py       # 매장 매칭
│   ├── item_data_loader.py    # 상품 데이터 로딩
│   └── schema_transformer.py  # 스키마 변환
│
├── prompts/                 # LLM 프롬프트 관리
│   ├── main_extraction_prompt.py
│   ├── entity_extraction_prompt.py
│   └── dag_extraction_prompt.py
│
├── config/                  # 설정 관리
│   └── settings.py         # 모든 설정 중앙화
│
├── utils/                   # 유틸리티 함수
│   ├── __init__.py         # 유사도 계산, DAG 시각화 등
│   ├── llm_factory.py      # LLM 모델 생성 Factory
│   ├── prompt_utils.py     # 프롬프트 관리
│   └── retry_utils.py      # 재시도 로직
│
├── apps/                    # 실행 인터페이스
│   ├── cli.py              # CLI 인터페이스
│   ├── api.py              # FastAPI 서버
│   └── batch.py            # 배치 처리
│
├── tests/                   # 테스트
│   ├── test_workflow.py
│   └── test_architecture_improvements.py
│
└── docs/                    # 문서
    └── LOGGING_GUIDELINES.md
```

### 핵심 설계 패턴

#### 1. **Workflow Pattern** (핵심!)

```python
# workflow_core.py
WorkflowState (Dataclass)  # 단계 간 데이터 전달
    ↓
WorkflowStep (Abstract)    # 각 처리 단계의 베이스 클래스
    ↓
WorkflowEngine             # 단계들을 순차 실행
```

**9개 Workflow 단계** (`mms_workflow_steps.py`):
1. `InputValidationStep`: 입력 검증
2. `EntityExtractionStep`: 엔티티 추출 (Kiwi/LLM)
3. `ProgramClassificationStep`: 프로그램 분류
4. `ContextPreparationStep`: RAG 컨텍스트 준비
5. `LLMExtractionStep`: LLM 기반 정보 추출
6. `ResponseParsingStep`: LLM 응답 파싱
7. `ResultConstructionStep`: 최종 결과 구성
8. `ValidationStep`: 결과 검증
9. `DAGExtractionStep`: DAG 추출 (선택적)

#### 2. **Service Layer Pattern**

각 서비스가 독립적인 책임을 가짐:
- `EntityRecognizer`: 엔티티 인식 전담
- `ResultBuilder`: 결과 구성 전담
- `ItemDataLoader`: 데이터 로딩 전담 (최근 리팩토링)

#### 3. **Factory Pattern**

- `LLMFactory`: LLM 모델 생성 중앙화
- 순환 의존성 제거

#### 4. **Config-Driven Design**

모든 설정이 `config/settings.py`에 중앙화:
- API 설정 (`APIConfig`)
- 모델 설정 (`ModelConfig`)
- 처리 설정 (`ProcessingConfig`)
- 메타데이터 경로 (`METADATAConfig`)

---

## Agent 작업 적합성 평가

### ✅ 강점 (Agent-Friendly)

#### 1. **명확한 모듈 분리**
- 각 컴포넌트의 역할이 명확함
- 파일명과 클래스명이 직관적
- 단일 책임 원칙 준수

**Agent 관점**: 어떤 파일을 수정해야 할지 쉽게 파악 가능

#### 2. **Workflow 기반 구조**
- 처리 흐름이 명시적으로 정의됨
- 각 단계가 독립적으로 테스트 가능
- 새로운 단계 추가가 용이

**Agent 관점**: 새로운 기능 추가 시 어디에 넣을지 명확

#### 3. **Config 중앙화**
- 모든 설정이 한 곳에 모여있음
- 타입 힌트가 잘 되어있음
- 환경별 설정 변경 용이

**Agent 관점**: 설정 변경 시 한 파일만 수정하면 됨

#### 4. **서비스 독립성**
- 각 서비스가 독립적으로 동작
- 의존성이 명시적 (생성자 주입)
- 테스트 용이

**Agent 관점**: 한 서비스 수정이 다른 서비스에 영향 최소화

#### 5. **타입 힌트**
```python
def process_message(self, message: str, message_id: str = '#') -> Dict[str, Any]:
```
- 함수 시그니처가 명확
- IDE/Agent가 타입 추론 가능

### ⚠️ 개선 필요 (Agent-Challenging)

#### 1. **문서화 부족**

**현재 상태**:
- Docstring은 있지만 간략함
- 각 서비스의 역할 설명 부족
- Workflow 전체 흐름 문서 없음

**Agent 영향**:
- 코드를 읽어야만 동작 이해 가능
- 컨텍스트 파악에 시간 소요

**개선 방안**:
```python
# 현재
class EntityRecognizer:
    """엔티티 인식 서비스"""
    
# 개선
class EntityRecognizer:
    """
    엔티티 인식 서비스
    
    책임:
    - Kiwi 형태소 분석기를 사용한 엔티티 추출
    - LLM 기반 엔티티 추출 (entity_extraction_mode='llm')
    - Fuzzy/Sequence 유사도 기반 상품 매칭
    
    주요 메서드:
    - extract_entities_from_kiwi(): Kiwi 기반 추출
    - extract_entities_by_llm(): LLM 기반 추출
    - match_entities(): 상품 DB와 매칭
    
    사용 예시:
        recognizer = EntityRecognizer(kiwi, item_pdf, stop_words, llm)
        entities = recognizer.extract_entities_from_kiwi(message)
    """
```

#### 2. **테스트 커버리지 부족**

**현재 상태**:
- 기본 Workflow 테스트만 존재
- 각 서비스별 단위 테스트 없음
- 통합 테스트 부족

**Agent 영향**:
- 변경 후 검증이 어려움
- 회귀 테스트 불가능

**개선 방안**:
```python
# tests/test_entity_recognizer.py (필요)
def test_extract_entities_from_kiwi():
    """Kiwi 기반 엔티티 추출 테스트"""
    recognizer = EntityRecognizer(...)
    entities = recognizer.extract_entities_from_kiwi("테스트 메시지")
    assert len(entities) > 0
```

#### 3. **복잡한 의존성 그래프**

**현재 상태**:
```
MMSExtractor
  ├─> EntityRecognizer
  │     ├─> Kiwi
  │     ├─> item_pdf_all
  │     └─> llm_model
  ├─> ResultBuilder
  │     ├─> EntityRecognizer
  │     ├─> StoreMatcher
  │     └─> LLMFactory
  └─> ProgramClassifier
        ├─> emb_model
        └─> pgm_pdf
```

**Agent 영향**:
- 한 컴포넌트 수정 시 영향 범위 파악 어려움
- 순환 의존성 위험

**개선 방안**:
- 의존성 다이어그램 문서화
- 인터페이스 분리 원칙 적용

#### 4. **프롬프트 관리 분산**

**현재 상태**:
- `prompts/` 디렉토리에 분산
- 프롬프트 버전 관리 없음
- 프롬프트 변경 이력 추적 어려움

**Agent 영향**:
- 프롬프트 개선 시 어떤 파일을 수정해야 할지 불명확
- A/B 테스트 어려움

**개선 방안**:
- 프롬프트 버전 관리 시스템
- 프롬프트 성능 메트릭 추적

---

## 핵심 컴포넌트 가이드

### 1. MMSExtractor (core/mms_extractor.py)

**역할**: 전체 추출 프로세스의 오케스트레이터

**주요 메서드**:
```python
__init__(...)                    # 초기화: 데이터 로드, 서비스 생성
process_message(message) -> dict # 메인 처리 함수
_load_data()                     # 데이터 로딩 오케스트레이션
_initialize_llm()                # LLM 초기화
```

**수정 시나리오**:
- 새로운 LLM 모델 추가 → `_initialize_llm()` 수정
- 새로운 데이터 소스 추가 → `_load_data()` 수정
- Workflow 단계 추가 → `__init__`에서 `workflow_engine.add_step()` 호출

### 2. Workflow Steps (core/mms_workflow_steps.py)

**역할**: 각 처리 단계의 구현

**새 단계 추가 방법**:
```python
class NewProcessingStep(WorkflowStep):
    """새로운 처리 단계"""
    
    def __init__(self, dependency1, dependency2):
        self.dep1 = dependency1
        self.dep2 = dependency2
    
    def execute(self, state: WorkflowState) -> WorkflowState:
        """
        단계 실행
        
        Args:
            state: 현재 상태
            
        Returns:
            업데이트된 상태
        """
        logger.info("🚀 새로운 처리 시작")
        
        # 처리 로직
        result = self.dep1.process(state.msg)
        
        # 상태 업데이트
        state.set("new_field", result)
        
        logger.info("✅ 새로운 처리 완료")
        return state
```

**MMSExtractor에 등록**:
```python
# mms_extractor.py __init__
self.workflow_engine.add_step(NewProcessingStep(dep1, dep2))
```

### 3. Services (services/)

**각 서비스의 역할**:

| 서비스 | 역할 | 주요 메서드 |
|--------|------|-------------|
| `EntityRecognizer` | 엔티티 인식 | `extract_entities_from_kiwi()`, `extract_entities_by_llm()` |
| `ResultBuilder` | 결과 구성 | `build_final_result()` |
| `ProgramClassifier` | 프로그램 분류 | `classify()` |
| `StoreMatcher` | 매장 매칭 | `match_store()` |
| `ItemDataLoader` | 상품 데이터 로딩 | `prepare_item_data()` |
| `SchemaTransformer` | 스키마 변환 | `transform_to_item_centric()` |

**서비스 수정 가이드**:
1. 단일 책임 유지
2. 생성자 주입으로 의존성 명시
3. 타입 힌트 필수
4. 에러 처리 포함

### 4. Config (config/settings.py)

**설정 추가 방법**:
```python
@dataclass
class ProcessingConfig:
    # 기존 설정들...
    
    # 새로운 설정 추가
    new_feature_enabled: bool = False
    new_threshold: float = 0.5
    new_model_name: str = "default-model"
```

**사용**:
```python
from config.settings import PROCESSING_CONFIG

if PROCESSING_CONFIG.new_feature_enabled:
    # 새로운 기능 실행
    pass
```

---

## 일반적인 개선 작업 패턴

### 패턴 1: 추출 성능 개선

**시나리오**: 엔티티 추출 정확도 향상

**작업 순서**:
1. **분석**: `services/entity_recognizer.py` 검토
2. **프롬프트 개선**: `prompts/entity_extraction_prompt.py` 수정
3. **임계값 조정**: `config/settings.py`에서 `entity_*_threshold` 조정
4. **테스트**: `tests/test_entity_recognizer.py` 작성/실행
5. **검증**: 샘플 메시지로 정확도 측정

**관련 파일**:
- `services/entity_recognizer.py`
- `prompts/entity_extraction_prompt.py`
- `config/settings.py`
- `core/mms_workflow_steps.py` (EntityExtractionStep)

### 패턴 2: 새로운 엔티티 타입 추가

**시나리오**: "이벤트" 엔티티 타입 추가

**작업 순서**:
1. **스키마 정의**: 출력 JSON 구조에 `event` 필드 추가
2. **프롬프트 수정**: `prompts/main_extraction_prompt.py`에 이벤트 추출 지시 추가
3. **파싱 로직**: `core/mms_workflow_steps.py` ResponseParsingStep 수정
4. **결과 구성**: `services/result_builder.py`에 이벤트 처리 로직 추가
5. **테스트**: 이벤트 포함 메시지로 테스트

**관련 파일**:
- `prompts/main_extraction_prompt.py`
- `core/mms_workflow_steps.py` (ResponseParsingStep, ResultConstructionStep)
- `services/result_builder.py`

### 패턴 3: 새로운 LLM 모델 추가

**시나리오**: "GPT-4o-mini" 모델 추가

**작업 순서**:
1. **설정 추가**: `config/settings.py` ModelConfig에 모델명 추가
2. **Factory 수정**: `utils/llm_factory.py`에 모델 매핑 추가
3. **초기화**: `core/mms_extractor.py` `_initialize_llm()` 확인 (Factory 사용하므로 수정 불필요)
4. **테스트**: CLI로 새 모델 테스트

**관련 파일**:
- `config/settings.py`
- `utils/llm_factory.py`

### 패턴 4: Workflow 단계 추가

**시나리오**: "감정 분석" 단계 추가

**작업 순서**:
1. **단계 구현**: `core/mms_workflow_steps.py`에 `SentimentAnalysisStep` 클래스 추가
2. **서비스 생성** (선택): `services/sentiment_analyzer.py` 생성
3. **등록**: `core/mms_extractor.py` `__init__`에서 단계 등록
4. **상태 필드**: `core/workflow_core.py` WorkflowState에 `sentiment` 필드 추가
5. **테스트**: Workflow 전체 테스트

**관련 파일**:
- `core/mms_workflow_steps.py`
- `core/mms_extractor.py`
- `core/workflow_core.py`
- `services/sentiment_analyzer.py` (신규)

---

## 개선 권장사항

### 우선순위 1: 문서화 강화

#### 1.1 아키텍처 다이어그램 추가

**생성할 문서**:
```markdown
# docs/ARCHITECTURE.md

## 시스템 아키텍처

### 컴포넌트 다이어그램
[Mermaid 다이어그램]

### 데이터 흐름
[Workflow 단계별 데이터 흐름 다이어그램]

### 의존성 그래프
[서비스 간 의존성 다이어그램]
```

#### 1.2 각 서비스에 상세 Docstring 추가

**템플릿**:
```python
class ServiceName:
    """
    [서비스 이름] - [한 줄 설명]
    
    책임:
    - [책임 1]
    - [책임 2]
    
    의존성:
    - [의존성 1]: [용도]
    - [의존성 2]: [용도]
    
    주요 메서드:
    - method1(): [설명]
    - method2(): [설명]
    
    사용 예시:
        service = ServiceName(dep1, dep2)
        result = service.method1(input)
    
    주의사항:
    - [주의사항 1]
    - [주의사항 2]
    """
```

#### 1.3 Workflow 가이드 문서

**생성할 문서**:
```markdown
# docs/WORKFLOW_GUIDE.md

## Workflow 단계 상세 가이드

### 1. InputValidationStep
- **목적**: 입력 검증 및 전처리
- **입력**: state.mms_msg
- **출력**: state.msg (검증된 메시지)
- **에러 처리**: 빈 메시지 → fallback

### 2. EntityExtractionStep
...
```

### 우선순위 2: 테스트 커버리지 확대

#### 2.1 서비스별 단위 테스트

**생성할 테스트**:
```
tests/
├── test_entity_recognizer.py
├── test_result_builder.py
├── test_program_classifier.py
├── test_store_matcher.py
├── test_item_data_loader.py
└── test_llm_factory.py
```

#### 2.2 통합 테스트

```python
# tests/test_integration.py
def test_full_pipeline():
    """전체 파이프라인 통합 테스트"""
    extractor = MMSExtractor()
    
    test_cases = [
        {"message": "...", "expected_products": [...]},
        {"message": "...", "expected_channels": [...]},
    ]
    
    for case in test_cases:
        result = extractor.process_message(case["message"])
        assert result["ext_result"]["product"] == case["expected_products"]
```

### 우선순위 3: Agent-Friendly 개선

#### 3.1 타입 힌트 완성도 향상

**현재**:
```python
def process(self, data):  # 타입 힌트 없음
    return result
```

**개선**:
```python
def process(self, data: Dict[str, Any]) -> ProcessingResult:
    return result
```

#### 3.2 에러 메시지 개선

**현재**:
```python
except Exception as e:
    logger.error(f"처리 실패: {e}")
```

**개선**:
```python
except ValueError as e:
    logger.error(f"입력 값 오류: {e}. 입력 데이터: {data}")
except KeyError as e:
    logger.error(f"필수 키 누락: {e}. 사용 가능한 키: {data.keys()}")
except Exception as e:
    logger.error(f"예상치 못한 오류: {e}")
    logger.error(f"상세: {traceback.format_exc()}")
```

#### 3.3 설정 검증 추가

```python
# config/settings.py
@dataclass
class ProcessingConfig:
    entity_fuzzy_threshold: float = 0.5
    
    def __post_init__(self):
        """설정 검증"""
        if not 0.0 <= self.entity_fuzzy_threshold <= 1.0:
            raise ValueError(
                f"entity_fuzzy_threshold는 0.0-1.0 사이여야 합니다. "
                f"현재 값: {self.entity_fuzzy_threshold}"
            )
```

### 우선순위 4: 모니터링 및 메트릭

#### 4.1 성능 메트릭 추가

```python
# utils/metrics.py (신규)
class PerformanceMetrics:
    """성능 메트릭 수집"""
    
    def __init__(self):
        self.metrics = {
            "total_messages": 0,
            "successful": 0,
            "failed": 0,
            "avg_processing_time": 0.0,
            "step_times": {}
        }
    
    def record_processing(self, duration: float, success: bool):
        self.metrics["total_messages"] += 1
        if success:
            self.metrics["successful"] += 1
        else:
            self.metrics["failed"] += 1
        # 평균 시간 업데이트
```

#### 4.2 추출 품질 메트릭

```python
# utils/quality_metrics.py (신규)
class QualityMetrics:
    """추출 품질 메트릭"""
    
    def calculate_precision_recall(self, 
                                   extracted: List[str], 
                                   ground_truth: List[str]) -> Dict[str, float]:
        """정확도/재현율 계산"""
        # 구현
```

---

## 종합 평가

### Agent 작업 적합성: ⭐⭐⭐⭐☆ (4/5)

**강점**:
- ✅ 명확한 모듈 분리
- ✅ Workflow 기반 구조
- ✅ Config 중앙화
- ✅ 서비스 독립성

**개선 필요**:
- ⚠️ 문서화 부족
- ⚠️ 테스트 커버리지
- ⚠️ 의존성 복잡도

### 권장 개선 순서

1. **즉시 (1-2일)**:
   - 이 가이드 문서 검토 및 보완
   - 주요 서비스 Docstring 강화
   - Workflow 가이드 작성

2. **단기 (1주)**:
   - 서비스별 단위 테스트 작성
   - 아키텍처 다이어그램 생성
   - 타입 힌트 완성도 향상

3. **중기 (2-4주)**:
   - 통합 테스트 확대
   - 성능/품질 메트릭 시스템 구축
   - 프롬프트 버전 관리 시스템

---

## 결론

현재 MMS Extractor는 **Agent가 작업하기에 양호한 구조**를 가지고 있습니다. 특히 Workflow 패턴과 서비스 분리는 Agent가 변경 영향 범위를 파악하고 안전하게 수정하기에 적합합니다.

다만, **문서화와 테스트 강화**를 통해 Agent의 작업 효율성을 크게 향상시킬 수 있습니다. 이 가이드 문서를 시작으로, 점진적으로 문서화를 확대하면 향후 Agent 기반 개발이 더욱 원활해질 것입니다.

---

*작성일: 2025-12-10*  
*대상: Agent 및 개발자*  
*버전: 1.0*
