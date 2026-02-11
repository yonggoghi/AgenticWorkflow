# Step 7 리팩토링 옵션 - Stage 1/2 완전 분리

**Date**: 2026-02-11
**Goal**: Stage 1의 두 가지 방식 (langextract, entity_extraction_prompt.py)을 모두 EntityMatchingStep에서 독립

---

## 현재 구조의 문제점

### EntityMatchingStep의 복잡한 책임
```
[EntityMatchingStep]
├─ Stage 1 Option A: langextract (조건부)
├─ Stage 1 Option B: entity_extraction_prompt.py (entity_recognizer 내부, 조건부)
└─ Stage 2: vocabulary filtering (항상)
```

**문제**:
1. EntityMatchingStep이 "사전 추출" + "매칭" + "필터링" 세 가지 책임
2. entity_recognizer 내부에 Stage 1 로직이 숨어있음
3. 관찰성 낮음: Stage 1의 두 방식을 구분하기 어려움

---

## 목표

### Stage 1과 Stage 2를 명확히 분리

**Stage 1: Entity + Context 추출**
- 방식 A: langextract
- 방식 B: entity_extraction_prompt.py

**Stage 2: Vocabulary Filtering**
- fuzzy matching + LLM filtering

**요구사항**:
- ✅ Stage 1의 두 방식을 동일한 수준으로 독립
- ✅ Stage 2는 Stage 1의 구현 방식과 무관하게 동작
- ✅ 명확한 단일 책임
- ✅ 높은 관찰성

---

## 옵션 1: 3-Step 분리 (권장)

### 구조
```
Step 7: EntityContextExtractionStep (새로 추가)
  - Stage 1 통합 (langextract + entity_extraction_prompt.py)
  - 출력: state.extracted_entities = {entities, context_text}

Step 8: VocabularyFilteringStep (새로 추가)
  - Stage 2만 담당 (vocabulary filtering)
  - 입력: state.extracted_entities
  - 출력: state.matched_products

Step 9: ResultConstructionStep (기존 Step 8 → renumber)
```

### 장점
✅ **Stage 1과 Stage 2 명확히 분리**
✅ **단일 책임**: 각 Step이 하나의 명확한 역할
✅ **관찰성 최고**:
```
✅ Step 7: EntityContextExtractionStep (1.8s)
    [langextract: 1.5s] 또는 [entity_extraction_prompt.py: 1.8s]
✅ Step 8: VocabularyFilteringStep (1.5s)
✅ Step 9: ResultConstructionStep (1.2s)
```
✅ **유연성**: 나중에 Stage 1에 새로운 방식 추가 가능

### 단점
⚠️ 10 → 11 steps
⚠️ entity_recognizer 리팩토링 필요 (Stage 1/2 분리)

### 구현 세부사항

#### Step 7: EntityContextExtractionStep
```python
class EntityContextExtractionStep(WorkflowStep):
    """
    엔티티 + 컨텍스트 추출 (Step 7)

    두 가지 방식 중 하나를 선택하여 실행:
    - langextract: Google langextract 라이브러리 사용
    - llm: entity_extraction_prompt.py의 프롬프트 사용

    데이터 흐름:
        입력: state.msg
        출력: state.extracted_entities = {
            'entities': [...],
            'context_text': "...",
            'context_mode': 'typed' | 'dag' | 'ont' | ...,
            'extraction_method': 'langextract' | 'llm'
        }
    """

    def __init__(self, extraction_engine='default', context_mode='dag',
                 llm_model='ax', entity_recognizer=None):
        self.extraction_engine = extraction_engine
        self.context_mode = context_mode
        self.llm_model = llm_model
        self.entity_recognizer = entity_recognizer

    def should_execute(self, state: WorkflowState) -> bool:
        """에러 없고, entity_extraction_mode='llm'일 때 실행"""
        if state.has_error():
            return False
        # logic 모드면 스킵 (fuzzy matching만 사용)
        return self.entity_recognizer.entity_extraction_mode == 'llm'

    def execute(self, state: WorkflowState) -> WorkflowState:
        logger.info("🔍 [Step 7] 엔티티 + 컨텍스트 추출 시작...")
        stage_start = time.time()

        if self.extraction_engine == 'langextract':
            # 방식 A: langextract
            result = self._extract_with_langextract(state.msg)
        else:
            # 방식 B: entity_extraction_prompt.py
            result = self._extract_with_llm(state.msg)

        state.extracted_entities = result

        elapsed = time.time() - stage_start
        logger.info(f"✅ 엔티티 추출 완료 ({result['extraction_method']}): "
                   f"{len(result['entities'])}개 엔티티 ({elapsed:.1f}s)")

        return state

    def _extract_with_langextract(self, msg: str) -> dict:
        """langextract 방식"""
        from core.lx_extractor import extract_mms_entities

        doc = extract_mms_entities(msg, model_id=self.llm_model)
        entities = []
        type_pairs = []

        for ext in (doc.extractions or []):
            if ext.extraction_class not in ('Channel', 'Purpose'):
                if len(ext.extraction_text) >= 2:
                    entities.append(ext.extraction_text)
                    type_pairs.append(f"{ext.extraction_text}({ext.extraction_class})")

        return {
            'entities': entities,
            'context_text': ", ".join(type_pairs),
            'context_mode': 'typed',
            'extraction_method': 'langextract'
        }

    def _extract_with_llm(self, msg: str) -> dict:
        """entity_extraction_prompt.py 방식"""
        # entity_recognizer의 Stage 1 로직 호출
        result = self.entity_recognizer._extract_entities_stage1(
            msg, context_mode=self.context_mode
        )

        return {
            'entities': result['entities'],
            'context_text': result['context_text'],
            'context_mode': self.context_mode,
            'extraction_method': 'llm'
        }
```

#### Step 8: VocabularyFilteringStep
```python
class VocabularyFilteringStep(WorkflowStep):
    """
    Vocabulary 기반 엔티티 필터링 (Step 8)

    Stage 1에서 추출한 엔티티들을 DB vocabulary와 비교하여 최종 선택.

    데이터 흐름:
        입력:
            - state.extracted_entities (from Step 7)
            - state.json_objects
            - state.entities_from_kiwi
        출력: state.matched_products
    """

    def __init__(self, entity_recognizer, alias_pdf_raw, stop_item_names,
                 use_external_candidates=True):
        self.entity_recognizer = entity_recognizer
        self.alias_pdf_raw = alias_pdf_raw
        self.stop_item_names = stop_item_names
        self.use_external_candidates = use_external_candidates

    def should_execute(self, state: WorkflowState) -> bool:
        """에러 없고, extracted_entities 있을 때 실행"""
        if state.has_error():
            return False
        if state.is_fallback:
            return False

        # Step 7에서 추출된 엔티티가 있어야 함
        extracted = getattr(state, 'extracted_entities', None)
        if extracted and len(extracted.get('entities', [])) > 0:
            return True

        # 또는 json_objects/kiwi에 엔티티가 있어야 함
        product_items = state.json_objects.get('product', [])
        if isinstance(product_items, dict):
            product_items = product_items.get('items', [])

        return len(product_items) > 0 or len(state.entities_from_kiwi) > 0

    def execute(self, state: WorkflowState) -> WorkflowState:
        logger.info("🔍 [Step 8] Vocabulary 필터링 시작...")
        stage_start = time.time()

        # Get extracted entities from Step 7
        extracted = getattr(state, 'extracted_entities', None)

        if extracted:
            # Step 7에서 추출한 엔티티 사용
            entities = extracted['entities']
            context_text = extracted['context_text']
            context_mode = extracted['context_mode']
        else:
            # Fallback: json_objects에서 추출 (logic 모드일 때)
            entities = []
            context_text = ""
            context_mode = 'none'

        # External candidates 추가
        if self.use_external_candidates:
            # ... existing logic ...
            pass

        # entity_recognizer의 Stage 2 로직 호출
        matched = self.entity_recognizer._filter_with_vocabulary(
            entities=entities,
            context_text=context_text,
            context_mode=context_mode,
            msg=state.msg
        )

        # Product mapping
        state.matched_products = self.entity_recognizer.map_products_to_entities(
            matched, state.json_objects
        )

        elapsed = time.time() - stage_start
        logger.info(f"✅ Vocabulary 필터링 완료: {len(state.matched_products)}개 매칭 ({elapsed:.1f}s)")

        return state
```

#### entity_recognizer 리팩토링
```python
# services/entity_recognizer.py

class EntityRecognizer:
    def _extract_entities_stage1(self, msg: str, context_mode: str = 'dag') -> dict:
        """
        Stage 1: Entity + Context 추출

        기존 extract_entities_with_llm()의 Stage 1 부분만 분리.
        Lines 712-925의 로직.

        Returns:
            {
                'entities': [...],
                'context_text': "...",
                'entity_types': {...},  # ont 모드일 때만
                'relationships': [...]   # ont 모드일 때만
            }
        """
        # ... existing Stage 1 logic (lines 712-925) ...
        pass

    def _filter_with_vocabulary(self, entities: list, context_text: str,
                                context_mode: str, msg: str) -> pd.DataFrame:
        """
        Stage 2: Vocabulary Filtering

        기존 extract_entities_with_llm()의 Stage 2 부분만 분리.
        Lines 940-1006의 로직.

        Returns:
            DataFrame with filtered entities
        """
        # ... existing Stage 2 logic (lines 940-1006) ...
        pass

    def extract_entities_with_llm(self, msg_text: str, ...):
        """
        기존 메서드 (backward compatibility 유지)

        내부적으로 _extract_entities_stage1 + _filter_with_vocabulary 호출
        """
        if pre_extracted:
            # Stage 1 스킵
            result = self._filter_with_vocabulary(
                entities=pre_extracted['entities'],
                context_text=pre_extracted['context_text'],
                context_mode='typed',
                msg=msg_text
            )
        else:
            # Stage 1 + Stage 2
            stage1 = self._extract_entities_stage1(msg_text, context_mode)
            result = self._filter_with_vocabulary(
                entities=stage1['entities'],
                context_text=stage1['context_text'],
                context_mode=context_mode,
                msg=msg_text
            )

        return result
```

### WorkflowState 수정
```python
@dataclass
class WorkflowState:
    # ... existing fields ...

    # Entity extraction (set by EntityContextExtractionStep)
    extracted_entities: Optional[Dict[str, Any]] = None  # {entities, context_text, context_mode, extraction_method}

    # Entity matching (set by VocabularyFilteringStep)
    matched_products: List[Dict[str, Any]] = field(default_factory=list)
```

### 파이프라인 순서
```
1. InputValidationStep
2. EntityExtractionStep (Kiwi)
3. ProgramClassificationStep
4. ContextPreparationStep
5. LLMExtractionStep
6. ResponseParsingStep
7. EntityContextExtractionStep (Stage 1: langextract 또는 entity_extraction_prompt.py)
8. VocabularyFilteringStep (Stage 2: vocabulary filtering)
9. ResultConstructionStep
10. ValidationStep
11. DAGExtractionStep
```

**Total: 11 steps**

---

## 옵션 2: 4-Step 분리 (최대 명확성)

### 구조
```
Step 7A: LangExtractStep
  - langextract만 담당
  - 조건: extraction_engine='langextract'

Step 7B: LLMEntityExtractionStep
  - entity_extraction_prompt.py만 담당
  - 조건: extraction_engine='default' and entity_extraction_mode='llm'

Step 8: VocabularyFilteringStep
  - Stage 2만 담당

Step 9: ResultConstructionStep
```

### 장점
✅ **최대 명확성**: 각 추출 방식이 독립된 Step
✅ **배타적 실행 명확**: 7A와 7B는 절대 동시 실행 안 됨
✅ **최고 관찰성**:
```
⏭️ Step 7A: LangExtractStep (skipped - extraction_engine=default)
✅ Step 7B: LLMEntityExtractionStep (1.8s)
✅ Step 8: VocabularyFilteringStep (1.5s)
```

### 단점
⚠️ 10 → 12 steps (너무 많음)
⚠️ Step 7A와 7B가 배타적 → 개념적으로 하나의 "역할"인데 2개 Step

---

## 옵션 3: 2-Step 분리 + entity_recognizer 캡슐화 유지

### 구조
```
Step 7: PreExtractionStep (선택적)
  - langextract만 담당
  - 조건: extraction_engine='langextract'
  - 출력: state.pre_extracted

Step 8: EntityMatchingStep (수정)
  - entity_recognizer.extract_entities_with_llm() 호출
  - pre_extracted 있으면 → Stage 2만
  - pre_extracted 없으면 → Stage 1 (entity_extraction_prompt.py) + Stage 2
```

### 장점
✅ **최소 변경**: entity_recognizer 리팩토링 불필요
✅ **10 → 11 steps**
✅ **캡슐화 유지**: entity_recognizer 내부 구조 숨김

### 단점
❌ **Stage 1 방식의 비대칭성**: langextract는 Step으로 분리, entity_extraction_prompt.py는 entity_recognizer 내부
❌ **사용자 요구사항 미충족**: "동일한 수준으로 독립"이 목표인데 비대칭적

---

## 비교표

| 기준 | 옵션 1 (3-Step) | 옵션 2 (4-Step) | 옵션 3 (2-Step) |
|------|----------------|----------------|----------------|
| **Steps 수** | 11 | 12 | 11 |
| **Stage 1/2 분리** | ✅ 완전 분리 | ✅ 완전 분리 | ⚠️ 부분 분리 |
| **두 방식 대칭성** | ✅ 동일 Step 내 | ✅ 각각 독립 Step | ❌ 비대칭 |
| **entity_recognizer 리팩토링** | 필요 (medium) | 필요 (medium) | 불필요 |
| **관찰성** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **단일 책임** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **구현 복잡도** | Medium | Medium | Low |
| **유지보수성** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **사용자 요구 충족** | ✅ | ✅ | ❌ |

---

## 최종 권장: 옵션 1 (3-Step 분리)

### 이유

1. **사용자 요구사항 충족**:
   - Stage 1의 두 방식을 동일한 수준으로 독립 ✅
   - EntityContextExtractionStep 내부에서 두 방식이 대등하게 처리됨

2. **명확한 Stage 1/2 분리**:
   - EntityContextExtractionStep: Stage 1만 담당
   - VocabularyFilteringStep: Stage 2만 담당
   - 역할이 명확함

3. **합리적인 복잡도**:
   - 11 steps (12 steps인 옵션 2보다 적음)
   - entity_recognizer 리팩토링 필요하지만 깔끔한 설계
   - 두 방식이 같은 Step 안에 있어서 "하나의 역할"이라는 개념 유지

4. **높은 관찰성**:
   ```
   ✅ Step 7: EntityContextExtractionStep (1.8s)
       - Method: langextract (or llm)
       - Entities: 5개
       - Context: "아이폰17(Product), 을지로점(Store)"
   ✅ Step 8: VocabularyFilteringStep (1.5s)
       - Matched: 3개
   ```

5. **확장성**:
   - 나중에 Stage 1에 새로운 방식 추가 가능
   - 예: OpenAI structured outputs, Anthropic tool use 등

---

## 구현 계획 (옵션 1)

### Phase 1: entity_recognizer 리팩토링 (2-3시간)

#### 1.1. Stage 1/2 분리
```python
# services/entity_recognizer.py

def _extract_entities_stage1(self, msg: str, context_mode: str = 'dag',
                             llm_models: list = None) -> dict:
    """Stage 1 로직 (lines 712-925)"""
    # ... 기존 로직 ...
    return {
        'entities': all_entities,
        'context_text': combined_context,
        'entity_types': all_entity_types,  # ont mode only
        'relationships': all_relationships  # ont mode only
    }

def _filter_with_vocabulary(self, entities: list, context_text: str,
                            context_mode: str, msg: str, rank_limit: int = 5) -> pd.DataFrame:
    """Stage 2 로직 (lines 940-1006)"""
    # ... 기존 로직 ...
    return cand_entities_sim

def extract_entities_with_llm(self, msg_text: str, ...):
    """Backward compatibility wrapper"""
    if pre_extracted:
        return self._filter_with_vocabulary(...)
    else:
        stage1 = self._extract_entities_stage1(...)
        return self._filter_with_vocabulary(...)
```

### Phase 2: 새 Step 클래스 생성 (2-3시간)

#### 2.1. EntityContextExtractionStep
```python
# core/mms_workflow_steps.py
class EntityContextExtractionStep(WorkflowStep):
    # ... 위의 구현 참조 ...
```

#### 2.2. VocabularyFilteringStep
```python
# core/mms_workflow_steps.py
class VocabularyFilteringStep(WorkflowStep):
    # ... 위의 구현 참조 ...
```

#### 2.3. WorkflowState 수정
```python
# core/workflow_core.py
@dataclass
class WorkflowState:
    # ... existing ...
    extracted_entities: Optional[Dict[str, Any]] = None
    matched_products: List[Dict[str, Any]] = field(default_factory=list)
```

#### 2.4. MMSExtractor 업데이트
```python
# core/mms_extractor.py

# Step 7: EntityContextExtractionStep
self.workflow_engine.add_step(
    EntityContextExtractionStep(
        extraction_engine=self.extraction_engine,
        context_mode=entity_extraction_context_mode,
        llm_model=llm_model,
        entity_recognizer=self.entity_recognizer
    )
)

# Step 8: VocabularyFilteringStep
self.workflow_engine.add_step(
    VocabularyFilteringStep(
        entity_recognizer=self.entity_recognizer,
        alias_pdf_raw=self.alias_pdf_raw,
        stop_item_names=self.stop_item_names,
        use_external_candidates=self.use_external_candidates
    )
)
```

### Phase 3: 문서 업데이트 (1시간)

1. ARCHITECTURE.md: 10 → 11 steps, Stage 1/2 분리 설명
2. WORKFLOW_GUIDE.md: Step 7 (EntityContextExtraction) + Step 8 (VocabularyFiltering)
3. EXECUTION_FLOW.md: 흐름도 업데이트
4. QUICK_REFERENCE.md: 단계 번호 수정
5. WORKFLOW_EXECUTIVE_SUMMARY.md: 11 steps 반영
6. WORKFLOW_SUMMARY.md: 11 steps 반영

### Phase 4: 테스트 (1시간)

```bash
# 1. Default engine (Step 7 uses entity_extraction_prompt.py)
python tests/trace_product_extraction.py \
    --message "테스트" \
    --extraction-engine default \
    --entity-matching-mode llm \
    --data-source local

# Expected:
# ✅ Step 7: EntityContextExtractionStep (1.8s) - method: llm
# ✅ Step 8: VocabularyFilteringStep (1.5s)

# 2. LangExtract engine (Step 7 uses langextract)
python tests/trace_product_extraction.py \
    --message "테스트" \
    --extraction-engine langextract \
    --entity-matching-mode llm \
    --data-source local

# Expected:
# ✅ Step 7: EntityContextExtractionStep (1.5s) - method: langextract
# ✅ Step 8: VocabularyFilteringStep (1.5s)

# 3. Logic mode (Step 7 skipped)
python tests/trace_product_extraction.py \
    --message "테스트" \
    --extraction-engine default \
    --entity-matching-mode logic \
    --data-source local

# Expected:
# ⏭️ Step 7: EntityContextExtractionStep (skipped - mode: logic)
# ⏭️ Step 8: VocabularyFilteringStep (skipped - no extracted entities)
```

---

## 예상 소요 시간

| Phase | 작업 | 시간 |
|-------|------|------|
| Phase 1 | entity_recognizer 리팩토링 | 2-3시간 |
| Phase 2 | 새 Step 클래스 생성 | 2-3시간 |
| Phase 3 | 문서 업데이트 | 1시간 |
| Phase 4 | 테스트 | 1시간 |
| **Total** | | **6-8시간** |

---

## 결론

**옵션 1 (3-Step 분리) 권장**

**핵심 이점**:
1. ✅ Stage 1의 두 방식을 동일한 수준으로 독립
2. ✅ Stage 1과 Stage 2 명확히 분리
3. ✅ 명확한 단일 책임
4. ✅ 높은 관찰성
5. ✅ 합리적인 복잡도 (11 steps)

**다음 단계**: 사용자 승인 후 Phase 1-4 구현 진행

---

*작성 날짜: 2026-02-11*
*예상 구현 시간: 6-8시간*
