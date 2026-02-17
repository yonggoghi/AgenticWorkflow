# Step 7 Stage 1 & 2 완전 분석

**Date**: 2026-02-11
**Purpose**: EntityMatchingStep의 Stage 1과 Stage 2 역할을 명확히 파악하고 분할 여부 결정

---

## Stage 1과 Stage 2의 역할

### Stage 1: Entity + Context 추출
**목적**: 메시지에서 후보 엔티티 목록과 컨텍스트 정보 추출
**출력**: `entities` (리스트) + `context_text` (문자열)

**두 가지 구현 방식**:

#### 방식 A: LangExtract (EntityMatchingStep에서 실행)
- **위치**: EntityMatchingStep.execute() lines 679-705
- **조건**: `extraction_engine='langextract'`
- **사용 프롬프트**: `core/lx_extractor.py`의 MMS_PROMPT_DESCRIPTION + lx_examples
- **처리 과정**:
  ```python
  from core.lx_extractor import extract_mms_entities
  doc = extract_mms_entities(msg, model_id=self.llm_model)
  entities = []
  type_pairs = []
  for ext in doc.extractions:
      if ext.extraction_class not in ('Channel', 'Purpose'):
          if len(ext.extraction_text) >= 2:
              entities.append(ext.extraction_text)
              type_pairs.append(f"{ext.extraction_text}({ext.extraction_class})")

  pre_extracted = {
      'entities': entities,
      'context_text': ", ".join(type_pairs)  # "아이폰17(Product), 을지로점(Store)"
  }
  ```
- **출력 예시**:
  ```python
  {
      'entities': ['아이폰17', '을지로점', '특가기변'],
      'context_text': '아이폰17(Product), 을지로점(Store), 특가기변(Program)'
  }
  ```

#### 방식 B: LLM with entity_extraction_prompt.py (entity_recognizer 내부)
- **위치**: entity_recognizer.extract_entities_with_llm() lines 712-925
- **조건**: `pre_extracted=None` (langextract를 사용하지 않았을 때)
- **사용 프롬프트**: `prompts/entity_extraction_prompt.py`
  - `dag` mode → `HYBRID_DAG_EXTRACTION_PROMPT`
  - `pairing` mode → `HYBRID_PAIRING_EXTRACTION_PROMPT`
  - `ont` mode → `ONTOLOGY_PROMPT`
  - `typed` mode → `TYPED_ENTITY_EXTRACTION_PROMPT`
  - `none` mode → `SIMPLE_ENTITY_EXTRACTION_PROMPT`
- **처리 과정**:
  ```python
  # 1. context_mode에 따라 프롬프트 선택
  if context_mode == 'dag':
      first_stage_prompt = HYBRID_DAG_EXTRACTION_PROMPT
      context_keyword = 'DAG'
  elif context_mode == 'ont':
      first_stage_prompt = ONTOLOGY_PROMPT
      context_keyword = 'ONT'
  # ... etc

  # 2. LLM 호출
  prompt = f"{first_stage_prompt}\n\n## message:\n{msg_text}"
  response = llm_model.invoke(prompt).content

  # 3. 파싱 (mode별로 다름)
  # - ont mode: JSON 파싱 → entities, entity_types, relationships, dag_text
  # - typed mode: JSON 파싱 → entities with types
  # - standard mode: Regex 파싱 → entities + DAG/PAIRING context

  # 4. 출력
  return {
      "entities": cand_entity_list,
      "context_text": combined_context
  }
  ```
- **출력 예시 (dag mode)**:
  ```python
  {
      'entities': ['아이폰17', '을지로점', '기변'],
      'context_text': '(을지로점:방문) -[방문하면]-> (아이폰17:기변)'
  }
  ```

---

### Stage 2: Vocabulary Filtering
**목적**: Stage 1에서 추출한 엔티티들을 DB vocabulary와 비교하여 최종 선택
**입력**: Stage 1의 `entities` + `context_text`
**출력**: Filtered entity DataFrame (item_nm_alias, item_name_in_msg, similarity)

**실행 위치**: entity_recognizer.extract_entities_with_llm() lines 940-1006

**처리 과정**:
```python
# 1. Stage 1 entities를 product DB와 fuzzy matching
cand_entities_sim = self._match_entities_with_products(cand_entity_list, rank_limit)

# 2. Vocabulary filtering을 위한 프롬프트 구성
second_stage_prompt = build_context_based_entity_extraction_prompt(context_keyword)

# 3. Context section 추가 (Stage 1에서 추출한 context_text 사용)
if context_keyword == 'TYPED':
    context_section = f"\n## TYPED Context (Entity Types):\n{combined_context}\n"
elif context_keyword:
    context_section = f"\n## {context_keyword} Context (User Action Paths):\n{combined_context}\n"

# 4. LLM 호출하여 vocabulary 필터링
prompt = f"""
{second_stage_prompt}

## message:
{msg_text}

{context_section}

## entities in message:
{', '.join(entities_in_message)}

## candidate entities in vocabulary:
{', '.join(cand_entities_voca)}
"""

response = llm_model.invoke(prompt).content
filtered_entities = self._parse_entity_response(response)

# 5. 최종 필터링
cand_entities_sim = cand_entities_sim.query("item_nm_alias in @filtered_entities")
```

**사용 프롬프트**: `prompts/entity_extraction_prompt.py`의 `build_context_based_entity_extraction_prompt()`

**핵심**: Stage 1에서 추출한 `context_text`를 Stage 2 프롬프트에 추가하여 LLM이 더 정확한 필터링을 수행하도록 함

---

## 전체 실행 흐름

### 시나리오 A: langextract 엔진 사용

```
[EntityMatchingStep]
  ├─ Stage 1 (langextract 방식)
  │   ├─ extract_mms_entities(msg)
  │   ├─ 6-type 분류: Product, Store, Program, Channel, Purpose, Other
  │   └─ pre_extracted = {entities: [...], context_text: "..."}
  │
  └─ entity_recognizer.extract_entities_with_llm(pre_extracted=pre_extracted)
      ├─ Stage 1 스킵 (pre_extracted 사용)
      └─ Stage 2만 실행 (vocabulary filtering)
          ├─ pre_extracted['entities'] → fuzzy matching
          ├─ pre_extracted['context_text'] → Stage 2 프롬프트에 추가
          └─ LLM 호출하여 최종 필터링
```

**LLM 호출 횟수**: 2회
- 1회: langextract (Stage 1)
- 1회: vocabulary filtering (Stage 2)

---

### 시나리오 B: default 엔진 사용 (entity_extraction_mode='llm')

```
[EntityMatchingStep]
  └─ entity_recognizer.extract_entities_with_llm(pre_extracted=None)
      ├─ Stage 1 (entity_extraction_prompt.py 방식)
      │   ├─ context_mode에 따라 프롬프트 선택
      │   ├─ LLM 호출하여 entities + context_text 추출
      │   │   - dag mode: HYBRID_DAG_EXTRACTION_PROMPT
      │   │   - ont mode: ONTOLOGY_PROMPT (entity types + relationships)
      │   │   - typed mode: TYPED_ENTITY_EXTRACTION_PROMPT
      │   └─ entities + context_text 생성
      │
      └─ Stage 2 (vocabulary filtering)
          ├─ Stage 1 entities → fuzzy matching
          ├─ Stage 1 context_text → Stage 2 프롬프트에 추가
          └─ LLM 호출하여 최종 필터링
```

**LLM 호출 횟수**: 2회
- 1회: entity + context 추출 (Stage 1)
- 1회: vocabulary filtering (Stage 2)

---

### 시나리오 C: default 엔진 + logic 모드

```
[EntityMatchingStep]
  └─ entity_recognizer.extract_entities_with_fuzzy_matching()
      └─ Fuzzy matching만 수행 (LLM 호출 없음)
```

**LLM 호출 횟수**: 0회 (logic 모드는 LLM 사용 안 함)

---

## 핵심 통찰

### 1. Stage 1의 두 가지 구현은 **대안적 관계**
- langextract 방식과 entity_extraction_prompt.py 방식은 동시에 실행되지 않음
- 한 번에 하나만 선택되어 실행
- 둘 다 같은 목적: `entities` + `context_text` 생성

### 2. Stage 2는 **항상 동일한 로직**
- Stage 1의 구현 방식과 무관하게 동일한 vocabulary filtering 수행
- `build_context_based_entity_extraction_prompt()` 사용
- Stage 1에서 생성된 `context_text`를 컨텍스트로 활용

### 3. EntityMatchingStep의 실제 역할
**현재**:
- Stage 1 Option A 구현 (langextract, 선택적)
- entity_recognizer 호출 (Stage 1 Option B + Stage 2 또는 Stage 2만)

**문제점**:
- EntityMatchingStep이 두 가지 일을 함:
  1. langextract 기반 사전 추출 (선택적)
  2. entity_recognizer를 통한 매칭 (필수)
- 명확한 단일 책임이 없음

### 4. entity_recognizer 내부의 Stage 1 + Stage 2
- entity_recognizer는 이미 "Stage 1 + Stage 2" 구조를 가지고 있음
- pre_extracted가 있으면 Stage 1 스킵, 없으면 Stage 1 실행
- 이건 entity_recognizer의 내부 로직으로 캡슐화되어 있음

---

## 분할 여부 분석

### Option 1: 분할 (권장)

**구조**:
```
Step 7: LangExtractStep (새로 추가)
  - 목적: langextract 기반 사전 추출
  - 조건: extraction_engine='langextract'
  - 출력: state.pre_extracted = {entities, context_text}

Step 8: EntityMatchingStep (수정)
  - 목적: 엔티티 매칭 (entity_recognizer 통한)
  - 입력: state.pre_extracted (옵션)
  - 처리:
    - entity_extraction_mode='llm':
      - pre_extracted 있으면 → Stage 2만 실행
      - pre_extracted 없으면 → Stage 1 + Stage 2 실행
    - entity_extraction_mode='logic': fuzzy matching만
  - 출력: state.matched_products
```

**장점**:
1. ✅ **Single Responsibility**:
   - LangExtractStep: "langextract 기반 사전 추출"만 담당
   - EntityMatchingStep: "엔티티 매칭 (entity_recognizer 호출)"만 담당

2. ✅ **명확한 조건부 실행**:
   ```python
   # Step 7: LangExtractStep
   def should_execute(self, state):
       return not state.has_error() and self.extraction_engine == 'langextract'

   # Step 8: EntityMatchingStep
   def should_execute(self, state):
       return not state.has_error() and not state.is_fallback and has_entities
   ```

3. ✅ **더 나은 관찰성**:
   ```
   ✅ Step 7: LangExtractStep (1.5s)
   ✅ Step 8: EntityMatchingStep (2.1s)
       - entity_recognizer Stage 2만 실행 (1.5s)
       - product matching (0.6s)
   ```

4. ✅ **독립적 테스트**:
   ```python
   # LangExtractStep만 테스트
   def test_langextract_step():
       step = LangExtractStep(...)
       state = step.execute(WorkflowState(msg="..."))
       assert state.pre_extracted is not None

   # EntityMatchingStep with pre_extracted 테스트
   def test_entity_matching_with_pre_extracted():
       state.pre_extracted = {...}
       step = EntityMatchingStep(...)
       state = step.execute(state)
       assert len(state.matched_products) > 0
   ```

5. ✅ **entity_recognizer 캡슐화 유지**:
   - entity_recognizer 내부의 Stage 1/2 로직은 그대로 유지
   - pre_extracted 유무에 따라 자동으로 분기
   - EntityMatchingStep은 entity_recognizer의 내부 구조를 알 필요 없음

**단점**:
- ⚠️ 10 → 11 steps (minor)
- ⚠️ 1개 클래스 추가 (~50 lines)
- ⚠️ 문서 업데이트 필요

---

### Option 2: 유지 (비권장)

**구조**: 현재 그대로 유지

**단점**:
1. ❌ EntityMatchingStep이 두 가지 책임:
   - langextract 사전 추출 (Stage 1 Option A)
   - entity_recognizer 호출 (Stage 1 Option B + Stage 2)

2. ❌ 복잡한 should_execute() 로직:
   ```python
   def should_execute(self, state):
       if state.has_error():
           return False
       if state.is_fallback and self.extraction_engine != 'langextract':
           return False
       # ... 복잡한 조건 ...
       return has_entities or self.extraction_engine == 'langextract'
   ```

3. ❌ 관찰성 낮음:
   ```
   ✅ Step 7: EntityMatchingStep (3.6s)
       [내부에서 langextract 1.5s + matching 2.1s, 구분 어려움]
   ```

---

## 최종 권장사항

### ✅ **분할 권장**

**이유**:
1. **명확한 책임 분리**: LangExtractStep은 "사전 추출", EntityMatchingStep은 "매칭"
2. **entity_recognizer 캡슐화**: entity_recognizer의 Stage 1/2 로직은 내부 구현으로 유지
3. **워크플로우 일관성**: 다른 Step들과 마찬가지로 "한 Step = 한 책임"
4. **테스트 용이성**: 각 Step을 독립적으로 테스트 가능
5. **관찰성 향상**: 각 Step의 소요 시간과 성공/실패 명확히 구분

**구현 복잡도**: 낮음 (2-3시간)
- LangExtractStep 클래스 추가 (~50 lines)
- EntityMatchingStep 수정 (Stage 1 제거, ~30 lines 삭제)
- WorkflowState에 pre_extracted 필드 추가 (1 line)
- 문서 업데이트 (6 files)

---

## 구현 계획 (분할 시)

### Phase 1: 코드 변경 (1-2시간)

#### 1.1. WorkflowState에 pre_extracted 추가
```python
# core/workflow_core.py
@dataclass
class WorkflowState:
    # ... existing fields ...

    # LangExtract pre-extraction
    pre_extracted: Optional[Dict[str, Any]] = None  # Set by LangExtractStep
    matched_products: List[Dict[str, Any]] = field(default_factory=list)
```

#### 1.2. LangExtractStep 생성
```python
# core/mms_workflow_steps.py

class LangExtractStep(WorkflowStep):
    """
    LangExtract 기반 엔티티 사전 추출 (Step 7)

    Google langextract를 사용하여 6-type 엔티티를 추출하고
    pre_extracted 결과를 state에 저장합니다.

    데이터 흐름:
        입력: state.msg
        출력: state.pre_extracted = {entities: [...], context_text: "..."}
    """

    def __init__(self, llm_model: str = 'ax', extraction_engine: str = 'default'):
        self.llm_model = llm_model
        self.extraction_engine = extraction_engine

    def should_execute(self, state: WorkflowState) -> bool:
        """langextract 엔진이 선택되었을 때만 실행"""
        if state.has_error():
            return False
        return self.extraction_engine == 'langextract'

    def execute(self, state: WorkflowState) -> WorkflowState:
        """langextract로 엔티티 사전 추출"""
        from core.lx_extractor import extract_mms_entities

        logger.info("🔗 [Step 7] LangExtract 엔티티 사전 추출 시작...")
        stage_start = time.time()

        try:
            doc = extract_mms_entities(state.msg, model_id=self.llm_model)

            entities = []
            type_pairs = []
            for ext in (doc.extractions or []):
                if ext.extraction_class not in ('Channel', 'Purpose'):
                    if len(ext.extraction_text) >= 2:
                        entities.append(ext.extraction_text)
                        type_pairs.append(f"{ext.extraction_text}({ext.extraction_class})")

            state.pre_extracted = {
                'entities': entities,
                'context_text': ", ".join(type_pairs)
            }

            elapsed = time.time() - stage_start
            logger.info(f"✅ LangExtract 완료: {len(entities)}개 엔티티 추출 ({elapsed:.1f}s)")
            logger.info(f"   Entities: {entities}")
            logger.info(f"   Context: {state.pre_extracted['context_text']}")

        except Exception as e:
            logger.error(f"❌ LangExtract 추출 실패: {e}")
            state.pre_extracted = None

        return state
```

#### 1.3. EntityMatchingStep 수정
```python
# core/mms_workflow_steps.py

class EntityMatchingStep(WorkflowStep):
    """
    엔티티 매칭 (Step 8)

    LLM 파싱 결과의 상품명을 DB 엔티티와 매칭하여 item_id 부여.
    pre_extracted가 있으면 이를 활용하여 매칭 수행.

    데이터 흐름:
        입력:
            - state.json_objects (product items)
            - state.entities_from_kiwi
            - state.pre_extracted (optional, from LangExtractStep)
        출력: state.matched_products
    """

    def __init__(self, entity_recognizer, alias_pdf_raw, stop_item_names,
                 entity_extraction_mode, llm_factory=None, llm_model='ax',
                 entity_extraction_context_mode='dag', use_external_candidates=True):
        self.entity_recognizer = entity_recognizer
        self.alias_pdf_raw = alias_pdf_raw
        self.stop_item_names = stop_item_names
        self.entity_extraction_mode = entity_extraction_mode
        self.llm_factory = llm_factory
        self.llm_model = llm_model
        self.entity_extraction_context_mode = entity_extraction_context_mode
        self.use_external_candidates = use_external_candidates

    def should_execute(self, state: WorkflowState) -> bool:
        """에러, 폴백, 또는 엔티티 없으면 스킵"""
        if state.has_error():
            return False
        if state.is_fallback:
            return False

        json_objects = state.json_objects
        product_items = json_objects.get('product', [])
        if isinstance(product_items, dict):
            product_items = product_items.get('items', [])

        has_entities = len(product_items) > 0 or len(state.entities_from_kiwi) > 0
        return has_entities

    def execute(self, state: WorkflowState) -> WorkflowState:
        """엔티티 매칭 수행"""
        logger.info("🔍 [Step 8] 엔티티 매칭 시작...")

        # Get pre_extracted from state (set by LangExtractStep if used)
        pre_extracted = getattr(state, 'pre_extracted', None)

        if pre_extracted:
            logger.info(f"   Using pre-extracted entities: {len(pre_extracted['entities'])} entities")

        # ... rest of existing matching logic ...
        # (Remove Stage 1 langextract code, keep Stage 2 logic)

        return state
```

#### 1.4. MMSExtractor 업데이트
```python
# core/mms_extractor.py

# Step 7: LangExtractStep (new)
self.workflow_engine.add_step(
    LangExtractStep(
        llm_model=llm_model,
        extraction_engine=self.extraction_engine
    )
)

# Step 8: EntityMatchingStep (modified)
self.workflow_engine.add_step(
    EntityMatchingStep(
        entity_recognizer=self.entity_recognizer,
        alias_pdf_raw=self.alias_pdf_raw,
        stop_item_names=self.stop_item_names,
        entity_extraction_mode=entity_extraction_mode,
        llm_factory=self.llm_factory,
        llm_model=llm_model,
        entity_extraction_context_mode=entity_extraction_context_mode,
        use_external_candidates=self.use_external_candidates
    )
)
```

### Phase 2: 문서 업데이트 (1시간)

1. ARCHITECTURE.md: 10 → 11 steps
2. WORKFLOW_GUIDE.md: Step 7 (LangExtract) + Step 8 (EntityMatching) 설명
3. EXECUTION_FLOW.md: 흐름도 업데이트
4. QUICK_REFERENCE.md: 단계 번호 수정
5. WORKFLOW_EXECUTIVE_SUMMARY.md: 11 steps 반영
6. WORKFLOW_SUMMARY.md: 11 steps 반영

### Phase 3: 테스트 (30분)

```bash
# 1. Default engine (Step 7 should skip)
python tests/trace_product_extraction.py --message "테스트" --data-source local

# 2. LangExtract engine (Step 7 should execute)
python tests/trace_product_extraction.py --message "테스트" --extraction-engine langextract --data-source local

# 3. Verify timing logs
# Expected output:
#   ⏭️ Step 7: LangExtractStep (skipped - extraction_engine=default)
#   ✅ Step 8: EntityMatchingStep (2.1s)
# or:
#   ✅ Step 7: LangExtractStep (1.5s)
#   ✅ Step 8: EntityMatchingStep (1.7s)
```

---

## 결론

**Stage 1과 Stage 2의 역할**:
- **Stage 1**: Entity + Context 추출 (2가지 방식: langextract 또는 entity_extraction_prompt.py)
- **Stage 2**: Vocabulary Filtering (항상 동일한 로직)

**분할 권장 이유**:
- EntityMatchingStep이 현재 "langextract 사전 추출 + entity_recognizer 호출"이라는 두 가지 책임을 가지고 있음
- LangExtractStep으로 분리하면 각 Step이 단일 책임을 가지게 됨
- entity_recognizer 내부의 Stage 1/2 로직은 캡슐화된 구현 세부사항으로 유지

**다음 단계**: 사용자 승인 후 Phase 1-3 구현 진행

---

*분석 날짜: 2026-02-11*
*예상 구현 시간: 2-3시간*
