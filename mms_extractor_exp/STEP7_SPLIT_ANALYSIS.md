# Step 7 Stage Split Analysis

**Date**: 2026-02-11
**Current Status**: Step 7 (EntityMatchingStep) contains two stages:
- **Stage 1**: LangExtract pre-extraction (conditional on `extraction_engine='langextract'`)
- **Stage 2**: Entity matching (logic or llm mode)

**Question**: Should these be split into two separate workflow steps?

---

## Current Implementation

### EntityMatchingStep (Step 7) - Lines 612-776

**Two Stages**:
1. **Stage 1 (Lines 679-705)**: LangExtract pre-extraction
   - Only runs when `extraction_engine='langextract'`
   - Extracts entities with 6-type classification
   - Creates `pre_extracted` dict with entities and context_text
   - Passes result to Stage 2

2. **Stage 2 (Lines 706-776)**: Entity matching
   - Uses `pre_extracted` from Stage 1 (if available)
   - Performs logic or llm-based entity matching
   - Filters aliases
   - Maps products to entities
   - Sets `state.matched_products`

**Current Conditional Logic**:
- `should_execute()` returns True even when `is_fallback=True` if using langextract
- Stage 1 runs conditionally based on `extraction_engine`
- Stage 2 always runs (if step executes)

---

## Proposed Split: Two Separate Steps

### Option A: Split into Step 7A and 7B

#### Step 7A: LangExtractStep (New)
```python
class LangExtractStep(WorkflowStep):
    """
    LangExtract 기반 엔티티 사전 추출 (Step 7A)

    책임:
        - Google langextract를 사용한 6-type 엔티티 추출
        - pre_extracted 결과를 state에 저장

    데이터 흐름:
        입력: msg
        출력: pre_extracted (entities, context_text)
    """

    def should_execute(self, state: WorkflowState) -> bool:
        if state.has_error():
            return False
        # Only run when langextract engine is selected
        return self.extraction_engine == 'langextract'

    def execute(self, state: WorkflowState) -> WorkflowState:
        from core.lx_extractor import extract_mms_entities
        logger.info("🔗 langextract 엔진으로 엔티티 추출 시작...")

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
            logger.info(f"✅ langextract 완료: {len(entities)}개 엔티티 추출")
        except Exception as e:
            logger.error(f"❌ langextract 추출 실패: {e}")
            state.pre_extracted = None

        return state
```

#### Step 7B: EntityMatchingStep (Modified)
```python
class EntityMatchingStep(WorkflowStep):
    """
    엔티티 매칭 단계 (Step 7B)

    책임:
        - LLM 추출 상품명과 DB 상품을 매칭
        - pre_extracted 컨텍스트 활용 (있으면)
        - 매칭 결과를 state.matched_products에 저장

    데이터 흐름:
        입력: json_objects, entities_from_kiwi, msg, pre_extracted (optional)
        출력: matched_products
    """

    def should_execute(self, state: WorkflowState) -> bool:
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
        # Get pre_extracted from state (set by Step 7A if langextract was used)
        pre_extracted = getattr(state, 'pre_extracted', None)

        # ... rest of matching logic (Stage 2 code)
        # Uses pre_extracted if available

        return state
```

#### WorkflowState Changes
```python
@dataclass
class WorkflowState:
    # ... existing fields ...

    # LangExtract pre-extraction (set by LangExtractStep)
    pre_extracted: Optional[Dict[str, Any]] = None
```

#### Pipeline Order (10 → 11 steps)
1. InputValidationStep
2. EntityExtractionStep
3. ProgramClassificationStep
4. ContextPreparationStep
5. LLMExtractionStep
6. ResponseParsingStep
7A. **LangExtractStep** (new, conditional)
7B. **EntityMatchingStep** (modified)
8. ResultConstructionStep
9. ValidationStep
10. DAGExtractionStep

---

## Analysis

### ✅ Pros of Splitting

#### 1. **Single Responsibility Principle**
- Each step does ONE thing
- LangExtractStep: Extract entities with types
- EntityMatchingStep: Match entities to DB
- Current: EntityMatchingStep does both extraction AND matching

#### 2. **Clearer Conditional Execution**
```python
# Current (confusing):
def should_execute(self, state: WorkflowState) -> bool:
    # Complex logic mixing langextract and matching conditions
    if state.is_fallback and self.extraction_engine != 'langextract':
        return False
    has_entities = ...
    return has_entities or self.extraction_engine == 'langextract'

# After split (clear):
# Step 7A:
def should_execute(self, state: WorkflowState) -> bool:
    return not state.has_error() and self.extraction_engine == 'langextract'

# Step 7B:
def should_execute(self, state: WorkflowState) -> bool:
    return not state.has_error() and not state.is_fallback and has_entities
```

#### 3. **Better Observability**
- Separate timing logs for each stage
- Clear success/failure status per step
- Current logs show "Stage 1" and "Stage 2" within one step

Example current logs:
```
✅ EntityMatchingStep 완료 (3.2초)
  - Stage 1 (langextract): 1.5초
  - Stage 2 (matching): 1.7초
```

Example after split:
```
✅ LangExtractStep 완료 (1.5초)
✅ EntityMatchingStep 완료 (1.7초)
```

#### 4. **Easier Testing**
```python
# Current: Must mock extraction_engine to test each stage
def test_entity_matching_with_langextract():
    step = EntityMatchingStep(..., extraction_engine='langextract')
    # Tests both stages together

# After split: Test independently
def test_langextract_step():
    step = LangExtractStep(...)
    # Test only extraction

def test_entity_matching_with_pre_extracted():
    state.pre_extracted = {...}
    step = EntityMatchingStep(...)
    # Test only matching
```

#### 5. **Explicit State Management**
- `pre_extracted` becomes explicit state field
- Aligns with architecture review recommendation: "explicit state fields instead of extractor passthrough"
- Currently: `pre_extracted` is local variable passed to `extract_entities_with_llm()`

#### 6. **No "Stages" in Steps**
- Consistent with other workflow steps (no other step has "stages")
- Each step = one unit of work
- Simpler mental model

#### 7. **Better Error Handling**
```python
# Current: If Stage 1 fails, Stage 2 still runs (pre_extracted=None)
# After split: Clear failure point
✅ LangExtractStep 완료 (or skipped, or failed)
✅ EntityMatchingStep 완료
```

#### 8. **Conditional Skip Shows Explicitly**
```
Workflow execution:
  ✅ Step 6: ResponseParsingStep (0.5s)
  ⏭️ Step 7A: LangExtractStep (skipped - extraction_engine=default)
  ✅ Step 7B: EntityMatchingStep (2.1s)
  ✅ Step 8: ResultConstructionStep (1.2s)
```

vs current:
```
  ✅ Step 7: EntityMatchingStep (2.1s)
      [internally skipped Stage 1]
```

---

### ⚠️ Cons of Splitting

#### 1. **More Steps**
- 10 → 11 steps (minor increase)
- Step numbering becomes 7A, 7B (or renumber to 7, 8, 9, ...)

#### 2. **More Classes**
- New `LangExtractStep` class (~40 lines)
- Not a significant burden

#### 3. **State Field Addition**
```python
@dataclass
class WorkflowState:
    # Add new field
    pre_extracted: Optional[Dict[str, Any]] = None
```
- One more field to manage
- But makes data flow explicit (pro!)

#### 4. **Documentation Updates**
- All docs need updating (again)
- Update from "10 steps" to "11 steps"
- Update diagrams, flowcharts
- **Mitigation**: Already updated docs today, fresh in mind

#### 5. **Migration Effort**
- Update `mms_extractor.py` to add new step
- Update all tests
- ~2 hours of work

#### 6. **Conceptual Grouping Lost**
- Currently, "entity matching" conceptually includes pre-extraction
- After split, this becomes two separate operations
- **Counter**: They ARE separate operations with different purposes

---

## Recommendation

### ✅ **RECOMMEND SPLITTING**

**Reasoning**:

1. **Architecture Principles**: Aligns with Single Responsibility and explicit state management
2. **Workflow Consistency**: No other step has "stages"
3. **Observability Gain**: Clear timing and status per operation
4. **Testing Benefit**: Independent test coverage
5. **Minor Cost**: Only 1 extra step, 1 new class, 1 state field

**The benefits outweigh the costs.**

---

## Implementation Plan

### Phase 1: Code Changes (1-2 hours)

1. **Add `pre_extracted` to WorkflowState**
   ```python
   # core/workflow_core.py
   @dataclass
   class WorkflowState:
       # ... existing fields ...
       pre_extracted: Optional[Dict[str, Any]] = None  # Set by LangExtractStep
   ```

2. **Create LangExtractStep**
   ```python
   # core/mms_workflow_steps.py
   class LangExtractStep(WorkflowStep):
       """LangExtract 기반 엔티티 사전 추출 (Step 7)"""
       # Extract Stage 1 logic (lines 679-705)
   ```

3. **Modify EntityMatchingStep**
   ```python
   # core/mms_workflow_steps.py
   class EntityMatchingStep(WorkflowStep):
       """엔티티 매칭 (Step 8)"""
       # Remove Stage 1 logic
       # Read pre_extracted from state instead of creating it
       # Simplify should_execute()
   ```

4. **Update MMSExtractor**
   ```python
   # core/mms_extractor.py
   # Add LangExtractStep before EntityMatchingStep
   self.workflow_engine.add_step(LangExtractStep(...))
   self.workflow_engine.add_step(EntityMatchingStep(...))
   ```

### Phase 2: Documentation Updates (1 hour)

Update all 6 docs from "10 steps" to "11 steps":
1. ARCHITECTURE.md
2. WORKFLOW_GUIDE.md
3. EXECUTION_FLOW.md
4. QUICK_REFERENCE.md
5. WORKFLOW_EXECUTIVE_SUMMARY.md
6. WORKFLOW_SUMMARY.md

**Note**: We just updated these today, so fresh context!

### Phase 3: Testing (30 min)

1. Update trace script to show new step
2. Test with `--extraction-engine langextract`
3. Test with `--extraction-engine default` (Step 7 should skip)
4. Verify timing logs show both steps

---

## Alternative: Keep As-Is

**Rationale**: If "entity matching" is conceptually one operation that optionally uses langextract pre-extraction, keep it as one step.

**When to choose this**:
- If tight coupling between extraction and matching is desired
- If code churn is too costly right now
- If 11 steps feels like too many

**However**: The current implementation already treats them as separate stages (7.1b and 7.2), so they're conceptually separate already.

---

## Decision Criteria

### Split if:
- ✅ We value clear separation of concerns
- ✅ We want better observability per operation
- ✅ We're willing to update docs (again)
- ✅ We have 2-3 hours for implementation + testing

### Keep as-is if:
- ❌ We prefer fewer steps over clarity
- ❌ We can't afford doc/code churn right now
- ❌ We consider extraction + matching as one atomic operation

---

## Final Verdict

**✅ SPLIT INTO TWO STEPS**

The architecture would be cleaner, more testable, and more observable. The cost is minimal (1 class, 1 field, 1 step number change), and we gain consistency with the rest of the workflow pattern.

**Next Steps**:
1. Get user approval
2. Implement Phase 1 (code changes)
3. Update docs (Phase 2)
4. Test (Phase 3)
5. Commit with clear message

---

*Analysis Date: 2026-02-11*
*Estimated Implementation Time: 2-3 hours total*
