# Extraction Engine Workflow Analysis

**Date**: 2026-02-11
**Purpose**: Comprehensive analysis of `--extraction-engine` argument flow through the entire MMS Extractor workflow

---

## Overview

The `--extraction-engine` argument controls which entity extraction approach is used in **Step 7 (EntityMatchingStep)**:
- **`default`**: Standard 10-step pipeline with LLM-based entity extraction
- **`langextract`**: Google langextract-based pre-extraction in Step 7 Stage 1

---

## 1. CLI Entry Point (apps/cli.py)

### Argument Definition (Line 102-103)
```python
parser.add_argument('--extraction-engine',
                   choices=['default', 'langextract'],
                   default='default',
                   help='추출 엔진 선택 (default: 10-step pipeline, langextract: Google langextract 기반)')
```

### Auto-Configuration Logic (Lines 132-136)
```python
# When using langextract engine, force entity_extraction_context_mode to 'typed'
entity_extraction_context_mode = args.entity_extraction_context_mode
if args.extraction_engine == 'langextract':
    entity_extraction_context_mode = 'typed'
    logger.info("langextract 엔진 선택: entity_extraction_context_mode를 'typed'로 강제 설정")
```

**Key Point**: When `--extraction-engine langextract` is used:
- `entity_extraction_context_mode` is FORCED to `'typed'`
- This overrides any `--entity-extraction-context-mode` CLI argument

### MMSExtractor Initialization (Lines 141-152)
```python
extractor = MMSExtractor(
    offer_info_data_src=args.offer_data_source,
    product_info_extraction_mode=args.product_info_extraction_mode,
    entity_extraction_mode=args.entity_matching_mode,
    llm_model=args.llm_model,
    entity_llm_model=args.entity_llm_model,
    extract_entity_dag=args.extract_entity_dag,
    entity_extraction_context_mode=entity_extraction_context_mode,  # ← 'typed' if langextract
    skip_entity_extraction=args.skip_entity_extraction,
    use_external_candidates=not args.no_external_candidates,
    extraction_engine=args.extraction_engine,  # ← Passed to MMSExtractor
)
```

---

## 2. MMSExtractor Initialization (core/mms_extractor.py)

### Constructor Signature (Line 352)
```python
def __init__(self, ...,
             extraction_engine='default'):  # ← New parameter
```

### Storing Configuration (Lines 387-393)
```python
self._set_default_config(
    model_path, data_dir, product_info_extraction_mode,
    entity_extraction_mode, offer_info_data_src, llm_model, entity_llm_model,
    extract_entity_dag, entity_extraction_context_mode,
    skip_entity_extraction, use_external_candidates,
    extraction_engine  # ← Stored in self.extraction_engine
)
```

### Instance Variable (Line 503)
```python
self.extraction_engine = extraction_engine  # ← Stored for later use
```

---

## 3. Workflow Step Registration (core/mms_extractor.py)

### EntityMatchingStep Registration (Lines 456-466)
```python
self.workflow_engine.add_step(EntityMatchingStep(
    entity_recognizer=self.entity_recognizer,
    alias_pdf_raw=self.alias_pdf_raw,
    stop_item_names=self.stop_item_names,
    entity_extraction_mode=self.entity_extraction_mode,
    llm_factory=self.llm_factory,
    llm_model=self.entity_llm_model_name,
    entity_extraction_context_mode=self.entity_extraction_context_mode,  # ← 'typed' if langextract
    use_external_candidates=self.use_external_candidates,
    extraction_engine=self.extraction_engine,  # ← CRITICAL: Passed to step
))
```

**Key Point**: The `extraction_engine` parameter is passed to EntityMatchingStep during workflow initialization.

---

## 4. EntityMatchingStep Execution (core/mms_workflow_steps.py)

### Constructor (Lines 628-642)
```python
class EntityMatchingStep(WorkflowStep):
    def __init__(self, entity_recognizer, alias_pdf_raw: pd.DataFrame,
                 stop_item_names: List[str], entity_extraction_mode: str,
                 llm_factory=None, llm_model: str = 'ax',
                 entity_extraction_context_mode: str = 'dag',
                 use_external_candidates: bool = True,
                 extraction_engine: str = 'default'):  # ← Stored
        # ... other assignments ...
        self.extraction_engine = extraction_engine  # ← Instance variable
```

### Conditional Skip Logic (Lines 644-656)
```python
def should_execute(self, state: WorkflowState) -> bool:
    if state.has_error():
        return False

    # ✅ SPECIAL CASE: langextract extracts entities independently of the main prompt,
    # so it should run even when is_fallback (main JSON parse failed)
    if state.is_fallback and self.extraction_engine != 'langextract':
        return False  # ← Skip if fallback UNLESS using langextract

    json_objects = state.json_objects
    product_items = json_objects.get('product', [])
    # ... check for entities ...
    has_entities = len(product_items) > 0 or len(state.entities_from_kiwi) > 0

    # ✅ SPECIAL CASE: Always run if using langextract (even without entities)
    return has_entities or self.extraction_engine == 'langextract'
```

**Key Points**:
1. **Fallback resilience**: langextract runs even when `is_fallback=True`
2. **Entity-less execution**: langextract runs even when no entities are found in previous steps
3. **Rationale**: langextract is independent of main prompt, so it can extract entities from scratch

### Stage 1: LangExtract Pre-Extraction (Lines 679-705)
```python
def execute(self, state: WorkflowState) -> WorkflowState:
    # ... stage preparation ...

    # ✅ CONDITIONAL: Only runs if extraction_engine='langextract'
    pre_extracted = None
    if self.extraction_engine == 'langextract':
        try:
            from core.lx_extractor import extract_mms_entities
            logger.info("🔗 langextract 엔진으로 Stage 1 엔티티 추출 시작...")

            # Call langextract
            doc = extract_mms_entities(msg, model_id=self.llm_model)

            # Extract entities (excluding Channel, Purpose)
            entities = []
            type_pairs = []
            for ext in (doc.extractions or []):
                name = ext.extraction_text
                if ext.extraction_class in ('Channel', 'Purpose'):
                    continue
                if name not in self.stop_item_names and len(name) >= 2:
                    entities.append(name)
                    type_pairs.append(f"{name}({ext.extraction_class})")

            # Create pre_extracted context
            pre_extracted = {
                'entities': entities,
                'context_text': ", ".join(type_pairs)
            }
            logger.info(f"✅ langextract Stage 1 완료: {len(entities)}개 엔티티 추출")
        except Exception as e:
            logger.error(f"❌ langextract 추출 실패, 기본 모드로 폴백: {e}")
            pre_extracted = None
```

**Prompts Used in Stage 1**:
- **NOT** `prompts/entity_extraction_prompt.py` (HYBRID_DAG, PAIRING, ONT, TYPED, SIMPLE)
- **INSTEAD**: `core/lx_extractor.py` uses:
  - `MMS_PROMPT_DESCRIPTION` (defined in lx_extractor.py)
  - `prompts/lx_examples.build_mms_examples()` (few-shot examples)
  - `config/lx_schemas.get_class_description_text()` (entity type definitions)

### Stage 2: Entity Matching (Lines 706-776)
```python
    # Stage 2: Entity matching based on mode
    if self.entity_extraction_mode == 'logic':
        # Logic-based fuzzy matching
        cand_entities = list(set(entities_from_kiwi + [item.get('name', '') ...]))
        similarities_fuzzy = self.entity_recognizer.extract_entities_with_fuzzy_matching(cand_entities)
    else:
        # ✅ LLM-based matching with pre_extracted context
        llm_result = self.entity_recognizer.extract_entities_with_llm(
            msg,
            llm_models=default_llm_models,
            rank_limit=100,
            external_cand_entities=external_cand,
            context_mode=self.entity_extraction_context_mode,  # ← 'typed' if langextract
            pre_extracted=pre_extracted,  # ← CRITICAL: Passed to entity_recognizer
        )

        if isinstance(llm_result, dict):
            similarities_fuzzy = llm_result.get('similarities_df', pd.DataFrame())
        else:
            similarities_fuzzy = llm_result
```

---

## 5. EntityRecognizer Processing (services/entity_recognizer.py)

### extract_entities_with_llm Method (Lines 594-596)
```python
def extract_entities_with_llm(self, msg_text: str, rank_limit: int = 50, llm_models: List = None,
                            external_cand_entities: List[str] = [], context_mode: str = 'dag',
                            pre_extracted: dict = None) -> pd.DataFrame:  # ← Receives pre_extracted
```

### Pre-Extracted Path (Lines 628-710)
```python
    # ✅ CRITICAL BRANCH: Pre-extracted entities skip Stage 1 entirely
    if pre_extracted:
        logger.info("=== Using pre-extracted entities (Stage 1 skipped) ===")
        cand_entity_list = list(pre_extracted['entities'])
        combined_context = pre_extracted.get('context_text', '')
        context_keyword = 'TYPED'  # ← langextract always uses typed context

        # ... normalization, n-gram expansion ...

        # Match with products
        cand_entities_sim = self._match_entities_with_products(cand_entity_list, rank_limit)

        # ✅ Stage 2 ONLY: Vocabulary filtering using LLM
        # Uses build_context_based_entity_extraction_prompt('TYPED')
        second_stage_prompt = build_context_based_entity_extraction_prompt(context_keyword)

        prompt = f"""
        {second_stage_prompt}

        ## message:
        {msg_text}

        ## TYPED Context (Entity Types):
        {combined_context}  # ← Uses langextract type annotations

        ## entities in message:
        {', '.join(entities_in_message)}

        ## candidate entities in vocabulary:
        {', '.join(cand_entities_voca)}
        """

        # Call LLM for Stage 2 filtering only
        response = llm_model.invoke(prompt).content

        return cand_entities_sim  # ← Filtered results
```

**Key Points**:
1. **Stage 1 SKIPPED**: No `HYBRID_DAG_EXTRACTION_PROMPT`, `ONTOLOGY_PROMPT`, etc.
2. **Stage 2 ONLY**: Uses `build_context_based_entity_extraction_prompt('TYPED')`
3. **Context**: Uses langextract's type annotations (e.g., "아이폰17(Product), 을지로점(Store)")

### Default Path (Lines 712+)
```python
    # ✅ Standard path (when pre_extracted is None)

    # Select prompt based on context_mode
    if context_mode == 'dag':
        first_stage_prompt = HYBRID_DAG_EXTRACTION_PROMPT
        context_keyword = 'DAG'
    elif context_mode == 'pairing':
        first_stage_prompt = HYBRID_PAIRING_EXTRACTION_PROMPT
        context_keyword = 'PAIRING'
    elif context_mode == 'ont':
        first_stage_prompt = ONTOLOGY_PROMPT
        context_keyword = 'ONT'
    elif context_mode == 'typed':
        first_stage_prompt = TYPED_ENTITY_EXTRACTION_PROMPT
        context_keyword = 'TYPED'
    else:  # 'none'
        first_stage_prompt = SIMPLE_ENTITY_EXTRACTION_PROMPT
        context_keyword = None

    # Stage 1: Extract entities with context
    # Stage 2: Filter entities from vocabulary
    # ... full 2-stage LLM extraction ...
```

---

## Complete Workflow Trace

### Scenario 1: `--extraction-engine default`

```
User Command:
  python apps/cli.py --message "광고" --extraction-engine default

┌─────────────────────────────────────────────────────────────┐
│ CLI (apps/cli.py)                                           │
├─────────────────────────────────────────────────────────────┤
│ args.extraction_engine = 'default'                          │
│ entity_extraction_context_mode = args.entity_extraction_   │
│                                  context_mode (e.g., 'dag') │
└─────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────┐
│ MMSExtractor.__init__()                                     │
├─────────────────────────────────────────────────────────────┤
│ self.extraction_engine = 'default'                          │
│ self.entity_extraction_context_mode = 'dag'                 │
└─────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────┐
│ Workflow Step Registration                                  │
├─────────────────────────────────────────────────────────────┤
│ EntityMatchingStep(                                         │
│   extraction_engine='default',                              │
│   entity_extraction_context_mode='dag'                      │
│ )                                                            │
└─────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 7: EntityMatchingStep.execute()                        │
├─────────────────────────────────────────────────────────────┤
│ ✅ should_execute() checks:                                 │
│    - has_error? No → Continue                               │
│    - is_fallback? Yes → SKIP (because extraction_engine    │
│                               != 'langextract')             │
│                                                              │
│ OR if not fallback:                                         │
│    - has_entities? → Continue                               │
│                                                              │
│ Stage 1: SKIPPED (extraction_engine != 'langextract')       │
│   pre_extracted = None                                      │
│                                                              │
│ Stage 2: entity_recognizer.extract_entities_with_llm(       │
│   pre_extracted=None  ← NULL                                │
│ )                                                            │
└─────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────┐
│ EntityRecognizer.extract_entities_with_llm()                │
├─────────────────────────────────────────────────────────────┤
│ if pre_extracted:  ← FALSE, skip this branch                │
│                                                              │
│ ✅ DEFAULT PATH (lines 712+):                               │
│   - Select prompt: HYBRID_DAG_EXTRACTION_PROMPT             │
│   - Stage 1: Extract entities + DAG context (LLM call 1)    │
│   - Stage 2: Filter vocabulary (LLM call 2)                 │
│                                                              │
│ Result: 2 LLM calls in entity extraction                    │
└─────────────────────────────────────────────────────────────┘
```

### Scenario 2: `--extraction-engine langextract`

```
User Command:
  python apps/cli.py --message "광고" --extraction-engine langextract

┌─────────────────────────────────────────────────────────────┐
│ CLI (apps/cli.py)                                           │
├─────────────────────────────────────────────────────────────┤
│ args.extraction_engine = 'langextract'                      │
│ ✅ AUTO-CONFIG:                                             │
│   entity_extraction_context_mode = 'typed' (FORCED!)        │
└─────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────┐
│ MMSExtractor.__init__()                                     │
├─────────────────────────────────────────────────────────────┤
│ self.extraction_engine = 'langextract'                      │
│ self.entity_extraction_context_mode = 'typed'               │
└─────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────┐
│ Workflow Step Registration                                  │
├─────────────────────────────────────────────────────────────┤
│ EntityMatchingStep(                                         │
│   extraction_engine='langextract',                          │
│   entity_extraction_context_mode='typed'                    │
│ )                                                            │
└─────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 7: EntityMatchingStep.execute()                        │
├─────────────────────────────────────────────────────────────┤
│ ✅ should_execute() checks:                                 │
│    - has_error? No → Continue                               │
│    - is_fallback? Yes → CONTINUE (because extraction_engine │
│                                   == 'langextract')         │
│    - OR: Always TRUE if extraction_engine=='langextract'    │
│                                                              │
│ ✅ Stage 1: LANGEXTRACT PRE-EXTRACTION                      │
│   - Call: lx_extractor.extract_mms_entities(msg)            │
│   - Prompt: MMS_PROMPT_DESCRIPTION + lx_examples            │
│   - Result: pre_extracted = {                               │
│       'entities': ['아이폰17', '을지로점', ...],            │
│       'context_text': "아이폰17(Product), 을지로점(Store)"  │
│     }                                                        │
│   - LLM call: 1 (via langextract)                           │
│                                                              │
│ Stage 2: entity_recognizer.extract_entities_with_llm(       │
│   pre_extracted=pre_extracted  ← POPULATED                  │
│ )                                                            │
└─────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────┐
│ EntityRecognizer.extract_entities_with_llm()                │
├─────────────────────────────────────────────────────────────┤
│ ✅ if pre_extracted:  ← TRUE, use this branch               │
│                                                              │
│   PRE-EXTRACTED PATH (lines 628-710):                       │
│   - SKIP Stage 1 (no HYBRID_DAG/ONT/PAIRING prompts)        │
│   - Use pre_extracted entities directly                     │
│   - Match with products                                     │
│   - Stage 2 ONLY: Vocabulary filtering (LLM call 2)         │
│     Prompt: build_context_based_entity_extraction_prompt    │
│             ('TYPED') + pre_extracted context               │
│                                                              │
│ Result: 1 LLM call in entity extraction (Stage 2 only)      │
│         + 1 LLM call from langextract (Stage 1)             │
│         = 2 total LLM calls (same as default)               │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Differences: Default vs LangExtract

| Aspect | `--extraction-engine default` | `--extraction-engine langextract` |
|--------|-------------------------------|-----------------------------------|
| **Step 7 Stage 1** | Skipped (no pre-extraction) | ✅ Runs langextract extraction |
| **Prompts Used (Stage 1)** | None | `MMS_PROMPT_DESCRIPTION` + `lx_examples` |
| **LLM Calls (Stage 1)** | 0 | 1 (langextract) |
| **EntityRecognizer Path** | Default path (lines 712+) | Pre-extracted path (lines 628-710) |
| **EntityRecognizer Stage 1** | ✅ Runs (HYBRID_DAG/ONT/etc.) | ❌ SKIPPED |
| **EntityRecognizer Stage 2** | ✅ Runs (vocabulary filtering) | ✅ Runs (vocabulary filtering) |
| **LLM Calls (EntityRecognizer)** | 2 (Stage 1 + Stage 2) | 1 (Stage 2 only) |
| **Total LLM Calls (Step 7)** | 2 | 2 (1 langextract + 1 filtering) |
| **Entity Types** | Extracted by LLM in Stage 1 | ✅ Explicitly typed (Product, Store, etc.) |
| **Context Mode** | User-specified (dag/pairing/ont/typed/none) | ✅ FORCED to 'typed' |
| **Fallback Resilience** | ❌ Skips if `is_fallback=True` | ✅ Runs even if `is_fallback=True` |
| **Entity-less Execution** | ❌ Skips if no entities found | ✅ Runs even without entities |

---

## Prompt Usage Summary

### Default Engine Prompts (from `prompts/entity_extraction_prompt.py`)
- ✅ Used in EntityRecognizer Stage 1:
  - `HYBRID_DAG_EXTRACTION_PROMPT` (if context_mode='dag')
  - `HYBRID_PAIRING_EXTRACTION_PROMPT` (if context_mode='pairing')
  - `ONTOLOGY_PROMPT` (if context_mode='ont')
  - `TYPED_ENTITY_EXTRACTION_PROMPT` (if context_mode='typed')
  - `SIMPLE_ENTITY_EXTRACTION_PROMPT` (if context_mode='none')
- ✅ Used in EntityRecognizer Stage 2:
  - `build_context_based_entity_extraction_prompt(context_keyword)`

### LangExtract Engine Prompts
- ✅ Stage 1 (lx_extractor.py):
  - `MMS_PROMPT_DESCRIPTION` (custom prompt for Korean MMS)
  - `prompts/lx_examples.build_mms_examples()` (few-shot examples)
  - `config/lx_schemas.get_class_description_text()` (entity type definitions)
- ✅ Stage 2 (entity_recognizer.py):
  - `build_context_based_entity_extraction_prompt('TYPED')` ONLY
  - Uses `pre_extracted['context_text']` for entity type annotations

**Critical Finding**: LangExtract does NOT use `prompts/entity_extraction_prompt.py` at all in Stage 1!

---

## Architecture Impact

### Current (Single Step with Two Stages)
- **Pros**:
  - All entity extraction in one place
  - Simpler workflow (10 steps)
- **Cons**:
  - Mixed responsibilities (extraction + matching)
  - Complex conditional logic in `should_execute()`
  - Hidden Stage 1 (not visible in workflow logs)
  - Different prompt systems mixed in one step

### Proposed (Split into Two Steps)
- **Pros**:
  - Clear separation: Step 7A = Extraction, Step 7B = Matching
  - Simple `should_execute()` per step
  - Explicit workflow visibility
  - Better observability (separate timing per step)
- **Cons**:
  - 10 → 11 steps
  - +1 class, +1 state field

**Recommendation**: Split is justified given the distinct purposes and prompt systems used.

---

## Conclusion

The `--extraction-engine` parameter fundamentally changes how Step 7 operates:

1. **Flow Control**: Determines whether Stage 1 pre-extraction runs
2. **Prompt Selection**: Completely different prompt systems (entity_extraction_prompt.py vs lx_extractor prompts)
3. **Context Mode**: Forces 'typed' mode when langextract is used
4. **Resilience**: langextract runs even in fallback/entity-less scenarios
5. **LLM Call Pattern**: Same total (2 calls), but different stages execute

This bi-modal behavior (with/without Stage 1) is a strong indicator that **splitting into two steps** would improve clarity and maintainability.

---

*Analysis Date: 2026-02-11*
*Next Steps: Decide whether to split Step 7 into two separate workflow steps*
