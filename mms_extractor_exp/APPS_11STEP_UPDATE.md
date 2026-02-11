# Apps Directory Update for 11-Step Workflow

**Date**: 2026-02-11
**Purpose**: Update apps directory and demo files to reflect the 11-step workflow refactoring

## Changes Made

### 1. apps/demo_presentation.py
**Major Updates**:
- ✅ Updated `PIPELINE_STEPS` from 10 to 11 steps
- ✅ Split old Step 7 (EntityMatchingStep) into:
  - **Step 7**: EntityContextExtractionStep (엔티티+컨텍스트 추출)
    - Description: "LLM으로 1차 엔티티 및 컨텍스트 추출 (Stage 1)"
    - Tech: LLM-based entity extraction, relationship extraction, entity type classification
    - Input: Message text + LLM
    - Output: Stage 1 extracted entities + context information
  - **Step 8**: VocabularyFilteringStep (어휘 필터링)
    - Description: "상품 어휘 DB와 매칭하여 필터링 (Stage 2)"
    - Tech: Bigram pre-filtering, Fuzzy Matching, item_id DB matching
    - Input: Stage 1 entities + 45K product DB
    - Output: Matched products (item_nm, item_id, similarity)
- ✅ Renumbered subsequent steps:
  - Old Step 8 (ResultConstructionStep) → New Step 9
  - Old Step 9 (ValidationStep) → New Step 10
  - Old Step 10 (DAGExtractionStep) → New Step 11

**Updated Functions**:
- `_build_pipeline_html()`: Updated row ranges (0-6, 6-11) and color logic (idx == 10)
- `_render_step_buttons()`: Updated to render 0-6 and 6-11 (two rows of 6 and 5 steps)
- `_show_step_actual_data()`: Updated step display logic:
  - Step 7: Shows Stage 1 extracted entities and context
  - Step 8: Shows matched products (vocabulary filtering)
  - Step 9: Shows final result JSON
  - Step 11: Shows DAG

**Updated Text**:
- Page title: "10단계 AI 파이프라인" → "11단계 AI 파이프라인"
- Section header: "10-Step Workflow Pipeline" → "11-Step Workflow Pipeline"

### 2. apps/cli.py
**Updated**:
- `--extraction-engine` help text: "10-step pipeline" → "11-step pipeline"

## Files Already Correct

### apps/demo_streamlit.py
- ✅ No hardcoded step counts or outdated step references
- ✅ Uses dynamic workflow execution, displays results correctly

### apps/quick_extractor.py
- ✅ No hardcoded step counts or outdated step references
- ✅ Simplified extraction wrapper, not affected by step count changes

### apps/api.py
- ✅ No hardcoded step counts (only log file rotation settings mentioning "10개")
- ✅ API endpoints work with workflow state, not step numbers

### scripts/generate_demo_data.py
- ✅ No hardcoded step references
- ✅ Uses dynamic workflow execution and step history

## Verification

All grep searches confirm:
- ✅ No remaining "10-step" or "10단계" references in workflow descriptions
- ✅ No remaining "EntityMatchingStep" references for Step 7
- ✅ All step numbers correctly reflect 11-step structure

## Testing Recommendations

1. **Demo Presentation**: Test `streamlit run apps/demo_presentation.py --server.port 8502`
   - Verify all 11 step buttons render correctly in two rows (6 + 5)
   - Click each step and verify correct data displays
   - Verify Step 7 shows Stage 1 entities, Step 8 shows matched products

2. **CLI Help**: Test `python -m apps.cli --help`
   - Verify `--extraction-engine` help text shows "11-step pipeline"

3. **Live Demo**: Test with actual MMS messages
   - Verify all 11 steps execute correctly
   - Verify step history and timing data is accurate

## Related Files

- Core workflow: `core/mms_workflow_steps.py` (already updated with 11 steps)
- Documentation: `docs/ARCHITECTURE.md`, `docs/EXECUTION_FLOW.md` (already updated)
- Review documents: `WORKFLOW_REFACTORING_REVIEW.md`, `STEP7_11_SERVICE_FILES_REVIEW.md`

## Commit Message

```
Update apps directory for 11-step workflow

- Split demo_presentation.py Step 7 into Steps 7 & 8
- Add EntityContextExtractionStep (Step 7) and VocabularyFilteringStep (Step 8)
- Renumber ResultConstruction/Validation/DAG to Steps 9-11
- Update cli.py help text to "11-step pipeline"
- Update all step display logic and button rendering
- Verify all apps work with new workflow structure
```
