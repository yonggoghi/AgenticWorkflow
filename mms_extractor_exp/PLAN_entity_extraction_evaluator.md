# Plan: Entity Extraction Evaluator Script - COMPLETED

## Objective
Create a Python script to evaluate LLM-based product entity extraction by:
1. Sampling 50 representative MMS messages (stratified by msg_nm patterns)
2. Running the full extraction workflow using `ProductExtractionTracer`
3. Capturing entities from the 1st stage of `HYBRID_DAG_EXTRACTION_PROMPT`
4. Saving results as CSV for human annotation

## Implementation: Option B - Leverage Existing Tracer ✅

### Script Location
`tests/generate_entity_extraction_eval.py`

### Categories for Stratified Sampling
Based on `msg_nm` pattern analysis:

| Category | Count | % |
|----------|-------|---|
| 기타 | 259 | 24.3% |
| 0 day 혜택 | 186 | 17.5% |
| 대리점 | 163 | 15.3% |
| T day 혜택 | 110 | 10.3% |
| 이벤트/프로모션 | 93 | 8.7% |
| 서비스/요금제 | 82 | 7.7% |
| 통화서비스 | 58 | 5.4% |
| special T/장기고객 | 52 | 4.9% |
| T 우주 | 45 | 4.2% |
| 단말기 | 17 | 1.6% |

### Output Format
```csv
mms,prev_extracted_entities,extracted_entities,correct_entities
"(광고)[SKT] T day 혜택 안내...","","T day, 스타벅스",""
```

**Columns:**
- `mms`: Original MMS message text
- `prev_extracted_entities`: Previous extraction result (empty for new evaluation, populated during re-evaluation)
- `extracted_entities`: Current extraction result from HYBRID_DAG_EXTRACTION_PROMPT
- `correct_entities`: Human annotated correct entities (empty initially)

### CLI Usage
```bash
/Users/yongwook/workspace/AgenticWorkflow/venv/bin/python \
    tests/generate_entity_extraction_eval.py \
    --input-file data/mms_data_251001_260205.csv \
    --output-dir outputs/ \
    --sample-size 50 \
    --llm-model ax \
    --random-seed 42
```

### Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--input-file` | `data/mms_data_251001_260205.csv` | Input CSV file |
| `--output-dir` | `outputs/` | Output directory |
| `--sample-size` | 50 | Number of samples |
| `--llm-model` | `ax` | LLM model (gem/ax/cld/gen/gpt) |
| `--random-seed` | 42 | Random seed for reproducibility |
| `--re-evaluate` | None | Re-evaluate existing CSV (preserves correct_entities, moves extracted_entities to prev_extracted_entities) |

### Test Results ✅
Tested with 3 samples - working correctly:
- Stratified sampling working
- Entity extraction via `ProductExtractionTracer` working
- CSV output with UTF-8 BOM for Excel compatibility

### Notes
- Uses `offer_info_data_src='local'` (CSV files) instead of Oracle DB
- Captures `first_stage_entities` from `HYBRID_DAG_EXTRACTION_PROMPT` response
- Estimated runtime: ~10-15 minutes for 50 samples
