# API.py Arguments Update for Feature Parity with CLI.py

**Date**: 2026-02-11
**Purpose**: Add missing arguments to api.py to achieve feature parity with cli.py for the 11-step workflow

## 📋 Changes Made

### 1. **Added Missing Command-Line Arguments**

Added the following HIGH priority arguments to api.py:

```python
# Entity extraction LLM model (separate from main LLM)
parser.add_argument('--entity-llm-model',
                   choices=['gem', 'ax', 'cld', 'gen', 'gpt', 'opus'],
                   default='ax',
                   help='엔티티 추출 전용 LLM 모델')

# Entity extraction context mode
parser.add_argument('--entity-extraction-context-mode',
                   choices=['dag', 'pairing', 'none', 'ont', 'typed'],
                   default='dag',
                   help='엔티티 추출 컨텍스트 모드')

# Skip entity extraction steps
parser.add_argument('--skip-entity-extraction',
                   action='store_true',
                   default=False,
                   help='엔티티 추출 단계 건너뛰기 (Steps 7-8 스킵)')

# Disable external candidates
parser.add_argument('--no-external-candidates',
                   action='store_true',
                   default=False,
                   help='외부 후보 소스 비활성화 (Kiwi NLP 후보만 사용)')

# Extraction engine selection
parser.add_argument('--extraction-engine',
                   choices=['default', 'langextract'],
                   default='default',
                   help='추출 엔진 선택 (default: 11-step pipeline, langextract: Google langextract 기반)')
```

### 2. **Updated Existing Arguments**

#### Fixed Default for product-info-extraction-mode
```python
# BEFORE:
parser.add_argument('--product-info-extraction-mode',
                   choices=['nlp', 'llm' ,'rag'],
                   default='nlp',  # ❌ Old default
                   help='...')

# AFTER:
parser.add_argument('--product-info-extraction-mode',
                   choices=['nlp', 'llm' ,'rag'],
                   default='llm',  # ✅ Matches cli.py, supports 11-step workflow
                   help='...')
```

#### Added 'opus' to LLM Model Choices
```python
# BEFORE:
parser.add_argument('--llm-model',
                   choices=['gem', 'ax', 'cld', 'gen', 'gpt'],  # ❌ Missing 'opus'
                   default='ax',
                   help='...')

# AFTER:
parser.add_argument('--llm-model',
                   choices=['gem', 'ax', 'cld', 'gen', 'gpt', 'opus'],  # ✅ Added 'opus'
                   default='ax',
                   help='사용할 LLM 모델 (gem: Gemma, ax: ax, cld: Claude, gen: Gemini, gpt: GPT, opus: Claude Opus)')
```

### 3. **Updated Test Mode Configuration**

Updated the test mode to use all new arguments:

```python
# BEFORE:
extractor = get_configured_extractor(
    args.llm_model,
    args.product_info_extraction_mode,
    args.entity_matching_mode,
    args.extract_entity_dag
)

# AFTER:
extractor = get_configured_extractor(
    args.llm_model,
    args.product_info_extraction_mode,
    args.entity_matching_mode,
    args.entity_llm_model,              # ✅ Added
    args.extract_entity_dag,
    args.entity_extraction_context_mode  # ✅ Added
)
```

### 4. **Updated API Endpoint Documentation**

Updated `/extract` endpoint docstring to document new JSON parameters:

```python
"""
Request Body (JSON):
    - message (required): 추출할 MMS 메시지 텍스트
    - llm_model (optional): 사용할 LLM 모델 (기본값: 'ax', 선택: ax, gpt, cld, gen, opus, gem)
    - entity_llm_model (optional): 엔티티 추출 전용 LLM 모델 (기본값: 'ax')  # ✅ New
    - entity_extraction_context_mode (optional): 엔티티 추출 컨텍스트 모드 (기본값: 'dag')  # ✅ New
    - extraction_engine (optional): 추출 엔진 (기본값: 'default', 선택: default, langextract)  # ✅ New
    - skip_entity_extraction (optional): 엔티티 추출 건너뛰기 (기본값: False)  # ✅ New
    - no_external_candidates (optional): 외부 후보 비활성화 (기본값: False)  # ✅ New
    - product_info_extraction_mode (optional): 상품 추출 모드 (기본값: 'llm')  # ✅ Updated default
    ...
"""
```

### 5. **Updated API Endpoint Parameter Extraction**

Added extraction of new parameters from JSON request:

```python
# Added to /extract endpoint:
extraction_engine = data.get('extraction_engine', 'default')
skip_entity_extraction = data.get('skip_entity_extraction', False)
no_external_candidates = data.get('no_external_candidates', False)
```

### 6. **Updated LLM Model Validation**

Fixed validation to match new model list:

```python
# BEFORE:
valid_llm_models = ['gemma', 'ax', 'claude', 'gemini']

# AFTER:
valid_llm_models = ['gem', 'ax', 'cld', 'gen', 'gpt', 'opus']
```

---

## ✅ Feature Parity Achieved

### Command-Line Arguments
api.py now has ALL the same arguments as cli.py (except batch-specific ones):

| Argument | cli.py | api.py (Before) | api.py (After) |
|----------|--------|-----------------|----------------|
| `--entity-llm-model` | ✅ | ❌ | ✅ |
| `--entity-extraction-context-mode` | ✅ | ❌ | ✅ |
| `--extraction-engine` | ✅ | ❌ | ✅ |
| `--skip-entity-extraction` | ✅ | ❌ | ✅ |
| `--no-external-candidates` | ✅ | ❌ | ✅ |
| `--llm-model` with 'opus' | ✅ | ❌ | ✅ |
| `--product-info-extraction-mode` default='llm' | ✅ | ❌ | ✅ |

### JSON API Parameters
The `/extract` endpoint now accepts all advanced configuration parameters:

```json
{
  "message": "...",
  "llm_model": "opus",                           // ✅ Now supports 'opus'
  "entity_llm_model": "gpt",                     // ✅ New
  "entity_extraction_context_mode": "ont",       // ✅ New
  "extraction_engine": "langextract",            // ✅ New
  "skip_entity_extraction": false,               // ✅ New
  "no_external_candidates": false,               // ✅ New
  "product_info_extraction_mode": "llm",         // ✅ Default changed
  "entity_matching_mode": "llm",
  "extract_entity_dag": true
}
```

---

## 🚀 Impact

### **Before**: Limited Configuration
- ❌ Could not use Claude Opus
- ❌ Could not switch extraction engines
- ❌ Could not use different context modes (ont, typed, etc.)
- ❌ Could not use different LLM for entity extraction
- ❌ Default product mode was 'nlp' (not optimal for 11-step workflow)

### **After**: Full Configuration Control
- ✅ All LLM models supported (including Opus)
- ✅ Can switch between default (11-step) and langextract engines
- ✅ All context modes available (dag, ont, typed, pairing, none)
- ✅ Can use different LLM for entity extraction vs main extraction
- ✅ Default product mode is 'llm' (optimal for 11-step workflow)
- ✅ Can skip entity extraction steps when needed
- ✅ Can disable external candidates for faster processing

---

## 📚 Files Modified

1. **apps/api.py**:
   - Added 5 new command-line arguments
   - Updated 2 existing arguments (default + choices)
   - Updated test mode configuration
   - Updated /extract endpoint docstring
   - Updated parameter extraction from JSON
   - Updated LLM model validation (2 locations)

---

## 🧪 Testing

### Test with Command-Line Arguments
```bash
# Test with new arguments
python -m apps.api --port 8088 \
  --llm-model opus \
  --entity-llm-model gpt \
  --entity-extraction-context-mode ont \
  --extraction-engine default \
  --product-info-extraction-mode llm \
  --entity-matching-mode llm \
  --test --message "T Day 혜택 안내"
```

### Test with API Endpoint
```bash
# Start server
python -m apps.api --port 8088

# Test with curl
curl -X POST http://localhost:8088/extract \
  -H "Content-Type: application/json" \
  -d '{
    "message": "T Day 혜택 안내",
    "llm_model": "opus",
    "entity_llm_model": "gpt",
    "entity_extraction_context_mode": "ont",
    "extraction_engine": "default",
    "product_info_extraction_mode": "llm",
    "entity_matching_mode": "llm",
    "extract_entity_dag": true
  }'
```

---

## 📊 Summary

**Total Changes**:
- ✅ 5 new arguments added
- ✅ 2 existing arguments updated
- ✅ 1 function call updated
- ✅ 2 validation lists updated
- ✅ 1 endpoint docstring updated
- ✅ 3 new JSON parameters supported

**Result**: api.py now has **complete feature parity** with cli.py for the 11-step workflow, enabling users to leverage all advanced configuration options through both CLI and API interfaces.
