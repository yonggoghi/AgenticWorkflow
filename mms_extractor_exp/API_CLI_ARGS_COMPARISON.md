# API.py vs CLI.py Arguments Comparison

**Date**: 2026-02-11

## 📊 Missing Arguments in api.py

The following arguments exist in **cli.py** but are **missing in api.py**:

### 1. **Batch Processing**
```python
# cli.py has:
--batch-file            # Batch processing from file
--max-workers          # Number of parallel workers

# api.py: MISSING (relies on API endpoint /extract/batch instead)
```

### 2. **Advanced Entity Extraction Control**
```python
# cli.py has:
--entity-llm-model                    # Separate LLM for entity extraction (ax, gpt, opus, etc.)
--entity-extraction-context-mode      # Context mode: dag, pairing, none, ont, typed
--skip-entity-extraction              # Skip entity extraction step
--no-external-candidates              # Disable external candidate sources

# api.py: ALL MISSING
```

### 3. **Message Metadata**
```python
# cli.py has:
--message-id           # Message identifier (default: '#')

# api.py: MISSING (relies on JSON request body instead)
```

### 4. **Storage & Persistence**
```python
# cli.py has:
--save-to-mongodb      # Save results to MongoDB (default: True)
--save-batch-results   # Save batch results to JSON files
--test-mongodb         # Test MongoDB connection only

# api.py: MISSING
```

### 5. **Extraction Engine Selection**
```python
# cli.py has:
--extraction-engine    # Choose: default (11-step) or langextract

# api.py: MISSING
```

### 6. **LLM Model Choices**
```python
# cli.py has:
--llm-model choices: ['gem', 'ax', 'cld', 'gen', 'gpt', 'opus']
--entity-llm-model choices: ['gem', 'ax', 'cld', 'gen', 'gpt', 'opus']

# api.py has:
--llm-model choices: ['gem', 'ax', 'cld', 'gen', 'gpt']  # MISSING: 'opus'
```

---

## ✅ Arguments Present in BOTH

| Argument | cli.py Default | api.py Default | Notes |
|----------|---------------|----------------|-------|
| `--message` | - | - | Test message text |
| `--offer-data-source` | `db` | `db` | local or db |
| `--product-info-extraction-mode` | `llm` | `nlp` | ⚠️ Different defaults! |
| `--entity-matching-mode` | `llm` | `llm` | Same |
| `--llm-model` | `ax` | `ax` | Same (but cli has 'opus') |
| `--log-level` | `INFO` | `INFO` | Same |
| `--extract-entity-dag` | `False` | `False` | Same |

---

## ⚠️ Different Defaults

### product-info-extraction-mode
- **cli.py**: `default='llm'` ← Recommended for 11-step workflow
- **api.py**: `default='nlp'` ← Old default, should update to 'llm'

---

## 🎯 Why the Differences?

### API Design Philosophy
api.py relies on **JSON request bodies** for most configuration:
```python
# API request format
{
  "message": "...",
  "message_id": "test_123",           # Instead of --message-id
  "llm_model": "ax",                   # Instead of --llm-model
  "entity_matching_mode": "llm",       # Instead of --entity-matching-mode
  "extract_entity_dag": true,          # Instead of --extract-entity-dag
  "save_to_mongodb": true,             # Instead of --save-to-mongodb
  "extraction_engine": "default",      # MISSING in api.py args!
  "entity_llm_model": "opus",          # MISSING in api.py args!
  "entity_extraction_context_mode": "dag"  # MISSING in api.py args!
}
```

### CLI Design Philosophy
cli.py uses **command-line arguments** for all configuration because it's meant for:
- Direct script execution
- Batch file processing (`--batch-file`)
- Automation and shell scripts
- Testing with specific configurations

---

## 🔧 Recommended Fixes for api.py

### 1. Add Missing Arguments (High Priority)
```python
# Add to api.py argument parser:
parser.add_argument('--entity-llm-model', choices=['gem', 'ax', 'cld', 'gen', 'gpt', 'opus'],
                   default='ax', help='Entity extraction LLM 모델')
parser.add_argument('--entity-extraction-context-mode',
                   choices=['dag', 'pairing', 'none', 'ont', 'typed'], default='dag',
                   help='엔티티 추출 컨텍스트 모드')
parser.add_argument('--extraction-engine', choices=['default', 'langextract'],
                   default='default', help='추출 엔진 (default: 11-step, langextract: Google)')
parser.add_argument('--skip-entity-extraction', action='store_true', default=False,
                   help='엔티티 추출 단계 건너뛰기')
parser.add_argument('--no-external-candidates', action='store_true', default=False,
                   help='외부 후보 비활성화')
```

### 2. Add 'opus' to LLM Choices (High Priority)
```python
# Update existing argument:
parser.add_argument('--llm-model',
                   choices=['gem', 'ax', 'cld', 'gen', 'gpt', 'opus'],  # Add 'opus'
                   default='ax', help='사용할 LLM 모델')
```

### 3. Fix Default for product-info-extraction-mode (Medium Priority)
```python
# Update from 'nlp' to 'llm' to match cli.py:
parser.add_argument('--product-info-extraction-mode',
                   choices=['nlp', 'llm', 'rag'],
                   default='llm',  # Changed from 'nlp'
                   help='상품 정보 추출 모드')
```

### 4. Add MongoDB/Storage Arguments (Low Priority)
These are less critical for API server since API endpoints handle storage differently:
```python
parser.add_argument('--save-to-mongodb', action='store_true', default=False,
                   help='MongoDB 저장 활성화')
parser.add_argument('--test-mongodb', action='store_true', default=False,
                   help='MongoDB 연결 테스트')
```

---

## 📋 Summary

| Category | Missing in api.py | Priority |
|----------|-------------------|----------|
| Advanced entity extraction | 4 args | **HIGH** |
| Extraction engine | 1 arg | **HIGH** |
| LLM model 'opus' | 1 arg | **HIGH** |
| Wrong default (nlp vs llm) | 1 arg | **MEDIUM** |
| Batch processing | 2 args | **LOW** (API has endpoints) |
| MongoDB/storage | 3 args | **LOW** (API has request params) |
| Message ID | 1 arg | **LOW** (API has request params) |

**Total missing/mismatched**: 13 arguments

**Recommendation**: Add the HIGH priority arguments (6 total) to ensure api.py supports all the advanced features that cli.py provides, especially for the 11-step workflow with different context modes and extraction engines.
