# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**langextract** is an AI-powered structured information extraction system for Korean MMS (multimedia message) advertisements. It takes unstructured Korean ad text and extracts entities (products, channels, programs), relationships, and structured metadata using a combination of LLM calls, Korean NLP (Kiwi morphological analyzer), fuzzy/sequence matching, and embedding similarity.

Based on the `mms_extractor_exp` architecture within the AgenticWorkflow monorepo.

## Python Environment

```bash
/Users/yongwook/workspace/AgenticWorkflow/venv/bin/python  # Python 3.12
```

Always use this venv for execution. The project has no `pyproject.toml` or `setup.py` — dependencies are in `requirements.txt`.

## Common Commands

```bash
# Single message extraction (default LLM: ax, data source: db)
python apps/cli.py --message "광고 메시지 텍스트" --offer-data-source local

# With DAG extraction
python apps/cli.py --message "메시지" --extract-entity-dag --offer-data-source local

# Batch processing
python apps/cli.py --batch-file messages.txt --offer-data-source local

# Choose LLM model (gem/ax/cld/gen/gpt/opus) and entity context mode (dag/pairing/none/ont/typed)
python apps/cli.py --message "메시지" --llm-model gpt --entity-extraction-context-mode ont

# Skip Kiwi+fuzzy pre-extraction (Step 2)
python apps/cli.py --message "메시지" --skip-entity-extraction

# API server
python apps/api.py --host 0.0.0.0 --port 8000

# Trace/debug product extraction pipeline (detailed step-by-step output)
python tests/trace_product_extraction.py --message "메시지" --data-source local

# Run workflow tests
python tests/test_workflow.py
```

Use `--data-source local` or `--offer-data-source local` to avoid Oracle DB dependency during development.

## Architecture

### 10-Step Workflow Pipeline

The core of the system is `WorkflowEngine` which executes a linear pipeline of `WorkflowStep` subclasses. Each step reads/writes to a shared `WorkflowState` dataclass. Steps can be conditionally skipped via `should_execute()`.

```
Step 1  InputValidationStep     → validates/trims message
Step 2  EntityExtractionStep    → Kiwi NLP + embedding similarity → candidate entities
Step 3  ProgramClassificationStep → embedding cosine similarity → top-N programs
Step 4  ContextPreparationStep  → builds RAG context string for LLM prompt
Step 5  LLMExtractionStep       → calls LLM with prompt → JSON response
Step 6  ResponseParsingStep     → parses JSON, detects schema responses
Step 7  EntityMatchingStep      → matches LLM-extracted names to DB products (fuzzy/LLM)
Step 8  ResultConstructionStep  → assembles final result (channels, programs, offers)
Step 9  ValidationStep          → validates required fields
Step 10 DAGExtractionStep       → (optional) extracts entity relationship graph
```

Steps 2 & 3 are independent and could be parallelized. Step 7 has `should_execute()` that skips when there are errors, fallback, or no product items.

### Key Classes & Files

| File | Class/Role |
|------|-----------|
| `core/workflow_core.py` | `WorkflowEngine`, `WorkflowState`, `WorkflowStep` (ABC) |
| `core/mms_workflow_steps.py` | All 10 step implementations |
| `core/mms_extractor.py` | `MMSExtractor` — orchestrator, initializes services and workflow |
| `core/mms_extractor_data.py` | `MMSExtractorDataMixin` — data loading mixin |
| `core/entity_dag_extractor.py` | `DAGParser`, `extract_dag()` — DAG graph extraction |
| `services/entity_recognizer.py` | `EntityRecognizer` — Kiwi + fuzzy + LLM entity extraction |
| `services/program_classifier.py` | `ProgramClassifier` — embedding-based program classification |
| `services/store_matcher.py` | `StoreMatcher` — store/organization matching |
| `services/result_builder.py` | `ResultBuilder` — final result assembly |
| `services/item_data_loader.py` | `ItemDataLoader` — item data loading + alias expansion |
| `config/settings.py` | Dataclass configs: `API_CONFIG`, `MODEL_CONFIG`, `PROCESSING_CONFIG`, `METADATA_CONFIG`, `EMBEDDING_CONFIG`, `STORAGE_CONFIG`, `DATABASE_CONFIG` |
| `utils/similarity_utils.py` | Bigram-prefiltered fuzzy/sequence matching (perf-critical) |
| `utils/llm_factory.py` | `LLMFactory` — creates LLM model instances |
| `prompts/` | All prompt templates (main extraction, entity extraction, DAG, ontology, retry) |
| `apps/cli.py` | CLI entry point |
| `apps/api.py` | Flask REST API server |
| `apps/batch.py` | Batch processing |

### Data Flow

`MMSExtractor.__init__()` loads: item data (45K+ product aliases from CSV/DB), program classification data, organization data, stop words, embedding model (ko-sbert-nli), Kiwi morphological analyzer, and LLM model.

`MMSExtractor.process_message(msg)` creates a `WorkflowState`, runs all steps via `WorkflowEngine.run()`, and returns `{ext_result, raw_result, prompts}`.

### Configuration

All config via `config/settings.py` dataclass singletons. Environment variables loaded from `.env` file override defaults. Key config groups:
- `API_CONFIG` — API keys/endpoints for LLM services
- `MODEL_CONFIG` — model names, loading mode (auto/local/remote), temperature
- `PROCESSING_CONFIG` — similarity thresholds, extraction modes (llm/logic, rag/llm/nlp), entity matching thresholds
- `METADATA_CONFIG` — paths to CSV data files (alias rules, stop words, offer data, org info, program info)
- `EMBEDDING_CONFIG` — embedding model and cache paths

### Entity Extraction Modes

- **product_info_extraction_mode**: `rag` (candidates in prompt), `llm` (reference candidates), `nlp` (pre-built elements)
- **entity_extraction_mode**: `llm` (2-stage LLM matching), `logic` (fuzzy + sequence similarity)
- **entity_extraction_context_mode**: `dag`, `pairing`, `none`, `ont` (ontology), `typed` (6-type)

### Performance Notes

- `similarity_utils.py` uses bigram pre-filtering index to reduce 904K → 48K comparisons (94.7% reduction)
- Single-process execution for <100K candidates (joblib IPC overhead dominates for smaller sets)
- Step 2 runs in ~1.5s (down from 16.2s after optimization)
- Local embedding model at `./models/ko-sbert-nli` avoids download latency

## Git

```bash
/usr/local/bin/git   # Must use Homebrew git; /usr/bin/git fails with Xcode license error
```

Git repo root is `/Users/yongwook/workspace/AgenticWorkflow/`, not the package directory.

## Data Files

Located in `data/` directory:
- `item_info_all_250527.csv` — product catalog (~45K aliases, columns: item_nm, item_id, item_nm_alias)
- `alias_rules.csv` — alias expansion rules
- `stop_words.csv` — stop words for entity filtering
- `pgm_tag_ext_250516.csv` — program classification tags
- `org_info_all_250605.csv` — organization/store info
- `offer_master_data*.csv` — offer master data

## Language Context

Code comments, docstrings, log messages, and data are predominantly in Korean. The system processes Korean MMS advertisement text. Be prepared to work with Korean language content throughout.
