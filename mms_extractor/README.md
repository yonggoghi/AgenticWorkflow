# MMS Extractor

A Python package for extracting structured information from MMS (Multimedia Messaging Service) messages using NLP and machine learning techniques.

## Features

- 🚀 **Fast API responses** - Models preloaded at startup (no more 25s delays!)
- 🔍 Entity extraction using Korean NLP (Kiwi)
- 🧮 Embedding-based similarity matching
- 🤖 LLM-powered information extraction
- 💾 Local model loading support with multiple modes
- ⚙️ Flexible configuration system

## Installation

```bash
pip install -r requirements.txt
```

## Model Loading Modes

The package supports three model loading modes for maximum flexibility:

### 🔧 Setup Options

1. **Download and save the model locally** (one-time setup):
   ```bash
   python download_model.py
   ```

2. **Choose your loading mode**:

   **Via Environment Variables:**
   ```bash
   export MODEL_LOADING_MODE=auto     # Default: auto-detect
   export MODEL_LOADING_MODE=local    # Offline only
   export MODEL_LOADING_MODE=remote   # Always download
   ```

   **Via Command Line:**
   ```bash
   python run.py --model-loading-mode local --text "your message"
   python api.py --model-loading-mode remote --port 8080
   ```

### 📋 Loading Mode Details

| Mode | Description | Use Case |
|------|-------------|----------|
| **`auto`** | Try local first, fallback to remote | **Default** - Best balance |
| **`local`** | Only use local models (fail if not found) | Offline environments |
| **`remote`** | Always download from internet | Testing, ensure latest version |

### 💡 Configuration Options

- **Local model path**: Set `LOCAL_EMBEDDING_MODEL_PATH` environment variable
- **Loading mode**: Set `MODEL_LOADING_MODE` environment variable
- Or modify values in `config/settings.py`

## Usage

```python
from mms_extractor import MMSExtractor
from mms_extractor.core.data_manager import DataManager

# Initialize data manager and extractor
data_manager = DataManager()
extractor = MMSExtractor(data_manager)

# Load data
extractor.load_data()

# Extract information from MMS message
message = "Your MMS message here"
result = extractor.extract(message)

print(result)
```

## Configuration

The package uses a configuration system located in `config/settings.py`. Key settings include:

- `MODEL_CONFIG.embedding_model`: The embedding model name
- `MODEL_CONFIG.local_embedding_model_path`: Path to local model directory
- `PROCESSING_CONFIG.product_info_extraction_mode`: Extraction mode ('rag', 'llm', 'nlp')

## Model Information

The package uses the `jhgan/ko-sbert-nli` model for Korean sentence embeddings. This model is automatically downloaded and saved locally for faster subsequent loading.

## Local vs. Remote Model Loading

- **Local loading**: Faster startup time, no internet required after initial download
- **Remote loading**: Always gets the latest model version, but requires internet connection

The system automatically detects if a local model exists and uses it, otherwise falls back to downloading from the internet.

### 🧪 Testing Loading Modes

Test all loading modes to see how they work:

```bash
python test_loading_modes.py
```

This script will test all three modes and provide recommendations based on your setup.
