# AgenticWorkflow - MMS Extractor

A comprehensive Python package for extracting structured information from MMS (Multimedia Messaging Service) messages using advanced NLP and machine learning techniques.

## 🚀 Features

- **Fast API responses** - Models preloaded at startup for instant processing
- **Entity extraction** - Advanced Korean NLP using Kiwi tokenizer
- **Embedding-based similarity** - Semantic matching with Korean SBERT
- **LLM-powered extraction** - Support for multiple LLM providers (OpenAI, Anthropic, Gemma)
- **Flexible model loading** - Local, remote, and auto-detection modes
- **RESTful API** - Easy integration with Flask-based web service
- **Multiple extraction modes** - NLP, LLM, and RAG-based approaches

## 📦 Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Quick Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yonggoghi/AgenticWorkflow.git
   cd AgenticWorkflow
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Edit .env with your API keys
   nano .env
   ```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# API Keys (required for LLM features)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
HUGGINGFACE_API_KEY=your_huggingface_token_here

# Model configuration
MODEL_LOADING_MODE=auto
LOCAL_EMBEDDING_MODEL_PATH=./models/ko-sbert-nli

# API configuration
FLASK_ENV=production
API_PORT=8080
```

### Model Loading Modes

The system supports three flexible model loading modes:

| Mode | Description | Use Case |
|------|-------------|----------|
| **`auto`** | Try local first, fallback to remote | **Default** - Best balance |
| **`local`** | Only use local models | Offline environments |
| **`remote`** | Always download from internet | Testing, latest versions |

## 🚦 Quick Start

### Python API

```python
from mms_extractor import MMSExtractor
from mms_extractor.core.data_manager import DataManager

# Initialize
data_manager = DataManager()
extractor = MMSExtractor(data_manager)
extractor.load_data()

# Extract information
message = "안녕하세요, 삼성 갤럭시 S23 구매 문의드립니다."
result = extractor.extract(message)
print(result)
```

### REST API

1. **Start the API server**
   ```bash
   cd mms_extractor
   python api.py --port 8080
   ```

2. **Make requests**
   ```bash
   curl -X POST http://localhost:8080/extract \
     -H "Content-Type: application/json" \
     -d '{"message": "iPhone 15 Pro 구매 문의", "model": "gemma_3"}'
   ```

## 📊 Extraction Modes

### NLP Mode
- Direct entity extraction using Korean NLP
- Fast processing, rule-based approach
- Best for structured, predictable messages

### LLM Mode  
- Large Language Model processing
- Supports GPT-4, Claude, Gemma models
- Best for complex, unstructured text

### RAG Mode
- Retrieval-Augmented Generation
- Combines knowledge base with LLM
- Best for domain-specific extraction

## 🏗️ Project Structure

```
AgenticWorkflow/
├── mms_extractor/           # Main package
│   ├── api.py              # Flask API server
│   ├── core/               # Core extraction logic
│   ├── models/             # Model definitions
│   ├── config/             # Configuration files
│   └── utils/              # Utility functions
├── app/                    # Application examples
├── modern_mms_extractor/   # Modern implementation
└── docs/                   # Documentation
```

## 🔍 API Endpoints

### Core Endpoints

- `GET /health` - Health check
- `GET /models` - List available models
- `POST /extract` - Extract information from message
- `GET /status` - Get API status and loaded models
- `GET /model-info` - Get model information

### Example Request

```json
{
  "message": "갤럭시 S24 Ultra 512GB 구매하고 싶습니다",
  "model": "gemma_3",
  "extraction_mode": "nlp",
  "mock_data": false
}
```

### Example Response

```json
{
  "success": true,
  "result": {
    "product_name": "갤럭시 S24 Ultra",
    "storage": "512GB",
    "intent": "purchase_inquiry",
    "confidence": 0.95
  },
  "metadata": {
    "model_used": "gemma_3",
    "processing_time_seconds": 0.245,
    "extraction_mode": "nlp"
  }
}
```

## 🧪 Testing

### Test Loading Modes
```bash
cd mms_extractor
python test_loading_modes.py
```

### Run API Tests
```bash
cd app
python example-client.py
```

## 🔧 Development

### Running in Development Mode

```bash
# Start API with auto-reload
cd mms_extractor
python api.py --debug --port 8080

# Run with specific model loading mode
python api.py --model-loading-mode local --port 8080
```

### Adding New Models

1. Update `models/language_models.py`
2. Add configuration in `config/settings.py`
3. Test with different extraction modes

## 📚 Documentation

- [API Usage Guide](mms_extractor/API_USAGE.md)
- [Model Configuration](mms_extractor/config/README.md)
- [Development Guide](docs/DEVELOPMENT.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Korean SBERT model by jhgan
- Kiwi Korean NLP library
- Flask and Flask-CORS for API framework
- All contributors and testers

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Check the documentation
- Review example usage in `/app` directory

---

**Made with ❤️ for Korean NLP and MMS processing** 