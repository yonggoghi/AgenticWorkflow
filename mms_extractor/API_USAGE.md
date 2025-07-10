# MMS Extractor API Usage Guide

This document provides comprehensive usage instructions for the MMS Extractor REST API service.

## Quick Start

### 1. Start the API Server
```bash
# Basic startup (recommended) - preloads gemma_3 with nlp mode
python api.py --port 8080

# With debug mode and additional models preloaded
python api.py --debug --preload claude_37 --port 8080

# Preload all extraction modes for faster switching
python api.py --preload-all-modes --port 8080

# Disable preloading (load on-demand for development)
python api.py --no-preload --port 8080

# With mock data for testing
python api.py --mock-data --port 8080

# With specific model loading mode
python api.py --model-loading-mode local --port 8080    # Offline only
python api.py --model-loading-mode remote --port 8080   # Always download
python api.py --model-loading-mode auto --port 8080     # Auto detect (default)

# Custom host and port
python api.py --host 0.0.0.0 --port 8080
```

> **Note:** Use port 8080 to avoid conflicts with macOS AirPlay Receiver (port 5000)

### 2. Test the API
```bash
# Health check
curl http://localhost:8080/health

# Quick extraction test
curl -X POST http://localhost:8080/extract \
  -H "Content-Type: application/json" \
  -d '{"message": "테스트 메시지입니다"}'
```

## API Overview

The MMS Extractor API provides endpoints for extracting structured information from Korean marketing messages (MMS). The service supports multiple LLM models and extraction modes for flexible information extraction.

### Base URL
- Local development: `http://localhost:8080`
- Local with IP: `http://127.0.0.1:8080`

### Available Extraction Modes
- **nlp**: Uses NLP-extracted entities directly in schema (default)
- **llm**: Uses LLM without additional context
- **rag**: Uses RAG context with candidate item names

## API Endpoints

### 1. Health Check
**GET** `/health`

Check if the API service is running.

```bash
curl http://localhost:8080/health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "MMS Extractor API",
  "version": "1.0.0",
  "timestamp": 1702123456.789
}
```

### 2. List Available Models
**GET** `/models`

Get list of available LLM models.

```bash
curl http://localhost:8080/models
```

**Response:**
```json
{
  "available_models": ["gemma_3", "claude_37", "gpt_4", "claude_sonnet_4"],
  "default_model": "gemma_3"
}
```

### 3. Extract Information
**POST** `/extract`

Extract structured information from MMS message.

#### Request Body
```json
{
  "message": "Your MMS message text here",
  "model": "gemma_3",
  "extraction_mode": "nlp",
  "mock_data": false
}
```

#### Parameters
- **message** (required, string): The MMS message text to process
- **model** (optional, string): LLM model to use (default: "gemma_3")
  - Available: "gemma_3", "claude_37", "gpt_4", "claude_sonnet_4"
- **extraction_mode** (optional, string): Extraction mode (default: "nlp")
  - Available: "nlp", "llm", "rag"
- **mock_data** (optional, boolean): Use mock data for testing (default: false)

**Response:**
```json
{
  "success": true,
  "result": {
    "title": "Extracted title",
    "purpose": ["혜택 안내", "쿠폰 제공 안내"],
    "product": [...],
    "channel": [...],
    "pgm": [...]
  },
  "metadata": {
    "model_used": "gemma_3",
    "mock_data": false,
    "processing_time_seconds": 2.456,
    "timestamp": 1734444123.456
  }
}
```

### 4. Get Status
**GET** `/status`

Get current API status, loaded extractors, and model configuration.

```bash
curl http://localhost:8080/status
```

**Response:**
```json
{
  "status": "running",
  "loaded_extractors": ["gemma_3_False_nlp"],
  "available_models": ["gemma_3", "claude_37", "gpt_4", "claude_sonnet_4"],
  "available_extraction_modes": ["nlp", "llm", "rag"],
  "model_loading_config": {
    "current_mode": "auto",
    "description": "Automatically use local model if available, otherwise download from internet",
    "local_model_path": "./models/ko-sbert-nli",
    "available_modes": ["auto", "local", "remote"]
  },
  "endpoints": [...]
}
```

### 5. Get Model Information
**GET** `/model-info`

Get detailed information about embedding models and their configuration.

```bash
curl http://localhost:8080/model-info
```

**Response:**
```json
{
  "embedding_model_config": {
    "model_name": "jhgan/ko-sbert-nli",
    "local_model_path": "./models/ko-sbert-nli",
    "loading_mode": "auto",
    "loading_mode_description": "Automatically use local model if available, otherwise download from internet"
  },
  "model_status": {
    "model_name": "jhgan/ko-sbert-nli",
    "device": "mps",
    "loading_mode": "auto",
    "local_model_path": "./models/ko-sbert-nli",
    "local_model_exists": true,
    "model_loaded": true
  },
  "llm_models": {
    "claude_model": "skt/claude-3-7-sonnet-20250219",
    "gemma_model": "skt/gemma3-12b-it",
    "gpt_model": "gpt-4.1",
    "claude_sonnet_model": "claude-sonnet-4-20250514"
  }
}
```

## Example Usage

### cURL
```bash
# Health check
curl http://localhost:8080/health

# Extract information
curl -X POST http://localhost:8080/extract \
  -H "Content-Type: application/json" \
  -d '{
    "message": "광고 제목:[SK텔레콤] 2월 0 day 혜택 안내",
    "model": "gemma_3"
  }'
```

### Python
```python
import requests

# Health check
response = requests.get('http://localhost:8080/health')
print(response.json())

# Extract information
data = {
    "message": "Korean MMS message here",
    "model": "gemma_3"
}
response = requests.post('http://localhost:8080/extract', json=data)
result = response.json()
print(result['result'])
```

### JavaScript
```javascript
// Health check
fetch('http://localhost:8080/health')
  .then(response => response.json())
  .then(data => console.log(data));

// Extract information
fetch('http://localhost:8080/extract', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    message: 'Korean MMS message here',
    model: 'gemma_3'
  })
})
.then(response => response.json())
.then(data => console.log(data.result));
```

## Available Models

- **gemma_3**: Gemma 3 12B (Korean-optimized, default)
- **claude_37**: Claude 3.7 Sonnet (Advanced reasoning)
- **gpt_4**: GPT-4.1 (General purpose)
- **claude_sonnet_4**: Claude Sonnet 4 (Latest Claude)

## Configuration

### Startup Options

The API supports the following startup options:

- `--host`: Host to bind to (default: 0.0.0.0)
- `--port`: Port to bind to (default: 5000, **recommended: 8080**)
- `--debug`: Enable debug mode
- `--preload`: Preload additional specific models (beyond defaults)
- `--preload-all-modes`: Preload all extraction modes for default model
- `--no-preload`: Disable default model preloading (load on-demand)
- `--mock-data`: Use mock data for testing
- `--model-loading-mode`: Choose model loading mode (`auto`, `local`, `remote`)

### Model Preloading Behavior

**🚀 By Default (NEW)**: The API now automatically preloads `gemma_3` with `nlp` mode at startup for instant response times.

**⚡ Fast First Request**: No more 25+ second delays on the first API call - models are ready immediately!

**🔧 Preloading Options**:
- **Default**: Preloads `gemma_3(nlp)` - fastest startup with good coverage
- **`--preload-all-modes`**: Preloads `gemma_3` with all modes (`nlp`, `rag`, `llm`)
- **`--preload claude_37`**: Preloads additional models beyond defaults
- **`--no-preload`**: Disables preloading for development (faster startup, slower first requests)

### Local Model Setup

For faster startup and offline operation, you can use local models instead of downloading from the internet:

1. **Download the model locally** (one-time setup):
   ```bash
   python download_model.py
   ```

2. **Set environment variable** (optional):
   ```bash
   export LOCAL_EMBEDDING_MODEL_PATH="./models/ko-sbert-nli"
   ```

3. **The API will automatically use local models** if available, providing:
   - ⚡ Faster startup times
   - 🔒 Offline operation capability
   - 📦 No repeated downloads

**Note**: The system automatically detects local models and falls back to internet downloads if needed.

## Troubleshooting

### Port Issues on macOS
If you get connection errors on port 5000, use port 8080 instead:
```bash
# Instead of localhost:5000
curl http://localhost:8080/health
```

**Root cause**: macOS AirPlay Receiver uses port 5000, causing conflicts.

### IPv4/IPv6 Issues
If `localhost` doesn't work, try the IP address directly:
```bash
# Use IP instead of localhost
curl http://127.0.0.1:8080/health
```

### Common API Errors

#### Invalid Model Error
```json
{"error": "Invalid model. Available: ['gemma_3', 'claude_37', 'gpt_4', 'claude_sonnet_4']"}
```
**Solution**: Use one of the supported model names.

#### Invalid Extraction Mode Error
```json
{"error": "Invalid extraction_mode. Available: ['nlp', 'llm', 'rag']"}
```
**Solution**: Use one of the supported extraction modes.

#### Empty Message Error
```json
{"error": "Message cannot be empty"}
```
**Solution**: Provide a non-empty message in the request body.

### Performance Considerations

#### 🚀 **NEW: Automatic Model Preloading**
- **First Request**: Now instant (was 25+ seconds) thanks to startup preloading
- **Startup Time**: ~30 seconds to preload models (one-time cost)
- **Memory Usage**: Higher initial memory footprint, but consistent performance

#### Extraction Mode Performance
- **NLP Mode**: Fastest, uses pre-extracted entities (~10-15 seconds)
- **RAG Mode**: Medium speed, provides context to LLM (~15-20 seconds)
- **LLM Mode**: Variable speed, depends on LLM complexity (~12-25 seconds)

#### Model Performance (with preloading)
- **gemma_3**: 10-15 seconds (recommended for production)
- **claude_37**: 15-20 seconds  
- **gpt_4**: 20-25 seconds
- **claude_sonnet_4**: 25-30 seconds

#### Startup vs Runtime Trade-offs

| Configuration | Startup Time | First Request | Memory Usage | Best For |
|---------------|--------------|---------------|--------------|----------|
| **Default preload** | ~30s | Instant | High | **Production** |
| **--preload-all-modes** | ~60s | Instant | Higher | High-traffic APIs |
| **--no-preload** | ~2s | 25+ seconds | Low | **Development** |

#### Optimization Tips
1. **Keep default preloading** for production (instant responses)
2. **Use `--no-preload`** during development for faster restarts
3. **Use `--preload-all-modes`** if you frequently switch extraction modes
4. **Use NLP mode** for fastest processing with preloaded entities
5. **Enable GPU acceleration** (MPS on macOS, CUDA on Linux) for faster processing

### Memory and Resource Usage

#### System Requirements
- **Minimum RAM**: 8GB
- **Recommended RAM**: 16GB+ for multiple concurrent requests
- **GPU**: Optional but recommended (MPS on Apple Silicon, CUDA on NVIDIA)
- **Disk Space**: ~5GB for models and data

#### Monitoring Resource Usage
```python
import psutil
import requests

# Monitor memory usage during extraction
def monitor_extraction():
    process = psutil.Process()
    
    # Before extraction
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # Perform extraction
    response = requests.post('http://localhost:8080/extract', 
                           json={'message': 'test message'})
    
    # After extraction
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    
    print(f"Memory usage: {mem_before:.1f}MB -> {mem_after:.1f}MB")
    return response.json()
```

### Development and Testing

#### Using Mock Data
For development without data files:
```bash
python api.py --mock-data --port 8080
```

#### Debug Mode
For detailed logging:
```bash
python api.py --debug --port 8080
```

#### Testing Different Configurations
```python
import requests

def test_configurations():
    """Test different model and extraction mode combinations."""
    base_url = "http://localhost:8080/extract"
    message = "테스트 메시지입니다"
    
    configs = [
        ("gemma_3", "nlp"),
        ("gemma_3", "rag"),
        ("claude_37", "llm"),
    ]
    
    for model, mode in configs:
        print(f"\nTesting {model} with {mode} mode...")
        response = requests.post(base_url, json={
            "message": message,
            "model": model,
            "extraction_mode": mode
        })
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success: {result['metadata']['processing_time_seconds']:.2f}s")
        else:
            print(f"❌ Failed: {response.status_code}")

test_configurations()
```

## Error Handling

The API returns appropriate HTTP status codes:

- `200`: Success
- `400`: Bad Request (missing/invalid parameters)
- `500`: Internal Server Error

Error responses include:
```json
{
  "success": false,
  "error": "Error description",
  "error_type": "ErrorClassName"
}
```

## Usage Examples

### Basic cURL Examples

#### 1. Default Extraction (NLP Mode)
```bash
curl -X POST http://localhost:8080/extract \
  -H "Content-Type: application/json" \
  -d '{
    "message": "광고 제목:[SK텔레콤] 2월 0 day 혜택 안내 광고 내용:(광고)[SKT] 2월 0 day 혜택 안내 만 13~34세 고객이라면 베어유 모든 강의 14일 무료 수강 쿠폰 드립니다!"
  }'
```

#### 2. RAG Mode with Claude Model
```bash
curl -X POST http://localhost:8080/extract \
  -H "Content-Type: application/json" \
  -d '{
    "message": "광고 제목:[SK텔레콤] 2월 0 day 혜택 안내 광고 내용:(광고)[SKT] 2월 0 day 혜택 안내 만 13~34세 고객이라면 베어유 모든 강의 14일 무료 수강 쿠폰 드립니다!",
    "model": "claude_37",
    "extraction_mode": "rag"
  }'
```

#### 3. LLM Mode with GPT-4
```bash
curl -X POST http://localhost:8080/extract \
  -H "Content-Type: application/json" \
  -d '{
    "message": "광고 제목:[SK텔레콤] 2월 0 day 혜택 안내 광고 내용:(광고)[SKT] 2월 0 day 혜택 안내 만 13~34세 고객이라면 베어유 모든 강의 14일 무료 수강 쿠폰 드립니다!",
    "model": "gpt_4",
    "extraction_mode": "llm"
  }'
```

### Python Examples

#### Using requests library
```python
import requests
import json

url = "http://localhost:8080/extract"
headers = {"Content-Type": "application/json"}

# Example with NLP mode
data = {
    "message": "광고 제목:[SK텔레콤] 2월 0 day 혜택 안내 광고 내용:(광고)[SKT] 2월 0 day 혜택 안내 만 13~34세 고객이라면 베어유 모든 강의 14일 무료 수강 쿠폰 드립니다!",
    "model": "gemma_3",
    "extraction_mode": "nlp"
}

response = requests.post(url, headers=headers, json=data)
result = response.json()

print(json.dumps(result, ensure_ascii=False, indent=2))
```

#### Comparing Different Extraction Modes
```python
import requests
import json

def extract_with_mode(message, extraction_mode, model="gemma_3"):
    """Extract information using specified mode."""
    url = "http://localhost:8080/extract"
    headers = {"Content-Type": "application/json"}
    
    data = {
        "message": message,
        "model": model,
        "extraction_mode": extraction_mode
    }
    
    response = requests.post(url, headers=headers, json=data)
    return response.json()

# Test message
message = "광고 제목:[SK텔레콤] 2월 0 day 혜택 안내 광고 내용:(광고)[SKT] 2월 0 day 혜택 안내 만 13~34세 고객이라면 베어유 모든 강의 14일 무료 수강 쿠폰 드립니다!"

# Compare different modes
for mode in ["nlp", "llm", "rag"]:
    print(f"\n=== {mode.upper()} Mode Results ===")
    result = extract_with_mode(message, mode)
    print(json.dumps(result["result"], ensure_ascii=False, indent=2))
```

### JavaScript/Node.js Examples

#### Using fetch API
```javascript
const extractMessage = async (message, extractionMode = 'nlp', model = 'gemma_3') => {
  const url = 'http://localhost:8080/extract';
  
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      message: message,
      model: model,
      extraction_mode: extractionMode
    })
  });
  
  return await response.json();
};

// Usage example
const message = "광고 제목:[SK텔레콤] 2월 0 day 혜택 안내 광고 내용:(광고)[SKT] 2월 0 day 혜택 안내 만 13~34세 고객이라면 베어유 모든 강의 14일 무료 수강 쿠폰 드립니다!";

// Extract with NLP mode
extractMessage(message, 'nlp')
  .then(result => console.log('NLP Result:', JSON.stringify(result, null, 2)))
  .catch(error => console.error('Error:', error));

// Extract with RAG mode
extractMessage(message, 'rag', 'claude_37')
  .then(result => console.log('RAG Result:', JSON.stringify(result, null, 2)))
  .catch(error => console.error('Error:', error));
```

## Extraction Mode Details

### NLP Mode (Default)
- **Best for**: High precision extraction with pre-identified entities
- **Behavior**: Uses Kiwi NLP to extract entities first, then fills LLM schema with these entities
- **Advantages**: Better accuracy for known products/entities
- **Use case**: Production environments with established vocabularies

### LLM Mode  
- **Best for**: General-purpose extraction without constraints
- **Behavior**: Uses pure LLM extraction without additional context
- **Advantages**: More flexible, can identify new entities
- **Use case**: Exploring new content types or when vocabulary is incomplete

### RAG Mode
- **Best for**: Balancing precision and recall with context
- **Behavior**: Provides candidate items as context to the LLM
- **Advantages**: Combines NLP entity extraction with LLM flexibility
- **Use case**: When you want to guide LLM with known entities but allow flexibility

# NLP Mode (default) - High precision with pre-extracted entities
curl -X POST http://localhost:8080/extract \
  -H "Content-Type: application/json" \
  -d '{"message": "테스트 메시지", "extraction_mode": "nlp"}'

# RAG Mode - Balanced approach with context
curl -X POST http://localhost:8080/extract \
  -H "Content-Type: application/json" \
  -d '{"message": "테스트 메시지", "extraction_mode": "rag"}'

# LLM Mode - Pure LLM flexibility
curl -X POST http://localhost:8080/extract \
  -H "Content-Type: application/json" \
  -d '{"message": "테스트 메시지", "extraction_mode": "llm"}'