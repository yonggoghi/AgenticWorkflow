#!/usr/bin/env python3
"""
Flask API service for MMS Extractor package.
"""
import sys
import os
import json
import logging
import time
import argparse
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS

# Add current directory to Python path for imports
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir.parent))

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global extractor instances
extractors = {}

def get_extractor(model_name='gemma_3', use_mock_data=False, extraction_mode='nlp'):
    """Get or create extractor instance."""
    key = f"{model_name}_{use_mock_data}_{extraction_mode}"
    
    if key not in extractors:
        logger.info(f"Creating new extractor: {key}")
        
        # Import the MMS extractor and DataManager
        try:
            from mms_extractor.core.mms_extractor import MMSExtractor
            from mms_extractor.core.data_manager import DataManager
            from mms_extractor.config.settings import PROCESSING_CONFIG
        except ImportError:
            from core.mms_extractor import MMSExtractor
            from core.data_manager import DataManager
            from config.settings import PROCESSING_CONFIG
        
        # Set the extraction mode
        PROCESSING_CONFIG.product_info_extraction_mode = extraction_mode
        
        # Initialize DataManager and extractor
        data_manager = DataManager(use_mock_data=use_mock_data)
        extractor = MMSExtractor(data_manager=data_manager, model_name=model_name)
        extractor.load_data()
        
        extractors[key] = extractor
        logger.info(f"Extractor ready: {key}")
    
    return extractors[key]

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "MMS Extractor API",
        "version": "1.0.0",
        "timestamp": time.time()
    })

@app.route('/models', methods=['GET'])
def list_models():
    """List available models."""
    return jsonify({
        "available_models": ["gemma_3", "claude_37", "gpt_4", "claude_sonnet_4"],
        "default_model": "gemma_3"
    })

@app.route('/extract', methods=['POST'])
def extract_message():
    """Extract information from MMS message."""
    try:
        # Get request data
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
        
        data = request.get_json()
        
        # Validate required fields
        if 'message' not in data:
            return jsonify({"error": "Missing required field: 'message'"}), 400
        
        message = data['message']
        if not message or not message.strip():
            return jsonify({"error": "Message cannot be empty"}), 400
        
        # Get optional parameters  
        model_name = data.get('model', 'gemma_3')
        use_mock_data = data.get('mock_data', False)
        extraction_mode = data.get('extraction_mode', 'nlp')
        
        # Validate model
        valid_models = ['gemma_3', 'claude_37', 'gpt_4', 'claude_sonnet_4']
        if model_name not in valid_models:
            return jsonify({"error": f"Invalid model. Available: {valid_models}"}), 400
        
        # Validate extraction mode
        valid_modes = ['nlp', 'llm', 'rag']
        if extraction_mode not in valid_modes:
            return jsonify({"error": f"Invalid extraction_mode. Available: {valid_modes}"}), 400
        
        # Get extractor and process
        start_time = time.time()
        extractor = get_extractor(model_name, use_mock_data, extraction_mode)
        
        logger.info(f"Processing message with model: {model_name}")
        result = extractor.extract(message)
        processing_time = time.time() - start_time
        
        # Return result
        response = {
            "success": True,
            "result": result,
            "metadata": {
                "model_used": model_name,
                "mock_data": use_mock_data,
                "extraction_mode": extraction_mode,
                "processing_time_seconds": round(processing_time, 3),
                "timestamp": time.time()
            }
        }
        
        logger.info(f"Extraction completed in {processing_time:.3f}s")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error during extraction: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }), 500

@app.route('/status', methods=['GET'])
def get_status():
    """Get API status."""
    try:
        from mms_extractor.config.settings import MODEL_CONFIG
    except ImportError:
        from config.settings import MODEL_CONFIG
    
    # Get extractor details
    extractor_details = {}
    for key, extractor in extractors.items():
        try:
            model_info = None
            if hasattr(extractor, 'embedding_manager'):
                model_info = extractor.embedding_manager.get_model_info()
            
            extractor_details[key] = {
                "model_loaded": True,
                "embedding_model_info": model_info
            }
        except Exception as e:
            extractor_details[key] = {
                "model_loaded": False,
                "error": str(e)
            }
    
    return jsonify({
        "status": "running", 
        "preloaded_extractors": {
            "count": len(extractors),
            "details": extractor_details
        },
        "available_models": ["gemma_3", "claude_37", "gpt_4", "claude_sonnet_4"],
        "available_extraction_modes": ["nlp", "llm", "rag"],
        "model_loading_config": {
            "current_mode": MODEL_CONFIG.model_loading_mode,
            "description": MODEL_CONFIG.get_loading_mode_description(),
            "local_model_path": MODEL_CONFIG.local_embedding_model_path,
            "available_modes": ["auto", "local", "remote"]
        },
        "extraction_mode_descriptions": {
            "nlp": "Uses NLP-extracted entities directly in schema",
            "llm": "Uses LLM without additional context",
            "rag": "Uses RAG context with candidate item names"
        },
        "startup_options": {
            "preload_by_default": "Models are preloaded at startup for fast response times",
            "disable_preload": "Use --no-preload to load models on-demand",
            "preload_all_modes": "Use --preload-all-modes to preload all extraction modes",
            "preload_additional": "Use --preload to preload additional models"
        },
        "endpoints": [
            "GET /health",
            "GET /models", 
            "POST /extract",
            "GET /status",
            "GET /model-info"
        ]
    })

@app.route('/model-info', methods=['GET'])
def get_model_info():
    """Get detailed model information."""
    try:
        from mms_extractor.config.settings import MODEL_CONFIG
    except ImportError:
        from config.settings import MODEL_CONFIG
    
    # Get model info from an extractor if available
    model_info = None
    if extractors:
        # Get info from the first available extractor
        first_extractor = next(iter(extractors.values()))
        if hasattr(first_extractor, 'embedding_manager'):
            model_info = first_extractor.embedding_manager.get_model_info()
    
    return jsonify({
        "embedding_model_config": {
            "model_name": MODEL_CONFIG.embedding_model,
            "local_model_path": MODEL_CONFIG.local_embedding_model_path,
            "loading_mode": MODEL_CONFIG.model_loading_mode,
            "loading_mode_description": MODEL_CONFIG.get_loading_mode_description()
        },
        "model_status": model_info if model_info else "No model loaded yet",
        "llm_models": {
            "claude_model": MODEL_CONFIG.claude_model,
            "gemma_model": MODEL_CONFIG.gemma_model,
            "gpt_model": MODEL_CONFIG.gpt_model,
            "claude_sonnet_model": MODEL_CONFIG.claude_sonnet_model
        }
    })

def preload_default_extractors(mock_data=False, models_to_preload=None):
    """Preload default extractors to avoid loading during API requests."""
    if models_to_preload is None:
        # Default models and extraction modes to preload
        models_to_preload = [
            ("gemma_3", "nlp"),
            ("gemma_3", "rag"),
        ]
    
    logger.info("🚀 Preloading default extractors...")
    
    for model_name, extraction_mode in models_to_preload:
        try:
            logger.info(f"📥 Loading {model_name} with {extraction_mode} mode...")
            start_time = time.time()
            
            extractor = get_extractor(model_name, mock_data, extraction_mode)
            load_time = time.time() - start_time
            
            logger.info(f"✅ Successfully preloaded {model_name}({extraction_mode}) in {load_time:.1f}s")
            
        except Exception as e:
            logger.error(f"❌ Failed to preload {model_name}({extraction_mode}): {e}")
            # Don't fail the entire startup for one model
            continue
    
    logger.info(f"🎯 Preloaded extractors: {list(extractors.keys())}")

def main():
    """Main function for Flask API server."""
    parser = argparse.ArgumentParser(description="MMS Extractor API Service")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--preload", nargs="+", 
                        choices=["gemma_3", "claude_37", "gpt_4", "claude_sonnet_4"],
                        help="Preload specific models (in addition to defaults)")
    parser.add_argument("--mock-data", action="store_true", 
                        help="Use mock data for testing")
    parser.add_argument("--model-loading-mode", 
                        choices=["auto", "local", "remote"], 
                        default="auto",
                        help="Model loading mode: auto (default), local (offline), remote (always download)")
    parser.add_argument("--no-preload", action="store_true",
                        help="Disable default model preloading (load on-demand)")
    parser.add_argument("--preload-all-modes", action="store_true",
                        help="Preload all extraction modes for default model")
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting MMS Extractor API Service")
    logger.info(f"🌐 Host: {args.host}, Port: {args.port}, Debug: {args.debug}")
    logger.info(f"🔧 Model loading mode: {args.model_loading_mode}")
    
    # Set model loading mode globally
    try:
        from mms_extractor.config.settings import MODEL_CONFIG
    except ImportError:
        from config.settings import MODEL_CONFIG
    
    MODEL_CONFIG.model_loading_mode = args.model_loading_mode
    logger.info(f"📋 Model loading mode set to: {MODEL_CONFIG.model_loading_mode}")
    logger.info(f"📝 Description: {MODEL_CONFIG.get_loading_mode_description()}")
    
    # Determine what to preload
    if not args.no_preload:
        # Prepare default models to preload
        default_models = [("gemma_3", "nlp")]
        
        if args.preload_all_modes:
            # Preload all extraction modes for default model
            default_models = [
                ("gemma_3", "nlp"),
                ("gemma_3", "rag"),
                ("gemma_3", "llm")
            ]
        
        # Preload default extractors
        preload_default_extractors(args.mock_data, default_models)
        
        # Preload additional models if specified
        if args.preload:
            logger.info(f"📥 Preloading additional models: {args.preload}")
            additional_models = [(model, "nlp") for model in args.preload]
            
            for model_name, extraction_mode in additional_models:
                try:
                    start_time = time.time()
                    get_extractor(model_name, args.mock_data, extraction_mode)
                    load_time = time.time() - start_time
                    logger.info(f"✅ Successfully preloaded additional {model_name} in {load_time:.1f}s")
                except Exception as e:
                    logger.error(f"❌ Failed to preload additional {model_name}: {e}")
    else:
        logger.info("⚠️  Model preloading disabled - models will load on first request")
    
    logger.info("🎯 API server ready to accept requests!")
    
    # Start Flask app
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()

# DEPRECATED ORIGINAL MAIN FUNCTION - REMOVE BELOW
def _old_main():
    """DEPRECATED - Original standalone execution."""
    parser = argparse.ArgumentParser(description="MMS Extractor API Service")
    parser.add_argument("--text", "-t", help="Text message to process")
    parser.add_argument("--mock-data", action="store_true", help="Use mock data for testing")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--model", "-m", default="gemma_3", 
                        choices=["gemma_3", "claude_37", "gpt_4", "claude_sonnet_4"],
                        help="LLM model to use (default: gemma_3)")
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    try:
        # Import the MMS extractor and DataManager
        try:
            from mms_extractor.core.mms_extractor import MMSExtractor
            from mms_extractor.core.data_manager import DataManager
        except ImportError:
            from core.mms_extractor import MMSExtractor
            from core.data_manager import DataManager
        
        logger.info("Initializing MMS Extractor...")
        logger.info(f"Selected LLM model: {args.model}")
        
        # Initialize DataManager with mock data setting
        data_manager = DataManager(use_mock_data=args.mock_data)
        
        # Initialize MMS Extractor with DataManager and selected model
        extractor = MMSExtractor(data_manager=data_manager, model_name=args.model)
        
        # Load data
        logger.info("Loading data...")
        extractor.load_data()
        logger.info("MMS Extractor ready!")
        
        # Process message
        if args.text:
            message = args.text
        else:
            # Default test message
            message = """광고 제목:[SK텔레콤] 2월 0 day 혜택 안내
광고 내용:(광고)[SKT] 2월 0 day 혜택 안내__[2월 10일(토) 혜택]_만 13~34세 고객이라면_베어유 모든 강의 14일 무료 수강 쿠폰 드립니다!_(선착순 3만 명 증정)_▶ 자세히 보기: http://t-mms.kr/t.do?m=#61&s=24589&a=&u=https://bit.ly/3SfBjjc__■ 에이닷 X T 멤버십 시크릿코드 이벤트_에이닷 T 멤버십 쿠폰함에 '에이닷이빵쏜닷'을 입력해보세요!_뚜레쥬르 데일리우유식빵 무료 쿠폰을 드립니다._▶ 시크릿코드 입력하러 가기: https://bit.ly/3HCUhLM__■ 문의: SKT 고객센터(1558, 무료)_무료 수신거부 1504
"""
            logger.info("No text provided, using sample message...")
        
        logger.info("Processing message...")
        result = extractor.extract(message)
        
        print("\n" + "="*60)
        print("EXTRACTION RESULTS:")
        print("="*60)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("="*60)
        
        logger.info("Processing completed successfully!")
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Please install dependencies: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        if not args.mock_data:
            logger.info("💡 Try using --mock-data flag for testing without data files")
        sys.exit(1)

if __name__ == "__main__":
    main()
