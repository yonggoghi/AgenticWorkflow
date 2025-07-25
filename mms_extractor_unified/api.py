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
from config import settings

# Add current directory to Python path for imports
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

# Import the new MMSExtractor
try:
    from mms_extractor import MMSExtractor
    from config.settings import API_CONFIG, MODEL_CONFIG, PROCESSING_CONFIG
except ImportError as e:
    print(f"Error importing MMSExtractor: {e}")
    print("Make sure mms_extractor.py is in the same directory")
    sys.exit(1)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global extractor instance - loaded once at startup
global_extractor = None

# Global configuration for CLI data source
CLI_DATA_SOURCE = 'local'

def initialize_global_extractor(offer_info_data_src='local'):
    """Initialize the global extractor instance once at startup."""
    global global_extractor
    
    if global_extractor is None:
        logger.info(f"Initializing global extractor with data source: {offer_info_data_src}")
        
        # Initialize extractor with data loading
        global_extractor = MMSExtractor(
            model_path='./models/ko-sbert-nli',
            data_dir='./data',
            offer_info_data_src=offer_info_data_src,
            llm_model='gemma',  # Default model, will be overridden per request
            product_info_extraction_mode='nlp',  # Default mode, will be overridden per request
            entity_extraction_mode='logic'  # Default mode, will be overridden per request
        )
        
        logger.info("Global extractor initialized successfully")
    
    return global_extractor

def get_configured_extractor(llm_model='gemma', product_info_extraction_mode='nlp', entity_matching_mode='logic'):
    """Get the global extractor with runtime configuration."""
    if global_extractor is None:
        raise RuntimeError("Global extractor not initialized. Call initialize_global_extractor() first.")
    
    # Update runtime configuration without reloading data
    global_extractor.llm_model_name = llm_model
    global_extractor.product_info_extraction_mode = product_info_extraction_mode
    global_extractor.entity_extraction_mode = entity_matching_mode
    
    # Reinitialize LLM if model changed
    global_extractor._initialize_llm()
    
    return global_extractor

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "MMS Extractor API",
        "version": "2.0.0",
        "model": "skt/gemma3-12b-it",
        "timestamp": time.time()
    })

@app.route('/models', methods=['GET'])
def list_models():
    """List available models and configuration options."""
    return jsonify({
        "available_llm_models": ["gemma", "gpt", "claude"],
        "default_llm_model": "gemma",
        "available_data_sources": ["local", "db"],
        "default_data_source": "local",
        "available_product_info_extraction_modes": ["nlp", "llm", "rag"],
        "default_product_info_extraction_mode": "nlp",
        "available_entity_matching_modes": ["logic", "llm"],
        "default_entity_matching_mode": "logic",
        "features": [
            "Korean morphological analysis (Kiwi)",
            "Embedding-based similarity search",
            "Entity extraction and matching",
            "Program classification",
            "Multiple LLM support (Gemma, GPT, Claude)"
        ]
    })

@app.route('/extract', methods=['POST'])
def extract_message():
    """Extract information from MMS message."""
    try:
        # Check if global extractor is initialized
        if global_extractor is None:
            return jsonify({"error": "Extractor not initialized. Server startup may have failed."}), 500
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
        data_source = data.get('data_source', CLI_DATA_SOURCE)
        offer_info_data_src = data.get('offer_info_data_src', CLI_DATA_SOURCE)
        llm_model = data.get('llm_model', settings.ModelConfig.llm_model)
        product_info_extraction_mode = data.get('product_info_extraction_mode', settings.ProcessingConfig.product_info_extraction_mode)
        entity_matching_mode = data.get('entity_matching_mode', settings.ProcessingConfig.entity_extraction_mode)
        
        # Validate parameters
        valid_sources = ['local', 'db']
        if offer_info_data_src not in valid_sources:
            return jsonify({"error": f"Invalid offer_info_data_src. Available: {valid_sources}"}), 400
            
        valid_llm_models = ['gemma', 'gpt', 'claude']
        if llm_model not in valid_llm_models:
            return jsonify({"error": f"Invalid llm_model. Available: {valid_llm_models}"}), 400
            
        valid_product_modes = ['nlp', 'llm', 'rag']
        if product_info_extraction_mode not in valid_product_modes:
            return jsonify({"error": f"Invalid product_info_extraction_mode. Available: {valid_product_modes}"}), 400
            
        valid_entity_modes = ['logic', 'llm']
        if entity_matching_mode not in valid_entity_modes:
            return jsonify({"error": f"Invalid entity_matching_mode. Available: {valid_entity_modes}"}), 400
        
        # Get configured extractor and process
        start_time = time.time()
        extractor = get_configured_extractor(llm_model, product_info_extraction_mode, entity_matching_mode)
        
        logger.info(f"Processing message with data_source: {offer_info_data_src}")
        result = extractor.process_message(message)
        processing_time = time.time() - start_time
        
        # Return result
        response = {
            "success": True,
            "result": result,
            "metadata": {
                "llm_model": llm_model,
                "offer_info_data_src": offer_info_data_src,
                "product_info_extraction_mode": product_info_extraction_mode,
                "entity_matching_mode": entity_matching_mode,
                "processing_time_seconds": round(processing_time, 3),
                "timestamp": time.time(),
                "message_length": len(message)
            }
        }
        
        logger.info(f"Extraction completed in {processing_time:.3f}s")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error during extraction: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": time.time()
        }), 500

@app.route('/extract/batch', methods=['POST'])
def extract_batch():
    """Extract information from multiple MMS messages."""
    try:
        # Check if global extractor is initialized
        if global_extractor is None:
            return jsonify({"error": "Extractor not initialized. Server startup may have failed."}), 500
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
        
        data = request.get_json()
        
        if 'messages' not in data:
            return jsonify({"error": "Missing required field: 'messages'"}), 400
        
        messages = data['messages']
        if not isinstance(messages, list):
            return jsonify({"error": "Field 'messages' must be a list"}), 400
        
        if len(messages) > 100:  # Limit batch size
            return jsonify({"error": "Maximum 100 messages per batch"}), 400
        
        # Get optional parameters
        offer_info_data_src = data.get('offer_info_data_src', CLI_DATA_SOURCE)
        llm_model = data.get('llm_model', settings.ModelConfig.llm_model)
        product_info_extraction_mode = data.get('product_info_extraction_mode', settings.ProcessingConfig.product_info_extraction_mode)
        entity_matching_mode = data.get('entity_matching_mode', settings.ProcessingConfig.entity_extraction_mode)
        
        # Validate parameters
        valid_sources = ['local', 'db']
        if offer_info_data_src not in valid_sources:
            return jsonify({"error": f"Invalid offer_info_data_src. Available: {valid_sources}"}), 400
            
        valid_llm_models = ['gemma', 'gpt', 'claude']
        if llm_model not in valid_llm_models:
            return jsonify({"error": f"Invalid llm_model. Available: {valid_llm_models}"}), 400
            
        valid_product_modes = ['nlp', 'llm', 'rag']
        if product_info_extraction_mode not in valid_product_modes:
            return jsonify({"error": f"Invalid product_info_extraction_mode. Available: {valid_product_modes}"}), 400
            
        valid_entity_modes = ['logic', 'llm']
        if entity_matching_mode not in valid_entity_modes:
            return jsonify({"error": f"Invalid entity_matching_mode. Available: {valid_entity_modes}"}), 400
        
        # Get configured extractor
        extractor = get_configured_extractor(llm_model, product_info_extraction_mode, entity_matching_mode)
        
        # Process all messages
        start_time = time.time()
        results = []
        
        for i, message in enumerate(messages):
            if not message or not message.strip():
                results.append({
                    "index": i,
                    "success": False,
                    "error": "Empty message"
                })
                continue
            
            try:
                result = extractor.process_message(message)
                results.append({
                    "index": i,
                    "success": True,
                    "result": result
                })
            except Exception as e:
                logger.error(f"Error processing message {i}: {e}")
                results.append({
                    "index": i,
                    "success": False,
                    "error": str(e)
                })
        
        processing_time = time.time() - start_time
        
        # Count successes and failures
        successful = sum(1 for r in results if r["success"])
        failed = len(results) - successful
        
        response = {
            "success": True,
            "results": results,
            "summary": {
                "total_messages": len(messages),
                "successful": successful,
                "failed": failed
            },
            "metadata": {
                "llm_model": llm_model,
                "offer_info_data_src": offer_info_data_src,
                "product_info_extraction_mode": product_info_extraction_mode,
                "entity_matching_mode": entity_matching_mode,
                "processing_time_seconds": round(processing_time, 3),
                "timestamp": time.time()
            }
        }
        
        logger.info(f"Batch extraction completed: {successful}/{len(messages)} successful in {processing_time:.3f}s")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error during batch extraction: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": time.time()
        }), 500

@app.route('/status', methods=['GET'])
def get_status():
    """Get API status and extractor information."""
    global global_extractor
    
    extractor_status = {
        "initialized": global_extractor is not None,
        "data_source": CLI_DATA_SOURCE if global_extractor else None,
        "current_llm_model": global_extractor.llm_model_name if global_extractor else None,
        "current_product_mode": global_extractor.product_info_extraction_mode if global_extractor else None,
        "current_entity_mode": global_extractor.entity_extraction_mode if global_extractor else None
    }
    
    return jsonify({
        "status": "running",
        "extractor": extractor_status,
        "timestamp": time.time()
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

def main():
    """Main function for CLI usage."""
    global CLI_DATA_SOURCE
    
    parser = argparse.ArgumentParser(description='MMS Extractor API Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--test', action='store_true', help='Run a test extraction')
    parser.add_argument('--message', type=str, help='Message to test with')
    parser.add_argument('--offer-data-source', choices=['local', 'db'], default='local',
                       help='Data source to use (local CSV or database)')
    parser.add_argument('--product-info-extraction-mode', choices=['nlp', 'llm' ,'rag'], default='nlp',
                       help='Product info extraction mode (nlp or llm)')
    parser.add_argument('--entity-matching-mode', choices=['logic', 'llm'], default='llm',
                       help='Entity matching mode (logic or llm)')
    parser.add_argument('--llm-model', choices=['gemma', 'gpt', 'claude'], default='gemma',
                       help='LLM model to use (gemma or gpt or claude)')
    
    args = parser.parse_args()
    
    # Set global CLI data source
    CLI_DATA_SOURCE = args.offer_data_source
    logger.info(f"CLI data source set to: {CLI_DATA_SOURCE}")
    
    # Initialize global extractor with the specified data source
    logger.info("Initializing global extractor...")
    initialize_global_extractor(CLI_DATA_SOURCE)
    
    if args.test:
        # Test mode
        logger.info("Running in test mode...")
        
        # Use provided message or default
        message = args.message or """
        [SK텔레콤] ZEM폰 포켓몬에디션3 안내
        (광고)[SKT] 우리 아이 첫 번째 스마트폰, ZEM 키즈폰__#04 고객님, 안녕하세요!
        우리 아이 스마트폰 고민 중이셨다면, 자녀 스마트폰 관리 앱 ZEM이 설치된 SKT만의 안전한 키즈폰,
        ZEM폰 포켓몬에디션3으로 우리 아이 취향을 저격해 보세요!
        """
        
        try:
            logger.info(f"Configuring extractor with parameters: llm_model={args.llm_model}, product_mode={args.product_info_extraction_mode}, entity_mode={args.entity_matching_mode}")
            extractor = get_configured_extractor(args.llm_model, args.product_info_extraction_mode, args.entity_matching_mode)
            
            if not message.strip():
                logger.info("No text provided, using sample message...")
            
            logger.info("Processing message...")
            result = extractor.process_message(message)
            
            print("\n" + "="*60)
            print("EXTRACTION RESULTS:")
            print("="*60)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            print("="*60)
            
            logger.info("Processing completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            sys.exit(1)
    else:
        # Server mode
        logger.info(f"Parsed arguments: host={args.host}, port={args.port}, debug={args.debug}")
        logger.info("✅ Global extractor initialized and ready for requests")
        logger.info(f"Starting MMS Extractor API server on {args.host}:{args.port}")
        logger.info("Available endpoints:")
        logger.info("  GET  /health - Health check")
        logger.info("  GET  /models - List available models")
        logger.info("  GET  /status - Get server status")
        logger.info("  POST /extract - Extract from single message")
        logger.info("  POST /extract/batch - Extract from multiple messages")
        
        # Ensure Flask uses the correct configuration
        app.config['DEBUG'] = args.debug
        
        try:
            app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False, threaded=True)
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()
