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

# Global extractor instances
extractors = {}

# Global configuration for CLI data source
CLI_DATA_SOURCE = 'local'

def get_extractor(data_source='local', offer_info_data_src='local'):
    """Get or create extractor instance."""
    key = f"{data_source}_{offer_info_data_src}"
    
    if key not in extractors:
        logger.info(f"Creating new extractor: {key}")
        
        # Initialize extractor with specified data source
        extractor = MMSExtractor(
            model_path='./models/ko-sbert-nli',
            data_dir='./data',
            offer_info_data_src=offer_info_data_src
        )
        
        extractors[key] = extractor
        logger.info(f"Extractor ready: {key}")
    
    return extractors[key]

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
    """List available models and data sources."""
    return jsonify({
        "model": "skt/gemma3-12b-it",
        "available_data_sources": ["local", "db"],
        "default_data_source": "local",
        "features": [
            "Korean morphological analysis (Kiwi)",
            "Embedding-based similarity search",
            "Entity extraction and matching",
            "Program classification"
        ]
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
        data_source = data.get('data_source', CLI_DATA_SOURCE)
        offer_info_data_src = data.get('offer_info_data_src', CLI_DATA_SOURCE)
        
        # Validate data source
        valid_sources = ['local', 'db']
        if offer_info_data_src not in valid_sources:
            return jsonify({"error": f"Invalid offer_info_data_src. Available: {valid_sources}"}), 400
        
        # Get extractor and process
        start_time = time.time()
        extractor = get_extractor(data_source, offer_info_data_src)
        
        logger.info(f"Processing message with data_source: {offer_info_data_src}")
        result = extractor.process_message(message)
        processing_time = time.time() - start_time
        
        # Return result
        response = {
            "success": True,
            "result": result,
            "metadata": {
                "model": "skt/gemma3-12b-it",
                "data_source": offer_info_data_src,
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
        
        # Get extractor
        extractor = get_extractor(CLI_DATA_SOURCE, offer_info_data_src)
        
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
                "model": "skt/gemma3-12b-it",
                "data_source": offer_info_data_src,
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
    """Get API status and loaded extractors."""
    return jsonify({
        "status": "running",
        "loaded_extractors": list(extractors.keys()),
        "total_extractors": len(extractors),
        "model": "skt/gemma3-12b-it",
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
    parser.add_argument('--data-source', choices=['local', 'db'], default='local',
                       help='Data source to use (local CSV or database)')
    
    args = parser.parse_args()
    
    # Set global CLI data source
    CLI_DATA_SOURCE = args.data_source
    logger.info(f"CLI data source set to: {CLI_DATA_SOURCE}")
    
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
            logger.info(f"Initializing extractor with data source: {args.data_source}")
            extractor = get_extractor(args.data_source, args.data_source)
            
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
