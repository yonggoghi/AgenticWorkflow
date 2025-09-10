#!/usr/bin/env python3
"""
Test Flask argument passing
"""
from flask import Flask
import argparse
import sys

# Create a simple Flask app
app = Flask(__name__)

@app.route('/test')
def test():
    return {"message": "Test endpoint", "config": {"host": app.config.get('HOST'), "port": app.config.get('PORT'), "debug": app.config.get('DEBUG')}}

def main():
    parser = argparse.ArgumentParser(description='Test Flask Args')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    print(f"Parsed arguments: host={args.host}, port={args.port}, debug={args.debug}")
    
    # Set Flask config
    app.config['HOST'] = args.host
    app.config['PORT'] = args.port
    app.config['DEBUG'] = args.debug
    
    print(f"Flask config: {app.config}")
    print(f"Starting Flask server...")
    
    # This should use the parsed arguments
    try:
        app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False, threaded=True)
    except Exception as e:
        print(f"Error starting server: {e}")

if __name__ == "__main__":
    main() 