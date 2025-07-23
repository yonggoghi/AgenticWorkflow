#!/usr/bin/env python3
"""
Test script to demonstrate different model loading modes.
This script shows how the three loading modes work in practice.
"""

import sys
import os
import time
from pathlib import Path

# Add current directory to Python path for imports
current_dir = Path(__file__).parent.absolute()
parent_dir = current_dir.parent
# sys.path.insert(0, str(parent_dir))  # Avoid importing from agentic files
sys.path.insert(0, str(current_dir))

# Try different import methods to handle both standalone and package execution
try:
    # Try relative imports first (when run as part of package)
    from .models.language_models import EmbeddingManager
    from .config.settings import MODEL_CONFIG
except ImportError:
    try:
        # Try package imports (when run from parent directory)
        from mms_extractor.models.language_models import EmbeddingManager
        from mms_extractor.config.settings import MODEL_CONFIG
    except ImportError:
        # Try direct imports (when run as standalone script)
        from models.language_models import EmbeddingManager
        from config.settings import MODEL_CONFIG

def test_loading_mode(mode: str, description: str):
    """Test a specific loading mode."""
    print(f"\n🔧 Testing {mode.upper()} mode")
    print(f"📝 {description}")
    print("-" * 50)
    
    try:
        start_time = time.time()
        
        # Create EmbeddingManager with specific loading mode
        embedding_manager = EmbeddingManager(model_name=MODEL_CONFIG.embedding_model)
        
        load_time = time.time() - start_time
        
        # Get model info
        model_info = embedding_manager.get_model_info()
        
        print(f"✅ SUCCESS - Loaded in {load_time:.2f} seconds")
        print(f"📍 Device: {model_info['device']}")
        print(f"📂 Local model exists: {model_info['local_model_exists']}")
        print(f"🎯 Model loaded: {model_info['model_loaded']}")
        
        # Test encoding
        test_texts = ["테스트 문장입니다", "Hello world"]
        embeddings = embedding_manager.encode(test_texts, show_progress_bar=False)
        print(f"🧮 Encoding test: {embeddings.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def main():
    """Test all loading modes."""
    print("🚀 Model Loading Mode Test")
    print("=" * 60)
    
    # Show current configuration
    print(f"🔧 Default model: {MODEL_CONFIG.embedding_model}")
    print(f"📂 Local path: {MODEL_CONFIG.local_embedding_model_path}")
    print(f"📍 Local exists: {os.path.exists(MODEL_CONFIG.local_embedding_model_path)}")
    print(f"🔄 Current mode: {MODEL_CONFIG.model_loading_mode}")
    
    # Test all modes
    modes = [
        ("auto", "Try local first, fallback to remote (recommended)"),
        ("local", "Only use local models (offline mode)"),
        ("remote", "Always download from internet")
    ]
    
    results = {}
    
    for mode, description in modes:
        results[mode] = test_loading_mode(mode, description)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    
    for mode, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{mode.upper():<8} {status}")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    
    if os.path.exists(MODEL_CONFIG.local_embedding_model_path):
        print("  • ✅ Local model found - 'auto' and 'local' modes available")
        print("  • 🚀 Use 'auto' for best performance with fallback")
        print("  • 🔒 Use 'local' for guaranteed offline operation")
    else:
        print("  • ⚠️  No local model found")
        print("  • 📥 Run 'python download_model.py' to enable local loading")
        print("  • 🌐 Currently only 'remote' mode will work")
    
    print("  • 🧪 Use 'remote' for testing or ensuring latest model version")
    
    # Usage examples
    print("\n🛠️  USAGE EXAMPLES:")
    print("  # Command line:")
    print("  python run.py --model-loading-mode local --text 'test'")
    print("  python api.py --model-loading-mode auto --port 8080")
    print()
    print("  # Environment variable:")
    print("  export MODEL_LOADING_MODE=local")
    print()
    print("  # Programmatic:")
    print("  manager = EmbeddingManager(model_name='jhgan/ko-sbert-nli')")

if __name__ == "__main__":
    main() 