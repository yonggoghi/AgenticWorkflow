#!/usr/bin/env python3
"""
Utility script to download and save SentenceTransformer models locally.
This script downloads the ko-sbert-nli model and saves it to the local models directory.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path to import our modules
current_dir = Path(__file__).parent.absolute()
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(current_dir))

# Try different import methods to handle both standalone and package execution
try:
    # Try relative imports first (when run as part of package)
    from .models.language_models import save_sentence_transformer
    from .config.settings import MODEL_CONFIG
except ImportError:
    try:
        # Try package imports (when run from parent directory)
        from mms_extractor.models.language_models import save_sentence_transformer
        from mms_extractor.config.settings import MODEL_CONFIG
    except ImportError:
        # Try direct imports (when run as standalone script)
        from models.language_models import save_sentence_transformer
        from config.settings import MODEL_CONFIG


def main():
    """Download and save the sentence transformer model locally."""
    print("📥 Downloading SentenceTransformer model locally...")
    print()
    
    # Model to download
    model_name = MODEL_CONFIG.embedding_model  # "jhgan/ko-sbert-nli"
    save_path = MODEL_CONFIG.local_embedding_model_path  # "./models/ko-sbert-nli"
    
    print(f"🔧 Model: {model_name}")
    print(f"📂 Save path: {save_path}")
    print(f"🔄 Current loading mode: {MODEL_CONFIG.model_loading_mode}")
    print()
    
    try:
        # Create the models directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Download and save the model
        model = save_sentence_transformer(model_name, save_path)
        
        print(f"✅ Successfully downloaded and saved model to {save_path}")
        print(f"📁 Model size: {get_dir_size(save_path):.2f} MB")
        print()
        print("🎯 Next steps:")
        print("  • The model will now be loaded locally by default (faster startup)")
        print("  • To force offline mode: set MODEL_LOADING_MODE=local")
        print("  • To always download fresh: set MODEL_LOADING_MODE=remote")
        print("  • For auto mode (default): set MODEL_LOADING_MODE=auto")
        print()
        print("🚀 Environment variable examples:")
        print("  export MODEL_LOADING_MODE=local    # Offline only")
        print("  export MODEL_LOADING_MODE=remote   # Always download")
        print("  export MODEL_LOADING_MODE=auto     # Auto detect (default)")
        
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        sys.exit(1)


def get_dir_size(path):
    """Get the size of a directory in MB."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    return total_size / (1024 * 1024)  # Convert to MB


if __name__ == "__main__":
    main() 