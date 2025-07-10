# Large Files Management Guide

## Overview

This project contains several large files (>100MB) that cannot be stored directly in Git due to GitHub's file size limits. This guide explains how to handle these files properly.

## Large Files Identified

The following files are over 100MB and are excluded from Git:

### Model Files
- `./models/ko-sbert-nli/model.safetensors` (~500MB)
- `./mms_extractor/models/ko-sbert-nli/model.safetensors` (~500MB)
- `./model/Llama3/model-00001-of-00004.safetensors` (~5GB)
- `./model/Llama3/model-00002-of-00004.safetensors` (~5GB)
- `./model/Llama3/model-00003-of-00004.safetensors` (~5GB)
- `./model/Llama3/model-00004-of-00004.safetensors` (~5GB)

### Data Files
- `./data/org_nm_embeddings_250605.npz` (~200MB)
- `./data/org_all_embeddings_250605.npz` (~300MB)
- `./data/item_embeddings_250527.npz` (~150MB)

### Package Files
- `./app/linux_packages_250617.zip` (~200MB)
- `./mms_extractor.zip` (~100MB)

### System Libraries
- `./.conda/lib/python3.9/site-packages/torch/lib/libtorch_cpu.dylib` (~500MB)

## Current Status

✅ **All large files are properly excluded** by `.gitignore`
✅ **Repository is clean and can be pushed to GitHub**
✅ **No large files will accidentally be committed**

## Recommended Solutions

### Option 1: Git LFS (Recommended for models)

Git LFS (Large File Storage) is the best solution for versioning large files. To set it up:

1. **Install Git LFS** (if not already installed):
   ```bash
   # On macOS with Homebrew
   brew install git-lfs
   
   # On Ubuntu/Debian
   sudo apt-get install git-lfs
   
   # Manual installation
   curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
   sudo apt-get install git-lfs
   ```

2. **Initialize Git LFS in your repository**:
   ```bash
   git lfs install
   ```

3. **Track large file types**:
   ```bash
   git lfs track "*.safetensors"
   git lfs track "*.npz"
   git lfs track "*.zip"
   git lfs track "*.dylib"
   ```

4. **Add and commit the .gitattributes file**:
   ```bash
   git add .gitattributes
   git commit -m "Add Git LFS tracking for large files"
   ```

5. **Add your large files**:
   ```bash
   git add models/ko-sbert-nli/model.safetensors
   git add data/item_embeddings_250527.npz
   git commit -m "Add large model and data files via Git LFS"
   git push
   ```

### Option 2: External Storage

For very large files like the Llama3 models (~20GB total), consider:

1. **Cloud Storage**: Store on Google Drive, AWS S3, or similar
2. **Download Script**: Create a script to download files when needed
3. **Model Hub**: Use Hugging Face Hub or similar model repositories

### Option 3: Local Development Only

Keep large files local and provide instructions for users to:
1. Download models from original sources
2. Place them in the correct directories
3. Use the provided configuration to load them

## Implementation Example

Here's how to set up automatic model downloading:

```python
# In your code, add automatic model downloading
import os
import requests
from pathlib import Path

def download_model_if_needed():
    model_path = Path("models/ko-sbert-nli/model.safetensors")
    if not model_path.exists():
        print("Downloading Korean SBERT model...")
        # Add download logic here
        pass
```

## Current .gitignore Configuration

The `.gitignore` file has been updated to exclude:
- All `.safetensors` files
- All `.npz` files  
- All `.zip` files
- All `.dylib` files
- Model directories (`models/`, `model/`)
- Data directory (`data/`)
- Large package files

## Next Steps

1. **Choose your preferred approach** from the options above
2. **Install Git LFS** if you want to version large files
3. **Set up download scripts** for models if keeping them external
4. **Update documentation** with setup instructions for new users

## Notes

- GitHub has a 100MB file size limit for regular Git
- Git LFS allows up to 2GB per file (with paid plans supporting more)
- The current setup ensures the repository remains clean and pushable
- All sensitive files with API keys are also properly excluded 