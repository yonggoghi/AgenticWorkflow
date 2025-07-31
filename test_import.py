#!/usr/bin/env python3
"""
Test script to verify correct imports and function signatures.
"""
import sys
from pathlib import Path

# Add mms_extractor to path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir / 'mms_extractor'))

try:
    from mms_extractor.utils.similarity import parallel_seq_similarity
    import inspect
    
    # Get function signature
    sig = inspect.signature(parallel_seq_similarity)
    print("Function signature:")
    print(f"parallel_seq_similarity{sig}")
    
    # Check parameter names
    param_names = list(sig.parameters.keys())
    print(f"\nParameter names: {param_names}")
    
    # Check if normalization_value exists
    if 'normalization_value' in param_names:
        print("✅ Correct parameter 'normalization_value' found")
    else:
        print("❌ Parameter 'normalization_value' NOT found")
    
    if 'normalizaton_value' in param_names:
        print("❌ Incorrect parameter 'normalizaton_value' found (typo)")
    else:
        print("✅ No typo parameter found")
        
    # Get module path
    print(f"\nModule path: {parallel_seq_similarity.__module__}")
    print(f"File location: {inspect.getfile(parallel_seq_similarity)}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc() 