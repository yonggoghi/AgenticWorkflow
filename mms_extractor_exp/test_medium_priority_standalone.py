#!/usr/bin/env python3
"""
Standalone Tests for Medium Priority Improvements
================================================

Tests for:
1. Retry utilities (retry_utils.py) - standalone
2. Product schema transformer (schema_transformer.py) - standalone
"""

import sys
import os
sys.path.insert(0, '.')

# Test imports work
print("Testing imports...")
try:
    # Import retry utils directly (doesn't need torch)
    from utils.retry_utils import RetryConfig, with_retry, LLMRetryManager
    print("✅ Retry utils imported")
except ImportError as e:
    print(f"❌ Failed to import retry utils: {e}")
    sys.exit(1)

# For schema transformer, we need pandas but not torch
try:
    import pandas as pd
    print("✅ Pandas imported")
except ImportError:
    print("❌ Pandas not available")
    sys.exit(1)

# Create minimal mocks for dependencies
class MockUtils:
    @staticmethod
    def select_most_comprehensive(items):
        """Mock implementation"""
        if not items:
            return []
        # Return longest items
        max_len = max(len(str(item)) for item in items)
        return [item for item in items if len(str(item)) == max_len]

# Monkey patch for testing
sys.modules['utils'] = sys.modules[__name__]
select_most_comprehensive = MockUtils.select_most_comprehensive

try:
    from services.schema_transformer import ProductSchemaTransformer
    print("✅ Schema transformer imported")
except ImportError as e:
    print(f"❌ Failed to import schema transformer: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("All imports successful! Running tests...")
print("="*60)


def test_retry_config():
    """Test RetryConfig dataclass"""
    print("\n### TEST: RetryConfig")
    
    # Default config
    config = RetryConfig()
    assert config.max_attempts == 3
    assert config.retry_delay_seconds == 1.0
    assert config.fallback_models == ['ax', 'gpt']
    print("  ✅ Default config initialized")
    
    # Custom config
    config = RetryConfig(max_attempts=5, retry_delay_seconds=2.0)
    assert config.max_attempts == 5
    assert config.retry_delay_seconds == 2.0
    print("  ✅ Custom config works")
    
    # Exponential backoff
    config = RetryConfig(exponential_backoff=True, retry_delay_seconds=1.0)
    assert config.get_delay(1) == 1.0
    assert config.get_delay(2) == 2.0
    assert config.get_delay(3) == 4.0
    print("  ✅ Exponential backoff calculation works")
    
    return True


def test_retry_decorator():
    """Test retry decorator"""
    print("\n### TEST: Retry Decorator")
    
    # Success on first try
    calls = []
    
    @with_retry(RetryConfig(max_attempts=3, retry_delay_seconds=0.01))
    def succeed_immediately():
        calls.append(1)
        return "success"
    
    result = succeed_immediately()
    assert result == "success"
    assert len(calls) == 1
    print("  ✅ Success on first attempt")
    
    # Success after retries
    calls2 = []
    
    @with_retry(RetryConfig(max_attempts=3, retry_delay_seconds=0.01))
    def succeed_on_third():
        calls2.append(1)
        if len(calls2) < 3:
            raise ValueError("Not yet")
        return "success"
    
    result = succeed_on_third()
    assert result == "success"
    assert len(calls2) == 3
    print("  ✅ Success after retries")
    
    # Exhaust retries
    calls3 = []
    
    @with_retry(RetryConfig(max_attempts=2, retry_delay_seconds=0.01))
    def always_fail():
        calls3.append(1)
        raise ValueError("Always fails")
    
    try:
        always_fail()
        assert False, "Should have raised"
    except ValueError:
        assert len(calls3) == 2
        print("  ✅ Properly exhausts retries")
    
    return True


def test_schema_transformer():
    """Test schema transformer"""
    print("\n### TEST: Schema Transformer")
    
    transformer = ProductSchemaTransformer()
    
    # Empty DataFrame
    result = transformer.transform_to_item_centric(pd.DataFrame())
    assert result == []
    print("  ✅ Empty DataFrame returns []")
    
    # Single product
    df = pd.DataFrame([{
        'item_nm': '갤럭시S24',
        'item_id': 'P1',
        'item_name_in_msg': '갤럭시'
    }])
    result = transformer.transform_to_item_centric(df)
    assert len(result) == 1
    assert result[0]['item_nm'] == '갤럭시S24'
    assert result[0]['item_id'] == ['P1']
    print("  ✅ Single product transforms correctly")
    
    # Multiple products, same item_nm
    df = pd.DataFrame([
        {'item_nm': '갤럭시S24', 'item_id': 'P1', 'item_name_in_msg': '갤럭시'},
        {'item_nm': '갤럭시S24', 'item_id': 'P2', 'item_name_in_msg': 'S24'}
    ])
    result = transformer.transform_to_item_centric(df)
    assert len(result) == 1
    assert set(result[0]['item_id']) == {'P1', 'P2'}
    print("  ✅ Aggregation works correctly")
    
    return True


if __name__ == "__main__":
    try:
        test_retry_config()
        test_retry_decorator()
        test_schema_transformer()
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED!")
        print("="*60)
        print("\nNew components verified:")
        print("  ✅ RetryConfig dataclass")
        print("  ✅ with_retry decorator")  
        print("  ✅ ProductSchemaTransformer")
        print("\nReady for integration!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed:  {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
