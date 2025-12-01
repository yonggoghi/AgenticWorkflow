#!/usr/bin/env python3
"""
Tests for Medium Priority Improvements
======================================

Tests for:
1. Retry utilities (retry_utils.py)
2. Product schema transformer (schema_transformer.py)
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
from utils.retry_utils import RetryConfig, with_retry, LLMRetryManager
from services.schema_transformer import ProductSchemaTransformer

def test_retry_decorator():
    """Test retry decorator functionality"""
    print("\n" + "="*60)
    print("TEST 1: Retry Decorator")
    print("="*60)
    
    # Test 1.1: Success on first attempt
    print("\n✓ Testing success on first attempt...")
    call_count = {'count': 0}
    
    @with_retry(RetryConfig(max_attempts=3))
    def succeed_first_time():
        call_count['count'] += 1
        return "success"
    
    result = succeed_first_time()
    assert result == "success"
    assert call_count['count'] == 1
    print(f"  ✅ Succeeded on first attempt (called {call_count['count']} time)")
    
    # Test 1.2: Retry on failure
    print("\n✓ Testing retry on failure...")
    call_count2 = {'count': 0}
    
    @with_retry(RetryConfig(max_attempts=3, retry_delay_seconds=0.1))
    def succeed_on_third():
        call_count2['count'] += 1
        if call_count2['count'] < 3:
            raise ValueError(f"Attempt {call_count2['count']} failed")
        return "success after retries"
    
    result = succeed_on_third()
    assert result == "success after retries"
    assert call_count2['count'] == 3
    print(f"  ✅ Succeeded after {call_count2['count']} attempts")
    
    # Test 1.3: All retries exhausted
    print("\n✓ Testing all retries exhausted...")
    call_count3 = {'count': 0}
    
    @with_retry(RetryConfig(max_attempts=2, retry_delay_seconds=0.1))
    def always_fail():
        call_count3['count'] += 1
        raise ValueError("Always fails")
    
    try:
        always_fail()
        assert False, "Should have raised exception"
    except ValueError:
        assert call_count3['count'] == 2
        print(f"  ✅ Failed as expected after {call_count3['count']} attempts")
    
    print("\n" + "="*60)
    print("✅ ALL RETRY DECORATOR TESTS PASSED!")
    print("="*60)
    return True


def test_llm_retry_manager():
    """Test LLMRetryManager functionality"""
    print("\n" + "="*60)
    print("TEST 2: LLM Retry Manager")
    print("="*60)
    
    manager = LLMRetryManager(RetryConfig(max_attempts=2, retry_delay_seconds=0.1))
    
    # Test 2.1: Primary succeeds
    print("\n✓ Testing primary call success...")
    call_log = []
    
    def primary_success():
        call_log.append("primary")
        return "primary_result"
    
    result = manager.call_with_fallback(primary_success)
    assert result == "primary_result"
    assert call_log == ["primary"]
    print("  ✅ Primary call succeeded")
    
    # Test 2.2: Primary fails, fallback succeeds
    print("\n✓ Testing fallback on primary failure...")
    call_log2 = []
    
    def primary_fail():
        call_log2.append("primary")
        raise ValueError("Primary failed")
    
    def fallback1_success():
        call_log2.append("fallback1")
        return "fallback1_result"
    
    result = manager.call_with_fallback(
        primary_fail,
        fallback_calls=[fallback1_success],
        fallback_models=['gpt']
    )
    assert result == "fallback1_result"
    assert "fallback1" in call_log2
    print("  ✅ Fallback succeeded after primary failure")
    
    print("\n" + "="*60)
    print("✅ ALL LLM RETRY MANAGER TESTS PASSED!")
    print("="*60)
    return True


def test_schema_transformer():
    """Test ProductSchemaTransformer functionality"""
    print("\n" + "="*60)
    print("TEST 3: Product Schema Transformer")
    print("="*60)
    
    transformer = ProductSchemaTransformer()
    
    # Test 3.1: Empty DataFrame
    print("\n✓ Testing empty DataFrame...")
    result = transformer.transform_to_item_centric(pd.DataFrame())
    assert result == []
    print("  ✅ Empty DataFrame returns empty list")
    
    # Test 3.2: Single product
    print("\n✓ Testing single product transformation...")
    df = pd.DataFrame([{
        "item_nm": "갤럭시S24",
        "item_id": "PROD123",
        "item_name_in_msg": "갤럭시 S24"
    }])
    
    result = transformer.transform_to_item_centric(df)
    assert len(result) == 1
    assert result[0]['item_nm'] == "갤럭시S24"
    assert result[0]['item_id'] == ["PROD123"]
    assert "갤럭시 S24" in result[0]['item_name_in_msg']
    print(f"  ✅ Single product transformed: {result[0]['item_nm']}")
    
    # Test 3.3: Multiple products with same item_nm
    print("\n✓ Testing aggregation of multiple products...")
    df = pd.DataFrame([
        {"item_nm": "갤럭시S24", "item_id": "P1", "item_name_in_msg": "갤럭시"},
        {"item_nm": "갤럭시S24", "item_id": "P2", "item_name_in_msg": "S24"},
        {"item_nm": "아이폰15", "item_id": "P3", "item_name_in_msg": "아이폰"}
    ])
    
    result = transformer.transform_to_item_centric(df)
    assert len(result) == 2  # Two unique item_nm values
    
    galaxy_item = [r for r in result if r['item_nm'] == "갤럭시S24"][0]
    assert set(galaxy_item['item_id']) == {"P1", "P2"}
    assert len(galaxy_item['item_name_in_msg']) >= 1
    print(f"  ✅ Aggregated 2 products into 1 item with {len(galaxy_item['item_id'])} IDs")
    
    # Test 3.4: With expected_action field
    print("\n✓ Testing with expected_action field...")
    df = pd.DataFrame([{
        "item_nm": "갤럭시S24",
        "item_id": "PROD123",
        "item_name_in_msg": "갤럭시",
        "expected_action": "구매"
    }])
    
    result = transformer.transform_to_item_centric(df)
    assert 'expected_action' in result[0]
    assert "구매" in result[0]['expected_action']
    print(f"  ✅ expected_action field preserved: {result[0]['expected_action']}")
    
    print("\n" + "="*60)
    print("✅ ALL SCHEMA TRANSFORMER TESTS PASSED!")
    print("="*60)
    return True


if __name__ == "__main__":
    try:
        success = True
        
        # Run tests
        success &= test_retry_decorator()
        success &= test_llm_retry_manager()
        success &= test_schema_transformer()
        
        print("\n" + "="*60)
        print("="*60)
        if success:
            print("🎉 ALL MEDIUM PRIORITY TESTS PASSED!")
            print("\nImplemented:")
            print("  1. ✅ Retry utilities (decorator + manager)")
            print("  2. ✅ Product schema transformer")
            print("\nNext steps:")
            print("  • Add retry configuration to settings")
            print("  • Update LLMExtractionStep to use retry")
            print("  • Update ResultBuilder to use schema transformer")
        else:
            print("❌ SOME TESTS FAILED")
        print("="*60)
        print("="*60)
        
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
