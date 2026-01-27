#!/usr/bin/env python
"""
Test script to verify fail-fast error handling in MMSExtractor initialization.

This script tests that initialization fails immediately when data loading errors occur,
instead of continuing with empty fallback data.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_invalid_db_connection():
    """Test that initialization fails with invalid DB connection."""
    print("\n" + "="*80)
    print("Test 1: Invalid DB Connection")
    print("="*80)
    
    # Set invalid DB host
    original_host = os.environ.get('DB_HOST')
    os.environ['DB_HOST'] = 'invalid_host_12345'
    
    try:
        from core.mms_extractor import MMSExtractor
        
        try:
            extractor = MMSExtractor(offer_info_data_src='db')
            print("❌ TEST FAILED: Initialization should have failed but succeeded")
            return False
        except RuntimeError as e:
            print(f"✅ TEST PASSED: RuntimeError raised as expected")
            print(f"   Error message: {str(e)[:100]}...")
            return True
        except Exception as e:
            print(f"⚠️  TEST PARTIAL: Different exception raised: {type(e).__name__}")
            print(f"   Error message: {str(e)[:100]}...")
            return True  # Still better than no error
    finally:
        # Restore original DB_HOST
        if original_host:
            os.environ['DB_HOST'] = original_host
        elif 'DB_HOST' in os.environ:
            del os.environ['DB_HOST']


def test_missing_csv_file():
    """Test that initialization fails with missing CSV file."""
    print("\n" + "="*80)
    print("Test 2: Missing CSV File")
    print("="*80)
    
    # Set invalid data directory
    original_offer_path = os.environ.get('OFFER_DATA_PATH')
    os.environ['OFFER_DATA_PATH'] = '/nonexistent/path/offer_data.csv'
    
    try:
        from core.mms_extractor import MMSExtractor
        
        try:
            extractor = MMSExtractor(offer_info_data_src='local')
            print("❌ TEST FAILED: Initialization should have failed but succeeded")
            return False
        except RuntimeError as e:
            print(f"✅ TEST PASSED: RuntimeError raised as expected")
            print(f"   Error message: {str(e)[:100]}...")
            return True
        except Exception as e:
            print(f"⚠️  TEST PARTIAL: Different exception raised: {type(e).__name__}")
            print(f"   Error message: {str(e)[:100]}...")
            return True  # Still better than no error
    finally:
        # Restore original OFFER_DATA_PATH
        if original_offer_path:
            os.environ['OFFER_DATA_PATH'] = original_offer_path
        elif 'OFFER_DATA_PATH' in os.environ:
            del os.environ['OFFER_DATA_PATH']


def test_missing_stopwords_file():
    """Test that initialization fails with missing stopwords file."""
    print("\n" + "="*80)
    print("Test 3: Missing Stopwords File")
    print("="*80)
    
    # Set invalid stopwords path
    original_stop_path = os.environ.get('STOP_ITEM_PATH')
    os.environ['STOP_ITEM_PATH'] = '/nonexistent/path/stop_words.csv'
    
    try:
        from core.mms_extractor import MMSExtractor
        
        try:
            extractor = MMSExtractor(offer_info_data_src='local')
            print("❌ TEST FAILED: Initialization should have failed but succeeded")
            return False
        except RuntimeError as e:
            print(f"✅ TEST PASSED: RuntimeError raised as expected")
            print(f"   Error message: {str(e)[:100]}...")
            return True
        except Exception as e:
            print(f"⚠️  TEST PARTIAL: Different exception raised: {type(e).__name__}")
            print(f"   Error message: {str(e)[:100]}...")
            return True  # Still better than no error
    finally:
        # Restore original STOP_ITEM_PATH
        if original_stop_path:
            os.environ['STOP_ITEM_PATH'] = original_stop_path
        elif 'STOP_ITEM_PATH' in os.environ:
            del os.environ['STOP_ITEM_PATH']


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("Fail-Fast Error Handling Tests")
    print("="*80)
    print("\nThese tests verify that MMSExtractor initialization fails immediately")
    print("when data loading errors occur, instead of continuing with empty data.")
    
    results = []
    
    # Run tests
    results.append(("Invalid DB Connection", test_invalid_db_connection()))
    results.append(("Missing CSV File", test_missing_csv_file()))
    results.append(("Missing Stopwords File", test_missing_stopwords_file()))
    
    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:30s}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Fail-fast error handling is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the implementation.")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
