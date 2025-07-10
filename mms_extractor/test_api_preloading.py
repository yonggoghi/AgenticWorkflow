#!/usr/bin/env python3
"""
Test script to demonstrate API preloading functionality.
This script shows how preloading improves API response times.
"""

import requests
import time
import json

def test_api_endpoint(url, description):
    """Test an API endpoint and measure response time."""
    print(f"\n🧪 Testing: {description}")
    print(f"📡 URL: {url}")
    
    try:
        start_time = time.time()
        response = requests.get(url, timeout=10)
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            print(f"✅ SUCCESS ({response.status_code}) - Response time: {response_time:.2f}s")
            return response.json(), response_time
        else:
            print(f"❌ FAILED ({response.status_code}) - Response time: {response_time:.2f}s")
            return None, response_time
            
    except requests.exceptions.ConnectionError:
        print("❌ FAILED - API server not running")
        return None, 0
    except requests.exceptions.Timeout:
        print("❌ TIMEOUT - Request took too long")
        return None, 10
    except Exception as e:
        print(f"❌ ERROR - {e}")
        return None, 0

def test_extraction(message, extraction_mode="nlp"):
    """Test message extraction and measure response time."""
    url = "http://localhost:8080/extract"
    
    print(f"\n🔍 Testing Extraction: {extraction_mode} mode")
    print(f"📝 Message: {message[:50]}...")
    
    data = {
        "message": message,
        "model": "gemma_3",
        "extraction_mode": extraction_mode
    }
    
    try:
        start_time = time.time()
        response = requests.post(url, json=data, timeout=30)
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            processing_time = result.get('metadata', {}).get('processing_time_seconds', 0)
            print(f"✅ SUCCESS - Total time: {response_time:.2f}s, Processing: {processing_time:.2f}s")
            
            # Show extracted products
            products = result.get('result', {}).get('product', [])
            if products:
                print(f"🎯 Extracted {len(products)} products:")
                for product in products[:3]:  # Show first 3
                    name = product.get('item_name_in_msg', 'Unknown')
                    print(f"   • {name}")
            else:
                print("🎯 No products extracted")
                
            return result, response_time
        else:
            print(f"❌ FAILED ({response.status_code}) - Response time: {response_time:.2f}s")
            print(f"Response: {response.text}")
            return None, response_time
            
    except requests.exceptions.ConnectionError:
        print("❌ FAILED - API server not running")
        return None, 0
    except requests.exceptions.Timeout:
        print("❌ TIMEOUT - Request took too long (>30s)")
        return None, 30
    except Exception as e:
        print(f"❌ ERROR - {e}")
        return None, 0

def main():
    """Main test function."""
    print("🚀 MMS Extractor API Preloading Test")
    print("=" * 60)
    
    # Test health check
    health_data, health_time = test_api_endpoint(
        "http://localhost:8080/health", 
        "Health Check"
    )
    
    if not health_data:
        print("\n❌ API server is not running. Please start it with:")
        print("   python api.py --model-loading-mode local --port 8080")
        return
    
    # Test status endpoint
    status_data, status_time = test_api_endpoint(
        "http://localhost:8080/status", 
        "Status Check (shows preloaded models)"
    )
    
    if status_data:
        extractors = status_data.get('preloaded_extractors', {})
        count = extractors.get('count', 0)
        print(f"📊 Preloaded extractors: {count}")
        
        if count > 0:
            details = extractors.get('details', {})
            for key in details.keys():
                print(f"   • {key}")
        else:
            print("⚠️  No extractors preloaded")
    
    # Test model info
    model_data, model_time = test_api_endpoint(
        "http://localhost:8080/model-info",
        "Model Information"
    )
    
    if model_data:
        config = model_data.get('embedding_model_config', {})
        loading_mode = config.get('loading_mode', 'unknown')
        print(f"🔧 Model loading mode: {loading_mode}")
        
        status = model_data.get('model_status', {})
        if isinstance(status, dict):
            model_loaded = status.get('model_loaded', False)
            device = status.get('device', 'unknown')
            print(f"📍 Model loaded: {model_loaded}, Device: {device}")
    
    # Test extraction (this should be fast if preloaded)
    test_message = "광고 제목:[SK텔레콤] 2월 0 day 혜택 안내 - 베어유 모든 강의 14일 무료 수강 쿠폰 드립니다!"
    
    extraction_result, extraction_time = test_extraction(test_message, "nlp")
    
    # Summary
    print("\n" + "=" * 60)
    print("📈 PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"Health check:     {health_time:.2f}s")
    print(f"Status check:     {status_time:.2f}s") 
    print(f"Model info:       {model_time:.2f}s")
    print(f"Extraction (NLP): {extraction_time:.2f}s")
    
    total_time = health_time + status_time + model_time + extraction_time
    print(f"Total test time:  {total_time:.2f}s")
    
    # Analysis
    print("\n💡 ANALYSIS:")
    if extraction_time < 15:
        print("✅ Excellent performance! Models are preloaded.")
        print("   First extraction request was fast (<15s)")
    elif extraction_time < 30:
        print("⚠️  Good performance, but could be better.")
        print("   Consider using --preload-all-modes for faster mode switching")
    else:
        print("🐌 Slow performance detected.")
        print("   Models might not be preloaded. Check server startup logs.")
    
    print("\n🔧 TIPS:")
    print("• For fastest API responses: Use default preloading")
    print("• For development: Use --no-preload for faster server restarts")
    print("• For production: Keep preloading enabled")

if __name__ == "__main__":
    main() 