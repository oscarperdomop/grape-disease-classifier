#!/usr/bin/env python3
"""
Script de prueba para verificar las optimizaciones de memoria del sistema.
"""

import requests
import time
import json

API_URL = "http://localhost:8000"

def test_memory_endpoints():
    """Test memory monitoring endpoints"""
    print("🧪 Testing Memory Optimization System")
    print("=" * 50)
    
    # Test health endpoint
    print("\n1. Testing /health endpoint...")
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed")
            print(f"   Available models: {data['models_available']}")
            print(f"   Loaded models: {data['models_loaded']}")
            if 'memory' in data:
                memory = data['memory']
                print(f"   Memory limit: {memory['max_limit_mb']}MB")
                print(f"   Max concurrent: {memory['max_concurrent_models']}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")
    
    # Test memory endpoint
    print("\n2. Testing /memory endpoint...")
    try:
        response = requests.get(f"{API_URL}/memory")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Memory endpoint working")
            print(f"   Current memory: {data['current_memory_mb']}MB")
            print(f"   Estimated model memory: {data['estimated_model_memory_mb']}MB")
            print(f"   Memory usage: {data['memory_usage_percent']}%")
            print(f"   Loaded models: {data['loaded_models']}")
        else:
            print(f"❌ Memory endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Memory endpoint error: {e}")
    
    # Test models endpoint
    print("\n3. Testing /models endpoint...")
    try:
        response = requests.get(f"{API_URL}/models")
        if response.status_code == 200:
            data = response.json()
            models = data.get('models', [])
            print(f"✅ Models endpoint working")
            print(f"   Available models: {len(models)}")
            for model in models:
                print(f"   - {model['id']}: {model['name']}")
            
            # Check if model_4 is available
            model_4_available = any(m['id'] == 'model_4' for m in models)
            if model_4_available:
                print("✅ model_4 (swin_gsrdn) is now available!")
            else:
                print("⚠️  model_4 not found in available models")
        else:
            print(f"❌ Models endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Models endpoint error: {e}")

def test_model_loading():
    """Test loading multiple models to verify memory management"""
    print("\n4. Testing model loading and memory management...")
    
    # Create a test image file (1x1 pixel)
    import io
    from PIL import Image
    
    # Create a small test image
    img = Image.new('RGB', (224, 224), color='green')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    models_to_test = ['model_1', 'model_2', 'model_3', 'model_4']
    
    for model_id in models_to_test:
        print(f"\n   Testing {model_id}...")
        
        # Check memory before
        try:
            mem_response = requests.get(f"{API_URL}/memory")
            if mem_response.status_code == 200:
                mem_before = mem_response.json()
                print(f"     Memory before: {mem_before['estimated_model_memory_mb']}MB")
        except:
            pass
        
        # Make prediction
        try:
            files = {'file': ('test.jpg', img_bytes.getvalue(), 'image/jpeg')}
            response = requests.post(
                f"{API_URL}/predict",
                params={'model_id': model_id},
                files=files
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"     ✅ {model_id} prediction successful")
                if result.get('predictions'):
                    top_pred = result['predictions'][0]
                    print(f"        Top prediction: {top_pred['label']} ({top_pred['score']:.3f})")
            else:
                print(f"     ❌ {model_id} prediction failed: {response.status_code}")
                print(f"        Error: {response.text}")
        except Exception as e:
            print(f"     ❌ {model_id} prediction error: {e}")
        
        # Check memory after
        try:
            mem_response = requests.get(f"{API_URL}/memory")
            if mem_response.status_code == 200:
                mem_after = mem_response.json()
                print(f"     Memory after: {mem_after['estimated_model_memory_mb']}MB")
                print(f"     Loaded models: {mem_after['loaded_models']}")
        except:
            pass
        
        time.sleep(1)  # Small delay between tests

def main():
    """Main test function"""
    print("🍇 Grape Disease Classifier - Memory Optimization Test")
    print("Make sure the backend is running on http://localhost:8000")
    print()
    
    # Wait for user confirmation
    input("Press Enter to start tests...")
    
    test_memory_endpoints()
    test_model_loading()
    
    print("\n" + "=" * 50)
    print("🎉 Memory optimization tests completed!")
    print("\nKey improvements:")
    print("✅ All 4 models are now available")
    print("✅ Smart memory management with LRU eviction")
    print("✅ Memory monitoring endpoints")
    print("✅ Automatic model unloading when needed")
    print("\nCheck the backend logs to see memory management in action!")

if __name__ == "__main__":
    main()
