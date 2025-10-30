#!/usr/bin/env python3
"""Test basic llama.cpp loading without multimodal to isolate the issue."""

import llama_cpp

def test_basic_llama_loading():
    """Test basic llama.cpp model loading without multimodal features."""
    
    model_path = "/models/qwen3-vl-32b/qwen3-vl-32b-q4-k-m-thinking-abliterated.gguf"
    
    print(f"🚀 Testing basic llama.cpp loading of: {model_path}")
    
    try:
        # Try minimal configuration first
        print("📋 Attempting minimal configuration...")
        llama = llama_cpp.Llama(
            model_path=model_path,
            n_ctx=512,  # Very small context to minimize memory usage
            n_gpu_layers=0,  # CPU only to avoid GPU issues
            verbose=False,
        )
        print("✅ Basic CPU loading successful!")
        del llama
        return True
        
    except Exception as e:
        print(f"❌ Basic CPU loading failed: {e}")
        
    try:
        # Try with GPU layers
        print("📋 Attempting with GPU layers...")
        llama = llama_cpp.Llama(
            model_path=model_path,
            n_ctx=512,
            n_gpu_layers=1,  # Just one layer
            verbose=True,  # Enable verbose to see what's happening
        )
        print("✅ GPU loading successful!")
        del llama
        return True
        
    except Exception as e:
        print(f"❌ GPU loading failed: {e}")
        
    return False

if __name__ == "__main__":
    success = test_basic_llama_loading()
    exit(0 if success else 1)