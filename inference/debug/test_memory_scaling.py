#!/usr/bin/env python3
"""
Quick test to check memory estimates with different context sizes.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runner.pipeline_cache import LocalPipelineCacheManager
from models import Model, ModelProfile, ModelProvider, ModelDetails, ModelParameters, ModelTask


def create_model() -> Model:
    return Model(
        id="test-model",
        name="Test Model 7B",
        model="test-model", 
        provider=ModelProvider.LLAMA_CPP,
        task=ModelTask.TEXTTOTEXT,
        modified_at="2023-01-01T00:00:00Z",
        digest="test-digest",
        size=4 * 1024 * 1024 * 1024,  # 4GB model file
        details=ModelDetails(
            format="gguf",
            family="llama",
            families=["llama"],
            parameter_size="7B",
            quantization_level="q4_k_m", 
            precision="fp16",
            size=4 * 1024 * 1024 * 1024,  # 4GB model file
            original_ctx=4096,
        ),
        lora_weights=[],
        ollama_keep_alive=None
    )


def test_context_sizes():
    """Test memory estimates with different context sizes."""
    cache_manager = LocalPipelineCacheManager()
    model = create_model()
    
    test_cases = [
        ("Small context (4K)", 4096, 512),
        ("Medium context (8K)", 8192, 512), 
        ("Large context (16K)", 16384, 512),
        ("Very large context (32K)", 32768, 512),
        ("Extreme context (32K + large batch)", 32768, 2048),
    ]
    
    print("🧪 Testing memory estimates with different context sizes:")
    print("=" * 60)
    
    for name, ctx_size, batch_size in test_cases:
        profile = ModelProfile(
            user_id="test_user",
            name="Test Profile",
            model_name="test-model",
            system_prompt="You are a helpful assistant.",
            type=1,
            parameters=ModelParameters(
                num_ctx=ctx_size,
                batch_size=batch_size,
                temperature=0.7,
            )
        )
        
        estimate = cache_manager.estimate_memory(model, profile)
        print(f"{name:30s}: {estimate/1e9:6.2f}GB (ctx:{ctx_size:5d}, batch:{batch_size:4d})")
    
    print("\n📊 Analysis:")
    print("- 4GB model file (7B parameters at q4_k_m quantization)")
    print("- Memory scales dramatically with context size due to KV cache")
    print("- Large batch size increases activation memory")


if __name__ == "__main__":
    test_context_sizes()