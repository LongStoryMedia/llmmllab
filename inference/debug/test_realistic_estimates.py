#!/usr/bin/env python3
"""
Test realistic memory estimates for different model sizes and context windows.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runner.pipeline_cache import LocalPipelineCacheManager
from models import Model, ModelProfile, ModelProvider, ModelDetails, ModelParameters, ModelTask


def create_model(name: str, param_size: str, file_size_gb: float) -> Model:
    """Create a model with specified parameters."""
    return Model(
        id=f"test-{name}",
        name=f"Test {name}",
        model=f"test-{name}", 
        provider=ModelProvider.LLAMA_CPP,
        task=ModelTask.TEXTTOTEXT,
        modified_at="2023-01-01T00:00:00Z",
        digest=f"test-digest-{name}",
        size=int(file_size_gb * 1024 * 1024 * 1024),
        details=ModelDetails(
            format="gguf",
            family="llama",
            families=["llama"],
            parameter_size=param_size,
            quantization_level="q4_k_m", 
            precision="fp16",
            size=int(file_size_gb * 1024 * 1024 * 1024),
            original_ctx=4096,
        ),
        lora_weights=[],
        ollama_keep_alive=None
    )


def create_profile(ctx_size: int) -> ModelProfile:
    """Create a profile with specified context size."""
    return ModelProfile(
        user_id="test_user",
        name=f"Profile {ctx_size}K",
        model_name="test-model",
        system_prompt="You are a helpful assistant.",
        type=1,
        parameters=ModelParameters(
            num_ctx=ctx_size,
            batch_size=512,  # Standard batch size
            temperature=0.7,
        )
    )


def test_realistic_estimates():
    """Test memory estimates for realistic model configurations."""
    cache_manager = LocalPipelineCacheManager()
    
    # Define test models (parameter size, typical file size in GB)
    models = [
        ("7B", 3.8),    # 7B model at q4_k_m
        ("13B", 7.3),   # 13B model at q4_k_m  
        ("30B", 17.0),  # 30B model at q4_k_m
        ("70B", 40.0),  # 70B model at q4_k_m
    ]
    
    # Define test context sizes
    contexts = [4096, 8192, 16384, 32768, 131072]
    
    print("🧪 Realistic Memory Estimates for LLM Pipeline Cache")
    print("=" * 75)
    print(f"{'Model':>8} {'File GB':>8} {'4K Ctx':>8} {'8K Ctx':>8} {'16K Ctx':>8} {'32K Ctx':>8} {'131K Ctx':>8}")
    print("-" * 75)
    
    for param_size, file_gb in models:
        model = create_model(param_size, param_size, file_gb)
        estimates = []
        
        for ctx_size in contexts:
            profile = create_profile(ctx_size)
            estimate = cache_manager.estimate_memory(model, profile)
            estimates.append(estimate / 1e9)  # Convert to GB
        
        print(f"{param_size:>8} {file_gb:>8.1f}   ", end="")
        for est in estimates:
            print(f"{est:>6.1f}GB ", end="")
        print()
    
    print("\n📊 Analysis:")
    print("- Memory scales with both model size and context window")
    print("- KV cache is the main driver of context-related memory usage")
    print("- 32K context adds significant memory overhead vs 8K")
    print("- 131K context (like Claude) requires substantial memory")
    print("- Estimates include: model weights + KV cache + activation buffer + overhead")


if __name__ == "__main__":
    test_realistic_estimates()