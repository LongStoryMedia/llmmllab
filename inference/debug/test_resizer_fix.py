#!/usr/bin/env python3
"""
Test the fixed Resizer class for realistic memory estimation.
"""

from models import Model, ModelDetails, OptimalParameters
from runner.utils.resizer import Resizer
from datetime import datetime

def test_resizer_fix():
    """Test that the fixed Resizer gives realistic estimates."""
    
    # Create test model (30B parameter model like in production)
    model = Model(
        id="Qwen/Qwen3-VL-30B-A3B-Thinking",
        name="Qwen3-VL-30B-A3B-Thinking",
        model="Qwen3-VL-30B-A3B-Thinking",
        task="TextToText",
        details=ModelDetails(
            format="gguf",
            family="qwen",
            families=["qwen"],
            parameter_size="30B",
            size=18_556_687_136,  # Actual size from production log
            original_ctx=262144,
            quantization_level="q4_k_m",
            dtype="Q4_K_M",
            clip_model_size=1083499616,  # Actual clip size
            n_layers=48,
            hidden_size=7168,
            n_heads=32,
            n_kv_heads=4,
        ),
        modified_at="2024-01-01T00:00:00Z",
        digest="abc123",
        provider="llama_cpp",
    )
    
    # Create optimal parameters from production
    params = OptimalParameters(
        n_ctx=131072,  # From production log
        n_batch=4096,  # From production log
        n_ubatch=4096,
        n_gpu_layers=48,  # All layers
    )
    
    print("🧪 Testing fixed Resizer with realistic 30B model")
    print(f"Model: {model.name}")
    print(f"Actual file size: {model.details.size / (1024**3):.2f}GB")
    print(f"Parameters: n_ctx={params.n_ctx}, n_batch={params.n_batch}, n_gpu_layers={params.n_gpu_layers}")
    print()
    
    # Test the fixed Resizer
    resizer = Resizer()
    breakdown = resizer.calculate_memory_breakdown(params, model)
    
    print("📊 Fixed Resizer Memory Breakdown:")
    print(f"   Total GPU Memory: {breakdown['total_gpu_gb']:.2f} GB")
    print(f"   Model Weights: {breakdown['model_weights_gpu_gb']:.2f} GB")
    print(f"   KV Cache: {breakdown['kv_cache_gb']:.2f} GB")
    print(f"   Activation: {breakdown['activation_gb']:.2f} GB") 
    print(f"   Overhead: {breakdown['overhead_gb']:.2f} GB")
    print(f"   Vision Tower: {breakdown['clip_model_gb']:.2f} GB")
    print()
    
    # Check if estimate is reasonable (should be 15-30GB for 30B model)
    total_gb = breakdown['total_gpu_gb']
    
    print("✅ Validation:")
    if 15 <= total_gb <= 30:
        print(f"✅ Total estimate {total_gb:.2f}GB looks reasonable for 30B model")
    else:
        print(f"⚠️  Total estimate {total_gb:.2f}GB may be off for 30B model")
    
    # Compare to real measurements (16.9-21.5GB)
    real_min, real_max = 16.9, 21.5
    if real_min <= total_gb <= real_max * 1.5:  # Allow 50% tolerance
        print(f"✅ Estimate {total_gb:.2f}GB is close to real usage ({real_min}-{real_max}GB)")
    else:
        print(f"⚠️  Estimate {total_gb:.2f}GB differs from real usage ({real_min}-{real_max}GB)")
        
    return total_gb

if __name__ == "__main__":
    test_resizer_fix()