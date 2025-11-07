#!/usr/bin/env python3
"""
Quick test of memory calculation accuracy after fixing the activation memory bug.
"""

import sys
import asyncio
sys.path.append('/app')

from models import Model, ModelDetails, OptimalParameters
from runner.utils.resizer import Resizer
from utils.logging import llmmllogger

def test_memory_calculation():
    """Test memory calculation with a typical 4B model configuration."""
    
    logger = llmmllogger.bind(component="MemoryTest")
    
    # Create a typical 4B model (like Qwen2-4B or similar)
    model_details = ModelDetails(
        family="qwen",
        families=["qwen"],
        format="gguf",
        parameter_size="4B",
        quantization_level="Q4_K_M",  # 4-bit quantization
        precision=None,
        size=2_400_000_000,  # ~2.4GB file size in bytes (typical for Q4 4B model)
        original_ctx=32768,
        n_layers=32,
        hidden_size=3584,
        n_heads=28,
        n_kv_heads=4,
    )
    
    model = Model(
        id="test-qwen2-4b",
        name="Test Qwen2 4B",
        model="test-qwen2-4b", 
        task="TextToText",
        modified_at="2024-01-01T00:00:00Z",
        digest="test-digest",
        provider="llama_cpp",
        details=model_details,
    )
    
    # Test different parameter configurations
    configs = [
        OptimalParameters(n_ctx=2048, n_batch=512, n_ubatch=512, n_gpu_layers=32),
        OptimalParameters(n_ctx=4096, n_batch=512, n_ubatch=512, n_gpu_layers=32),
        OptimalParameters(n_ctx=8192, n_batch=256, n_ubatch=256, n_gpu_layers=32),
        OptimalParameters(n_ctx=2048, n_batch=1024, n_ubatch=512, n_gpu_layers=16),  # Half layers
    ]
    
    resizer = Resizer()
    
    logger.info("🧮 Testing memory calculations for 4B model...")
    
    for i, config in enumerate(configs):
        logger.info(f"\n📊 Configuration {i+1}: {config}")
        
        breakdown = resizer.calculate_memory_breakdown(config, model)
        
        logger.info(f"  Model weights: {breakdown['model_weights_gpu_gb']:.2f}GB")
        logger.info(f"  KV cache: {breakdown['kv_cache_gb']:.2f}GB") 
        logger.info(f"  Activation: {breakdown['activation_gb']:.2f}GB")
        logger.info(f"  Overhead: {breakdown['overhead_gb']:.2f}GB")
        logger.info(f"  TOTAL GPU: {breakdown['total_gpu_gb']:.2f}GB")
        
        # Sanity checks
        if breakdown['total_gpu_gb'] > 50:
            logger.error(f"❌ FAIL: Total GPU memory {breakdown['total_gpu_gb']:.1f}GB is unreasonably high!")
        elif breakdown['total_gpu_gb'] < 1:
            logger.error(f"❌ FAIL: Total GPU memory {breakdown['total_gpu_gb']:.1f}GB is unreasonably low!")
        else:
            logger.info(f"✅ PASS: Total GPU memory {breakdown['total_gpu_gb']:.1f}GB looks reasonable")

if __name__ == "__main__":
    test_memory_calculation()