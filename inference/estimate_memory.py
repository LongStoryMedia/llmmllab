#!/usr/bin/env python3
"""
Memory estimation script for container usage.
Estimates memory requirements for llama.cpp model loading.
"""

import argparse
import sys
import os
sys.path.append('/app')

from runner.pipeline_cache import LocalPipelineCacheManager
from models import Model, ModelProfile, ModelParameters, ModelDetails

def estimate_memory_for_model(model_path: str, ctx_size: int, batch_size: int, gpu_layers: int, mmproj_path: str = None) -> float:
    """Estimate memory for a specific model configuration"""
    
    try:
        # Create a mock model object
        model_size = os.path.getsize(model_path) if os.path.exists(model_path) else 20 * 1024**3  # 20GB fallback
        mmproj_size = 0
        if mmproj_path and os.path.exists(mmproj_path):
            mmproj_size = os.path.getsize(mmproj_path)
        
        # Create mock model details
        model_details = ModelDetails(
            parent_model="test",
            format="gguf", 
            size=model_size,
            family="qwen",
            families=["Qwen"],
            parameter_size="30B",
            dtype="Q4_K_M",
            quantization_level="q4_k_m",
            specialization="Text",
            gguf_file=model_path,
            clip_model_path=mmproj_path,
            clip_model_size=mmproj_size,
            supports_thinking=True,
            supports_vision=bool(mmproj_path),
            original_ctx=262144,
            n_layers=64,
            hidden_size=5120,
            n_heads=64,
            n_kv_heads=8
        )
        
        model = Model(
            id="test-model",
            name="Test Model",
            model=model_path,
            provider="llama_cpp",
            pipeline="TestPipeline",
            modified_at="2025-01-01",
            digest="test",
            details=model_details,
            task="TextToText"
        )
        
        # Create model parameters
        model_params = ModelParameters(
            num_ctx=ctx_size,
            num_batch=batch_size,
            num_ubatch=batch_size,
            num_gpu_layers=gpu_layers,
            num_threads=24,
            num_threads_batch=24
        )
        
        # Create model profile
        model_profile = ModelProfile(
            model_id="test-model",
            parameters=model_params
        )
        
        # Use pipeline cache manager to estimate memory
        cache_manager = LocalPipelineCacheManager()
        estimated_memory_bytes = cache_manager.estimate_memory(model, model_profile)
        estimated_memory_gb = estimated_memory_bytes / (1024**3)
        
        return estimated_memory_gb
        
    except Exception as e:
        print(f"Error estimating memory: {e}", file=sys.stderr)
        return 20.0  # Fallback estimate

def main():
    parser = argparse.ArgumentParser(description="Estimate memory for llama.cpp model")
    parser.add_argument("--model", required=True, help="Path to GGUF model file")
    parser.add_argument("--ctx-size", type=int, required=True, help="Context size")
    parser.add_argument("--batch-size", type=int, required=True, help="Batch size")
    parser.add_argument("--gpu-layers", type=int, required=True, help="GPU layers")
    parser.add_argument("--mmproj", help="Path to mmproj file (optional)")
    
    args = parser.parse_args()
    
    estimated_gb = estimate_memory_for_model(
        args.model, args.ctx_size, args.batch_size, args.gpu_layers, args.mmproj
    )
    
    print(f"Model: {args.model}")
    print(f"Context Size: {args.ctx_size}")
    print(f"Batch Size: {args.batch_size}")
    print(f"GPU Layers: {args.gpu_layers}")
    if args.mmproj:
        print(f"MMPROJ: {args.mmproj}")
    print(f"Total GPU Memory: {estimated_gb:.2f}GB")

if __name__ == "__main__":
    main()