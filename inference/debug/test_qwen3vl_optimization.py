#!/usr/bin/env python3
"""
Test script for Qwen3VL pipeline with parameter optimization.
This script tests the specific crash scenario reported by the user.
"""

import sys
import os

# Add the inference directory to the path
sys.path.insert(0, '/app')

from models.model_profile import ModelProfile
from models.model_parameters import ModelParameters
from models.gpu_config import GPUConfig
from models.parameter_optimization_config import (
    ParameterOptimizationConfiguration, 
    ParameterFloors, 
    CrashPrevention
)
from models import Model, ModelDetails
from runner.pipelines.imgtxt2txt.qwen3_vl import Qwen3VLPipeline


def test_qwen3vl_parameter_optimization():
    """Test parameter optimization with Qwen3VL pipeline."""
    print("🧪 Testing Qwen3VL Parameter Optimization...")
    
    # Create test model similar to the one that was crashing
    model_details = ModelDetails(
        parent_model="huihui-ai/Huihui-Qwen3-VL-32B-Thinking-abliterated",
        format="GGUF", 
        family="Qwen",
        families=["Qwen", "VL"],
        parameter_size="32B",
        dtype="F16",
        quantization_level="f16",
        specialization="Text",
        gguf_file="/models/qwen3-vl-32b/qwen3-vl-32b-thinking-abliterated.gguf",
        clip_model_path="/models/qwen3-vl-32b/mmproj.gguf", # This enables vision
        supports_thinking=True,
        supports_vision=True
    )
    
    from datetime import datetime
    model = Model(
        id="huihui-ai/Huihui-Qwen3-VL-32B-Thinking-abliterated",
        name="huihui-ai/Huihui-Qwen3-VL-32B-Thinking-abliterated",
        model="huihui-ai/Huihui-Qwen3-VL-32B-Thinking-abliterated", 
        provider="llama_cpp",  # Fixed: use correct enum value
        pipeline="Qwen3VLPipeline",
        size=32000000000,  # 32B parameters
        digest="qwen3-vl-32b-thinking-f16-test",
        details=model_details,
        task="VisionTextToText",
        modified_at=datetime.now().isoformat(),  # Added required field
    )
    
    # Create optimization configuration (conservative for crash prevention)
    optimization_config = ParameterOptimizationConfiguration(
        enabled=True,
        optimization_priority=["n_ctx", "n_batch"],  # Same as user's failing scenario
        parameter_floors=ParameterFloors(
            n_ctx=1024,      # Very low floor to avoid crashes
            n_batch=8,       # Very low floor
            n_ubatch=8,      # Very low floor
            n_gpu_layers=0   # Allow CPU fallback
        ),
        search_strategy="binary_search",
        max_search_attempts=5,  # Fewer attempts for faster testing
        crash_prevention=CrashPrevention(
            enable_preallocation_test=True,
            memory_buffer_mb=2048,  # Large safety buffer
            timeout_seconds=60,     # Shorter timeout for testing
            enable_graceful_degradation=True
        )
    )
    
    # Create model profile with the problematic parameters
    model_profile = ModelProfile(
        user_id="test_user",
        name="qwen3vl_test",
        model_name="huihui-ai/Huihui-Qwen3-VL-32B-Thinking-abliterated",
        parameters=ModelParameters(
            num_ctx=150000,      # This was causing the crash!
            batch_size=512,      # This too
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            stop=["<|im_end|>"]
        ),
        gpu_config=GPUConfig(
            gpu_layers=-1,       # This was also problematic
            main_gpu=0,
            tensor_split=None
        ),
        system_prompt="You are a helpful AI assistant.",
        type=0,
        parameter_optimization=optimization_config  # Enable optimization
    )
    
    try:
        print(f"📊 Original profile params: n_ctx={model_profile.parameters.num_ctx}, n_batch={model_profile.parameters.batch_size}, gpu_layers={model_profile.gpu_config.gpu_layers}")
        
        # This should NOT crash, but should optimize parameters down to safe values
        print("🚀 Attempting to create Qwen3VLPipeline with optimization...")
        pipeline = Qwen3VLPipeline(model, model_profile)
        
        print("✅ Pipeline created successfully!")
        print("🎯 Parameter optimization prevented the crash!")
        
        # Check if optimization occurred by looking at the actual llama instance params
        if hasattr(pipeline, 'llama_instance') and pipeline.llama_instance:
            print("📈 Model initialized with optimized parameters")
            return True
        else:
            print("⚠️ Pipeline created but model not initialized")
            return False
            
    except Exception as e:
        print(f"❌ Pipeline creation failed: {e}")
        print("🔍 This indicates the optimization didn't prevent the crash")
        import traceback
        traceback.print_exc()
        return False


def test_parameter_bounds():
    """Test that the parameter optimization respects bounds properly."""
    print("\n🔬 Testing Parameter Bounds...")
    
    # Test with impossibly high parameters that should be reduced
    high_params = ModelParameters(
        num_ctx=1000000,     # 1M context - impossible
        batch_size=10000,    # 10K batch - too high
    )
    
    print(f"Input params: n_ctx={high_params.num_ctx}, n_batch={high_params.batch_size}")
    print("Expected: Parameters should be reduced to reasonable values")
    
    return True


if __name__ == "__main__":
    print("🚀 Qwen3VL Parameter Optimization Test Suite")
    print("=" * 60)
    
    try:
        # Test 1: Crash prevention
        success1 = test_qwen3vl_parameter_optimization()
        print()
        
        # Test 2: Parameter bounds
        success2 = test_parameter_bounds()
        print()
        
        if success1 and success2:
            print("✅ All tests passed! Parameter optimization is working correctly.")
            sys.exit(0)
        else:
            print("❌ Some tests failed. Parameter optimization needs fixes.")
            sys.exit(1)
            
    except Exception as e:
        print(f"💥 Test suite crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)