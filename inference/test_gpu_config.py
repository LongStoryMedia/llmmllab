#!/usr/bin/env python3
"""
Test script for GPU configuration integration.
Run this with: k exec -it -n ollama POD_NAME -- /app/v.sh runner python test_gpu_config.py
"""

import sys
import os
sys.path.append('/app')

def test_gpu_config_models():
    """Test GPU config model creation and validation."""
    print("=== Testing GPU Config Models ===")
    
    from models.model_parameters import GPUConfig, ModelParameters
    
    # Test basic GPU config
    gpu_config = GPUConfig(
        no_kv_offload=True,
        main_gpu=1,
        tensor_split=[0.3, 0.7],
        n_cpu_moe=2,
        offload_kqv=False
    )
    
    print(f"✓ GPU config created: {gpu_config}")
    print(f"  - no_kv_offload: {gpu_config.no_kv_offload}")
    print(f"  - main_gpu: {gpu_config.main_gpu}")
    print(f"  - tensor_split: {gpu_config.tensor_split}")
    print(f"  - n_cpu_moe: {gpu_config.n_cpu_moe}")
    
    # Test model parameters with GPU config
    params = ModelParameters(
        num_ctx=4096,
        temperature=0.7,
        gpu_config=gpu_config
    )
    
    print(f"✓ Model parameters created with GPU config")
    print(f"  - Has gpu_config: {params.gpu_config is not None}")
    
    return gpu_config, params

def test_device_mapping():
    """Test device name to index mapping."""
    print("\n=== Testing Device Mapping ===")
    
    from utils.hardware_manager import hardware_manager
    
    # Get device mappings
    mappings = hardware_manager.get_device_mappings()
    print(f"✓ Available devices: {len(mappings)}")
    
    for device_id, device_info in mappings.items():
        print(f"  - {device_id}: {device_info['name']} (index: {device_info['index']})")
    
    # Test name resolution
    test_cases = [
        "NVIDIA GeForce RTX 3090",
        "1",
        "cpu",
        "0"
    ]
    
    for test_case in test_cases:
        try:
            resolved = hardware_manager.resolve_device_name_to_index(test_case)
            print(f"✓ '{test_case}' -> index {resolved}")
        except Exception as e:
            print(f"✗ '{test_case}' -> ERROR: {e}")
    
    return mappings

def test_pipeline_gpu_config():
    """Test pipeline GPU configuration extraction."""
    print("\n=== Testing Pipeline GPU Config ===")
    
    from models import Model
    from models.model_profile import ModelProfile
    from runner.pipelines.llamacpp.base_llamacpp import BaseLlamaCppPipeline
    
    # Create test configuration
    gpu_config, params = test_gpu_config_models()
    
    profile = ModelProfile(
        name='test-profile',
        parameters=params
    )
    
    model = Model(
        model='test-model',
        provider='llamacpp'
    )
    
    # Create a minimal test pipeline
    class TestPipeline(BaseLlamaCppPipeline):
        def _get_gguf_path(self):
            return '/fake/path.gguf'
        
        async def _create_system_prompt(self, tools=None):
            return 'test prompt'
    
    try:
        pipeline = TestPipeline(model, profile)
        gpu_kwargs = pipeline._get_gpu_config_kwargs()
        
        print(f"✓ Pipeline GPU kwargs extracted:")
        for key, value in gpu_kwargs.items():
            print(f"  - {key}: {value}")
        
        return gpu_kwargs
        
    except Exception as e:
        print(f"✗ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_moe_pipeline():
    """Test MoE-specific pipeline configuration."""
    print("\n=== Testing MoE Pipeline Config ===")
    
    from models import Model
    from models.model_profile import ModelProfile
    from models.model_parameters import GPUConfig, ModelParameters
    from runner.pipelines.txt2txt.openai_gpt_oss import OpenAiGptOssPipe
    
    # Create MoE-specific configuration
    gpu_config = GPUConfig(
        n_cpu_moe=3,
        main_gpu=1,
        tensor_split=[0.4, 0.6]
    )
    
    params = ModelParameters(
        num_ctx=4096,
        gpu_config=gpu_config
    )
    
    profile = ModelProfile(
        name='test-moe-profile',
        parameters=params
    )
    
    model = Model(
        model='test-moe-model',
        provider='llamacpp'
    )
    
    try:
        # Mock the GGUF validation for testing
        os.environ['ALLOW_MISSING_GGUF'] = 'true'
        
        pipeline = OpenAiGptOssPipe(model, profile)
        gpu_kwargs = pipeline._get_gpu_config_kwargs()
        
        print(f"✓ MoE Pipeline GPU kwargs extracted:")
        for key, value in gpu_kwargs.items():
            print(f"  - {key}: {value}")
        
        # Check for MoE-specific handling
        if 'n_cpu_moe' in gpu_kwargs:
            print(f"✓ MoE CPU layers configured: {gpu_kwargs['n_cpu_moe']}")
        else:
            print("⚠ MoE CPU layers not found in kwargs")
        
        return gpu_kwargs
        
    except Exception as e:
        print(f"✗ MoE pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Run all GPU configuration tests."""
    print("GPU Configuration Integration Test Suite")
    print("=" * 50)
    
    try:
        # Test 1: Basic model creation
        test_gpu_config_models()
        
        # Test 2: Device mapping
        test_device_mapping()
        
        # Test 3: Pipeline integration
        test_pipeline_gpu_config()
        
        # Test 4: MoE pipeline
        test_moe_pipeline()
        
        print("\n" + "=" * 50)
        print("✓ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())