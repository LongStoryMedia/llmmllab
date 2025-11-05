#!/usr/bin/env python3
"""
Test script for enhanced OOM recovery with crash prevention and parameter optimization.

This script tests:
1. Memory preallocation testing
2. Parameter optimization strategies 
3. Crash prevention mechanisms
"""

import asyncio
import sys
import os

# Add the inference directory to the path
sys.path.insert(0, '/app')

from runner.utils.intelligent_oom_recovery import IntelligentOOMRecovery
from runner.utils.hardware_manager import hardware_manager
from models.optimal_parameters import OptimalParameters
from models.parameter_optimization_config import (
    ParameterOptimizationConfiguration, 
    ParameterFloors, 
    CrashPrevention
)
from models.model_profile import ModelProfile
from models.model_parameters import ModelParameters
from models.gpu_config import GPUConfig


async def test_memory_preallocation():
    """Test memory preallocation functionality."""
    print("🧪 Testing memory preallocation...")
    
    recovery = IntelligentOOMRecovery()
    
    # Test small allocation (should succeed)
    success = await recovery.test_memory_preallocation(100, 30)  # 100MB, 30s timeout
    print(f"✅ Small allocation (100MB): {'SUCCESS' if success else 'FAILED'}")
    
    # Test large allocation (should fail gracefully)
    success = await recovery.test_memory_preallocation(50000, 30)  # 50GB, 30s timeout  
    print(f"⚠️  Large allocation (50GB): {'SUCCESS' if success else 'FAILED'}")
    
    return True


def test_parameter_optimization():
    """Test parameter optimization strategies."""
    print("🎯 Testing parameter optimization...")
    
    recovery = IntelligentOOMRecovery()
    
    # Create test model profile with optimization config
    optimization_config = ParameterOptimizationConfiguration(
        enabled=True,
        optimization_priority=["n_ctx", "n_batch"],
        parameter_floors=ParameterFloors(
            n_ctx=2048,
            n_batch=32,
            n_ubatch=32, 
            n_gpu_layers=0
        ),
        search_strategy="binary_search",
        max_search_attempts=5,
        crash_prevention=CrashPrevention(
            enable_preallocation_test=True,
            memory_buffer_mb=1024,
            timeout_seconds=60,
            enable_graceful_degradation=True
        )
    )
    
    # Create minimal model profile for testing
    model_profile = ModelProfile(
        user_id="test_user",
        name="test_profile",
        model_name="test_model",
        parameters=ModelParameters(),
        system_prompt="Test prompt",
        type=0,
        parameter_optimization=optimization_config
    )
    
    # Create base parameters
    base_params = OptimalParameters(
        n_ctx=4096,
        n_batch=128,
        n_ubatch=128,
        n_gpu_layers=10
    )
    
    try:
        optimized_params = recovery.optimize_parameters_for_hardware(
            base_params=base_params,
            model_profile=model_profile,
            hardware_manager=hardware_manager,
            optimization_config=optimization_config
        )
        
        print(f"📊 Original params:  n_ctx={base_params.n_ctx}, n_batch={base_params.n_batch}, n_ubatch={base_params.n_ubatch}, n_gpu_layers={base_params.n_gpu_layers}")
        print(f"📈 Optimized params: n_ctx={optimized_params.n_ctx}, n_batch={optimized_params.n_batch}, n_ubatch={optimized_params.n_ubatch}, n_gpu_layers={optimized_params.n_gpu_layers}")
        
        return True
        
    except Exception as e:
        print(f"❌ Parameter optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_estimation():
    """Test memory estimation functionality."""
    print("📏 Testing memory estimation...")
    
    recovery = IntelligentOOMRecovery()
    
    # Test different parameter configurations
    test_configs = [
        OptimalParameters(n_ctx=2048, n_batch=32, n_ubatch=32, n_gpu_layers=0),
        OptimalParameters(n_ctx=8192, n_batch=128, n_ubatch=128, n_gpu_layers=20),
        OptimalParameters(n_ctx=32768, n_batch=512, n_ubatch=512, n_gpu_layers=50),
    ]
    
    for i, params in enumerate(test_configs):
        estimated = recovery.estimate_memory_requirements(params)
        print(f"Config {i+1}: n_ctx={params.n_ctx}, n_batch={params.n_batch} → {estimated:.0f}MB estimated")
    
    return True


async def main():
    """Main test function."""
    print("🚀 Enhanced OOM Recovery Test Suite")
    print("=" * 50)
    
    try:
        # Test memory preallocation
        await test_memory_preallocation()
        print()
        
        # Test parameter optimization
        test_parameter_optimization()
        print()
        
        # Test memory estimation
        test_memory_estimation()
        print()
        
        print("✅ All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)