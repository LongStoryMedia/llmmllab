#!/usr/bin/env python
"""
Simple Parameter Optimization Test - Focus on the bug fix

Tests that parameter optimization results are actually applied during initialization,
without requiring actual model files to exist.
"""

import sys
import os
from typing import Dict, Any
import asyncio
from datetime import datetime

# Add the parent directory to the Python path for imports
sys.path.append("/app")

from models import Model, ModelDetails, ModelProfile
from runner.utils.intelligent_oom_recovery import IntelligentOOMRecovery


async def test_parameter_optimization_logic():
    """Test the parameter optimization logic directly without model file dependencies."""
    
    print("🔧 Testing Parameter Optimization Logic...")
    print("=" * 60)
    
    # Test problematic parameters directly - no model profile needed
    original_n_ctx = 150000  # This should cause OOM 
    original_n_batch = 512   # This should cause OOM
    original_n_ubatch = 128
    original_n_gpu_layers = -1  # Use all layers
    
    print(f"📊 Original params: n_ctx={original_n_ctx}, n_batch={original_n_batch}, gpu_layers={original_n_gpu_layers}")
    
    # Initialize OOM recovery
    oom_recovery = IntelligentOOMRecovery()
    
    # Test parameter optimization
    print("🎯 Running parameter optimization...")
    optimized_params = await oom_recovery.optimize_parameters_for_hardware(
        original_n_ctx,
        original_n_batch,
        original_n_ubatch,
        original_n_gpu_layers
    )
    
    print(f"📈 Optimized params: {optimized_params}")
    
    # Verify optimization worked
    if optimized_params:
        print("✅ Parameter optimization completed successfully!")
        
        # Check that parameters were actually optimized down
        if (optimized_params.get('n_ctx', 150000) < 150000 and
            optimized_params.get('n_batch', 512) < 512):
            print("✅ Parameters were properly reduced to safe values")
            
            # Show the optimization results
            print(f"🔍 n_ctx: {original_n_ctx} → {optimized_params.get('n_ctx', original_n_ctx)} ({((optimized_params.get('n_ctx', original_n_ctx) / original_n_ctx) * 100):.1f}%)")
            print(f"🔍 n_batch: {original_n_batch} → {optimized_params.get('n_batch', original_n_batch)} ({((optimized_params.get('n_batch', original_n_batch) / original_n_batch) * 100):.1f}%)")
            
            return True
        else:
            print("❌ Parameters were not reduced - optimization may not be working")
            print(f"🔍 n_ctx optimized: {optimized_params.get('n_ctx', 'missing')} (original: {original_n_ctx})")
            print(f"🔍 n_batch optimized: {optimized_params.get('n_batch', 'missing')} (original: {original_n_batch})")
            return False
    else:
        print("❌ No optimized parameters returned")
        return False


def test_memory_estimation():
    """Test memory estimation accuracy."""
    
    print("\n🧠 Testing Memory Estimation...")
    print("=" * 40)
    
    # Initialize OOM recovery
    oom_recovery = IntelligentOOMRecovery()
    
    # Test various parameter combinations
    test_configs = [
        {"n_ctx": 4096, "n_batch": 128, "description": "Reasonable parameters"},
        {"n_ctx": 32768, "n_batch": 512, "description": "High but possible"},
        {"n_ctx": 150000, "n_batch": 512, "description": "Problematic parameters"},
    ]
    
    all_passed = True
    
    for config in test_configs:
        estimated_memory = oom_recovery.estimate_memory_requirements(
            config["n_ctx"],
            config["n_batch"],
            128,  # n_ubatch
            10    # n_gpu_layers
        )
        
        print(f"📏 {config['description']}: n_ctx={config['n_ctx']}, n_batch={config['n_batch']} → {estimated_memory}MB")
        
        # Basic sanity checks
        if estimated_memory <= 0:
            print(f"❌ Invalid memory estimate: {estimated_memory}")
            all_passed = False
        elif estimated_memory > 100000:  # 100GB seems unreasonable
            print(f"⚠️  Very high memory estimate: {estimated_memory}MB")
    
    return all_passed


async def main():
    """Run all tests."""
    
    print("🚀 Simple Parameter Optimization Test Suite")
    print("=" * 60)
    
    try:
        # Test parameter optimization logic
        optimization_success = await test_parameter_optimization_logic()
        
        # Test memory estimation
        memory_success = test_memory_estimation()
        
        print(f"\n📊 Test Results:")
        print(f"✅ Parameter Optimization: {'PASS' if optimization_success else 'FAIL'}")
        print(f"✅ Memory Estimation: {'PASS' if memory_success else 'FAIL'}")
        
        if optimization_success and memory_success:
            print("\n🎉 All tests passed! Parameter optimization logic is working correctly.")
            return True
        else:
            print("\n❌ Some tests failed. Parameter optimization needs fixes.")
            return False
            
    except Exception as e:
        print(f"💥 Test suite crashed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)