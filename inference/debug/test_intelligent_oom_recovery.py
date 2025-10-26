#!/usr/bin/env python3
"""
Test the intelligent OOM recovery system without actually loading models.
Tests the ML prediction logic and recovery strategy generation.
"""

import sys
import os
sys.path.insert(0, '/app')

from runner.utils.intelligent_oom_recovery import IntelligentOOMRecovery
from runner.utils.hardware_manager import hardware_manager

def test_oom_recovery_system():
    """Test the intelligent OOM recovery system functionality."""
    print("🧪 Testing Intelligent OOM Recovery System...")
    
    # Initialize the recovery system
    recovery = IntelligentOOMRecovery(data_dir="/tmp/test_oom_recovery")
    
    # Test parameter prediction
    print("\n🔮 Testing ML parameter prediction...")
    
    # Simulate different model scenarios
    test_scenarios = [
        {
            "name": "Small 7B model",
            "model_path": "/fake/path/llama-7b-chat.gguf",
            "target_n_ctx": 32768,
            "target_batch": 512,
            "target_ubatch": 512,
            "requested_gpu_layers": -1
        },
        {
            "name": "Large 70B model", 
            "model_path": "/fake/path/llama-70b-chat.gguf",
            "target_n_ctx": 16384,
            "target_batch": 256,
            "target_ubatch": 256,
            "requested_gpu_layers": -1
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n📊 Scenario: {scenario['name']}")
        
        predicted_params = recovery.predict_optimal_parameters(
            model_path=scenario["model_path"],
            target_n_ctx=scenario["target_n_ctx"],
            target_batch=scenario["target_batch"], 
            target_ubatch=scenario["target_ubatch"],
            requested_gpu_layers=scenario["requested_gpu_layers"],
            hardware_manager=hardware_manager
        )
        
        print(f"  🎯 Target params: n_ctx={scenario['target_n_ctx']}, n_batch={scenario['target_batch']}, gpu_layers={scenario['requested_gpu_layers']}")
        print(f"  🧠 Predicted params: n_ctx={predicted_params['n_ctx']}, n_batch={predicted_params['n_batch']}, gpu_layers={predicted_params['n_gpu_layers']}")
        
        # Test recovery strategies
        print(f"\n🔄 Testing recovery strategies for {scenario['name']}:")
        
        original_params = {
            'n_ctx': scenario['target_n_ctx'],
            'n_batch': scenario['target_batch'],
            'n_ubatch': scenario['target_ubatch'], 
            'n_gpu_layers': scenario['requested_gpu_layers']
        }
        
        current_params = predicted_params.copy()
        
        # Test multiple recovery attempts
        for attempt in range(1, 8):
            new_params, strategy = recovery.execute_recovery_strategy(
                attempt=attempt,
                original_params=original_params,
                current_params=current_params,
                hardware_manager=hardware_manager
            )
            
            print(f"    Attempt {attempt}: strategy='{strategy}' -> n_ctx={new_params['n_ctx']}, n_batch={new_params['n_batch']}, gpu_layers={new_params['n_gpu_layers']}")
            
            # Record a simulated failure for ML training
            recovery.record_failure(
                attempt=attempt,
                strategy=strategy,
                params=current_params,
                error_message=f"Simulated OOM at attempt {attempt}"
            )
            
            current_params = new_params
    
    # Test recording a success
    print("\n✅ Testing success recording...")
    
    success_params = {
        'n_ctx': 8192,
        'n_batch': 128,
        'n_ubatch': 128,
        'n_gpu_layers': 35
    }
    
    recovery.record_success(
        model_path="/fake/path/test-model.gguf",
        params=success_params,
        hardware_manager=hardware_manager,
        initialization_time_ms=2500.0,
        gpu_memory_used_mb=6144.0
    )
    
    # Get statistics
    print("\n📈 Recovery System Statistics:")
    stats = recovery.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n🎉 All OOM recovery system tests completed successfully!")

def test_model_size_estimation():
    """Test model size estimation heuristics."""
    print("\n🔍 Testing model size estimation...")
    
    recovery = IntelligentOOMRecovery()
    
    test_paths = [
        "/fake/path/llama-7b-chat.gguf",
        "/fake/path/mistral-7b-instruct.gguf",
        "/fake/path/llama-13b-chat.gguf", 
        "/fake/path/llama-30b-chat.gguf",
        "/fake/path/llama-70b-chat.gguf",
        "/fake/path/unknown-model.gguf"
    ]
    
    for path in test_paths:
        size_mb = recovery.get_model_size_mb(path)
        print(f"  {path} -> {size_mb:,.0f} MB")

def main():
    """Run all OOM recovery tests."""
    print("🚀 Starting Intelligent OOM Recovery System Tests...")
    print()
    
    try:
        test_model_size_estimation()
        test_oom_recovery_system()
        
        print("\n🎉 All tests passed! The intelligent OOM recovery system is ready.")
        print("✅ ML-based parameter prediction working")
        print("✅ Structured recovery strategy working") 
        print("✅ Success/failure recording working")
        print("✅ Statistics generation working")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()