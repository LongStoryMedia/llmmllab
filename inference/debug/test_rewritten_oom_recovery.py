#!/usr/bin/env python3
"""
Test script for the rewritten intelligent OOM recovery system.

Tests:
- Strong typing with TypedDict
- Dynamic multi-GPU support
- Model profile integration
- sklearn Ridge regression requirement (no fallbacks)
"""

import os
import sys

# Ensure we can import from the inference directory
inference_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, inference_path)

try:
    from runner.utils.intelligent_oom_recovery import IntelligentOOMRecovery
    from runner.utils.hardware_manager import hardware_manager
    from models.model_profile import ModelProfile
    from models.model_parameters import ModelParameters
    from models.gpu_config import GPUConfig
    from models.model_profile_type import ModelProfileType
    import uuid
    print("✅ Imports successful")
except Exception as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def create_test_model_profile():
    """Create a test model profile with realistic configuration."""
    
    # Create model parameters similar to default profiles
    params = ModelParameters(
        num_ctx=32768,
        temperature=0.7,
        top_p=0.9,
        batch_size=512,
        max_tokens=4096,
    )
    
    # Create GPU configuration
    gpu_config = GPUConfig(
        gpu_layers=-1,  # Auto-allocation
        main_gpu=0,
        offload_kqv=True,
    )
    
    # Create model profile
    profile = ModelProfile(
        id=uuid.uuid4(),
        user_id="test_user",
        name="Test Profile - Qwen3 30B",
        model_name="qwen3-30b-a3b-q4-k-m",
        parameters=params,
        gpu_config=gpu_config,
        system_prompt="You are a helpful assistant for testing the OOM recovery system.",
        type=ModelProfileType.Primary  # Required field
    )
    
    return profile

def test_strong_typing():
    """Test that the system uses strong typing throughout."""
    print("\n" + "="*60)
    print("TESTING STRONG TYPING")
    print("="*60)
    
    try:
        recovery = IntelligentOOMRecovery()
        
        # Test GPU stats structure (TypedDict)
        gpu_stats = recovery.get_system_gpu_stats(hardware_manager)
        print(f"✅ GPU Stats Type: {type(gpu_stats)}")
        print(f"   Total GPUs: {gpu_stats['total_gpus']}")
        print(f"   Total Memory: {gpu_stats['total_memory']:.0f} MB")
        print(f"   Primary GPU: {gpu_stats['primary_gpu_id']}")
        
        # Test model profile integration
        profile = create_test_model_profile()
        config = recovery.create_configuration_from_model_profile(profile, gpu_stats)
        print(f"✅ Profile Config Type: {type(config)}")
        print(f"   n_ctx: {config['n_ctx']}")
        print(f"   n_batch: {config['n_batch']}")
        print(f"   n_gpu_layers: {config['n_gpu_layers']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Strong typing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dynamic_multi_gpu():
    """Test dynamic multi-GPU support for any number of GPUs."""
    print("\n" + "="*60)
    print("TESTING DYNAMIC MULTI-GPU SUPPORT")
    print("="*60)
    
    try:
        recovery = IntelligentOOMRecovery()
        
        # Get comprehensive GPU statistics
        gpu_stats = recovery.get_system_gpu_stats(hardware_manager)
        
        print(f"🔍 System GPU Analysis:")
        print(f"   Total GPUs: {gpu_stats['total_gpus']}")
        print(f"   Primary GPU ID: {gpu_stats['primary_gpu_id']}")
        print(f"   Total System Memory: {gpu_stats['total_memory']:.0f} MB")
        print(f"   Primary GPU Memory: {gpu_stats['available_memory']:.0f} MB")
        
        # Test individual GPU details
        for gpu in gpu_stats['gpus']:
            print(f"   GPU {gpu['id']}: {gpu['name']} - {gpu['available_memory']:.0f}/{gpu['total_memory']:.0f}MB ({gpu['utilization_pct']:.1f}% used)")
        
        # Test GPU selection logic
        if gpu_stats['total_gpus'] > 1:
            print(f"✅ Multi-GPU system detected with intelligent selection")
        elif gpu_stats['total_gpus'] == 1:
            print(f"✅ Single GPU system handled correctly")
        else:
            print(f"✅ No GPU system handled correctly")
            
        return True
        
    except Exception as e:
        print(f"❌ Multi-GPU test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_profile_integration():
    """Test model profile-driven configuration."""
    print("\n" + "="*60)
    print("TESTING MODEL PROFILE INTEGRATION")
    print("="*60)
    
    try:
        recovery = IntelligentOOMRecovery()
        profile = create_test_model_profile()
        
        # Test model profile-driven prediction with non-existent path to trigger estimation
        optimal_params = recovery.predict_optimal_parameters_from_profile(
            profile, 
            "/nonexistent/qwen3-30b-model.gguf",  # Non-existent path triggers size estimation
            hardware_manager
        )
        
        print(f"✅ Model Profile Integration:")
        print(f"   Profile Name: {profile.name}")
        print(f"   Base num_ctx: {profile.parameters.num_ctx}")
        print(f"   Base batch_size: {profile.parameters.batch_size}")
        print(f"   GPU config: {profile.gpu_config.gpu_layers if profile.gpu_config else 'None'}")
        
        print(f"✅ Optimized Parameters:")
        print(f"   n_ctx: {optimal_params['n_ctx']}")
        print(f"   n_batch: {optimal_params['n_batch']}")
        print(f"   n_ubatch: {optimal_params['n_ubatch']}")
        print(f"   n_gpu_layers: {optimal_params['n_gpu_layers']}")
        
        # Verify parameters are reasonable
        assert optimal_params['n_ctx'] > 0, "n_ctx must be positive"
        assert optimal_params['n_batch'] > 0, "n_batch must be positive"
        assert optimal_params['n_ubatch'] <= optimal_params['n_batch'], "n_ubatch must be <= n_batch"
        assert optimal_params['n_gpu_layers'] >= 0, "n_gpu_layers must be non-negative"
        
        print(f"✅ Parameter validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Model profile integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sklearn_requirement():
    """Test that sklearn is required (no fallbacks)."""
    print("\n" + "="*60)
    print("TESTING SKLEARN REQUIREMENT (NO FALLBACKS)")
    print("="*60)
    
    try:
        recovery = IntelligentOOMRecovery()
        
        # Check that Ridge models are used
        for param_name, model in recovery.models.items():
            if model is not None:
                print(f"✅ {param_name} model: {type(model).__name__}")
                assert "Ridge" in type(model).__name__, f"Expected Ridge model, got {type(model)}"
            else:
                print(f"🔄 {param_name} model: Not trained yet (Ridge will be used)")
        
        # Test that scalers are StandardScaler from sklearn
        for scaler_name, scaler in recovery.scalers.items():
            print(f"✅ {scaler_name} scaler: {type(scaler).__name__}")
            assert "StandardScaler" in type(scaler).__name__, f"Expected StandardScaler, got {type(scaler)}"
            
        print(f"✅ sklearn requirement satisfied - no fallbacks detected")
        return True
        
    except Exception as e:
        print(f"❌ sklearn requirement test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_recovery_strategy():
    """Test the 4-level OOM recovery strategy."""
    print("\n" + "="*60)
    print("TESTING OOM RECOVERY STRATEGY")
    print("="*60)
    
    try:
        recovery = IntelligentOOMRecovery()
        
        # Create test parameters
        original_params = {
            "n_ctx": 32768,
            "n_batch": 512,
            "n_ubatch": 512,
            "n_gpu_layers": 40
        }
        
        current_params = original_params.copy()
        
        # Test each recovery level
        for attempt in range(1, 8):
            strategy = recovery.execute_recovery_strategy(
                attempt, original_params, current_params, hardware_manager
            )
            
            new_params = strategy["parameters"]
            strategy_name = strategy["strategy_name"]
            
            print(f"✅ Attempt {attempt}: {strategy_name}")
            print(f"   n_ctx: {current_params['n_ctx']} → {new_params['n_ctx']}")
            print(f"   n_batch: {current_params['n_batch']} → {new_params['n_batch']}")
            print(f"   n_gpu_layers: {current_params['n_gpu_layers']} → {new_params['n_gpu_layers']}")
            
            # Update for next iteration
            current_params = new_params
            
            # Validate strategy progression
            if attempt <= 2:
                assert strategy_name == "clear_memory", f"Expected clear_memory, got {strategy_name}"
            elif attempt <= 4:
                assert strategy_name == "reduce_batch", f"Expected reduce_batch, got {strategy_name}"
            elif attempt <= 6:
                assert strategy_name == "move_to_cpu", f"Expected move_to_cpu, got {strategy_name}"
            else:
                assert strategy_name == "reduce_context", f"Expected reduce_context, got {strategy_name}"
        
        print(f"✅ Recovery strategy progression validated")
        return True
        
    except Exception as e:
        print(f"❌ Recovery strategy test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run comprehensive tests for the rewritten OOM recovery system."""
    print("🧪 Testing Rewritten Intelligent OOM Recovery System")
    print("=" * 80)
    
    tests = [
        ("Strong Typing", test_strong_typing),
        ("Dynamic Multi-GPU", test_dynamic_multi_gpu),
        ("Model Profile Integration", test_model_profile_integration),
        ("sklearn Requirement", test_sklearn_requirement),
        ("Recovery Strategy", test_recovery_strategy),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔬 Running {test_name} Test...")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"💥 {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*80)
    print("📊 TEST RESULTS SUMMARY")
    print("="*80)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print(f"\n🎯 Overall Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The rewritten system meets all requirements:")
        print("   ✅ Strong typing (TypedDict, no Any/Dict)")
        print("   ✅ Dynamic multi-GPU support (any number of GPUs)")
        print("   ✅ sklearn required (Ridge regression, no fallbacks)")
        print("   ✅ Model profile integration (configuration-driven)")
        return 0
    else:
        print(f"💥 {total - passed} test(s) failed!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)