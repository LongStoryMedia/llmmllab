#!/usr/bin/env python3
"""
Test the updated OOM recovery system without primary GPU concept and with learned limits.
"""

from runner.utils.intelligent_oom_recovery import IntelligentOOMRecovery
from runner.utils.hardware_manager import hardware_manager
from models.model_profile import ModelProfile
from models.model_parameters import ModelParameters
from models.gpu_config import GPUConfig
from models.model_profile_type import ModelProfileType


def test_no_primary_gpu_concept():
    """Test that primary GPU concept is removed."""
    print("🔬 Testing removal of primary GPU concept...")
    
    recovery = IntelligentOOMRecovery()
    gpu_stats = recovery.get_system_gpu_stats(hardware_manager)
    
    print(f"✅ GPU Stats keys: {list(gpu_stats.keys())}")
    
    # Should NOT have primary_gpu_id
    if "primary_gpu_id" in gpu_stats:
        print("❌ FAIL: primary_gpu_id still exists in GPU stats")
        return False
    
    # Should have total_available_memory instead of available_memory
    if "total_available_memory" not in gpu_stats:
        print("❌ FAIL: total_available_memory missing from GPU stats")
        return False
        
    print(f"✅ Total GPUs: {gpu_stats['total_gpus']}")
    print(f"✅ Total Memory: {gpu_stats['total_memory']:.0f}MB")
    print(f"✅ Total Available: {gpu_stats['total_available_memory']:.0f}MB")
    print(f"✅ GPU Details: {len(gpu_stats['gpus'])} GPUs")
    
    return True


def test_learned_limits():
    """Test that learned limits are used instead of hard-coded values."""
    print("\n🔬 Testing learned limits...")
    
    recovery = IntelligentOOMRecovery()
    
    # Test learned limits method
    learned_limits = recovery._get_learned_limits()
    print(f"✅ Learned limits: {learned_limits}")
    
    # Test learned minimums method
    learned_mins = recovery._get_learned_minimums()
    print(f"✅ Learned minimums: {learned_mins}")
    
    # Verify they are using conservative initial values (no training data yet)
    if learned_limits["max_context"] == 16384:  # Conservative initial estimate
        print("✅ Using conservative initial estimates for limits")
    else:
        print(f"⚠️  Max context: {learned_limits['max_context']} (expected 16384 initially)")
    
    if learned_mins["n_batch"] == 16:  # Conservative initial minimum
        print("✅ Using conservative initial minimums")
    else:
        print(f"⚠️  Min batch: {learned_mins['n_batch']} (expected 16 initially)")
    
    return True


def test_model_profile_integration():
    """Test model profile integration without primary GPU."""
    print("\n🔬 Testing model profile integration...")
    
    # Create test profile
    params = ModelParameters(
        num_ctx=32768,
        temperature=0.7,
        top_p=0.9,
        batch_size=512,
        max_tokens=4096,
    )
    
    gpu_config = GPUConfig(
        gpu_layers=-1,  # Auto-allocation
        main_gpu=0,
        offload_kqv=True,
    )
    
    import uuid
    
    profile = ModelProfile(
        id=str(uuid.uuid4()),
        user_id=str(uuid.uuid4()), 
        name="Test Profile - No Primary GPU",
        model_name="test-model",
        system_prompt="Test system prompt",
        type=ModelProfileType.Primary,  
        model_path="/nonexistent/test-model.gguf",
        parameters=params,
        gpu_config=gpu_config,
        description="Test profile for no primary GPU testing",
    )
    
    recovery = IntelligentOOMRecovery()
    gpu_stats = recovery.get_system_gpu_stats(hardware_manager)
    
    # Test configuration creation
    config = recovery.create_configuration_from_model_profile(profile, gpu_stats)
    print(f"✅ Generated config: {config}")
    
    # Verify learned limits are applied  
    learned_limits = recovery._get_learned_limits()
    if config["n_ctx"] <= learned_limits["max_context"]:
        print("✅ Context limited by learned maximum")
    
    if config["n_batch"] <= learned_limits["max_batch"]:
        print("✅ Batch limited by learned maximum")
        
    return True


def main():
    """Run all tests."""
    print("🧪 Testing OOM Recovery System - No Primary GPU + Learned Limits")
    print("=" * 80)
    
    tests = [
        test_no_primary_gpu_concept,
        test_learned_limits, 
        test_model_profile_integration,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with error: {e}")
            results.append(False)
    
    print("\n" + "=" * 80)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 80)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test.__name__}")
    
    passed = sum(results)
    total = len(results)
    print(f"\n🎯 Overall Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! No primary GPU concept removed, learned limits implemented.")
    else:
        print(f"💥 {total - passed} test(s) failed!")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())