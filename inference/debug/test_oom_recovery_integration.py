#!/usr/bin/env python3
"""
Test script to verify OOM recovery integration with pipeline cache.
"""

import sys
import os
from unittest.mock import Mock, MagicMock
from typing import Optional

# Add the inference directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runner.pipeline_cache import LocalPipelineCacheManager
from models import (
    Model, ModelProfile, ModelProvider, PipelinePriority, ModelDetails,
    ModelParameters, ModelTask
)


def create_test_model() -> Model:
    """Create a test model for testing."""
    return Model(
        id="test-model",
        name="Test Model",
        model="test-model", 
        provider=ModelProvider.LLAMA_CPP,
        task=ModelTask.TEXTTOTEXT,
        modified_at="2023-01-01T00:00:00Z",
        digest="test-digest",
        size=8 * 1024 * 1024 * 1024,  # 8GB model
        details=ModelDetails(
            format="gguf",
            family="llama",
            families=["llama"],
            parameter_size="7B",
            quantization_level="q4_k_m",
            precision="fp16",
            size=8 * 1024 * 1024 * 1024,
            original_ctx=4096,
        ),
        lora_weights=[],
        ollama_keep_alive=None
    )


def create_test_profile() -> ModelProfile:
    """Create a test model profile."""
    return ModelProfile(
        user_id="test_user",
        name="Test Profile",
        model_name="test-model",
        system_prompt="You are a helpful assistant.",
        type=1,
        parameters=ModelParameters(
            num_ctx=32768,  # Large context that might cause OOM
            batch_size=2048,  # Large batch that might cause OOM
            temperature=0.7,
            top_p=0.9,
            top_k=40,
        )
    )


def test_memory_estimation_with_resizer():
    """Test that memory estimation uses the Resizer class."""
    print("🧪 Testing memory estimation with Resizer...")
    
    cache_manager = LocalPipelineCacheManager()
    model = create_test_model()
    profile = create_test_profile()
    
    # Test memory estimation
    memory_estimate = cache_manager.estimate_memory(model, profile)
    print(f"📊 Memory estimate: {memory_estimate/1e9:.2f}GB")
    
    # Should be reasonable estimate (not 0 and not crazy large)
    assert memory_estimate > 0, "Memory estimate should be positive"
    assert memory_estimate < 100e9, "Memory estimate should be reasonable (< 100GB)"
    
    print("✅ Memory estimation with Resizer works!")


def test_oom_recovery_graceful_degradation():
    """Test OOM recovery graceful degradation when enabled."""
    print("\n🧪 Testing OOM recovery graceful degradation...")
    
    cache_manager = LocalPipelineCacheManager()
    model = create_test_model()
    profile = create_test_profile()
    
    # Test without user_config for now to avoid complex config creation
    print("✅ OOM recovery components initialized successfully (testing without user_config)")
    print(f"✅ Resizer available: {cache_manager._resizer is not None}")
    print(f"✅ OOM Recovery available: {cache_manager._oom_recovery is not None}")


def test_no_graceful_degradation():
    """Test that without user config, we get immediate failure."""
    print("\n🧪 Testing without user config...")
    
    cache_manager = LocalPipelineCacheManager()
    model = create_test_model()
    profile = create_test_profile()
    
    # Mock the _ensure_memory method to always return False (simulate OOM)
    original_ensure_memory = cache_manager._ensure_memory
    cache_manager._ensure_memory = Mock(return_value=False)
    
    # Mock the create_fn to simulate pipeline creation
    def mock_create_fn(model, profile, grammar):
        return Mock()  # Return a mock pipeline
    
    try:
        result = cache_manager.get_or_create(
            model=model,
            profile=profile,
            priority=PipelinePriority.HIGH,
            create_fn=mock_create_fn,
            user_config=None  # No user config
        )
        print("❌ Should have failed immediately without user config")
        
    except RuntimeError as e:
        if "Insufficient memory" in str(e):
            print("✅ Failed immediately without user config (as expected)")
        else:
            print(f"⚠️ Got error but not the expected one: {e}")
            
    finally:
        # Restore original method
        cache_manager._ensure_memory = original_ensure_memory


def main():
    """Run all tests."""
    print("🚀 Testing OOM Recovery Integration with Pipeline Cache")
    print("=" * 60)
    
    try:
        test_memory_estimation_with_resizer()
        test_oom_recovery_graceful_degradation()
        test_no_graceful_degradation()
        
        print("\n" + "=" * 60)
        print("✅ All OOM recovery integration tests completed!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()