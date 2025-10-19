#!/usr/bin/env python3
"""
Simple test to verify BaseLlamaCppPipeline initialization works.
"""

import asyncio
import sys
import os

# Add the inference directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

async def test_base_llamacpp_init():
    """Test that BaseLlamaCppPipeline can be initialized with proper parameters."""
    print("🧪 Testing BaseLlamaCppPipeline initialization...")
    
    try:
        from models import Model, ModelDetails, ModelTask, ModelProvider
        from runner.pipelines.llamacpp.base_llamacpp import BaseLlamaCppPipeline
        
        # Create minimal test objects
        class MockProfile:
            def __init__(self):
                self.model_name = "test-model"
                self.parameters = MockParameters()
                self.gpu_config = None
                
        class MockParameters:
            def __init__(self):
                self.num_ctx = 2048
                self.temperature = 0.7
                self.batch_size = 512
                self.seed = None
                self.top_p = 0.8
                self.top_k = 20
                self.min_p = 0.05
                self.repeat_penalty = 1.05
                self.stop = None
                self.max_tokens = 100
        
        # Create test model
        test_model = Model(
            id="test-model",
            name="Test Model",
            model="/fake/path/model.gguf",
            provider=ModelProvider.LLAMA_CPP,
            modified_at="2025-10-19T00:00:00Z",
            size=1000000,
            digest="test-digest",
            pipeline="TestPipe",
            details=ModelDetails(
                parent_model="test",
                format="gguf",
                family="test",
                families=["test"],
                parameter_size="1B",
                gguf_file="/fake/path/model.gguf"
            ),
            task=ModelTask.TEXTTOTEXT
        )
        
        test_profile = MockProfile()
        
        print(f"   📝 Testing BaseLlamaCppPipeline initialization...")
        print(f"   📝 Model: {test_model.name}")
        print(f"   📝 Profile: {test_profile.model_name}")
        
        try:
            # This should work now with our fix - we're NOT actually initializing llama
            # because the file doesn't exist, but the Pydantic validation should pass
            pipeline = BaseLlamaCppPipeline(test_model, test_profile, None)
            print("❌ Expected RuntimeError due to fake GGUF path, but validation passed!")
        except RuntimeError as e:
            if "Failed to initialize" in str(e):
                print("✅ Pydantic validation passed! (RuntimeError from llama initialization is expected)")
                print(f"   Expected error: {e}")
            else:
                print(f"❌ Unexpected RuntimeError: {e}")
                raise
        except Exception as e:
            if "ValidationError" in str(type(e)):
                print(f"❌ Pydantic validation still failing: {e}")
                raise
            else:
                print(f"❌ Unexpected error: {e}")
                raise
            
        print("\n🎉 BaseLlamaCppPipeline initialization test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_base_llamacpp_init())