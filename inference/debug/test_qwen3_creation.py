#!/usr/bin/env python3
"""
Test script to verify Qwen3Moe pipeline creation works after the fix.
"""

import asyncio
import sys
import os

# Add the inference directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

async def test_qwen3_creation():
    """Test that Qwen3Moe can be created with proper parameters."""
    print("🧪 Testing Qwen3Moe pipeline creation...")
    
    try:
        from models import Model, ModelProfile, ModelDetails, ModelTask, ModelProvider
        from runner.pipelines.txt2txt.qwen3moe import Qwen3Moe
        
        # Create a test model
        test_model = Model(
            id="test-qwen3-4b",
            name="Qwen3-4B",
            model="/test/path/qwen3-4b.gguf",
            provider=ModelProvider.LLAMA_CPP,
            modified_at="2025-10-19T00:00:00Z",
            size=4000000000,
            digest="test-digest",
            pipeline="Qwen3Pipe",
            details=ModelDetails(
                parent_model="qwen3",
                format="gguf",
                family="qwen",
                families=["qwen"],
                parameter_size="4B",
                gguf_file="/test/path/qwen3-4b.gguf"
            ),
            task=ModelTask.TEXTTOTEXT
        )
        
        # Create a test profile  
        test_profile = ModelProfile(
            model_name="Qwen3-4B",
            n_ctx=2048,
            temperature=0.7,
            max_tokens=100
        )
        
        print(f"   📝 Creating Qwen3Moe with model: {test_model.name}")
        print(f"   📝 Profile: {test_profile.model_name}")
        
        # Try to create the pipeline
        pipeline = Qwen3Moe(test_model, test_profile, None)
        
        print(f"✅ Successfully created Qwen3Moe pipeline!")
        print(f"   Pipeline type: {type(pipeline).__name__}")
        print(f"   LLM type: {pipeline._llm_type}")
        print(f"   Model name: {pipeline.model.name}")
        
        # Clean up
        if hasattr(pipeline, 'close'):
            pipeline.close()
            
        print("\n🎉 Qwen3Moe creation test completed successfully!")
        
    except Exception as e:
        print(f"❌ Qwen3Moe creation test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_qwen3_creation())