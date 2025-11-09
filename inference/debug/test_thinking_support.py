#!/usr/bin/env python3
"""
Test thinking support in LlamaCppServerPipeline.
Verifies that thinking models properly handle <think> tags.
"""

import asyncio
from models.model_parameters import ModelParameters
from models.user_config import UserConfig
from runner.pipeline_factory import PipelineFactory

async def test_thinking_support():
    """Test thinking support with thinking enabled and disabled."""
    
    # Test with thinking enabled (think=True)
    print("🧠 Testing thinking support with think=True")
    
    # Load user config
    user_config = UserConfig.load_system_default()
    
    # Create pipeline with thinking enabled
    think_params = ModelParameters(
        model_name="qwen3-vl-30b-a3b-thinking",
        think=True,  # Enable thinking
        num_ctx=4096,
        temperature=0.7,
    )
    
    pipeline = await PipelineFactory.create_pipeline(
        model_name="qwen3-vl-30b-a3b-thinking",
        parameters=think_params,
        user_config=user_config
    )
    
    try:
        from langchain_core.messages import HumanMessage
        
        # Test message that might trigger thinking
        messages = [
            HumanMessage(content="Think step by step: What is 15 + 27? Show your reasoning.")
        ]
        
        print("📝 Generating response with thinking enabled...")
        result = pipeline._generate(messages)
        
        if result.generations:
            content = result.generations[0].message.content
            print(f"✅ Response with thinking: {content[:200]}...")
            
            # Check if thinking tags are preserved
            if "<think>" in content:
                print("✅ Thinking tags preserved (as expected with think=True)")
            else:
                print("ℹ️ No thinking tags in response")
        
        # Now test with thinking disabled
        print("\n🚫 Testing thinking support with think=False")
        
        no_think_params = ModelParameters(
            model_name="qwen3-vl-30b-a3b-thinking",
            think=False,  # Disable thinking
            num_ctx=4096,
            temperature=0.7,
        )
        
        pipeline_no_think = await PipelineFactory.create_pipeline(
            model_name="qwen3-vl-30b-a3b-thinking",
            parameters=no_think_params,
            user_config=user_config
        )
        
        print("📝 Generating response with thinking disabled...")
        result_no_think = pipeline_no_think._generate(messages)
        
        if result_no_think.generations:
            content_no_think = result_no_think.generations[0].message.content
            print(f"✅ Response without thinking: {content_no_think[:200]}...")
            
            # Check if thinking tags are removed
            if "<think>" not in content_no_think:
                print("✅ Thinking tags filtered out (as expected with think=False)")
            else:
                print("⚠️ Thinking tags still present (unexpected with think=False)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        pipeline.close()
        if 'pipeline_no_think' in locals():
            pipeline_no_think.close()

if __name__ == "__main__":
    asyncio.run(test_thinking_support())