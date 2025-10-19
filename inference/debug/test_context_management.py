#!/usr/bin/env python3
"""Test context window management for llama-cpp-python pipelines."""

import asyncio
from langchain_core.messages import HumanMessage, SystemMessage

async def test_context_trimming():
    """Test that context trimming works correctly."""
    print("🧪 Testing context window management...")
    
    try:
        from runner.pipeline_factory import PipelineFactory
        from models import ModelTask
        
        # Create factory instance
        factory = PipelineFactory()
        
        # Create a pipeline that should have context window issues
        pipeline = await factory.create_pipeline("qwen3-30b-a3b-q4-k-m", ModelTask.TEXTTOTEXT)
        
        if not hasattr(pipeline, '_trim_messages_to_context'):
            print("❌ Context trimming method not found")
            return
        
        # Create messages that would exceed context
        system_msg = SystemMessage(content="You are a helpful assistant.")
        
        # Create many large messages to exceed context
        large_messages = [system_msg]
        for i in range(100):  # 100 large messages
            content = f"This is message {i}. " + "A" * 1000  # 1000+ char message
            large_messages.append(HumanMessage(content=content))
        
        print(f"📊 Created {len(large_messages)} messages for testing")
        
        # Test token counting
        message_tokens = pipeline._count_message_tokens(large_messages)
        print(f"📏 Total message tokens (estimated): {message_tokens}")
        
        # Test context trimming
        trimmed = pipeline._trim_messages_to_context(large_messages, max_tokens=8192)
        trimmed_tokens = pipeline._count_message_tokens(trimmed)
        
        print(f"✂️  Trimmed to {len(trimmed)} messages ({trimmed_tokens} tokens)")
        print(f"✅ Context trimming test completed successfully")
        
        # Verify system message is preserved
        has_system = any(isinstance(msg, SystemMessage) for msg in trimmed)
        print(f"🔒 System message preserved: {has_system}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(test_context_trimming())