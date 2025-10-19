#!/usr/bin/env python3
"""Test OpenAI tool format conversion."""

import sys
import traceback
from runner.pipeline_factory import pipeline_factory
from models import ModelProfile, ModelProfileType
from langchain_core.tools import tool

@tool
def test_tool(query: str) -> str:
    """A simple test tool for validation.
    
    Args:
        query: The test query
        
    Returns:
        A test response
    """
    return f"Test response for: {query}"

async def test_openai_tool_conversion():
    """Test that LangChain tools can be converted to OpenAI format."""
    print("🧪 Testing OpenAI tool format conversion...")
    
    try:
        # Use global pipeline factory
        factory = pipeline_factory
        
        # Get a model and create a pipeline
        model_name = "llama-chat-summary-3_2-3b-q5-k-m"
        model = factory.models[model_name]
        
        # Create a basic profile
        from models import ModelProfile, ModelProfileType
        profile = ModelProfile(
            name="test_profile",
            model_name=model.name,
            parameters={},
            user_id="test_user",
            system_prompt="You are a helpful assistant.",
            type=ModelProfileType.Primary
        )
        
        # Create a pipeline
        pipeline = factory.create_pipeline(model, profile)
        print(f"✅ Created pipeline: {type(pipeline).__name__}")
        
        # Test tool conversion
        tools = [test_tool]
        converted_tools = pipeline._convert_tools_to_openai_format(tools)
        
        if converted_tools:
            print(f"✅ Converted {len(converted_tools)} tools to OpenAI format")
            for i, tool_dict in enumerate(converted_tools):
                print(f"  Tool {i+1}: {tool_dict}")
                
                # Verify structure
                assert "type" in tool_dict
                assert tool_dict["type"] == "function"
                assert "function" in tool_dict
                assert "name" in tool_dict["function"]
                assert "description" in tool_dict["function"]
                print(f"    ✓ Tool {i+1} has correct structure")
        else:
            print("❌ No tools converted")
            return False
            
        print("🎉 OpenAI tool conversion test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        print("Traceback:")
        traceback.print_exc()
        return False

async def main():
    """Run the test."""
    success = await test_openai_tool_conversion()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())