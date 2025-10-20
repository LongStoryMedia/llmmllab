#!/usr/bin/env python3
"""
Test the fixed LlamaCpp function calling
"""

from runner.pipeline_factory import pipeline_factory
from models.model_profile import ModelProfile, ModelParameters
from models.model_profile_type import ModelProfileType
from models.complexity_level import ComplexityLevel
from models.model_task import ModelTask
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage


@tool
def get_current_weather(location: str, unit: str = "celsius") -> str:
    """Get the current weather in a given location."""
    return f"The weather in {location} is 22°{unit} and sunny."


def test_fixed_tool_calling():
    """Test the fixed tool calling implementation."""
    print("🔧 Testing FIXED LlamaCpp tool calling...")
    
    # Create model profile
    profile = ModelProfile(
        user_id="debug-user",
        name="debug-profile", 
        system_prompt="You are a helpful assistant that can call functions when needed.",
        type=ModelProfileType.Primary,
        model_name="qwen3-30b-a3b-q4-k-m",
        provider="ollama",
        parameters=ModelParameters(
            temperature=0.2,
            max_tokens=1000,
            num_ctx=40000
        ),
        complexity_level=ComplexityLevel.MODERATE,
        supports_tools=True,
        supports_streaming=True
    )
    
    # Get the model from the factory
    model = pipeline_factory._get_model_by_id("qwen3-30b-a3b-q4-k-m")
    if not model:
        print("❌ Model not found!")
        return
    
    # Create pipeline using the model object
    pipeline = pipeline_factory.create_pipeline(model, profile)
    
    print(f"✅ Created pipeline: {type(pipeline).__name__}")
    
    # Bind tools
    tools = [get_current_weather]
    pipeline_with_tools = pipeline.bind_tools(tools)
    
    # Create test message
    messages = [HumanMessage(content="What's the weather like in Paris?")]
    
    print("\n📤 Testing parsing function first...")
    
    try:
        # Test the parsing directly first
        test_content = '<think>\nI need to call get_weather for Paris.\n</think>\n\n<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "Paris"}}\n</tool_call>'
        
        cleaned, tool_calls = pipeline_with_tools._parse_tool_calls_from_content(test_content)
        print(f"✅ Parsing test:")
        print(f"   Cleaned content: '{cleaned}'")
        print(f"   Tool calls found: {len(tool_calls)}")
        for i, call in enumerate(tool_calls):
            print(f"   Tool {i+1}: {call}")
        
        print("\n📤 Testing actual generation...")
        # Test actual generation
        result = pipeline_with_tools._generate(messages, tools=tools)
        
        if result.generations:
            message = result.generations[0].message
            print(f"\n📥 Generation result:")
            print(f"   Content: '{message.content}'")
            print(f"   Has tool_calls: {hasattr(message, 'tool_calls') and message.tool_calls is not None}")
            
            if hasattr(message, 'tool_calls') and message.tool_calls:
                print(f"   ✅ TOOL CALLS EXTRACTED: {len(message.tool_calls)} calls")
                for i, call in enumerate(message.tool_calls):
                    print(f"      Call {i+1}: {call}")
            else:
                print(f"   ❌ NO TOOL CALLS EXTRACTED")
                print(f"   Raw content (first 500 chars): {message.content[:500]}")
    
    except Exception as e:
        print(f"💥 Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_fixed_tool_calling()