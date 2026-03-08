#!/usr/bin/env python3
"""
Debug LlamaCpp function calling - check if tool_calls are in the response
"""

import asyncio
import json
from runner.pipelines.llamacpp.base_llamacpp import BaseLlamaCppPipeline
from models.model_profile import ModelProfile, ModelParameters
from models.complexity_level import ComplexityLevel
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage


@tool
def get_current_weather(location: str, unit: str = "celsius") -> str:
    """Get the current weather in a given location."""
    return f"The weather in {location} is 22°{unit} and sunny."


async def debug_llamacpp_tool_calls():
    """Debug actual llama-cpp-python tool calling response format."""
    print("🔍 Testing LlamaCpp tool calling response format...")
    
    # Create model profile
    from models.model_profile_type import ModelProfileType
    profile = ModelProfile(
        user_id="debug-user",
        name="debug-profile", 
        system_prompt="You are a helpful assistant that can call functions when needed.",
        type=ModelProfileType.PRIMARY,
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
    
    # Create pipeline
    pipeline = BaseLlamaCppPipeline(profile=profile)
    
    # Bind tools
    tools = [get_current_weather]
    pipeline_with_tools = pipeline.bind_tools(tools)
    
    # Create test message that should trigger tool call
    messages = [HumanMessage(content="What's the weather like in Paris?")]
    
    print("\n📤 Sending request with tool binding...")
    print(f"Tools bound: {[t.name for t in tools]}")
    
    try:
        # Test non-streaming first
        print("\n🔄 Testing non-streaming response...")
        response = pipeline_with_tools._get_res(
            messages=messages,
            tools=tools,
            stream=False
        )
        
        print(f"📥 Raw response type: {type(response)}")
        print(f"📥 Raw response keys: {response.keys() if isinstance(response, dict) else 'Not a dict'}")
        
        if isinstance(response, dict):
            choices = response.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                print(f"📝 Message keys: {message.keys()}")
                print(f"📝 Content: {message.get('content', 'NO CONTENT')[:200]}...")
                
                # CHECK FOR TOOL CALLS!
                tool_calls = message.get("tool_calls")
                print(f"🔧 Tool calls: {tool_calls}")
                
                if tool_calls:
                    print("✅ TOOL CALLS FOUND IN RESPONSE!")
                    for i, call in enumerate(tool_calls):
                        print(f"   Tool {i+1}: {call}")
                else:
                    print("❌ NO TOOL CALLS FOUND - THIS IS THE ISSUE!")
                    print("   LlamaCpp might be outputting tool calls as text content")
        
        # Test streaming
        print("\n🔄 Testing streaming response...")
        
        response_stream = pipeline_with_tools._get_res(
            messages=messages,
            tools=tools,
            stream=True
        )
        
        chunks_with_tool_calls = 0
        total_chunks = 0
        all_content = ""
        
        for chunk in response_stream:
            total_chunks += 1
            if isinstance(chunk, dict) and "choices" in chunk:
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "") or ""
                tool_calls = delta.get("tool_calls")
                
                if content:
                    all_content += content
                    
                if tool_calls:
                    chunks_with_tool_calls += 1
                    print(f"🔧 Chunk {total_chunks} has tool calls: {tool_calls}")
        
        print(f"\n📊 Streaming summary:")
        print(f"   Total chunks: {total_chunks}")
        print(f"   Chunks with tool calls: {chunks_with_tool_calls}")
        print(f"   Final content: {all_content[:500]}...")
        
        if chunks_with_tool_calls == 0:
            print("❌ NO TOOL CALLS IN STREAMING - checking if content contains function markup")
            if "<function" in all_content or "function_call" in all_content:
                print("🎯 FOUND FUNCTION MARKUP IN CONTENT - this needs parsing!")
            
    except Exception as e:
        print(f"💥 Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(debug_llamacpp_tool_calls())