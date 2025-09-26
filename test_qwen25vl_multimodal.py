#!/usr/bin/env python3
"""
Test script for Qwen2.5VL multimodal pipeline with tool calling.
"""

import asyncio
import sys
import os

# Add the inference path to make imports work
sys.path.append('/app/runner')
sys.path.append('/app/server') 
sys.path.append('/app/evaluation')

from models import Message, MessageContent, MessageContentType, MessageRole, Model, ModelProfile

def create_vision_test_message():
    """Create a test message with image for vision testing."""
    return Message(
        role=MessageRole.USER,
        content=[
            MessageContent(
                type=MessageContentType.TEXT,
                text="Describe what you see in this image and use the web_search tool if you need more information."
            ),
            MessageContent(
                type=MessageContentType.IMAGE,
                url="https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
            )
        ]
    )

def create_text_only_test_message():
    """Create a test message with only text for tool calling testing.""" 
    return Message(
        role=MessageRole.USER,
        content=[
            MessageContent(
                type=MessageContentType.TEXT,
                text="What are the latest developments in quantum computing? Please search for current information."
            )
        ]
    )

async def test_qwen25vl_parsing():
    """Test the tool call parsing functionality."""
    print("🧪 Testing Qwen2.5VL Tool Call Parsing")
    print("=" * 50)
    
    # Import here to avoid module loading issues
    from runner.pipelines.imgtxt2txt.qwen25_vl import Qwen25VLPipeline
    
    # Create a dummy model and profile for testing
    model = Model(
        id="qwen2.5-vl-32b-instruct-q4-k-m",
        name="Qwen2.5-VL-32B-Instruct",
        model="qwen2.5-vl-32b-instruct-q4-k-m",
        pipeline="Qwen25VLGGUFPipeline"
    )
    
    profile = ModelProfile(
        id="test-profile", 
        name="Test Profile",
        model_name="qwen2.5-vl-32b-instruct-q4-k-m"
    )
    
    # Create pipeline instance (but don't initialize LLM)
    pipeline = Qwen25VLPipeline(model, profile)
    
    # Test 1: Mixed function call format
    test_content1 = '''I can see this is a nature scene with a wooden boardwalk. Let me search for more information.

<function_call>
{"name": "web_search", "arguments": {"query": "wooden boardwalk nature park Wisconsin"}}
</FunctionCall>'''
    
    print("\n📝 Test 1: Mixed function_call format")
    result1 = pipeline._parse_qwen_tool_calls(test_content1)
    print(f"✅ Found {len(result1)} tool calls: {result1}")
    
    # Test 2: Proper Qwen format
    test_content2 = '''This appears to be a beautiful nature boardwalk. {"function_call": {"name": "web_search", "arguments": "{\\"query\\": \\"nature boardwalk photography\\"}"}}'''
    
    print("\n📝 Test 2: Proper Qwen function_call format") 
    result2 = pipeline._parse_qwen_tool_calls(test_content2)
    print(f"✅ Found {len(result2)} tool calls: {result2}")
    
    # Test 3: Content cleaning
    test_content3 = '''This is a scenic wooden boardwalk through what appears to be a natural area.

<function_call>
{"name": "web_search", "arguments": {"query": "nature boardwalk Wisconsin"}}
</function_call>

The image shows a peaceful walkway extending into the distance with natural vegetation on both sides.'''
    
    print("\n📝 Test 3: Content cleaning")
    cleaned = pipeline._clean_tool_calls_from_content(test_content3)
    print(f"Original length: {len(test_content3)}")
    print(f"Cleaned length: {len(cleaned)}")
    print(f"Cleaned content preview: {cleaned[:200]}...")
    
    print("\n🎉 Qwen2.5VL parsing tests completed!")

async def main():
    """Run the test."""
    try:
        await test_qwen25vl_parsing()
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)