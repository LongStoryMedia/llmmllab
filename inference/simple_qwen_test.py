#!/usr/bin/env python3
"""
Simple QwenMoE Tool Calling Test
Test QwenMoE model's tool calling capabilities with detailed validation
"""

import asyncio
import json
import re
from runner.pipeline_factory import PipelineFactory

async def test_qwen_tool_calling():
    """Test QwenMoE tool calling with detailed validation"""
    
    print("🚀 Starting Simple QwenMoE Tool Calling Test")
    
    # Initialize pipeline factory
    factory = PipelineFactory()
    
    # Test request for tool calling
    test_request = {
        "messages": [
            {
                "role": "user", 
                "content": "Can you search for information about recent AI breakthroughs in 2024? Use the web_search tool to find current information."
            }
        ]
    }
    
    # Get QwenMoE model
    model_name = "qwen3-30b-a3b-q4-k-m"
    
    try:
        print(f"📋 Testing model: {model_name}")
        
        # Create pipeline
        pipeline = await factory.get_pipeline_async(
            model_name=model_name,
            user_id="test_user",
            profile_name=None,
            task="text_generation"
        )
        
        print(f"✅ Pipeline created: {type(pipeline).__name__}")
        
        # Test streaming with tool context
        print("🔍 Testing tool calling capabilities...")
        
        # Simulate tool availability
        available_tools = [
            {"name": "web_search", "description": "Search the web for current information"},
            {"name": "memory_retrieval", "description": "Retrieve relevant memories"},
            {"name": "summarization", "description": "Summarize content"}
        ]
        
        # Add tools to request context
        test_request["available_tools"] = available_tools
        
        response_chunks = []
        
        # Stream the response
        async for chunk in pipeline.stream(test_request):
            response_chunks.append(chunk)
            
            # Check if this chunk contains tool calls
            if hasattr(chunk, 'message') and chunk.message:
                if hasattr(chunk.message, 'content') and chunk.message.content:
                    content_text = ""
                    for content_item in chunk.message.content:
                        if hasattr(content_item, 'text'):
                            content_text += content_item.text
                    
                    # Look for JSON tool calling patterns
                    if content_text and any(pattern in content_text for pattern in ['```json', 'tool_calls', '"name":', '"arguments":']):
                        print(f"🔧 Found potential tool call pattern in chunk: {content_text[:100]}...")
        
        print(f"📊 Collected {len(response_chunks)} response chunks")
        
        # Analyze full response for tool calls
        full_response = ""
        for chunk in response_chunks:
            if hasattr(chunk, 'message') and chunk.message and hasattr(chunk.message, 'content'):
                for content_item in chunk.message.content:
                    if hasattr(content_item, 'text'):
                        full_response += content_item.text
        
        print(f"📄 Full response length: {len(full_response)} chars")
        print(f"📄 Response preview: {full_response[:200]}...")
        
        # Check for tool calling patterns
        tool_patterns = {
            'json_blocks': len(re.findall(r'```json.*?```', full_response, re.DOTALL)),
            'tool_calls_array': len(re.findall(r'"tool_calls":\s*\[', full_response)),
            'name_field': len(re.findall(r'"name":\s*"[\w_]+"', full_response)),
            'arguments_field': len(re.findall(r'"arguments":\s*{', full_response)),
            'web_search_mention': 'web_search' in full_response.lower(),
            'search_mention': 'search' in full_response.lower()
        }
        
        print("🔍 Tool Pattern Analysis:")
        for pattern, count in tool_patterns.items():
            status = "✅" if (count > 0 if isinstance(count, int) else count) else "❌"
            print(f"   {status} {pattern}: {count}")
        
        # Determine if tools were used
        tool_usage_detected = any([
            tool_patterns['json_blocks'] > 0,
            tool_patterns['tool_calls_array'] > 0,
            tool_patterns['name_field'] > 0 and tool_patterns['arguments_field'] > 0
        ])
        
        print(f"\n🎯 Tool Usage Result: {'✅ DETECTED' if tool_usage_detected else '❌ NOT DETECTED'}")
        
        return {
            'success': True,
            'model': model_name,
            'response_length': len(full_response),
            'tool_patterns': tool_patterns,
            'tool_usage_detected': tool_usage_detected,
            'response_preview': full_response[:500]
        }
        
    except Exception as e:
        print(f"❌ Error testing {model_name}: {e}")
        return {
            'success': False,
            'model': model_name,
            'error': str(e)
        }

if __name__ == "__main__":
    result = asyncio.run(test_qwen_tool_calling())
    print(f"\n📊 Final Result: {json.dumps(result, indent=2)}")