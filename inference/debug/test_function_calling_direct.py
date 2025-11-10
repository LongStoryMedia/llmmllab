#!/usr/bin/env python3
"""
Direct function calling test with ChatOpenAI to debug tool calling.
"""
import asyncio
import sys
import os
from pathlib import Path

# Add the inference directory to the Python path for imports
sys.path.append(str(Path(__file__).parent.parent))

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="test_function_calling_direct")

# Test function calling with a simple tool
test_tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

async def test_direct_function_calling():
    """Test direct function calling with ChatOpenAI."""
    logger.info("🧪 Starting direct function calling test")
    
    try:
        # Create ChatOpenAI instance
        chat_model = ChatOpenAI(
            base_url="http://localhost:8001/v1",
            api_key="dummy",
            model="local-model",
            temperature=0.1,
        )
        
        # Bind tools
        chat_with_tools = chat_model.bind_tools(test_tools)
        
        # Test message that should trigger function calling
        test_message = HumanMessage(content="What's the weather like in San Francisco?")
        
        logger.info("📤 Sending test message to ChatOpenAI...")
        
        # Make the call
        response = await chat_with_tools.ainvoke([test_message])
        
        logger.info(f"📨 Response type: {type(response)}")
        logger.info(f"📨 Response content: {response.content}")
        logger.info(f"📨 Response tool_calls: {getattr(response, 'tool_calls', None)}")
        logger.info(f"📨 Response additional_kwargs: {getattr(response, 'additional_kwargs', {})}")
        
        # Check for tool calls
        if hasattr(response, 'tool_calls') and response.tool_calls:
            logger.info(f"✅ Tool calls detected: {len(response.tool_calls)}")
            for i, tool_call in enumerate(response.tool_calls):
                logger.info(f"🔧 Tool call {i}: {tool_call}")
        else:
            logger.warning("❌ No tool calls in response")
            
        # Check raw response
        if hasattr(response, 'response_metadata'):
            logger.info(f"🔍 Response metadata: {response.response_metadata}")
            
        return response
        
    except Exception as e:
        logger.error(f"❌ Function calling test failed: {e}", exc_info=True)
        return None

if __name__ == "__main__":
    asyncio.run(test_direct_function_calling())