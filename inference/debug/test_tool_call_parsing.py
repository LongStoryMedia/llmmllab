#!/usr/bin/env python3

"""Test tool call parsing with different message formats."""

import json
from unittest.mock import MagicMock
from typing import Dict, Any

from langchain_core.messages import AIMessage, BaseMessage
from utils.tool_call_types import extract_tool_calls_as_models, is_langchain_tool_call
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="test_tool_call_parsing")

def create_mock_openai_tool_call():
    """Create a mock ChatCompletionMessageFunctionToolCall object."""
    mock_function = MagicMock()
    mock_function.name = "web_search"
    mock_function.arguments = '{"query": "test query"}'
    
    mock_tool_call = MagicMock()
    mock_tool_call.id = "test_id_123"
    mock_tool_call.function = mock_function
    mock_tool_call.type = "function"
    
    return mock_tool_call

def create_dict_tool_call():
    """Create a dictionary-format tool call."""
    return {
        "id": "test_id_456",
        "name": "web_search",
        "args": {"query": "test query dict"}
    }

def test_tool_call_parsing():
    """Test tool call parsing with different formats."""
    logger.info("🧪 Testing tool call parsing...")
    
    # Test 1: Dictionary format tool call (LangChain native)
    logger.info("Test 1: Dictionary format")
    dict_tool_call = create_dict_tool_call()
    
    logger.info(f"Dict tool call: {dict_tool_call}")
    is_valid_dict = is_langchain_tool_call(dict_tool_call)
    logger.info(f"is_langchain_tool_call(dict_format): {is_valid_dict}")
    
    # Test 2: Create AIMessage with dict tool call (this should work)
    logger.info("Test 2: AIMessage with dictionary tool call")
    try:
        ai_message_dict = AIMessage(
            content="I'll search for that.",
            tool_calls=[dict_tool_call]
        )
        
        logger.info(f"✅ AIMessage created successfully with tool_calls: {ai_message_dict.tool_calls}")
        
        # Test extract_tool_calls_as_models
        extracted_dict = extract_tool_calls_as_models(ai_message_dict)
        logger.info(f"✅ Successfully extracted {len(extracted_dict)} dict tool calls")
        for i, tc in enumerate(extracted_dict):
            logger.info(f"Dict tool call {i}: name={tc.name}, id={tc.execution_id}, args={tc.args}")
            
    except Exception as e:
        logger.error(f"❌ AIMessage with dict tool calls failed: {e}", exc_info=True)
    
    # Test 3: Mock OpenAI response format (simulate what we actually get)
    logger.info("Test 3: Simulating OpenAI ChatCompletionMessage format")
    try:
        # Create a mock response like what we get from OpenAI
        from unittest.mock import MagicMock
        
        mock_response = MagicMock()
        mock_response.content = "I'll search for that information."
        
        # Create mock tool call in OpenAI format
        mock_function = MagicMock()
        mock_function.name = "web_search"
        mock_function.arguments = '{"query": "test search"}'
        
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function = mock_function
        mock_tool_call.type = "function"
        
        mock_response.tool_calls = [mock_tool_call]
        
        logger.info(f"Mock OpenAI tool call: {mock_tool_call}")
        logger.info(f"Tool call type: {type(mock_tool_call)}")
        
        # Test our parsing function directly on the tool call
        is_valid_openai = is_langchain_tool_call(mock_tool_call)
        logger.info(f"is_langchain_tool_call(openai_format): {is_valid_openai}")
        
        # Test the extract_tool_call_requests function directly
        from utils.tool_call_types import extract_tool_call_requests
        from langchain_core.messages import AIMessage
        
        # Create an AIMessage that simulates what we get after OpenAI response conversion
        # This might happen in the server pipeline when converting responses
        ai_msg_openai = AIMessage(content="Test response")
        ai_msg_openai.tool_calls = [mock_tool_call]  # Set directly to bypass validation
        
        logger.info(f"AIMessage with OpenAI tool calls created")
        
        # Test extract_tool_call_requests
        extracted_requests = extract_tool_call_requests(ai_msg_openai)
        logger.info(f"✅ Extracted {len(extracted_requests)} requests from OpenAI format")
        for i, req in enumerate(extracted_requests):
            logger.info(f"Request {i}: {req}")
            
        # Test extract_tool_calls_as_models on this message
        extracted_models = extract_tool_calls_as_models(ai_msg_openai)
        logger.info(f"✅ Extracted {len(extracted_models)} models from OpenAI format")
        for i, tc in enumerate(extracted_models):
            logger.info(f"Model {i}: name={tc.name}, id={tc.execution_id}, args={tc.args}")
            
    except Exception as e:
        logger.error(f"❌ OpenAI format simulation failed: {e}", exc_info=True)

if __name__ == "__main__":
    test_tool_call_parsing()
    logger.info("🎉 Tool call parsing test completed")