#!/usr/bin/env python3
"""
Simple test to validate typing changes without database dependencies.
Tests the new ToolCall model and ChatResponse structure changes.
"""

import sys
import os
sys.path.insert(0, '/app')

from models import ToolCall, ChatResponse, Message, MessageRole, MessageContent, MessageContentType, Thought

def test_tool_call_model():
    """Test ToolCall model creation and validation."""
    print("🧪 Testing ToolCall model...")
    
    tool_call = ToolCall(
        tool_name="test_tool",
        execution_id="call_123",
        success=True,
        args={"param1": "value1"},
        result_data={"output": "test result"},
        execution_time_ms=150.5,
        error_message=None
    )
    
    assert tool_call.tool_name == "test_tool"
    assert tool_call.execution_id == "call_123"
    assert tool_call.success is True
    assert tool_call.args == {"param1": "value1"}
    assert tool_call.result_data == {"output": "test result"}
    assert tool_call.execution_time_ms == 150.5
    
    print("✅ ToolCall model validation passed")

def test_message_with_tool_calls():
    """Test Message model with tool_calls array."""
    print("🧪 Testing Message with tool_calls...")
    
    tool_calls = [
        ToolCall(
            tool_name="search_tool",
            execution_id="call_1",
            success=True,
            args={"query": "test"},
            result_data={"results": ["item1", "item2"]}
        ),
        ToolCall(
            tool_name="analysis_tool", 
            execution_id="call_2",
            success=False,
            error_message="Tool failed",
            execution_time_ms=75.0
        )
    ]
    
    message = Message(
        role=MessageRole.ASSISTANT,
        content=[MessageContent(type=MessageContentType.TEXT, text="Here are the results:")],
        tool_calls=tool_calls
    )
    
    assert message.role == MessageRole.ASSISTANT
    assert len(message.tool_calls) == 2
    assert message.tool_calls[0].tool_name == "search_tool"
    assert message.tool_calls[1].success is False
    
    print("✅ Message with tool_calls validation passed")

def test_message_with_thoughts():
    """Test Message model with thoughts array."""
    print("🧪 Testing Message with thoughts...")
    
    thoughts = [
        Thought(text="First thought about the problem"),
        Thought(text="Second thought with analysis")
    ]
    
    message = Message(
        role=MessageRole.ASSISTANT,
        content=[MessageContent(type=MessageContentType.TEXT, text="Final response")],
        thoughts=thoughts
    )
    
    assert message.role == MessageRole.ASSISTANT
    assert len(message.thoughts) == 2
    assert message.thoughts[0].text == "First thought about the problem"
    assert message.thoughts[1].text == "Second thought with analysis"
    
    print("✅ Message with thoughts validation passed")

def test_chat_response_structure():
    """Test ChatResponse with new message-centric structure."""
    print("🧪 Testing ChatResponse structure...")
    
    # Create a message with both thoughts and tool_calls
    thoughts = [Thought(text="Thinking about this request")]
    tool_calls = [ToolCall(
        tool_name="web_search",
        execution_id="search_1", 
        success=True,
        args={"query": "test search"},
        result_data={"count": 3}
    )]
    
    message = Message(
        role=MessageRole.ASSISTANT,
        content=[MessageContent(type=MessageContentType.TEXT, text="Response text")],
        thoughts=thoughts,
        tool_calls=tool_calls
    )
    
    response = ChatResponse(
        done=True,
        message=message,
        finish_reason="stop"
    )
    
    assert response.done is True
    assert response.message.role == MessageRole.ASSISTANT
    assert len(response.message.thoughts) == 1
    assert len(response.message.tool_calls) == 1
    assert response.message.tool_calls[0].tool_name == "web_search"
    
    print("✅ ChatResponse structure validation passed")

def main():
    """Run all typing validation tests."""
    print("🚀 Starting typing validation tests...")
    print()
    
    try:
        test_tool_call_model()
        test_message_with_tool_calls()
        test_message_with_thoughts()
        test_chat_response_structure()
        
        print()
        print("🎉 All typing validation tests passed!")
        print("✅ Schema refactoring completed successfully")
        print("✅ ToolCall model is working correctly")
        print("✅ Message.tool_calls and Message.thoughts are properly implemented")
        print("✅ ChatResponse uses new message-centric structure")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()