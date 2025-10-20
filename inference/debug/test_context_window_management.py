#!/usr/bin/env python3
"""
Test context window management in assemble_context_messages function.
"""

import sys
sys.path.append('/app')

from composer.utils.state import assemble_context_messages, _estimate_tokens, _count_message_tokens, _trim_messages_to_context_window
from composer.graph.state import WorkflowState
from models import (
    Message, 
    MessageRole, 
    MessageContent, 
    MessageContentType,
    LangChainMessage
)

def test_token_counting():
    """Test basic token counting functionality."""
    print("Testing token counting...")
    
    # Test text token estimation
    test_text = "This is a test message with some content."
    tokens = _estimate_tokens(test_text)
    print(f"Text: '{test_text}' -> {tokens} tokens (length: {len(test_text)})")
    
    # Test message token counting
    message = Message(
        content=[MessageContent(type=MessageContentType.TEXT, text=test_text)],
        role=MessageRole.USER
    )
    message_tokens = _count_message_tokens([message])
    print(f"Message tokens: {message_tokens}")

def test_context_trimming():
    """Test context window trimming functionality."""
    print("\nTesting context trimming...")
    
    # Create a large set of messages
    messages = []
    
    # Add a system message
    messages.append(Message(
        content=[MessageContent(type=MessageContentType.TEXT, text="You are a helpful assistant.")],
        role=MessageRole.SYSTEM
    ))
    
    # Add many user/assistant messages
    for i in range(20):
        messages.append(Message(
            content=[MessageContent(type=MessageContentType.TEXT, text=f"User message {i}: " + "This is a long message with lots of content. " * 50)],
            role=MessageRole.USER
        ))
        messages.append(Message(
            content=[MessageContent(type=MessageContentType.TEXT, text=f"Assistant response {i}: " + "This is an equally long response with lots of content. " * 50)],
            role=MessageRole.ASSISTANT
        ))
    
    total_tokens = _count_message_tokens(messages)
    print(f"Total messages: {len(messages)}, Total tokens: {total_tokens}")
    
    # Test trimming to different limits
    for limit in [1000, 5000, 10000]:
        trimmed = _trim_messages_to_context_window(messages, limit)
        trimmed_tokens = _count_message_tokens(trimmed)
        print(f"Limit {limit}: {len(trimmed)} messages, {trimmed_tokens} tokens")

def test_assemble_with_context_limit():
    """Test assemble_context_messages with context limit."""
    print("\nTesting assemble_context_messages with context limit...")
    
    # Create a simple workflow state
    langchain_messages = [
        LangChainMessage(type="human", content="What is the capital of France?"),
        LangChainMessage(type="ai", content="The capital of France is Paris.")
    ]
    
    state = WorkflowState(
        messages=langchain_messages,
        conversation_id=123,
        user_id="test_user"
    )
    
    # Test without context limit
    messages_unlimited = assemble_context_messages(state)
    print(f"Unlimited: {len(messages_unlimited)} messages")
    
    # Test with context limit
    messages_limited = assemble_context_messages(state, max_tokens=1000)
    print(f"Limited to 1000 tokens: {len(messages_limited)} messages")
    
    # Test with very small limit
    messages_tiny = assemble_context_messages(state, max_tokens=100)
    print(f"Limited to 100 tokens: {len(messages_tiny)} messages")

if __name__ == "__main__":
    print("Context Window Management Test")
    print("=" * 40)
    
    try:
        test_token_counting()
        test_context_trimming()
        test_assemble_with_context_limit()
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)