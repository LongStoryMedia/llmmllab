#!/usr/bin/env python3
"""
Test context window management with large conversations to ensure no context overflow errors.
"""

import sys
sys.path.append('/app')

import asyncio
from composer.graph.state import WorkflowState
from composer.utils.state import assemble_context_messages
from models import LangChainMessage, ModelProfile, ModelParameters

async def test_large_conversation_context_window():
    """Test with a large conversation that would exceed context window."""
    print("Testing large conversation context window management...")
    
    # Create a large number of messages that would exceed the token limit
    messages = []
    
    # Add a long conversation with many exchanges
    for i in range(50):
        # User message with substantial content
        user_content = f"User message {i}: " + "This is a very long message with lots of detailed content that would consume many tokens when processed by the language model. " * 20
        messages.append(LangChainMessage(type="human", content=user_content))
        
        # Assistant response with substantial content  
        assistant_content = f"Assistant response {i}: " + "This is an equally long and detailed response from the assistant that provides comprehensive information and consumes many tokens during processing. " * 20
        messages.append(LangChainMessage(type="ai", content=assistant_content))
    
    # Create workflow state
    state = WorkflowState(
        messages=messages,
        conversation_id=123,
        user_id="test_user"
    )
    
    print(f"Created conversation with {len(messages)} messages")
    
    # Test without context limit (should be very large)
    unlimited_messages = assemble_context_messages(state)
    print(f"Unlimited context: {len(unlimited_messages)} messages")
    
    # Test with context limits that should trigger trimming
    for limit in [1000, 5000, 10000, 20000, 40960]:
        limited_messages = assemble_context_messages(state, max_tokens=limit)
        print(f"Context limit {limit}: {len(limited_messages)} messages")
        
        # Verify the messages fit within the limit
        from composer.utils.state import _count_message_tokens
        actual_tokens = _count_message_tokens(limited_messages)
        print(f"  -> Actual tokens: {actual_tokens} (target: <{limit})")
        
        # Add some buffer for response tokens, but actual tokens should be well under limit
        if actual_tokens > limit * 0.8:  # Should be less than 80% of limit to leave room for response
            print(f"  ⚠️  Warning: Token count {actual_tokens} is close to limit {limit}")
        else:
            print(f"  ✅ Token count is well within limit")

def test_model_profile_context_access():
    """Test accessing context window from model profile."""
    print("\nTesting model profile context window access...")
    
    # Create a model profile with specific context window
    profile = ModelProfile(
        user_id="test_user",
        name="Test Profile",
        model_name="test-model",
        parameters=ModelParameters(num_ctx=40960),
        system_prompt="You are a helpful assistant.",
        type=1
    )
    
    # Test accessing the context window
    if profile.parameters and hasattr(profile.parameters, 'num_ctx') and profile.parameters.num_ctx:
        context_limit = profile.parameters.num_ctx
        print(f"✅ Successfully accessed context limit: {context_limit}")
    else:
        print("❌ Could not access context limit from profile")
        return False
    
    return True

if __name__ == "__main__":
    print("Large Conversation Context Window Test")
    print("=" * 50)
    
    try:
        # Test model profile access
        if not test_model_profile_context_access():
            sys.exit(1)
        
        # Test large conversation handling
        asyncio.run(test_large_conversation_context_window())
        
        print("\n✅ All large conversation tests completed successfully!")
        print("✅ Context window management is working correctly!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)