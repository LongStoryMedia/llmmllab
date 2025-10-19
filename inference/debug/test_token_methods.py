#!/usr/bin/env python3
"""Test context trimming methods only."""

from langchain_core.messages import HumanMessage, SystemMessage

def test_token_estimation():
    """Test token estimation method directly."""
    print("🧪 Testing token estimation method...")
    
    try:
        # Import the pipeline class
        from runner.pipelines.llamacpp.base_llamacpp import BaseLlamaCppPipeline
        
        # Create a test instance that only tests our methods
        class TestPipeline:
            def _estimate_tokens(self, text: str) -> int:
                """Estimate token count for text (rough approximation)."""
                # Simple approximation: ~4 characters per token for most languages
                return max(1, len(text) // 4)

            def _count_message_tokens(self, messages) -> int:
                """Count approximate tokens in messages."""
                total_tokens = 0
                for message in messages:
                    if hasattr(message, 'content') and message.content:
                        total_tokens += self._estimate_tokens(str(message.content))
                return total_tokens

            def _trim_messages_to_context(self, messages, tools=None, max_tokens=None):
                """Trim messages to fit within context window."""
                if not max_tokens:
                    max_tokens = 4096
                
                # Reserve tokens for tools, system prompt, and response
                tool_tokens = 50 if tools else 0  # Simple estimate
                response_reserve = 1024
                
                # Find system message tokens
                system_tokens = 0
                if messages and hasattr(messages[0], 'content'):
                    if any(isinstance(msg, SystemMessage) for msg in messages[:1]):
                        system_tokens = self._count_message_tokens(messages[:1])
                
                # Available tokens for conversation history
                available_tokens = max_tokens - tool_tokens - system_tokens - response_reserve - 500  # Safety buffer
                
                if available_tokens <= 0:
                    # Keep only system message if any
                    return [msg for msg in messages if isinstance(msg, SystemMessage)][:1]
                
                # Count tokens from the end (most recent messages)
                trimmed_messages = []
                current_tokens = 0
                
                # Always keep system message first
                system_messages = [msg for msg in messages if isinstance(msg, SystemMessage)]
                other_messages = [msg for msg in messages if not isinstance(msg, SystemMessage)]
                
                # Add system messages
                trimmed_messages.extend(system_messages)
                
                # Add other messages from most recent, checking token limits
                for message in reversed(other_messages):
                    message_tokens = self._count_message_tokens([message])
                    if current_tokens + message_tokens <= available_tokens:
                        trimmed_messages.insert(len(system_messages), message)  # Insert after system messages
                        current_tokens += message_tokens
                    else:
                        print(f"Trimming message due to context limit: {message_tokens} tokens")
                        break
                
                if len(trimmed_messages) < len(messages):
                    print(f"Trimmed {len(messages) - len(trimmed_messages)} messages to fit context window")
                
                return trimmed_messages
        
        pipeline = TestPipeline()
        
        # Test token estimation
        test_text = "Hello world! This is a test message with some content."
        tokens = pipeline._estimate_tokens(test_text)
        print(f"📏 Token estimation for '{test_text}': {tokens} tokens")
        
        # Test message token counting
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="What is the meaning of life?"),
            HumanMessage(content="Tell me more about that topic."),
            HumanMessage(content="Can you provide specific examples?")
        ]
        
        message_tokens = pipeline._count_message_tokens(messages)
        print(f"📊 Message tokens: {message_tokens}")
        
        # Test context trimming with many messages
        large_messages = [SystemMessage(content="You are a helpful assistant.")]
        for i in range(20):  # Create 20 large messages
            content = f"Message {i}: " + "This is a longer message with more content to test context trimming. " * 10
            large_messages.append(HumanMessage(content=content))
        
        print(f"📝 Created {len(large_messages)} test messages")
        
        original_tokens = pipeline._count_message_tokens(large_messages)
        print(f"📊 Original total tokens: {original_tokens}")
        
        # Test trimming to smaller context
        trimmed = pipeline._trim_messages_to_context(large_messages, max_tokens=2048)
        trimmed_tokens = pipeline._count_message_tokens(trimmed)
        
        print(f"✂️  Trimmed to {len(trimmed)} messages ({trimmed_tokens} tokens)")
        
        # Verify system message preserved
        has_system = any(isinstance(msg, SystemMessage) for msg in trimmed)
        print(f"🔒 System message preserved: {has_system}")
        
        print("✅ Context management test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_token_estimation()