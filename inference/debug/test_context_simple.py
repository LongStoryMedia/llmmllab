#!/usr/bin/env python3
"""Simple test for context window management methods."""

from langchain_core.messages import HumanMessage, SystemMessage
import json

def test_context_methods():
    """Test the context management methods directly."""
    print("🧪 Testing context management methods...")
    
    try:
        # Import the pipeline class
        from runner.pipelines.llamacpp.base_llamacpp import BaseLlamaCppPipeline
        from models import Model, ModelProfile, ModelParameters
        
        # Create a mock model and profile for testing
        model = Model(
            name="test-model", 
            model="test-path",
            details=None,
            task="texttotext"
        )
        
        profile = ModelProfile(
            name="test-profile",
            parameters=ModelParameters(
                num_ctx=8192,  # 8K context
                max_tokens=1024,
                temperature=0.7
            )
        )
        
        # Create a minimal test instance (without llama initialization)
        class TestPipeline(BaseLlamaCppPipeline):
            def __init__(self):
                self.model = model
                self.profile = profile
                self.grammar = None
                self._bound_tools = []
                self.llama_instance = None
                # Mock llama instance with n_ctx
                class MockLlama:
                    def n_ctx(self):
                        return 8192
                self.llama_instance = MockLlama()
        
        pipeline = TestPipeline()
        
        # Test token estimation
        test_text = "Hello world! This is a test message."
        tokens = pipeline._estimate_tokens(test_text)
        print(f"📏 Token estimation for '{test_text}': {tokens} tokens")
        
        # Test message token counting
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="What is the meaning of life?"),
            HumanMessage(content="Tell me more about that.")
        ]
        
        message_tokens = pipeline._count_message_tokens(messages)
        print(f"📊 Message tokens: {message_tokens}")
        
        # Test tool token counting
        mock_tools = [{
            'function': {
                'name': 'web_search',
                'description': 'Search the web for information',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'query': {'type': 'string', 'description': 'Search query'}
                    },
                    'required': ['query']
                }
            }
        }]
        
        # Convert to the format the method expects
        class MockTool:
            def __init__(self, data):
                self.function = MockFunction(data['function'])
                
        class MockFunction:
            def __init__(self, data):
                self.name = data['name']
                self.description = data['description']
                self.parameters = data['parameters']
        
        tools = [MockTool(tool) for tool in mock_tools]
        tool_tokens = pipeline._count_tool_tokens(tools)
        print(f"🔧 Tool tokens: {tool_tokens}")
        
        # Test context trimming with many messages
        large_messages = [SystemMessage(content="You are a helpful assistant.")]
        for i in range(50):  # Create 50 large messages
            content = f"Message {i}: " + "A" * 200  # 200+ char message
            large_messages.append(HumanMessage(content=content))
        
        print(f"📝 Created {len(large_messages)} test messages")
        
        original_tokens = pipeline._count_message_tokens(large_messages)
        print(f"📊 Original total tokens: {original_tokens}")
        
        # Test trimming to smaller context
        trimmed = pipeline._trim_messages_to_context(large_messages, tools, max_tokens=4096)
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
    test_context_methods()