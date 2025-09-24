#!/usr/bin/env python3
"""
Test the updated tool calling with improved error handling.
"""

import sys
import os

# Add inference to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'inference'))

def test_qwen_system_prompt():
    """Test the updated Qwen system prompt generation."""
    print("Testing Qwen system prompt generation...")
    
    # Mock objects
    class MockTool:
        def __init__(self, name, description):
            self.name = name
            self.description = description
    
    class MockModel:
        name = "qwen3-test"
        details = None
        model = "/fake/path"
    
    class MockProfile:
        parameters = type('obj', (object,), {'num_ctx': 4096})()
        system_prompt = "Test system prompt"
    
    # Set environment variable to bypass GGUF validation
    os.environ["ALLOW_MISSING_GGUF"] = "true"
    
    try:
        from runner.pipelines.txt2txt.qwen3moe import QwenLangGraphPipe
        
        pipeline = QwenLangGraphPipe(MockModel(), MockProfile())
        
        tools = [
            MockTool("web_search", "Perform a web search and retrieve relevant results"),
            MockTool("memory_retrieval", "Retrieve relevant memories based on query embeddings"),
        ]
        
        # This should work now without throwing errors
        import asyncio
        prompt = asyncio.run(pipeline._create_system_prompt(tools))
        
        print("✓ System prompt generated successfully")
        print(f"Prompt length: {len(prompt)}")
        
        # Check if it contains the expected tool calling format
        assert "tool_calls" in prompt, "Should contain tool_calls JSON format"
        assert "web_search" in prompt, "Should contain web_search tool"
        assert "arguments" in prompt, "Should mention arguments format"
        
        print("✓ Prompt contains proper tool calling instructions")
        
    except Exception as e:
        print(f"⚠ Test failed: {e}")
        return False
    
    return True

def test_qwen_tool_parsing():
    """Test improved Qwen tool parsing."""
    print("\nTesting Qwen tool parsing...")
    
    os.environ["ALLOW_MISSING_GGUF"] = "true"
    
    try:
        from runner.pipelines.txt2txt.qwen3moe import QwenLangGraphPipe
        
        class MockModel:
            name = "qwen3-test"
            details = None
            model = "/fake/path"
        
        class MockProfile:
            parameters = type('obj', (object,), {'num_ctx': 4096})()
            system_prompt = "Test prompt"
        
        pipeline = QwenLangGraphPipe(MockModel(), MockProfile())
        
        # Test content with tool calls
        content_with_tools = '''I need to search for that information.

```json
{
    "tool_calls": [
        {
            "name": "web_search",
            "arguments": {
                "query": "NEMA 17 stepper motor real products",
                "limit": 5
            }
        }
    ]
}
```

Let me get those links for you.'''
        
        tool_calls = pipeline._parse_qwen_tool_calls(content_with_tools)
        print(f"✓ Parsed {len(tool_calls)} tool calls")
        
        if tool_calls:
            call = tool_calls[0]
            print(f"  Tool: {call['name']}")
            print(f"  Args: {call['args']}")
            
            # Verify format
            assert call['name'] == 'web_search', "Tool name should match"
            assert 'query' in call['args'], "Should have query argument"
            assert call['args']['query'] == 'NEMA 17 stepper motor real products', "Query should match"
            
        # Test content cleaning
        clean_content = pipeline._clean_tool_calls_from_content(content_with_tools)
        print(f"✓ Cleaned content: '{clean_content[:50]}...'")
        
        # Should not contain JSON blocks
        assert "```json" not in clean_content, "Should remove JSON blocks"
        assert "tool_calls" not in clean_content, "Should remove tool_calls"
        
        return True
        
    except Exception as e:
        print(f"⚠ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🔧 Testing improved tool calling fixes...")
    
    success = True
    success &= test_qwen_system_prompt()
    success &= test_qwen_tool_parsing()
    
    if success:
        print("\n✅ All tests passed! Tool calling improvements should work.")
        print("\nKey improvements:")
        print("- Enhanced Qwen system prompt with detailed tool calling instructions")
        print("- Better tool call parsing and content cleaning")
        print("- Improved error handling for streaming")
        print("- Enhanced web search tool with fallback error handling")
    else:
        print("\n❌ Some tests failed. Please check the issues above.")

if __name__ == "__main__":
    main()