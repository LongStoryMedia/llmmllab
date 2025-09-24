#!/usr/bin/env python3
"""
Test script to directly test OpenAI GPT OSS tool calling
This bypasses the API auth layer and tests the pipeline directly
"""

import sys
import os
import asyncio
import json

# Add the paths for imports
sys.path.append("/app")
sys.path.append("/app/runner")
sys.path.append("/app/server")


async def test_tool_calling():
    """Test tool calling by examining our debugging code"""

    try:
        from runner.pipelines.txt2txt.openai_gpt_oss import OpenAiGptOssPipe
        from models.lang_chain_message import LangChainMessage

        print("✅ Imports successful")

        # Let's just test the parsing functions to see if our debugging is working
        pipeline_class = OpenAiGptOssPipe

        # Create a mock instance just to test methods
        class MockPipeline:
            def __init__(self):
                self._logger = logging.getLogger("test-pipeline")

        import logging

        logging.basicConfig(level=logging.DEBUG)

        mock = MockPipeline()

        # Bind the methods to our mock instance
        parse_harmony_tool_calls = OpenAiGptOssPipe._parse_harmony_tool_calls
        extract_final_content = OpenAiGptOssPipe._extract_final_content

        # Test harmony parsing with the correct format
        test_content = """
        <|channel|>analysis<|message|>
        I need to search for iPhone 16 models on Amazon to help the user.
        <|end|>
        
        <|channel|>commentary to=functions <|constrain|>json<|message|>
        {
            "name": "web_search",
            "arguments": {
                "query": "iPhone 16 Amazon models latest"
            }
        }
        <|end|>
        
        <|channel|>final<|message|>
        I'll search for the iPhone 16 models on Amazon for you.
        <|end|>
        """

        print("✅ Testing harmony tool call parsing...")
        tool_calls = parse_harmony_tool_calls(mock, test_content)
        print(f"Parsed tool calls: {tool_calls}")

        final_content = extract_final_content(mock, test_content)
        print(f"Final content: {final_content}")

        if tool_calls:
            print("✅ Tool call parsing is working!")
            for tool_call in tool_calls:
                print(f"  - Tool: {tool_call.get('name', 'unknown')}")
                print(f"  - Args: {tool_call.get('args', {})}")
        else:
            print("❌ No tool calls parsed")

        return tool_calls

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    result = asyncio.run(test_tool_calling())
    if result:
        print("\n🎉 Tool calling test completed successfully!")
    else:
        print("\n💥 Tool calling test failed!")
