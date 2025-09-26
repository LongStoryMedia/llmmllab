#!/usr/bin/env python3
"""
Simple test of OpenAI GPT OSS tool calling graph functionality
"""

import sys
import os
import asyncio
import logging

# Add the paths for imports
sys.path.append("/app")
sys.path.append("/app/runner")
sys.path.append("/app/server")


async def test_graph_tool_calling():
    """Test tool calling by directly using the create_graph method"""

    try:
        # Set up logging to capture our debug messages
        logging.basicConfig(
            level=logging.DEBUG, format="%(levelname)s:%(name)s:%(message)s"
        )
        logger = logging.getLogger("graph-test")

        from runner.pipelines.txt2txt.openai_gpt_oss import OpenAiGptOssPipe
        from models.lang_chain_message import LangChainMessage
        from models.lang_graph_state import LangGraphState

        print("✅ Imports successful")

        # Create a minimal mock pipeline to test the graph creation
        class MockProfile:
            def __init__(self):
                self.parameters = type(
                    "obj",
                    (object,),
                    {
                        "temperature": 0.7,
                        "max_tokens": 1000,
                        "reasoning_effort": "medium",
                    },
                )()

        class MockModel:
            def __init__(self):
                self.model = "test-model"

        # Create pipeline instance with mock objects
        os.environ["ALLOW_MISSING_GGUF"] = "true"
        pipeline = OpenAiGptOssPipe.__new__(
            OpenAiGptOssPipe
        )  # Create without calling __init__

        # Manually set the required attributes from all parent classes
        pipeline.model = MockModel()
        pipeline.profile = MockProfile()
        pipeline._logger = logging.getLogger("openai-gpt-oss-test")
        pipeline._current_tools = None
        pipeline.llm = None

        # Add attributes from BaseChatPipeline
        pipeline.graph_cache = {}
        pipeline.allowed_return_types = (
            type(None),
        )  # ChatResponse would require import
        pipeline.default_return_type = None

        # Add attributes from BasePipelineCore
        pipeline.expected_return_type = type(None)

        # Add memory for LangGraph checkpointer
        from langgraph.checkpoint.memory import MemorySaver

        pipeline.memory = MemorySaver()

        # Add OpenAI GPT OSS specific attributes
        pipeline.context_manager = None  # Mock
        pipeline._reasoning_effort = "medium"
        pipeline.harmony_buffer = ""
        pipeline.current_channel = "final"
        pipeline.in_analysis_channel = False
        pipeline.analysis_complete = False
        pipeline.detected_channels = set()

        print("✅ Mock pipeline created")

        # Create a simple mock tool
        from langchain.tools import BaseTool
        from typing import Type
        from pydantic import BaseModel, Field

        class MockSearchInput(BaseModel):
            query: str = Field(description="Search query")

        class MockWebSearchTool(BaseTool):
            name: str = "web_search"
            description: str = "Mock web search tool for testing"
            args_schema: Type[BaseModel] = MockSearchInput

            def _run(self, query: str) -> str:
                return f"Mock search results for: {query}"

            async def _arun(self, query: str) -> str:
                return f"Mock search results for: {query}"

        tools = [MockWebSearchTool()]
        print(f"✅ Created {len(tools)} mock tools")

        # Test the graph creation (this should work without LLM)
        try:
            graph = pipeline.create_graph(tools)
            print("✅ Graph created successfully with tools!")

            # Create test state
            messages = [
                LangChainMessage(
                    role="user",
                    content="Please search for iPhone 16 on Amazon using web_search tool.",
                )
            ]

            initial_state = LangGraphState(
                messages=messages,
                user_input="Please search for iPhone 16 on Amazon using web_search tool.",
                current_iteration=0,
                max_iterations=5,
            )

            print("✅ Test state created")

            # Test the tools_condition function directly to see our debugging
            result = pipeline._debug_tools_condition(initial_state)
            print(f"✅ Tools condition result: {result}")

            # Now let's test the actual tool calling by simulating a response with tool calls
            print("\n🔍 Testing tool call parsing...")

            # Simulate harmony response with tool calls
            harmony_response = """
            <|channel|>analysis<|message|>
            The user wants to search for iPhone 16 models on Amazon. I should use the web_search tool.
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
            I'll search for iPhone 16 models on Amazon for you.
            <|end|>
            """

            # Test our parsing functions
            tool_calls = pipeline._parse_harmony_tool_calls(harmony_response)
            print(f"✅ Parsed {len(tool_calls)} tool calls from harmony response")

            if tool_calls:
                for i, tool_call in enumerate(tool_calls):
                    print(
                        f"  Tool {i+1}: {tool_call['name']} with args {tool_call['args']}"
                    )

            # Test final content extraction
            final_content = pipeline._extract_final_content(harmony_response)
            print(f"✅ Final content: {final_content[:100]}...")

            return True

        except Exception as graph_error:
            print(f"❌ Graph creation error: {graph_error}")
            import traceback

            traceback.print_exc()
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    result = asyncio.run(test_graph_tool_calling())
    if result:
        print("\n🎉 Graph tool calling test completed successfully!")
    else:
        print("\n💥 Graph tool calling test failed!")
