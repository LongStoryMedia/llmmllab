"""Test what tool format is being sent to Qwen3-VL"""

import os
import sys
from typing import List
import json

from models import (
    Message,
    MessageContent,
    MessageContentType,
    MessageRole,
    NodeMetadata,
)
from models.default_model_profiles import DEFAULT_TEXT_TO_TEXT_MODEL, DEFAULT_PROFILES
from runner.pipeline_factory import pipeline_factory
from composer.agents import ChatAgent
from composer.tools.registry import ToolRegistry
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="test_qwen3vl_tool_format")


def get_profile():
    model_name = DEFAULT_TEXT_TO_TEXT_MODEL
    profile = DEFAULT_PROFILES.get("primary")
    if getattr(profile, "model_name", None) != model_name:
        for _, p in DEFAULT_PROFILES.items():
            if getattr(p, "model_name", None) == model_name:
                profile = p
                break
    if profile is None:
        print(f"[error] No default model profile found for {model_name}")
        sys.exit(1)
    return profile


async def test_tool_format() -> None:
    os.environ.setdefault("LOG_LEVEL", "INFO")

    logger.info("🔧 Testing Qwen3-VL tool format")

    # Get the pipeline and profile
    profile = get_profile()
    logger.info(f"📊 Using model profile: {profile.model_name}")

    # Create pipeline directly to test tool format
    pipeline = await pipeline_factory.get_pipeline(profile)
    logger.info(f"🔧 Pipeline type: {type(pipeline).__name__}")

    # Get tools from registry
    registry = ToolRegistry(pipeline_factory)
    executable_tools = registry.get_all_executable_tools()
    tools_list = list(executable_tools.values())
    
    logger.info(f"🛠️ Available tools: {[tool.name for tool in tools_list]}")
    
    # Test the tool conversion
    converted_tools = pipeline._convert_tools_to_simple_format(tools_list)
    
    print("\n" + "="*80)
    print("CONVERTED TOOLS FORMAT FOR QWEN3-VL")
    print("="*80)
    
    if converted_tools:
        for i, tool in enumerate(converted_tools):
            print(f"\nTool {i+1}:")
            print(json.dumps(tool, indent=2))
    else:
        print("No tools converted!")
    
    print("\n" + "="*80)
    
    # Test with a simple text message to see what gets generated
    messages = [
        Message(
            role=MessageRole.USER,
            content=[
                MessageContent(
                    type=MessageContentType.TEXT,
                    text="Please search the web for 'Python programming tips' and summarize the results."
                )
            ]
        )
    ]
    
    logger.info("🎯 Testing with simple tool-requiring message...")
    
    # Create agent to test tool usage
    agent = ChatAgent(
        pipeline_factory,
        profile,
        NodeMetadata(node_name="test_node", node_id="test_001", node_type="test"),
    )
    
    # Test streaming to see if tools are called
    logger.info("🔄 Starting agent stream test...")
    chunk_count = 0
    has_tool_calls = False
    
    try:
        async for chunk in agent.stream(messages, tools=tools_list):
            chunk_count += 1
            if chunk.message and chunk.message.content:
                for content in chunk.message.content:
                    if content.type == MessageContentType.TOOL_CALL:
                        has_tool_calls = True
                        logger.info(f"✅ Found tool call: {content.text}")
                        print(f"TOOL CALL DETECTED: {content.text}")
                    elif content.type == MessageContentType.TEXT and content.text:
                        # Print first few chars to see what's being generated
                        if chunk_count <= 5:  # Only show first few chunks
                            print(f"Chunk {chunk_count}: {content.text[:50]}...")
    
    except Exception as e:
        logger.error(f"Error during streaming: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n📊 Results: {chunk_count} chunks processed, tool calls detected: {has_tool_calls}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_tool_format())