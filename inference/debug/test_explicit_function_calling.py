#!/usr/bin/env python3
"""
Test simple function calling with explicit instruction.
"""
import asyncio
import sys
import os
from pathlib import Path

# Add the inference directory to the Python path for imports
sys.path.append(str(Path(__file__).parent.parent))

from composer.agents.chat_agent import ChatAgent
from runner.pipeline_factory import PipelineFactory
from models import ModelProfile, ModelProvider, ModelTask, NodeMetadata, PipelinePriority
from composer.tools.registry import ToolRegistry
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="test_explicit_function_calling")

async def test_explicit_function_calling():
    """Test with explicit function calling instruction."""
    logger.info("🧪 Starting explicit function calling test")
    
    try:
        # Set up model profile
        profile = ModelProfile(
            model_name='qwen3-vl-30b-a3b-thinking',
            provider=ModelProvider.LLAMA_CPP,
            path='/models/qwen3-vl-30b-a3b/qwen3-vl-30b-a3b-q4-k-m.gguf',
            projector_path='/models/qwen3-vl-30b-a3b/mmproj.gguf',
            task=ModelTask.VISIONTEXTTOTEXT,
            size=0,
            original_ctx=65536
        )
        
        # Create pipeline factory and agent
        pipeline_factory = PipelineFactory()
        node_metadata = NodeMetadata(
            node_id="test_001",
            node_name="test_node", 
            node_type="test",
            user_id="test_user",
            conversation_id=123
        )
        
        chat_agent = ChatAgent(
            pipeline_factory=pipeline_factory,
            profile=profile,
            node_metadata=node_metadata,
            priority=PipelinePriority.MEDIUM
        )
        
        # Set up tools
        tool_registry = ToolRegistry()
        tools = tool_registry.get_all_executable_tools()
        tools_list = list(tools.values()) if tools else None
        
        logger.info(f"📋 Available tools: {len(tools_list) if tools_list else 0}")
        
        # Test with very explicit function calling instruction
        from models import Message, MessageRole, MessageContent, MessageContentType
        
        test_message = Message(
            role=MessageRole.USER,
            conversation_id=123,
            content=[
                MessageContent(
                    type=MessageContentType.TEXT,
                    text="You MUST use the web_search function to search for 'weather in San Francisco'. Do not respond with text, only call the function."
                )
            ]
        )
        
        logger.info("📤 Sending explicit function calling message...")
        
        # Call with tools
        response = await chat_agent.chat_completion(
            messages=[test_message],
            tools=tools_list,
            stream=False,
        )
        
        if response and response.message:
            logger.info(f"📨 Response: {response.message.content}")
            
            # Check for tool calls
            from utils.tool_call_extraction import extract_tool_calls_from_message_content
            tool_calls = extract_tool_calls_from_message_content(response.message.content)
            
            if tool_calls:
                logger.info(f"✅ Found {len(tool_calls)} tool calls!")
                for i, tc in enumerate(tool_calls):
                    logger.info(f"🔧 Tool call {i}: {tc}")
            else:
                logger.warning("❌ No tool calls found in response")
        else:
            logger.error("❌ No response received")
            
    except Exception as e:
        logger.error(f"❌ Explicit function calling test failed: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(test_explicit_function_calling())