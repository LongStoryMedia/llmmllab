#!/usr/bin/env python3
"""
Test the tool calls parsing fix in BaseAgent streaming with direct BaseAgent testing.
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    Message, MessageRole, MessageContent, MessageContentType,
    ModelProfile, Model, NodeMetadata, PipelinePriority
)
from utils.logging import llmmllogger
from runner.pipeline_factory import pipeline_factory

logger = llmmllogger.logger.bind(component="DirectToolCallsTest")

def create_test_messages():
    """Create test messages requesting AI search with tools."""
    return [
        Message(
            role=MessageRole.USER,
            content=[
                MessageContent(
                    type=MessageContentType.TEXT,
                    text="I need current information about the latest developments in artificial intelligence in 2024. Search for major AI model releases in 2024."
                )
            ]
        )
    ]

def create_test_profile():
    """Get an existing test model profile from config."""
    from db.config_storage import get_config
    
    config = get_config()
    # Use the first available model profile
    for profile_id in config.models.model_profiles:
        profile = config.models.model_profiles[profile_id]
        if profile:
            # Update system prompt for tools
            profile.system_prompt = "You are a helpful AI assistant with access to web search tools. When the user asks for current information, use the web_search tool to find the latest information."
            return profile
    raise ValueError("No model profiles found in config")

def create_test_node_metadata():
    """Create test node metadata."""
    return NodeMetadata(
        node_name="test_tool_calls_node",
        node_type="chat_agent",
        user_id="test_user",
        conversation_id=12345
    )

async def test_base_agent_tool_calls():
    """Test BaseAgent streaming with tool calls parsing."""
    try:
        # Import BaseAgent
        from composer.agents.base_agent import BaseAgent
        
        logger.info("🧪 Testing BaseAgent tool calls parsing fix")
        
        # Create test components
        profile = create_test_profile()
        node_metadata = create_test_node_metadata()
        messages = create_test_messages()
        
        # Create BaseAgent instance
        agent = BaseAgent(
            pipeline_factory=pipeline_factory,
            profile=profile,
            node_metadata=node_metadata
        )
        
        logger.info("🤖 Created BaseAgent, starting streaming test")
        
        # Test streaming with mock tools
        from langchain.tools import tool
        
        @tool
        def web_search(query: str) -> str:
            """Search the web for information."""
            return f"Mock search results for: {query}"
        
        tools = [web_search]
        
        # Stream with tools to trigger tool calls
        chunk_count = 0
        tool_call_chunks = 0
        text_chunks = 0
        
        async for chunk in agent.stream(
            messages=messages,
            tools=tools,
            priority=PipelinePriority.MEDIUM
        ):
            chunk_count += 1
            
            if hasattr(chunk, 'message') and chunk.message:
                # Check for tool calls
                if chunk.message.tool_calls:
                    tool_call_chunks += 1
                    logger.info(
                        f"🔧 Found tool calls in chunk {chunk_count}:",
                        tool_calls=len(chunk.message.tool_calls),
                        tools=[tc.get('name', 'unnamed') for tc in chunk.message.tool_calls]
                    )
                
                # Check for text content
                if chunk.message.content:
                    for content in chunk.message.content:
                        if content.type == MessageContentType.TEXT and content.text.strip():
                            text_chunks += 1
                            logger.debug(f"📝 Text chunk {chunk_count}: {content.text[:100]}...")
                            break
        
        logger.info(
            "✅ BaseAgent streaming test completed",
            total_chunks=chunk_count,
            tool_call_chunks=tool_call_chunks,
            text_chunks=text_chunks
        )
        
        # Success if we found any chunks with tool calls
        if tool_call_chunks > 0:
            logger.info("🎉 SUCCESS: Tool calls detected in BaseAgent streaming chunks!")
            return True
        else:
            logger.warning("⚠️  WARNING: No tool calls found in BaseAgent streaming chunks")
            return False
            
    except Exception as e:
        logger.error(f"❌ BaseAgent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(test_base_agent_tool_calls())