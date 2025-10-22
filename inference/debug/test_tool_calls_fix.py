#!/usr/bin/env python3
"""
Test the tool calls parsing fix in BaseAgent streaming.
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import Message, MessageRole, MessageContent, MessageContentType
from utils.logging import llmmllogger

logger = llmmllogger.logger.bind(component="ToolCallsFixTest")

def create_test_messages():
    """Create test messages requesting AI search."""
    return [
        Message(
            role=MessageRole.USER,
            content=[
                MessageContent(
                    type=MessageContentType.TEXT,
                    text="I need current information about the latest developments in artificial intelligence in 2024. Specifically, I'm interested in: 1. Major AI model releases in 2024, 2. Recent breakthroughs in AI research, 3. Current AI safety developments. Please search for the most recent information and provide a comprehensive summary."
                )
            ]
        )
    ]

async def test_tool_calls_parsing():
    """Test that streaming chunks now properly parse tool calls."""
    try:
        # Import composer service and workflow components
        from composer.service import ComposerService
        from composer.config import config
        
        logger.info("🧪 Testing tool calls parsing fix in streaming")
        
        # Create composer service
        service = ComposerService(config)
        
        # Create test state
        messages = create_test_messages()
        
        # Execute workflow with streaming
        logger.info("📡 Executing streaming workflow to test tool calls")
        
        response_count = 0
        tool_call_chunks = 0
        message_chunks = 0
        
        async for response in service.execute_workflow_stream(
            messages=messages,
            user_id="test_tool_calls_fix",
            conversation_id=12345
        ):
            response_count += 1
            
            # Check if this chunk has a message
            if hasattr(response, 'message') and response.message:
                message_chunks += 1
                
                # Check if this chunk has tool calls
                if response.message.tool_calls:
                    tool_call_chunks += 1
                    logger.info(
                        f"🔧 Found tool calls in chunk {response_count}:",
                        tool_calls=len(response.message.tool_calls),
                        tools=[tc.get('name', 'unnamed') for tc in response.message.tool_calls]
                    )
                
                # Check content
                content_text = ""
                if response.message.content:
                    for content in response.message.content:
                        if content.type == MessageContentType.TEXT:
                            content_text += content.text
                
                if content_text.strip():
                    logger.debug(f"📝 Chunk {response_count} content: {content_text[:200]}...")
        
        logger.info(
            "✅ Streaming test completed",
            total_responses=response_count,
            message_chunks=message_chunks,
            tool_call_chunks=tool_call_chunks
        )
        
        # Success if we found tool calls in streaming chunks
        if tool_call_chunks > 0:
            logger.info("🎉 SUCCESS: Tool calls detected in streaming chunks!")
            return True
        else:
            logger.warning("⚠️  WARNING: No tool calls found in streaming chunks")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_tool_calls_parsing())