#!/usr/bin/env python3

"""Test to debug streaming chunks format in ChatAgent."""

import asyncio
from models.user_config_defaults import create_default_user_config
from models.message import Message, MessageContentType, MessageContent
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="test_chat_agent_streaming")

async def test_chat_agent_streaming():
    """Test ChatAgent streaming to debug chunk format."""
    logger.info("🧪 Testing ChatAgent streaming to debug chunk format...")
    
    try:
        from composer.agents.chat_agent import ChatAgent
        
        # Create a ChatAgent instance
        chat_agent = ChatAgent(
            user_id="test_user",
            conversation_id=123,
            user_config=create_default_user_config("test_user"),
            node_id="debug_chat",
            node_name="debug_chat",
            node_type="debug"
        )
        
        # Create a simple test message
        test_message = Message(
            role="user",
            content=[
                MessageContent(
                    type=MessageContentType.TEXT,
                    text="Please search for 'test search query' using web search tool."
                )
            ],
            conversation_id=123
        )
        
        # Get available tools to pass to chat agent
        from composer.tools.registry import ToolRegistry
        tool_registry = ToolRegistry()
        tools_dict = tool_registry.get_all_executable_tools()
        tools_list = list(tools_dict.values()) if tools_dict else []
        
        logger.info(f"Available tools: {len(tools_list)}")
        
        # Test streaming completion
        logger.info("Testing streaming chat completion...")
        
        result = await chat_agent.chat_completion(
            messages=[test_message],
            tools=tools_list,
            stream=True
        )
        
        logger.info(f"Streaming result: {result}")
        
        if result and result.message:
            logger.info(f"Final message role: {result.message.role}")
            logger.info(f"Final message content length: {len(result.message.content) if result.message.content else 0}")
            
            if hasattr(result.message, 'tool_calls') and result.message.tool_calls:
                logger.info(f"Tool calls found: {len(result.message.tool_calls)}")
                for i, tc in enumerate(result.message.tool_calls):
                    logger.info(f"Tool call {i}: {type(tc)} - {tc}")
            else:
                logger.info("No tool calls in final message")
        
    except Exception as e:
        logger.error(f"❌ ChatAgent streaming test failed: {e}", exc_info=True)
        
        # Let's check if the error matches our expected pattern
        if "'ChatCompletionMessageFunctionToolCall' object is not subscriptable" in str(e):
            logger.error("🎯 Found the exact error we're debugging!")

if __name__ == "__main__":
    asyncio.run(test_chat_agent_streaming())
    logger.info("🎉 ChatAgent streaming test completed")