#!/usr/bin/env python3
"""
Test script for the redesigned tools agent subgraph with proper LangGraph agent pattern.

This script tests the complete agent workflow with chat_agent + tool_node cycling.
"""

import asyncio
from models import LangChainMessage
from composer.graph.subgraphs.tools_agent import tools_agent_subgraph
from composer.graph.state import WorkflowState
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="ToolsAgentTest")


async def test_tools_agent_subgraph():
    """Test the tools agent subgraph with a simple query that should trigger web search."""
    try:
        logger.info("Starting tools agent subgraph test")
        
        # Create a minimal WorkflowState for testing
        test_state = WorkflowState()
        test_state.user_id = "test_user"
        test_state.conversation_id = 123
        test_state.current_date = "2024-10-21"
        test_state.messages = [
            LangChainMessage(
                content="What is the current weather in San Francisco? Please search for recent information.",
                type="human"
            )
        ]
        
        # Set up minimal user config for web search
        from models.default_configs import create_default_user_config
        test_state.user_config = create_default_user_config("test_user")
        
        logger.info("Created test state with web search query")
        
        # Execute the subgraph
        result_command = await tools_agent_subgraph.execute(test_state)
        
        logger.info(f"Subgraph completed, result type: {type(result_command)}")
        
        if hasattr(result_command, 'update') and result_command.update:
            logger.info(f"State updates: {list(result_command.update.keys())}")
            
            # Check for new messages
            if 'messages' in result_command.update:
                new_messages = result_command.update['messages']
                logger.info(f"Added {len(new_messages)} new messages")
                
                for i, msg in enumerate(new_messages):
                    logger.info(f"Message {i+1}: type={type(msg).__name__}, content_length={len(str(msg.content))}")
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        logger.info(f"  Tool calls: {[call['name'] for call in msg.tool_calls]}")
        else:
            logger.warning("No state updates in result")
        
        logger.info("✅ Tools agent subgraph test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Tools agent subgraph test failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    asyncio.run(test_tools_agent_subgraph())