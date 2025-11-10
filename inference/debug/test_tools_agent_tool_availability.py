#!/usr/bin/env python3
"""
Debug script to test tools_agent tool availability and tool call generation.

This script specifically tests if tools are properly available to the tools_agent
subgraph and if the model can generate proper tool calls.
"""

import asyncio
import sys
from pathlib import Path

# Add inference directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from models import UserConfig, Message, MessageContent, MessageContentType, MessageRole, NodeMetadata
from models.default_configs import create_default_user_config
from models.default_model_profiles import DEFAULT_PROFILES
from runner.pipeline_factory import pipeline_factory
from composer.tools.registry import ToolRegistry
from composer.agents.chat_agent import ChatAgent  
from composer.graph.subgraphs.tools_agent import ToolsAgentSubgraph
from composer.graph.state import ToolsState
from utils.message_conversion import messages_to_lc_messages
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="ToolsAgentDebug")

async def test_tools_agent_tool_availability():
    """Test if tools_agent has access to tools and can generate tool calls."""
    try:
        logger.info("🔧 Starting tools_agent tool availability test")
        
        # Create default user config (don't need full Config for this test)
        user_config = create_default_user_config("test_tools_debug")
        logger.info(f"✅ User config created with web_search enabled: {user_config.web_search.enabled}")
        
        # Initialize tool registry with pipeline_factory (not user_config)  
        tool_registry = ToolRegistry(pipeline_factory)
        logger.info("✅ ToolRegistry initialized")
        
        # Check tools before agent creation
        all_tools = tool_registry.get_all_executable_tools()
        logger.info(f"📋 Available tools: {list(all_tools.keys()) if all_tools else 'None'}")
        if all_tools:
            for name, tool in all_tools.items():
                logger.info(f"   - {name}: {tool.description[:50]}...")
        else:
            logger.warning("❌ No tools available in registry!")
            return False
            
        # Initialize chat agent with proper arguments
        profile = list(DEFAULT_PROFILES.values())[0]  # Use first available profile
        node_metadata = NodeMetadata(
            node_name="tools_test_node", 
            node_id="test_001", 
            node_type="test"
        )
        chat_agent = ChatAgent(pipeline_factory, profile, node_metadata)
        logger.info("✅ ChatAgent initialized")
        
        # Create tools agent subgraph
        tools_agent = ToolsAgentSubgraph(tool_registry, chat_agent)
        logger.info("✅ ToolsAgentSubgraph initialized")
        
        # Test message requesting web search (user config already created above)
        
        # Create simple test message requesting web search
        test_message = Message(
            role=MessageRole.USER,
            content=[MessageContent(
                type=MessageContentType.TEXT,
                text="Search the web for information about LangGraph agents"
            )],
            conversation_id=999,
            thoughts=[],
            tool_calls=[],
            analyses=[]
        )
        
        # Create tools state
        tools_state = ToolsState(
            messages=messages_to_lc_messages([test_message]),
            user_id="test_tools_debug", 
            conversation_id=999,
            user_config=user_config,
            tool_call_count=0
        )
        
        logger.info("🎯 Testing _chat_agent_node directly")
        
        # Call the chat agent node directly to see if it can generate tool calls
        result_state = await tools_agent._chat_agent_node(tools_state)
        
        logger.info(f"📤 Chat agent returned {len(result_state.messages)} messages")
        
        if result_state.messages:
            last_message = result_state.messages[-1]
            logger.info(f"📝 Last message type: {type(last_message)}")
            logger.info(f"📝 Last message content preview: {str(last_message)[:200]}...")
            
            # Check if the message has tool calls
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                logger.info(f"🔧 Found {len(last_message.tool_calls)} tool calls!")
                for i, tool_call in enumerate(last_message.tool_calls):
                    logger.info(f"   Tool Call {i+1}: {tool_call}")
                return True
            else:
                logger.warning("❌ No tool calls found in model response")
                logger.info(f"🔍 Message attributes: {dir(last_message)}")
                
                # Check message content for indication of tool intent
                content = str(getattr(last_message, 'content', ''))
                if 'search' in content.lower():
                    logger.warning("🤔 Model mentions search but didn't generate tool call")
                    
        logger.warning("❌ Model did not generate expected tool calls")
        return False
        
    except Exception as e:
        logger.error(f"💥 Error testing tools_agent: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    result = asyncio.run(test_tools_agent_tool_availability())
    if result:
        print("\n✅ SUCCESS: Tools are available and model can generate tool calls")
        exit(0)
    else:
        print("\n❌ FAILED: Tools not available or model cannot generate tool calls") 
        exit(1)