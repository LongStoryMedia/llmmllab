#!/usr/bin/env python3
"""
Test to verify that agents can see conversation history including tool results.
This will help diagnose why the infinite tool loop is still happening.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from composer.core.service import ComposerService
from composer.agents.chat_agent import ChatAgent
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from models.default_model_profiles import create_default_model_profile
from utils.logging_util import setup_logger

logger = setup_logger(__name__)


async def test_conversation_history():
    """Test if agent can see and use conversation history."""
    
    logger.info("🧪 Testing agent conversation history visibility")
    
    try:
        # Initialize composer service
        service = ComposerService()
        await service.initialize()
        
        # Create a simple chat agent with the problematic model
        model_profile = create_default_model_profile("llama-chat-summary-3_2-3b-q5-k-m")
        chat_agent = ChatAgent(
            "test_agent",
            user_id="test_user",
            conversation_id=None,
            model_profile=model_profile
        )
        
        # Create a conversation that includes previous search results
        messages = [
            HumanMessage(content="What are the latest AI developments in 2024?"),
            AIMessage(
                content="I'll search for the latest AI developments in 2024.",
                tool_calls=[{
                    "name": "web_search",
                    "args": {"query": "latest AI developments 2024"},
                    "id": "call_1",
                    "type": "tool_call"
                }]
            ),
            ToolMessage(
                content="Search results for 'latest AI developments 2024':\n\n1. GPT-4 Turbo released in March 2024\n2. Claude 3 family launched by Anthropic\n3. Major breakthroughs in multimodal AI\n4. Open source models like Llama 3 released\n5. Significant advances in AI safety research",
                tool_call_id="call_1"
            ),
            HumanMessage(content="Can you tell me more about AI safety developments specifically?")
        ]
        
        logger.info(f"📋 Testing with {len(messages)} messages in conversation history")
        
        # Test the agent's response
        response = await chat_agent.chat_completion_with_conversion(
            messages=messages,
            tools=None  # No tools to force text response
        )
        
        logger.info(f"🤖 Agent response: {response.content[:200]}...")
        
        # Check if response references the previous search results
        content = response.content.lower()
        if any(term in content for term in ["gpt-4 turbo", "claude 3", "llama 3", "previous", "mentioned", "above"]):
            logger.info("✅ SUCCESS: Agent referenced previous search results")
            return True
        else:
            logger.warning("⚠️ PROBLEM: Agent didn't reference previous search results")
            logger.info(f"Full response: {response.content}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False


async def test_with_tools():
    """Test with tools enabled to see if agent makes redundant calls."""
    
    logger.info("🧪 Testing agent behavior with tools enabled")
    
    try:
        # Initialize composer service
        service = ComposerService()
        await service.initialize()
        
        # Get tools from registry
        from composer.tools.registry import ToolRegistry
        tool_registry = ToolRegistry()
        tools = list(tool_registry.get_all_executable_tools().values())
        
        # Create a simple chat agent with the problematic model
        model_profile = create_default_model_profile("llama-chat-summary-3_2-3b-q5-k-m")
        chat_agent = ChatAgent(
            "test_agent",
            user_id="test_user", 
            conversation_id=None,
            model_profile=model_profile
        )
        
        # Simulate conversation with previous tool results
        messages = [
            HumanMessage(content="What are the latest AI safety developments?"),
            AIMessage(
                content="I'll search for current AI safety developments.",
                tool_calls=[{
                    "name": "web_search",
                    "args": {"query": "current AI safety developments"},
                    "id": "call_1",
                    "type": "tool_call"
                }]
            ),
            ToolMessage(
                content="Found 5 results about current AI safety developments:\n1. Constitutional AI research\n2. AI alignment progress\n3. Safety evaluations framework\n4. Responsible AI deployment guidelines\n5. International AI safety cooperation initiatives",
                tool_call_id="call_1"
            )
        ]
        
        logger.info(f"📋 Testing with tools and {len(messages)} messages")
        
        # Test agent response with tools available
        response = await chat_agent.chat_completion_with_conversion(
            messages=messages,
            tools=tools
        )
        
        # Check if agent makes redundant tool calls
        if hasattr(response, 'tool_calls') and response.tool_calls:
            for tc in response.tool_calls:
                if tc.get('name') == 'web_search' and 'safety' in str(tc.get('args', {})).lower():
                    logger.warning(f"⚠️ REDUNDANT: Agent making duplicate safety search: {tc}")
                    return False
            logger.info(f"✅ Agent made different tool calls: {[tc.get('name') for tc in response.tool_calls]}")
        else:
            logger.info("✅ Agent provided final answer without redundant tool calls")
        
        logger.info(f"🤖 Agent response: {response.content[:200]}...")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    async def run_tests():
        logger.info("🧪 Starting conversation history visibility tests")
        
        # Test 1: Can agent see conversation history?
        test1_result = await test_conversation_history()
        
        # Test 2: Does agent avoid redundant tool calls?
        test2_result = await test_with_tools()
        
        # Summary
        logger.info(f"📊 Test Results:")
        logger.info(f"  ✅ History visibility: {'PASS' if test1_result else 'FAIL'}")
        logger.info(f"  ✅ Redundancy avoidance: {'PASS' if test2_result else 'FAIL'}")
        
        if not (test1_result and test2_result):
            logger.error("❌ Some tests failed - infinite loop issue likely persists")
        else:
            logger.info("✅ All tests passed - conversation history working properly")
    
    asyncio.run(run_tests())