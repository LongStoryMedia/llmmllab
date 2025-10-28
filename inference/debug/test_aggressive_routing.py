#!/usr/bin/env python3
"""
Quick test script for ultra-aggressive routing fixes.
Skip all the workflow setup and go directly to tools execution.
"""
import asyncio
import sys
import os

# Add the current directory to the path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from composer.graph.subgraphs.tools_agent import ToolsAgentSubgraph
from composer.agents.chat_agent import ChatAgent
from composer.tools.registry import ToolRegistry

async def test_aggressive_routing():
    """Test the ultra-aggressive routing fixes directly."""
    print("🧪 Testing ultra-aggressive routing fixes...")
    
    # Create minimal dependencies
    chat_agent = ChatAgent(
        model_name="llama-chat-summary-3_2-3b-q5-k-m",
        conversation_id=None,
        user_id="test_user"
    )
    
    tool_registry = ToolRegistry()
    
    # Create the tools agent subgraph
    tools_subgraph = ToolsAgentSubgraph(
        chat_agent=chat_agent,
        tool_registry=tool_registry
    )
    
    # Create test state with 2 tool executions (should trigger ultra-aggressive END)
    test_messages = [
        HumanMessage(content="I need current information about AI developments in 2024."),
        AIMessage(content='{"name": "web_search", "parameters": {"query": "AI developments 2024"}}', 
                 tool_calls=[{"name": "web_search", "args": {"query": "AI developments 2024"}, "id": "call_1", "type": "tool_call"}]),
        ToolMessage(content="Search results: AI has advanced significantly in 2024...", name="web_search", tool_call_id="call_1"),
        AIMessage(content='{"name": "web_search", "parameters": {"query": "AI safety 2024"}}',
                 tool_calls=[{"name": "web_search", "args": {"query": "AI safety 2024"}, "id": "call_2", "type": "tool_call"}]),
        ToolMessage(content="Search results: AI safety measures improved in 2024...", name="web_search", tool_call_id="call_2"),
        AIMessage(content='{"name": "web_search", "parameters": {"query": "more searches"}}',
                 tool_calls=[{"name": "web_search", "args": {"query": "more searches"}, "id": "call_3", "type": "tool_call"}])
    ]
    
    test_state = {
        "messages": test_messages
    }
    
    # Test the should_continue_after_tools function
    # This should return END because we have 2+ tool executions
    result = await tools_subgraph.graph.ainvoke(test_state)
    
    print(f"✅ Test completed. Result: {result}")
    return result

if __name__ == "__main__":
    asyncio.run(test_aggressive_routing())