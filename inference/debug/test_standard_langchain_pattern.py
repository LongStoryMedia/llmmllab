#!/usr/bin/env python3
"""
Test standard LangChain ChatOpenAI tool calling pattern.

This validates that our simplified implementation follows the 
exact LangChain documentation patterns without any custom logic.
"""

import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from typing import List, TypedDict


@tool
def simple_test_tool(query: str) -> str:
    """A simple test tool that returns a greeting."""
    return f"Hello, {query}! This is a test response."


class SimpleState(TypedDict):
    messages: List


def create_standard_agent():
    """Create a standard LangChain agent following documentation exactly."""
    # Create ChatOpenAI with test base URL
    chat_model = ChatOpenAI(
        base_url="http://localhost:8003/v1",  # Use our llama.cpp server
        model="local-model",
        temperature=0.7
    )
    
    # Bind tools
    tools = [simple_test_tool]
    chat_model_with_tools = chat_model.bind_tools(tools)
    
    # Create agent node
    def agent_node(state: SimpleState) -> SimpleState:
        response = chat_model_with_tools.invoke(state["messages"])
        state["messages"].append(response)
        return state
    
    # Create tool node
    tool_node = ToolNode(tools)
    
    # Build graph
    builder = StateGraph(SimpleState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)
    
    # Add routing
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")
    builder.add_edge(START, "agent")
    
    return builder.compile()


async def test_standard_pattern():
    """Test the standard LangChain pattern."""
    print("🧪 Testing standard LangChain agent pattern...")
    
    try:
        # Create agent
        agent = create_standard_agent()
        
        # Test input
        initial_state = {
            "messages": [
                HumanMessage(content="Please greet me using the simple test tool.")
            ]
        }
        
        print("📤 Invoking standard agent...")
        result = await agent.ainvoke(initial_state)
        
        print(f"📨 Agent completed with {len(result['messages'])} messages")
        for i, msg in enumerate(result["messages"]):
            print(f"  {i}: {type(msg).__name__} - {str(msg)[:100]}...")
            
        print("✅ Standard LangChain pattern test completed")
        
    except Exception as e:
        print(f"❌ Standard pattern test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_standard_pattern())