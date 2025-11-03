#!/usr/bin/env python3
"""
Test script to investigate how LangChain ToolNode handles state injection for ToolRuntime.

This script tests different approaches to fix the "Missing user_id in state" issue.
"""

import asyncio
from typing import Dict, Any, List

from langchain_core.tools import tool
from langchain.tools import ToolRuntime, BaseTool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

# Mock the ToolsState structure
class ToolsState(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def __getitem__(self, key):
        return super().get(key)
        
    def get(self, key, default=None):
        return super().get(key, default)


# Test tool that uses ToolRuntime
@tool
async def test_memory_tool(
    query: str,
    runtime: ToolRuntime,
) -> str:
    """Test memory tool to debug state injection."""
    print(f"=== Tool Runtime Debug ===")
    print(f"Tool called with query: {query}")
    print(f"Runtime state type: {type(runtime.state)}")
    print(f"Runtime state keys: {list(runtime.state.keys()) if hasattr(runtime.state, 'keys') else 'No keys method'}")
    print(f"Runtime state content: {runtime.state}")
    
    # Check for user_id specifically
    user_id = runtime.state.get('user_id')
    print(f"User ID from runtime: {user_id} (type: {type(user_id).__name__})")
    
    return f"Tool executed successfully. User ID: {user_id}"


async def test_standard_tool_node():
    """Test how standard LangChain ToolNode handles state injection."""
    print("\n=== Testing Standard ToolNode ===")
    
    # Create ToolNode directly and test its input/output
    tool_node = ToolNode([test_memory_tool])
    
    # Test with a message that has tool calls (what ToolNode expects)
    tool_call_message = AIMessage(
        content="Calling test tool",
        tool_calls=[{
            "name": "test_memory_tool",
            "args": {"query": "test query"},
            "id": "test_call_123",
            "type": "tool_call"
        }]
    )
    
    # This is what ToolNode expects - messages in the input
    tool_node_input = {
        "messages": [tool_call_message],
        "user_id": "test_user_123", 
        "conversation_id": 456,
        "user_config": {"memory": {"enabled": True}},
        "current_date": "2025-11-03"
    }
    
    print(f"ToolNode input keys: {list(tool_node_input.keys())}")
    print(f"ToolNode input user_id: {tool_node_input.get('user_id')}")
    
    # Execute ToolNode directly
    try:
        result = await tool_node.ainvoke(tool_node_input)
        print(f"ToolNode result: {result}")
        print(f"ToolNode result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
        return result
    except Exception as e:
        print(f"ToolNode execution failed: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_custom_tool_node():
    """Test a custom ToolNode implementation that properly injects state."""
    print("\n=== Testing Custom ToolNode with State Injection ===")
    
    class StateInjectedToolNode:
        """Custom ToolNode that properly injects full state into ToolRuntime."""
        
        def __init__(self, tools: List[BaseTool]):
            self.tools = {tool.name: tool for tool in tools}
        
        async def __call__(self, state: ToolsState) -> ToolsState:
            """Execute tools with proper state injection."""
            messages = state.get("messages", [])
            
            if not messages:
                return state
            
            last_message = messages[-1]
            
            # Check if last message has tool calls
            if not (hasattr(last_message, "tool_calls") and last_message.tool_calls):
                return state
            
            # Execute each tool call with proper state injection
            tool_messages = []
            
            for tool_call in last_message.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_call_id = tool_call["id"]
                
                if tool_name in self.tools:
                    tool = self.tools[tool_name]
                    
                    # Create mock ToolRuntime with full state
                    class MockToolRuntime:
                        def __init__(self, state_dict, call_id):
                            self.state = state_dict
                            self.tool_call_id = call_id
                    
                    runtime = MockToolRuntime(state, tool_call_id)
                    
                    try:
                        # Call tool with runtime injection
                        if asyncio.iscoroutinefunction(tool._arun):
                            result = await tool._arun(runtime=runtime, **tool_args)
                        else:
                            result = tool._run(runtime=runtime, **tool_args)
                        
                        tool_messages.append(ToolMessage(
                            content=result,
                            tool_call_id=tool_call_id,
                            name=tool_name
                        ))
                        
                    except Exception as e:
                        print(f"Tool execution error: {e}")
                        tool_messages.append(ToolMessage(
                            content=f"Error: {str(e)}",
                            tool_call_id=tool_call_id,
                            name=tool_name
                        ))
            
            # Return updated state with tool messages
            updated_messages = messages + tool_messages
            return {**state, "messages": updated_messages}
    
    # Create graph with custom tool node
    builder = StateGraph(ToolsState)
    
    async def chat_node(state: ToolsState) -> ToolsState:
        print(f"Custom chat node received state keys: {list(state.keys())}")
        
        # Create a tool call message
        tool_call_message = AIMessage(
            content="Calling test tool",
            tool_calls=[{
                "name": "test_memory_tool", 
                "args": {"query": "test query"},
                "id": "test_call_456",
                "type": "tool_call"
            }]
        )
        
        messages = state.get("messages", [])
        messages.append(tool_call_message)
        
        return {**state, "messages": messages}
    
    # Use custom tool node
    custom_tool_node = StateInjectedToolNode([test_memory_tool])
    
    builder.add_node("chat", chat_node)
    builder.add_node("tools", custom_tool_node)
    
    builder.add_conditional_edges(
        "chat",
        tools_condition,
        {
            "tools": "tools", 
            "__end__": END,
        },
    )
    
    builder.add_edge("tools", "chat")
    builder.add_edge(START, "chat")
    
    graph = builder.compile()
    
    # Test with proper state including user_id
    test_state = ToolsState(
        messages=[HumanMessage(content="Hello")],
        user_id="test_user_456",
        conversation_id=789,
        user_config={"memory": {"enabled": True}},
        current_date="2025-11-03"
    )
    
    print(f"Custom input state keys: {list(test_state.keys())}")
    print(f"Custom input user_id: {test_state.get('user_id')}")
    
    # Execute the graph
    result = await graph.ainvoke(test_state)
    
    print(f"Custom final result keys: {list(result.keys())}")
    print(f"Custom final messages count: {len(result.get('messages', []))}")
    
    return result


async def main():
    """Run all tests to debug ToolRuntime state injection."""
    print("Testing LangChain ToolNode state injection patterns...")
    
    try:
        # Test 1: Standard LangChain ToolNode
        await test_standard_tool_node()
        
        # Test 2: Custom ToolNode with proper state injection
        await test_custom_tool_node()
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())