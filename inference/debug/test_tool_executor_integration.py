#!/usr/bin/env python3
"""
Simple test to verify the ToolExecutorNode with subgraph integration works correctly.
"""

import asyncio
from langchain_core.messages import HumanMessage, AIMessage
from composer.graph.state import WorkflowState
from models import LangChainMessage


async def test_tool_executor_node():
    """Test the ToolExecutorNode with subgraph integration."""
    
    print("🧪 Testing ToolExecutorNode with subgraph...")
    
    try:
        from composer.nodes.tools.tool_executor import ToolExecutorNode
        from composer.tools.registry import ToolRegistry
        from runner.pipeline_factory import pipeline_factory
        
        # Create tool registry and executor node
        registry = ToolRegistry(pipeline_factory)
        executor_node = ToolExecutorNode(registry)
        
        print("✅ ToolExecutorNode created successfully")
        
        # Create test state with tool calls
        state = WorkflowState()
        state.user_id = "test_user_executor"
        state.conversation_id = 997
        state.current_date = "2025-01-01"
        state.messages = [
            LangChainMessage(
                content="What's the current date?",
                type="human"
            ),
            LangChainMessage(
                content="I'll get the current date for you.",
                type="ai",
                tool_calls=[{
                    "id": "test_call_executor",
                    "name": "get_current_date", 
                    "args": {}
                }]
            )
        ]
        state.things_to_remember = []
        state.web_search_results = []
        state.tool_calls = []
        
        # Remove the manual tool_calls setting since it's now in the message
        
        print("✅ Test state with tool calls created")
        
        # Execute the tool executor node
        result_state = await executor_node(state)
        
        print("✅ ToolExecutorNode execution completed")
        print(f"   Messages count: {len(result_state.messages)}")
        
        # Check if tool results were added
        if len(result_state.messages) > len(state.messages):
            print("✅ Tool execution added new messages")
            last_message = result_state.messages[-1]
            print(f"   Last message type: {last_message.type}")
            print(f"   Content: {last_message.content[:100]}...")
        else:
            print("⚠️ No new messages added by tool execution")
        
        # Check things_to_remember
        if result_state.things_to_remember:
            print(f"✅ Tool results added to things_to_remember: {len(result_state.things_to_remember)}")
        
        return True
        
    except Exception as e:
        print(f"❌ ToolExecutorNode test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_subgraph_command_pattern():
    """Test the subgraph command pattern directly."""
    
    print("\n🧪 Testing subgraph Command pattern...")
    
    try:
        from composer.graph.subgraphs import tools_agent_subgraph
        from composer.graph.state import WorkflowState
        from models import LangChainMessage
        
        # Create test WorkflowState
        state = WorkflowState()
        state.user_id = "test_user_command" 
        state.conversation_id = 996
        state.current_date = "2025-01-01"
        state.messages = [
            LangChainMessage(
                content="What's the current date?",
                type="human"
            ),
            LangChainMessage(
                content="I'll get the current date for you.",
                type="ai",
                tool_calls=[{
                    "id": "test_call_command",
                    "name": "get_current_date",
                    "args": {}
                }]
            )
        ]
        state.things_to_remember = []
        state.web_search_results = []
        state.tool_calls = []
        
        print("✅ WorkflowState created for command test")
        
        # Test the command execution
        command = await tools_agent_subgraph.execute(state)
        
        print("✅ Subgraph execute() completed")
        print(f"   Command type: {type(command)}")
        
        if command and hasattr(command, 'update'):
            print(f"   Command updates: {list(command.update.keys()) if command.update else 'None'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Command pattern test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run the integration tests."""
    print("🚀 Starting ToolExecutorNode integration tests...\n")
    
    success = True
    
    # Test the command pattern directly
    if not await test_subgraph_command_pattern():
        success = False
    
    # Test the tool executor node integration
    if not await test_tool_executor_node():
        success = False
    
    if success:
        print("\n✅ All integration tests passed!")
    else:
        print("\n❌ Some tests failed")
    
    return success


if __name__ == "__main__":
    asyncio.run(main())