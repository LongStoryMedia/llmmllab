#!/usr/bin/env python3
"""
Debug routing decisions in the composer workflow.
"""

import asyncio
import logging
from pathlib import Path

# Import composer components
from composer.core.service import ComposerService
from models.workflow_state import WorkflowState
from models.lang_chain_message import LangChainMessage


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_routing_debug():
    """Test routing decisions with simulated states."""
    print("🔍 Testing routing decisions...")
    
    # Create composer service
    service = ComposerService()
    await service.initialize()
    
    # Build workflow for test user
    test_user_id = "debug_routing_test"
    workflow = await service.build_workflow(test_user_id)
    
    # Test scenario 1: Initial message (should go to tool_executor)
    print("\n--- Test 1: AI message with tool calls ---")
    initial_state = WorkflowState(
        messages=[
            LangChainMessage(content="Search query", type="human"),
            LangChainMessage(
                content="I'll search for information", 
                type="ai", 
                tool_calls=[{"name": "web_search", "args": {"query": "test"}}]
            )
        ],
        user_id=test_user_id
    )
    
    # Import the routing function (we need to test it directly)
    from composer.graph.builder import GraphBuilder
    builder = GraphBuilder(service.logger)
    
    # We need to extract the routing function from the builder
    # This is tricky since it's defined inside build_workflow
    # Let's simulate the routing logic instead
    
    def should_execute_tools_debug(state: WorkflowState):
        """Debug version of should_execute_tools"""
        print(f"🔍 Messages in state: {len(state.messages)}")
        
        if not state.messages:
            print("❌ No messages - returning memory_creation")
            return "memory_creation"

        # Check if we have any tool results in recent messages
        has_recent_tool_results = False
        for i, msg in enumerate(state.messages[-10:]):
            print(f"  Message {i}: type={getattr(msg, 'type', 'unknown')}, class={type(msg).__name__}")
            if hasattr(msg, '__class__') and 'ToolMessage' in str(type(msg)):
                print(f"    ✅ Found ToolMessage: {type(msg)}")
                has_recent_tool_results = True
                break
            elif hasattr(msg, 'type') and str(getattr(msg, 'type', '')).lower() == 'tool':
                print(f"    ✅ Found tool type message")
                has_recent_tool_results = True
                break

        if has_recent_tool_results:
            print("✅ Found tool results - returning memory_creation")
            return "memory_creation"

        last_message = state.messages[-1]
        print(f"🔍 Last message: type={getattr(last_message, 'type', 'unknown')}")
        print(f"🔍 Has tool_calls: {hasattr(last_message, 'tool_calls') and bool(getattr(last_message, 'tool_calls', None))}")

        # If last message is from assistant and has tool calls, execute tools
        if (
            hasattr(last_message, "type")
            and last_message.type == "ai"
            and hasattr(last_message, "tool_calls")
            and last_message.tool_calls
        ):
            print("✅ AI message with tool calls - returning tool_executor")
            return "tool_executor"

        print("✅ No tool calls - returning chat_summary")
        return "chat_summary"
    
    result1 = should_execute_tools_debug(initial_state)
    print(f"Result: {result1}")
    
    # Test scenario 2: After tool execution (should NOT go to tool_executor again)
    print("\n--- Test 2: After tool execution ---")
    post_tool_state = WorkflowState(
        messages=[
            LangChainMessage(content="Search query", type="human"),
            LangChainMessage(
                content="I'll search for information", 
                type="ai", 
                tool_calls=[{"name": "web_search", "args": {"query": "test"}}]
            ),
            # Simulate tool result message
            LangChainMessage(content="Tool result: Search completed", type="tool")
        ],
        user_id=test_user_id
    )
    
    result2 = should_execute_tools_debug(post_tool_state)
    print(f"Result: {result2}")
    
    # Test scenario 3: AI response after tools (should go to memory_creation)
    print("\n--- Test 3: AI response after tools ---")
    final_state = WorkflowState(
        messages=[
            LangChainMessage(content="Search query", type="human"),
            LangChainMessage(
                content="I'll search for information", 
                type="ai", 
                tool_calls=[{"name": "web_search", "args": {"query": "test"}}]
            ),
            LangChainMessage(content="Tool result: Search completed", type="tool"),
            LangChainMessage(content="Here's the information I found...", type="ai")
        ],
        user_id=test_user_id
    )
    
    result3 = should_execute_tools_debug(final_state)
    print(f"Result: {result3}")
    
    print("\n🎯 Summary:")
    print(f"  Test 1 (AI with tools): {result1}")
    print(f"  Test 2 (After tools): {result2}")  
    print(f"  Test 3 (Final AI): {result3}")


if __name__ == "__main__":
    asyncio.run(test_routing_debug())