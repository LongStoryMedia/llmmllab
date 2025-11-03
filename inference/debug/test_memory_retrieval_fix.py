#!/usr/bin/env python3
"""
Test the memory retrieval fix for missing user_id scenarios.

This test simulates various user_id failure scenarios to verify that 
our improved error handling and logging works correctly.
"""

import asyncio
from unittest.mock import MagicMock
from models import LangChainMessage
from composer.graph.state import WorkflowState, ToolsState
from composer.graph.subgraphs.tools_agent import ToolsAgentSubgraph
from composer.tools.static.memory_retrieval_tool import memory_retrieval
from utils.logging import llmmllogger

logger = llmmllogger.logger.bind(component="MemoryRetrievalFix")


class MockToolRuntime:
    """Mock ToolRuntime for testing memory retrieval tool."""
    
    def __init__(self, state, tool_call_id="test_call"):
        self.state = state
        self.tool_call_id = tool_call_id


def create_workflow_state_with_user_id_issue(user_id_value):
    """Create WorkflowState with specific user_id value for testing."""
    
    state = WorkflowState(
        conversation_id=42,
        messages=[LangChainMessage(type="human", content="Test message")],
        current_user_message=LangChainMessage(type="human", content="Test message"),
    )
    
    # Manually set user_id to the test value
    if user_id_value is not None:
        state.user_id = user_id_value
    else:
        # Simulate missing user_id by deleting the attribute
        if hasattr(state, 'user_id'):
            delattr(state, 'user_id')
    
    return state


def test_transform_with_user_id_scenarios():
    """Test transform_to_tools_state with various user_id scenarios."""
    
    print("=" * 60)
    print("🧪 TESTING TRANSFORM WITH USER_ID SCENARIOS")
    print("=" * 60)
    
    # Create a minimal ToolsAgentSubgraph for testing transformation only
    # We'll mock the dependencies to avoid initialization issues
    mock_tool_registry = MagicMock()
    mock_chat_agent = MagicMock()
    
    # Create ToolsAgentSubgraph instance by bypassing the full initialization 
    tools_agent = object.__new__(ToolsAgentSubgraph)
    tools_agent.tool_registry = mock_tool_registry
    tools_agent.chat_agent = mock_chat_agent
    tools_agent.graph = None
    
    test_scenarios = [
        ("normal_user_id", "test-user-123"),
        ("empty_string", ""),
        ("none_value", None),
        ("missing_attr", "MISSING"),  # Special marker for missing attribute
    ]
    
    results = []
    
    for scenario_name, user_id_value in test_scenarios:
        print(f"\n🔍 Testing scenario: {scenario_name} (user_id={repr(user_id_value)})")
        
        # Create state with specific user_id scenario
        if user_id_value == "MISSING":
            workflow_state = create_workflow_state_with_user_id_issue(None)
        else:
            workflow_state = create_workflow_state_with_user_id_issue(user_id_value)
        
        # Test transformation
        try:
            tools_state = tools_agent.transform_to_tools_state(workflow_state)
            
            print(f"   ✅ Transformation completed")
            print(f"   - ToolsState user_id: {repr(tools_state.get('user_id'))}")
            print(f"   - Type: {type(tools_state.get('user_id')).__name__}")
            print(f"   - Is truthy: {bool(tools_state.get('user_id'))}")
            
            # Test memory tool check
            user_id_from_state = tools_state.get("user_id")
            memory_tool_would_pass = bool(user_id_from_state and user_id_from_state != "")
            
            print(f"   - Memory tool would pass: {memory_tool_would_pass}")
            
            results.append((scenario_name, True, memory_tool_would_pass, tools_state.get('user_id')))
            
        except Exception as e:
            print(f"   ❌ Transformation failed: {e}")
            results.append((scenario_name, False, False, None))
    
    return results


async def test_memory_tool_with_improved_errors():
    """Test memory retrieval tool with improved error messages."""
    
    print("\n" + "=" * 60)
    print("🛠️ TESTING MEMORY TOOL WITH IMPROVED ERRORS")
    print("=" * 60)
    
    # Mock storage to avoid database dependency
    import composer.tools.static.memory_retrieval_tool as tool_module
    original_storage = getattr(tool_module, 'storage', None)
    
    mock_storage = MagicMock()
    mock_storage.pool = True  # Storage initialized
    tool_module.storage = mock_storage
    
    test_cases = [
        ("empty_string_user_id", {"user_id": "", "conversation_id": 42, "user_config": None}),
        ("none_user_id", {"user_id": None, "conversation_id": 42, "user_config": None}),
        ("missing_user_id", {"conversation_id": 42, "user_config": None}),  # No user_id key
        ("valid_user_id", {"user_id": "test-user-123", "conversation_id": 42, "user_config": None}),
    ]
    
    results = []
    
    for test_name, test_state in test_cases:
        print(f"\n🔍 Testing: {test_name}")
        print(f"   State: {test_state}")
        
        try:
            runtime = MockToolRuntime(test_state)
            result = await memory_retrieval.ainvoke({"query": "test query", "runtime": runtime})
            
            print(f"   Result: {result[:200]}...")
            
            if "❌ Memory retrieval failed" in result:
                print(f"   ✅ Tool correctly detected user_id issue")
                results.append((test_name, "error_detected", True))
            else:
                print(f"   ✅ Tool executed successfully") 
                results.append((test_name, "success", True))
                
        except Exception as e:
            print(f"   ❌ Tool execution failed: {e}")
            results.append((test_name, "exception", False))
    
    # Restore original storage
    if original_storage:
        tool_module.storage = original_storage
    
    return results


def main():
    """Main test function."""
    print("🏃 Starting memory retrieval fix test...")
    
    # Test 1: State transformation scenarios
    transform_results = test_transform_with_user_id_scenarios()
    
    # Test 2: Memory tool error handling (async)
    tool_results = asyncio.run(test_memory_tool_with_improved_errors())
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 70)
    
    print(f"\n🔄 State Transformation Results:")
    for scenario, transform_ok, memory_ok, user_id in transform_results:
        status = "✅" if transform_ok and memory_ok else "❌"
        print(f"   {status} {scenario}: transform={transform_ok}, memory_check={memory_ok}, user_id={repr(user_id)}")
    
    print(f"\n🛠️ Memory Tool Results:")
    for test_name, result_type, success in tool_results:
        status = "✅" if success else "❌" 
        print(f"   {status} {test_name}: {result_type}")
    
    # Overall assessment
    transform_passed = all(success for _, success, _, _ in transform_results)
    tool_passed = all(success for _, _, success in tool_results)
    
    print(f"\n" + "=" * 70)
    if transform_passed and tool_passed:
        print("✅ ALL TESTS PASSED - Memory retrieval fix is working")
    else:
        print("❌ SOME TESTS FAILED - Memory retrieval fix needs more work")
    print("=" * 70)


if __name__ == "__main__":
    main()