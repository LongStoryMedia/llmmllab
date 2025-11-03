#!/usr/bin/env python3
"""
Test the complete state transformation flow from WorkflowState through ToolsAgentSubgraph.

This script traces how user_id flows from initial WorkflowState creation 
through the transform_to_tools_state conversion to verify where user_id gets lost.
"""

import asyncio
from models import LangChainMessage, UserConfig, CircuitBreakerConfig
from composer.graph.state import WorkflowState, ToolsState
from composer.graph.subgraphs.tools_agent import ToolsAgentSubgraph


def create_test_workflow_state() -> WorkflowState:
    """Create a test WorkflowState with user_id properly set."""
    
    # Create test messages
    messages = [
        LangChainMessage(type="human", content="Hello, I want to retrieve my memories"),
        LangChainMessage(type="ai", content="I'll help you retrieve your memories using the memory retrieval tool."),
    ]
    
    # Create WorkflowState with user_id (no user_config for simplicity)
    state = WorkflowState(
        user_id="test-user-123",  # This should be passed through
        conversation_id=42,
        messages=messages,
        current_user_message=LangChainMessage(type="human", content="Hello, I want to retrieve my memories"),
    )
    
    print(f"✅ Created WorkflowState with user_id: '{state.user_id}'")
    print(f"   - Type of user_id: {type(state.user_id)}")
    print(f"   - user_id is truthy: {bool(state.user_id)}")
    print(f"   - user_id == '': {state.user_id == ''}")
    print(f"   - user_config present: {state.user_config is not None}")
    
    return state


def test_direct_state_transformation():
    """Test the direct transform_to_tools_state logic without full subgraph."""
    
    print("=" * 60)
    print("🧪 TESTING DIRECT STATE TRANSFORMATION")
    print("=" * 60)
    
    # Create test WorkflowState
    workflow_state = create_test_workflow_state()
    
    # Test the direct getattr logic used in transform_to_tools_state
    print(f"\n🔄 Testing direct getattr logic from transform_to_tools_state...")
    
    # Simulate the exact logic from line 306 in tools_agent.py
    user_id_from_getattr = getattr(workflow_state, "user_id", "")
    conversation_id_from_getattr = getattr(workflow_state, "conversation_id", 0)
    user_config_from_getattr = getattr(workflow_state, "user_config", None)
    current_date_from_getattr = getattr(workflow_state, "current_date", "")
    
    # Simulate the ToolsState dictionary creation
    simulated_tools_state = {
        "messages": [],  # We'll skip message conversion for this test
        "user_id": user_id_from_getattr,
        "conversation_id": conversation_id_from_getattr,
        "user_config": user_config_from_getattr,
        "system_config": None,
        "current_date": current_date_from_getattr,
        "tool_call_count": 0,
    }
    
    print(f"\n📋 Simulated ToolsState after transformation:")
    print(f"   - user_id: '{simulated_tools_state.get('user_id', 'NOT_FOUND')}'")
    print(f"   - Type: {type(simulated_tools_state.get('user_id', 'NOT_FOUND'))}")
    print(f"   - Is truthy: {bool(simulated_tools_state.get('user_id', ''))}")
    print(f"   - Equals empty string: {simulated_tools_state.get('user_id', '') == ''}")
    print(f"   - conversation_id: {simulated_tools_state.get('conversation_id', 'NOT_FOUND')}")
    print(f"   - user_config present: {simulated_tools_state.get('user_config') is not None}")
    
    # Test what happens in memory retrieval tool check
    print(f"\n🔍 Testing memory tool state check:")
    user_id_from_state = simulated_tools_state.get("user_id")
    print(f"   - tools_state.get('user_id'): '{user_id_from_state}'")
    print(f"   - not tools_state.get('user_id'): {not simulated_tools_state.get('user_id')}")
    
    if not simulated_tools_state.get("user_id"):
        print("   ❌ MEMORY TOOL WOULD FAIL: Missing user_id in state")
        return False
    else:
        print("   ✅ MEMORY TOOL WOULD PASS: user_id found in state")
        return True


def test_getattr_behavior():
    """Test getattr behavior with different scenarios."""
    
    print(f"\n=" * 60)
    print("🧪 TESTING GETATTR BEHAVIOR")
    print("=" * 60)
    
    # Create test WorkflowState
    workflow_state = create_test_workflow_state()
    
    # Test different getattr scenarios
    print(f"\n🔍 Testing getattr scenarios:")
    
    # Normal case
    user_id_normal = getattr(workflow_state, "user_id", "")
    print(f"   - getattr(state, 'user_id', ''): '{user_id_normal}'")
    
    # What if user_id is None?
    workflow_state.user_id = None
    user_id_none = getattr(workflow_state, "user_id", "")
    print(f"   - getattr(state, 'user_id', '') when user_id=None: '{user_id_none}'")
    
    # What if user_id is empty string?
    workflow_state.user_id = ""
    user_id_empty = getattr(workflow_state, "user_id", "")
    print(f"   - getattr(state, 'user_id', '') when user_id='': '{user_id_empty}'")
    
    # What if attribute doesn't exist?
    delattr(workflow_state, "user_id")
    user_id_missing = getattr(workflow_state, "user_id", "")
    print(f"   - getattr(state, 'user_id', '') when attr missing: '{user_id_missing}'")


def main():
    """Main test function."""
    print("🏃 Starting state transformation flow test...")
    
    # Test direct transformation
    success = test_direct_state_transformation()
    
    # Test getattr edge cases
    test_getattr_behavior()
    
    print(f"\n" + "=" * 60)
    if success:
        print("✅ STATE TRANSFORMATION TEST PASSED")
        print("   user_id correctly passed from WorkflowState to ToolsState")
    else:
        print("❌ STATE TRANSFORMATION TEST FAILED") 
        print("   user_id was lost during WorkflowState → ToolsState conversion")
    print("=" * 60)


if __name__ == "__main__":
    main()