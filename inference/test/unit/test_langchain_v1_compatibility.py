"""
Test LangChain v1.0 compatibility for the composer service.
"""

import pytest
import asyncio
from typing import List

# Test imports are compatible with LangChain v1.0
def test_langchain_v1_imports():
    """Test that all LangChain v1.0 imports work correctly."""
    try:
        # Core LangGraph imports (should remain unchanged)
        from langgraph.graph import StateGraph, END, add_messages
        from langgraph.graph.state import CompiledStateGraph
        
        # LangChain v1.0 imports
        from langgraph.prebuilt import ToolNode  # Moved location in v1.0
        from langchain_core.tools import BaseTool
        from langchain_core.runnables import RunnableParallel, RunnableLambda
        
        print("✅ All LangChain v1.0 imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def test_message_structure():
    """Test that message structure is compatible with v1.0."""
    from models import LangChainMessage
    
    # Test creating messages with v1.0 compatible structure
    test_cases = [
        {"type": "human", "content": "Hello"},
        {"type": "ai", "content": "Hi there!"},
        {"type": "system", "content": "System message"},
        {"type": "tool", "content": "Tool result"},
    ]
    
    for case in test_cases:
        try:
            message = LangChainMessage(**case)
            assert message.type == case["type"]
            assert message.content == case["content"]
            print(f"✅ Message type '{case['type']}' compatible")
        except Exception as e:
            print(f"❌ Message type '{case['type']}' failed: {e}")
            return False
    
    return True


def test_tool_node_v1_compatibility():
    """Test that ToolNode usage is compatible with v1.0."""
    try:
        from langgraph.prebuilt import ToolNode
        from langchain_core.tools import BaseTool
        
        # Create a simple test tool
        class TestTool(BaseTool):
            name = "test_tool"
            description = "A test tool"
            
            def _run(self, query: str) -> str:
                return f"Test result for: {query}"
            
            async def _arun(self, query: str) -> str:
                return self._run(query)
        
        # Test ToolNode creation with v1.0 features
        tools = [TestTool()]
        
        # Test with handle_tool_errors parameter (v1.0 feature)
        tool_node = ToolNode(tools, handle_tool_errors=True)
        
        print("✅ ToolNode v1.0 compatibility successful")
        return True
    except Exception as e:
        print(f"❌ ToolNode compatibility test failed: {e}")
        return False


@pytest.mark.asyncio
async def test_workflow_state_compatibility():
    """Test that WorkflowState is compatible with LangGraph v1.0."""
    try:
        from composer.graph.state import WorkflowState
        from models import LangChainMessage
        
        # Create test state
        state = WorkflowState(
            messages=[
                LangChainMessage(type="human", content="Test message"),
                LangChainMessage(type="ai", content="Test response")
            ],
            user_id="test_user"
        )
        
        # Test state operations
        assert len(state.messages) == 2
        assert state.messages[0].type == "human"
        assert state.messages[1].type == "ai"
        assert state.user_id == "test_user"
        
        print("✅ WorkflowState v1.0 compatibility successful")
        return True
    except Exception as e:
        print(f"❌ WorkflowState compatibility test failed: {e}")
        return False


def run_all_compatibility_tests():
    """Run all LangChain v1.0 compatibility tests."""
    tests = [
        ("LangChain v1.0 Imports", test_langchain_v1_imports),
        ("Message Structure", test_message_structure),
        ("ToolNode v1.0", test_tool_node_v1_compatibility),
    ]
    
    async_tests = [
        ("WorkflowState Compatibility", test_workflow_state_compatibility),
    ]
    
    print("🔍 Testing LangChain v1.0 Compatibility...")
    print("=" * 50)
    
    results = {}
    
    # Run sync tests
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Run async tests
    for test_name, test_func in async_tests:
        print(f"\n📋 {test_name}")
        try:
            result = asyncio.run(test_func())
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All LangChain v1.0 compatibility tests passed!")
    else:
        print("⚠️  Some compatibility issues found. See details above.")
    
    return passed == total


if __name__ == "__main__":
    run_all_compatibility_tests()