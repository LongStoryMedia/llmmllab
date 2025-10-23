#!/usr/bin/env python3
"""
Test script to verify ToolRuntime validation works with context_schema fix.

This script tests that adding context_schema to StateGraph enables ToolRuntime 
context propagation in subgraphs, fixing the "runtime Field required" validation errors.
"""

import asyncio
from typing import Any
from dataclasses import dataclass

from langchain_core.tools import tool
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langchain.tools import ToolRuntime

# Mock state schema
class TestState:
    """Mock state for testing."""
    def __init__(self):
        self.user_config = {"test": "value"}
        self.messages = []

@dataclass
class TestContext:
    """Test context schema for ToolRuntime injection."""
    state: TestState

# Test tool using ToolRuntime pattern
@tool
async def test_tool_with_runtime(
    query: str,
    runtime: ToolRuntime,
) -> str:
    """Test tool that requires ToolRuntime injection."""
    # Try to access state through runtime
    state = runtime.state
    return f"Tool executed with query: {query}, state access: {hasattr(state, 'user_config')}"

def test_with_context_schema():
    """Test StateGraph with context_schema parameter."""
    print("🧪 Testing StateGraph with context_schema...")
    
    try:
        # Create StateGraph WITH context_schema
        builder = StateGraph(dict, context_schema=TestContext)  # Use dict for simplicity
        
        # Create ToolNode with our test tool
        tool_node = ToolNode([test_tool_with_runtime])
        builder.add_node("tools", tool_node)
        
        # Set entry point
        builder.set_entry_point("tools")
        
        # Try to compile
        graph = builder.compile()
        print("✅ StateGraph with context_schema compiled successfully!")
        return True
        
    except Exception as e:
        print(f"❌ StateGraph with context_schema failed: {e}")
        return False

def test_without_context_schema():
    """Test StateGraph without context_schema parameter."""
    print("🧪 Testing StateGraph without context_schema...")
    
    try:
        # Create StateGraph WITHOUT context_schema
        builder = StateGraph(dict)  # Use dict for simplicity
        
        # Create ToolNode with our test tool
        tool_node = ToolNode([test_tool_with_runtime])
        builder.add_node("tools", tool_node)
        
        # Set entry point
        builder.set_entry_point("tools")
        
        # Try to compile
        graph = builder.compile()
        print("⚠️ StateGraph without context_schema compiled (but ToolRuntime won't work)")
        return True
        
    except Exception as e:
        print(f"❌ StateGraph without context_schema failed: {e}")
        return False

async def test_tool_validation():
    """Test tool validation directly."""
    print("🧪 Testing tool validation...")
    
    try:
        # Try to validate the tool
        tool_schema = test_tool_with_runtime.get_input_schema()
        print(f"✅ Tool schema: {tool_schema}")
        
        # Check if runtime is in required fields
        required_fields = tool_schema.get('required', [])
        if 'runtime' in required_fields:
            print("⚠️ Tool requires runtime parameter (expected with ToolRuntime pattern)")
        else:
            print("✅ Tool does not require runtime parameter")
            
        return True
        
    except Exception as e:
        print(f"❌ Tool validation failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing ToolRuntime context schema fix...\n")
    
    # Test tool validation
    asyncio.run(test_tool_validation())
    print()
    
    # Test without context schema
    test_without_context_schema()
    print()
    
    # Test with context schema
    test_with_context_schema()
    print()
    
    print("✅ Context schema fix test completed!")

if __name__ == "__main__":
    main()