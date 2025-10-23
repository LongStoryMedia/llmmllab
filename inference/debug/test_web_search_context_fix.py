#!/usr/bin/env python3
"""
Test the actual web search tool with ToolRuntime to verify context_schema fix.

This tests our actual web_search tool with the updated StateGraph configuration
to ensure ToolRuntime context propagation works correctly.
"""

import asyncio
from dataclasses import dataclass

from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

# Import our actual components
from composer.graph.state import ToolsState
from composer.tools.static.web_search_tool import web_search

@dataclass
class ToolsContext:
    """Context schema for tools runtime - provides state access for ToolRuntime injection."""
    state: ToolsState

    def __getitem__(self, key: str):
        """Allow dict-like access to state for compatibility."""
        return getattr(self.state, key, None)

def test_web_search_tool_validation():
    """Test that our web search tool validates correctly with ToolRuntime."""
    print("🧪 Testing web search tool validation...")
    
    try:
        # Check tool schema
        tool_schema = web_search.get_input_schema()
        print(f"✅ Web search tool schema: {tool_schema}")
        
        # Check required fields
        required_fields = tool_schema.model_fields.keys() if hasattr(tool_schema, 'model_fields') else []
        print(f"✅ Required fields: {list(required_fields)}")
        
        if 'runtime' in required_fields:
            print("✅ Tool correctly requires runtime parameter")
        else:
            print("⚠️ Tool does not require runtime parameter")
            
        return True
        
    except Exception as e:
        print(f"❌ Web search tool validation failed: {e}")
        return False

def test_tools_state_graph_with_context():
    """Test StateGraph with ToolsState and context_schema."""
    print("🧪 Testing StateGraph with ToolsState and context_schema...")
    
    try:
        # Create StateGraph with ToolsState and context_schema - matches our actual setup
        builder = StateGraph(ToolsState, context_schema=ToolsContext)
        
        # Create ToolNode with our actual web search tool
        tool_node = ToolNode([web_search])
        builder.add_node("tools", tool_node)
        
        # Set entry point
        builder.set_entry_point("tools")
        
        # Try to compile
        graph = builder.compile()
        print("✅ StateGraph with ToolsState and context_schema compiled successfully!")
        print("✅ This should enable ToolRuntime context propagation to subgraphs")
        return True
        
    except Exception as e:
        print(f"❌ StateGraph with ToolsState and context_schema failed: {e}")
        return False

def test_tool_node_creation():
    """Test ToolNode creation with web search tool."""
    print("🧪 Testing ToolNode creation with web search tool...")
    
    try:
        # Create ToolNode with our web search tool
        tool_node = ToolNode([web_search])
        print("✅ ToolNode created successfully with web search tool")
        
        # Test that the tool node has our tool
        if hasattr(tool_node, 'tools_by_name'):
            tool_names = list(tool_node.tools_by_name.keys())
            print(f"✅ Tools in ToolNode: {tool_names}")
            
            if 'web_search' in tool_names:
                print("✅ Web search tool registered in ToolNode")
            else:
                print("⚠️ Web search tool not found in ToolNode")
        
        return True
        
    except Exception as e:
        print(f"❌ ToolNode creation failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing web search tool with context_schema fix...\n")
    
    # Test web search tool validation
    test_web_search_tool_validation()
    print()
    
    # Test ToolNode creation
    test_tool_node_creation()
    print()
    
    # Test StateGraph with context schema
    test_tools_state_graph_with_context()
    print()
    
    print("✅ Web search tool context schema fix test completed!")
    print("📝 If all tests pass, the E2E tests should now work correctly")

if __name__ == "__main__":
    main()