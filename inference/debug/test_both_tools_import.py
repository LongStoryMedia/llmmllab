#!/usr/bin/env python3
"""
Test both memory retrieval and summarization tools can be imported together.
"""

def test_both_tools_import():
    """Test that both tools can be imported together."""
    print("🧪 Testing both tools import together...")
    
    try:
        # Import both tools
        from composer.tools.static.memory_retrieval_tool import memory_retrieval
        from composer.tools.static.summarization_tool import summarization
        
        print("✅ Successfully imported both tools")
        
        # Check they have different names
        assert memory_retrieval.name != summarization.name, "Tools should have different names"
        print(f"✅ Memory tool name: {memory_retrieval.name}")
        print(f"✅ Summarization tool name: {summarization.name}")
        
        # Check they both have the required attributes
        for tool, name in [(memory_retrieval, "memory_retrieval"), (summarization, "summarization")]:
            assert hasattr(tool, 'name'), f"{name} should have 'name' attribute"
            assert hasattr(tool, 'description'), f"{name} should have 'description' attribute"
            assert hasattr(tool, 'run'), f"{name} should have 'run' method"
            assert hasattr(tool, 'arun'), f"{name} should have 'arun' method"
        
        print("✅ Both tools have required attributes")
        
        # Test importing from the static tools module
        from composer.tools.static import memory_retrieval as mem_tool, summarization as sum_tool, web_search
        
        # Verify the imports work
        assert mem_tool is not None, "Memory tool import should work"
        assert sum_tool is not None, "Summarization tool import should work" 
        assert web_search is not None, "Web search tool import should work"
        
        print("✅ Successfully imported from static tools module")
        print("✅ Available tools: memory_retrieval, summarization, web_search")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    success = test_both_tools_import()
    sys.exit(0 if success else 1)