#!/usr/bin/env python3
"""
Test memory retrieval tool import only (isolated test).
"""

def test_memory_tool_import_only():
    """Test that memory retrieval tool can be imported."""
    print("🧪 Testing memory retrieval tool import...")
    
    try:
        # Just import the memory retrieval tool directly
        from composer.tools.static.memory_retrieval_tool import memory_retrieval
        
        print("✅ Successfully imported memory_retrieval function")
        
        # Check basic attributes
        assert hasattr(memory_retrieval, 'name'), "Function should have 'name' attribute from @tool decorator"
        assert hasattr(memory_retrieval, 'description'), "Function should have 'description' attribute from @tool decorator"
        
        print(f"✅ Tool name: {memory_retrieval.name}")
        print(f"✅ Tool description: {memory_retrieval.description[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    success = test_memory_tool_import_only()
    sys.exit(0 if success else 1)