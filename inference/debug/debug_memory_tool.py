#!/usr/bin/env python3
"""
Debug memory retrieval tool object type.
"""

def debug_memory_tool():
    """Debug what the memory retrieval tool actually is."""
    print("🧪 Debugging memory retrieval tool object...")
    
    try:
        from composer.tools.static.memory_retrieval_tool import memory_retrieval
        
        print(f"✅ Type: {type(memory_retrieval)}")
        print(f"✅ Callable: {callable(memory_retrieval)}")
        print(f"✅ Dir: {dir(memory_retrieval)}")
        
        if hasattr(memory_retrieval, 'name'):
            print(f"✅ Name: {memory_retrieval.name}")
        if hasattr(memory_retrieval, 'description'): 
            print(f"✅ Description: {memory_retrieval.description[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    success = debug_memory_tool()
    sys.exit(0 if success else 1)