"""
Simple validation test for the simplified server architecture.
Tests imports and basic functionality without complex database setup.
"""

import asyncio
import sys

# Add the inference directory to the path
sys.path.insert(0, '/app')


async def test_simplified_imports_and_functionality():
    """Test that all imports work and basic functionality is intact."""
    print("🧪 Testing simplified server architecture imports and functionality...")
    
    try:
        # Test 1: Server router imports
        print("   📦 Testing server router imports...")
        from server.routers.chat import router, chat_completion
        print("   ✅ Chat router imports successfully")
        
        # Test 2: Composer imports
        print("   📦 Testing composer imports...")
        import composer
        print("   ✅ Composer module imports successfully")
        
        # Test 3: Models imports
        print("   📦 Testing models imports...")
        from models import Message, MessageRole, MessageContentType
        print("   ✅ Models import successfully")
        
        # Test 4: FastAPI app imports
        print("   📦 Testing FastAPI app imports...")
        from server.app import app
        print("   ✅ FastAPI app imports successfully")
        
        # Test 5: Verify no old handler imports exist
        print("   🚫 Verifying old handlers are removed...")
        try:
            from server.handlers import completion
            print("   ❌ ERROR: Old completion handler still exists!")
            return False
        except ImportError:
            print("   ✅ Old completion handler properly removed")
        
        # Test 6: Verify no tools imports exist  
        print("   🚫 Verifying server tools are removed...")
        try:
            from server.tools import integration
            print("   ❌ ERROR: Server tools still exist!")
            return False
        except ImportError:
            print("   ✅ Server tools properly removed")
        
        # Test 7: Test composer interface functions
        print("   🎼 Testing composer interface...")
        await composer.initialize_composer()
        print("   ✅ Composer service initialized successfully")
        
        # Test 8: Verify the chat router structure
        print("   🔍 Checking chat router endpoints...")
        routes = [route for route in router.routes if hasattr(route, 'path')]
        completion_routes = [route for route in routes if 'completions' in route.path]
        
        print(f"   📊 Found {len(completion_routes)} completion route(s)")
        for route in completion_routes:
            print(f"   📍 Route: {route.path}")
        
        # Should have exactly one completion route  
        if len(completion_routes) == 1 and '/completions' in completion_routes[0].path:
            print("   ✅ Single /completions endpoint confirmed")
        else:
            print(f"   ❌ ERROR: Expected 1 /completions route, found {len(completion_routes)}")
            return False
        
        print("   🎉 Simplified server architecture validation PASSED!")
        print("   ✨ All imports work, old handlers/tools removed, single endpoint confirmed!")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Simplified server architecture validation FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_simplified_imports_and_functionality())
    sys.exit(0 if success else 1)