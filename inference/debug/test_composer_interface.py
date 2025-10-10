"""
Test the server-composer integration.
Simple test to verify that the composer interface works properly.
"""

import asyncio
import sys

# Add the inference directory to the path
sys.path.insert(0, '/app')

# Import composer interface directly
import composer


async def test_composer_interface():
    """Test the composer interface functions."""
    print("🧪 Testing composer interface functions...")
    
    try:
        # Test composer initialization
        print("   🔧 Initializing composer...")
        initialize_result = await composer.initialize_composer()
        print(f"   ✅ Composer initialized: {initialize_result}")
        
        # Test user and conversation
        user_id = "test_server_composer_user"
        conversation_id = 1
        
        print(f"   🔧 Testing with user_id: {user_id}, conversation_id: {conversation_id}")
        
        # Test workflow composition
        print("   🎼 Composing workflow...")
        workflow_result = await composer.compose_workflow(user_id)
        print(f"   ✅ Workflow composed: {type(workflow_result)}")
        
        # Test initial state creation
        print("   🏁 Creating initial state...")
        initial_state = await composer.create_initial_state(user_id, conversation_id)
        print(f"   ✅ Initial state created: {type(initial_state)}")
        
        # Test workflow execution (just start it, don't wait for completion)
        print("   🚀 Starting workflow execution...")
        async for event in composer.execute_workflow(workflow_result, initial_state):
            print(f"   📡 Received event: {event[:100]}...")
            # Just get one event to verify streaming works
            break
            
        print("   🎉 Composer interface test PASSED!")
        return True
        
    except Exception as e:
        print(f"   ❌ Composer interface test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_composer_interface())
    sys.exit(0 if success else 1)