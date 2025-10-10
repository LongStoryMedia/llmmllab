"""
Test the server-composer integration with proper database initialization.
"""

import asyncio
import sys

# Add the inference directory to the path
sys.path.insert(0, '/app')

# Import necessary modules
import composer
from db import storage
import os


async def test_composer_integration_with_db():
    """Test the composer interface with proper database initialization."""
    print("🧪 Testing composer integration with database...")
    
    try:
        # Initialize the database first
        print("   💾 Initializing database...")
        
        # Build connection string from environment variables
        db_host = os.environ.get("DB_HOST", "localhost")
        db_port = os.environ.get("DB_PORT", "5432") 
        db_user = os.environ.get("DB_USER", "lsm")
        db_password = os.environ.get("DB_PASSWORD", "")
        db_name = os.environ.get("DB_NAME", "llmmll")
        
        connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        
        await storage.initialize(connection_string)
        print("   ✅ Database initialized")
        
        # Test composer initialization
        print("   🔧 Initializing composer...")
        await composer.initialize_composer()
        print("   ✅ Composer initialized")
        
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
        event_count = 0
        async for event in composer.execute_workflow(workflow_result, initial_state):
            print(f"   📡 Event {event_count}: {str(event)[:100]}...")
            event_count += 1
            # Just get a few events to verify streaming works
            if event_count >= 3:
                break
            
        print(f"   🎉 Composer integration test PASSED! ({event_count} events received)")
        return True
        
    except Exception as e:
        print(f"   ❌ Composer integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_composer_integration_with_db())
    sys.exit(0 if success else 1)