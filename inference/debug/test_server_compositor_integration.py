"""
Test the server-composer integration.
Simple test to verify that the chat router properly delegates to composer interface.
"""

import asyncio
import sys

# Add the inference directory to the path
sys.path.insert(0, '/app')

from server.handlers.composer_completion import composer_chat_completion
from fastapi import BackgroundTasks


async def test_composer_integration():
    """Test the composer integration from server side."""
    print("🧪 Testing server-composer integration...")
    
    # Create a background tasks instance
    background_tasks = BackgroundTasks()
    
    try:
        # Test user and conversation (using values from previous e2e test)
        user_id = "test_server_composer_user"
        conversation_id = 1  # Simple test conversation
        
        print(f"   🔧 Testing with user_id: {user_id}, conversation_id: {conversation_id}")
        
        # Call the composer completion handler
        print("   🎼 Calling composer_chat_completion...")
        
        event_count = 0
        async for event in composer_chat_completion(user_id, conversation_id, background_tasks):
            print(f"   📡 Received event {event_count}: {event[:100]}...")
            event_count += 1
            
            # Just collect a few events to verify streaming works
            if event_count >= 5:
                break
                
        print(f"   ✅ Successfully received {event_count} streaming events")
        print("   🎉 Server-composer integration test PASSED!")
        
    except Exception as e:
        print(f"   ❌ Integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    return True


if __name__ == "__main__":
    success = asyncio.run(test_composer_integration())
    sys.exit(0 if success else 1)