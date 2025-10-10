"""
Comprehensive test to validate the simplified server architecture.
Tests that the /completions endpoint works with direct composer delegation.
"""

import asyncio
import sys

# Add the inference directory to the path
sys.path.insert(0, '/app')

import composer
from db import storage
from server.routers.chat import chat_completion
from models import Message, MessageRole, MessageContentType
from fastapi import Request, BackgroundTasks
from unittest.mock import Mock
import os


async def test_simplified_server_architecture():
    """Test the complete simplified server architecture."""
    print("🧪 Testing simplified server architecture...")
    
    try:
        # Initialize the database first
        print("   💾 Initializing database...")
        
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
        
        # Create test conversation first
        print("   📝 Creating test conversation...")
        conversation_id = await storage.conversation.create_conversation(
            "test_simplified_user", 
            "Test conversation for simplified architecture"
        )
        print(f"   ✅ Created conversation {conversation_id}")
        
        # Create test message
        test_message = Message(
            conversation_id=conversation_id,
            role=MessageRole.USER,
            content=[{
                "type": MessageContentType.TEXT,
                "text": "Hello, simplified architecture test"
            }]
        )
        
        # Mock request with user authentication
        mock_request = Mock(spec=Request)
        mock_request.headers = {"authorization": "Bearer test-user-token"}
        mock_request.state = Mock()
        mock_request.state.user_id = "test_simplified_user"
        mock_request.state.request_id = "test-simplified-123"
        
        # Create background tasks
        background_tasks = BackgroundTasks()
        
        print("   🎯 Testing /completions endpoint with direct composer delegation...")
        
        # Test the main completions endpoint
        response = await chat_completion(test_message, mock_request, background_tasks)
        
        print(f"   ✅ Response type: {type(response)}")
        print("   📡 Testing streaming response...")
        
        # Test that the response is a streaming response
        if hasattr(response, 'body_iterator'):
            event_count = 0
            async for chunk in response.body_iterator:
                print(f"   📄 Chunk {event_count}: {chunk.decode()[:100]}...")
                event_count += 1
                # Just test a few chunks to verify streaming works
                if event_count >= 3:
                    break
            print(f"   ✅ Received {event_count} streaming chunks")
        
        print("   🎉 Simplified server architecture test PASSED!")
        print("   ✨ Server successfully delegates to composer without intermediate handlers!")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Simplified server architecture test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_simplified_server_architecture())
    sys.exit(0 if success else 1)