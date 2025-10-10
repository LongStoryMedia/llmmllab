"""
Test the main /completions endpoint to ensure it properly delegates to composer.
"""

import asyncio
import sys

# Add the inference directory to the path
sys.path.insert(0, '/app')

from server.routers.chat import chat_completion
from models import Message, MessageRole, MessageContentType
from fastapi import Request, BackgroundTasks
from unittest.mock import Mock
from db import storage
import os


async def test_main_completions_endpoint():
    """Test that the main /completions endpoint delegates to composer correctly."""
    print("🧪 Testing main /completions endpoint...")
    
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
        
        # Create test message
        test_message = Message(
            conversation_id=1,
            role=MessageRole.USER,
            content=[{
                "type": MessageContentType.TEXT,
                "text": "Hello, test message"
            }]
        )
        
        # Mock request with user authentication
        mock_request = Mock(spec=Request)
        mock_request.headers = {"authorization": "Bearer test-user-token"}
        mock_request.state = Mock()
        mock_request.state.user_id = "test_server_composer_user"
        mock_request.state.request_id = "test-request-123"
        
        # Create background tasks
        background_tasks = BackgroundTasks()
        
        print("   🎯 Calling main /completions endpoint...")
        
        # Test the main completions endpoint
        response = await chat_completion(test_message, mock_request, background_tasks)
        
        print(f"   ✅ Response type: {type(response)}")
        print("   🎉 Main /completions endpoint test PASSED!")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Main completions endpoint test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_main_completions_endpoint())
    sys.exit(0 if success else 1)