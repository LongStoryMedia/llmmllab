#!/usr/bin/env python3
"""
Simple smoke test for chat completion endpoint.
Tests basic "hello" message through the chat completion flow.
"""

import asyncio
import sys
import os

# Add the app directory to Python path
sys.path.append('/app')

async def test_chat_completion():
    """Test basic chat completion with a simple hello message."""
    
    # Import after path setup
    from fastapi.testclient import TestClient
    from server.app import app
    from models import Message, MessageRole, MessageContent, MessageContentType
    
    print("🚀 Starting chat completion smoke test...")
    
    # Disable auth for testing
    os.environ['DISABLE_AUTH'] = 'true'
    print("✅ Auth disabled for testing")
    
    # Create test client
    client = TestClient(app)
    
    # First create a conversation
    print("📝 Creating test conversation...")
    conv_response = client.post("/v1/chat/conversations")
    if conv_response.status_code != 200:
        print(f"❌ Failed to create conversation: {conv_response.status_code} - {conv_response.text}")
        return False
    
    conversation = conv_response.json()
    conversation_id = conversation.get("id")
    print(f"✅ Created conversation with ID: {conversation_id}")
    
    # Test message
    test_message = Message(
        conversation_id=conversation_id,
        role=MessageRole.USER,
        content=[
            MessageContent(
                type=MessageContentType.TEXT,
                text="hello"
            )
        ]
    )
    
    print(f"📝 Testing with message: {test_message.content[0].text}")
    
    try:
        # Make the request
        response = client.post(
            "/v1/chat/completions",
            json=test_message.model_dump(),
            headers={"Content-Type": "application/json"}
        )
        
        print(f"📡 Response status: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Chat completion endpoint is working!")
            # For streaming response, just check it starts correctly
            content = response.content.decode()
            if content:
                print(f"📄 Response preview: {content[:200]}...")
            return True
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"📄 Error details: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Run the test
    success = asyncio.run(test_chat_completion())
    sys.exit(0 if success else 1)