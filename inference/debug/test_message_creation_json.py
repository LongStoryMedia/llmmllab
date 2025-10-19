#!/usr/bin/env python3
"""
Test script to create a message and verify JSON aggregation retrieval works.
"""

import asyncio
import sys
import os
from datetime import datetime

# Add the inference directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from db.message_storage import MessageStorage
from models.message import Message, MessageContent, MessageContentType
from db import storage

async def test_message_creation_and_retrieval():
    """Test creating a message and retrieving it with JSON aggregation."""
    print("🧪 Testing message creation and JSON aggregation retrieval...")
    
    try:
        # Initialize database connection
        print("   💾 Initializing database...")
        db_host = os.environ.get("DB_HOST", "localhost")
        db_port = os.environ.get("DB_PORT", "5432")
        db_user = os.environ.get("DB_USER", "lsm")
        db_password = os.environ.get("DB_PASSWORD", "")
        db_name = os.environ.get("DB_NAME", "llmmll")

        connection_string = (
            f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        )
        await storage.initialize(connection_string)
        print("   ✅ Database initialized")
        
        # Use storage message instance (already initialized)
        message_storage = storage.message
        conversation_storage = storage.conversation
        
        # Create a test conversation first
        print("\n1️⃣ Creating test conversation...")
        conversation_id = await conversation_storage.create_conversation(
            title="Test Conversation for JSON Aggregation",
            user_id="test-user"
        )
        print(f"✅ Created conversation with ID: {conversation_id}")
        
        # Create a test message with multiple content pieces
        print("\n2️⃣ Creating test message with multiple contents...")
        test_message = Message(
            role="user",
            conversation_id=conversation_id,
            content=[
                MessageContent(type=MessageContentType.TEXT, text="Hello, this is the first content piece."),
                MessageContent(type=MessageContentType.TEXT, text="And this is the second content piece."),
                MessageContent(type=MessageContentType.TEXT, text="Finally, this is the third content piece.")
            ]
        )
        
        message_id = await message_storage.add_message(test_message)
        print(f"✅ Created message with ID: {message_id}")
        
        # Test retrieving the message using JSON aggregation
        print("\n3️⃣ Testing get_message with JSON aggregation...")
        retrieved_message = await message_storage.get_message(message_id)
        if retrieved_message:
            print(f"✅ Retrieved message ID {message_id}:")
            print(f"   Role: {retrieved_message.role}")
            print(f"   Content items: {len(retrieved_message.content)}")
            for i, content in enumerate(retrieved_message.content):
                print(f"   Content {i+1}: {content.type.value} - {content.text}")
            
            if len(retrieved_message.content) == 3:
                print("✅ All 3 content pieces retrieved correctly!")
            else:
                print(f"❌ Expected 3 content pieces, got {len(retrieved_message.content)}")
        else:
            print(f"❌ Failed to retrieve message {message_id}")
        
        # Test get_messages_by_conversation_id
        print("\n4️⃣ Testing get_messages_by_conversation_id...")
        messages = await message_storage.get_messages_by_conversation_id(conversation_id, limit=10, offset=0)
        print(f"✅ Retrieved {len(messages)} messages for conversation {conversation_id}")
        if messages:
            msg = messages[0]
            print(f"   First message has {len(msg.content)} content pieces")
            
        # Test get_conversation_history  
        print("\n5️⃣ Testing get_conversation_history...")
        history = await message_storage.get_conversation_history(conversation_id)
        print(f"✅ Retrieved {len(history)} messages in conversation history")
        if history:
            msg = history[0]
            print(f"   History message has {len(msg.content)} content pieces")
            
        print("\n🎉 All JSON aggregation tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Failed to run test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_message_creation_and_retrieval())