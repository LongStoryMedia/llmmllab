#!/usr/bin/env python3
"""
Test script to verify the updated message queries work with JSON aggregation.
"""

import asyncio
import sys
import os

# Add the inference directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from db.message_storage import MessageStorage
from models.message import Message, MessageContent, MessageContentType
from db import storage

async def test_message_queries():
    """Test the updated message queries with JSON aggregation."""
    print("🧪 Testing message queries with JSON aggregation...")
    
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
        
        # Test 1: Get a single message by ID (using updated get_message query)
        print("\n1️⃣ Testing get_message with JSON aggregation...")
        try:
            # Try to get message ID 1 (if it exists)
            message = await message_storage.get_message(1)
            if message:
                print(f"✅ Retrieved message ID 1:")
                print(f"   Role: {message.role}")
                print(f"   Content items: {len(message.content)}")
                for i, content in enumerate(message.content):
                    print(f"   Content {i+1}: {content.type} - {content.text[:50]}...")
            else:
                print("❌ No message found with ID 1")
        except Exception as e:
            print(f"❌ Error getting message: {e}")
        
        # Test 2: Get messages by conversation ID
        print("\n2️⃣ Testing get_messages_by_conversation_id with JSON aggregation...")
        try:
            messages = await message_storage.get_messages_by_conversation_id(1, limit=5, offset=0)
            print(f"✅ Retrieved {len(messages)} messages for conversation 1")
            for i, msg in enumerate(messages):
                print(f"   Message {i+1}: {msg.role} - {len(msg.content)} contents")
        except Exception as e:
            print(f"❌ Error getting messages by conversation ID: {e}")
            
        # Test 3: Get conversation history
        print("\n3️⃣ Testing get_conversation_history with JSON aggregation...")
        try:
            history = await message_storage.get_conversation_history(1)
            print(f"✅ Retrieved {len(history)} messages in conversation history")
            for i, msg in enumerate(history):
                print(f"   History {i+1}: {msg.role} - {len(msg.content)} contents")
        except Exception as e:
            print(f"❌ Error getting conversation history: {e}")
            
        print("\n🎉 All tests completed!")
        
    except Exception as e:
        print(f"❌ Failed to initialize or run tests: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_message_queries())