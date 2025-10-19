#!/usr/bin/env python3
"""
Test script to verify delete message functionality works correctly.
"""

import asyncio
import sys
import os

# Add the inference directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from db import storage
from models.message import Message, MessageContent, MessageContentType

async def test_delete_functionality():
    """Test that delete functionality works correctly."""
    print("🧪 Testing delete message functionality...")
    
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
        
        # Create a test conversation and message
        print("\n1️⃣ Creating test message...")
        conversation_id = await storage.conversation.create_conversation(
            title="Delete Test",
            user_id="test-user"
        )
        
        test_message = Message(
            role="user",
            conversation_id=conversation_id,
            content=[
                MessageContent(type=MessageContentType.TEXT, text="This message will be deleted."),
                MessageContent(type=MessageContentType.TEXT, text="Second content piece.")
            ]
        )
        
        message_id = await storage.message.add_message(test_message)
        print(f"✅ Created message {message_id} with 2 content pieces")
        
        # Verify message exists
        print("\n2️⃣ Verifying message exists...")
        message = await storage.message.get_message(message_id)
        if message and len(message.content) == 2:
            print(f"✅ Message found with {len(message.content)} content pieces")
        else:
            print(f"❌ Message verification failed")
            return
            
        # Delete the message
        print(f"\n3️⃣ Deleting message {message_id}...")
        await storage.message.delete_message(message_id)
        print("✅ Delete operation completed without error")
        
        # Verify message is deleted
        print("\n4️⃣ Verifying message deletion...")
        deleted_message = await storage.message.get_message(message_id)
        if deleted_message is None:
            print("✅ Message successfully deleted")
        else:
            print("❌ Message still exists after deletion")
            
        # Verify contents are also deleted
        print("\n5️⃣ Verifying contents deletion...")
        async with storage.pool.acquire() as conn:
            contents_count = await conn.fetchval(
                "SELECT COUNT(*) FROM message_contents WHERE message_id = $1", 
                message_id
            )
            if contents_count == 0:
                print("✅ All message contents successfully deleted")
            else:
                print(f"❌ {contents_count} content pieces still exist")
            
        print("\n🎉 Delete functionality test completed successfully!")
        
    except Exception as e:
        print(f"❌ Delete test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_delete_functionality())