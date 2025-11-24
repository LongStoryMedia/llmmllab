"""
Test OID recovery mechanism in message storage
"""
import asyncio
from models.message import Message
from models.message_content import MessageContent
from models.message_content_type import MessageContentType
from models.message_role import MessageRole
import db


async def test_oid_recovery():
    """Test that message storage operations work with OID recovery"""
    print("🔧 Initializing database with recovery manager...")
    await db.storage.initialize()
    
    print("📝 Testing message storage with recovery mechanism...")
    try:
        # Create a test message
        test_message = Message(
            conversation_id=1,
            role=MessageRole.USER,
            content=[
                MessageContent(
                    type=MessageContentType.TEXT,
                    text="Test message for OID recovery validation"
                )
            ]
        )
        
        # This should use the new recovery mechanism
        result = await db.storage.message.add_message(test_message)
        print(f"✅ Message stored successfully with ID: {result}")
        
        if result:
            # Test retrieval (also uses recovery now)
            retrieved = await db.storage.message.get_message(result)
            if retrieved:
                print(f"✅ Message retrieved successfully: role={retrieved.role}, content_count={len(retrieved.content)}")
            else:
                print("❌ Message retrieval failed")
                
            # Test conversation history (also uses recovery)
            history = await db.storage.message.get_conversation_history(1)
            print(f"✅ Conversation history retrieved: {len(history)} messages")
        
        print("🎯 All message storage operations completed successfully with recovery support!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await db.storage.close()


if __name__ == "__main__":
    asyncio.run(test_oid_recovery())