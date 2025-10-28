#!/usr/bin/env python3
"""
Test script to verify message creation fix.
"""

import asyncio
from datetime import datetime, timezone
from models.conversation import Conversation
from models.message import Message
from models.message_role import MessageRole
from models.message_content import MessageContent, MessageContentType
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="message_fix_test")


async def test_message_creation():
    """Test message creation with proper content insertion."""
    try:
        # Initialize database
        logger.info("Initializing database...")
        from db import storage
        import os
        
        # Build connection string from environment variables
        db_host = os.getenv("DB_HOST", "localhost")
        db_port = os.getenv("DB_PORT", "5432")
        db_user = os.getenv("DB_USER", "lsm")
        db_password = os.getenv("DB_PASSWORD", "")
        db_name = os.getenv("DB_NAME", "llmmll")
        db_sslmode = os.getenv("DB_SSLMODE", "disable")

        connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}?sslmode={db_sslmode}"
        await storage.initialize(connection_string)
        
        logger.info("✅ Database initialized successfully")
        
        # Create test conversation
        test_conv = Conversation(
            id=0,  # Will be set by database
            user_id='debug_test_user_fix',
            title='Message Fix Verification Test',
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )
        
        conv_id = await storage.conversation.create_conversation(test_conv)
        logger.info(f"✅ Created conversation: {conv_id}")
        
        # Create message with content
        content_list = [
            MessageContent(
                type=MessageContentType.TEXT, 
                text='Test message to verify message_id is set correctly on content'
            )
        ]
        
        message = Message(
            conversation_id=conv_id,
            role=MessageRole.USER,
            content=content_list,
            created_at=datetime.now(timezone.utc)
        )
        
        # Add message
        logger.info("Creating message with content...")
        message_id = await storage.message.add_message(message)
        logger.info(f"✅ Created message: {message_id}")
        
        # Verify persistence by retrieving conversation history
        logger.info("Retrieving conversation history...")
        messages = await storage.message.get_conversation_history(conv_id)
        logger.info(f"✅ Retrieved {len(messages)} messages from conversation")
        
        if messages:
            msg = messages[0]
            logger.info(f"Message details: ID={msg.id}, role={msg.role}")
            if msg.content:
                logger.info(f"Content count: {len(msg.content)}")
                first_content = msg.content[0]
                logger.info(f"Content type: {first_content.type}")
                logger.info(f"Content message_id: {first_content.message_id}")
                if first_content.text:
                    logger.info(f"First content text: '{first_content.text[:100]}...'")
                else:
                    logger.warning(f"Content text is None/empty")
            else:
                logger.error("❌ No content found in retrieved message!")
        else:
            logger.error("❌ No messages retrieved from conversation!")
            
        return {
            "success": True,
            "conversation_id": conv_id,
            "message_id": message_id,
            "messages_retrieved": len(messages)
        }
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    result = asyncio.run(test_message_creation())
    print(f"\nTest Result: {result}")