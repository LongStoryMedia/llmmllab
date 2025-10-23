#!/usr/bin/env python3
"""
Test script to verify structured data storage in chat completions.
This script tests that thoughts, analyses, and tool_calls are properly stored 
in the database during chat completions.
"""

import asyncio
import json
import uuid
from datetime import datetime, timezone
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="test_structured_storage")

async def test_structured_data_storage():
    """Test that structured data is properly stored during chat completions."""
    try:
        # Initialize database and composer
        from db import storage
        from models.message import Message
        from models.message_role import MessageRole
        from models.message_content import MessageContent, MessageContentType
        from server.routers.chat import store_structured_response_data
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
        
        if not storage.initialized:
            raise RuntimeError("Storage failed to initialize")
            
        logger.info("✅ Database initialized")
        
        # Create a test user and conversation (minimal setup)
        test_user_id = f"test_user_{uuid.uuid4().hex[:8]}"
        
        # Create a test conversation first
        test_conversation_id = await storage.conversation.create_conversation(test_user_id, "Test Structured Storage")
        
        if not test_conversation_id:
            raise RuntimeError("Failed to create test conversation")
            
        logger.info(f"✅ Test conversation created with ID: {test_conversation_id}")
        
        # Create a test assistant message
        test_message = Message(
            conversation_id=test_conversation_id,
            role=MessageRole.ASSISTANT,
            content=[
                MessageContent(
                    type=MessageContentType.TEXT,
                    text="This is a test response with structured data.",
                )
            ],
        )
        
        # Store the message and get the ID
        message_id = await storage.message.add_message(test_message)
        
        if not message_id:
            raise RuntimeError("Failed to store test message")
            
        logger.info(f"✅ Test message stored with ID: {message_id}")
        
        # Test structured data
        test_thinking = "This is test thinking content. The user asked about testing."
        test_structured_data = {
            "analyses": [
                {
                    "workflow_type": "chat",
                    "complexity_level": "low",
                    "confidence": 0.95,
                    "technical_domain": "testing"
                }
            ],
            "tool_calls": [
                {
                    "tool_name": "test_tool",
                    "execution_id": "test_call_1",
                    "success": True,
                    "args": {"query": "test"},
                    "result_data": {"output": "test result"},
                    "execution_time_ms": 123.45
                }
            ]
        }
        
        # Test the structured data storage function
        await store_structured_response_data(
            message_id=message_id,
            thinking_content=test_thinking,
            structured_data=test_structured_data
        )
        
        logger.info("✅ Structured data storage function executed")
        
        # Verify the data was stored by retrieving it
        thought_service = getattr(storage, 'thought', None)
        analysis_service = getattr(storage, 'analysis', None)
        tool_call_service = getattr(storage, 'tool_call', None)
        
        if thought_service:
            thoughts = await thought_service.get_thoughts_by_message(message_id)
            logger.info(f"✅ Retrieved {len(thoughts)} thoughts for message {message_id}")
            if thoughts:
                logger.info(f"   First thought: {thoughts[0].text[:50]}...")
        else:
            logger.warning("❌ Thought service not available")
            
        if analysis_service:
            analyses = await analysis_service.get_analyses_by_message(message_id)
            logger.info(f"✅ Retrieved {len(analyses)} analyses for message {message_id}")
            if analyses:
                logger.info(f"   First analysis: {json.dumps(analyses[0]['analysis_data'], indent=2)}")
        else:
            logger.warning("❌ Analysis service not available")
            
        if tool_call_service:
            tool_calls = await tool_call_service.get_tool_calls_by_message(message_id)
            logger.info(f"✅ Retrieved {len(tool_calls)} tool calls for message {message_id}")
            if tool_calls:
                logger.info(f"   First tool call: {json.dumps(tool_calls[0]['tool_data'], indent=2)}")
        else:
            logger.warning("❌ Tool call service not available")
        
        # Clean up test data
        if thought_service:
            await thought_service.delete_thoughts_by_message(message_id)
        if analysis_service:
            await analysis_service.delete_analyses_by_message(message_id)
        if tool_call_service:
            await tool_call_service.delete_tool_calls_by_message(message_id)
            
        # Clean up conversation and message
        try:
            await storage.conversation.delete_conversation(test_conversation_id)
        except Exception as e:
            logger.warning(f"Failed to cleanup test conversation: {e}")
            
        logger.info("✅ Test data cleaned up")
        logger.info("✅ Structured data storage test completed successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Structured data storage test failed: {e}")
        return False
    finally:
        if storage.pool:
            await storage.close()

if __name__ == "__main__":
    success = asyncio.run(test_structured_data_storage())
    exit(0 if success else 1)