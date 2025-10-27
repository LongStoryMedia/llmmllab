"""
Test script for the new transaction-based multi-query architecture.
Demonstrates proper transaction handling across multiple storage operations.
"""

import asyncio
import json
from datetime import datetime
from typing import Optional

from models.message import Message
from models.message_content import MessageContent
from models.message_content_type import MessageContentType
from models.message_role import MessageRole
from models.intent_analysis import IntentAnalysis
from models.workflow_type import WorkflowType
from models.complexity_level import ComplexityLevel
from models.required_capability import RequiredCapability
from models.computational_requirement import ComputationalRequirement
from db import storage


async def test_transaction_architecture():
    """
    Test the new transaction-based architecture with proper multi-query handling.
    """
    print("🧪 Testing Transaction-Based Multi-Query Architecture")
    print("=" * 60)
    
    # Initialize storage
    try:
        await storage.initialize("postgresql://lsm:listenbrainz@localhost:5432/llmmll")
        print("✅ Database initialized successfully")
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return
    
    # Test 1: Create a message with transaction support
    print("\n📝 Test 1: Create message with transaction support")
    try:
        # Create a test message with content
        message = Message(
            id=None,
            conversation_id=1,  # Using existing conversation
            role=MessageRole.USER,
            content=[
                MessageContent(
                    type=MessageContentType.TEXT,
                    text="Test message for transaction architecture"
                )
            ],
            created_at=datetime.utcnow()
        )
        
        # Use the new transaction-aware add_message method
        message_id = await storage.message.add_message(message)
        
        if message_id:
            print(f"✅ Created message {message_id} with transaction support")
            
            # Test 2: Retrieve message using new multi-query approach
            print(f"\n🔍 Test 2: Retrieve message {message_id} using multi-query")
            retrieved_message = await storage.message.get_message(message_id)
            
            if retrieved_message:
                print(f"✅ Retrieved message successfully")
                print(f"   - ID: {retrieved_message.id}")
                print(f"   - Role: {retrieved_message.role}")
                print(f"   - Content count: {len(retrieved_message.content) if retrieved_message.content else 0}")
                print(f"   - Created at: {retrieved_message.created_at}")
            else:
                print("❌ Failed to retrieve message")
                
        else:
            print("❌ Failed to create message")
            
    except Exception as e:
        print(f"❌ Message operations failed: {e}")
    
    # Test 3: Test connection parameter passing
    print(f"\n🔗 Test 3: Test explicit connection handling")
    try:
        async with storage.message.typed_pool.acquire() as conn:
            async with conn.transaction():
                # Create message using explicit connection
                test_message = Message(
                    id=None,
                    conversation_id=1,
                    role=MessageRole.ASSISTANT, 
                    content=[
                        MessageContent(
                            type=MessageContentType.TEXT,
                            text="Message created with explicit connection"
                        )
                    ],
                    created_at=datetime.utcnow()
                )
                
                message_id = await storage.message.add_message(test_message, conn=conn)
                
                if message_id:
                    print(f"✅ Created message {message_id} with explicit connection")
                    
                    # Retrieve using same connection
                    retrieved = await storage.message.get_message(message_id, conn=conn)
                    if retrieved:
                        print(f"✅ Retrieved message using same connection")
                    else:
                        print("❌ Failed to retrieve with same connection")
                else:
                    print("❌ Failed to create message with explicit connection")
                    
    except Exception as e:
        print(f"❌ Connection handling test failed: {e}")
    
    print("\n🏁 Transaction Architecture Testing Complete")


if __name__ == "__main__":
    asyncio.run(test_transaction_architecture())