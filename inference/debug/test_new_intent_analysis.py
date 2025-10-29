"""
Test script to verify that NEW intent analyses don't leak to message content.
This will create a fresh conversation and check if intent analysis appears in the message.
"""

import asyncio
# Config import removed to simplify test
from composer.graph.chat_router import ChatRouter
from models import Message, MessageContent, MessageContentType, MessageRole
from server.storage.conversation_storage import ConversationStorage
from db import init_db
import uuid


async def test_new_intent_analysis():
    """Test that NEW intent analyses don't leak to message content."""
    print("🧪 Testing new intent analysis leakage prevention...")
    
    # Initialize database 
    await init_db()
    
    # Initialize storage
    storage = ConversationStorage()
    
    # Create a new conversation
    conversation_id = str(uuid.uuid4())
    user_id = "test_user"
    
    # Create a test message that should trigger intent analysis
    test_message = Message(
        role=MessageRole.USER,
        content=[MessageContent(
            type=MessageContentType.TEXT,
            text="What are the latest developments in AI? I want to know about major model releases, recent research breakthroughs, and any safety concerns that have been raised."
        )]
    )
    
    print(f"📝 Creating conversation {conversation_id}")
    print(f"📤 Sending message: {test_message.content[0].text[:100]}...")
    
    # Create ChatRouter and process the message  
    chat_router = ChatRouter()
    
    # Process the message (this should trigger intent analysis)
    response_generator = chat_router.route_message(
        message=test_message,
        conversation_id=conversation_id,
        user_id=user_id,
        config_override=None  # Use default config
    )
    
    # Collect all response chunks
    print("🌊 Collecting response chunks...")
    response_chunks = []
    async for chunk in response_generator:
        response_chunks.append(chunk)
        if chunk.done:
            break
    
    print(f"📊 Got {len(response_chunks)} response chunks")
    
    # Get the final message from storage
    conversations = await storage.get_conversations(user_id)
    if not conversations:
        print("❌ No conversations found!")
        return False
        
    target_conversation = None
    for conv in conversations:
        if conv.id == conversation_id:
            target_conversation = conv
            break
            
    if not target_conversation:
        print("❌ Test conversation not found!")
        return False
        
    messages = await storage.get_messages(conversation_id)
    assistant_messages = [m for m in messages if m.role == MessageRole.ASSISTANT]
    
    if not assistant_messages:
        print("❌ No assistant messages found!")
        return False
        
    # Check the latest assistant message content
    latest_assistant = assistant_messages[-1]
    message_text = ""
    for content in latest_assistant.content:
        if content.type == MessageContentType.TEXT:
            message_text += content.text
            
    print(f"📄 Assistant message content (first 500 chars):")
    print(f"'{message_text[:500]}{'...' if len(message_text) > 500 else ''}'")
    
    # Check if intent analysis JSON appears in the message content
    if '"intents"' in message_text or '"workflow_type"' in message_text:
        print("❌ INTENT ANALYSIS JSON STILL LEAKING TO MESSAGE CONTENT!")
        print(f"🔍 Found JSON in message: {message_text[:200]}...")
        return False
    else:
        print("✅ Intent analysis JSON NOT found in message content - leak prevention working!")
        return True


if __name__ == "__main__":
    result = asyncio.run(test_new_intent_analysis())
    if result:
        print("🎉 NEW INTENT ANALYSIS LEAK PREVENTION: WORKING ✅")
    else:
        print("💥 NEW INTENT ANALYSIS LEAK PREVENTION: FAILED ❌")