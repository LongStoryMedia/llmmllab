#!/usr/bin/env python3
"""
Test script to verify conversation history retrieval for context assembly debugging.
Checks if both user and assistant messages are being retrieved and passed to composer.
"""

import asyncio
import json
from db import storage
from models import MessageRole
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="context_assembly_test")


async def test_conversation_history(conversation_id: int = 717):
    """Test conversation history retrieval and context assembly."""
    
    try:
        # Initialize storage - use the same connection as the app
        await storage.initialize("postgresql://lsm:7cb9c812e384e16c911a72f1066517d205e8641b78edb3b1b3c78d0c351b1885@192.168.0.71:32345/llmmll")
        
        logger.info(f"🔍 Testing conversation history for conversation {conversation_id}")
        
        # Get conversation history using the same method as composer
        messages = await storage.get_service(storage.message).get_conversation_history(conversation_id)
        
        logger.info(f"📚 Retrieved {len(messages)} messages from conversation {conversation_id}")
        
        # Analyze message types
        user_messages = [msg for msg in messages if msg.role == MessageRole.USER]
        assistant_messages = [msg for msg in messages if msg.role == MessageRole.ASSISTANT]
        
        logger.info(f"👤 User messages: {len(user_messages)}")
        logger.info(f"🤖 Assistant messages: {len(assistant_messages)}")
        
        # Print recent messages for verification
        print("\n" + "="*80)
        print(f"📋 CONVERSATION {conversation_id} HISTORY ANALYSIS")
        print("="*80)
        print(f"Total messages: {len(messages)}")
        print(f"User messages: {len(user_messages)}")
        print(f"Assistant messages: {len(assistant_messages)}")
        print("\n📝 Recent messages (last 5):")
        
        for i, msg in enumerate(messages[-5:], start=len(messages)-4):
            role_emoji = "👤" if msg.role == MessageRole.USER else "🤖"
            content_preview = ""
            if msg.content:
                content_text = msg.content[0].text if msg.content else ""
                content_preview = content_text[:100] + "..." if len(content_text) > 100 else content_text
            
            print(f"{i:2d}. {role_emoji} {msg.role.value:9} | {content_preview}")
        
        print("="*80)
        
        # Check for potential issues
        issues = []
        
        if len(assistant_messages) == 0:
            issues.append("❌ No assistant messages found - context assembly will be incomplete")
        
        if len(user_messages) == 0:
            issues.append("❌ No user messages found - invalid conversation state")
            
        # Check for messages with empty content
        empty_content_messages = [msg for msg in messages if not msg.content or not any(c.text.strip() for c in msg.content)]
        if empty_content_messages:
            issues.append(f"⚠️ {len(empty_content_messages)} messages with empty content found")
        
        # Check message chronological order
        timestamps = [msg.created_at for msg in messages if hasattr(msg, 'created_at') and msg.created_at]
        if len(timestamps) > 1 and timestamps != sorted(timestamps):
            issues.append("⚠️ Messages not in chronological order")
        
        if issues:
            print("\n🚨 ISSUES DETECTED:")
            for issue in issues:
                print(f"  {issue}")
        else:
            print("\n✅ No issues detected in conversation history")
        
        return {
            "total_messages": len(messages),
            "user_messages": len(user_messages), 
            "assistant_messages": len(assistant_messages),
            "issues": issues,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to test conversation history: {e}")
        print(f"\n❌ ERROR: {e}")
        return {"success": False, "error": str(e)}


async def main():
    """Main function to run the test."""
    result = await test_conversation_history()
    print(f"\n🎯 Test Result: {json.dumps(result, indent=2)}")


if __name__ == "__main__":
    asyncio.run(main())