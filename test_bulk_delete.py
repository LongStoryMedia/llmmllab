#!/usr/bin/env python3
"""
Test script for bulk delete messages from timestamp functionality.
Tests the new bulk delete API endpoint and storage method.
"""

import asyncio
import aiohttp
from datetime import datetime, timedelta
import json

async def test_bulk_delete_api():
    """Test the bulk delete API endpoint"""
    # Test configuration
    base_url = "http://192.168.0.122:8000"
    user_id = "CgNsc20SBGxkYXA"  # Using the standard test user
    conversation_id = 717  # Using the standard test conversation
    
    headers = {
        "Content-Type": "application/json",
        "X-User-ID": user_id
    }
    
    print("🧪 Testing Bulk Delete Messages From Timestamp API")
    print("=" * 60)
    print(f"Base URL: {base_url}")
    print(f"User ID: {user_id}")
    print(f"Conversation ID: {conversation_id}")
    print()
    
    async with aiohttp.ClientSession() as session:
        try:
            # First, get current messages to see what we have
            print("📋 Getting current messages...")
            async with session.get(
                f"{base_url}/chat/conversations/{conversation_id}/messages",
                headers=headers
            ) as response:
                if response.status == 200:
                    messages = await response.json()
                    print(f"   Found {len(messages)} messages in conversation")
                    if messages:
                        print(f"   Oldest: {messages[0].get('created_at', 'N/A')}")
                        print(f"   Newest: {messages[-1].get('created_at', 'N/A')}")
                else:
                    print(f"   ❌ Failed to get messages: {response.status}")
                    return
                    
            if not messages:
                print("   No messages found - creating a test scenario would require adding messages first")
                return
                
            # Test bulk delete with a timestamp that should delete some messages
            # Use a timestamp from 1 hour ago to delete recent messages
            test_timestamp = (datetime.now() - timedelta(hours=1)).isoformat() + "Z"
            print(f"📝 Testing bulk delete for messages >= {test_timestamp}")
            
            # Count how many messages should be deleted
            messages_to_delete = [
                msg for msg in messages 
                if msg.get('created_at') and msg['created_at'] >= test_timestamp
            ]
            print(f"   Expected to delete: {len(messages_to_delete)} messages")
            
            # Perform the bulk delete
            async with session.delete(
                f"{base_url}/chat/conversations/{conversation_id}/messages/bulk/from-timestamp",
                headers=headers,
                params={"from_timestamp": test_timestamp}
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"   ✅ Bulk delete successful:")
                    print(f"      Status: {result.get('status')}")
                    print(f"      Message: {result.get('message')}")
                    print(f"      Deleted count: {result.get('deleted_count')}")
                else:
                    error_text = await response.text()
                    print(f"   ❌ Bulk delete failed: {response.status}")
                    print(f"      Error: {error_text}")
                    return
                    
            # Verify the delete worked by getting messages again
            print("📋 Verifying deletion...")
            async with session.get(
                f"{base_url}/chat/conversations/{conversation_id}/messages",
                headers=headers
            ) as response:
                if response.status == 200:
                    remaining_messages = await response.json()
                    print(f"   Messages remaining: {len(remaining_messages)}")
                    print(f"   Messages deleted: {len(messages) - len(remaining_messages)}")
                    
                    # Check that no messages remain with created_at >= test_timestamp
                    future_messages = [
                        msg for msg in remaining_messages
                        if msg.get('created_at') and msg['created_at'] >= test_timestamp
                    ]
                    if future_messages:
                        print(f"   ⚠️  Warning: {len(future_messages)} messages still exist with timestamp >= {test_timestamp}")
                    else:
                        print("   ✅ All messages with timestamp >= test_timestamp were deleted")
                else:
                    print(f"   ❌ Failed to verify deletion: {response.status}")
                    
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_bulk_delete_api())