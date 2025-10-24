#!/usr/bin/env python3
"""
Test script for simulating the complete message replay workflow.
This tests the full replay scenario with bulk delete.
"""

import asyncio
import aiohttp
from datetime import datetime, timedelta
import json

async def test_replay_workflow():
    """Test the complete replay workflow with bulk delete"""
    # Test configuration
    base_url = "http://192.168.0.122:8000"
    user_id = "CgNsc20SBGxkYXA"  # Using the standard test user
    conversation_id = 717  # Using the standard test conversation
    
    headers = {
        "Content-Type": "application/json",
        "X-User-ID": user_id
    }
    
    print("🔄 Testing Complete Message Replay Workflow")
    print("=" * 60)
    print(f"Base URL: {base_url}")
    print(f"User ID: {user_id}")
    print(f"Conversation ID: {conversation_id}")
    print()
    
    async with aiohttp.ClientSession() as session:
        try:
            # Step 1: Create some test messages to work with
            print("📝 Step 1: Creating test messages...")
            test_messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "First test message"}],
                    "conversation_id": conversation_id
                },
                {
                    "role": "user", 
                    "content": [{"type": "text", "text": "Second test message"}],
                    "conversation_id": conversation_id
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Third test message - this is the one we'll replay"}],
                    "conversation_id": conversation_id
                }
            ]
            
            created_messages = []
            for i, msg_data in enumerate(test_messages):
                # Small delay between messages to ensure different timestamps
                if i > 0:
                    await asyncio.sleep(0.1)
                    
                async with session.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json=msg_data
                ) as response:
                    if response.status == 200:
                        # For streaming responses, we just verify it started
                        print(f"   ✅ Created test message {i+1}")
                    else:
                        error_text = await response.text()
                        print(f"   ❌ Failed to create message {i+1}: {response.status} - {error_text}")
                        
            # Wait a bit for messages to be processed
            await asyncio.sleep(2)
            
            # Step 2: Get all messages to see what we have
            print("\n📋 Step 2: Getting current messages...")
            async with session.get(
                f"{base_url}/chat/conversations/{conversation_id}/messages",
                headers=headers
            ) as response:
                if response.status == 200:
                    all_messages = await response.json()
                    print(f"   Found {len(all_messages)} messages total")
                    
                    # Find our test messages (they should be the most recent ones)
                    user_messages = [msg for msg in all_messages if msg.get('role') == 'user']
                    if len(user_messages) >= 3:
                        target_message = user_messages[-2]  # Second to last user message
                        print(f"   Target message for replay: ID={target_message.get('id')}, created_at={target_message.get('created_at')}")
                        
                        # Step 3: Simulate replay - bulk delete messages >= target message timestamp
                        print(f"\n🔄 Step 3: Simulating replay from message {target_message.get('id')}...")
                        target_timestamp = target_message.get('created_at')
                        
                        if not target_timestamp:
                            print("   ❌ Target message missing created_at timestamp")
                            return
                            
                        # Count expected deletions
                        messages_to_delete = [
                            msg for msg in all_messages 
                            if msg.get('created_at') and msg['created_at'] >= target_timestamp
                        ]
                        print(f"   Expected to delete: {len(messages_to_delete)} messages")
                        
                        # Perform bulk delete
                        async with session.delete(
                            f"{base_url}/chat/conversations/{conversation_id}/messages/bulk/from-timestamp",
                            headers=headers,
                            params={"from_timestamp": target_timestamp}
                        ) as response:
                            if response.status == 200:
                                result = await response.json()
                                print(f"   ✅ Bulk delete successful: {result.get('deleted_count')} messages deleted")
                            else:
                                error_text = await response.text()
                                print(f"   ❌ Bulk delete failed: {response.status} - {error_text}")
                                return
                                
                        # Step 4: Verify the replay worked
                        print("\n✅ Step 4: Verifying replay results...")
                        async with session.get(
                            f"{base_url}/chat/conversations/{conversation_id}/messages",
                            headers=headers
                        ) as response:
                            if response.status == 200:
                                remaining_messages = await response.json()
                                print(f"   Messages before replay: {len(all_messages)}")
                                print(f"   Messages after bulk delete: {len(remaining_messages)}")
                                print(f"   Messages deleted: {len(all_messages) - len(remaining_messages)}")
                                
                                # Check no messages remain with timestamp >= target
                                future_messages = [
                                    msg for msg in remaining_messages
                                    if msg.get('created_at') and msg['created_at'] >= target_timestamp
                                ]
                                
                                if not future_messages:
                                    print("   ✅ Replay successful: All messages >= target timestamp deleted")
                                    print("   🎯 Ready for message re-posting (would happen next in real replay)")
                                else:
                                    print(f"   ⚠️  Warning: {len(future_messages)} messages still exist with timestamp >= target")
                            else:
                                print(f"   ❌ Failed to verify replay: {response.status}")
                    else:
                        print("   ❌ Not enough user messages found for replay test")
                else:
                    print(f"   ❌ Failed to get messages: {response.status}")
                    
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_replay_workflow())