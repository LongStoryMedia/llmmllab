#!/usr/bin/env python3
"""
Target test to reproduce the chat node tool registry bug.
This test specifically triggers the chat node execution to expose the tool registry issue.
"""

import asyncio
import sys
import traceback
from models.message import Message, MessageRole, MessageContent, MessageContentType
from composer.core.service import ComposerService
from db import storage
from utils.suppress_warnings import suppress_async_warnings

# Suppress async warnings early
suppress_async_warnings()

async def test_chat_node_tool_registry_bug():
    """
    Targeted test to reproduce the exact chat node failure.
    This should trigger the tool registry bug that causes:
    'function' object has no attribute 'name'
    """
    print("🎯 Testing Chat Node Tool Registry Bug")
    print("=" * 50)
    
    try:
        # Initialize database first
        print("💾 Initializing database...")
        import os
        db_host = os.environ.get("DB_HOST", "192.168.0.71")
        db_port = os.environ.get("DB_PORT", "32345")
        db_user = os.environ.get("DB_USER", "lsm")
        db_password = os.environ.get("DB_PASSWORD", "lsm")
        db_name = os.environ.get("DB_NAME", "llmmll")
        
        connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}?sslmode=disable"
        await storage.initialize(connection_string)
        
        # Initialize services (ComposerService initializes lazily)
        print("🚀 Initializing composer service...")
        composer = ComposerService()
        
        # Create test user and conversation
        test_user_id = "test_chat_node_user"
        print(f"👤 Creating test conversation for user: {test_user_id}")
        
        conversation_id = await storage.conversation.create_conversation(
            title="Test Chat Node Bug",
            user_id=test_user_id
        )
        print(f"💬 Created conversation: {conversation_id}")
        
        # Create test message that should trigger the workflow
        test_message = Message(
            conversation_id=conversation_id,
            role=MessageRole.USER,
            content=[
                MessageContent(
                    type=MessageContentType.TEXT,
                    text="What are some common machine learning algorithms?" # This should trigger tool collection
                )
            ]
        )
        
        print("📝 Storing test message...")
        await storage.message.add_message(test_message)
        
        print("🎼 Composing workflow...")
        workflow = await composer.compose_workflow(test_user_id)
        
        print("🏁 Creating initial state...")
        initial_state = await composer.create_initial_state(test_user_id, conversation_id)
        
        print("🚀 Executing workflow (this should trigger the tool registry bug)...")
        event_count = 0
        
        try:
            async for event in composer.execute_workflow(workflow, initial_state, stream=True):
                print(f"   📡 Event {event_count}: {event.get('event', 'unknown')} - {event.get('name', 'no_name')}")
                event_count += 1
                
                # Look specifically for chat node execution
                if 'chat' in str(event).lower() or 'tool' in str(event).lower():
                    print(f"      🔍 Tool/Chat related event: {str(event)[:200]}...")
                
                if event_count >= 20:  # Give it more events to reach the chat node
                    print(f"   ⏱️  Stopping after {event_count} events")
                    break
            
            print(f"✅ Workflow executed successfully ({event_count} events)")
            print("⚠️  WARNING: The tool registry bug may not have been triggered!")
            
        except Exception as workflow_error:
            if "'function' object has no attribute 'name'" in str(workflow_error):
                print(f"🎯 SUCCESS: Reproduced the tool registry bug!")
                print(f"   Error: {workflow_error}")
                return True
            else:
                print(f"❌ Different error occurred: {workflow_error}")
                traceback.print_exc()
                return False
        
        # Cleanup
        print("🧹 Cleaning up...")
        await storage.conversation.delete_conversation(conversation_id)
        await storage.close()
        
        return False  # Bug not reproduced
        
    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        traceback.print_exc()
        return False

async def main():
    """Run the targeted chat node test."""
    print("🔬 Chat Node Tool Registry Bug Reproduction Test")
    print("=" * 60)
    
    bug_reproduced = await test_chat_node_tool_registry_bug()
    
    print("=" * 60)
    if bug_reproduced:
        print("🎯 Bug reproduced successfully - tool registry needs fixing!")
        sys.exit(1)  # Exit with error to indicate bug found
    else:
        print("✅ Bug not reproduced - may be fixed or test needs adjustment")
        sys.exit(0)

if __name__ == "__main__":
    asyncio.run(main())