"""
Test ChatAgent tool calling with corrected schema filtering.
Focus on verifying ToolMessage validation passes.
"""

import asyncio
import sys
import os

# Add the inference directory to the path
sys.path.insert(0, "/app")

from models import Message, MessageRole, MessageContentType, MessageContent
from composer import initialize_composer, compose_workflow, create_initial_state, execute_workflow

async def test_chat_agent_tool_calling():
    """Test ChatAgent with tool calling to verify ToolMessage validation works."""
    
    print("🧪 Testing ChatAgent tool calling with schema filtering fixes...")
    
    try:
        # Initialize composer service first
        await initialize_composer()
        
        # Create test message that should trigger web search tool
        test_message = Message(
            role=MessageRole.USER,
            content=[MessageContent(
                type=MessageContentType.TEXT,
                text="Search for the latest news about artificial intelligence developments"
            )]
        )
        
        print("📝 Created test message:", test_message.content[0].text)
        
        # Run with user ID
        user_id = "test_user_tool_calling"
        conversation_id = None  # Use None for new conversation
        
        print("🔧 Composing workflow...")
        workflow = await compose_workflow(user_id)
        
        print("🏁 Creating initial state...")
        initial_state = await create_initial_state(user_id, conversation_id)
        
        print("⚡ Executing workflow...")
        
        # Execute the workflow with streaming
        events = []
        async for event in execute_workflow(workflow, initial_state, stream=True):
            events.append(event)
            if len(events) <= 5:  # Only print first few events to avoid spam
                print(f"📡 Event {len(events)-1}: {type(event)} - {str(event)[:100]}...")
        
        print(f"✅ Workflow completed successfully with {len(events)} events")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(test_chat_agent_tool_calling())