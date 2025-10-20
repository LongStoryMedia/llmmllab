#!/usr/bin/env python3
"""
Debug the exact chat workflow to identify duplication and tool call issues
"""

import asyncio
from composer import execute_workflow, compose_workflow, create_initial_state
from models import Message, MessageRole, MessageContent, MessageContentType


async def debug_chat_workflow():
    """Debug the exact workflow path the UI uses."""
    print("🧪 DEBUGGING CHAT WORKFLOW...")
    
    # Create the same message structure as UI
    test_message = Message(
        role=MessageRole.USER,
        content=[
            MessageContent(
                type=MessageContentType.TEXT,
                text="What are the most commonly used shallow learning algorithms in 2025? Provide examples."
            )
        ],
        conversation_id=717
    )
    
    try:
        # Setup workflow same as chat router
        user_id = "CgNsc20SBGxkYXA"
        
        workflow = await compose_workflow(user_id)
        initial_state = await create_initial_state(
            message=test_message,
            user_id=user_id
        )
        
        print("✅ Workflow setup complete")
        print(f"   Initial state keys: {list(initial_state.__dict__.keys()) if hasattr(initial_state, '__dict__') else 'N/A'}")
        
        # Execute with streaming to see what events come through
        print("📡 STREAMING EVENTS:")
        print("=" * 80)
        
        event_count = 0
        chat_model_stream_count = 0
        final_state = None
        
        async for event in execute_workflow(workflow, initial_state, stream=True):
            event_count += 1
            
            if isinstance(event, dict):
                event_type = event.get("event", "")
                event_data = event.get("data", {})
                
                print(f"[{event_count:03d}] Event: {event_type}")
                
                if event_type == "on_chat_model_stream":
                    chat_model_stream_count += 1
                    chunk = event_data.get("chunk", {})
                    
                    content = ""
                    if isinstance(chunk, dict):
                        content = chunk.get("content", "")
                    elif hasattr(chunk, "content"):
                        content = str(chunk.content)
                    
                    print(f"      💬 Chat stream #{chat_model_stream_count}: '{content[:50]}{'...' if len(content) > 50 else ''}'")
                    
                elif event_type == "on_chain_end":
                    final_state = event_data.get("output", {})
                    print("      🏁 Chain end - final state captured")
                    
                elif event_type in ["on_chain_stream", "on_tool_end", "on_chain_start"]:
                    output = event_data.get("output", {})
                    print(f"      📦 {event_type}: {type(output)}")
                    
                    if isinstance(output, dict):
                        messages = output.get("messages", [])
                        if messages:
                            print(f"         📝 Messages in output: {len(messages)}")
                            for i, msg in enumerate(messages[-2:]):  # Show last 2
                                if isinstance(msg, dict):
                                    msg_type = msg.get("type", "?")
                                    content = str(msg.get("content", ""))[:30]
                                    print(f"            [{i}] {msg_type}: {content}...")
        
        print("📊 SUMMARY:")
        print(f"   Total events: {event_count}")
        print(f"   Chat model stream events: {chat_model_stream_count}")
        print(f"   Final state available: {final_state is not None}")
        
        if final_state:
            messages = final_state.get("messages", [])
            print(f"   Final state messages: {len(messages)}")
            
            if messages:
                last_msg = messages[-1]
                if isinstance(last_msg, dict):
                    content = str(last_msg.get("content", ""))
                    has_tool_calls = "tool_calls" in last_msg
                    print(f"   Last message content length: {len(content)}")
                    print(f"   Last message has tool_calls: {has_tool_calls}")
                    
                    # Check for function-call markup in content
                    if "<function-call>" in content:
                        print("   ⚠️  Found <function-call> markup in content!")
                    if "<tool_call>" in content:
                        print("   ⚠️  Found <tool_call> markup in content!")
        
    except Exception as e:
        print(f"💥 Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(debug_chat_workflow())