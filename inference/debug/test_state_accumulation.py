#!/usr/bin/env python3
"""
Test for state accumulation fix - check if tools and messages grow exponentially.
"""

import asyncio
import time

# Import the necessary components
from composer import compose_workflow, create_initial_state, execute_workflow

async def test_state_accumulation():
    """Test if state accumulation is fixed."""
    print("🧪 Testing state accumulation fix...")
    
    try:
        # Create a simple test user
        user_id = "test_accumulation_user"
        
        # Create workflow
        workflow = await compose_workflow(user_id=user_id)
        print(f"✅ Workflow created")
        
        # Create simple initial state with minimal data
        initial_state = await create_initial_state(
            user_id=user_id,
            conversation_id=999,
        )
        
        print(f"📊 Initial state - Messages: {len(initial_state.messages)}, Tools: {len(initial_state.available_tools)}")
        
        # Track state changes during execution
        message_counts = []
        tool_counts = []
        event_count = 0
        
        print("🔄 Starting workflow execution...")
        start_time = time.time()
        
        async for event in execute_workflow(workflow, initial_state, stream=True):
            event_count += 1
            
            # Every 50 events, check state size
            if event_count % 50 == 0:
                # Try to extract state from event
                if hasattr(event, 'values') and event.values:
                    state = list(event.values())[0]
                    if hasattr(state, 'messages') and hasattr(state, 'available_tools'):
                        msg_count = len(state.messages) if state.messages else 0
                        tool_count = len(state.available_tools) if state.available_tools else 0
                        message_counts.append(msg_count)
                        tool_counts.append(tool_count)
                        print(f"📊 Event {event_count}: Messages={msg_count}, Tools={tool_count}")
            
            # Stop after reasonable number of events to avoid infinite execution
            if event_count > 500:
                print("⏹️ Stopping test after 500 events")
                break
                
            # Stop if execution takes too long  
            if time.time() - start_time > 60:
                print("⏹️ Stopping test after 60 seconds")
                break
        
        execution_time = time.time() - start_time
        print(f"⏱️ Execution time: {execution_time:.2f}s")
        print(f"📊 Total events processed: {event_count}")
        
        # Analyze accumulation patterns
        if message_counts and tool_counts:
            print("\n📈 State Growth Analysis:")
            print(f"Messages: {message_counts[0]} → {message_counts[-1]} (growth: {message_counts[-1] - message_counts[0]})")  
            print(f"Tools: {tool_counts[0]} → {tool_counts[-1]} (growth: {tool_counts[-1] - tool_counts[0]})")
            
            # Check for exponential growth (bad)
            if len(message_counts) >= 2:
                msg_growth_rate = message_counts[-1] / max(message_counts[0], 1)
                tool_growth_rate = tool_counts[-1] / max(tool_counts[0], 1) 
                
                print(f"\n📊 Growth Rates:")
                print(f"Messages growth: {msg_growth_rate:.1f}x")
                print(f"Tools growth: {tool_growth_rate:.1f}x")
                
                # Determine if fix worked
                if msg_growth_rate > 10 or tool_growth_rate > 10:
                    print("❌ FAIL: State still growing exponentially!")
                    return False
                elif msg_growth_rate < 3 and tool_growth_rate < 3:
                    print("✅ SUCCESS: State accumulation under control!")
                    return True
                else:
                    print("⚠️ PARTIAL: Some improvement but still concerning growth")
                    return False
        else:
            print("⚠️ No state samples collected")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function."""
    print("🚀 Starting State Accumulation Test")
    print("="*50)
    
    success = await test_state_accumulation()
    
    print("\n" + "="*50) 
    if success:
        print("🎉 State accumulation fix SUCCESSFUL!")
    else:
        print("💥 State accumulation fix FAILED!")
    
    return 0 if success else 1

if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))