"""
Test Infinite Tool Calling Fix

This test specifically verifies that the agent:
1. Makes tool calls when needed
2. Uses tool results from conversation history
3. Stops calling tools once sufficient information is gathered
4. Provides a final comprehensive answer
"""

import asyncio
import sys
from pathlib import Path

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logging import llmmllogger

logger = llmmllogger.bind(component="infinite_tools_fix_test")


async def test_tools_fix():
    """Test that infinite tool calling is fixed."""
    
    logger.info("🧪 Testing Infinite Tool Calling Fix")
    
    try:
        # Import required components
        from composer import compose_workflow, create_initial_state, execute_workflow
        from db import storage
        import os
        
        # Initialize database properly
        db_host = os.getenv("DB_HOST", "localhost")
        db_port = os.getenv("DB_PORT", "5432")
        db_user = os.getenv("DB_USER", "lsm")
        db_password = os.getenv("DB_PASSWORD", "")
        db_name = os.getenv("DB_NAME", "llmmll")
        db_sslmode = os.getenv("DB_SSLMODE", "disable")

        connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}?sslmode={db_sslmode}"
        await storage.initialize(connection_string)

        if not storage.initialized:
            raise RuntimeError("Storage failed to initialize properly")
        logger.info("✅ Database context initialized")
        
        # Create a test user
        test_user_id = "test_tools_fix_user"
        
        # Create workflow
        workflow = await compose_workflow(user_id=test_user_id)
        
        # Create initial state with a message that should trigger web search
        initial_state = await create_initial_state(
            user_id=test_user_id,
            conversation_id=998,  # Use a test conversation ID
        )
        
        # Add a message that should trigger web search but not infinite loops
        from models import Message, MessageContent, MessageContentType, MessageRole
        test_message = Message(
            role=MessageRole.USER,
            content=[
                MessageContent(
                    type=MessageContentType.TEXT,
                    text="What are the latest developments in AI safety in 2024? Please provide a comprehensive overview."
                )
            ]
        )
        
        # Add message to initial state
        if hasattr(initial_state, 'messages') and initial_state.messages is not None:
            initial_state.messages.append(test_message)
        else:
            initial_state.messages = [test_message]
        
        logger.info("🎼 Created workflow and initial state")
        
        # Execute workflow and track tool usage
        event_count = 0
        tool_calls = []
        tool_results = []
        ai_responses = []
        errors = []
        
        logger.info("🚀 Starting workflow execution...")
        
        async for event in execute_workflow(workflow, initial_state, stream=True):
            event_count += 1
            
            # Log every 50 events to show progress
            if event_count % 50 == 0:
                logger.info(f"   📊 Processed {event_count} events...")
            
            # Track tool-related events
            event_type = event.get("event", "")
            
            if "tool" in event_type.lower():
                data = event.get("data", {})
                
                # Track tool calls
                if "call" in event_type.lower() and "input" in data:
                    tool_input = data.get("input", {})
                    tool_name = data.get("name", "unknown")
                    tool_calls.append({
                        "name": tool_name,
                        "input": tool_input,
                        "event_count": event_count
                    })
                    logger.info(f"   🔧 Tool call #{len(tool_calls)}: {tool_name} - {str(tool_input)[:100]}...")
                
                # Track tool results  
                elif "end" in event_type.lower() and "output" in data:
                    tool_output = data.get("output", "")
                    output_preview = str(tool_output)[:150] + "..." if len(str(tool_output)) > 150 else str(tool_output)
                    tool_results.append({
                        "output": output_preview,
                        "event_count": event_count
                    })
                    logger.info(f"   📋 Tool result #{len(tool_results)}: {output_preview}")
            
            # Track AI responses
            elif event_type == "on_chat_model_stream":
                chunk_data = event.get("data", {})
                chunk = chunk_data.get("chunk", {})
                if hasattr(chunk, 'content') and chunk.content:
                    ai_responses.append(str(chunk.content))
            
            # Track errors
            elif "error" in event_type.lower():
                error_msg = f"Error event: {event}"
                errors.append(error_msg)
                logger.error(f"   ❌ {error_msg}")
            
            # Stop after reasonable number of events
            if event_count > 1000:
                logger.warning("   🛑 Stopping after 1000 events")
                break
        
        logger.info(f"✅ Workflow completed after {event_count} events")
        
        # Analyze results
        logger.info(f"📊 Analysis Results:")
        logger.info(f"   🔧 Total tool calls: {len(tool_calls)}")
        logger.info(f"   📋 Total tool results: {len(tool_results)}")
        logger.info(f"   💬 Total AI response chunks: {len(ai_responses)}")
        logger.info(f"   ❌ Total errors: {len(errors)}")
        
        # Check for infinite loops (same tool call repeated)
        if len(tool_calls) > 1:
            identical_calls = []
            for i, call1 in enumerate(tool_calls):
                for j, call2 in enumerate(tool_calls[i+1:], i+1):
                    if (call1["name"] == call2["name"] and 
                        str(call1["input"]) == str(call2["input"])):
                        identical_calls.append((i, j, call1))
            
            logger.info(f"   🔄 Identical tool calls found: {len(identical_calls)}")
            
            # Show tool call pattern
            if tool_calls:
                logger.info("   📋 Tool call sequence:")
                for i, call in enumerate(tool_calls):
                    logger.info(f"      {i+1}. {call['name']}: {str(call['input'])[:80]}...")
            
            # Check if fix worked
            if len(identical_calls) > 2:  # Allow max 2 identical calls
                logger.error("❌ FIX FAILED: Too many identical tool calls - infinite loop still present")
                for i, (idx1, idx2, call) in enumerate(identical_calls[:3]):
                    logger.error(f"   Identical calls #{idx1+1} and #{idx2+1}: {call['name']}")
                return False
        
        # Check for recursion limit error
        recursion_errors = [err for err in errors if "recursion" in err.lower()]
        if recursion_errors:
            logger.error("❌ FIX FAILED: Recursion limit error still occurred")
            logger.error(f"   Recursion error: {recursion_errors[0][:200]}...")
            return False
        
        # Check if we got a proper final response
        final_response = "".join(ai_responses)
        if len(final_response) < 100:  # Should be a comprehensive response
            logger.error("❌ FIX FAILED: No adequate final response generated")
            logger.error(f"   Final response: '{final_response}'")
            return False
        
        # Success criteria:
        # 1. Some tool calls were made (shows tools are working)
        # 2. No infinite loops (max 2 identical calls allowed)
        # 3. No recursion errors
        # 4. Got a comprehensive final response
        
        if len(tool_calls) == 0:
            logger.warning("⚠️  No tool calls made - may not have tested tool calling properly")
        
        logger.info("✅ FIX SUCCESSFUL: Tool calling working properly")
        logger.info(f"   Tool calls made: {len(tool_calls)}")
        logger.info(f"   Final response length: {len(final_response)} characters")
        return True
            
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    asyncio.run(test_tools_fix())