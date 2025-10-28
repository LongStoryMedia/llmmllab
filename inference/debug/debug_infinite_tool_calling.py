"""
Debug Infinite Tool Calling Issue

This script reproduces and analyzes the infinite tool calling issue where
the agent makes 9+ identical web searches instead of providing a final response.
"""

import asyncio
import sys
from pathlib import Path

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logging import llmmllogger

logger = llmmllogger.bind(component="infinite_tools_debug")


async def test_infinite_tools_issue():
    """Test to reproduce infinite tool calling issue."""
    
    logger.info("🧪 Testing Infinite Tool Calling Issue")
    
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
        
        # Create a simple test user
        test_user_id = "test_infinite_tools_user"
        
        # Create workflow for a simple chat that should trigger tools
        workflow = await compose_workflow(user_id=test_user_id)
        
        # Create initial state with a message that should trigger tool calls
        initial_state = await create_initial_state(
            user_id=test_user_id,
            conversation_id=999,  # Use a test conversation ID
        )
        
        # Add a message that should trigger web search tools
        from models import Message, MessageContent, MessageContentType, MessageRole
        test_message = Message(
            role=MessageRole.USER,
            content=[
                MessageContent(
                    type=MessageContentType.TEXT,
                    text="What are the current AI safety developments?"
                )
            ]
        )
        
        # Add message to initial state
        if hasattr(initial_state, 'messages') and initial_state.messages is not None:
            initial_state.messages.append(test_message)
        else:
            initial_state.messages = [test_message]
        
        logger.info("🎼 Created workflow and initial state")
        
        # Execute workflow and track tool calls
        event_count = 0
        tool_calls = []
        tool_results = []
        ai_responses = []
        errors = []
        
        logger.info("🚀 Starting workflow execution...")
        
        async for event in execute_workflow(workflow, initial_state, stream=True):
            event_count += 1
            
            # Log every 25 events to show progress
            if event_count % 25 == 0:
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
                    logger.info(f"   🔧 Tool call #{len(tool_calls)}: {tool_name} with {tool_input}")
                
                # Track tool results  
                elif "end" in event_type.lower() and "output" in data:
                    tool_output = data.get("output", "")
                    tool_results.append({
                        "output": str(tool_output)[:200] + "..." if len(str(tool_output)) > 200 else str(tool_output),
                        "event_count": event_count
                    })
                    logger.info(f"   📋 Tool result #{len(tool_results)}: {tool_results[-1]['output']}")
            
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
            
            # Stop after reasonable number of events to detect infinite loops
            if event_count > 500:
                logger.warning("   🛑 Stopping after 500 events - possible infinite loop detected")
                break
        
        logger.info(f"✅ Workflow completed after {event_count} events")
        
        # Analyze tool call patterns
        logger.info(f"📊 Analysis Results:")
        logger.info(f"   🔧 Total tool calls: {len(tool_calls)}")
        logger.info(f"   📋 Total tool results: {len(tool_results)}")
        logger.info(f"   💬 Total AI response chunks: {len(ai_responses)}")
        logger.info(f"   ❌ Total errors: {len(errors)}")
        
        # Check for identical tool calls (infinite loop indicator)
        if len(tool_calls) > 1:
            identical_calls = []
            for i, call1 in enumerate(tool_calls):
                for j, call2 in enumerate(tool_calls[i+1:], i+1):
                    if (call1["name"] == call2["name"] and 
                        call1["input"] == call2["input"]):
                        identical_calls.append((i, j, call1))
            
            logger.info(f"   🔄 Identical tool calls found: {len(identical_calls)}")
            
            if len(identical_calls) > 3:
                logger.error("❌ INFINITE LOOP DETECTED: Too many identical tool calls")
                for i, (idx1, idx2, call) in enumerate(identical_calls[:5]):  # Show first 5
                    logger.error(f"   Call #{idx1} == Call #{idx2}: {call['name']} with {call['input']}")
                return False
            
        # Check if we got a final AI response
        final_response = "".join(ai_responses)
        if len(final_response) < 50:  # Very short or no response
            logger.error("❌ NO PROPER RESPONSE: Agent didn't provide adequate final answer")
            logger.error(f"   Final response: '{final_response}'")
            return False
        
        logger.info("✅ TOOL CALLING WORKING CORRECTLY: No infinite loops detected")
        logger.info(f"   Final response length: {len(final_response)} characters")
        return True
            
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    asyncio.run(test_infinite_tools_issue())