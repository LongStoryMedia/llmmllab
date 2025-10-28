"""
Test Assistant Token Loop Fix

This test verifies that the fix for infinite "assistant" token generation works correctly.
"""

import asyncio
import sys
from pathlib import Path

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logging import llmmllogger

logger = llmmllogger.bind(component="assistant_loop_test")


async def test_assistant_loop_fix():
    """Test that assistant token loops are prevented."""
    
    logger.info("🧪 Testing Assistant Token Loop Fix")
    
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
        test_user_id = "test_assistant_loop_user"
        
        # Create workflow for a simple chat
        workflow = await compose_workflow(user_id=test_user_id)
        
        # Create initial state with a message that should trigger tool calls
        initial_state = await create_initial_state(
            user_id=test_user_id,
            conversation_id=999,  # Use a test conversation ID
        )
        
        logger.info("🎼 Created workflow and initial state")
        
        # Execute workflow and capture events
        event_count = 0
        assistant_responses = []
        errors = []
        
        logger.info("🚀 Starting workflow execution...")
        
        async for event in execute_workflow(workflow, initial_state, stream=True):
            event_count += 1
            
            # Log every 50 events to show progress
            if event_count % 50 == 0:
                logger.info(f"   📊 Processed {event_count} events...")
            
            # Capture AI message chunks for analysis
            if event.get("event") == "on_chat_model_stream":
                chunk_data = event.get("data", {})
                chunk = chunk_data.get("chunk", {})
                if hasattr(chunk, 'content') and chunk.content:
                    content = str(chunk.content)
                    assistant_responses.append(content)
                    
                    # Check for assistant token loops
                    if "assistant" in content.lower():
                        logger.warning(f"   ⚠️  Found 'assistant' in content: {content[:100]}...")
                        
                        # Count consecutive "assistant" tokens
                        assistant_count = content.lower().count("assistant")
                        if assistant_count > 2:
                            error_msg = f"ASSISTANT LOOP DETECTED: {assistant_count} 'assistant' tokens in single chunk"
                            logger.error(f"   ❌ {error_msg}")
                            errors.append(error_msg)
            
            # Check for error events
            elif "error" in event.get("event", "").lower():
                error_msg = f"Workflow error: {event}"
                logger.error(f"   ❌ {error_msg}")
                errors.append(error_msg)
            
            # Stop after reasonable number of events to prevent infinite loops
            if event_count > 1000:
                logger.warning("   🛑 Stopping after 1000 events to prevent infinite execution")
                break
        
        logger.info(f"✅ Workflow completed after {event_count} events")
        
        # Analyze results
        total_assistant_content = "".join(assistant_responses)
        assistant_token_count = total_assistant_content.lower().count("assistant")
        
        logger.info(f"📊 Analysis Results:")
        logger.info(f"   📝 Total assistant response chunks: {len(assistant_responses)}")
        logger.info(f"   🔤 Total 'assistant' token occurrences: {assistant_token_count}")
        logger.info(f"   ❌ Errors detected: {len(errors)}")
        
        # Check if fix worked
        if assistant_token_count > 10:  # Allow a few occurrences but not loops
            logger.error("❌ FIX FAILED: Too many 'assistant' tokens detected")
            for error in errors[-5:]:  # Show last 5 errors
                logger.error(f"   {error}")
            return False
        elif errors:
            logger.error("❌ FIX FAILED: Errors occurred during execution")
            for error in errors[-5:]:
                logger.error(f"   {error}")
            return False
        else:
            logger.info("✅ FIX SUCCESSFUL: No assistant token loops detected")
            return True
            
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    asyncio.run(test_assistant_loop_fix())