#!/usr/bin/env python3
"""
Simple test to verify the tools agent subgraph fix.
Test that composer can now generate final responses after tool calls.
"""

import asyncio

async def test_composer_with_short_request():
    """Test composer with a simple request that should generate a final response."""
    
    try:
        from composer import initialize_composer, compose_workflow, create_initial_state, execute_workflow
        from utils.logging import llmmllogger
        
        logger = llmmllogger.bind(component="test_fix")
        
        logger.info("🧪 Testing tools agent subgraph fix...")
        
        # Initialize composer first
        await initialize_composer()
        logger.info("✅ Composer initialized")
        
        # Set up test user and conversation
        user_id = "test_user_fix"
        conversation_id = 1
        
        # Create workflow
        logger.info("🧪 Creating workflow...")
        workflow = await compose_workflow(user_id)
        logger.info("✅ Workflow created")
        
        # Create initial state
        logger.info("🧪 Creating initial state...")
        initial_state = await create_initial_state(user_id, conversation_id)
        logger.info("✅ Initial state created")
        
        # Execute workflow with streaming
        logger.info("🧪 Executing workflow...")
        events = []
        async for event in execute_workflow(workflow, initial_state, stream=True):
            events.append(event)
            logger.info(f"📦 Event: {type(event)} - {str(event)[:200]}")
        
        logger.info(f"🧪 Workflow completed with {len(events)} events")
        
        # Check results - look for actual assistant responses in events
        logger.info("🧪 Analyzing events for assistant responses...")
        final_response_found = False
        assistant_messages = []
        
        for event in events:
            # Look for final assistant messages in the events
            if hasattr(event, 'data') and isinstance(event.data, dict):
                if 'messages' in event.data:
                    messages = event.data['messages']
                    if isinstance(messages, list):
                        for msg in messages:
                            if hasattr(msg, 'role') and getattr(msg, 'role', None) == 'assistant':
                                if hasattr(msg, 'content') and msg.content:
                                    assistant_messages.append(str(msg.content))
                                    final_response_found = True
        
        if final_response_found:
            logger.info(f"✅ Found {len(assistant_messages)} assistant response(s)")
            for i, content in enumerate(assistant_messages):
                logger.info(f"✅ Response {i+1}: {content[:100]}...")
            logger.info("🎉 TOOLS AGENT SUBGRAPH FIX APPEARS TO BE WORKING!")
            return True
        else:
            logger.error("❌ No final assistant responses found - tools agent subgraph may still need fixing")
            logger.info(f"📊 Total events processed: {len(events)}")
            # Log some event details for debugging
            for i, event in enumerate(events[:5]):  # Just first 5 events
                logger.info(f"📊 Event {i}: {type(event)} - {str(event)[:100]}")
            return False
            
    except Exception as e:
        from utils.logging import llmmllogger
        logger = llmmllogger.bind(component="test_fix")
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_composer_with_short_request())
    exit(0 if success else 1)