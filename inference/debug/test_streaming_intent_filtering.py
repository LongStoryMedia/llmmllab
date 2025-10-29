#!/usr/bin/env python3
"""
Test streaming response state intent analysis filtering.

This test verifies that the StreamingResponseState can properly filter
intent-analysis XML tags when they appear in streaming content.
"""

import asyncio
import logging

from server.streaming_response_state import StreamingResponseState, StreamingState

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_streaming_response_state_filtering():
    """Test that StreamingResponseState properly filters intent-analysis content."""
    logger.info("🌊 Testing StreamingResponseState filtering...")
    
    try:
        streaming_state = StreamingResponseState()
        
        # Test content that should be filtered out
        test_chunk = """<intent-analysis>
{
  "intents": [
    {
      "description": "Get weather information",
      "workflow_type": "simple_query",
      "complexity_level": "low",
      "confidence": 0.95,
      "requires_tools": true,
      "requires_custom_tools": false,
      "tool_complexity_score": 0.2
    }
  ]
}
</intent-analysis>"""
        
        logger.info("📝 Testing streaming filter with intent analysis content...")
        response = streaming_state.process_chunk(test_chunk)
        
        # Check that response is empty (filtered out)
        response_content = ""
        if response.message and response.message.content:
            response_content = "".join([str(c.text) for c in response.message.content if hasattr(c, 'text')])
        
        if response_content.strip() == "":
            logger.info("✅ Intent analysis content properly filtered out")
            logger.info(f"📊 Current streaming state: {streaming_state.state}")
            logger.info(f"🔍 Intent analysis buffer length: {len(streaming_state.intent_analysis_buffer)}")
        else:
            logger.error(f"❌ Expected empty response, got: '{response_content}'")
            return False
        
        # Test regular content that should pass through
        regular_chunk = "This is regular chat content that should appear in the UI."
        response = streaming_state.process_chunk(regular_chunk)
        
        if response.content.strip() == regular_chunk:
            logger.info("✅ Regular content properly passed through")
        else:
            logger.error(f"❌ Expected '{regular_chunk}', got: '{response.content}'")
            return False
            
        # Test mixed content (intent analysis followed by regular content)
        mixed_chunk = """<intent-analysis>
{
  "intents": [
    {
      "description": "Mixed content test",
      "workflow_type": "simple_query",  
      "complexity_level": "low",
      "confidence": 0.90
    }
  ]
}
</intent-analysis>

This is the response the user should see."""
        
        logger.info("📝 Testing mixed content (intent + response)...")
        
        # Process mixed content chunk by chunk to simulate streaming
        lines = mixed_chunk.split('\n')
        collected_responses = []
        
        for line in lines:
            response = streaming_state.process_chunk(line + '\n')
            if response.content.strip():
                collected_responses.append(response.content)
        
        # Should only see the regular content, not the intent analysis
        full_response = ''.join(collected_responses).strip()
        expected_response = "This is the response the user should see."
        
        if expected_response in full_response:
            logger.info(f"✅ Mixed content properly filtered: '{full_response}'")
        else:
            logger.error(f"❌ Expected '{expected_response}' in response, got: '{full_response}'")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ StreamingResponseState test failed: {e}")
        raise


async def test_intent_analysis_state_transitions():
    """Test that the streaming state properly transitions in and out of INTENT_ANALYSIS mode."""
    logger.info("🔄 Testing intent analysis state transitions...")
    
    try:
        streaming_state = StreamingResponseState()
        
        # Initially should be in NORMAL state
        assert streaming_state.state == StreamingState.NORMAL
        logger.info("✅ Initial state is NORMAL")
        
        # Process opening tag
        response = streaming_state.process_chunk("<intent-analysis>")
        assert streaming_state.state == StreamingState.INTENT_ANALYSIS
        logger.info("✅ State changed to INTENT_ANALYSIS after opening tag")
        
        # Process content inside tags (should be filtered)
        response = streaming_state.process_chunk('{"test": "content"}')
        assert streaming_state.state == StreamingState.INTENT_ANALYSIS
        assert response.content == ""
        logger.info("✅ Content filtered while in INTENT_ANALYSIS state")
        
        # Process closing tag
        response = streaming_state.process_chunk("</intent-analysis>")
        assert streaming_state.state == StreamingState.NORMAL
        logger.info("✅ State changed back to NORMAL after closing tag")
        
        # Process regular content after (should pass through)
        response = streaming_state.process_chunk("Regular content after intent analysis.")
        assert streaming_state.state == StreamingState.NORMAL
        assert "Regular content after intent analysis." in response.content
        logger.info("✅ Regular content passes through after intent analysis")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ State transition test failed: {e}")
        raise


async def main():
    """Run all streaming response state tests."""
    logger.info("🚀 Starting StreamingResponseState intent analysis tests...")
    
    try:
        # Test 1: Basic filtering
        await test_streaming_response_state_filtering()
        
        # Test 2: State transitions
        await test_intent_analysis_state_transitions()
        
        logger.info("🎉 All StreamingResponseState tests completed successfully!")
        
    except Exception as e:
        logger.error(f"💥 Test suite failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())