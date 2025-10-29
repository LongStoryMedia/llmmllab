#!/usr/bin/env python3
"""
Test intent analysis XML wrapping and streaming filtering.

This test verifies that:
1. ClassifierAgent wraps its output in <intent-analysis> tags
2. The output is properly parsed after XML extraction
3. The StreamingResponseState can filter this content when streaming
"""

import asyncio
import logging
from typing import List

from composer.agents.classifier_agent import ClassifierAgent
from runner.pipeline_factory import pipeline_factory
from langchain_core.messages import HumanMessage
from models.intent_analysis import IntentAnalysis
from models.tool import Tool
from server.streaming_response_state import StreamingResponseState, StreamingState
from models import ModelProfile, NodeMetadata
from models.default_model_profiles import DEFAULT_ANALYSIS_PROFILE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_intent_analysis_xml_wrapping():
    """Test that ClassifierAgent properly wraps output in XML tags."""
    logger.info("🧪 Testing ClassifierAgent XML wrapping...")
    
    try:
        # Set up classifier agent
        node_metadata = NodeMetadata(
            node_name="TestClassifierAgent", 
            node_id="test_classifier",
            node_type="agent",
            execution_id="test-123"
        )
        
        classifier_agent = ClassifierAgent(
            pipeline_factory=pipeline_factory,
            profile=DEFAULT_ANALYSIS_PROFILE,
            node_metadata=node_metadata
        )
        
        # Prepare test data
        messages = [HumanMessage(content="What is the weather like today?")]
        available_tools = [
            Tool(
                name="weather_api",
                description="Get current weather information for a location",
                parameters_schema={},
                enabled=True
            )
        ]
        
        logger.info("📝 Running intent analysis...")
        intent_analyses: List[IntentAnalysis] = await classifier_agent.analyze(
            messages=messages,
            available_static_tools=available_tools
        )
        
        logger.info(f"✅ Intent analysis completed successfully")
        logger.info(f"📊 Found {len(intent_analyses)} intents:")
        
        for i, intent in enumerate(intent_analyses, 1):
            logger.info(f"  {i}. {intent.description}")
            logger.info(f"     - Workflow: {intent.workflow_type}")
            logger.info(f"     - Complexity: {intent.complexity_level}")
            logger.info(f"     - Tools needed: {intent.requires_tools}")
            logger.info(f"     - Custom tools: {intent.requires_custom_tools}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Intent analysis test failed: {e}")
        raise


async def test_streaming_response_state_filtering():
    """Test that StreamingResponseState properly filters intent-analysis content."""
    logger.info("🌊 Testing StreamingResponseState filtering...")
    
    try:
        streaming_state = StreamingResponseState()
        
        # Test content that should be filtered out
        test_chunk = """
<intent-analysis>
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
</intent-analysis>
"""
        
        logger.info("📝 Testing streaming filter with intent analysis content...")
        response = streaming_state.process_chunk(test_chunk)
        
        # Check that response is empty (filtered out)
        if response.content.strip() == "":
            logger.info("✅ Intent analysis content properly filtered out")
            logger.info(f"📊 Current streaming state: {streaming_state.state}")
            logger.info(f"🔍 Intent analysis buffer length: {len(streaming_state.intent_analysis_buffer)}")
        else:
            logger.error(f"❌ Expected empty response, got: '{response.content}'")
            return False
        
        # Test regular content that should pass through
        regular_chunk = "This is regular chat content that should appear in the UI."
        response = streaming_state.process_chunk(regular_chunk)
        
        if response.content.strip() == regular_chunk:
            logger.info("✅ Regular content properly passed through")
        else:
            logger.error(f"❌ Expected '{regular_chunk}', got: '{response.content}'")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ StreamingResponseState test failed: {e}")
        raise


async def main():
    """Run all intent analysis XML wrapping tests."""
    logger.info("🚀 Starting intent analysis XML wrapping tests...")
    
    try:
        # Test 1: ClassifierAgent XML wrapping
        await test_intent_analysis_xml_wrapping()
        
        # Test 2: StreamingResponseState filtering
        await test_streaming_response_state_filtering()
        
        logger.info("🎉 All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"💥 Test suite failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())