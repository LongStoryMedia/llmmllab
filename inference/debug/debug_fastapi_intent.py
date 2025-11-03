"""
Debug specific intent analysis output for FastAPI REST API request.
"""
import asyncio
from runner.pipeline_factory import pipeline_factory
from composer.agents.classifier_agent import ClassifierAgent
from models import (
    Message,
    ModelProfile,
    NodeMetadata,
    ModelParameters,
    MessageContent,
    MessageContentType,
    MessageRole,
)
from datetime import datetime, timezone
import logging
import uuid

logger = logging.getLogger("debug_fastapi_intent")
logging.basicConfig(level=logging.INFO)


async def debug_fastapi_intent():
    """Debug the FastAPI REST API intent analysis."""
    try:
        # Use the global pipeline factory instance
        factory = pipeline_factory

        # Create classifier agent
        profile = ModelProfile(
            id=str(uuid.uuid4()),
            user_id="test-user",
            name="analysis_profile",
            model_name="qwen3-vl-2b-thinking-abliterated",
            parameters=ModelParameters(temperature=0.1, num_predict=512),
            system_prompt="You are an intent analysis system.",
            type=1,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        node_metadata = NodeMetadata(
            node_id="test_classifier",
            node_name="test_classifier",
            node_type="classifier",
            execution_order=1,
        )

        classifier = ClassifierAgent(factory, profile, node_metadata)

        # Test the exact query that's causing issues
        query = "I need help building a REST API in Python using FastAPI. Can you provide a complete implementation with proper error handling, authentication, and database integration?"

        messages = [
            Message(
                role=MessageRole.USER,
                content=[MessageContent(type=MessageContentType.TEXT, text=query)],
                message_type="text"
            )
        ]

        # Analyze intent
        logger.info("Testing FastAPI REST API intent analysis...")
        logger.info(f"Query: {query}")
        
        intent_analyses = await classifier.analyze(messages=messages, available_static_tools=[])

        logger.info(f"Received {len(intent_analyses)} intent analyses")

        for i, intent in enumerate(intent_analyses):
            logger.info(f"Intent {i+1}:")
            logger.info(f"  workflow_type: {intent.workflow_type}")
            logger.info(f"  technical_domain: {intent.technical_domain}")
            logger.info(f"  response_format: {intent.response_format}")
            logger.info(f"  complexity_level: {intent.complexity_level}")
            logger.info(f"  confidence: {intent.confidence}")
            logger.info(f"  required_capabilities: {intent.required_capabilities}")

            # Check if engineering fields are populated
            if intent.workflow_type.value == "engineering":
                logger.info("✅ Workflow correctly classified as ENGINEERING")
                
                if intent.technical_domain:
                    logger.info(f"✅ technical_domain populated: {intent.technical_domain}")
                else:
                    logger.error("❌ technical_domain is None!")
                    
                if intent.response_format:
                    logger.info(f"✅ response_format populated: {intent.response_format}")
                else:
                    logger.error("❌ response_format is None!")
                    logger.error("This is causing the engineering agent to skip execution!")
            else:
                logger.error(f"❌ Workflow misclassified as {intent.workflow_type.value}, should be engineering")

        return intent_analyses

    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


if __name__ == "__main__":
    asyncio.run(debug_fastapi_intent())