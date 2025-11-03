"""
Test with a clearly engineering-focused query to ensure proper classification.
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

logger = logging.getLogger("test_clear_engineering_intent")
logging.basicConfig(level=logging.INFO)


async def test_clear_engineering_intent():
    """Test with clear engineering request that should definitely be classified as ENGINEERING."""
    try:
        # Use the global pipeline factory instance
        factory = pipeline_factory

        # Create classifier agent with minimal profile
        profile = ModelProfile(
            id=str(uuid.uuid4()),
            user_id="test-user",
            name="analysis_profile",
            model_name="qwen3-vl-2b-thinking-abliterated",  # Use coder model for better results
            parameters=ModelParameters(
                temperature=0.1, num_predict=512
            ),  # Lower temp for consistency
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

        # Test clearly engineering-focused query
        engineering_query = """I need to build a REST API with FastAPI that includes:
        1. User authentication with JWT tokens
        2. CRUD operations for a user management system  
        3. Database integration with PostgreSQL
        4. Error handling and validation
        5. Rate limiting and security middleware
        
        Please provide the complete implementation with code examples and explain the architecture."""

        messages = [
            Message(
                role=MessageRole.USER,
                content=[
                    MessageContent(type=MessageContentType.TEXT, text=engineering_query)
                ],
                message_type="text",
            )
        ]

        # Analyze intent
        logger.info("Testing clear engineering intent analysis...")
        intent_analyses = await classifier.analyze(
            messages=messages, available_static_tools=[]
        )

        logger.info(f"Received {len(intent_analyses)} intent analyses")

        for i, intent in enumerate(intent_analyses):
            logger.info(f"Intent {i+1}:")
            logger.info(f"  workflow_type: {intent.workflow_type}")
            logger.info(f"  technical_domain: {intent.technical_domain}")
            logger.info(f"  response_format: {intent.response_format}")
            logger.info(f"  complexity_level: {intent.complexity_level}")
            logger.info(f"  confidence: {intent.confidence}")

            # Check if engineering fields are populated
            if intent.workflow_type.value == "engineering":
                if intent.technical_domain:
                    logger.info(
                        "✅ SUCCESS: technical_domain is populated for engineering workflow"
                    )
                    logger.info(f"   technical_domain: {intent.technical_domain}")
                else:
                    logger.error(
                        "❌ FAILURE: technical_domain is None for engineering workflow"
                    )

                if intent.response_format:
                    logger.info(
                        "✅ SUCCESS: response_format is populated for engineering workflow"
                    )
                    logger.info(f"   response_format: {intent.response_format}")
                else:
                    logger.error(
                        "❌ FAILURE: response_format is None for engineering workflow"
                    )
            else:
                logger.error(
                    f"❌ CRITICAL FAILURE: Intent was classified as {intent.workflow_type.value}, not engineering"
                )

        return intent_analyses

    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return None


if __name__ == "__main__":
    asyncio.run(test_clear_engineering_intent())
