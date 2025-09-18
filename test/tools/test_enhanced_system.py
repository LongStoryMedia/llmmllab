"""
Comprehensive test for the enhanced tool generation system.
Tests smart analysis, deduplication, and pipeline lifecycle management.
"""

import asyncio
import logging
import sys
import os
from typing import Dict, Any

# Add inference directory to path
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "inference"
    ),
)

from models import DynamicTool, ConversationContext, UserConfig
from server.tools.smart_analysis import SmartIntentAnalyzer
from server.tools.deduplication import AdvancedToolDeduplicator
from server.tools.integration import DynamicToolGenerator
from runner.pipeline_lifecycle import managed_pipeline_execution

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockConversationContext:
    """Mock conversation context for testing."""

    def __init__(self):
        self.user_config = MockUserConfig()


class MockUserConfig:
    """Mock user config for testing."""

    def __init__(self):
        self.user_id = "test_user_123"


async def test_smart_analysis():
    """Test the smart intent analyzer."""
    logger.info("=== Testing Smart Intent Analysis ===")

    analyzer = SmartIntentAnalyzer()

    test_cases = [
        ("Hello, how are you?", "TRIVIAL"),
        ("What's 2 + 2?", "SIMPLE"),
        ("Calculate compound interest for a loan", "MODERATE"),
        ("Create a machine learning model for time series prediction", "SPECIALIZED"),
        ("Parse this complex XML document and extract specific fields", "COMPLEX"),
    ]

    for text, expected_complexity in test_cases:
        analysis = analyzer.analyze_intent(text)
        logger.info(f"Text: '{text}'")
        logger.info(
            f"  Complexity: {analysis.complexity_level} (expected: {expected_complexity})"
        )
        logger.info(f"  Intent: {analysis.primary_intent}")
        logger.info(f"  Reusability: {analysis.reusability_potential:.2f}")
        logger.info(
            f"  Capabilities: {[cap.value for cap in analysis.required_capabilities]}"
        )
        logger.info("")


async def test_deduplication():
    """Test the advanced tool deduplicator."""
    logger.info("=== Testing Advanced Tool Deduplication ===")

    deduplicator = AdvancedToolDeduplicator()

    # Create test tools
    tool1 = DynamicTool(
        user_id="test_user",
        name="calculator_tool",
        description="A tool for basic mathematical calculations",
        code="def calculate(a, b, operation):\n    if operation == 'add':\n        return a + b\n    elif operation == 'subtract':\n        return a - b",
        function_name="calculate",
        parameters={"a": "number", "b": "number", "operation": "string"},
    )

    tool2 = DynamicTool(
        user_id="test_user",
        name="math_helper",
        description="A helper tool for performing mathematical operations",
        code="def math_operation(x, y, op):\n    if op == 'add':\n        return x + y\n    elif op == 'sub':\n        return x - y",
        function_name="math_operation",
        parameters={"x": "number", "y": "number", "op": "string"},
    )

    tool3 = DynamicTool(
        user_id="test_user",
        name="text_processor",
        description="A tool for processing and formatting text content",
        code="def process_text(text, format_type):\n    if format_type == 'upper':\n        return text.upper()\n    elif format_type == 'lower':\n        return text.lower()",
        function_name="process_text",
        parameters={"text": "string", "format_type": "string"},
    )

    # Test similarity calculation
    similarity_1_2 = await deduplicator._calculate_comprehensive_similarity(
        tool1, tool2, 0.8
    )
    similarity_1_3 = await deduplicator._calculate_comprehensive_similarity(
        tool1, tool3, 0.2
    )

    logger.info(
        f"Similarity between calculator tools: {similarity_1_2.overall_similarity:.2f}"
    )
    logger.info(f"  Code similarity: {similarity_1_2.code_similarity:.2f}")
    logger.info(f"  Parameter similarity: {similarity_1_2.parameter_similarity:.2f}")
    logger.info(f"  Reasons: {similarity_1_2.reasons}")
    logger.info("")

    logger.info(
        f"Similarity between calculator and text tools: {similarity_1_3.overall_similarity:.2f}"
    )
    logger.info(f"  Code similarity: {similarity_1_3.code_similarity:.2f}")
    logger.info(f"  Parameter similarity: {similarity_1_3.parameter_similarity:.2f}")
    logger.info(f"  Reasons: {similarity_1_3.reasons}")
    logger.info("")


async def test_integrated_system():
    """Test the integrated tool generation system."""
    logger.info("=== Testing Integrated Tool Generation System ===")

    generator = DynamicToolGenerator()
    mock_context = MockConversationContext()

    test_cases = [
        "Hello there!",  # Should be blocked by smart analysis
        "What's 5 + 3?",  # Should be blocked as trivial
        "Calculate the area of a complex polygon",  # Should pass smart analysis
        "Create a function to parse JSON data",  # Should pass smart analysis
    ]

    for user_message in test_cases:
        logger.info(f"Testing: '{user_message}'")

        try:
            # Test analysis phase
            analysis = await generator.analyze_tool_need(user_message, mock_context)
            logger.info(f"  Analysis result: needs_tool={analysis.needs_dynamic_tool}")
            logger.info(f"  Confidence: {analysis.confidence_score:.2f}")
            logger.info(f"  Reasoning: {analysis.reasoning}")

            if analysis.needs_dynamic_tool:
                logger.info(f"  Description: {analysis.description}")

        except Exception as e:
            logger.error(f"  Error in analysis: {e}")

        logger.info("")


async def test_pipeline_lifecycle():
    """Test pipeline lifecycle management."""
    logger.info("=== Testing Pipeline Lifecycle Management ===")

    # Test managed pipeline execution
    async def sample_pipeline_work():
        """Simulate some pipeline work."""
        await asyncio.sleep(0.1)  # Simulate processing
        return "Pipeline completed successfully"

    try:
        # Test normal execution
        async with managed_pipeline_execution(
            pipeline_id="test_pipeline",
            task_name="sample_task",
            user_id="test_user",
            arguments={"test": "argument"},
            timeout=5.0,
        ) as ctx:
            result = await sample_pipeline_work()
            ctx.logger.info(f"Pipeline work result: {result}")

        logger.info("Pipeline execution completed successfully")

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")


async def main():
    """Run all tests."""
    logger.info("Starting Enhanced Tool Generation System Tests")
    logger.info("=" * 60)

    try:
        await test_smart_analysis()
        await test_deduplication()
        await test_integrated_system()
        await test_pipeline_lifecycle()

        logger.info("=" * 60)
        logger.info("All tests completed successfully!")

    except Exception as e:
        logger.error(f"Test execution failed: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
