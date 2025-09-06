#!/usr/bin/env python3
"""
Test the pipeline factory return type through the search service path.
"""

import os
import sys
import asyncio
import logging

# Add the parent directory to path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from runner.pipeline_factory import PipelineFactory
from models import (
    ModelProfile,
    Model,
    Message,
    MessageContent,
    MessageRole,
    MessageContentType,
)
from models.model_parameters import ModelParameters

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_search_pipeline_return_type():
    """Test that pipeline factory correctly passes return types for search service."""

    # Create a mock model and profile
    model_params = ModelParameters(
        model_provider="ollama",
        model="qwen2.5:7b-instruct-q4_K_M",
        temperature=0.7,
        top_p=0.9,
        max_tokens=1000,
    )

    profile = ModelProfile(name="test_profile", parameters=model_params)

    # Get global factory instance (similar to how search service uses it)
    import runner.pipeline_factory as pf

    print("Testing pipeline creation through search service path...")

    # Test the exact pattern used in search service: with pipeline_factory.pipeline(mp, str) as pipe:
    try:
        with pf.pipeline(profile, str) as pipe:
            print(f"✓ Pipeline created successfully: {type(pipe)}")
            print(
                f"✓ Expected return type: {getattr(pipe, 'expected_return_type', 'Not set')}"
            )

            # Test that the pipeline is configured for string return type
            if hasattr(pipe, "expected_return_type"):
                if pipe.expected_return_type == str:
                    print("✓ Pipeline correctly configured for string return type")

                    # Create a test message
                    test_content = MessageContent(
                        type=MessageContentType.TEXT,
                        text="test query for search formatting",
                    )
                    test_message = Message(
                        role=MessageRole.USER, content=[test_content]
                    )

                    print("Testing message processing...")
                    result = await pipe.process_messages([test_message])
                    print(f"✓ Processing result type: {type(result)}")

                    if isinstance(result, str):
                        print("✓ Pipeline correctly returned string type")
                        print(f"Result preview: {result[:100]}...")
                    else:
                        print(f"✗ Pipeline returned {type(result)}, not str")
                        print(f"Result: {result}")

                        # Test if it has .strip() method (the error from logs)
                        if hasattr(result, "strip"):
                            print("✓ Result has strip method")
                        else:
                            print("✗ Result does not have strip method")

                else:
                    print(
                        f"✗ Pipeline expected_return_type is {pipe.expected_return_type}, not str"
                    )
            else:
                print("⚠ Pipeline doesn't have expected_return_type attribute")

    except Exception as e:
        print(f"✗ Error in pipeline test: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_search_pipeline_return_type())
