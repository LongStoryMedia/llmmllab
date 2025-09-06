#!/usr/bin/env python3
"""
Test the pipeline factory return type parameter passing.
"""

import os
import sys
import asyncio
import logging

# Add the parent directory to path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from runner.pipeline_factory import PipelineFactory
from models import ModelProfile, Model
from models.model_parameters import ModelParameters

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_pipeline_return_type():
    """Test that pipeline factory correctly passes return types."""

    # Create a mock model and profile
    model_params = ModelParameters(
        model_provider="ollama",
        model="qwen2.5:7b-instruct-q4_K_M",
        temperature=0.7,
        top_p=0.9,
        max_tokens=1000,
    )

    model = Model(
        name="qwen2.5:7b-instruct-q4_K_M",
        pipeline="Qwen30A3BQ4KMPipe",
        model="qwen2.5:7b-instruct-q4_K_M",
    )

    profile = ModelProfile(name="test_profile", parameters=model_params)

    # Test string return type request
    factory = PipelineFactory(prefer_langgraph=True)

    print("Testing pipeline creation with string return type...")
    pipeline = factory.get_pipeline(profile, str)

    if pipeline:
        print(f"✓ Pipeline created successfully: {type(pipeline)}")
        print(
            f"✓ Expected return type: {getattr(pipeline, 'expected_return_type', 'Not set')}"
        )

        # Test that the pipeline is configured for string return type
        if hasattr(pipeline, "expected_return_type"):
            if pipeline.expected_return_type == str:
                print("✓ Pipeline correctly configured for string return type")
            else:
                print(
                    f"✗ Pipeline expected_return_type is {pipeline.expected_return_type}, not str"
                )
        else:
            print("⚠ Pipeline doesn't have expected_return_type attribute")

    else:
        print("✗ Failed to create pipeline")


if __name__ == "__main__":
    asyncio.run(test_pipeline_return_type())
