#!/usr/bin/env python3
"""
Test script for LlamaCppServerPipeline.

This validates that the new server-based pipeline works correctly
for replacing llama-cpp-python dependency.
"""

import os
import sys
import asyncio
from pathlib import Path

# Add inference directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models import Model, ModelProfile, ModelParameters, ModelDetails
from runner.pipelines.llamacpp.llamacpp_server_pipeline import LlamaCppServerPipeline
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="test_server_pipeline")


def create_test_model() -> Model:
    """Create a test model configuration."""
    return Model(
        id="test-model",
        name="Test Model", 
        model="/models/qwen3-4b/qwen3-4b-ud-q6-k-xl.gguf",  # Assuming this exists
        task="TextToText",
        modified_at="2025-11-08",
        digest="test-digest",
        details=ModelDetails(
            gguf_file="/models/qwen3-4b/qwen3-4b-ud-q6-k-xl.gguf",
            format="gguf",
            size=3658223392,
            family="qwen",
            families=["Qwen"],
            parameter_size="4B",
            original_ctx=40960,  # Required field
        ),
        provider="llama_cpp"
    )


def create_test_profile() -> ModelProfile:
    """Create a test model profile."""
    return ModelProfile(
        id=None,
        user_id="test-user",
        name="Test Profile",
        model_name="test-model",
        parameters=ModelParameters(
            temperature=0.7,
            max_tokens=100,
            top_p=0.9,
            top_k=40,
            num_ctx=2048,
            batch_size=256,
        ),
        system_prompt="You are a helpful AI assistant.",
        type=1
    )


def test_pipeline_initialization():
    """Test that the pipeline initializes correctly."""
    logger.info("Testing pipeline initialization...")
    
    model = create_test_model()
    profile = create_test_profile()
    
    try:
        # Initialize pipeline
        pipeline = LlamaCppServerPipeline(model, profile)
        logger.info("✅ Pipeline initialized successfully")
        
        # Check that server is running
        if pipeline.server_manager.is_running():
            logger.info("✅ Server is running and responsive")
        else:
            logger.error("❌ Server is not running")
            return False
        
        # Clean up
        pipeline.close()
        logger.info("✅ Pipeline closed successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline initialization failed: {e}")
        return False


def test_simple_generation():
    """Test basic text generation."""
    logger.info("Testing simple text generation...")
    
    model = create_test_model()
    profile = create_test_profile()
    
    try:
        # Initialize pipeline
        pipeline = LlamaCppServerPipeline(model, profile)
        
        # Test generation
        from langchain_core.messages import HumanMessage
        
        messages = [HumanMessage(content="What is 2+2?")]
        result = pipeline._generate(messages)
        
        if result and result.generations:
            response = result.generations[0].message.content
            logger.info(f"✅ Generation successful: {response[:100]}...")
        else:
            logger.error("❌ No response generated")
            pipeline.close()
            return False
        
        # Clean up
        pipeline.close()
        logger.info("✅ Generation test completed successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Generation test failed: {e}")
        return False


def test_streaming():
    """Test streaming generation."""
    logger.info("Testing streaming generation...")
    
    model = create_test_model()
    profile = create_test_profile()
    
    try:
        # Initialize pipeline
        pipeline = LlamaCppServerPipeline(model, profile)
        
        # Test streaming
        from langchain_core.messages import HumanMessage
        
        messages = [HumanMessage(content="Count from 1 to 5:")]
        
        chunks = []
        for chunk in pipeline._stream(messages):
            if chunk and chunk.message and chunk.message.content:
                chunks.append(chunk.message.content)
        
        if chunks:
            full_response = "".join(chunks)
            logger.info(f"✅ Streaming successful: {full_response[:100]}...")
        else:
            logger.error("❌ No streaming chunks received")
            pipeline.close()
            return False
        
        # Clean up
        pipeline.close()
        logger.info("✅ Streaming test completed successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Streaming test failed: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("🚀 Starting LlamaCppServerPipeline tests...")
    
    tests = [
        test_pipeline_initialization,
        test_simple_generation,
        test_streaming,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            logger.error(f"Test {test.__name__} crashed: {e}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    logger.info(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Server pipeline is working correctly.")
        return 0
    else:
        logger.error("❌ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    exit(main())