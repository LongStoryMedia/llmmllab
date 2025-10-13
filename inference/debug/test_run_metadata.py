#!/usr/bin/env python3
"""
Test the simplified run.py implementation with metadata tracking.
"""

import asyncio
import logging
from typing import List

from models import Message, MessageContent, MessageContentType, MessageRole
from runner.pipelines.run import run_pipeline, PipelineExecutionMetadata
from runner.pipeline_factory import pipeline_factory

# Set up logging to see metadata
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

async def test_metadata_tracking():
    """Test the new metadata tracking functionality."""
    
    try:
        # Create a simple test message
        test_message = Message(
            role=MessageRole.USER,
            content=[MessageContent(type=MessageContentType.TEXT, text="Hello, world!")]
        )
        
        # Get a pipeline from the factory
        from models import ModelProfile, ModelParameters
        import uuid
        from datetime import datetime, timezone
        
        profile = ModelProfile(
            id=str(uuid.uuid4()),
            user_id="test-user",
            name="test-profile", 
            description="Test profile for metadata testing",
            model_name="qwen3-4b-ud-q6-k-xl",
            parameters=ModelParameters(
                temperature=0.7,
                max_tokens=100,
                top_p=0.9,
                frequency_penalty=0.0,
                presence_penalty=0.0
            ),
            system_prompt="You are a helpful assistant.",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            type=1
        )
        
        pipeline = pipeline_factory.get_pipeline(profile, expected_type=str)
        
        print(f"Pipeline type: {type(pipeline).__name__}")
        print(f"Model: {getattr(pipeline.model, 'name', 'unknown')}")
        print(f"Provider: {getattr(pipeline.model, 'provider', 'unknown')}")
        
        # Test metadata creation
        metadata = PipelineExecutionMetadata(pipeline)
        print(f"\nMetadata:")
        print(f"  Execution ID: {metadata.execution_id}")
        print(f"  Pipeline: {metadata.pipeline_name}")
        print(f"  Model: {metadata.model_name}")
        print(f"  Provider: {metadata.provider}")
        print(f"  Cached: {metadata.is_cached}")
        print(f"  Return type: {metadata.expected_return_type}")
        
        # Test actual pipeline execution
        print(f"\nTesting pipeline execution...")
        result = await run_pipeline([test_message], pipeline)
        
        print(f"Result type: {type(result)}")
        if hasattr(result, 'message') and result.message:
            from utils.message import extract_message_text
            text = extract_message_text(result.message)
            print(f"Response text: {text[:100]}...")
        
        print("✅ Metadata tracking test completed successfully")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Clean up
        try:
            if 'pipeline' in locals():
                pipeline.cleanup()
        except:
            pass

if __name__ == "__main__":
    asyncio.run(test_metadata_tracking())