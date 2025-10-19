#!/usr/bin/env python3
"""
Test script to verify pipeline creation works correctly with grammar parameters.
"""

import asyncio
import sys
import os

# Add the inference directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from runner.pipeline_factory import PipelineFactory
from models.model_profile import ModelProfile
from models.pipeline_priority import PipelinePriority

async def test_pipeline_creation():
    """Test that pipeline creation works with proper parameter passing."""
    print("🧪 Testing pipeline creation with grammar parameter fix...")
    
    try:
        # Create pipeline factory
        factory = PipelineFactory({})
        
        # Test with a simple profile that should trigger the Qwen3-4B model
        test_profile = ModelProfile(
            model_name="Qwen3-4B",
            # Add minimal required fields
            n_ctx=2048,
            temperature=0.7,
            max_tokens=100
        )
        
        print(f"   📝 Testing pipeline creation for {test_profile.model_name}...")
        
        # Try to get pipeline (this should trigger the fixed parameter passing)
        with factory.pipeline(test_profile, PipelinePriority.NORMAL) as pipeline:
            print(f"✅ Successfully created pipeline: {type(pipeline).__name__}")
            print(f"   Model: {getattr(pipeline, 'model', {}).get('name', 'Unknown')}")
            
        print("\n🎉 Pipeline creation test completed successfully!")
        
    except Exception as e:
        print(f"❌ Pipeline creation test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_pipeline_creation())