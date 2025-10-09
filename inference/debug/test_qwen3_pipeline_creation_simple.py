#!/usr/bin/env python3
"""
Simple test to verify Qwen3 pipeline creation works after fixing import issues.
This version doesn't require database access.
"""

import asyncio
import logging
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_qwen3_pipeline_creation():
    """Test that we can create a Qwen3 pipeline successfully."""
    
    try:
        # Import required modules
        from runner.pipeline_factory import pipeline_factory
        from models import Model, ModelProfile, ModelProfileType, ModelTask, ModelParameters
        from models.default_configs import DEFAULT_MODEL_PROFILE_CONFIG
        import uuid
        
        logger.info("🧪 Testing Qwen3 pipeline creation...")
        
        # Find the Qwen3 Coder model in the factory's models
        qwen3_model = None
        target_model_id = "qwen3-coder-30b-a3b"
        
        if target_model_id in pipeline_factory.models:
            qwen3_model = pipeline_factory.models[target_model_id]
            logger.info(f"Found Qwen3 model: {target_model_id}")
        else:
            logger.error(f"❌ Qwen3 Coder model '{target_model_id}' not found in pipeline factory")
            logger.info(f"Available models: {list(pipeline_factory.models.keys())}")
            return False
        
        # Create a basic model profile for testing
        parameters = ModelParameters(
            temperature=0.7,
            max_tokens=4096,
            num_ctx=4096
        )
        
        model_profile = ModelProfile(
            id=uuid.uuid4(),
            name="Test Qwen3 Profile",
            user_id="test_user_pipeline_creation", 
            profile_type=ModelProfileType.Engineering,
            model_name=qwen3_model.id,
            parameters=parameters,
            system_prompt="You are a helpful AI assistant.",
            type=1  # Primary profile type
        )
        
        logger.info(f"Model profile: {model_profile.model_name}")
        logger.info(f"Profile name: {model_profile.name}")
        
        # Try to create the pipeline using the factory
        logger.info("Creating pipeline...")
        
        with pipeline_factory.pipeline(model_profile, str) as pipe:
            logger.info(f"✅ Successfully created pipeline: {type(pipe).__name__}")
            logger.info(f"Pipeline model: {pipe.model.name}")
            logger.info(f"Pipeline class: {pipe.__class__.__name__}")
            
            # Verify it's the correct class
            if "Qwen3Moe" in str(type(pipe)):
                logger.info("✅ Correctly using Qwen3Moe class")
            else:
                logger.warning(f"⚠️  Expected Qwen3Moe, got {type(pipe).__name__}")
            
        logger.info("🎉 Qwen3 pipeline creation test completed successfully!")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Pipeline creation failed: {e}")
        logger.exception("Full traceback:")
        return False


if __name__ == "__main__":
    try:
        result = test_qwen3_pipeline_creation()
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)