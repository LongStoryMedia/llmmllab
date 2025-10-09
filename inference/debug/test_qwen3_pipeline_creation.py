#!/usr/bin/env python3
"""
Simple test to verify Qwen3 pipeline creation works after fixing import issues.
"""

import asyncio
import logging
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def test_qwen3_pipeline_creation():
    """Test that we can create a Qwen3 pipeline successfully."""
    
    try:
        # Import required modules
        from runner.pipeline_factory import pipeline_factory
        from utils.model_profile import get_model_profile
        from models import ModelProfileType
        
        logger.info("🧪 Testing Qwen3 pipeline creation...")
        
        # Try to get a model profile for Qwen3 Coder 30B
        # Use a test user ID for this test
        test_user_id = "test_user_pipeline_creation"
        
        logger.info(f"Getting model profile for user: {test_user_id}")
        
        # Get an engineering profile (typically used for code generation)
        model_profile = await get_model_profile(test_user_id, ModelProfileType.Engineering)
        
        # Override the model to use Qwen3 Coder specifically
        model_profile.model_name = "qwen3-coder-30b-a3b"
        
        logger.info(f"Model profile: {model_profile.model_name}")
        logger.info(f"Profile type: {model_profile.profile_type}")
        
        # Try to create the pipeline using the factory
        logger.info("Creating pipeline...")
        
        with pipeline_factory.pipeline(model_profile, str) as pipe:
            logger.info(f"✅ Successfully created pipeline: {type(pipe).__name__}")
            logger.info(f"Pipeline model: {pipe.model.name}")
            logger.info(f"Pipeline class: {pipe.__class__.__name__}")
            
            # Test basic functionality by getting the LLM instance
            if hasattr(pipe, 'llm') and pipe.llm is None:
                logger.info("Initializing LLM...")
                pipe._initialize_llm()
                logger.info("✅ LLM initialized successfully")
            
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
        result = asyncio.run(test_qwen3_pipeline_creation())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)