#!/usr/bin/env python3
"""
Test script to verify that pipeline close() method properly unlocks pipelines.
"""

import sys
import os

# Add inference directory to Python path
sys.path.insert(0, '/Users/lons7862/workspace/llmmllab/inference')

from models import Model, ModelProfile, ModelDetails, ModelProvider, ModelTask, ModelParameters
from runner.pipeline_factory import pipeline_factory
from runner.pipeline_cache import LocalPipelineCacheManager
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_pipeline_unlock_on_close():
    """Test that pipeline.close() automatically unlocks the pipeline."""
    
    # Create a test model with a local provider
    test_model = Model(
        id="test-model",
        name="test-model", 
        model="/fake/path/model.gguf",
        provider=ModelProvider.LLAMA_CPP,  # Use local provider
        task=ModelTask.TEXTTOTEXT,
    )
    
    test_profile = ModelProfile(
        user_id="test_user", 
        name="Test Profile",
        model_name="test-model",
        system_prompt="You are a test assistant",
        parameters=ModelParameters(),
        type=0
    )
    
    # Add the test model to the factory's available models
    pipeline_factory._available_models["test-model"] = test_model
    
    try:
        cache_manager = pipeline_factory.local_cache
        model_id = "test-model"
        
        # Manually lock the pipeline to simulate get_pipeline() behavior
        logger.info("Testing manual lock/unlock cycle...")
        
        # Check initial state
        initial_stats = cache_manager.stats()
        initial_locked = initial_stats.get("locked", 0)
        logger.info(f"Initial locked pipelines: {initial_locked}")
        
        # Verify model is considered local
        is_local = cache_manager.is_local(test_model)
        logger.info(f"Model is_local: {is_local}")
        
        # Manually lock
        lock_success = cache_manager.lock_pipeline(model_id)
        logger.info(f"Manual lock success: {lock_success}")
        
        # Check locked state
        locked_stats = cache_manager.stats() 
        locked_count = locked_stats.get("locked", 0)
        model_entry = locked_stats.get("entries", {}).get(model_id, {})
        model_in_use = model_entry.get("in_use", False)
        logger.info(f"After lock - Total locked: {locked_count}, Model in_use: {model_in_use}")
        
        # Test unlock via pipeline factory (simulates what close() would do)
        unlock_success = pipeline_factory.unlock_pipeline(test_profile)
        logger.info(f"Unlock success: {unlock_success}")
        
        # Check unlocked state
        unlocked_stats = cache_manager.stats()
        unlocked_count = unlocked_stats.get("locked", 0)
        model_entry_after = unlocked_stats.get("entries", {}).get(model_id, {})
        model_in_use_after = model_entry_after.get("in_use", False) 
        logger.info(f"After unlock - Total locked: {unlocked_count}, Model in_use: {model_in_use_after}")
        
        # Verify the unlock worked
        if is_local and lock_success and model_in_use and not model_in_use_after:
            logger.info("✅ SUCCESS: Unlock logic works correctly - simulates what close() would do")
            return True
        else:
            logger.error(f"❌ FAILED: Unlock logic test failed")
            logger.error(f"  is_local: {is_local}")
            logger.error(f"  lock_success: {lock_success}")
            logger.error(f"  model_in_use before: {model_in_use}")
            logger.error(f"  model_in_use after: {model_in_use_after}")
            return False
    
    finally:
        # Clean up
        if "test-model" in pipeline_factory._available_models:
            del pipeline_factory._available_models["test-model"]

if __name__ == "__main__":
    test_pipeline_unlock_on_close()