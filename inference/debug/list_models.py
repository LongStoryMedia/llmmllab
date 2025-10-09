#!/usr/bin/env python3
"""
Debug script to list available models in pipeline factory.
"""

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def list_available_models():
    """List all models available in the pipeline factory."""
    
    try:
        from runner.pipeline_factory import pipeline_factory
        
        logger.info("📋 Available models in pipeline factory:")
        
        for model_id, model in pipeline_factory.models.items():
            logger.info(f"  - {model_id}: {model.name} (pipeline: {model.pipeline})")
        
        # Look specifically for qwen models
        qwen_models = {k: v for k, v in pipeline_factory.models.items() if "qwen" in k.lower()}
        if qwen_models:
            logger.info("🔍 Qwen models found:")
            for model_id, model in qwen_models.items():
                logger.info(f"  - {model_id}: {model.name}")
        else:
            logger.warning("⚠️  No Qwen models found!")
            
        return True
        
    except Exception as e:
        logger.error(f"❌ Error listing models: {e}")
        return False


if __name__ == "__main__":
    list_available_models()