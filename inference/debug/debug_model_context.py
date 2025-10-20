#!/usr/bin/env python3
"""
Debug model context window configuration and actual usage.
"""

import logging
import asyncio
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from runner.pipelines.llamacpp.qwen3_moe import Qwen3Moe
from models.model_profile import ModelProfile, ModelParameters
from models.model import Model, ModelTask, ModelProvider, ModelDetails
from uuid import uuid4

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

async def debug_model_context():
    """Debug model context window configuration."""
    logger.info("🔍 Starting model context window debugging...")
    
    try:
        # Create test model configuration
        model = Model(
            id='qwen3-30b-a3b-q4-k-m',
            name='qwen3-30b-a3b-q4-k-m',
            model='/models/qwen3-30b-a3b/qwen3-30b-a3b-iq4-xs-abliterated.gguf',
            task=ModelTask.TEXTTOTEXT,
            modified_at='2025-07-20',
            size=16557092832,
            digest='qwen3-30b-a3b-20250720',
            details=ModelDetails(
                parent_model='Qwen/Qwen3-30B-A3B',
                format='gguf',
                gguf_file='/models/qwen3-30b-a3b/qwen3-30b-a3b-iq4-xs-abliterated.gguf',
                family='qwen',
                families=['Qwen', 'MoE'],
                parameter_size='30.5B',
                quantization_level='Q4_K_M',
                dtype='BF16',
                precision='fp16',
                specialization='TextToText',
                description=None,
                weight=1.0
            ),
            pipeline='Qwen3Pipe',
            lora_weights=[],
            provider=ModelProvider.LLAMA_CPP
        )
        
        # Create test profile with 100K context
        profile = ModelProfile(
            id=uuid4(),
            user_id='debug_user',
            name='Debug Profile',
            description='Debug profile for context testing',
            model_name='qwen3-30b-a3b-q4-k-m',
            parameters=ModelParameters(
                num_ctx=100000,  # 100K context
                temperature=0.7,
                max_tokens=4000
            ),
            system_prompt='You are a helpful AI assistant.',
            model_version='1.0',
            type=0
        )
        
        logger.info(f"📋 Test Profile Context: {profile.parameters.num_ctx}")
        
        # Try to initialize model
        logger.info("🔧 Initializing Qwen3Moe pipeline...")
        pipeline = Qwen3Moe(model=model, profile=profile)
        
        # Check if llama_instance exists and its context
        if hasattr(pipeline, 'llama_instance') and pipeline.llama_instance:
            actual_ctx = pipeline.llama_instance.n_ctx()
            logger.info(f"✅ Llama instance found with context: {actual_ctx}")
            logger.info(f"📊 Configured vs Actual: {profile.parameters.num_ctx} vs {actual_ctx}")
            
            if actual_ctx != profile.parameters.num_ctx:
                logger.warning(f"⚠️  Context mismatch! Requested {profile.parameters.num_ctx}, got {actual_ctx}")
                
                # Check if it's a known fallback value
                if actual_ctx == 16384:
                    logger.error("🚨 Model fell back to 16K context - this is the source of our problem!")
                elif actual_ctx == 4096:
                    logger.error("🚨 Model fell back to 4K context - memory insufficient!")
                else:
                    logger.warning(f"🔍 Unknown context fallback to {actual_ctx}")
            else:
                logger.info("✅ Context configuration matches perfectly!")
        else:
            logger.warning("❌ No llama_instance found - model not initialized")
            
        # Try to get some model information
        logger.info("🔍 Checking model initialization details...")
        if hasattr(pipeline, '_identifying_params'):
            params = pipeline._identifying_params()
            logger.info(f"📋 Identifying params: {params}")
            
    except Exception as e:
        logger.error(f"❌ Error during model context debugging: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_model_context())