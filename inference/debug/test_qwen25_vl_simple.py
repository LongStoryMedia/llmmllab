#!/usr/bin/env python3
"""
Simple test to verify Qwen2.5VL pipeline without complex imports
"""

import sys
import os
import asyncio
import logging

# Add the necessary paths
sys.path.append('/app')
sys.path.append('/app/server')
sys.path.append('/app/runner')
sys.path.append('/app/utils')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_qwen25_vl_basic():
    """Basic test of Qwen2.5VL pipeline"""
    
    try:
        # Import what we need
        from runner.pipelines.qwen25_vl import Qwen25VLPipeline
        from models.generate_req import GenerateReq  
        from models.message import Message
        from models.message_source import MessageSource
        from models.pipeline_config import PipelineConfig
        
        logger.info("✅ All imports successful")
        
        # Create the pipeline
        pipeline = Qwen25VLPipeline()
        logger.info("✅ Pipeline created successfully")
        
        # Create a simple test request
        request = GenerateReq(
            model="qwen2.5-vl-32b-instruct-q4-k-m",
            messages=[
                Message(
                    role="user", 
                    content="Hello! Can you tell me about machine learning vs deep learning?",
                    source=MessageSource.USER
                )
            ],
            stream=False,
            config=PipelineConfig(
                enable_web_search=False,  # Disable web search for simplicity
                enable_tool_calling=False,  # Disable tool calling for simplicity
                max_tokens=500,
                temperature=0.1
            )
        )
        
        logger.info("✅ Request created successfully")
        
        # Try to generate a response
        logger.info("🚀 Starting generation...")
        response = await pipeline.generate(request)
        
        if response and hasattr(response, 'choices') and response.choices:
            content = response.choices[0].message.content
            logger.info(f"✅ Generation successful! Response length: {len(content)} characters")
            logger.info(f"Response preview: {content[:200]}...")
            return True
        else:
            logger.error("❌ No response generated")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_qwen25_vl_basic())
    if success:
        logger.info("🎉 QWEN2.5VL BASIC TEST: PASSED")
    else:
        logger.info("💥 QWEN2.5VL BASIC TEST: FAILED")