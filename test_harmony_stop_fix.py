#!/usr/bin/env python3
"""Test OpenAI Harmony format stop token fix."""

import asyncio
import logging
import sys
import os

# Add paths for inference modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "inference"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

async def test_harmony_stop_tokens():
    """Test that GPT OSS model can complete harmony format channel transitions."""
    logger.info("🧪 Testing OpenAI Harmony format stop token fix")
    
    try:
        # Mock the model and profile objects that would normally come from database
        from models import Model, ModelProfile, ModelParameters
        
        # Create mock model and profile for testing
        model = Model(
            id="test-gpt-oss",
            name="Test GPT OSS 20B",
            details={"supports_tools": True},
            model="gpt-oss-20b",
            base_url="http://localhost:8000",
            provider="openai-gpt-oss"
        )
        
        # Create test parameters
        parameters = ModelParameters(
            temperature=0.3,
            top_p=0.8,
            top_k=20,
            max_tokens=1000,
            num_ctx=32768,
            stop=["<|im_end|>", "<|endoftext|>", "<|end|>"]  # Default includes <|end|>
        )
        
        profile = ModelProfile(
            id="test-profile",
            name="Test Profile",
            parameters=parameters,
            system_prompt="Test system prompt",
            provider="openai-gpt-oss"
        )
        
        logger.info("📋 Created mock model and profile")
        
        # Import pipeline after sys.path setup
        from runner.pipelines.txt2txt.openai_gpt_oss import OpenAiGptOssPipe
        
        # Create pipeline instance
        pipeline = OpenAiGptOssPipe(
            model=model,
            profile=profile,
            expected_return_type=None,
            circuit_config=None
        )
        
        logger.info("🏗️  Created OpenAI GPT OSS pipeline")
        
        # Check if the stop token fix is applied during initialization
        # This should modify the profile to remove <|end|> from stop tokens
        
        logger.info(f"📝 Original stop tokens: {profile.parameters.stop}")
        
        # Initialize the pipeline (this should apply the stop token fix)
        # Note: This will fail if GGUF file doesn't exist, but we can check the stop token fix
        try:
            gguf_path = "/tmp/nonexistent.gguf"  # This will fail, but we can check the parameter setting
            await pipeline._initialize_llm(gguf_path, None)
        except Exception as e:
            logger.info(f"💡 Expected initialization error (GGUF file missing): {e}")
        
        # Check if stop tokens were modified by the fix
        logger.info(f"✨ Modified stop tokens: {profile.parameters.stop}")
        
        # Verify the fix worked
        if "<|end|>" not in profile.parameters.stop:
            logger.info("✅ SUCCESS: <|end|> successfully removed from stop tokens!")
            logger.info("✅ Harmony format channel transitions should now work correctly")
            return True
        else:
            logger.error("❌ FAILED: <|end|> still present in stop tokens")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(test_harmony_stop_tokens())
    if result:
        logger.info("🎉 Stop token fix test PASSED")
        sys.exit(0)
    else:
        logger.error("💥 Stop token fix test FAILED")
        sys.exit(1)