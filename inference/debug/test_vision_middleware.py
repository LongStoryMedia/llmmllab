"""
Test Vision Middleware directly to validate functionality.
"""

import os
import sys
from langchain_core.messages import HumanMessage, AIMessage
from composer.middleware import VisionSummarizationMiddleware
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="test_vision_middleware")

def test_vision_middleware():
    """Test the vision middleware directly."""
    logger.info("🧪 Testing Vision Middleware")
    
    # Create middleware instance
    middleware = VisionSummarizationMiddleware(
        max_image_reprocessing=1,
        enable_logging=True
    )
    
    # Test message with vision content
    test_message = HumanMessage(content="Picture 1: <|vision_start|> <|vision_end|>What do you see in this image?")
    
    # Test image extraction
    logger.info("🔍 Testing image content extraction")
    images = middleware._extract_image_content(test_message)
    logger.info(f"📋 Found {len(images)} images: {images}")
    
    # Test with AI response
    ai_response = AIMessage(content="The image shows a woman in a checkered shirt sitting on a sandy beach, gently holding the paw of a light-colored dog.")
    
    # Test analysis extraction
    logger.info("🔍 Testing analysis extraction")
    analysis = middleware._extract_image_analysis_from_response(ai_response)
    logger.info(f"📝 Extracted analysis: {analysis}")
    
    # Test cache update
    messages = [test_message, ai_response]
    logger.info("🔍 Testing cache update")
    middleware._update_processed_images_cache(messages)
    
    # Check cache
    cache_stats = middleware.get_cache_stats()
    logger.info(f"📊 Cache stats: {cache_stats}")
    
    # Test message replacement
    logger.info("🔍 Testing message replacement")
    optimized_messages = middleware._replace_processed_images_in_messages([test_message])
    logger.info(f"🔄 Original: {len([test_message])} messages, Optimized: {len(optimized_messages)} messages")
    
    if optimized_messages != [test_message]:
        logger.info(f"✅ Vision middleware successfully optimized messages!")
        for i, msg in enumerate(optimized_messages):
            logger.info(f"   Message {i}: {msg.content[:100]}...")
    else:
        logger.info("ℹ️ No optimization applied (expected for first processing)")
    
    logger.info("🎉 Vision middleware test completed")

if __name__ == "__main__":
    test_vision_middleware()