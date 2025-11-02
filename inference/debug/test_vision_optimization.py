"""
Test Vision Middleware with multiple identical images to see optimization.
"""

import asyncio
from langchain_core.messages import HumanMessage, AIMessage
from composer.middleware import VisionSummarizationMiddleware
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="test_vision_optimization")

async def test_vision_optimization():
    """Test the vision middleware with multiple identical images."""
    logger.info("🧪 Testing Vision Optimization Flow")
    
    # Create middleware instance
    middleware = VisionSummarizationMiddleware(
        max_image_reprocessing=1,
        enable_logging=True
    )
    
    # Simulate first conversation turn
    logger.info("🔄 Turn 1: First image processing")
    user_message_1 = HumanMessage(content="Picture 1: <|vision_start|> <|vision_end|>What do you see in this image?")
    ai_response_1 = AIMessage(content="The image shows a woman in a checkered shirt sitting on a sandy beach, gently holding the paw of a light-colored dog during golden hour.")
    
    # Update cache with first turn
    conversation_1 = [user_message_1, ai_response_1]
    middleware._update_processed_images_cache(conversation_1)
    
    cache_stats = middleware.get_cache_stats()
    logger.info(f"📊 After Turn 1 - Cache stats: {cache_stats}")
    
    # Simulate second conversation turn with SAME image
    logger.info("🔄 Turn 2: Same image appears again")
    user_message_2 = HumanMessage(content="Picture 1: <|vision_start|> <|vision_end|>Describe this image in more detail.")
    
    # Test optimization - should replace image with summary
    conversation_2 = conversation_1 + [user_message_2]
    optimized_messages = middleware._replace_processed_images_in_messages(conversation_2)
    
    logger.info(f"📋 Original conversation: {len(conversation_2)} messages")
    logger.info(f"📋 Optimized conversation: {len(optimized_messages)} messages")
    
    # Show the optimization
    for i, (orig, opt) in enumerate(zip(conversation_2, optimized_messages)):
        if orig.content != opt.content:
            logger.info(f"✨ Message {i} optimized:")
            logger.info(f"   Original: {orig.content[:100]}...")
            logger.info(f"   Optimized: {opt.content[:100]}...")
    
    cache_stats_final = middleware.get_cache_stats()
    logger.info(f"📊 Final cache stats: {cache_stats_final}")
    
    logger.info("🎉 Vision optimization test completed")

if __name__ == "__main__":
    asyncio.run(test_vision_optimization())