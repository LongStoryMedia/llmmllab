#!/usr/bin/env python3
"""
Simple test of vision optimization logic without full framework initialization.
"""

import json
import hashlib
import structlog
from langchain_core.messages import HumanMessage

logger = structlog.get_logger(__name__)

def test_vision_optimization_logic():
    """Test the core vision optimization logic."""
    
    logger.info("🚀 Testing vision optimization logic directly")
    
    # Create a simple vision cache (mimicking the subgraph's cache)
    vision_cache = {}
    
    # Test message with image
    test_message = HumanMessage(content=[
        {
            "type": "image_url", 
            "image_url": {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"}
        },
        {
            "type": "text",
            "text": "What do you see in this image? Please describe it briefly."
        }
    ])
    
    def optimize_vision_content_simple(messages):
        """Simplified version of the optimization logic."""
        optimized_messages = []
        for msg in messages:
            has_vision = False
            content_hash = None
            content = getattr(msg, 'content', '')
            
            if isinstance(content, list):
                # Check for LangChain content blocks with images
                for block in content:
                    if isinstance(block, dict):
                        if block.get('type') == 'image_url' or block.get('type') == 'image':
                            has_vision = True
                            # Create hash from image URL or data
                            image_content = json.dumps(block, sort_keys=True)
                            content_hash = hashlib.md5(image_content.encode()).hexdigest()[:8]
                            break
            
            if has_vision and content_hash:
                logger.info(f"🖼️ Found vision content with hash: {content_hash}")
                
                # Check if we've processed this before
                if content_hash in vision_cache:
                    # Replace with cached summary
                    cached_summary = vision_cache[content_hash]
                    
                    # Replace image content blocks with text summary
                    new_content = []
                    for block in content:
                        if isinstance(block, dict) and (block.get('type') == 'image_url' or block.get('type') == 'image'):
                            new_content.append({
                                'type': 'text', 
                                'text': f"[Previous image analysis: {cached_summary}]"
                            })
                        else:
                            new_content.append(block)
                    new_msg = HumanMessage(content=new_content)
                    
                    optimized_messages.append(new_msg)
                    logger.info(f"🖼️ Using cached vision analysis (hash: {content_hash})")
                else:
                    # First time seeing this image - store hash for later caching
                    setattr(msg, '_vision_hash', content_hash)
                    optimized_messages.append(msg)
                    logger.info(f"🖼️ New vision content detected (hash: {content_hash})")
            else:
                optimized_messages.append(msg)
                
        return optimized_messages
    
    logger.info("� First processing - should detect new vision content")
    
    # First processing
    messages1 = optimize_vision_content_simple([test_message])
    
    # Simulate adding a cache entry
    if hasattr(messages1[0], '_vision_hash'):
        vision_hash = messages1[0]._vision_hash
        vision_cache[vision_hash] = "The image shows a woman in a checkered shirt with a light-colored dog on a beach at sunset."
        logger.info(f"📝 Added cache entry for hash: {vision_hash}")
    
    logger.info("� Second processing - should use cached analysis")
    
    # Second processing (same image) - should use cache
    messages2 = optimize_vision_content_simple([test_message])
    
    # Check if optimization worked
    if messages2:
        msg2_content = messages2[0].content
        if isinstance(msg2_content, list):
            for block in msg2_content:
                if isinstance(block, dict) and block.get('type') == 'text':
                    if 'Previous image analysis' in block.get('text', ''):
                        logger.info("✅ Vision optimization working! Found cached analysis in content.")
                        logger.info(f"📄 Optimized content: {block['text']}")
                        return True
        
        logger.info(f"📄 Second processing result content: {msg2_content}")
    
    logger.warning("❌ Vision optimization may not be working as expected")
    return False

if __name__ == "__main__":
    test_vision_optimization_logic()