#!/usr/bin/env python3
"""
Test Qwen3VL multimodal inference capabilities
"""
import json
import base64
from io import BytesIO
from PIL import Image
from runner.pipeline_factory import PipelineFactory
from models import ChatRequest, ConversationContext, Message, MessageRole
import structlog

# Configure logging
logger = structlog.get_logger("qwen3vl_multimodal_test")

def create_test_image():
    """Create a simple test image"""
    # Create a simple 200x200 red square with text
    img = Image.new('RGB', (200, 200), color='red')
    
    # Convert to base64
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    img_data = buffer.getvalue()
    img_base64 = base64.b64encode(img_data).decode('utf-8')
    
    return f"data:image/png;base64,{img_base64}"

def test_multimodal_inference():
    """Test multimodal inference with image and text"""
    logger.info("🚀 Testing Qwen3VL multimodal inference...")
    
    try:
        # Create pipeline factory
        factory = PipelineFactory()
        
        # Get the pipeline for qwen3-vl-32b
        model_id = "huihui-ai/Huihui-Qwen3-VL-32B-Thinking-abliterated"
        pipeline = factory.get_pipeline(model_id)
        
        logger.info("✅ Pipeline retrieved successfully")
        
        # Create test image
        test_image = create_test_image()
        logger.info("✅ Test image created")
        
        # Create multimodal chat request
        messages = [
            Message(
                role=MessageRole.USER,
                content="What color is this image? Please describe what you see.",
                images=[test_image]  # Image in base64 format
            )
        ]
        
        conversation_ctx = ConversationContext(
            conversation_id="test-multimodal",
            messages=messages
        )
        
        chat_request = ChatRequest(
            conversation_ctx=conversation_ctx,
            model=model_id,
            temperature=0.1,
            max_tokens=100,
            stream=False
        )
        
        logger.info("🔄 Sending multimodal request to pipeline...")
        
        # Test the pipeline
        response = pipeline.process_request(chat_request)
        
        logger.info("✅ Multimodal inference successful!")
        logger.info(f"📝 Response: {response.content[:200]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Multimodal inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_multimodal_inference()
    if success:
        print("🎉 Qwen3VL multimodal pipeline test completed successfully!")
    else:
        print("💥 Qwen3VL multimodal pipeline test failed!")
        exit(1)