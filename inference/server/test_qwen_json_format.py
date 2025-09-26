#!/usr/bin/env python3
"""
Direct QwenMoE Tool Calling Test - Tests if QwenMoE follows JSON format instructions.
"""

import sys
import os
import asyncio
import logging
from typing import List

# Add the server path for imports
sys.path.insert(0, '/app/server')
sys.path.insert(0, '/app/runner')

from services.conversation_service import ConversationService
from services.model_service import ModelService
from models.conversation import Conversation
from models.message import Message
from models.message_content import MessageContent, MessageContentType
from models.message_role import MessageRole
from models.model_profile import ModelProfile


async def test_qwen_json_format_adherence():
    """Test if QwenMoE follows explicit JSON tool calling format instructions."""
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    print("🧪 Testing QwenMoE JSON Format Adherence")
    print("=" * 60)
    
    # Initialize services
    try:
        conversation_service = ConversationService()
        model_service = ModelService()
        
        # Get the QwenMoE model
        model = await model_service.get_model("qwen3-30b-a3b-q4-k-m")
        if not model:
            raise Exception("QwenMoE model not found")
            
        print(f"✅ Model found: {model.name}")
        
        # Create a test profile with explicit JSON instructions  
        profile = ModelProfile(
            id="test-json-format",
            user_id="test",
            name="QwenMoE JSON Format Test",
            description="Test profile for QwenMoE JSON format adherence",
            model_name="qwen3-30b-a3b-q4-k-m",
            parameters={
                "temperature": 0.1,  # Low temperature for consistent behavior
                "max_tokens": 200,   # Short response to focus on format
                "top_p": 0.9
            },
            system_prompt="""You are a helpful assistant with access to tools.

CRITICAL: When asked for current information, you MUST respond with the EXACT JSON format shown below:

```json
{
    "tool_calls": [
        {
            "name": "web_search",
            "arguments": {
                "query": "your search query here"
            }
        }
    ]
}
```

DO NOT write any other text. ONLY respond with the JSON format when asked for current information.

Tools available:
- web_search: Search for current information on the web
"""
        )
        
        print(f"✅ Profile created with explicit JSON instructions")
        
        # Create test message requesting current information
        test_message = Message(
            role=MessageRole.USER,
            content=[
                MessageContent(
                    type=MessageContentType.TEXT,
                    text="What are the latest quantum computing breakthroughs announced in 2024?"
                )
            ]
        )
        
        print(f"✅ Test message: {test_message.content[0].text}")
        
        # Test the response
        print("\n🚀 Executing QwenMoE pipeline...")
        
        response = await conversation_service.stream_chat_completion(
            messages=[test_message],
            model=model,
            profile=profile,
            user_id="test"
        )
        
        print("\n📝 QwenMoE Response Analysis:")
        print("-" * 40)
        
        response_text = ""
        if response and response.message and response.message.content:
            for content in response.message.content:
                if content.type == MessageContentType.TEXT:
                    response_text += content.text
                    
        print(f"Response text: {response_text}")
        
        # Check if response follows JSON format
        has_json_block = "```json" in response_text
        has_tool_calls = "tool_calls" in response_text
        has_web_search = "web_search" in response_text
        
        print(f"\n🔍 Format Analysis:")
        print(f"   Contains ```json block: {has_json_block}")
        print(f"   Contains 'tool_calls': {has_tool_calls}")
        print(f"   Contains 'web_search': {has_web_search}")
        
        # Check thinking content
        thinking = response.message.thinking if response.message else None
        if thinking:
            print(f"   Thinking content: {len(thinking)} chars")
            print(f"   Thinking preview: {thinking[:200]}...")
        
        # Final assessment
        follows_format = has_json_block and has_tool_calls and has_web_search
        
        print(f"\n🎯 RESULT:")
        print(f"   QwenMoE follows JSON format: {follows_format}")
        
        if not follows_format:
            print(f"\n❌ ISSUE IDENTIFIED:")
            print(f"   QwenMoE received explicit JSON format instructions")
            print(f"   QwenMoE was asked for current 2024 information") 
            print(f"   QwenMoE should have used web_search tool in JSON format")
            print(f"   QwenMoE generated: {response_text[:200]}...")
            print(f"\n🔍 This suggests QwenMoE may not be properly trained")
            print(f"   for this specific JSON tool calling format, despite")
            print(f"   having the _parse_qwen_tool_calls method in the pipeline.")
            
        return follows_format
        
    except Exception as e:
        logger.error(f"Error in QwenMoE JSON format test: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    result = asyncio.run(test_qwen_json_format_adherence())
    sys.exit(0 if result else 1)