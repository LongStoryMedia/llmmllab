#!/usr/bin/env python3
"""
Simple test to verify title generation fix through REST API.
Tests that title generation properly updates the conversation database.
"""

import asyncio
import sys
import json
import aiohttp
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, '/app')

from db import storage
from models.conversation import Conversation
from models.message import Message, MessageRole, MessageContent, MessageContentType
import utils.logging as logging

logger = logging.llmmllogger.logger.bind(component="TitleGenerationAPITest")


class TitleGenerationAPITester:
    """Test title generation through chat completion API."""
    
    def __init__(self):
        self.test_user_id = "title_api_test_user"
        self.test_conversation_id = None
        self.api_base_url = "http://localhost:8000"
        
    async def setup(self):
        """Initialize test environment."""
        logger.info("🚀 Setting up title generation API test environment")
        
        # Import and use the same connection string as the server
        from server.config import DB_CONNECTION_STRING
        
        # Initialize storage with connection string
        if DB_CONNECTION_STRING:
            await storage.initialize(DB_CONNECTION_STRING)
            logger.info("✅ Database storage initialized")
        else:
            raise Exception("DB_CONNECTION_STRING not available")
        
        logger.info("✅ Test environment initialized")
    
    async def create_test_conversation(self) -> int:
        """Create a test conversation."""
        logger.info("📝 Creating test conversation")
        
        conversation_id = await storage.conversation.create_conversation(
            user_id=self.test_user_id,
            title="Original Title - Should Be Updated"
        )
        
        if not conversation_id:
            raise Exception("Failed to create test conversation")
        
        self.test_conversation_id = conversation_id
        logger.info(f"✅ Created conversation {self.test_conversation_id}")
        
        return self.test_conversation_id
    
    async def test_chat_completion_api(self) -> Dict[str, Any]:
        """Test chat completion API with title generation."""
        logger.info("🎭 Testing chat completion API")
        
        # Message that should trigger title generation
        test_message_data = {
            "conversation_id": self.test_conversation_id,
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Can you explain the fundamental concepts of quantum computing and how they apply to modern cryptography?"
                }
            ]
        }
        
        headers = {
            "Content-Type": "application/json",
            "X-User-ID": self.test_user_id,
            "X-Request-ID": "test-title-generation-001"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                logger.info(f"🔄 Sending request to {self.api_base_url}/chat/completions")
                
                async with session.post(
                    f"{self.api_base_url}/chat/completions",
                    json=test_message_data,
                    headers=headers
                ) as response:
                    
                    logger.info(f"📊 Response status: {response.status}")
                    
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"API request failed with status {response.status}: {error_text}")
                    
                    # Read the streaming response
                    response_lines = []
                    async for line in response.content:
                        line_text = line.decode('utf-8').strip()
                        if line_text:
                            logger.debug(f"📝 Response line: {line_text[:100]}...")
                            response_lines.append(line_text)
                    
                    logger.info(f"✅ API request completed with {len(response_lines)} response lines")
                    
                    # Parse the last response line (should contain "done": true)
                    final_response = None
                    for line in reversed(response_lines):
                        try:
                            parsed = json.loads(line)
                            if isinstance(parsed, dict) and parsed.get("done") is True:
                                final_response = parsed
                                break
                        except json.JSONDecodeError:
                            continue
                    
                    if final_response:
                        logger.info("🎉 Found final response with done=true")
                        return {"success": True, "final_response": final_response}
                    else:
                        logger.warning("⚠️  No final response found")
                        return {"success": False, "response_lines": response_lines}
                        
        except Exception as e:
            logger.error(f"❌ API request failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def verify_title_persistence(self) -> tuple[bool, str]:
        """Verify that the conversation title was updated in the database."""
        logger.info("🔍 Verifying title persistence in database")
        
        # Add a small delay to ensure database operations complete
        await asyncio.sleep(1)
        
        # Get updated conversation from database
        updated_conversation = await storage.conversation.get_conversation(self.test_conversation_id)
        
        if not updated_conversation:
            raise Exception(f"Could not retrieve conversation {self.test_conversation_id}")
        
        logger.info(f"📋 Current conversation title: '{updated_conversation.title}'")
        
        # Check if title was updated from original
        if updated_conversation.title == "Original Title - Should Be Updated":
            logger.warning("⚠️  Title was not updated from original value")
            return False, updated_conversation.title
        
        # Check if title is meaningful (not empty/default)
        if not updated_conversation.title or updated_conversation.title.strip() == "":
            logger.warning("⚠️  Title is empty or whitespace")
            return False, updated_conversation.title
        
        if updated_conversation.title == "New conversation":
            logger.warning("⚠️  Title is still default value")
            return False, updated_conversation.title
        
        logger.info(f"✅ Title successfully updated to: '{updated_conversation.title}'")
        return True, updated_conversation.title
    
    async def cleanup(self):
        """Clean up test data."""
        logger.info("🧹 Cleaning up test data")
        
        try:
            if self.test_conversation_id:
                await storage.conversation.delete_conversation(self.test_conversation_id)
                logger.info(f"✅ Deleted test conversation {self.test_conversation_id}")
        except Exception as e:
            logger.warning(f"⚠️  Failed to cleanup conversation: {e}")
    
    async def run_test(self):
        """Run complete title generation API test."""
        logger.info("🎯 Starting title generation API test")
        
        success = False
        
        try:
            # Setup
            await self.setup()
            
            # Create test conversation
            await self.create_test_conversation()
            
            # Test API call
            logger.info("\n" + "="*50)
            logger.info("TEST: Chat Completion API with Title Generation")
            logger.info("="*50)
            
            api_result = await self.test_chat_completion_api()
            
            if not api_result.get("success", False):
                logger.error("❌ API test failed")
                return False
            
            # Verify database persistence
            logger.info("\n" + "="*50)
            logger.info("TEST: Database Title Persistence Verification")
            logger.info("="*50)
            
            title_updated, final_title = await self.verify_title_persistence()
            
            if title_updated:
                logger.info("🎉 SUCCESS: Title generation and persistence working correctly!")
                logger.info(f"📋 Final title: '{final_title}'")
                success = True
            else:
                logger.error("❌ FAILURE: Title was not properly persisted to database")
                logger.error(f"📋 Current title: '{final_title}'")
                
        except Exception as e:
            logger.error(f"❌ Test failed with error: {e}")
            
        finally:
            # Always cleanup
            await self.cleanup()
            
        return success


async def main():
    """Main test function."""
    tester = TitleGenerationAPITester()
    
    success = await tester.run_test()
    
    if success:
        print("\n🎉 API TEST PASSED - Title generation fix is working!")
        sys.exit(0)
    else:
        print("\n❌ API TEST FAILED - Title generation needs more work")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())