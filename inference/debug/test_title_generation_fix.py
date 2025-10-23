#!/usr/bin/env python3
"""
Test title generation fix to ensure titles are properly persisted to the database.
This test validates that the title generation workflow updates the conversation title.
"""

import asyncio
import sys
import json
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, '/app')

from db import storage
from models.conversation import Conversation
from models.message import Message, MessageRole, MessageContent, MessageContentType
from composer.interface import ComposerInterface
from composer.agents import ClassifierAgent
from runner import PipelineFactory
import utils.logging as logging

logger = logging.llmmllogger.logger.bind(component="TitleGenerationTest")


class TitleGenerationTester:
    """Test title generation and database persistence."""
    
    def __init__(self):
        self.test_user_id = "title_test_user_12345"
        self.test_conversation_id = None
        self.pipeline_factory = None
        self.composer = None
        
    async def setup(self):
        """Initialize test environment."""
        logger.info("🚀 Setting up title generation test environment")
        
        # Initialize storage
        await storage.init_db()
        
        # Initialize pipeline factory and composer
        self.pipeline_factory = PipelineFactory()
        await self.pipeline_factory.initialize()
        
        self.composer = ComposerInterface(self.pipeline_factory)
        await self.composer.initialize()
        
        logger.info("✅ Test environment initialized")
    
    async def create_test_conversation(self) -> int:
        """Create a test conversation."""
        logger.info("📝 Creating test conversation")
        
        conversation = await storage.conversation.create_conversation(
            user_id=self.test_user_id,
            title="Original Title - Should Be Updated"
        )
        
        self.test_conversation_id = conversation.id
        logger.info(f"✅ Created conversation {self.test_conversation_id}")
        
        return self.test_conversation_id
    
    async def add_test_message(self, content: str) -> int:
        """Add a test message to trigger title generation."""
        logger.info(f"💬 Adding test message: {content[:50]}...")
        
        message = Message(
            conversation_id=self.test_conversation_id,
            role=MessageRole.USER,
            content=[
                MessageContent(
                    type=MessageContentType.TEXT,
                    text=content
                )
            ]
        )
        
        message_id = await storage.message.add_message(message)
        logger.info(f"✅ Added message {message_id}")
        
        return message_id
    
    async def test_title_generation_workflow(self):
        """Test that the title generation workflow produces a title."""
        logger.info("🎯 Testing title generation workflow directly")
        
        # Create a message that should trigger title generation
        user_message = "Can you help me understand quantum computing principles and their applications in modern technology?"
        await self.add_test_message(user_message)
        
        # Get conversation messages for title generation
        messages = await storage.message.get_messages_by_conversation(self.test_conversation_id)
        
        # Test classifier agent directly
        classifier_agent = ClassifierAgent(self.pipeline_factory)
        
        # Convert to format expected by classifier
        message_list = []
        for msg in messages:
            content_text = ""
            if msg.content:
                for content_item in msg.content:
                    if content_item.type == MessageContentType.TEXT:
                        content_text += content_item.text
            
            message_list.append({
                "role": msg.role.value,
                "content": content_text
            })
        
        logger.info(f"📋 Testing title generation with {len(message_list)} messages")
        
        try:
            generated_title = await classifier_agent.generate_title(message_list)
            logger.info(f"🎉 Generated title: '{generated_title}'")
            
            # Verify title was generated
            if not generated_title or len(generated_title.strip()) == 0:
                raise Exception("Title generation returned empty result")
                
            return generated_title
            
        except Exception as e:
            logger.error(f"❌ Title generation failed: {e}")
            raise
    
    async def test_composer_chat_completion(self):
        """Test complete chat completion workflow with title generation."""
        logger.info("🎭 Testing complete composer workflow")
        
        # Message that should trigger title generation
        test_message = "Explain the basics of machine learning and how neural networks work"
        
        try:
            # Use composer for full workflow
            response_events = []
            async for event in self.composer.chat_completion(
                user_id=self.test_user_id,
                conversation_id=self.test_conversation_id,
                message=test_message
            ):
                response_events.append(event)
                
                # Log key events
                if isinstance(event, dict):
                    event_type = event.get("event", "")
                    if "title" in str(event).lower():
                        logger.info(f"📌 Title-related event: {event_type}")
            
            logger.info(f"✅ Composer workflow completed with {len(response_events)} events")
            return response_events
            
        except Exception as e:
            logger.error(f"❌ Composer workflow failed: {e}")
            raise
    
    async def verify_title_persistence(self, expected_title: str = None):
        """Verify that the conversation title was updated in the database."""
        logger.info("🔍 Verifying title persistence in database")
        
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
    
    async def run_comprehensive_test(self):
        """Run complete title generation test suite."""
        logger.info("🎯 Starting comprehensive title generation test")
        
        success = False
        
        try:
            # Setup
            await self.setup()
            
            # Create test conversation
            await self.create_test_conversation()
            
            # Test 1: Direct title generation
            logger.info("\n" + "="*50)
            logger.info("TEST 1: Direct Title Generation")
            logger.info("="*50)
            
            generated_title = await self.test_title_generation_workflow()
            
            # Test 2: Full composer workflow (this should update the database)
            logger.info("\n" + "="*50)
            logger.info("TEST 2: Complete Composer Workflow")
            logger.info("="*50)
            
            await self.test_composer_chat_completion()
            
            # Test 3: Verify database persistence
            logger.info("\n" + "="*50)
            logger.info("TEST 3: Database Persistence Verification")
            logger.info("="*50)
            
            title_updated, final_title = await self.verify_title_persistence(generated_title)
            
            if title_updated:
                logger.info("🎉 SUCCESS: Title generation and persistence working correctly!")
                success = True
            else:
                logger.error("❌ FAILURE: Title was not properly persisted to database")
                
        except Exception as e:
            logger.error(f"❌ Test failed with error: {e}")
            
        finally:
            # Always cleanup
            await self.cleanup()
            
        return success


async def main():
    """Main test function."""
    tester = TitleGenerationTester()
    
    success = await tester.run_comprehensive_test()
    
    if success:
        print("\n🎉 ALL TESTS PASSED - Title generation fix is working!")
        sys.exit(0)
    else:
        print("\n❌ TESTS FAILED - Title generation needs more work")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())