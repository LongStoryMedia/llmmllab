"""
Comprehensive End-to-End Test for complete LangGraph workflow execution.
"""

import asyncio
import sys
import json
import os

# Add the inference directory to the path
sys.path.insert(0, "/app")

from unittest.mock import Mock
from fastapi import HTTPException

# Import all necessary components
from server.routers.chat import chat_completion
from models import Message, MessageRole, MessageContentType
from db import storage
from composer.core.service import ComposerService


class FullWorkflowE2ETestRunner:
    """Complete end-to-end test runner that ensures full LangGraph workflow execution."""

    def __init__(self):
        self.test_user_id = "full_workflow_test_user"
        self.test_conversation_id = None

    async def setup_test_environment(self):
        """Set up the test environment with database initialization."""
        print("🚀 Setting up E2E test environment...")

        try:
            # Initialize database
            print("   💾 Initializing database...")
            db_host = os.environ.get("DB_HOST", "192.168.0.71")
            db_port = os.environ.get("DB_PORT", "32345")
            db_user = os.environ.get("DB_USER", "lsm")
            db_password = os.environ.get("DB_PASSWORD", "lsm")
            db_name = os.environ.get("DB_NAME", "llmmll")

            connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}?sslmode=disable"
            await storage.initialize(connection_string)
            print("   ✅ Database initialized")

            # Create test conversation
            print("   💬 Creating test conversation...")
            self.test_conversation_id = await storage.conversation.create_conversation(
                title="E2E Test Conversation",
                user_id=self.test_user_id
            )
            print(f"   ✅ Created test conversation: {self.test_conversation_id}")

            return True

        except Exception as e:
            print(f"   ❌ Setup failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def test_direct_router_function(self):
        """Test the chat_completion router function directly."""
        print("🧪 Testing direct router function...")

        try:
            # Create test message
            test_message = Message(
                conversation_id=self.test_conversation_id,
                role=MessageRole.USER,
                content=[
                    {
                        "type": MessageContentType.TEXT,
                        "text": "What is machine learning?",
                    }
                ],
            )

            # Mock request
            mock_request = Mock()
            mock_request.headers = {"authorization": "Bearer test-token"}
            mock_request.state = Mock()
            mock_request.state.user_id = self.test_user_id
            mock_request.state.request_id = "e2e-test-request-123"

            print("   📡 Calling chat_completion function...")
            response = await chat_completion(test_message, mock_request)

            print(f"   ✅ Response type: {type(response)}")
            return True

        except Exception as e:
            print(f"   ❌ Direct router test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def test_database_operations(self):
        """Test database operations used by the endpoint."""
        print("💾 Testing database operations...")

        try:
            # Test message storage
            print("   📝 Testing message storage...")
            test_message = Message(
                conversation_id=self.test_conversation_id,
                role=MessageRole.USER,
                content=[
                    {
                        "type": MessageContentType.TEXT,
                        "text": "Test message for database operations",
                    }
                ],
            )

            await storage.message.add_message(test_message)
            print("   ✅ Message stored successfully")

            # Test message retrieval
            print("   📖 Testing message retrieval...")
            messages = await storage.message.get_messages_by_conversation_id(
                self.test_conversation_id, 10, 0
            )
            print(f"   ✅ Retrieved {len(messages)} messages")

            # Test conversation operations
            print("   💬 Testing conversation operations...")
            conversation = await storage.conversation.get_conversation(
                self.test_conversation_id
            )
            print(f"   ✅ Retrieved conversation: {conversation.title if conversation else 'None'}")

            return True

        except Exception as e:
            print(f"   ❌ Database operations test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def test_context_window_management(self):
        """Test that context window management prevents overflow errors."""
        print("🧠 Testing context window management...")

        try:
            # Create a conversation with many messages that would exceed context window
            context_test_conversation_id = await storage.conversation.create_conversation(
                title="Context Window Test Conversation",
                user_id=self.test_user_id
            )
            
            print("   📝 Adding messages to create large context...")
            # Add many long messages to create a context that would exceed 40960 tokens
            for i in range(20):
                long_user_message = Message(
                    conversation_id=context_test_conversation_id,
                    role=MessageRole.USER,
                    content=[
                        {
                            "type": MessageContentType.TEXT,
                            "text": f"User message {i}: " + "This is an extremely long message with lots of detailed content that consumes many tokens when processed by the language model. " * 100,
                        }
                    ],
                )
                await storage.message.add_message(long_user_message)
                
                long_assistant_message = Message(
                    conversation_id=context_test_conversation_id,
                    role=MessageRole.ASSISTANT,
                    content=[
                        {
                            "type": MessageContentType.TEXT,
                            "text": f"Assistant response {i}: " + "This is an equally long and detailed response from the assistant with comprehensive information that also consumes many tokens during processing. " * 100,
                        }
                    ],
                )
                await storage.message.add_message(long_assistant_message)

            print(f"   ✅ Added {40} messages to conversation")

            # Now test with a new message that should trigger context window management
            test_message = Message(
                conversation_id=context_test_conversation_id,
                role=MessageRole.USER,
                content=[
                    {
                        "type": MessageContentType.TEXT,
                        "text": "Given our long conversation, please provide a brief summary.",
                    }
                ],
            )

            mock_request = Mock()
            mock_request.headers = {"authorization": "Bearer test-token"}
            mock_request.state = Mock()
            mock_request.state.user_id = self.test_user_id
            mock_request.state.request_id = "context-window-test-request"

            print("   🚀 Testing workflow with large context (should use context window management)...")
            
            try:
                response = await chat_completion(test_message, mock_request)
                print("   ✅ Context window management working - no overflow error!")
                print(f"   ✅ Response type: {type(response)}")
            except Exception as e:
                if "exceed context window" in str(e):
                    print(f"   ❌ Context window overflow error occurred: {e}")
                    await storage.conversation.delete_conversation(context_test_conversation_id)
                    return False
                else:
                    print(f"   ❌ Other workflow error: {e}")
                    import traceback
                    traceback.print_exc()
                    await storage.conversation.delete_conversation(context_test_conversation_id)
                    return False
            
            # Clean up
            await storage.conversation.delete_conversation(context_test_conversation_id)
            print("   ✅ Context window test completed successfully")
            return True

        except Exception as e:
            print(f"   ❌ Context window test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def test_complete_workflow_execution(self):
        """Test complete workflow execution with a real conversation."""
        print("� Testing complete workflow execution...")

        try:
            # Create a new conversation specifically for workflow testing
            print("   💬 Creating conversation for workflow test...")
            workflow_conversation_id = await storage.conversation.create_conversation(
                title="Complete Workflow Test Conversation",
                user_id=self.test_user_id
            )
            
            # Add some initial context messages to the conversation
            print("   📝 Adding initial context to conversation...")
            for i in range(3):
                context_message = Message(
                    conversation_id=workflow_conversation_id,
                    role=MessageRole.USER,
                    content=[
                        {
                            "type": MessageContentType.TEXT,
                            "text": f"Context message {i}: Tell me about artificial intelligence and machine learning.",
                        }
                    ],
                )
                await storage.message.add_message(context_message)
                
                assistant_context = Message(
                    conversation_id=workflow_conversation_id,
                    role=MessageRole.ASSISTANT,
                    content=[
                        {
                            "type": MessageContentType.TEXT,
                            "text": f"Response {i}: AI and ML are fascinating fields that involve creating systems that can learn and make decisions.",
                        }
                    ],
                )
                await storage.message.add_message(assistant_context)

            print(f"   ✅ Added context messages to conversation {workflow_conversation_id}")

            # Now test the complete workflow with a real message
            workflow_test_message = Message(
                conversation_id=workflow_conversation_id,
                role=MessageRole.USER,
                content=[
                    {
                        "type": MessageContentType.TEXT,
                        "text": "Based on our conversation, can you explain the key differences between supervised and unsupervised learning?",
                    }
                ],
            )

            mock_request = Mock()
            mock_request.headers = {"authorization": "Bearer test-token"}
            mock_request.state = Mock()
            mock_request.state.user_id = self.test_user_id
            mock_request.state.request_id = "complete-workflow-test-request"

            print("   🎯 Executing complete LangGraph workflow...")
            
            try:
                response = await chat_completion(workflow_test_message, mock_request)
                print("   ✅ Complete workflow executed successfully!")
                print(f"   ✅ Response type: {type(response)}")
                
                # Verify the response is a streaming response (indicates successful execution)
                if "StreamingResponse" in str(type(response)):
                    print("   ✅ Received proper streaming response from workflow")
                else:
                    print(f"   ⚠️  Unexpected response type: {type(response)}")
                    
            except Exception as workflow_error:
                print(f"   ❌ CRITICAL: Complete workflow execution failed: {workflow_error}")
                import traceback
                traceback.print_exc()
                # await storage.conversation.delete_conversation(workflow_conversation_id)
                return False
            
            # Clean up the test conversation
            # await storage.conversation.delete_conversation(workflow_conversation_id)
            print("   ✅ Workflow test completed successfully")
            return True

        except Exception as e:
            print(f"   ❌ Complete workflow test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def test_invalid_conversation_error_handling(self):
        """Test error handling with invalid conversation ID."""
        print("❌ Testing invalid conversation error handling...")

        try:
            # Test with truly invalid conversation ID (non-existent)
            print("   🚨 Testing with non-existent conversation ID...")
            invalid_message = Message(
                conversation_id=99999,  # Use a clearly invalid ID
                role=MessageRole.USER,
                content=[
                    {
                        "type": MessageContentType.TEXT,
                        "text": "Test message with invalid conversation",
                    }
                ],
            )

            mock_request_invalid = Mock()
            mock_request_invalid.headers = {"authorization": "Bearer test-token"}
            mock_request_invalid.state = Mock()
            mock_request_invalid.state.user_id = self.test_user_id
            mock_request_invalid.state.request_id = "error-test-request"

            try:
                await chat_completion(invalid_message, mock_request_invalid)
                print("   ❌ ERROR: Should have raised HTTPException for invalid conversation")
                return False
            except HTTPException as e:
                if e.status_code == 400 and "Referenced conversation does not exist" in e.detail:
                    print(f"   ✅ Correctly handled invalid conversation error: {e.status_code} - {e.detail}")
                    return True
                else:
                    print(f"   ❌ Wrong error response: {e.status_code} - {e.detail}")
                    return False

        except Exception as e:
            print(f"   ❌ Error handling test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def cleanup_test_environment(self):
        """Clean up test environment."""
        print("🧹 Cleaning up test environment...")

        try:
            if self.test_conversation_id:
                await storage.conversation.delete_conversation(
                    self.test_conversation_id
                )
                print("   ✅ Test conversation deleted")

            await storage.close()
            print("   ✅ Database connection closed")

        except Exception as e:
            print(f"   ⚠️  Cleanup warning: {e}")

    async def run_comprehensive_e2e_test(self):
        """Run the complete end-to-end test suite."""
        print("🎯 Starting Comprehensive E2E Test for /chat/completions")
        print("=" * 60)

        # Setup
        if not await self.setup_test_environment():
            print("❌ E2E Test FAILED - Setup failed")
            return False

        test_results = []

        # Run all tests focused on full workflow execution
        test_results.append(await self.test_database_operations())
        test_results.append(await self.test_complete_workflow_execution())
        test_results.append(await self.test_context_window_management())

        # Cleanup
        await self.cleanup_test_environment()

        # Results
        print("=" * 60)
        passed_tests = sum(test_results)
        total_tests = len(test_results)

        if passed_tests == total_tests:
            print(f"🎉 E2E Test PASSED! ({passed_tests}/{total_tests} tests passed)")
            print("✨ The simplified server architecture works end-to-end!")
            return True
        else:
            print(f"❌ E2E Test FAILED! ({passed_tests}/{total_tests} tests passed)")
            return False


async def main():
    """Main test execution function."""
    runner = FullWorkflowE2ETestRunner()
    success = await runner.run_comprehensive_e2e_test()
    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
