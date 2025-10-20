"""
Comprehensive End-to-End Test for /chat/completions endpoint.

Tests the complete flow from HTTP request through FastAPI routing,
authentication, database operations, and composer workflow execution.
"""

import asyncio
import sys
import json
import os
from typing import AsyncGenerator

# Add the inference directory to the path
sys.path.insert(0, "/app")

import pytest
from fastapi.testclient import TestClient
from fastapi import HTTPException
from unittest.mock import Mock, AsyncMock, patch

# Import all necessary components
from server.app import app
from server.routers.chat import chat_completion
from models import Message, MessageRole, MessageContentType
from db import storage
import composer


class E2ETestRunner:
    """Comprehensive end-to-end test runner for the chat completions endpoint."""

    def __init__(self):
        self.test_user_id = "e2e_test_user"
        self.test_conversation_id = None
        self.setup_complete = False

    async def setup_test_environment(self):
        """Set up the test environment with database and composer initialization."""
        print("🚀 Setting up E2E test environment...")

        try:
            # Initialize database
            print("   💾 Initializing database...")
            db_host = os.environ.get("DB_HOST", "localhost")
            db_port = os.environ.get("DB_PORT", "5432")
            db_user = os.environ.get("DB_USER", "lsm")
            db_password = os.environ.get("DB_PASSWORD", "")
            db_name = os.environ.get("DB_NAME", "llmmll")

            connection_string = (
                f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
            )
            await storage.initialize(connection_string)
            print("   ✅ Database initialized")

            # Initialize composer
            print("   🎼 Initializing composer...")
            await composer.initialize_composer()
            print("   ✅ Composer initialized")

            # Create test user if not exists (simulate user creation)
            print("   👤 Setting up test user...")

            # Create test conversation
            print("   💬 Creating test conversation...")
            self.test_conversation_id = await storage.conversation.create_conversation(
                self.test_user_id, "E2E Test Conversation"
            )
            print(f"   ✅ Created test conversation: {self.test_conversation_id}")

            self.setup_complete = True
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
                        "text": "Hello, this is an e2e test message for the simplified architecture!",
                    }
                ],
            )

            # Mock request with authentication
            mock_request = Mock()
            mock_request.headers = {"authorization": "Bearer test-token"}
            mock_request.state = Mock()
            mock_request.state.user_id = self.test_user_id
            mock_request.state.request_id = "e2e-test-request-123"

            # Mock background tasks
            from fastapi import BackgroundTasks

            background_tasks = BackgroundTasks()

            print("   📡 Calling chat_completion function...")
            response = await chat_completion(
                test_message, mock_request, background_tasks
            )

            print(f"   ✅ Response type: {type(response)}")

            # Test streaming response
            if hasattr(response, "body_iterator"):
                print("   📄 Testing streaming response...")
                event_count = 0
                content_received = ""

                async for chunk in response.body_iterator:
                    decoded_chunk = chunk.decode("utf-8")
                    print(f"   📦 Chunk {event_count}: {decoded_chunk[:100]}...")

                    # Parse SSE data
                    if decoded_chunk.startswith("data: "):
                        try:
                            data_json = json.loads(decoded_chunk[6:])
                            if "content" in data_json:
                                content_received += data_json["content"]
                        except json.JSONDecodeError:
                            pass

                    event_count += 1
                    if event_count >= 10:  # Test first 10 chunks
                        break

                print(f"   ✅ Received {event_count} streaming chunks")
                print(f"   📝 Content preview: {content_received[:200]}...")

            return True

        except Exception as e:
            print(f"   ❌ Direct router test failed: {e}")
            import traceback

            traceback.print_exc()
            return False

    async def test_composer_integration_flow(self):
        """Test the full composer integration flow."""
        print("🎼 Testing composer integration flow...")

        try:
            # Test workflow composition
            print("   🔧 Testing workflow composition...")
            workflow = await composer.compose_workflow(self.test_user_id)
            print(f"   ✅ Workflow created: {type(workflow)}")

            # Test initial state creation
            print("   🏁 Testing initial state creation...")
            initial_state = await composer.create_initial_state(
                self.test_user_id, self.test_conversation_id
            )
            print(f"   ✅ Initial state created: {type(initial_state)}")

            # Test workflow execution
            print("   🚀 Testing workflow execution...")
            event_count = 0
            async for event in composer.execute_workflow(
                workflow, initial_state, stream=True
            ):
                print(f"   📡 Event {event_count}: {str(event)[:100]}...")
                event_count += 1
                if event_count >= 5:  # Test first 5 events
                    break

            print(
                f"   ✅ Composer workflow executed successfully ({event_count} events)"
            )
            return True

        except Exception as e:
            print(f"   ❌ Composer integration test failed: {e}")
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
            print(
                f"   ✅ Retrieved conversation: {conversation.title if conversation else 'None'}"
            )

            return True

        except Exception as e:
            print(f"   ❌ Database operations test failed: {e}")
            import traceback

            traceback.print_exc()
            return False

    async def test_error_handling(self):
        """Test error handling scenarios."""
        print("🚨 Testing error handling...")

        try:
            # Test with invalid conversation ID
            print("   ❌ Testing invalid conversation ID...")
            invalid_message = Message(
                conversation_id=99999,  # Non-existent conversation
                role=MessageRole.USER,
                content=[
                    {
                        "type": MessageContentType.TEXT,
                        "text": "Test message with invalid conversation",
                    }
                ],
            )

            mock_request = Mock()
            mock_request.headers = {"authorization": "Bearer test-token"}
            mock_request.state = Mock()
            mock_request.state.user_id = self.test_user_id
            mock_request.state.request_id = "error-test-request"

            from fastapi import BackgroundTasks

            background_tasks = BackgroundTasks()

            try:
                await chat_completion(invalid_message, mock_request, background_tasks)
                print("   ❌ ERROR: Should have raised HTTPException")
                return False
            except HTTPException as e:
                print(f"   ✅ Correctly handled error: {e.detail}")
                return True

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

        # Run all tests
        test_results.append(await self.test_database_operations())
        test_results.append(await self.test_composer_integration_flow())
        test_results.append(await self.test_direct_router_function())
        test_results.append(await self.test_error_handling())

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
    runner = E2ETestRunner()
    success = await runner.run_comprehensive_e2e_test()
    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
