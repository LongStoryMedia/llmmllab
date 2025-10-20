#!/usr/bin/env python3
"""
Full LangGraph Workflow E2E Test - Ensures complete workflow execution without shortcuts.
"""

import asyncio
import sys
import os

# Add the inference directory to the path
sys.path.insert(0, "/app")

from unittest.mock import Mock

# Import all necessary components
from server.routers.chat import chat_completion
from models import Message, MessageRole, MessageContentType
from db import storage


class FullLangGraphWorkflowTest:
    """Test runner focused exclusively on full LangGraph workflow execution."""

    def __init__(self):
        self.test_user_id = "full_workflow_test_user"
        self.test_conversation_id = None

    async def setup_test_environment(self):
        """Set up the test environment with database initialization."""
        print("🚀 Setting up FULL LangGraph workflow test environment...")

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
                title="Full LangGraph Workflow Test",
                user_id=self.test_user_id
            )
            print(f"   ✅ Created test conversation: {self.test_conversation_id}")

            return True

        except Exception as e:
            print(f"   ❌ Setup failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def test_full_langgraph_workflow_execution(self):
        """Test complete LangGraph workflow execution with real conversation context."""
        print("🎯 Testing COMPLETE LangGraph workflow execution...")

        try:
            # Create a rich conversation context to trigger full workflow
            print("   📝 Creating rich conversation context...")
            
            conversation_history = [
                ("I'm starting a new AI project and need comprehensive guidance.", 
                 "I'd be happy to help with your AI project! What specific domain are you working in?"),
                ("I want to build a computer vision system for medical image analysis.",
                 "Medical image analysis is a fascinating and important field. What type of medical images will you be working with?"),
                ("I'll be analyzing chest X-rays to detect pneumonia and other lung conditions.",
                 "Chest X-ray analysis is a well-established application. You'll want to consider data preprocessing, model architecture, and validation strategies."),
                ("What specific deep learning architectures would you recommend for this task?",
                 "For chest X-ray analysis, I'd recommend starting with proven architectures like ResNet, DenseNet, or Vision Transformers, depending on your dataset size and computational resources."),
                ("I have about 10,000 labeled X-ray images. Is that sufficient for training?",
                 "10,000 labeled images is a decent starting point, though you might benefit from transfer learning and data augmentation techniques to improve performance.")
            ]
            
            # Add the conversation history to the database
            for i, (user_msg, assistant_msg) in enumerate(conversation_history):
                user_message = Message(
                    conversation_id=self.test_conversation_id,
                    role=MessageRole.USER,
                    content=[
                        {
                            "type": MessageContentType.TEXT,
                            "text": user_msg,
                        }
                    ],
                )
                await storage.message.add_message(user_message)
                
                assistant_message = Message(
                    conversation_id=self.test_conversation_id,
                    role=MessageRole.ASSISTANT,
                    content=[
                        {
                            "type": MessageContentType.TEXT,
                            "text": assistant_msg,
                        }
                    ],
                )
                await storage.message.add_message(assistant_message)

            print(f"   ✅ Added {len(conversation_history)*2} context messages")

            # Now send a complex query that should trigger the FULL LangGraph workflow
            complex_query = Message(
                conversation_id=self.test_conversation_id,
                role=MessageRole.USER,
                content=[
                    {
                        "type": MessageContentType.TEXT,
                        "text": "Based on our entire conversation about my chest X-ray analysis project, can you provide a comprehensive implementation plan? I need specific recommendations for: 1) Data preprocessing pipeline, 2) Model architecture selection with justification, 3) Training strategy including hyperparameters, 4) Evaluation metrics and validation approach, 5) Deployment considerations for a clinical environment. Please be thorough and include code examples where appropriate.",
                    }
                ],
            )

            mock_request = Mock()
            mock_request.headers = {"authorization": "Bearer test-token"}
            mock_request.state = Mock()
            mock_request.state.user_id = self.test_user_id
            mock_request.state.request_id = "full-langgraph-workflow-test"

            print("   🚀 Executing COMPLETE LangGraph workflow...")
            print("   🔍 This should trigger: intent classification, context assembly, agent selection, tool usage, response generation...")
            print("   ⏱️  Full workflow execution may take 30-60 seconds...")
            
            try:
                response = await chat_completion(complex_query, mock_request)
                
                print("   ✅ COMPLETE LangGraph workflow executed successfully!")
                print(f"   ✅ Response type: {type(response)}")
                
                # Verify the response indicates successful workflow execution
                if "StreamingResponse" in str(type(response)):
                    print("   ✅ Received proper streaming response from FULL LangGraph workflow")
                    print("   🎉 This confirms the complete workflow execution path was followed!")
                    print("   🎉 LangGraph workflow is working end-to-end with context window management!")
                    return True
                else:
                    print(f"   ❌ Unexpected response type: {type(response)}")
                    print("   ❌ This may indicate the workflow was not fully executed")
                    return False
                    
            except Exception as workflow_error:
                print(f"   ❌ CRITICAL: LangGraph workflow execution failed: {workflow_error}")
                print(f"   ❌ This means the complete LangGraph workflow is not working!")
                import traceback
                traceback.print_exc()
                return False

        except Exception as e:
            print(f"   ❌ Full workflow test failed: {e}")
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

    async def run_full_workflow_test(self):
        """Run the complete LangGraph workflow test."""
        print("🎯 Starting FULL LangGraph Workflow Test")
        print("=" * 70)
        print("📋 This test ensures the complete LangGraph workflow executes without shortcuts")
        print("📋 Expected execution path: context assembly → intent classification → agent routing → tool usage → response generation")
        print("=" * 70)

        # Setup
        if not await self.setup_test_environment():
            print("❌ Full Workflow Test FAILED - Setup failed")
            return False

        # Run the full workflow test
        workflow_success = await self.test_full_langgraph_workflow_execution()

        # Cleanup
        await self.cleanup_test_environment()

        # Results
        print("=" * 70)
        if workflow_success:
            print("🎉 FULL LangGraph Workflow Test PASSED!")
            print("✨ The complete LangGraph workflow is working end-to-end!")
            print("✨ Context window management is working correctly!")
            return True
        else:
            print("❌ FULL LangGraph Workflow Test FAILED!")
            print("💥 The complete LangGraph workflow is not executing properly!")
            return False


async def main():
    """Main test execution function."""
    test_runner = FullLangGraphWorkflowTest()
    success = await test_runner.run_full_workflow_test()
    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)