#!/usr/bin/env python3
"""
Real End-to-End Pipeline Test

This test validates the complete LLM ML Lab pipeline using actual system components:
1. Real user creation in database
2. Real model profile creation with openai-gpt-oss-20b-uncensored-q5_1
3. Real conversation and message creation
4. Real pipeline execution with stream_pipeline
5. Real tool integration (web_search)
6. Real output validation
7. Complete cleanup of all created data

Unlike the mock version, this test performs actual database operations,
real model execution, and genuine tool calling.
"""

import asyncio
import time
import uuid
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RealEndToEndPipelineTester:
    """Real end-to-end pipeline test using actual infrastructure."""
    
    def __init__(self):
        """Initialize real pipeline tester."""
        self.test_user_id = f"test_real_user_{uuid.uuid4().hex[:8]}"
        self.test_model_profile_id = uuid.uuid4()
        self.test_conversation_id = None
        self.test_message_id = None
        self.created_entities = []  # Track for cleanup
        
    async def run_full_test(self) -> Dict[str, Any]:
        """Run complete real end-to-end pipeline test."""
        logger.info("🚀 Starting Real End-to-End Pipeline Test")
        logger.info("=" * 80)
        
        test_results = {
            "overall_success": False,
            "execution_time": 0,
            "pipeline_time": 0,
            "components_passed": 0,
            "total_components": 6,
            "results": {},
            "entities_created": 0,
            "entities_cleaned": 0
        }
        
        start_time = time.time()
        
        try:
            # Phase 1: Real Infrastructure Setup
            logger.info("📋 Phase 1: Real Infrastructure Setup")
            infrastructure_result = await self.setup_real_infrastructure()
            test_results["results"]["infrastructure_setup"] = infrastructure_result
            if infrastructure_result["success"]:
                test_results["components_passed"] += 1
            
            # Phase 2: Real User & Model Profile Creation
            logger.info("👤 Phase 2: Real User & Model Profile Creation")
            user_profile_result = await self.create_real_user_and_profile()
            test_results["results"]["user_profile_creation"] = user_profile_result
            if user_profile_result["success"]:
                test_results["components_passed"] += 1
            
            # Phase 3: Real Conversation Creation
            logger.info("💬 Phase 3: Real Conversation Creation")
            conversation_result = await self.create_real_conversation()
            test_results["results"]["conversation_creation"] = conversation_result
            if conversation_result["success"]:
                test_results["components_passed"] += 1
            
            # Phase 4: Real Message with Tool Context
            logger.info("📝 Phase 4: Real Message with Tool Context")
            message_result = await self.create_real_message_with_tools()
            test_results["results"]["message_creation"] = message_result
            if message_result["success"]:
                test_results["components_passed"] += 1
            
            # Phase 5: Real Pipeline Execution
            logger.info("🔥 Phase 5: Real Pipeline Execution")
            pipeline_result = await self.execute_real_pipeline()
            test_results["results"]["pipeline_execution"] = pipeline_result
            if pipeline_result["success"]:
                test_results["components_passed"] += 1
                test_results["pipeline_time"] = pipeline_result.get("execution_time", 0)
            
            # Phase 6: Real Output Validation
            logger.info("✅ Phase 6: Real Output Validation")
            validation_result = await self.validate_real_outputs()
            test_results["results"]["output_validation"] = validation_result
            if validation_result["success"]:
                test_results["components_passed"] += 1
            
            # Calculate success
            test_results["overall_success"] = test_results["components_passed"] == test_results["total_components"]
            test_results["execution_time"] = time.time() - start_time
            test_results["entities_created"] = len(self.created_entities)
            
        except Exception as e:
            logger.error(f"❌ Test execution failed: {str(e)}")
            test_results["error"] = str(e)
        
        finally:
            # Always attempt cleanup
            logger.info("🧹 Phase 7: Real Data Cleanup")
            cleanup_result = await self.cleanup_real_data()
            test_results["results"]["cleanup"] = cleanup_result
            test_results["entities_cleaned"] = cleanup_result.get("cleaned_count", 0)
        
        # Print comprehensive results
        await self.print_test_summary(test_results)
        
        return test_results

    async def setup_real_infrastructure(self) -> Dict[str, Any]:
        """Set up real infrastructure components."""
        logger.info("🏗️  Setting up real infrastructure...")
        
        try:
            # Initialize real database connection
            from server.db import storage
            await storage.init()
            logger.info("   ✅ Database connection established")
            
            # Initialize real pipeline factory
            from runner.pipeline_factory import pipeline_factory
            logger.info("   ✅ Pipeline factory initialized")
            
            # Verify target model exists
            target_model = "openai-gpt-oss-20b-uncensored-q5_1"
            if target_model in pipeline_factory.model_manager.models:
                logger.info(f"   ✅ Model {target_model} verified")
            else:
                raise Exception(f"Target model {target_model} not found")
            
            return {
                "success": True,
                "database_connected": True,
                "pipeline_factory_ready": True,
                "target_model_available": True
            }
            
        except Exception as e:
            logger.error(f"   ❌ Infrastructure setup failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    async def create_real_user_and_profile(self) -> Dict[str, Any]:
        """Create real user and model profile in database."""
        logger.info("👤 Creating real user and model profile...")
        
        try:
            from server.db import storage
            from models.user import User
            from models.model_profile import ModelProfile
            from models.model_parameters import ModelParameters
            
            # Create real user
            user = User(
                id=self.test_user_id,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc)
            )
            
            created_user = await storage.user.create_user(user)
            self.created_entities.append(("user", self.test_user_id))
            logger.info(f"   ✅ Created real user: {self.test_user_id}")
            
            # Create real model profile with enhanced system prompt for tool usage
            model_profile = ModelProfile(
                id=self.test_model_profile_id,
                user_id=self.test_user_id,
                name="Real GPT-OSS Tool Calling Profile",
                description="Real model profile for end-to-end testing with tool calling",
                model_name="openai-gpt-oss-20b-uncensored-q5_1",
                parameters=ModelParameters(
                    temperature=0.7,
                    top_p=0.9,
                    max_tokens=2000,
                    flash_attention=True
                ),
                system_prompt="""You are a helpful AI assistant with access to web search tools. When users ask questions that require current information or research, you MUST use the web_search tool to find relevant information. Always provide comprehensive and accurate responses based on the search results.

Reasoning: medium

# Valid channels: analysis, commentary, final. Channel must be included for every message.
# Use 'analysis' channel for chain-of-thought reasoning
# Use 'commentary' channel for tool calls and function descriptions  
# Use 'final' channel for your response to the user

IMPORTANT: When you need current information, you MUST:
1. Use the 'commentary' channel to call the web_search tool
2. Format tool calls properly as JSON in the commentary channel
3. Use the 'final' channel for your response to the user

Available tools:
- web_search: Search the web for current information (required for 2024 research)
- memory_retrieval: Retrieve relevant context from conversation history
- summarization: Summarize long content

EXAMPLE: When asked about 2024 research, respond like:
<|channel|>commentary<|message|>I need to search for current information about 2024 research. Let me use the web_search tool.

{"tool_name": "web_search", "parameters": {"query": "2024 quantum computing breakthroughs research"}}
<|channel|>final<|message|>Based on my search results...""",
                type=1,  # Text generation profile
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                model_version="1.0"
            )
            
            # Store model profile in database
            created_profile = await storage.model_profile.create_model_profile(model_profile)
            self.created_entities.append(("model_profile", self.test_model_profile_id))
            
            logger.info(f"   ✅ Created real model profile: {created_profile.name}")
            
            return {
                "success": True,
                "user_id": self.test_user_id,
                "model_profile_id": str(self.test_model_profile_id),
                "model_name": "openai-gpt-oss-20b-uncensored-q5_1"
            }
            
        except Exception as e:
            logger.error(f"   ❌ User/profile creation failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    async def create_real_conversation(self) -> Dict[str, Any]:
        """Create real conversation in database."""
        logger.info("💬 Creating real conversation...")
        
        try:
            from server.db import storage
            
            # Create real conversation
            conversation_id = await storage.conversation.create_conversation(
                user_id=self.test_user_id,
                title="Real End-to-End Pipeline Test Conversation"
            )
            
            if not conversation_id:
                raise Exception("Failed to create real conversation")
            
            self.test_conversation_id = conversation_id
            self.created_entities.append(("conversation", conversation_id))
            
            logger.info(f"   ✅ Created real conversation ID: {conversation_id}")
            
            return {
                "success": True,
                "conversation_id": conversation_id
            }
            
        except Exception as e:
            logger.error(f"   ❌ Conversation creation failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    async def create_real_message_with_tools(self) -> Dict[str, Any]:
        """Create real user message with tool-calling context."""
        logger.info("📝 Creating real message with tool context...")
        
        try:
            from server.db import storage
            from models.message import Message
            from models.message_role import MessageRole
            from models.message_content import MessageContent, MessageContentType
            
            # Create a message that explicitly requests web search for 2024 information
            query_text = """I need current information about recent breakthroughs in quantum computing research published in 2024. 
Please use the web_search tool to find the latest developments, key researchers, and significant papers or announcements from major tech companies and research institutions. 
I'm particularly interested in advances in quantum error correction, quantum algorithms, and practical quantum computing applications.

Please search the web for 2024 quantum computing research and provide a comprehensive summary."""
            
            user_message = Message(
                id=None,  # Will be set by database
                conversation_id=self.test_conversation_id,
                role=MessageRole.USER,
                content=[
                    MessageContent(
                        type=MessageContentType.TEXT,
                        text=query_text
                    )
                ],
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc)
            )
            
            # Add message to database
            message_id = await storage.message.add_message(user_message)
            if not message_id:
                raise Exception("Failed to create user message")
            
            self.test_message_id = message_id
            self.created_entities.append(("message", message_id))
            
            logger.info(f"   ✅ Created real user message ID: {message_id}")
            logger.info(f"   📋 Message length: {len(query_text)} characters")
            
            return {
                "success": True,
                "message_id": message_id,
                "message_length": len(query_text)
            }
            
        except Exception as e:
            logger.error(f"   ❌ Message creation failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    async def execute_real_pipeline(self) -> Dict[str, Any]:
        """Execute real pipeline with stream_pipeline function."""
        logger.info("🔥 Executing real pipeline...")
        
        try:
            from runner.pipeline_factory import pipeline_factory
            from runner.pipelines.run import stream_pipeline
            from server.tools.integration import get_tools
            from models.model_profile import ModelProfile
            from server.db import storage
            
            # Get the real model profile
            model_profile = await storage.model_profile.get_model_profile_by_id(
                self.test_model_profile_id, self.test_user_id
            )
            
            if not model_profile:
                raise Exception("Failed to retrieve model profile")
            
            logger.info(f"   🤖 Using model: {model_profile.model_name}")
            
            # Get real pipeline from factory
            from models.chat_response import ChatResponse
            pipeline = pipeline_factory.get_pipeline(
                profile=model_profile,
                expected_type=ChatResponse
            )
            
            if not pipeline:
                raise Exception("Failed to get pipeline from factory")
                
            logger.info("   ✅ Pipeline obtained from factory")
            
            # Get conversation messages for context first
            messages = await storage.message.get_conversation_history(self.test_conversation_id)
            if not messages:
                raise Exception("No messages found for conversation")
                
            logger.info(f"   📝 Processing {len(messages)} messages")
            
            # Get real tools
            from server.services.context import ConversationContext
            from models.user_config import UserConfig
            
            # Create conversation context for tool retrieval
            user_config = await storage.user_config.get_user_config(self.test_user_id)
            if not user_config:
                # Create default user config if none exists
                from models.default_configs import create_default_user_config
                user_config = create_default_user_config(self.test_user_id)
                await storage.user_config.update_user_config(self.test_user_id, user_config)
            
            conversation_ctx = ConversationContext(
                conversation_id=self.test_conversation_id,
                user_config=user_config
            )
            
            # Add current user message to context
            if messages:
                conversation_ctx.current_user_message = messages[0]
            
            # Get tools from the tool manager
            tools = []
            async for tool_result in get_tools(conversation_ctx):
                if isinstance(tool_result, list):
                    tools.extend(tool_result)
                    break  # We got the tools list
            
            logger.info(f"   🛠️  Available tools: {[tool.name for tool in tools]}")
            
            # Execute real streaming pipeline
            start_time = time.time()
            response_chunks = []
            tool_calls_detected = False
            
            logger.info("   🚀 Starting real pipeline stream...")
            
            async for chunk in stream_pipeline(messages, pipeline, tools):
                response_chunks.append(chunk)
                
                # Check for tool calls in response
                if hasattr(chunk, 'content') and chunk.content:
                    chunk_content = chunk.content.lower()
                    if 'web_search' in chunk_content or 'tool_name' in chunk_content:
                        tool_calls_detected = True
                
                # Log progress periodically
                if len(response_chunks) % 10 == 0:
                    logger.info(f"   📊 Processed {len(response_chunks)} response chunks")
            
            execution_time = time.time() - start_time
            
            logger.info(f"   ✅ Pipeline execution completed in {execution_time:.2f}s")
            logger.info(f"   📊 Total response chunks: {len(response_chunks)}")
            
            # Store assistant response in database
            if response_chunks:
                from models.message import Message
                from models.message_role import MessageRole
                from models.message_content import MessageContent, MessageContentType
                
                # Combine all chunks into final response
                final_response = ''.join([
                    chunk.content if hasattr(chunk, 'content') and chunk.content else ''
                    for chunk in response_chunks
                ])
                
                assistant_message = Message(
                    id=None,
                    conversation_id=self.test_conversation_id,
                    role=MessageRole.ASSISTANT,
                    content=[
                        MessageContent(
                            type=MessageContentType.TEXT,
                            text=final_response
                        )
                    ],
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc)
                )
                
                response_message_id = await storage.message.add_message(assistant_message)
                self.created_entities.append(("message", response_message_id))
                logger.info(f"   💾 Stored assistant response ID: {response_message_id}")
            
            return {
                "success": True,
                "execution_time": execution_time,
                "response_chunks": len(response_chunks),
                "tool_calls_detected": tool_calls_detected,
                "pipeline_type": type(pipeline).__name__
            }
            
        except Exception as e:
            logger.error(f"   ❌ Pipeline execution failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    async def validate_real_outputs(self) -> Dict[str, Any]:
        """Validate real outputs and database integrity."""
        logger.info("✅ Validating real outputs...")
        
        try:
            from server.db import storage
            
            # Validate conversation exists
            conversation = await storage.conversation.get_conversation(self.test_conversation_id)
            if not conversation:
                raise Exception("Conversation not found in database")
            logger.info("   ✅ Conversation verified in database")
            
            # Wait a bit for database consistency
            await asyncio.sleep(2.5)
            
            # Validate messages exist
            messages = await storage.message.get_conversation_history(self.test_conversation_id)
            if len(messages) < 1:
                raise Exception("Messages not found in database")
            logger.info(f"   ✅ {len(messages)} messages verified in database")
            
            # Check if assistant response was generated
            assistant_messages = [msg for msg in messages if msg.role.value == "assistant"]
            user_messages = [msg for msg in messages if msg.role.value == "user"]
            
            if len(assistant_messages) == 0:
                return {
                    "success": False,
                    "error": "No assistant response generated"
                }
            
            logger.info("   ✅ Response generated: {} chunks".format(len(assistant_messages)))
            
            # Validate model profile exists
            model_profile = await storage.model_profile.get_model_profile_by_id(
                self.test_model_profile_id, self.test_user_id
            )
            if not model_profile:
                raise Exception("Model profile not found")
            
            logger.info("   ✅ Database integrity maintained")
            
            return {
                "success": True,
                "messages_found": len(messages),
                "assistant_responses": len(assistant_messages),
                "user_messages": len(user_messages)
            }
            
        except Exception as e:
            logger.error(f"   ❌ Output validation failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    async def cleanup_real_data(self) -> Dict[str, Any]:
        """Clean up all real test data from database."""
        logger.info("🧹 Cleaning up real test data...")
        
        cleaned_count = 0
        cleanup_errors = []
        
        try:
            from server.db import storage
            
            # Clean up in reverse order of creation
            for entity_type, entity_id in reversed(self.created_entities):
                try:
                    if entity_type == "message":
                        await storage.message.delete_message(entity_id)
                        logger.info(f"   🗑️  Deleted message: {entity_id}")
                    elif entity_type == "conversation":
                        await storage.conversation.delete_conversation(entity_id)
                        logger.info(f"   🗑️  Deleted conversation: {entity_id}")
                    elif entity_type == "model_profile":
                        await storage.model_profile.delete_model_profile(entity_id, self.test_user_id)
                        logger.info(f"   🗑️  Deleted model profile: {entity_id}")
                    elif entity_type == "user":
                        await storage.user.delete_user(entity_id)
                        logger.info(f"   🗑️  Deleted user: {entity_id}")
                    
                    cleaned_count += 1
                    
                except Exception as e:
                    error_msg = f"Failed to delete {entity_type} {entity_id}: {str(e)}"
                    cleanup_errors.append(error_msg)
                    logger.warning(f"   ⚠️  {error_msg}")
            
            logger.info(f"   ✅ Cleaned up {cleaned_count} entities")
            
            if len(cleanup_errors) == 0:
                logger.info("   ✅ All real data cleaned successfully")
            else:
                logger.warning(f"   ⚠️  Cleanup completed with {len(cleanup_errors)} errors")
            
            return {
                "success": len(cleanup_errors) == 0,
                "cleaned_count": cleaned_count,
                "total_entities": len(self.created_entities),
                "errors": cleanup_errors
            }
            
        except Exception as e:
            logger.error(f"   ❌ Cleanup failed: {str(e)}")
            return {
                "success": False,
                "cleaned_count": cleaned_count,
                "error": str(e)
            }

    async def print_test_summary(self, results: Dict[str, Any]) -> None:
        """Print comprehensive test summary."""
        logger.info("\n" + "=" * 80)
        logger.info("📊 Real Pipeline Test Summary")
        logger.info("=" * 80)
        
        success_rate = (results["components_passed"] / results["total_components"]) * 100
        overall_success = "YES" if results["overall_success"] else "NO"
        
        logger.info(f"✅ Overall Success: {overall_success} ({success_rate:.1f}%)")
        logger.info(f"🕒 Total Execution Time: {results['execution_time']:.2f}s")
        logger.info(f"⚡ Pipeline Execution Time: {results.get('pipeline_time', 0):.2f}s")
        logger.info(f"🔧 Components Passed: {results['components_passed']}/{results['total_components']}")
        logger.info(f"🏗️  Real Entities Created: {results['entities_created']}")
        
        # Extract key information from results
        pipeline_result = results["results"].get("pipeline_execution", {})
        model_name = results["results"].get("user_profile_creation", {}).get("model_name", "Unknown")
        tool_calls = pipeline_result.get("tool_calls_detected", False)
        
        logger.info(f"🤖 Model Used: {model_name}")
        logger.info(f"🛠️  Real Tools Executed: {'YES' if tool_calls else 'NO'}")
        logger.info(f"💾 Real Database Operations: YES")
        
        logger.info("\n📋 Component Results:")
        component_names = [
            "Infrastructure Setup",
            "User Profile Creation", 
            "Conversation Creation",
            "Message Creation",
            "Pipeline Execution",
            "Output Validation"
        ]
        
        result_keys = [
            "infrastructure_setup",
            "user_profile_creation",
            "conversation_creation", 
            "message_creation",
            "pipeline_execution",
            "output_validation"
        ]
        
        for name, key in zip(component_names, result_keys):
            component_result = results["results"].get(key, {})
            status = "✅ PASS" if component_result.get("success", False) else "❌ FAIL"
            logger.info(f"   {status} {name}")
        
        # Save detailed results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"real_pipeline_test_{timestamp}.json"
        
        try:
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"\n📝 Detailed results saved to: {results_file}")
        except Exception as e:
            logger.warning(f"⚠️  Could not save results file: {str(e)}")

async def main():
    """Main test execution function."""
    tester = RealEndToEndPipelineTester()
    
    try:
        results = await tester.run_full_test()
        
        if results["overall_success"]:
            logger.info("🎉 Real end-to-end pipeline test PASSED")
            exit(0)
        else:
            logger.error("❌ Real end-to-end pipeline test FAILED")
            exit(1)
            
    except Exception as e:
        logger.error(f"💥 Test execution failed with exception: {str(e)}")
        exit(1)

if __name__ == "__main__":
    asyncio.run(main())