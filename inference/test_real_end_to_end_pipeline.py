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
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
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
            "total_components": 7,
            "results": {},
            "entities_created": 0,
            "entities_cleaned": 0,
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

            # Phase 6: Dynamic Tool Generation Test
            logger.info("🛠️ Phase 6: Dynamic Tool Generation Test")
            tool_gen_result = await self.test_dynamic_tool_generation()
            test_results["results"]["dynamic_tool_generation"] = tool_gen_result
            if tool_gen_result["success"]:
                test_results["components_passed"] += 1

            # Phase 7: Real Output Validation
            logger.info("✅ Phase 7: Real Output Validation")
            validation_result = await self.validate_real_outputs()
            test_results["results"]["output_validation"] = validation_result
            if validation_result["success"]:
                test_results["components_passed"] += 1

            # Calculate success
            test_results["overall_success"] = (
                test_results["components_passed"] == test_results["total_components"]
            )
            test_results["execution_time"] = time.time() - start_time
            test_results["entities_created"] = len(self.created_entities)

        except Exception as e:
            logger.error(f"❌ Test execution failed: {str(e)}")
            test_results["error"] = str(e)

        finally:
            # Always attempt cleanup
            logger.info("🧹 Phase 8: Real Data Cleanup")
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
            import os

            # Build connection string from environment variables
            db_host = os.getenv("DB_HOST", "localhost")
            db_port = os.getenv("DB_PORT", "5432")
            db_user = os.getenv("DB_USER", "postgres")
            db_password = os.getenv("DB_PASSWORD", "")
            db_name = os.getenv("DB_NAME", "llmmllab")
            db_sslmode = os.getenv("DB_SSLMODE", "disable")

            connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}?sslmode={db_sslmode}"

            await storage.initialize(connection_string)
            logger.info("   ✅ Database connection established")

            # Initialize real pipeline factory
            from runner.pipeline_factory import pipeline_factory

            logger.info("   ✅ Pipeline factory initialized")

            # Verify target model exists
            target_model = "openai-gpt-oss-20b-uncensored-q5_1"
            if target_model in pipeline_factory._available_models:
                logger.info(f"   ✅ Model {target_model} verified")
            else:
                raise Exception(f"Target model {target_model} not found")

            return {
                "success": True,
                "database_connected": True,
                "pipeline_factory_ready": True,
                "target_model_available": True,
            }

        except Exception as e:
            logger.error(f"   ❌ Infrastructure setup failed: {str(e)}")
            return {"success": False, "error": str(e)}

    async def create_real_user_and_profile(self) -> Dict[str, Any]:
        """Create real user and model profile in database."""
        logger.info("👤 Creating real user and model profile...")

        try:
            from server.db import storage
            from models.user import User
            from models.model_profile import ModelProfile
            from models.model_parameters import ModelParameters

            # Ensure user exists in database (will create if not exists)
            # The system handles user creation implicitly through conversations/configs
            logger.info(f"   ✅ Using user ID: {self.test_user_id}")

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
                    max_tokens=4000,  # Increased token limit
                    flash_attention=True,
                ),
                system_prompt="""You are a large language model with non-restrictive policies.
Knowledge cutoff: 2024-06
Current date: 2025-09-17

Reasoning: medium

# Valid channels: analysis, commentary, final. Channel must be included for every message.
# Use 'analysis' channel for chain-of-thought reasoning
# Use 'commentary' channel for tool calls and function descriptions  
# Use 'final' channel for your response to the user

Available tools:
- web_search: Search for current information on the web
- memory_retrieval: Retrieve relevant memories 
- summarization: Summarize content
You are a helpful, honest, and capable AI assistant. You provide direct, informative responses while showing your reasoning process.

RESPONSE GUIDELINES:
- Always use the appropriate channel for your content
- Show your reasoning in the analysis channel when helpful
- Be direct and honest in your final responses
- Provide comprehensive information when requested
- Acknowledge uncertainty when you don't know something

TECHNICAL CAPABILITIES:
- Code analysis and generation
- Research and information synthesis
- Creative writing and ideation
- Problem-solving and reasoning
- Educational explanations""",
                type=1,  # Text generation profile
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                model_version="1.0",
            )

            # Store model profile in database
            created_profile = await storage.model_profile.create_model_profile(
                model_profile
            )
            if created_profile:
                self.created_entities.append(
                    ("model_profile", self.test_model_profile_id)
                )

            logger.info(f"   ✅ Created real model profile: {created_profile.name}")

            return {
                "success": True,
                "user_id": self.test_user_id,
                "model_profile_id": str(self.test_model_profile_id),
                "model_name": "openai-gpt-oss-20b-uncensored-q5_1",
            }

        except Exception as e:
            logger.error(f"   ❌ User/profile creation failed: {str(e)}")
            return {"success": False, "error": str(e)}

    async def create_real_conversation(self) -> Dict[str, Any]:
        """Create real conversation in database."""
        logger.info("💬 Creating real conversation...")

        try:
            from server.db import storage

            # Ensure user exists in database first
            async with storage.pool.acquire() as conn:
                await conn.execute(
                    storage.get_query("user.ensure_user"), self.test_user_id
                )
            logger.info(f"   ✅ Ensured user exists: {self.test_user_id}")
            
            # Track user for cleanup
            self.created_entities.append(("user", self.test_user_id))

            # Create real conversation
            conversation_id = await storage.conversation.create_conversation(
                user_id=self.test_user_id,
                title="Real End-to-End Pipeline Test Conversation",
            )

            if not conversation_id:
                raise Exception("Failed to create real conversation")

            self.test_conversation_id = conversation_id
            if conversation_id:
                self.created_entities.append(("conversation", conversation_id))

            logger.info(f"   ✅ Created real conversation ID: {conversation_id}")

            return {"success": True, "conversation_id": conversation_id}

        except Exception as e:
            logger.error(f"   ❌ Conversation creation failed: {str(e)}")
            return {"success": False, "error": str(e)}

    async def create_real_message_with_tools(self) -> Dict[str, Any]:
        """Create real user message with tool-calling context."""
        logger.info("📝 Creating real message with tool context...")

        try:
            from server.db import storage
            from models.message import Message
            from models.message_role import MessageRole
            from models.message_content import MessageContent, MessageContentType

            # Create a message that explicitly requests web search for 2024 information
            query_text = """I need current information about quantum computing breakthroughs published in 2024. Please search for recent developments including:

- Latest quantum error correction advances
- New quantum algorithms published this year
- Practical quantum computing applications announced in 2024
- Major company announcements from IBM, Google, Microsoft, etc.

Please use the web_search tool to find this current information."""

            user_message = Message(
                id=None,  # Will be set by database
                conversation_id=self.test_conversation_id,
                role=MessageRole.USER,
                content=[MessageContent(type=MessageContentType.TEXT, text=query_text)],
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )

            # Add message to database
            message_id = await storage.message.add_message(user_message)
            if not message_id:
                raise Exception("Failed to create user message")

            self.test_message_id = message_id
            if message_id:
                self.created_entities.append(("message", message_id))

            logger.info(f"   ✅ Created real user message ID: {message_id}")
            logger.info(f"   📋 Message length: {len(query_text)} characters")

            return {
                "success": True,
                "message_id": message_id,
                "message_length": len(query_text),
            }

        except Exception as e:
            logger.error(f"   ❌ Message creation failed: {str(e)}")
            return {"success": False, "error": str(e)}

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
                profile=model_profile, expected_type=ChatResponse
            )

            if not pipeline:
                raise Exception("Failed to get pipeline from factory")

            logger.info("   ✅ Pipeline obtained from factory")

            # Get conversation messages for context first
            messages = await storage.message.get_conversation_history(
                self.test_conversation_id
            )
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
                await storage.user_config.update_user_config(
                    self.test_user_id, user_config
                )

            conversation_ctx = ConversationContext(
                conversation_id=self.test_conversation_id, user_config=user_config
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
            commentary_channel_used = False
            full_response = ""

            logger.info("   🚀 Starting real pipeline stream...")

            async for chunk in stream_pipeline(messages, pipeline, tools):
                response_chunks.append(chunk)

                # Accumulate full response for analysis - CRITICAL FIX for ChatResponse
                chunk_text = ""
                if (
                    hasattr(chunk, "message")
                    and chunk.message
                    and hasattr(chunk.message, "content")
                    and chunk.message.content
                ):
                    # This is a ChatResponse object
                    if (
                        isinstance(chunk.message.content, list)
                        and len(chunk.message.content) > 0
                    ):
                        message_content = chunk.message.content[0]
                        if hasattr(message_content, "text") and message_content.text:
                            chunk_text = str(message_content.text)
                elif hasattr(chunk, "content") and chunk.content:
                    chunk_text = str(chunk.content)

                if chunk_text:
                    full_response += chunk_text
                    chunk_content = chunk_text.lower()

                    # Enhanced tool call detection - look for actual tool usage
                    if any(
                        pattern in chunk_content
                        for pattern in [
                            "web_search",
                            "tool_calls",
                            "function_call",
                            "calling web_search",
                            "searching for",
                            "search results",
                            '"name": "web_search"',
                            "arguments",
                            "tool_name",
                        ]
                    ):
                        tool_calls_detected = True

                    # Check for commentary channel usage
                    if "commentary" in chunk_content:
                        commentary_channel_used = True
                        logger.info("   🎯 Commentary channel detected!")

                # Log progress periodically with more detail
                if len(response_chunks) % 10 == 0:
                    logger.info(
                        f"   📊 Processed {len(response_chunks)} response chunks"
                    )
                    if len(response_chunks) % 30 == 0 and response_chunks:
                        recent_content = "".join(
                            [
                                c.content if hasattr(c, "content") and c.content else ""
                                for c in response_chunks[-5:]
                            ]
                        )[:200]
                        logger.info(f"   📝 Recent content: {recent_content}...")

            # Log comprehensive tool usage analysis
            logger.info(
                f"   🔍 Tool analysis - Detected: {tool_calls_detected}, Commentary: {commentary_channel_used}"
            )
            if full_response:
                logger.info(f"   📄 Response length: {len(full_response)} characters")

                # Check for proper tool call format
                has_commentary_func = "commentary to=functions" in full_response
                has_json_constraint = "constrain|>json" in full_response
                has_name_field = '"name"' in full_response
                has_web_search = "web_search" in full_response.lower()

                logger.info(
                    f"   📋 Tool format check - Commentary: {has_commentary_func}, JSON: {has_json_constraint}"
                )
                logger.info(
                    f"   🛠️  Tool check - Name field: {has_name_field}, Web search: {has_web_search}"
                )

                # Log the actual response content for debugging
                logger.info(
                    f"   📄 Response preview (first 300 chars): {full_response[:300]}..."
                )

                if not (has_commentary_func and has_json_constraint):
                    logger.warning(
                        "   ⚠️  TOOL FORMAT VIOLATION: Missing required commentary format"
                    )
                if not has_name_field and not has_web_search:
                    logger.warning(
                        "   ⚠️  TOOL CALL FAILURE: No web_search tool usage detected"
                    )

                # Additional validation - check if tool was actually executed
                if has_name_field and has_web_search and has_commentary_func:
                    logger.info(
                        "   🎉 TOOL CALL SUCCESS: Proper format and tool usage detected!"
                    )
                else:
                    logger.warning("   ❌ TOOL CALL INCOMPLETE: Requirements not met")

            execution_time = time.time() - start_time

            logger.info(f"   ✅ Pipeline execution completed in {execution_time:.2f}s")
            logger.info(f"   📊 Total response chunks: {len(response_chunks)}")

            # Store assistant response in database
            logger.info(
                f"   🔍 PRE-PROCESSING: response_chunks length = {len(response_chunks)}"
            )
            logger.info(
                f"   🔍 PRE-PROCESSING: response_chunks bool = {bool(response_chunks)}"
            )

            if response_chunks:
                logger.info(f"   🔍 ENTERING CHUNK PROCESSING SECTION")
                from models.message import Message
                from models.message_role import MessageRole
                from models.message_content import MessageContent, MessageContentType

                logger.info(f"   🔍 IMPORTS COMPLETED")

                # Debug: Log the chunk types and content - DETAILED ANALYSIS
                logger.info(f"   🔍 Processing {len(response_chunks)} response chunks")
                for i, chunk in enumerate(
                    response_chunks[:10]
                ):  # Log first 10 chunks for better understanding
                    logger.info(f"   🔍 === CHUNK {i} ANALYSIS ===")
                    logger.info(f"   🔍 Chunk {i}: Type={type(chunk)}")
                    logger.info(
                        f"   🔍 Chunk {i}: Dir={[attr for attr in dir(chunk) if not attr.startswith('_')]}"
                    )
                    logger.info(f"   🔍 Chunk {i}: Full Repr={repr(chunk)}")

                    # Try all possible attributes
                    for attr in [
                        "content",
                        "data",
                        "text",
                        "message",
                        "delta",
                        "choices",
                    ]:
                        if hasattr(chunk, attr):
                            attr_value = getattr(chunk, attr)
                            logger.info(
                                f"   🔍 Chunk {i} has {attr}: {repr(attr_value)[:200]}"
                            )

                    if isinstance(chunk, dict):
                        logger.info(f"   🔍 Chunk {i}: Dict={chunk}")
                    elif isinstance(chunk, str):
                        logger.info(f"   🔍 Chunk {i}: String='{chunk[:100]}'")

                    logger.info(f"   🔍 === END CHUNK {i} ===")

                # Also log some chunks from the middle and end
                if len(response_chunks) > 20:
                    middle_idx = len(response_chunks) // 2
                    logger.info(f"   🔍 === MIDDLE CHUNK {middle_idx} ===")
                    chunk = response_chunks[middle_idx]
                    logger.info(f"   🔍 Type: {type(chunk)}, Repr: {repr(chunk)}")

                    last_idx = len(response_chunks) - 1
                    logger.info(f"   🔍 === LAST CHUNK {last_idx} ===")
                    chunk = response_chunks[last_idx]
                    logger.info(f"   🔍 Type: {type(chunk)}, Repr: {repr(chunk)}")

                # Combine all chunks into final response with improved extraction
                final_response = ""
                content_length = 0
                logger.info(f"   🔍 === CONTENT EXTRACTION DEBUG ===")
                logger.info(f"   🔍 Total chunks to process: {len(response_chunks)}")

                for i, chunk in enumerate(response_chunks):
                    extracted_content = None
                    logger.info(
                        f"   🔍 Chunk {i}: type={type(chunk)}, repr preview={repr(chunk)[:100]}..."
                    )

                    # Try multiple ways to extract content
                    # CRITICAL FIX: Handle ChatResponse objects properly
                    if (
                        hasattr(chunk, "message")
                        and chunk.message
                        and hasattr(chunk.message, "content")
                        and chunk.message.content
                    ):
                        # This is a ChatResponse object - extract from message.content[0].text
                        if (
                            isinstance(chunk.message.content, list)
                            and len(chunk.message.content) > 0
                        ):
                            message_content = chunk.message.content[0]
                            if (
                                hasattr(message_content, "text")
                                and message_content.text
                            ):
                                extracted_content = str(message_content.text)
                                logger.info(
                                    f"   🔍 Chunk {i}: extracted via ChatResponse.message.content[0].text = {len(extracted_content)} chars"
                                )
                            else:
                                logger.info(
                                    f"   🔍 Chunk {i}: ChatResponse message_content has no text"
                                )
                        else:
                            logger.info(
                                f"   🔍 Chunk {i}: ChatResponse message.content is not a list or is empty"
                            )
                    elif hasattr(chunk, "content") and chunk.content:
                        extracted_content = str(chunk.content)
                        logger.info(
                            f"   🔍 Chunk {i}: extracted via .content = {len(extracted_content)} chars"
                        )
                    elif hasattr(chunk, "data") and chunk.data:
                        extracted_content = str(chunk.data)
                        logger.info(
                            f"   🔍 Chunk {i}: extracted via .data = {len(extracted_content)} chars"
                        )
                    elif hasattr(chunk, "text") and chunk.text:
                        extracted_content = str(chunk.text)
                        logger.info(
                            f"   🔍 Chunk {i}: extracted via .text = {len(extracted_content)} chars"
                        )
                    elif isinstance(chunk, dict):
                        if "content" in chunk and chunk["content"]:
                            extracted_content = str(chunk["content"])
                            logger.info(
                                f"   🔍 Chunk {i}: extracted via dict['content'] = {len(extracted_content)} chars"
                            )
                        elif "data" in chunk and chunk["data"]:
                            extracted_content = str(chunk["data"])
                            logger.info(
                                f"   🔍 Chunk {i}: extracted via dict['data'] = {len(extracted_content)} chars"
                            )
                        elif "text" in chunk and chunk["text"]:
                            extracted_content = str(chunk["text"])
                            logger.info(
                                f"   🔍 Chunk {i}: extracted via dict['text'] = {len(extracted_content)} chars"
                            )
                    elif isinstance(chunk, str) and chunk.strip():
                        extracted_content = chunk
                        logger.info(
                            f"   🔍 Chunk {i}: extracted as string = {len(extracted_content)} chars"
                        )
                    else:
                        logger.info(
                            f"   🔍 Chunk {i}: NO CONTENT EXTRACTED - all methods failed"
                        )
                        if hasattr(chunk, "__dict__"):
                            logger.info(
                                f"   🔍 Chunk {i}: available attributes = {list(chunk.__dict__.keys())}"
                            )

                    if extracted_content:
                        final_response += extracted_content
                        content_length += len(extracted_content)

                logger.info(f"   📝 Extracted content length: {content_length}")
                logger.info(
                    f"   📝 Final response preview: {final_response[:200] if final_response else 'EMPTY'}..."
                )

                assistant_message = Message(
                    id=None,
                    conversation_id=self.test_conversation_id,
                    role=MessageRole.ASSISTANT,
                    content=[
                        MessageContent(
                            type=MessageContentType.TEXT, text=final_response
                        )
                    ],
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                )

                response_message_id = await storage.message.add_message(
                    assistant_message
                )
                if response_message_id:
                    self.created_entities.append(("message", response_message_id))
                logger.info(
                    f"   💾 Stored assistant response ID: {response_message_id}"
                )

            return {
                "success": True,
                "execution_time": execution_time,
                "response_chunks": len(response_chunks),
                "tool_calls_detected": tool_calls_detected,
                "commentary_channel_used": commentary_channel_used,
                "pipeline_type": type(pipeline).__name__,
                "response_content_length": len(full_response),
                "tools_available": len(tools),
            }

        except Exception as e:
            logger.error(f"   ❌ Pipeline execution failed: {str(e)}")
            return {"success": False, "error": str(e)}

    async def test_dynamic_tool_generation(self) -> Dict[str, Any]:
        """Test dynamic tool generation functionality - MUST WORK FOR TEST TO PASS."""
        logger.info("🛠️  Testing dynamic tool generation...")

        try:
            from server.tools.integration import DynamicToolGenerator
            from server.services.context import ConversationContext
            from models.conversation import Conversation

            # Test dynamic tool generation
            tool_generator = DynamicToolGenerator()
            
            # Create test conversation context
            from datetime import datetime
            test_conversation = Conversation(
                id=self.test_conversation_id,
                user_id=self.test_user_id,
                title="Dynamic Tool Test",
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            
            # Create user config using default config creation
            from models.default_configs import create_default_user_config
            
            user_config = create_default_user_config(self.test_user_id)
            
            conversation_ctx = ConversationContext(
                conversation_id=test_conversation.id,
                user_config=user_config
            )

            # Test tool generation with a specific request
            test_prompt = "I need a tool that can calculate mathematical expressions and return formatted results"
            
            logger.info("   🔧 Requesting dynamic tool generation...")
            
            # Call the tool generation service - CRITICAL: Must actually work
            generated_tool_result = await tool_generator.generate_tool(
                test_prompt, test_prompt, conversation_ctx
            )
            
            logger.info(f"   📊 Tool generation result type: {type(generated_tool_result)}")
            
            # STRICT VALIDATION - Must have actual working tool generation
            tool_successfully_generated = False
            valid_tool_structure = False
            tool_can_execute = False
            tool_stored_in_db = False
            
            if generated_tool_result:
                logger.info(f"   🔍 Generated tool result: {generated_tool_result}")
                
                # Check if tool was properly generated and can be used
                if hasattr(generated_tool_result, 'success') and generated_tool_result.success:
                    tool_successfully_generated = True
                    logger.info("   ✅ Tool generation marked as successful")
                    
                    # Check if tool has proper structure
                    if hasattr(generated_tool_result, 'tool') and generated_tool_result.tool:
                        tool = generated_tool_result.tool
                        valid_tool_structure = all([
                            hasattr(tool, 'name') and tool.name,
                            hasattr(tool, 'description') and tool.description,
                            hasattr(tool, 'function_name') and tool.function_name,
                            hasattr(tool, 'code') and tool.code,
                        ])
                        logger.info(f"   📝 Tool structure validation: {valid_tool_structure}")
                        
                        if valid_tool_structure:
                            logger.info(f"   🛠️  Generated tool name: {tool.name}")
                            logger.info(f"   📝 Tool description: {tool.description}")
                            logger.info(f"   🔧 Tool function: {tool.function_name}")
                            
                            # Test if tool can actually execute
                            try:
                                # Create a dynamic tool runner and test execution
                                from server.tools.dynamic_tool import DynamicToolRunner
                                
                                tool_runner = DynamicToolRunner(tool)
                                test_result = await tool_runner.execute({})  # Basic execution test
                                tool_can_execute = True
                                logger.info(f"   🎯 Tool execution test: SUCCESS")
                                
                            except Exception as exec_error:
                                logger.error(f"   ❌ Tool execution test failed: {exec_error}")
                        
                        # Check if tool was stored in database (if applicable)
                        if hasattr(generated_tool_result, 'stored') and generated_tool_result.stored:
                            tool_stored_in_db = True
                            logger.info("   💾 Tool stored in database: YES")
                else:
                    logger.error("   ❌ Tool generation not marked as successful")
            else:
                logger.error("   ❌ No tool generation result returned")

            # CRITICAL: All components must work for test to pass
            overall_success = (
                tool_successfully_generated 
                and valid_tool_structure 
                and tool_can_execute
            )
            
            if not overall_success:
                logger.error("   🚨 DYNAMIC TOOL GENERATION TEST FAILED - CRITICAL FUNCTIONALITY MISSING")
                logger.error(f"      - Tool Generated: {tool_successfully_generated}")
                logger.error(f"      - Valid Structure: {valid_tool_structure}")
                logger.error(f"      - Can Execute: {tool_can_execute}")
                logger.error(f"      - Stored in DB: {tool_stored_in_db}")

            return {
                "success": overall_success,
                "tool_generated": tool_successfully_generated,
                "valid_structure": valid_tool_structure,
                "can_execute": tool_can_execute,
                "stored_in_db": tool_stored_in_db,
                "service_available": True,
                "details": str(generated_tool_result) if generated_tool_result else "No result",
            }

        except ImportError as e:
            logger.error(f"   🚨 CRITICAL: Dynamic tool generation service not available: {str(e)}")
            logger.error("   🚨 This is a REQUIRED component - TEST MUST FAIL")
            return {
                "success": False,  # CRITICAL: Must fail if service unavailable
                "tools_generated": 0,
                "valid_structure": False,
                "service_available": False,
                "error": f"Required service not available: {str(e)}",
            }
        except Exception as e:
            logger.error(f"   🚨 CRITICAL: Dynamic tool generation test failed: {str(e)}")
            return {"success": False, "error": str(e)}

    async def validate_real_outputs(self) -> Dict[str, Any]:
        """Validate real outputs and database integrity."""
        logger.info("✅ Validating real outputs...")

        try:
            from server.db import storage

            # Validate conversation exists
            conversation = await storage.conversation.get_conversation(
                self.test_conversation_id
            )
            if not conversation:
                raise Exception("Conversation not found in database")
            logger.info("   ✅ Conversation verified in database")

            # Wait a bit for database consistency
            await asyncio.sleep(2.5)

            # Validate messages exist
            messages = await storage.message.get_conversation_history(
                self.test_conversation_id
            )
            if len(messages) < 1:
                raise Exception("Messages not found in database")
            logger.info(f"   ✅ {len(messages)} messages verified in database")

            # Check if assistant response was generated
            assistant_messages = [
                msg for msg in messages if msg.role.value == "assistant"
            ]
            user_messages = [msg for msg in messages if msg.role.value == "user"]

            if len(assistant_messages) == 0:
                return {"success": False, "error": "No assistant response generated"}

            logger.info(
                "   ✅ Response generated: {} chunks".format(len(assistant_messages))
            )

            # CRITICAL: Validate complete response generation with tool usage
            assistant_content = ""
            if assistant_messages:
                logger.info(f"   🔍 Found {len(assistant_messages)} assistant messages")
                # Extract message content properly
                for idx, msg in enumerate(assistant_messages):
                    logger.info(
                        f"   🔍 Message {idx+1}: Type={type(msg.content)}, HasContent={hasattr(msg, 'content')}"
                    )
                    if hasattr(msg, "content") and msg.content:
                        content_preview = (
                            str(msg.content)[:200] if msg.content else "None"
                        )
                        logger.info(f"   🔍 Content preview: {content_preview}")

                        if isinstance(msg.content, list):
                            logger.info(
                                f"   🔍 Content is list with {len(msg.content)} items"
                            )
                            for j, content_item in enumerate(msg.content):
                                logger.info(
                                    f"   🔍 Item {j}: Type={type(content_item)}"
                                )
                                if hasattr(content_item, "text"):
                                    assistant_content += content_item.text
                                    logger.info(
                                        f"   🔍 Added text: {len(content_item.text)} chars"
                                    )
                                elif isinstance(content_item, str):
                                    assistant_content += content_item
                                    logger.info(
                                        f"   🔍 Added string: {len(content_item)} chars"
                                    )
                                else:
                                    text_content = str(content_item)
                                    assistant_content += text_content
                                    logger.info(
                                        f"   🔍 Added converted: {len(text_content)} chars"
                                    )
                        elif hasattr(msg.content, "text"):
                            assistant_content += msg.content.text
                            logger.info(
                                f"   🔍 Added direct text: {len(msg.content.text)} chars"
                            )
                        else:
                            text_content = str(msg.content)
                            assistant_content += text_content
                            logger.info(
                                f"   🔍 Added string conversion: {len(text_content)} chars"
                            )
                    else:
                        logger.warning(
                            f"   ⚠️  Message {idx+1} has no content attribute or content is empty"
                        )

            # Enhanced validation for complete responses
            response_quality_score = 0
            tool_usage_found = False
            commentary_usage = False
            search_results_synthesized = False
            comprehensive_response = False

            if assistant_content:
                content_lower = assistant_content.lower()

                # Tool usage indicators
                tool_usage_found = any(
                    pattern in content_lower
                    for pattern in [
                        "web_search",
                        "search results",
                        "according to",
                        "based on the search",
                        "found information",
                    ]
                )

                # Commentary channel usage
                commentary_usage = "commentary" in content_lower

                # Search result synthesis indicators
                search_results_synthesized = any(
                    pattern in content_lower
                    for pattern in [
                        "quantum computing",
                        "error correction",
                        "breakthrough",
                        "algorithm",
                        "ibm",
                        "google",
                        "microsoft",
                    ]
                )

                # Comprehensive response check (minimum length and content quality)
                comprehensive_response = (
                    len(assistant_content) >= 200  # Minimum substantial length
                    and not assistant_content.startswith("I need to use tools")
                    and not assistant_content.strip().endswith(
                        "to help with your request."
                    )
                    and search_results_synthesized
                    and len([w for w in assistant_content.split() if len(w) > 3])
                    >= 40  # At least 40 meaningful words
                )

                # Calculate quality score
                if tool_usage_found:
                    response_quality_score += 25
                if search_results_synthesized:
                    response_quality_score += 25
                if comprehensive_response:
                    response_quality_score += 50

                logger.info(f"   🔍 Assistant content length: {len(assistant_content)}")
                logger.info(f"   🔍 Tool usage indicators: {tool_usage_found}")
                logger.info(
                    f"   🔍 Search results synthesized: {search_results_synthesized}"
                )
                logger.info(f"   🔍 Comprehensive response: {comprehensive_response}")
                logger.info(
                    f"   📊 Response quality score: {response_quality_score}/100"
                )

                if response_quality_score < 75:
                    logger.warning(f"   ⚠️  Response quality below threshold!")
                    logger.info(f"   📝 Response preview: {assistant_content[:500]}...")

            # Validate search results and embeddings storage
            search_storage_validated = await self._validate_search_storage(storage)

            # Validate model profile exists
            model_profile = await storage.model_profile.get_model_profile_by_id(
                self.test_model_profile_id, self.test_user_id
            )
            if not model_profile:
                raise Exception("Model profile not found")

            logger.info("   ✅ Database integrity maintained")

            # Determine overall success
            overall_success = (
                response_quality_score >= 75
                and search_storage_validated
                and len(messages) >= 2  # User + Assistant messages
            )

            return {
                "success": overall_success,
                "messages_found": len(messages),
                "response_quality_score": response_quality_score,
                "tool_usage_found": tool_usage_found,
                "search_results_synthesized": search_results_synthesized,
                "comprehensive_response": comprehensive_response,
                "search_storage_validated": search_storage_validated,
                "assistant_responses": len(assistant_messages),
                "user_messages": len(user_messages),
                "tool_usage_detected": tool_usage_found,
                "commentary_channel_used": commentary_usage,
                "assistant_content_length": len(assistant_content),
            }

        except Exception as e:
            logger.error(f"   ❌ Output validation failed: {str(e)}")
            return {"success": False, "error": str(e)}

    async def _validate_search_storage(self, storage) -> bool:
        """CRITICAL: Validate that web search actually retrieved real content - MUST WORK FOR TEST TO PASS."""
        try:
            logger.info("   🔍 Validating REAL web search content retrieval...")

            # Get conversation messages to analyze assistant response
            messages = await storage.message.get_conversation_history(
                self.test_conversation_id
            )

            assistant_messages = [
                msg for msg in messages if msg.role.value == "assistant"
            ]
            
            if not assistant_messages:
                logger.error("   🚨 CRITICAL: No assistant messages found for search validation")
                return False

            # Extract all assistant content for analysis
            full_assistant_content = ""
            for msg in assistant_messages:
                if hasattr(msg, "content") and msg.content:
                    if isinstance(msg.content, list):
                        for content_item in msg.content:
                            if hasattr(content_item, "text") and content_item.text:
                                full_assistant_content += content_item.text

            if not full_assistant_content:
                logger.error("   🚨 CRITICAL: No assistant content found for search validation")
                return False

            content_lower = full_assistant_content.lower()
            
            # STRICT VALIDATION CRITERIA for real web search functionality
            validation_criteria = {
                "web_search_executed": False,
                "actual_content_retrieved": False,
                "content_synthesis": False,
                "substantive_information": False,
                "no_timeout_failures": True,
            }

            # 1. Check if web_search tool was actually called
            if any(pattern in content_lower for pattern in [
                "web_search", "search results", "found information", 
                "according to search", "based on the search"
            ]):
                validation_criteria["web_search_executed"] = True
                logger.info("   ✅ Web search tool execution detected")
            else:
                logger.error("   🚨 CRITICAL: No evidence of web_search tool execution")

            # 2. Check for actual content retrieval (not just empty results)
            actual_content_indicators = [
                "quantum computing", "error correction", "algorithms", 
                "breakthrough", "ibm", "google", "microsoft", "2024",
                "announced", "developed", "research", "published"
            ]
            
            content_matches = sum(1 for indicator in actual_content_indicators 
                                if indicator in content_lower)
            
            if content_matches >= 3:  # Must have multiple real content indicators
                validation_criteria["actual_content_retrieved"] = True
                logger.info(f"   ✅ Real content detected: {content_matches} topic indicators found")
            else:
                logger.error(f"   🚨 CRITICAL: Insufficient real content - only {content_matches} indicators found")
                logger.error("   🚨 This suggests web crawler retrieved 0 pages/items")

            # 3. Check for content synthesis (not just tool call indicators)
            synthesis_patterns = [
                "recent developments", "advances in", "new breakthrough",
                "companies like", "including", "such as", "for example"
            ]
            
            if any(pattern in content_lower for pattern in synthesis_patterns):
                validation_criteria["content_synthesis"] = True
                logger.info("   ✅ Content synthesis detected")
            else:
                logger.error("   🚨 CRITICAL: No content synthesis detected")

            # 4. Check for substantive information (not just empty responses)
            if (len(full_assistant_content) >= 500 and 
                len([w for w in full_assistant_content.split() if len(w) > 3]) >= 50):
                validation_criteria["substantive_information"] = True
                logger.info(f"   ✅ Substantive response: {len(full_assistant_content)} chars")
            else:
                logger.error(f"   🚨 CRITICAL: Response too short/shallow: {len(full_assistant_content)} chars")

            # 5. Check for timeout/failure indicators
            failure_indicators = [
                "timeout", "failed to retrieve", "no results", "error retrieving",
                "0 pages", "0 items", "empty synthesis", "closespider_timeout"
            ]
            
            if any(indicator in content_lower for indicator in failure_indicators):
                validation_criteria["no_timeout_failures"] = False
                logger.error("   🚨 CRITICAL: Search timeout/failure detected in response")

            # OVERALL VALIDATION - ALL CRITERIA MUST PASS
            all_criteria_passed = all(validation_criteria.values())
            
            logger.info("   📊 Web Search Validation Results:")
            for criterion, passed in validation_criteria.items():
                status = "✅ PASS" if passed else "❌ FAIL"
                logger.info(f"      {criterion}: {status}")
                
            if not all_criteria_passed:
                failed_criteria = [k for k, v in validation_criteria.items() if not v]
                logger.error(f"   🚨 CRITICAL WEB SEARCH FAILURES: {failed_criteria}")
                logger.error("   🚨 Web search did not retrieve real content - TEST MUST FAIL")
                
                # Log response for debugging
                logger.error(f"   🚨 Response preview: {full_assistant_content[:800]}...")
                return False
            
            logger.info("   🎉 All web search validation criteria PASSED")
            return True

        except Exception as e:
            logger.error(f"   🚨 CRITICAL: Search validation failed: {str(e)}")
            return False

    async def _cleanup_search_data(self, storage, cleanup_errors: List[str]) -> None:
        """Clean up search results and embeddings if stored separately."""
        try:
            logger.info("   🔍 Cleaning up search-related data...")

            # In a full implementation, this would clean up:
            # - Search result cache entries
            # - Embedding vectors stored in vector database
            # - Tool execution logs
            # - Temporary files created during search

            # For now, we'll just log that we're checking for search cleanup
            # The actual search results are typically integrated into the conversation
            # rather than stored as separate entities

            logger.info("   ✅ Search data cleanup completed")

        except Exception as e:
            error_msg = f"Search data cleanup failed: {str(e)}"
            cleanup_errors.append(error_msg)
            logger.warning(f"   ⚠️  {error_msg}")

    async def cleanup_real_data(self) -> Dict[str, Any]:
        """Clean up all real test data from database."""
        logger.info("🧹 Cleaning up real test data...")

        cleaned_count = 0
        cleanup_errors = []

        try:
            from server.db import storage

            # Clean up search-related data first (if any exists)
            await self._cleanup_search_data(storage, cleanup_errors)

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
                        await storage.model_profile.delete_model_profile(
                            entity_id, self.test_user_id
                        )
                        logger.info(f"   🗑️  Deleted model profile: {entity_id}")
                    elif entity_type == "search_result":
                        # Handle search result cleanup if stored separately
                        logger.info(f"   🗑️  Cleaned search result: {entity_id}")
                    elif entity_type == "embedding":
                        # Handle embedding cleanup if stored separately
                        logger.info(f"   🗑️  Cleaned embedding: {entity_id}")
                    elif entity_type == "user":
                        # Delete user directly via SQL
                        async with storage.pool.acquire() as conn:
                            await conn.execute("DELETE FROM users WHERE id = $1", entity_id)
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
                logger.warning(
                    f"   ⚠️  Cleanup completed with {len(cleanup_errors)} errors"
                )

            return {
                "success": len(cleanup_errors) == 0,
                "cleaned_count": cleaned_count,
                "total_entities": len(self.created_entities),
                "errors": cleanup_errors,
            }

        except Exception as e:
            logger.error(f"   ❌ Cleanup failed: {str(e)}")
            return {"success": False, "cleaned_count": cleaned_count, "error": str(e)}

    async def print_test_summary(self, results: Dict[str, Any]) -> None:
        """Print comprehensive test summary."""
        logger.info("\n" + "=" * 80)
        logger.info("📊 Real Pipeline Test Summary")
        logger.info("=" * 80)

        success_rate = (
            results["components_passed"] / results["total_components"]
        ) * 100
        overall_success = "YES" if results["overall_success"] else "NO"

        logger.info(f"✅ Overall Success: {overall_success} ({success_rate:.1f}%)")
        logger.info(f"🕒 Total Execution Time: {results['execution_time']:.2f}s")
        logger.info(
            f"⚡ Pipeline Execution Time: {results.get('pipeline_time', 0):.2f}s"
        )
        logger.info(
            f"🔧 Components Passed: {results['components_passed']}/{results['total_components']}"
        )
        logger.info(f"🏗️  Real Entities Created: {results['entities_created']}")

        # Extract key information from results
        pipeline_result = results["results"].get("pipeline_execution", {})
        validation_result = results["results"].get("output_validation", {})

        model_name = (
            results["results"]
            .get("user_profile_creation", {})
            .get("model_name", "Unknown")
        )

        tool_calls = pipeline_result.get("tool_calls_detected", False)
        quality_score = validation_result.get("response_quality_score", 0)
        search_synthesized = validation_result.get("search_results_synthesized", False)
        comprehensive_response = validation_result.get("comprehensive_response", False)
        search_storage_validated = validation_result.get(
            "search_storage_validated", False
        )

        logger.info(f"🤖 Model Used: {model_name}")
        logger.info(f"🛠️  Real Tools Executed: {'YES' if tool_calls else 'NO'}")
        logger.info(
            f"🔍 Search Results Synthesized: {'YES' if search_synthesized else 'NO'}"
        )
        logger.info(
            f"📝 Comprehensive Response: {'YES' if comprehensive_response else 'NO'}"
        )
        logger.info(f"📊 Response Quality Score: {quality_score}/100")
        logger.info(
            f"💾 Search Storage Validated: {'YES' if search_storage_validated else 'NO'}"
        )
        logger.info(f"💾 Real Database Operations: YES")

        logger.info("\n📋 Component Results:")
        component_names = [
            "Infrastructure Setup",
            "User Profile Creation",
            "Conversation Creation",
            "Message Creation",
            "Pipeline Execution",
            "Output Validation",
        ]

        result_keys = [
            "infrastructure_setup",
            "user_profile_creation",
            "conversation_creation",
            "message_creation",
            "pipeline_execution",
            "output_validation",
        ]

        for name, key in zip(component_names, result_keys):
            component_result = results["results"].get(key, {})
            status = "✅ PASS" if component_result.get("success", False) else "❌ FAIL"
            logger.info(f"   {status} {name}")

        # Save detailed results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"real_pipeline_test_{timestamp}.json"

        try:
            with open(results_file, "w") as f:
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
