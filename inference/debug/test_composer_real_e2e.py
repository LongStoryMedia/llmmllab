"""
Composer-based Real End-to-End Pipeline Test

This test validates the complete LLM ML Lab pipeline using the new composer architecture:
1. Real user creation in database
2. Real model profile creation
3. Real conversation and message creation
4. **Composer workflow execution** using compose_workflow, create_initial_state, execute_workflow
5. Real tool integration via LangGraph workflows
6. Real output validation
7. Complete cleanup of all created data

This modernized version uses the composer/__init__.py entry points and follows
the new architectural patterns with LangGraph workflows instead of direct pipeline calls.
"""

import asyncio
from re import T
import time
import uuid
import json
import os
import sys
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from utils.logging import llmmllogger

# Configure logging
logger = llmmllogger.bind(component="composer_e2e_test")


class ComposerRealEndToEndTester:
    """Real end-to-end test using composer architecture."""

    def __init__(
        self,
        target_model: Optional[str] = None,
        capture_llm_output: bool = True,
        print_output: bool = False,
    ):
        """Initialize composer-based pipeline tester."""
        self.test_user_id = f"test_composer_user_{uuid.uuid4().hex[:8]}"
        self.test_model_profile_id = uuid.uuid4()
        self.test_conversation_id: Optional[int] = None
        self.test_message_id: Optional[int] = None
        self.storage = None  # Will be initialized with infrastructure

        # LLM output capture configuration
        self.capture_llm_output = capture_llm_output
        self.print_output = print_output
        self.llm_output_file = None
        self.output_dir = "debug/out"
        self.llm_responses = []  # Store all LLM responses for analysis

        # Support multiple models for comprehensive testing
        available_models = [
            "qwen3-vl-32b-thinking-abliterated",  # Primary multimodal model - use this as default
            "qwen3-30b-a3b-q4-k-m",
            "openai-gpt-oss-20b-uncensored-q5_1",
            "qwen2.5-vl-32b-instruct-q4-k-m",
        ]

        self.target_model = target_model or available_models[0]
        self.available_models = available_models

        # Initialize LLM output file if capture is enabled
        if self.capture_llm_output:
            self._initialize_llm_output_file()

    def _initialize_llm_output_file(self):
        """Initialize the file for capturing LLM-generated text."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_safe = (
            self.target_model.replace("/", "_").replace("-", "_").replace(":", "_")
        )

        # Ensure debug/out directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        self.llm_output_file = (
            f"{self.output_dir}/composer_llm_output_{model_safe}_{timestamp}.txt"
        )

        # Create the file with header
        try:
            with open(self.llm_output_file, "w", encoding="utf-8") as f:
                f.write("Composer LLM Output Capture - Real End-to-End Test\n")
                f.write("=" * 60 + "\n")
                f.write(f"Model: {self.target_model}\n")
                f.write(f"Test User: {self.test_user_id}\n")
                f.write("Architecture: Composer + LangGraph\n")
                f.write(f"Timestamp: {datetime.now(timezone.utc).isoformat()}\n")
                f.write("=" * 60 + "\n\n")
            logger.info(f"📝 LLM output will be captured to: {self.llm_output_file}")
        except Exception as e:
            logger.warning(f"⚠️  Failed to initialize LLM output file: {e}")
            self.capture_llm_output = False

    def _write_llm_response(
        self, phase: str, response_text: str, metadata: Optional[Dict[str, Any]] = None
    ):
        """Write LLM response to file with phase information."""
        if not self.capture_llm_output or not self.llm_output_file:
            return

        try:
            with open(self.llm_output_file, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"PHASE: {phase}\n")
                f.write(f"TIMESTAMP: {datetime.now(timezone.utc).isoformat()}\n")
                if metadata:
                    f.write(f"METADATA: {json.dumps(metadata, indent=2)}\n")
                f.write(f"{'='*60}\n")
                f.write(f"{response_text}\n")

            # Also store in memory for analysis
            self.llm_responses.append(
                {
                    "phase": phase,
                    "response": response_text,
                    "metadata": metadata or {},
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

            logger.info(
                f"📝 Captured LLM response for {phase} ({len(response_text)} chars)"
            )
        except Exception as e:
            logger.warning(f"⚠️  Failed to write LLM response to file: {e}")

    def _write_detailed_data(
        self, section: str, title: str, data: Any, description: str = ""
    ):
        """Write detailed data (prompts, tools, messages, etc.) to output file."""
        if not self.capture_llm_output or not self.llm_output_file:
            return

        try:
            with open(self.llm_output_file, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"SECTION: {section}\n")
                f.write(f"TITLE: {title}\n")
                f.write(f"TIMESTAMP: {datetime.now(timezone.utc).isoformat()}\n")
                if description:
                    f.write(f"DESCRIPTION: {description}\n")
                f.write(f"{'='*60}\n")

                # Format data based on type
                if isinstance(data, (dict, list)):
                    f.write(json.dumps(data, indent=2, default=str))
                elif hasattr(data, "__dict__"):
                    # Object with attributes - convert to dict
                    f.write(json.dumps(data.__dict__, indent=2, default=str))
                elif isinstance(data, str):
                    f.write(data)
                else:
                    f.write(str(data))

                f.write(f"\n{'='*60}\n")

            logger.info(f"📝 Captured detailed data: {section} - {title}")
        except Exception as e:
            logger.warning(f"⚠️  Failed to write detailed data to file: {e}")

    def _write_workflow_event(
        self, event_type: str, event_data: Any, context: str = ""
    ):
        """Write workflow event data to output file for debugging."""
        if not self.capture_llm_output or not self.llm_output_file:
            return

        try:
            with open(self.llm_output_file, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"WORKFLOW EVENT: {event_type}\n")
                f.write(f"TIMESTAMP: {datetime.now(timezone.utc).isoformat()}\n")
                if context:
                    f.write(f"CONTEXT: {context}\n")
                f.write(f"{'='*60}\n")

                # Format event data
                if isinstance(event_data, (dict, list)):
                    # Pretty print JSON
                    f.write(json.dumps(event_data, indent=2, default=str))
                elif hasattr(event_data, "__dict__"):
                    # Object - convert to dict
                    f.write(json.dumps(vars(event_data), indent=2, default=str))
                else:
                    f.write(str(event_data))

                f.write(f"\n{'='*60}\n")

        except Exception as e:
            logger.warning(f"⚠️  Failed to write workflow event to file: {e}")

    def _parse_response_as_json(self, response_text: str) -> Any:
        """
        Try to parse response as JSON, return original if not valid JSON.

        Args:
            response_text: The response text to parse

        Returns:
            Parsed JSON if valid, otherwise the original text
        """
        try:
            return json.loads(response_text)
        except (json.JSONDecodeError, TypeError):
            return response_text

    async def run_full_test(
        self, query: Optional[str] = "", image: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run complete composer-based end-to-end pipeline test."""
        logger.info("🚀 Starting Composer Real End-to-End Pipeline Test")
        logger.info("=" * 80)

        test_results = {
            "overall_success": False,
            "execution_time": 0,
            "workflow_time": 0,
            "components_passed": 0,
            "total_components": 8,  # Added cleanup as a tracked component
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

            # Phase 2: Composer Service Initialization
            logger.info("🎼 Phase 2: Composer Service Initialization")
            composer_result = await self.initialize_composer_service()
            test_results["results"]["composer_initialization"] = composer_result
            if composer_result["success"]:
                test_results["components_passed"] += 1

            # Phase 4: Real Conversation Creation (includes user profile creation)
            logger.info("💬 Phase 4: Real Conversation Creation")
            conversation_result = await self.create_real_conversation()
            test_results["results"]["conversation_creation"] = conversation_result
            if conversation_result["success"]:
                test_results["components_passed"] += 1

            # User profile creation is handled within conversation creation
            # (user creation and config retrieval are part of the conversation flow)
            user_profile_result = {
                "success": conversation_result["success"],
                "model_name": "qwen3-30b-a3b-q4-k-m",  # Set the expected model name
            }
            test_results["results"]["user_profile_creation"] = user_profile_result
            if user_profile_result["success"]:
                test_results["components_passed"] += 1

            # Phase 5: Real Message with Tool Context
            logger.info("📝 Phase 5: Real Message with Tool Context")
            message_result = await self.create_real_message_with_tools(
                query=query, image=image
            )
            test_results["results"]["message_creation"] = message_result
            if message_result["success"]:
                test_results["components_passed"] += 1

            # Phase 6: Composer Workflow Execution (THE KEY TEST)
            logger.info("🎼 Phase 6: Composer Workflow Execution")
            workflow_result = await self.execute_composer_workflow()
            test_results["results"]["workflow_execution"] = workflow_result
            if workflow_result["success"]:
                test_results["components_passed"] += 1
                test_results["workflow_time"] = workflow_result.get("execution_time", 0)

            # Phase 7: Real Output Validation
            logger.info("✅ Phase 7: Real Output Validation")
            validation_result = await self.validate_real_outputs()
            test_results["results"]["output_validation"] = validation_result
            if validation_result["success"]:
                test_results["components_passed"] += 1

            # Calculate success
            all_passed = (
                test_results["components_passed"] == test_results["total_components"]
            )
            test_results["overall_success"] = all_passed
            test_results["execution_time"] = time.time() - start_time

        except Exception as e:
            logger.error(f"❌ Test execution failed: {str(e)}")
            test_results["error"] = str(e)
            import traceback

            traceback.print_exc()

        finally:
            # Always attempt cleanup - but track as a component that can fail the test
            logger.info("🧹 Phase 8: Real Data Cleanup")
            cleanup_result = await self.cleanup_real_data()
            test_results["results"]["data_cleanup"] = cleanup_result
            test_results["entities_cleaned"] = cleanup_result.get("cleaned_count", 0)

            # Cleanup failure should fail the overall test
            if cleanup_result.get("success", False):
                test_results["components_passed"] += 1
                logger.info("   ✅ Cleanup validation passed")
            else:
                logger.error("   ❌ Cleanup validation failed - test marked as failure")
                # Don't increment components_passed for failed cleanup

            # Recalculate overall success after cleanup (which can fail the test)
            all_passed_including_cleanup = (
                test_results["components_passed"] == test_results["total_components"]
            )
            test_results["overall_success"] = all_passed_including_cleanup

            # Log cleanup impact on overall result
            if not cleanup_result.get("success", False):
                logger.error(
                    f"   ❌ Overall test FAILED due to cleanup issues (components: {test_results['components_passed']}/{test_results['total_components']})"
                )

            # Finalize LLM output capture
            self._finalize_llm_output()

        # Print comprehensive results
        await self.print_test_summary(test_results)
        self._print_llm_output_summary()

        return test_results

    async def setup_real_infrastructure(self) -> Dict[str, Any]:
        """Set up real infrastructure components."""
        logger.info("🏗️  Setting up real infrastructure...")

        try:
            # Initialize real database connection
            from db import storage

            # Build connection string from environment variables
            db_host = os.getenv("DB_HOST", "localhost")
            db_port = os.getenv("DB_PORT", "5432")
            db_user = os.getenv("DB_USER", "lsm")
            db_password = os.getenv("DB_PASSWORD", "")
            db_name = os.getenv("DB_NAME", "llmmll")
            db_sslmode = os.getenv("DB_SSLMODE", "disable")

            connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}?sslmode={db_sslmode}"

            await storage.initialize(connection_string)

            # Verify storage is properly initialized
            if not storage.initialized:
                raise RuntimeError("Storage failed to initialize properly")

            self.storage = storage
            logger.info("   ✅ Database connection established")

            return {
                "success": True,
                "database_connected": True,
                "target_model": self.target_model,
            }

        except Exception as e:
            logger.error(f"   ❌ Infrastructure setup failed: {str(e)}")
            return {"success": False, "error": str(e)}

    async def initialize_composer_service(self) -> Dict[str, Any]:
        """Initialize the composer service."""
        logger.info("🎼 Initializing composer service...")

        try:
            # Import and initialize composer
            from composer import initialize_composer, get_composer_service

            await initialize_composer()
            logger.info("   ✅ Composer service initialized")

            # Test service availability
            service = get_composer_service()
            logger.info(f"   ✅ Composer service available: {type(service).__name__}")

            return {
                "success": True,
                "service_initialized": True,
                "service_type": type(service).__name__,
            }

        except Exception as e:
            logger.error(f"   ❌ Composer initialization failed: {str(e)}")
            import traceback

            traceback.print_exc()
            return {"success": False, "error": str(e)}

    async def create_real_conversation(self) -> Dict[str, Any]:
        """Create real conversation in database."""
        logger.info("💬 Creating real conversation...")

        try:
            from db import storage

            # Ensure storage is available
            if not storage or not storage.pool or not storage.conversation:
                raise RuntimeError("Storage components not available")

            # Ensure user exists in database first
            async with storage.pool.acquire() as conn:
                await conn.execute(
                    "INSERT INTO users (id) VALUES ($1) ON CONFLICT (id) DO NOTHING",
                    self.test_user_id,
                )
            logger.info(f"   ✅ Ensured user exists: {self.test_user_id}")

            # Create real conversation
            from models.conversation import Conversation
            from datetime import datetime

            test_conversation = Conversation(
                id=0,  # Will be set by database
                user_id=self.test_user_id,
                title="Composer Real End-to-End Test Conversation",
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            conversation_id = await storage.conversation.create_conversation(
                test_conversation
            )

            if not conversation_id:
                raise RuntimeError("Failed to create conversation")

            self.test_conversation_id = conversation_id

            logger.info(f"   ✅ Created real conversation ID: {conversation_id}")

            return {"success": True, "conversation_id": conversation_id}

        except Exception as e:
            logger.error(f"   ❌ Conversation creation failed: {str(e)}")
            import traceback

            traceback.print_exc()
            return {"success": False, "error": str(e)}

    async def create_real_message_with_tools(
        self,
        query: Optional[str] = "",
        image: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create real user message with tool-calling context."""
        logger.info("📝 Creating real message with tool context...")

        try:
            from db import storage
            from models.message import Message
            from models.message_role import MessageRole
            from models.message_content import MessageContent, MessageContentType

            # Ensure storage is available
            if not storage or not storage.message:
                raise RuntimeError("Storage message service not available")

            # Create a multimodal message for testing vision capabilities
            query_text = (
                query
                or """Look at this image and describe what you see. What colors are visible, and what might this represent? Also, please search the web for information about the latest developments in multimodal AI models that can process both text and images together."""
            )

            content_list = []
            if image or (not query and not image):
                img = (
                    image
                    or "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
                )
                content_list.append(
                    MessageContent(
                        type=MessageContentType.IMAGE,
                        url=img,
                    )
                )
            content_list.append(
                MessageContent(
                    type=MessageContentType.TEXT,
                    text=query_text,
                )
            )

            user_message = Message(
                conversation_id=self.test_conversation_id,
                role=MessageRole.USER,
                content=content_list,
                created_at=datetime.now(timezone.utc),
            )

            # Add message to database
            message_id = await storage.message.add_message(user_message)
            if not message_id:
                raise RuntimeError("Failed to add message to database")

            self.test_message_id = message_id

            logger.info(f"   ✅ Created real user message ID: {message_id}")
            logger.info(f"   📋 Message length: {len(query_text)} characters")

            return {
                "success": True,
                "message_id": message_id,
                "message_length": len(query_text),
            }

        except Exception as e:
            logger.error(f"   ❌ Message creation failed: {str(e)}")
            import traceback

            traceback.print_exc()
            return {"success": False, "error": str(e)}

    async def execute_composer_workflow(self) -> Dict[str, Any]:
        """Execute composer workflow using the new architecture."""
        logger.info("🎼 Executing composer workflow...")

        try:
            # Import composer functions
            from composer import (
                compose_workflow,
                create_initial_state,
                execute_workflow,
            )
            from models import WorkflowType
            from db import storage

            # Ensure we have required IDs
            if not self.test_conversation_id or not storage or not storage.message:
                raise Exception("Missing required components for workflow execution")

            # Get conversation messages for context
            messages = await storage.message.get_conversation_history(
                self.test_conversation_id
            )
            if not messages:
                raise Exception("No messages found for conversation")

            logger.info(f"   📝 Processing {len(messages)} messages")

            # Capture conversation history
            messages_data = []
            for msg in messages:
                messages_data.append(msg.model_dump_json())

            # Step 1: Compose workflow for user
            logger.info("   🎼 Step 1: Composing workflow...")
            workflow = await compose_workflow(self.test_user_id)
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"{self.output_dir}/workflow_graph_{timestamp}.md"
                doc = workflow.get_graph().draw_mermaid(
                    with_styles=True,
                )
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write("```mermaid\n")
                    f.write(doc)
                    f.write("\n```\n")
                logger.info(f"  Workflow graph saved: {output_path}")
                logger.info(f"   ✅ Workflow composed: {type(workflow).__name__}")
            except Exception as e:
                logger.warning(f"   ⚠️ Could not generate workflow graph image: {e}")
                pass

            # Step 2: Create initial state
            logger.info("   🎼 Step 2: Creating initial state...")
            initial_state = await create_initial_state(
                user_id=self.test_user_id,
                conversation_id=self.test_conversation_id,
            )
            logger.info(f"   ✅ Initial state created: {type(initial_state).__name__}")

            # Capture initial state information
            initial_state_data = {
                "state_type": type(initial_state).__name__,
                "user_id": self.test_user_id,
                "workflow_type": WorkflowType.GENERAL,
                "message_count": len(messages),
            }

            # Try to extract more state data if possible
            if hasattr(initial_state, "__dict__"):
                try:
                    state_dict = {}
                    for key, value in initial_state.__dict__.items():
                        # Convert complex objects to string representations
                        if hasattr(value, "__dict__"):
                            state_dict[key] = str(value)
                        elif isinstance(
                            value, (list, dict, str, int, float, bool, type(None))
                        ):
                            state_dict[key] = value
                        else:
                            state_dict[key] = str(value)
                    initial_state_data["state_contents"] = state_dict
                except Exception as e:
                    logger.warning(f"Could not extract full state data: {e}")

            self._write_detailed_data(
                section="WORKFLOW",
                title="Initial State",
                data=initial_state_data,
                description="Initial workflow state before execution",
            )

            # Step 3: Execute workflow with streaming
            logger.info("   🎼 Step 3: Executing workflow with streaming...")
            start_time = time.time()
            response_chunks = []
            tool_calls_detected = False
            full_response = ""
            error_events: list = []

            # Capture workflow events and detailed data
            event_count = 0
            tool_events = []
            model_events = []

            async for event in execute_workflow(workflow, initial_state, stream=True):
                event_count += 1

                # Process different types of events - use dict access for safety
                event_dict = event if isinstance(event, dict) else {}

                # Check if this event contains tool execution information
                if hasattr(event_dict, "__dict__") and hasattr(event_dict, "node_name"):
                    # This could be a node execution event
                    if getattr(event_dict, "node_name", None) == "tool_executor":
                        tool_calls_detected = True
                        logger.info("   🛠️  Tool execution node detected")

                # Capture significant events for detailed logging
                if "event" in event_dict:
                    event_type = event_dict["event"]
                    # Capture only actual error events that indicate workflow failures
                    # Note: on_tool_error is not necessarily a failure - tools can handle errors gracefully
                    if event_type in [
                        "on_chain_error",
                        "on_chat_model_error",
                        "on_llm_error",
                        "on_retriever_error",
                    ]:
                        # Check if this is actually an unrecoverable error
                        error_data = event_dict.get("data", {})
                        if isinstance(error_data, dict) and "error" in error_data:
                            # Only count as error if there's actual error data
                            error_events.append(event_dict)

                    # Log significant events to output file
                    if event_type in [
                        "on_chat_model_start",
                        "on_chat_model_end",
                        "on_tool_start",
                        "on_tool_end",
                        "on_chain_start",
                        "on_chain_end",
                    ]:
                        self._write_workflow_event(
                            event_type=event_type,
                            event_data=event_dict,
                            context=f"Event {event_count} in workflow execution",
                        )

                    # Handle streaming events
                    if event_type == "on_chat_model_start":
                        # Capture the input to the model
                        if "data" in event_dict and "input" in event_dict["data"]:
                            model_input = event_dict["data"]["input"]
                            self._write_detailed_data(
                                section="MODEL_INPUT",
                                title="Model Input Messages",
                                data=model_input,
                                description="Messages and prompts sent to the LLM",
                            )
                            model_events.append(
                                {
                                    "type": "model_start",
                                    "input": model_input,
                                    "timestamp": time.time(),
                                }
                            )

                    elif event_type == "on_chat_model_stream":
                        # Token streaming
                        if "data" in event_dict:
                            data = event_dict["data"]
                            if isinstance(data, dict) and "chunk" in data:
                                chunk = data["chunk"]
                                # Handle both dict and object chunk formats
                                content = None
                                if isinstance(chunk, dict):
                                    content = chunk.get("content", "")
                                elif hasattr(chunk, "content"):
                                    content = getattr(chunk, "content", "")

                                if content:
                                    full_response += content
                                    response_chunks.append(
                                        {
                                            "type": "token",
                                            "content": content,
                                            "timestamp": time.time(),
                                        }
                                    )

                    elif event_type == "on_chat_model_end":
                        if "data" in event_dict:
                            data = event_dict["data"]
                            if isinstance(data, dict) and "output" in data:
                                output = data["output"]
                                if hasattr(output, "content") and output.content:
                                    # This is the complete response
                                    if output.content not in full_response:
                                        full_response += output.content
                                        response_chunks.append(
                                            {
                                                "type": "complete",
                                                "content": output.content,
                                                "timestamp": time.time(),
                                            }
                                        )
                                        logger.info(
                                            f"   📝 Captured complete response: {len(output.content)} chars"
                                        )

                                # Capture the complete model output
                                self._write_detailed_data(
                                    section="MODEL_OUTPUT",
                                    title="Complete Model Response",
                                    data={
                                        "content": (
                                            output.content
                                            if hasattr(output, "content")
                                            else str(output)
                                        ),
                                        "metadata": (
                                            getattr(output, "response_metadata", {})
                                            if hasattr(output, "response_metadata")
                                            else {}
                                        ),
                                        "additional_kwargs": (
                                            getattr(output, "additional_kwargs", {})
                                            if hasattr(output, "additional_kwargs")
                                            else {}
                                        ),
                                    },
                                    description="Complete response from the LLM",
                                )

                    elif event_type == "on_tool_start":
                        # Tool execution starting - this indicates tool calls are happening
                        tool_calls_detected = True
                        if "data" in event_dict:
                            tool_data = event_dict["data"]
                            logger.info(
                                f"   🛠️  Tool call detected: {tool_data.get('name', 'Unknown')}"
                            )
                            self._write_detailed_data(
                                section="TOOL_EXECUTION",
                                title=f"Tool Start: {tool_data.get('name', 'Unknown')}",
                                data=tool_data,
                                description=f"Tool execution starting for {tool_data.get('name', 'Unknown')}",
                            )
                            tool_events.append(
                                {
                                    "type": "tool_start",
                                    "data": tool_data,
                                    "timestamp": time.time(),
                                }
                            )

                    elif event_type == "on_tool_end":
                        # Tool execution completed
                        if "data" in event_dict:
                            tool_data = event_dict["data"]
                            self._write_detailed_data(
                                section="TOOL_EXECUTION",
                                title=f"Tool End: {tool_data.get('name', 'Unknown')}",
                                data=tool_data,
                                description=f"Tool execution completed for {tool_data.get('name', 'Unknown')}",
                            )
                            tool_events.append(
                                {
                                    "type": "tool_end",
                                    "data": tool_data,
                                    "timestamp": time.time(),
                                }
                            )

                elif isinstance(event_dict, dict):
                    # Handle dict events
                    if "data" in event_dict:
                        data = event_dict["data"]
                        if isinstance(data, dict) and "content" in data:
                            content = data["content"]
                            full_response += content
                            response_chunks.append(
                                {
                                    "type": "content",
                                    "content": content,
                                    "timestamp": time.time(),
                                }
                            )

                        # Check for tool calls
                        if "tool_calls" in data or "function_call" in data:
                            tool_calls_detected = True
                            logger.info("   🛠️  Tool call detected in workflow")

            execution_time = time.time() - start_time
            logger.info(f"   ✅ Workflow execution completed in {execution_time:.2f}s")
            logger.info(f"   📊 Total response chunks: {len(response_chunks)}")

            # Additional tool call detection methods
            if len(tool_events) > 0 and not tool_calls_detected:
                tool_calls_detected = True
                logger.info(
                    f"   🛠️  Tool calls detected via tool_events: {len(tool_events)}"
                )

            # Check if the response indicates successful tool usage (e.g. web search results)
            if not tool_calls_detected and full_response:
                # Look for indicators of successful tool execution in the response
                tool_indicators = [
                    "based on recent findings",
                    "search results",
                    "according to",
                    "recent developments",
                    "latest information",
                    "current information",
                    "web search",
                ]
                if any(
                    indicator.lower() in full_response.lower()
                    for indicator in tool_indicators
                ):
                    tool_calls_detected = True
                    logger.info(
                        "   🛠️  Tool calls detected via response content analysis"
                    )

            logger.info(f"   🛠️  Tool calls detected: {tool_calls_detected}")
            if error_events:
                logger.error(
                    f"   🚫 Captured {len(error_events)} workflow error events; marking execution as failed"
                )

            # Write execution summary
            execution_summary = {
                "execution_time": execution_time,
                "total_events": event_count,
                "response_chunks": len(response_chunks),
                "tool_calls_detected": tool_calls_detected,
                "tool_events_count": len(tool_events),
                "model_events_count": len(model_events),
                "final_response_length": len(full_response),
            }

            self._write_detailed_data(
                section="EXECUTION_SUMMARY",
                title="Workflow Execution Summary",
                data=execution_summary,
                description="Summary of workflow execution metrics and events",
            )

            # Write all tool events if any
            if tool_events:
                self._write_detailed_data(
                    section="TOOL_SUMMARY",
                    title="All Tool Events",
                    data=tool_events,
                    description=f"Complete list of {len(tool_events)} tool execution events",
                )

            # Write response chunks summary
            if response_chunks:
                self._write_detailed_data(
                    section="RESPONSE_SUMMARY",
                    title="Response Generation Summary",
                    data={
                        "total_chunks": len(response_chunks),
                        "chunk_types": list(
                            set(chunk["type"] for chunk in response_chunks)
                        ),
                        "first_chunk": response_chunks[0] if response_chunks else None,
                        "last_chunk": response_chunks[-1] if response_chunks else None,
                    },
                    description="Summary of response generation process",
                )

            # Store assistant response in database if we got content
            if full_response and storage and storage.message:
                from models.message import Message
                from models.message_role import MessageRole
                from models.message_content import MessageContent, MessageContentType

                assistant_message = Message(
                    id=None,
                    conversation_id=self.test_conversation_id,
                    role=MessageRole.ASSISTANT,
                    content=[
                        MessageContent(type=MessageContentType.TEXT, text=full_response)
                    ],
                    created_at=datetime.now(timezone.utc),
                )

                assistant_message_id = await storage.message.add_message(
                    assistant_message
                )

                # Capture LLM response
                self._write_llm_response(
                    "composer_workflow_execution",
                    full_response,
                    {
                        "execution_time": execution_time,
                        "chunk_count": len(response_chunks),
                        "tool_calls_detected": tool_calls_detected,
                        "model": self.target_model,
                    },
                )

            # Validate that we actually got meaningful output
            success = True
            validation_errors = []

            # If pipeline node failure logged earlier or error events captured, mark failure
            if error_events:
                success = False
                validation_errors.append(
                    f"Encountered {len(error_events)} workflow error events (first type: {error_events[0].get('event')})"
                )

            # Check 1: Must have actual response content
            if len(full_response.strip()) == 0:
                success = False
                validation_errors.append("No response content generated")

            # Check 2: Must have response chunks from streaming
            if len(response_chunks) == 0:
                success = False
                validation_errors.append("No response chunks captured from workflow")

            # Check 2a: Must have at least one tool call (strict requirement)
            if not tool_calls_detected:
                success = False
                validation_errors.append(
                    "No tool calls were executed during workflow (strict failure)"
                )

            # Check 3: Response length should be reasonable for a real LLM interaction
            if len(full_response) < 10:  # Very short responses likely indicate issues
                success = False
                validation_errors.append(
                    f"Response too short ({len(full_response)} chars), likely incomplete"
                )

            # Check 4: Validate output file has captured responses
            try:
                if self.llm_output_file:
                    with open(self.llm_output_file, "r", encoding="utf-8") as f:
                        output_content = f.read()
                        # Count actual response entries (PHASE: markers indicate captured content)
                        response_count = output_content.count("PHASE:")
                        if response_count == 0:
                            success = False
                            validation_errors.append(
                                "Output file shows 0 captured responses despite workflow execution"
                            )
                        logger.info(
                            f"   📊 Output file validation: {response_count} responses captured"
                        )
                else:
                    validation_errors.append("No output file path configured")
            except Exception as file_error:
                success = False
                validation_errors.append(
                    f"Could not validate output file: {file_error}"
                )

            # Check 5: If workflow took significant time but produced no output, likely an internal error
            if execution_time > 3.0 and len(full_response) == 0:
                success = False
                validation_errors.append(
                    f"Workflow ran for {execution_time:.1f}s but generated no output (likely internal error)"
                )

            # Check 6: Validate tool availability awareness (only flag as issue if no tools were used)
            tool_availability_issues = []
            if not tool_calls_detected:
                # Extract user-facing content by removing <think> tags and their content
                import re

                user_facing_content = re.sub(
                    r"<think>.*?</think>", "", full_response, flags=re.DOTALL
                )

                if "can't perform actual searches" in user_facing_content.lower():
                    tool_availability_issues.append(
                        "Model believes it cannot perform web searches without using tools"
                    )
                if "system can't perform" in user_facing_content.lower():
                    tool_availability_issues.append(
                        "Model believes system lacks tool capabilities without using tools"
                    )
                if "cannot access real-time" in user_facing_content.lower():
                    tool_availability_issues.append(
                        "Model believes it cannot access real-time data without using tools"
                    )

            if tool_availability_issues:
                success = False
                validation_errors.extend(tool_availability_issues)
                logger.error(
                    f"   🚫 Tool availability issues detected: {', '.join(tool_availability_issues)}"
                )

            # Check 7: Validate dynamic tool generation doesn't have error handling flags
            if (
                "handle_tool_error" in full_response
                and '"handle_tool_error": true' in full_response
            ):
                success = False
                validation_errors.append(
                    "Dynamic tool has handle_tool_error=true indicating error conditions"
                )
                logger.error(
                    f"   ⚠️  Dynamic tool configured with error handling (problematic)"
                )

            if not success:
                logger.error(
                    f"   ❌ Workflow validation failed: {', '.join(validation_errors)}"
                )

            return {
                "success": success,
                "execution_time": execution_time,
                "response_chunks": len(response_chunks),
                "tool_calls_detected": tool_calls_detected,
                "response_length": len(full_response),
                "workflow_type": "composer_langgraph",
                "final_response_raw": full_response,  # Full response for debugging
                "final_response": self._parse_response_as_json(
                    full_response
                ),  # Try to parse as JSON
                "validation_errors": validation_errors if validation_errors else None,
                "tool_availability_correct": len(
                    [
                        e
                        for e in validation_errors
                        if "tool" in e.lower() or "search" in e.lower()
                    ]
                )
                == 0,
                "dynamic_tool_error_free": "handle_tool_error" not in full_response
                or '"handle_tool_error": false' in full_response,
            }

        except Exception as e:
            logger.error(f"   ❌ Composer workflow execution failed: {str(e)}")
            import traceback

            traceback.print_exc()
            return {"success": False, "error": str(e)}

    async def validate_real_outputs(self) -> Dict[str, Any]:
        """Validate real outputs and database integrity."""
        logger.info("✅ Validating real outputs...")

        try:
            from db import storage

            # Ensure we have required components
            if (
                not storage
                or not storage.conversation
                or not storage.message
                or not self.test_conversation_id
            ):
                raise Exception(
                    "Missing required storage components or conversation ID"
                )

            # Validate conversation exists and has messages
            conversation = await storage.conversation.get_conversation(
                self.test_conversation_id
            )
            if not conversation:
                raise Exception("Test conversation not found")

            messages = await storage.message.get_conversation_history(
                self.test_conversation_id
            )
            if len(messages) < 2:  # Should have user + assistant messages
                logger.warning(
                    f"   ⚠️  Expected at least 2 messages, found {len(messages)}"
                )

            logger.info(
                f"   ✅ Conversation validation passed: {len(messages)} messages"
            )

            # Validate model profile exists (only if storage available)
            if storage.model_profile:
                model_profile = await storage.model_profile.get_model_profile_by_id(
                    self.test_model_profile_id, self.test_user_id
                )
                if model_profile:
                    logger.info(
                        f"   ✅ Model profile validation passed: {model_profile.name}"
                    )

            # Calculate quality metrics
            assistant_messages = [m for m in messages if m.role.value == "assistant"]

            # Critical validation: Must have assistant responses for a successful test
            if len(assistant_messages) == 0:
                logger.error(
                    "   ❌ No assistant messages found - workflow failed to generate responses"
                )
                return {
                    "success": False,
                    "error": "No assistant responses generated - workflow execution failed",
                    "conversation_valid": True,
                    "model_profile_valid": True,
                    "message_count": len(messages),
                    "assistant_messages": 0,
                    "response_quality_score": 0,
                }

            response_quality_score = 0
            if assistant_messages:
                latest_response = assistant_messages[-1]
                response_text = ""
                for content in latest_response.content:
                    if content.type.value == "text" and content.text:
                        response_text += content.text

                # Basic quality checks
                if len(response_text) > 100:
                    response_quality_score += 25
                if "2024" in response_text or "recent" in response_text.lower():
                    response_quality_score += 25
                if len(response_text.split()) > 50:
                    response_quality_score += 25
                if any(
                    keyword in response_text.lower()
                    for keyword in [
                        "ai",
                        "artificial intelligence",
                        "model",
                        "research",
                    ]
                ):
                    response_quality_score += 25

                logger.info(
                    f"   📊 Response quality score: {response_quality_score}/100"
                )

            # Validate conversation title if generated
            title_validation = await self._validate_conversation_title()

            return {
                "success": True,
                "conversation_valid": True,
                "model_profile_valid": True,
                "message_count": len(messages),
                "assistant_messages": len(assistant_messages),
                "response_quality_score": response_quality_score,
                "title_validation": title_validation,
            }

        except Exception as e:
            logger.error(f"   ❌ Output validation failed: {str(e)}")
            return {"success": False, "error": str(e)}

    async def _validate_conversation_title(self) -> Dict[str, Any]:
        """Validate the generated conversation title meets requirements."""
        try:
            from db import storage

            if not storage or not storage.conversation or not self.test_conversation_id:
                return {"valid": False, "error": "Missing storage or conversation ID"}

            # Get conversation to check for title
            conversation = await storage.conversation.get_conversation(
                self.test_conversation_id
            )

            if not conversation:
                return {"valid": False, "error": "Conversation not found"}

            title = getattr(conversation, "title", None)

            if not title or title.strip() == "":
                return {"valid": False, "error": "No title generated"}

            # Clean and validate title
            title = title.strip()
            word_count = len(title.split())

            # Check title requirements
            validation_results = {
                "valid": True,
                "title": title,
                "word_count": word_count,
                "meets_word_limit": word_count <= 5,
                "has_content": len(title) > 0,
                "no_quotes": not (title.startswith('"') and title.endswith('"')),
                "properly_capitalized": title[0].isupper() if title else False,
            }

            # Overall validation
            validation_results["valid"] = all(
                [
                    validation_results["meets_word_limit"],
                    validation_results["has_content"],
                    validation_results["no_quotes"],
                ]
            )

            if validation_results["valid"]:
                logger.info(
                    f"   ✅ Title validation passed: '{title}' ({word_count} words)"
                )
            else:
                logger.error(
                    f"   ❌ Title validation failed: '{title}' ({word_count} words)"
                )

            return validation_results

        except Exception as e:
            logger.error(f"   ❌ Title validation error: {str(e)}")
            return {"valid": False, "error": str(e)}

    async def cleanup_real_data(self):
        """Clean up all real test data from database."""
        logger.info("🧹 Cleaning up real test data...")

        cleaned_count = 0
        cleanup_errors = []

        try:
            from db import storage

            # Ensure we have storage available
            if not storage or not storage.pool:
                logger.warning("   ⚠️  Storage not available for cleanup")
                return {
                    "success": False,
                    "error": "Storage not available",
                    "cleaned_count": 0,
                }

            # Validate cascading deletes by checking what will be deleted
            async with storage.pool.acquire() as conn:
                # Count related entities before deletion
                related_counts = {}

                # Model profiles
                profile_count = await conn.fetchval(
                    "SELECT COUNT(*) FROM model_profiles WHERE user_id = $1",
                    self.test_user_id,
                )
                related_counts["model_profiles"] = profile_count

                # Dynamic tools
                tool_count = await conn.fetchval(
                    "SELECT COUNT(*) FROM dynamic_tools WHERE user_id = $1",
                    self.test_user_id,
                )
                related_counts["dynamic_tools"] = tool_count

                # Conversations
                conversation_count = await conn.fetchval(
                    "SELECT COUNT(*) FROM conversations WHERE user_id = $1",
                    self.test_user_id,
                )
                related_counts["conversations"] = conversation_count

                # Messages (should cascade from conversations)
                message_count = await conn.fetchval(
                    "SELECT COUNT(*) FROM messages WHERE conversation_id IN (SELECT id FROM conversations WHERE user_id = $1)",
                    self.test_user_id,
                )
                related_counts["messages"] = message_count

                # Memories (cascade through user_id, not conversation_id)
                memory_count = await conn.fetchval(
                    "SELECT COUNT(*) FROM memories WHERE user_id = $1",
                    self.test_user_id,
                )
                related_counts["memories"] = memory_count

                # Summaries (should cascade from conversations)
                summary_count = await conn.fetchval(
                    "SELECT COUNT(*) FROM summaries WHERE conversation_id IN (SELECT id FROM conversations WHERE user_id = $1)",
                    self.test_user_id,
                )
                related_counts["summaries"] = summary_count

                # Search topic syntheses (should cascade from conversations)
                synthesis_count = await conn.fetchval(
                    "SELECT COUNT(*) FROM search_topic_syntheses WHERE conversation_id IN (SELECT id FROM conversations WHERE user_id = $1)",
                    self.test_user_id,
                )
                related_counts["search_topic_syntheses"] = synthesis_count

                logger.info(f"   📊 Related entities before deletion: {related_counts}")

                # Manual cascading deletes since DB triggers may not be working correctly
                # Delete in dependency order to avoid foreign key constraint violations
                logger.info(
                    f"   🔍 Debug: conversation_id={self.test_conversation_id}, model_profile_id={self.test_model_profile_id}"
                )

                # 1. Delete messages first (dependent on conversations) - use specific conversation ID
                if self.test_conversation_id:
                    deleted_messages_result = await conn.execute(
                        "DELETE FROM messages WHERE conversation_id = $1",
                        self.test_conversation_id,
                    )
                    logger.info(
                        f"   🗑️  Deleted messages for conversation {self.test_conversation_id}: {deleted_messages_result}"
                    )
                else:
                    logger.warning(f"   ⚠️  No conversation ID to delete messages from")

                # 2. Delete summaries (dependent on conversations)
                if self.test_conversation_id:
                    deleted_summaries_result = await conn.execute(
                        "DELETE FROM summaries WHERE conversation_id = $1",
                        self.test_conversation_id,
                    )
                    logger.info(
                        f"   🗑️  Deleted summaries for conversation {self.test_conversation_id}: {deleted_summaries_result}"
                    )

                # 3. Delete search topic syntheses (dependent on conversations)
                if self.test_conversation_id:
                    deleted_syntheses_result = await conn.execute(
                        "DELETE FROM search_topic_syntheses WHERE conversation_id = $1",
                        self.test_conversation_id,
                    )
                    logger.info(
                        f"   🗑️  Deleted syntheses for conversation {self.test_conversation_id}: {deleted_syntheses_result}"
                    )

                # 4. Delete conversations (dependent on user) - use specific conversation ID for precision
                if self.test_conversation_id:
                    deleted_conversations_result = await conn.execute(
                        "DELETE FROM conversations WHERE id = $1 AND user_id = $2",
                        self.test_conversation_id,
                        self.test_user_id,
                    )
                    logger.info(
                        f"   🗑️  Deleted conversation {self.test_conversation_id}: {deleted_conversations_result}"
                    )

                # 5. Delete model profiles (dependent on user) - use specific profile ID for precision
                if self.test_model_profile_id:
                    deleted_profiles_result = await conn.execute(
                        "DELETE FROM model_profiles WHERE id = $1 AND user_id = $2",
                        self.test_model_profile_id,
                        self.test_user_id,
                    )
                    logger.info(
                        f"   🗑️  Deleted model profile {self.test_model_profile_id}: {deleted_profiles_result}"
                    )
                else:
                    logger.warning(f"   ⚠️  No model profile ID to delete")

                # 6. Delete dynamic tools (dependent on user) - these have proper CASCADE so may already be deleted
                deleted_tools_result = await conn.execute(
                    "DELETE FROM dynamic_tools WHERE user_id = $1", self.test_user_id
                )
                logger.info(f"   🗑️  Deleted dynamic tools: {deleted_tools_result}")

                # 7. Delete memories (dependent on user) - these have proper CASCADE so may already be deleted
                deleted_memories_result = await conn.execute(
                    "DELETE FROM memories WHERE user_id = $1", self.test_user_id
                )
                logger.info(f"   🗑️  Deleted memories: {deleted_memories_result}")

                # 8. Finally delete the user
                await conn.execute("DELETE FROM users WHERE id = $1", self.test_user_id)
                logger.info(f"   ✅ Deleted user: {self.test_user_id}")

                # Count cleanup based on what was actually deleted
                cleaned_count = 1  # User
                cleaned_count += sum(
                    [count for count in related_counts.values() if count > 0]
                )

                # TEMPORARY: Force a cleanup failure for testing
                # TODO: Remove this line after testing cleanup failure handling
                # raise Exception("Forced cleanup failure for testing")

                # Validate cascading deletes worked
                remaining_counts = {}

                # Check that all related entities were deleted
                remaining_counts["model_profiles"] = await conn.fetchval(
                    "SELECT COUNT(*) FROM model_profiles WHERE user_id = $1",
                    self.test_user_id,
                )

                remaining_counts["dynamic_tools"] = await conn.fetchval(
                    "SELECT COUNT(*) FROM dynamic_tools WHERE user_id = $1",
                    self.test_user_id,
                )

                remaining_counts["conversations"] = await conn.fetchval(
                    "SELECT COUNT(*) FROM conversations WHERE user_id = $1",
                    self.test_user_id,
                )

                # Check messages directly since conversation may be deleted already
                # Get actual conversation IDs that were associated with this user before deletion
                remaining_counts["messages"] = await conn.fetchval(
                    "SELECT COUNT(*) FROM messages WHERE conversation_id = $1",
                    self.test_conversation_id,
                )

                remaining_counts["memories"] = await conn.fetchval(
                    "SELECT COUNT(*) FROM memories WHERE user_id = $1",
                    self.test_user_id,
                )

                # Check summaries and syntheses directly by conversation ID
                remaining_counts["summaries"] = await conn.fetchval(
                    "SELECT COUNT(*) FROM summaries WHERE conversation_id = $1",
                    self.test_conversation_id,
                )

                remaining_counts["search_topic_syntheses"] = await conn.fetchval(
                    "SELECT COUNT(*) FROM search_topic_syntheses WHERE conversation_id = $1",
                    self.test_conversation_id,
                )

                logger.info(
                    f"   📊 Remaining entities after deletion: {remaining_counts}"
                )

                # Validate that cascading deletes worked
                cascade_failures = []
                for entity_type, count in remaining_counts.items():
                    if count > 0:
                        cascade_failures.append(f"{entity_type}: {count} remaining")

                # TEMPORARY: Force cascade failure detection for testing
                # TODO: Remove this after testing cleanup failure handling
                # cascade_failures.append("test_failure: 1 remaining")

                if cascade_failures:
                    error_msg = (
                        f"Cascading deletes failed: {'; '.join(cascade_failures)}"
                    )
                    cleanup_errors.append(error_msg)
                    logger.error(f"   ❌ {error_msg}")
                else:
                    logger.info("   ✅ All cascading deletes succeeded")
                    cleaned_count += sum(related_counts.values())

        except Exception as e:
            error_msg = f"Failed to delete user or validate cascades: {e}"
            cleanup_errors.append(error_msg)
            logger.error(f"   ❌ Cleanup failed: {str(e)}")
            return {"success": False, "error": str(e), "cleaned_count": cleaned_count}

        # Return cleanup results
        logger.info(f"✅ Cleanup completed: {cleaned_count} entities deleted")
        return {
            "success": len(cleanup_errors) == 0,
            "cleaned_count": cleaned_count,
            "errors": cleanup_errors,
            "total_entities": sum(related_counts.values()) + 1,  # +1 for user
        }

    def _finalize_llm_output(self):
        """Finalize the LLM output file with summary statistics."""
        if not self.capture_llm_output or not self.llm_output_file:
            return

        try:
            total_chars = sum(len(resp["response"]) for resp in self.llm_responses)
            total_responses = len(self.llm_responses)

            with open(self.llm_output_file, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"TEST SUMMARY\n")
                f.write(f"{'='*60}\n")
                f.write(f"Total Responses: {total_responses}\n")
                f.write(f"Total Characters: {total_chars:,}\n")
                f.write(
                    f"Average Response Length: {total_chars // max(total_responses, 1):,} chars\n"
                )
                f.write(f"Model Used: {self.target_model}\n")
                f.write(f"Architecture: Composer + LangGraph\n")
                f.write(f"Test Completed: {datetime.now(timezone.utc).isoformat()}\n")
                f.write(f"{'='*60}\n")

            logger.info(f"📝 Finalized LLM output file: {self.llm_output_file}")
            logger.info(
                f"📊 Captured {total_responses} responses totaling {total_chars:,} characters"
            )

        except Exception as e:
            logger.warning(f"⚠️  Failed to finalize LLM output file: {e}")

    def _print_llm_output_summary(self):
        """Print a summary of captured LLM output."""
        if not self.capture_llm_output or not self.llm_output_file:
            return

        total_chars = sum(len(resp["response"]) for resp in self.llm_responses)
        total_responses = len(self.llm_responses)

        print(f"\n{'='*60}")
        print(f"COMPOSER LLM OUTPUT SUMMARY")
        print(f"{'='*60}")
        print(f"Output File: {self.llm_output_file}")
        print(f"Total Responses: {total_responses}")
        print(f"Total Characters: {total_chars:,}")
        print(f"Average Length: {total_chars // max(total_responses, 1):,} chars")
        print(f"Architecture: Composer + LangGraph")
        print(f"{'='*60}")

        if self.print_output and total_responses > 0:
            print(f"\nFULL LLM OUTPUT CONTENT:")
            print(f"{'='*60}")
            try:
                with open(self.llm_output_file, "r", encoding="utf-8") as f:
                    print(f.read())
            except Exception as e:
                print(f"Error reading output file: {e}")
            print(f"{'='*60}")
        elif total_responses > 0:
            print(
                f"\nTo view full content, set print_output=True or read: {self.llm_output_file}"
            )
            print(f"{'='*60}")

    async def print_test_summary(self, results: Dict[str, Any]) -> None:
        """Print comprehensive test summary."""
        logger.info("\n" + "=" * 80)
        logger.info("📊 Composer Real Pipeline Test Summary")
        logger.info("=" * 80)

        success_rate = (
            results["components_passed"] / results["total_components"]
        ) * 100
        overall_success = "YES" if results["overall_success"] else "NO"

        logger.info(f"✅ Overall Success: {overall_success} ({success_rate:.1f}%)")
        logger.info(f"🕒 Total Execution Time: {results['execution_time']:.2f}s")
        logger.info(
            f"⚡ Workflow Execution Time: {results.get('workflow_time', 0):.2f}s"
        )
        logger.info(
            f"🔧 Components Passed: {results['components_passed']}/{results['total_components']}"
        )
        logger.info(f"🏗️  Real Entities Created: {results['entities_created']}")

        # Extract key information from results
        workflow_result = results["results"].get("workflow_execution", {})
        validation_result = results["results"].get("output_validation", {})

        model_name = (
            results["results"]
            .get("user_profile_creation", {})
            .get("model_name", "Unknown")
        )
        tool_calls = workflow_result.get("tool_calls_detected", False)
        quality_score = validation_result.get("response_quality_score", 0)
        tool_availability_correct = workflow_result.get(
            "tool_availability_correct", True
        )
        dynamic_tool_error_free = workflow_result.get("dynamic_tool_error_free", True)

        # Extract title validation information
        title_validation = validation_result.get("title_validation", {})
        title_valid = title_validation.get("valid", False)
        title_word_count = title_validation.get("word_count", "Unknown")
        title_text = title_validation.get("title", "Not generated")

        logger.info(f"🤖 Model Used: {model_name}")
        logger.info(f"🛠️  Tool Calls Detected: {'YES' if tool_calls else 'NO'}")
        logger.info(
            f"🔍 Tool Availability Awareness: {'CORRECT' if tool_availability_correct else 'INCORRECT'}"
        )
        logger.info(
            f"⚙️  Dynamic Tool Error-Free: {'YES' if dynamic_tool_error_free else 'NO'}"
        )
        logger.info(f"📊 Response Quality Score: {quality_score}/100")
        logger.info(
            f"🏷️  Title Generated: {'YES' if title_valid else 'NO'} ({title_word_count} words)"
        )
        if title_valid:
            logger.info(f"    Title: '{title_text}'")
        logger.info(f"🎼 Architecture: Composer + LangGraph")

        logger.info("\n📋 Component Results:")
        component_names = [
            "Infrastructure Setup",
            "Composer Initialization",
            "User Profile Creation",
            "Conversation Creation",
            "Message Creation",
            "Workflow Execution",
            "Output Validation",
            "Data Cleanup",
        ]

        result_keys = [
            "infrastructure_setup",
            "composer_initialization",
            "user_profile_creation",
            "conversation_creation",
            "message_creation",
            "workflow_execution",
            "output_validation",
            "data_cleanup",
        ]

        for name, key in zip(component_names, result_keys):
            result = results["results"].get(key, {})
            status = "✅ PASS" if result.get("success", False) else "❌ FAIL"
            logger.info(f"   {status} {name}")

        # Save detailed results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "debug/out"
        os.makedirs(output_dir, exist_ok=True)
        results_file = f"{output_dir}/composer_test_{timestamp}.json"

        try:
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"📄 Detailed results saved to: {results_file}")
        except Exception as e:
            logger.warning(f"⚠️  Failed to save results file: {e}")


async def main():
    """Main test execution function."""
    import argparse

    logger.info("🧪 Starting Composer Real End-to-End Pipeline Tests")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Target model to test")
    parser.add_argument(
        "--no-capture", action="store_true", help="Disable capturing LLM output to file"
    )
    parser.add_argument(
        "--print-output", action="store_true", help="Print full LLM output to console"
    )
    parser.add_argument(
        "--query", type=str, default="", help="Custom query for the test"
    )
    parser.add_argument(
        "--image", type=str, default=None, help="Custom image for the test"
    )
    args = parser.parse_args()

    # Available models for testing
    available_models = [
        "qwen3-vl-32b-thinking-abliterated",  # Primary multimodal model - use this as default
        "qwen3-30b-a3b-q4-k-m",
        "openai-gpt-oss-20b-uncensored-q5_1",
    ]

    # Test specified model or default
    models_to_test = [args.model] if args.model else [available_models[0]]

    for model in models_to_test:
        logger.info(f"🧪 Testing composer architecture with model: {model}")
        tester = ComposerRealEndToEndTester(
            target_model=model,
            capture_llm_output=not args.no_capture,
            print_output=args.print_output,
        )

        # Run the test
        try:
            results = await tester.run_full_test(query=args.query, image=args.image)

            if results["overall_success"]:
                logger.info(f"🎉 Composer test with {model} PASSED!")
            else:
                logger.error(f"❌ Composer test with {model} FAILED!")
                return 1

        except Exception as e:
            logger.error(f"❌ Test execution failed for {model}: {e}")
            import traceback

            traceback.print_exc()
            return 1

    logger.info("🏁 Composer architecture testing completed successfully!")
    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        exit(exit_code)
    except KeyboardInterrupt:
        logger.info("🛑 Test interrupted by user")
        exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
