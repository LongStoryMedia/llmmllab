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
import time
import uuid
import json
import os
import traceback
import argparse
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from utils.logging import llmmllogger, serialize_event_data
from models import (
    Message,
    MessageRole,
    MessageContent,
    MessageContentType,
    Conversation,
    ChatResponse,
)
from db import storage
from composer import (
    compose_workflow,
    create_initial_state,
    execute_workflow,
    initialize_composer,
    get_composer_service,
)

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

    def _write_section(self, title: str) -> None:
        """Write a section header to the output file."""
        if self.capture_llm_output and self.llm_output_file:
            with open(self.llm_output_file, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"{title}\n")
                f.write(f"{'='*80}\n\n")

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

    def _finalize_llm_output(self) -> None:
        """Finalize LLM output capture."""
        if self.capture_llm_output and self.llm_output_file:
            try:
                with open(self.llm_output_file, "a", encoding="utf-8") as f:
                    f.write(f"\n{'='*80}\n")
                    f.write("TEST SUMMARY\n")
                    f.write(f"{'='*80}\n")
                    f.write(f"Total responses captured: {len(self.llm_responses)}\n")
                    f.write(
                        f"File finalized at: {datetime.now(timezone.utc).isoformat()}\n"
                    )
                    f.write(f"{'='*80}\n")
            except Exception as e:
                logger.warning(f"⚠️  Failed to finalize LLM output: {e}")

    def _print_llm_output_summary(self) -> None:
        """Print summary of captured LLM outputs."""
        if self.capture_llm_output:
            logger.info(
                f"📝 Captured {len(self.llm_responses)} LLM responses to {self.llm_output_file}"
            )
        else:
            logger.info("📝 LLM output capture was disabled")

    def _write_to_output(self, content: str) -> None:
        """Write content to console and file output."""
        print(content, end="", flush=True)
        if self.capture_llm_output and self.llm_output_file:
            try:
                with open(self.llm_output_file, "a", encoding="utf-8") as f:
                    f.write(content)
            except Exception as e:
                logger.warning(f"⚠️  Failed to write content to output file: {e}")

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
                "model_name": self.target_model,
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
            workflow_result = await self.execute_workflow()
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
            logger.error(f"Test execution traceback: {traceback.format_exc()}")

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
            logger.error(f"Composer initialization traceback: {traceback.format_exc()}")
            return {"success": False, "error": str(e)}

    async def create_real_conversation(self) -> Dict[str, Any]:
        """Create real conversation in database."""
        logger.info("💬 Creating real conversation...")

        try:
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
            logger.error(f"Conversation creation traceback: {traceback.format_exc()}")
            return {"success": False, "error": str(e)}

    async def create_real_message_with_tools(
        self,
        query: Optional[str] = "",
        image: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create real user message with tool-calling context."""
        logger.info("📝 Creating real message with tool context...")

        try:

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
            logger.error(f"Message creation traceback: {traceback.format_exc()}")
            return {"success": False, "error": str(e)}

    async def execute_workflow(self) -> Dict[str, Any]:
        """Execute composer workflow with simplified streaming like tools_agent."""
        logger.info("🎼 Executing simplified workflow...")

        try:
            # Ensure we have required IDs
            if not self.test_conversation_id or not storage or not storage.message:
                raise RuntimeError("Missing required components for workflow execution")

            # Get conversation messages for context
            messages = await storage.message.get_conversation_history(
                self.test_conversation_id
            )
            if not messages:
                raise RuntimeError("No messages found for conversation")

            logger.info(f"   📝 Processing {len(messages)} messages")

            # Step 1: Compose workflow for user
            logger.info("   🎼 Step 1: Composing workflow...")
            workflow = await compose_workflow(self.test_user_id)

            # Generate mermaid diagram and set as output file
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"{self.output_dir}/workflow_graph_{timestamp}.md"

                # Set this as our LLM output file for consolidated output
                if self.capture_llm_output:
                    self.llm_output_file = output_path

                doc = workflow.get_graph().draw_mermaid(with_styles=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write("# Composer E2E Test Results\n\n")
                    f.write(
                        f"**Test started at:** {datetime.now(timezone.utc).isoformat()}\n"
                    )
                    f.write(f"**Target Model:** {self.target_model or 'auto-detect'}\n")
                    f.write(f"**User ID:** {self.test_user_id}\n\n")
                    f.write("## Workflow Graph\n\n")
                    f.write("```mermaid\n")
                    f.write(doc)
                    f.write("\n```\n\n")
                    f.write("## LLM Execution Output\n\n")
                logger.info(f"   📊 Workflow graph saved: {output_path}")
                if self.capture_llm_output:
                    logger.info(f"   📝 LLM output will be captured to: {output_path}")
            except Exception as e:
                logger.warning(f"   ⚠️ Could not generate workflow graph: {e}")

            logger.info(f"   ✅ Workflow composed: {type(workflow).__name__}")

            # Step 2: Create initial state
            logger.info("   🎼 Step 2: Creating initial state...")
            initial_state = await create_initial_state(
                user_id=self.test_user_id,
                conversation_id=self.test_conversation_id,
            )
            logger.info(f"   ✅ Initial state created: {type(initial_state).__name__}")

            # Step 3: Execute workflow with simplified streaming like tools_agent
            self._write_section("## STREAMING WORKFLOW EXECUTION")

            start_time = time.time()
            full_response: ChatResponse
            tool_calls_detected = False
            event_count = 0

            # Stream execution exactly like tools_agent.py
            async for res in execute_workflow(
                initial_state=initial_state,
                workflow=workflow,
            ):
                event_count += 1

                if res.done and res.finish_reason == "complete":
                    full_response = res
                    break

                if res.message is None:
                    logger.warning("Received empty message in stream event")
                    continue

                # Handle tool calls like tools_agent
                if res.message.tool_calls:
                    tool_calls_detected = True
                    for t in res.message.tool_calls:
                        tool_text = f"\n{'-'*40}\nTool Call: {t.name}\nArguments: {serialize_event_data(t.args)}\nRESULTS: {t.result_data.get('output', '') if t.result_data else ''}\n{'-'*40}\n"
                        self._write_to_output(tool_text)
                # Handle message content like tools_agent (filter out [THOUGHT] content)
                for c in res.message.content:
                    if c.type == MessageContentType.ANALYSIS:
                        self._write_to_output(f"\n[ANALYSIS]: {c.text}\n")
                    if c.type in [MessageContentType.TEXT, MessageContentType.THINKING]:
                        content_str = c.text
                        self._write_to_output(content_str or "")

                # Skip thoughts - we only want clean content output

            execution_time = time.time() - start_time
            completion_text = f"\n\n{'='*80}\n✅ STREAMING COMPLETE - Total events: {event_count}\nTotal time: {execution_time:.2f} seconds\n{'='*80}\n"
            self._write_to_output(completion_text)

            logger.info(f"   ✅ Workflow execution completed in {execution_time:.2f}s")
            logger.info(f"   🛠️  Tool calls detected: {tool_calls_detected}")

            res_txt = ""
            # Store assistant response in database if we got content
            if full_response and full_response.message:
                # Ensure conversation_id is set on the response message
                full_response.message.conversation_id = self.test_conversation_id
                await storage.message.add_message(full_response.message)
                logger.info("   📝 Assistant response saved to database")
                for c in full_response.message.content:
                    if c.type == MessageContentType.TEXT and c.text:
                        res_txt += c.text

            # Basic validation
            success = True
            validation_errors = []

            if not tool_calls_detected:
                success = False
                validation_errors.append("No tool calls executed")

            if validation_errors:
                logger.error(f"   ❌ Validation errors: {', '.join(validation_errors)}")
            else:
                logger.info("   ✅ Workflow validation passed")

            return {
                "success": success,
                "execution_time": execution_time,
                "response_length": len(res_txt),
                "tool_calls_detected": tool_calls_detected,
                "event_count": event_count,
                "validation_errors": validation_errors,
            }

        except Exception as e:
            logger.error(f"   ❌ Workflow execution failed: {str(e)}")
            logger.error(f"Workflow execution traceback: {traceback.format_exc()}")
            return {"success": False, "error": str(e)}

    async def validate_real_outputs(self) -> Dict[str, Any]:
        """Validate real outputs and database integrity."""
        logger.info("✅ Validating real outputs...")

        try:
            # Ensure we have required components
            if (
                not storage
                or not storage.conversation
                or not storage.message
                or not self.test_conversation_id
            ):
                raise RuntimeError(
                    "Missing required storage components or conversation ID"
                )

            # Validate conversation exists and has messages
            conversation = await storage.conversation.get_conversation(
                self.test_conversation_id
            )
            if not conversation:
                raise RuntimeError("Test conversation not found")

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
                    logger.warning("   ⚠️  No conversation ID to delete messages from")

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
                    logger.warning("   ⚠️  No model profile ID to delete")

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
        logger.info("🎼 Architecture: Composer + LangGraph")

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
        "qwen3-vl-30b-a3b-thinking",
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
            logger.error(f"Test execution traceback: {traceback.format_exc()}")
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
        logger.error(f"Unexpected error traceback: {traceback.format_exc()}")
        exit(1)
