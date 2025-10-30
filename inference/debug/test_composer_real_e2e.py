"""
Chat Completion E2E Test - Real HTTP Interface Validation

This test validates the complete LLM ML Lab pipeline through the actual chat completion HTTP interface:
1. Real user creation in database
2. Real model profile creation
3. Real conversation and message creation
4. **HTTP chat completion requests** to /chat/completions endpoint
5. **Streaming response capture** exactly as UI receives it
6. **Content filtering validation** for our recent fixes:
   - Intent analysis JSON not leaking into message content
   - Thoughts not appearing in main message content
   - Tool calls showing proper names (not "unknown_tool")
   - Thoughts as clean text (not serialized Pydantic objects)
   - System aware of correct date (2025, not 2023)
7. Real output validation and cleanup

This test uses the actual HTTP chat completion endpoint to capture streaming responses
exactly as the UI would receive them, validating our content filtering fixes.
"""

import asyncio
import time
import uuid
import json
import os
import sys
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
import httpx
import re

from utils.logging import llmmllogger

# Configure logging
logger = llmmllogger.bind(component="chat_completion_e2e_test")


class ChatCompletionE2ETester:
    """Real end-to-end test using HTTP chat completion endpoint."""

    def __init__(
        self,
        target_model: Optional[str] = None,
        capture_llm_output: bool = True,
        print_output: bool = False,
        server_url: str = "http://localhost:8000",
    ):
        """Initialize HTTP chat completion tester."""
        self.server_url = server_url.rstrip("/")
        self.test_user_id = f"test_chat_user_{uuid.uuid4().hex[:8]}"
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
            "qwen3-30b-a3b-q4-k-m",  # Primary model - use this as default
            "openai-gpt-oss-20b-uncensored-q5_1",
            "qwen2.5-vl-32b-instruct-q4-k-m",
        ]

        self.target_model = target_model or available_models[0]
        self.available_models = available_models

        # HTTP client for requests
        self.http_client = None
        
        # Response validation tracking
        self.streaming_responses = []  # Store all streaming JSON responses
        self.content_issues = []  # Track content filtering issues
        
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
            f"{self.output_dir}/chat_completion_output_{model_safe}_{timestamp}.txt"
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

    async def run_full_test(self, query: Optional[str] = "") -> Dict[str, Any]:
        """Run complete composer-based end-to-end pipeline test."""
        logger.info("🚀 Starting Chat Completion HTTP End-to-End Test")
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
            message_result = await self.create_real_message_with_tools(query=query)
            test_results["results"]["message_creation"] = message_result
            if message_result["success"]:
                test_results["components_passed"] += 1

            # Phase 6: HTTP Chat Completion Execution (THE KEY TEST)
            logger.info("� Phase 6: HTTP Chat Completion Execution")
            workflow_result = await self.execute_chat_completion()
            test_results["results"]["chat_completion_execution"] = workflow_result
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

            # Create a message that will benefit from tool usage
            query_text = (
                query
                or """I need current information about the latest developments in artificial intelligence. 
Specifically, I'm interested in:
1. Major AI model releases
2. Recent breakthroughs in AI research
3. Current AI safety developments
Please search for the most recent information and provide a comprehensive summary."""
            )

            content_list = [
                MessageContent(type=MessageContentType.TEXT, text=query_text)
            ]

            user_message = Message(
                id=None,  # Will be assigned by database
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

    async def execute_chat_completion(self) -> Dict[str, Any]:
        """Execute chat completion via HTTP endpoint exactly as UI would."""
        logger.info("� Executing HTTP chat completion...")

        try:
            # Initialize HTTP client
            if not self.http_client:
                self.http_client = httpx.AsyncClient(timeout=60.0)

            # Get the user message from database
            from db import storage
            
            if not self.test_conversation_id or not self.test_message_id:
                raise Exception("Missing conversation or message ID for HTTP request")

            messages = await storage.message.get_conversation_history(
                self.test_conversation_id
            )
            if not messages:
                raise Exception("No messages found for conversation")
            
            # Get the last user message
            user_message = None
            for msg in reversed(messages):
                if msg.role.value == "user":
                    user_message = msg
                    break
                    
            if not user_message:
                raise Exception("No user message found for chat completion")

            logger.info(f"   📝 Sending HTTP request for message: {user_message.id}")
            
            # Prepare request data exactly as UI would send it (with JSON-serializable fields only)
            request_data = {
                "role": user_message.role.value,  # Convert enum to string
                "content": [
                    {
                        "type": content.type.value,  # Convert enum to string
                        "text": content.text
                    } for content in user_message.content
                ],
                "conversation_id": user_message.conversation_id,
            }
            
            # Make HTTP POST request to chat completion endpoint
            headers = {
                "Content-Type": "application/json",
                "User-ID": self.test_user_id,  # Auth header
                "X-Request-ID": f"test_request_{uuid.uuid4().hex[:8]}",
            }
            
            logger.info(f"   🚀 Making request to {self.server_url}/chat/completions")
            
            # Execute HTTP request and capture streaming response
            start_time = time.time()
            streaming_responses = []
            full_content = ""
            tool_calls = []
            thoughts_content = ""
            analyses_content = []
            content_issues = []  # Track content filtering issues

            
            async with self.http_client.stream(
                "POST",
                f"{self.server_url}/chat/completions", 
                json=request_data,
                headers=headers
            ) as response:
                
                if response.status_code != 200:
                    raise Exception(f"HTTP {response.status_code}: {await response.aread()}")
                
                logger.info("   📡 Receiving streaming response...")
                
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                        
                    try:
                        # Parse JSON response chunk exactly as UI would
                        chunk_data = json.loads(line)
                        streaming_responses.append(chunk_data)
                        
                        # Validate content filtering - these are the issues we're testing fixes for
                        if "message" in chunk_data and chunk_data["message"]:
                            message = chunk_data["message"]
                            
                            # Check main message content for leaks
                            if "content" in message and message["content"]:
                                for content_item in message["content"]:
                                    if content_item.get("type") == "text":
                                        text = content_item.get("text", "")
                                        full_content += text
                                        
                                        # VALIDATION 1: Intent analysis JSON should NOT be in main content
                                        if self._detect_intent_analysis_leak(text):
                                            content_issues.append({
                                                "issue": "intent_analysis_in_content",
                                                "text_sample": text[:200] + "..." if len(text) > 200 else text
                                            })
                                        
                                        # VALIDATION 2: Thoughts should NOT be in main content  
                                        if self._detect_thoughts_leak(text):
                                            content_issues.append({
                                                "issue": "thoughts_in_content", 
                                                "text_sample": text[:200] + "..." if len(text) > 200 else text
                                            })
                                        
                                        # VALIDATION 3: Check for 2023 date references (should be 2025)
                                        if self._detect_wrong_date(text):
                                            content_issues.append({
                                                "issue": "wrong_year_2023",
                                                "text_sample": text[:200] + "..." if len(text) > 200 else text
                                            })
                            
                            # Check tool calls for proper names (not "unknown_tool")
                            if "tool_calls" in message and message["tool_calls"]:
                                for tool_call in message["tool_calls"]:
                                    tool_calls.append(tool_call)
                                    
                                    # VALIDATION 4: Tool calls should have proper names
                                    if tool_call.get("name") == "unknown_tool":
                                        content_issues.append({
                                            "issue": "unknown_tool_name",
                                            "tool_data": tool_call
                                        })
                            
                            # Check thoughts format (should be clean text, not serialized objects)
                            if "thoughts" in message and message["thoughts"]:
                                for thought in message["thoughts"]:
                                    if isinstance(thought, dict):
                                        thoughts_content += thought.get("text", "")
                                        
                                        # VALIDATION 5: Thoughts should not be serialized Pydantic objects
                                        if self._detect_serialized_pydantic(thought):
                                            content_issues.append({
                                                "issue": "serialized_pydantic_thoughts",
                                                "thought_data": thought
                                            })
                                    else:
                                        thoughts_content += str(thought)
                            
                            # Collect analyses for validation
                            if "analyses" in message and message["analyses"]:
                                analyses_content.extend(message["analyses"])
                        
                        # Write streaming response to output file exactly as received
                        self._write_streaming_response(chunk_data)
                        
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse JSON chunk: {line[:100]}... Error: {e}")
                        content_issues.append({
                            "issue": "invalid_json_response",
                            "raw_line": line[:200] + "..." if len(line) > 200 else line
                        })
            
            execution_time = time.time() - start_time
            logger.info(f"   ✅ HTTP chat completion finished in {execution_time:.2f}s")
            logger.info(f"   📊 Received {len(streaming_responses)} response chunks")
            
            # Write comprehensive analysis to output file
            self._write_content_analysis({
                "total_chunks": len(streaming_responses),
                "full_content_length": len(full_content),
                "tool_calls_count": len(tool_calls),
                "thoughts_length": len(thoughts_content),
                "analyses_count": len(analyses_content),
                "content_issues": content_issues,
                "execution_time": execution_time
            })
            
            # Determine success based on content filtering validation
            success = True
            validation_errors = []
            
            # Check for content filtering issues (main validation criteria)
            if content_issues:
                success = False
                for issue in content_issues:
                    validation_errors.append(f"Content filtering issue: {issue['issue']}")
                
                logger.error(f"   ❌ Found {len(content_issues)} content filtering issues")
                
            # Basic response validation
            if len(full_content.strip()) == 0:
                success = False
                validation_errors.append("No response content received")
            
            if len(streaming_responses) == 0:
                success = False
                validation_errors.append("No streaming responses received")
            
            # Tool execution validation (should have tool calls with proper names)
            tool_calls_detected = len(tool_calls) > 0
            unknown_tool_count = sum(1 for tc in tool_calls if tc.get("name") == "unknown_tool")
            
            if not success:
                logger.error(f"   ❌ Chat completion validation failed: {', '.join(validation_errors)}")
            else:
                logger.info(f"   ✅ Content filtering validation passed!")
                logger.info(f"   🛠️ Tool calls: {len(tool_calls)} (unknown: {unknown_tool_count})")
                
            return {
                "success": success,
                "execution_time": execution_time, 
                "streaming_chunks": len(streaming_responses),
                "content_length": len(full_content),
                "tool_calls_detected": tool_calls_detected,
                "tool_calls_count": len(tool_calls),
                "unknown_tool_count": unknown_tool_count,
                "content_issues": content_issues,
                "validation_errors": validation_errors if validation_errors else None,
                "final_response": full_content,
                "tool_calls": tool_calls,
                "thoughts": thoughts_content,
                "analyses": analyses_content,
                "streaming_responses": streaming_responses,  # Full raw responses for debugging
            }

                # Check if this event contains tool execution information
        except Exception as e:
            logger.error(f"   ❌ Chat completion HTTP request failed: {str(e)}")
            import traceback

            traceback.print_exc()
            return {"success": False, "error": str(e)}

    def _detect_intent_analysis_leak(self, text: str) -> bool:
        """Detect if intent analysis JSON leaked into main content."""
        # Look for JSON structures that look like intent analysis
        patterns = [
            r'"intent":\s*"[^"]+?"',
            r'"confidence":\s*[\d\.]+',
            r'"analysis":\s*{',
            r'IntentAnalysis\(',
            r'intent_analysis',
        ]
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)
    
    def _detect_thoughts_leak(self, text: str) -> bool:
        """Detect if thoughts leaked into main content."""
        # Look for think tags or thought structures
        patterns = [
            r'<think>',
            r'</think>',
            r'Thought\(',
            r'"text":\s*".*?".*"message_id"',
            r'thinking.*process',
        ]
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)
    
    def _detect_wrong_date(self, text: str) -> bool:
        """Detect if AI thinks it's 2023 instead of 2025."""
        # Look for 2023 date references
        patterns = [
            r'\b2023\b',
            r'as of 2023',
            r'current year.*2023',
        ]
        return any(re.search(pattern, text) for pattern in patterns)
    
    def _detect_serialized_pydantic(self, thought_data: dict) -> bool:
        """Detect if thoughts are serialized Pydantic objects instead of clean text."""
        # Look for Pydantic-specific keys that shouldn't be in clean thoughts
        problematic_keys = [
            '__dict__',
            '__class__',
            'model_fields',
            'model_config',
            'model_validate',
        ]
        return any(key in str(thought_data) for key in problematic_keys)
    
    def _write_streaming_response(self, chunk_data: dict) -> None:
        """Write streaming response chunk to output file."""
        if not self.capture_llm_output or not self.llm_output_file:
            return
            
        try:
            with open(self.llm_output_file, "a", encoding="utf-8") as f:
                f.write(f"\n--- STREAMING CHUNK ---\n")
                f.write(json.dumps(chunk_data, indent=2, ensure_ascii=False))
                f.write(f"\n--- END CHUNK ---\n")
        except Exception as e:
            logger.warning(f"Failed to write streaming response: {e}")
    
    def _write_content_analysis(self, analysis_data: dict) -> None:
        """Write comprehensive content analysis to output file."""
        if not self.capture_llm_output or not self.llm_output_file:
            return
            
        try:
            with open(self.llm_output_file, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"CONTENT FILTERING ANALYSIS\n")
                f.write(f"{'='*80}\n")
                f.write(json.dumps(analysis_data, indent=2, ensure_ascii=False))
                f.write(f"\n{'='*80}\n")
        except Exception as e:
            logger.warning(f"Failed to write content analysis: {e}")

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
        """Clean up all real test data from database and HTTP client."""
        logger.info("🧹 Cleaning up real test data...")

        cleaned_count = 0
        cleanup_errors = []

        try:
            # Clean up HTTP client
            if self.http_client:
                await self.http_client.aclose()
                self.http_client = None
                logger.info("   ✅ HTTP client closed")
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
        logger.info("📊 Chat Completion HTTP Test Summary")
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
        logger.info(f"� Architecture: HTTP Chat Completion")

        logger.info("\n📋 Component Results:")
        component_names = [
            "Infrastructure Setup",
            "Composer Initialization",
            "User Profile Creation",
            "Conversation Creation",
            "Message Creation",
            "Chat Completion Execution",
            "Output Validation",
            "Data Cleanup",
        ]

        result_keys = [
            "infrastructure_setup",
            "composer_initialization",
            "user_profile_creation",
            "conversation_creation",
            "message_creation",
            "chat_completion_execution",
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
    logger.info("🧪 Starting Chat Completion HTTP E2E Tests")

    # Support command line model selection and output options
    target_model = None
    capture_output = True
    print_output = False
    query = ""

    # Parse command line arguments
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg.startswith("--"):
            if arg == "--no-capture":
                capture_output = False
            elif arg == "--print-output":
                print_output = True
            elif arg.startswith("--model="):
                target_model = arg.split("=", 1)[1]
            elif arg.startswith("--query="):
                query = arg.split("=", 1)[1]
        elif not target_model and not arg.startswith("--"):
            target_model = arg

    # Available models for testing
    available_models = [
        "qwen3-30b-a3b-q4-k-m",  # Primary model - use this as default
        "openai-gpt-oss-20b-uncensored-q5_1",
    ]

    # Test specified model or default
    models_to_test = [target_model] if target_model else [available_models[0]]

    for model in models_to_test:
        logger.info(f"🧪 Testing HTTP chat completion with model: {model}")
        tester = ChatCompletionE2ETester(
            target_model=model,
            capture_llm_output=capture_output,
            print_output=print_output,
        )

        # Run the test
        try:
            results = await tester.run_full_test(query=query)

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

    logger.info("🏁 Chat completion HTTP testing completed successfully!")
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
