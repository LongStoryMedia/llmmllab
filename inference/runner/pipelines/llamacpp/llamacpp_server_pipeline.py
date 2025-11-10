"""
LlamaCppServerPipeline - Direct llama.cpp server integration.

This pipeline replaces llama-cpp-python with direct llama.cpp server management,
providing better performance, compatibility, and feature support.

Features:
- Direct llama.cpp server process management
- OpenAI-compatible API interface via LangChain
- Full feature parity: streaming, tool calling, grammar constraints
- Better memory management and OOM recovery
- Support for all llama.cpp server features
"""

import re
import json
from typing import Any, Dict, List, Optional, Type, Iterator

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools.base import BaseTool
from pydantic import BaseModel

from models import (
    Model,
    ModelProfile,
)
from utils.logging import llmmllogger
from utils.tool_call_types import LangChainToolCall
from runner.pipelines.base import BasePipeline
from runner.server_manager import LlamaCppServerManager
from openai import OpenAI


logger = llmmllogger.bind(component="LlamaCppServerPipeline")


class LlamaCppServerPipeline(BasePipeline):
    """
    llama.cpp server-based pipeline with persistent server management.

    Behaves like the old llama-cpp-python approach:
    - Server starts once during initialization (like loading llama_instance)
    - Server stays running and model stays loaded for fast reuse
    - Server shuts down when pipeline is destroyed
    """

    def __init__(
        self,
        model: Model,
        profile: ModelProfile,
        grammar: Optional[Type[BaseModel]] = None,
        **kwargs,
    ):
        # DEBUG: Add detailed instance creation logging
        import traceback

        call_stack = traceback.extract_stack()[-4:-1]
        call_info = " → ".join(
            [f"{frame.filename.split('/')[-1]}:{frame.lineno}" for frame in call_stack]
        )

        super().__init__(model, profile, grammar, **kwargs)
        self.user_config = kwargs.get("user_config", None)
        self._logger = llmmllogger.bind(
            component=self.__class__.__name__, model=model.name
        )

        # Log each instance creation with stack trace
        self._logger.info(
            f"🏗️ Creating LlamaCppServerPipeline instance #{id(self):x} for {model.name} (called from {call_info})"
        )

        # Create server manager with new architecture
        self.server_manager = LlamaCppServerManager(
            model=self.model,
            profile=self.profile,
            user_config=self.user_config,
        )

        # Initialize server once, just like the old approach loaded llama_instance once
        self._initialize_persistent_server()

        # Initialize additional attributes after server startup
        super().__init__(model=model, profile=profile, grammar=grammar, **kwargs)
        self.openai_client: Optional[OpenAI] = None
        self._server_started = False
        self._bound_tools = (kwargs.get("_bound_tools", []) or []) + (
            kwargs.get("tools", []) or []
        )
        self._tool_choice = kwargs.get("tool_choice", None)  # Initialize tool choice attribute

    def _initialize_persistent_server(self):
        """
        Initialize persistent llama.cpp server (equivalent to old llama_instance initialization).
        Server stays running for the lifetime of this pipeline instance.
        """
        try:
            # DEBUG: Add call stack logging to track multiple calls
            import traceback

            call_stack = traceback.extract_stack()[-4:-1]
            call_info = " → ".join(
                [
                    f"{frame.filename.split('/')[-1]}:{frame.lineno}"
                    for frame in call_stack
                ]
            )

            self._logger.info(
                f"🚀 _initialize_persistent_server() called for instance #{id(self):x} (called from {call_info})"
            )

            # Start the server ONCE - model loads here and stays in memory
            logger.info(
                f"Loading model {self.model.name} into persistent server...",
                component="LlamaCppServerPipeline",
                model=self.model.name,
            )

            success = self.server_manager.start()
            if not success:
                raise RuntimeError(
                    f"Failed to start persistent server for model {self.model.name}"
                )

            self._server_started = True

            # Initialize OpenAI client using new architecture
            self._initialize_openai_client()

            chat_endpoint = self.server_manager.get_api_endpoint("/chat/completions")
            logger.info(
                f"Persistent server ready at {chat_endpoint} - model loaded and cached",
                component="LlamaCppServerPipeline",
                model=self.model.name,
            )

        except Exception as e:
            logger.error(f"Failed to initialize persistent server: {e}")
            raise

    @property
    def _llm_type(self) -> str:
        return "llamacpp_server"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": self.model.name,
            "model_path": self.server_manager.get_gguf_path(),
            "server_port": self.server_manager.port,
            "parameters": (
                self.profile.parameters.model_dump() if self.profile.parameters else {}
            ),
        }

    def _initialize_openai_client(self) -> None:
        """Initialize OpenAI client to connect to our llama.cpp server."""
        try:
            # Use the new base URL without hardcoded /v1
            base_url = self.server_manager.get_api_endpoint("")  # Get base /v1 endpoint

            self.openai_client = OpenAI(
                base_url=base_url,
                api_key="dummy",  # llama.cpp server doesn't require real API key
                max_retries=3,
                timeout=self.server_manager.startup_timeout,
            )

            self._logger.info(f"OpenAI client initialized with base_url: {base_url}")

        except Exception as e:
            self._logger.error(f"Failed to initialize OpenAI client: {e}")
            raise

    def _build_openai_request_params(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Build OpenAI-compatible request parameters from model profile and user config."""
        params = {
            "model": "local-model",  # llama.cpp servers typically use this
            "messages": self._format_messages_for_openai(messages),
        }

        # Add stop sequences
        if stop:
            params["stop"] = stop

        # Temperature from profile or kwargs
        if kwargs.get("temperature") is not None:
            params["temperature"] = kwargs["temperature"]
        elif (
            hasattr(self.profile.parameters, "temperature")
            and self.profile.parameters.temperature is not None
        ):
            params["temperature"] = self.profile.parameters.temperature

        # Max tokens - use OpenAI's preferred parameter names
        max_tokens = kwargs.get("max_tokens")
        if max_tokens is None and hasattr(self.profile.parameters, "max_tokens"):
            max_tokens = self.profile.parameters.max_tokens
        if max_tokens is None and hasattr(self.profile.parameters, "n_predict"):
            max_tokens = self.profile.parameters.n_predict

        if max_tokens is not None:
            # Use max_completion_tokens for newer OpenAI API compatibility
            params["max_completion_tokens"] = max_tokens

        # Top-p sampling
        if kwargs.get("top_p") is not None:
            params["top_p"] = kwargs["top_p"]
        elif (
            hasattr(self.profile.parameters, "top_p")
            and self.profile.parameters.top_p is not None
        ):
            params["top_p"] = self.profile.parameters.top_p

        # Frequency and presence penalties
        if kwargs.get("frequency_penalty") is not None:
            params["frequency_penalty"] = kwargs["frequency_penalty"]
        elif (
            hasattr(self.profile.parameters, "frequency_penalty")
            and self.profile.parameters.frequency_penalty is not None
        ):
            params["frequency_penalty"] = self.profile.parameters.frequency_penalty

        if kwargs.get("presence_penalty") is not None:
            params["presence_penalty"] = kwargs["presence_penalty"]
        elif (
            hasattr(self.profile.parameters, "presence_penalty")
            and self.profile.parameters.presence_penalty is not None
        ):
            params["presence_penalty"] = self.profile.parameters.presence_penalty

        # Seed for reproducibility
        if kwargs.get("seed") is not None:
            params["seed"] = kwargs["seed"]
        elif (
            hasattr(self.profile.parameters, "seed")
            and self.profile.parameters.seed is not None
        ):
            params["seed"] = self.profile.parameters.seed

        # Tools and tool choice
        if self._bound_tools:
            # Convert LangChain tools to OpenAI format
            openai_tools = []
            for tool in self._bound_tools:
                try:
                    tool_def = {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                        },
                    }
                    # Add parameters schema if available
                    if hasattr(tool, "args_schema") and tool.args_schema:
                        try:
                            tool_def["function"][
                                "parameters"
                            ] = tool.args_schema.model_json_schema()
                        except Exception as schema_error:
                            self._logger.warning(
                                f"Failed to serialize schema for tool '{tool.name}': {schema_error}"
                            )
                            # Create a simple fallback schema
                            tool_def["function"]["parameters"] = {
                                "type": "object",
                                "properties": {},
                                "required": [],
                            }

                    openai_tools.append(tool_def)

                except Exception as tool_error:
                    self._logger.error(
                        f"Failed to convert tool '{tool.name}' to OpenAI format: {tool_error}"
                    )
                    continue

            if openai_tools:
                params["tools"] = openai_tools

            # Set tool choice if specified
            tool_choice = kwargs.get("tool_choice")
            if tool_choice:
                params["tool_choice"] = tool_choice
            elif len(openai_tools) == 1:
                # If only one tool, auto-select it
                params["tool_choice"] = {
                    "type": "function",
                    "function": {"name": openai_tools[0]["function"]["name"]},
                }

        # Streaming
        if kwargs.get("stream") is not None:
            params["stream"] = kwargs["stream"]

        # Additional llama.cpp specific parameters
        extra_body = {}

        # Top-k sampling
        if (
            hasattr(self.profile.parameters, "top_k")
            and self.profile.parameters.top_k is not None
        ):
            extra_body["top_k"] = self.profile.parameters.top_k

        # Repetition penalty
        if (
            hasattr(self.profile.parameters, "repeat_penalty")
            and self.profile.parameters.repeat_penalty is not None
        ):
            extra_body["repeat_penalty"] = self.profile.parameters.repeat_penalty

        # Grammar constraints
        if self.grammar:
            try:
                # Convert Pydantic model to JSON schema for grammar
                schema = self.grammar.model_json_schema()
                extra_body["grammar"] = json.dumps(schema)
            except Exception as e:
                self._logger.warning(f"Failed to convert grammar to schema: {e}")

        # Thinking model support - detect if we should enable thinking
        if (
            hasattr(self.profile.parameters, "think")
            and self.profile.parameters.think is not None
        ):
            extra_body["think"] = self.profile.parameters.think
        elif "thinking" in self.model.name.lower():
            # Auto-enable for thinking models
            extra_body["think"] = True

        # Add any extra parameters
        if extra_body:
            params["extra_body"] = extra_body

        return params

    def _format_messages_for_openai(
        self, messages: List[BaseMessage]
    ) -> List[Dict[str, Any]]:
        """Convert LangChain messages to OpenAI format."""
        openai_messages = []

        # Add system message if present in profile
        if self.profile.system_prompt:
            openai_messages.append(
                {"role": "system", "content": self.profile.system_prompt}
            )

        # Convert conversation messages
        for msg in messages:
            if isinstance(msg, HumanMessage):
                openai_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                ai_msg = {"role": "assistant", "content": msg.content}
                # Add tool calls if present
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    ai_msg["tool_calls"] = msg.tool_calls
                openai_messages.append(ai_msg)
            elif isinstance(msg, SystemMessage):
                openai_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, ToolMessage):
                openai_messages.append(
                    {
                        "role": "tool",
                        "content": msg.content,
                        "tool_call_id": getattr(msg, "tool_call_id", "unknown"),
                    }
                )

        return openai_messages

    def _parse_tool_calls_from_content(
        self, content: str
    ) -> tuple[str, List[LangChainToolCall]]:
        """
        Parse tool calls from LlamaCpp text output and clean content.

        Handles XML-wrapped format:
            <tool_call>{"name": "func", "arguments": {...}}</tool_call>

        Returns:
            Tuple of (cleaned_content, tool_calls_list)
        """
        tool_calls: List[LangChainToolCall] = []
        cleaned_content = content

        # Try XML-wrapped format with generic tags
        tool_call_pattern = (
            r"<(?:tool|function)[-_]call>\s*(\{.*?\})\s*</(?:tool|function)[-_]call>"
        )

        matches = re.finditer(tool_call_pattern, content, re.DOTALL | re.IGNORECASE)

        for match in matches:
            try:
                # Parse the JSON inside the tool_call tags
                json_str = match.group(1).strip()
                tool_data = json.loads(json_str)

                # Handle both regular and Hermes-style arguments
                arguments = tool_data.get("arguments", {})

                # If arguments is a string (Hermes format), parse it as JSON
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        self._logger.warning(
                            f"Failed to parse arguments string as JSON: {arguments}"
                        )
                        arguments = {}

                # Convert to LangChain flat format
                tool_call = LangChainToolCall(
                    id=f"call_{len(tool_calls)}",  # Generate ID
                    name=tool_data.get("name", ""),
                    args=arguments,
                    type="tool_call",
                )

                tool_calls.append(tool_call)

                # Remove this tool call from content
                cleaned_content = cleaned_content.replace(match.group(0), "").strip()

            except (json.JSONDecodeError, KeyError) as e:
                self._logger.warning(
                    f"Failed to parse XML tool call: {e}, content: {match.group(1)}"
                )
                continue

        return cleaned_content, tool_calls

    def _calculate_usage_metadata(
        self, prompt_tokens: int, completion_tokens: int
    ) -> UsageMetadata:
        """Calculate usage metadata for response."""
        return UsageMetadata(
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate response using llama.cpp server."""
        if not self.server_manager.is_running():
            raise RuntimeError("llama.cpp server is not running")

        if not self.openai_client:
            raise RuntimeError("OpenAI client not initialized")

        try:
            # Build comprehensive request parameters
            request_params = self._build_openai_request_params(
                messages=messages, stop=stop, stream=False, **kwargs
            )

            # Make the request
            response = self.openai_client.chat.completions.create(**request_params)

            # Extract content and handle thinking models
            content = response.choices[0].message.content or ""

            # Debug: Log response structure
            self._logger.debug(
                f"🔍 Response message attributes: {dir(response.choices[0].message)}"
            )
            if hasattr(response.choices[0].message, "tool_calls"):
                tool_calls = response.choices[0].message.tool_calls
                self._logger.info(
                    f"🔧 OpenAI response has {len(tool_calls) if tool_calls else 0} tool_calls"
                )
                if tool_calls:
                    for i, tc in enumerate(tool_calls):
                        self._logger.debug(f"🔧 Tool call {i}: {type(tc)} - {tc}")
            else:
                self._logger.debug(f"🔍 No tool_calls attribute in OpenAI response")

            # Handle thinking model output - filter <think> tags if needed
            if (
                hasattr(self.profile.parameters, "think")
                and not self.profile.parameters.think
            ):
                # Remove thinking content if think=False
                content = re.sub(
                    r"<think>.*?</think>",
                    "[Thinking content filtered]",
                    content,
                    flags=re.DOTALL,
                )

            # Parse tool calls from content (llama.cpp generates <tool_call> tags)
            self._logger.debug(
                f"🔍 Parsing content for tool calls. Content length: {len(content)}"
            )
            self._logger.debug(f"🔍 Content preview (first 500 chars): {content[:500]}")
            cleaned_content, parsed_tool_calls = self._parse_tool_calls_from_content(
                content
            )

            if parsed_tool_calls:
                self._logger.info(
                    f"🔧 Found {len(parsed_tool_calls)} tool calls in content"
                )
            else:
                self._logger.debug(f"🔍 No tool calls found in content")

            # Create AI message with cleaned content
            ai_message = AIMessage(content=cleaned_content)

            # Handle tool calls - check both OpenAI format and parsed format
            if (
                hasattr(response.choices[0].message, "tool_calls")
                and response.choices[0].message.tool_calls
            ):
                # Convert OpenAI format tool calls to LangChain format
                langchain_tool_calls = []
                for tc in response.choices[0].message.tool_calls:
                    try:
                        args = {}
                        if tc.function.arguments:
                            args = json.loads(tc.function.arguments)

                        # Convert to LangChain dictionary format
                        langchain_tool_call = {
                            "name": tc.function.name,
                            "args": args,
                            "id": tc.id,
                            "type": "function",
                        }
                        langchain_tool_calls.append(langchain_tool_call)

                        self._logger.debug(
                            f"🔧 Converted tool call: {langchain_tool_call}"
                        )

                    except Exception as convert_error:
                        self._logger.warning(
                            f"Failed to convert tool call {tc}: {convert_error}"
                        )

                if langchain_tool_calls:
                    ai_message.tool_calls = langchain_tool_calls
                    self._logger.info(
                        f"🔧 Converted {len(langchain_tool_calls)} OpenAI tool calls to LangChain format"
                    )

            elif parsed_tool_calls:
                # Use parsed tool calls from content
                ai_message.tool_calls = parsed_tool_calls
                self._logger.info(
                    f"🔧 Parsed {len(parsed_tool_calls)} tool calls from content"
                )

            # Calculate usage metadata
            usage = response.usage
            usage_metadata = self._calculate_usage_metadata(
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
            )

            # Create generation with metadata
            generation = ChatGeneration(
                message=ai_message,
                generation_info={
                    "model": response.model,
                    "usage": usage_metadata,
                    "finish_reason": response.choices[0].finish_reason,
                },
            )

            return ChatResult(
                generations=[generation], llm_output={"model": response.model}
            )

        except Exception as e:
            self._logger.error(f"Generation failed: {e}")
            raise

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream response using llama.cpp server."""
        if not self.server_manager.is_running():
            raise RuntimeError("llama.cpp server is not running")

        if not self.openai_client:
            raise RuntimeError("OpenAI client not initialized")

        try:
            # Build comprehensive request parameters for streaming
            request_params = self._build_openai_request_params(
                messages=messages, stop=stop, stream=True, **kwargs
            )

            # Make streaming request
            stream = self.openai_client.chat.completions.create(**request_params)

            accumulated_content = ""
            for chunk in stream:
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta
                content = delta.content or ""

                # Accumulate content for thinking model processing
                accumulated_content += content

                # Handle tool calls in delta
                tool_calls = getattr(delta, "tool_calls", None)

                # Create AI message chunk
                ai_chunk = AIMessageChunk(content=content)
                if tool_calls:
                    ai_chunk.tool_calls = tool_calls

                # Create generation chunk
                generation_chunk = ChatGenerationChunk(
                    message=ai_chunk,
                    generation_info={
                        "model": chunk.model,
                        "finish_reason": chunk.choices[0].finish_reason,
                    },
                )

                yield generation_chunk

            # Post-process accumulated content for thinking models (if needed)
            if accumulated_content and (
                (
                    hasattr(self.profile.parameters, "think")
                    and not self.profile.parameters.think
                )
                or (
                    "thinking" in self.model.name.lower()
                    and not kwargs.get("show_thinking", False)
                )
            ):
                # For streaming, we could send a final chunk with filtered content
                # but typically streaming preserves the raw output
                pass

        except Exception as e:
            self._logger.error(f"Streaming failed: {e}")
            raise

    def bind_tools(
        self, tools: List[BaseTool], *, tool_choice: str | None = None, **kwargs: Any
    ) -> "LlamaCppServerPipeline":
        """Bind tools to this pipeline WITHOUT creating a new server instance."""
        # CRITICAL: Do NOT create a new pipeline instance - reuse the existing server!
        # Creating a new instance would start another server, causing duplicate servers.
        self._logger.info(
            f"🔗 Binding {len(tools)} tools to EXISTING pipeline instance #{id(self):x} for {self.model.name}"
        )

        # Store tools on the existing instance instead of creating a new one
        self._bound_tools += tools
        self._tool_choice = tool_choice

        # Apply any additional kwargs to the existing instance
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Return the same instance with tools bound
        return self

    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics from server."""
        if self.server_manager:
            return self.server_manager.get_stats()
        return {}

    def close(self):
        """Clean up resources."""
        try:
            if self.server_manager:
                self.server_manager.stop()
            self._logger.info("LlamaCppServerPipeline closed successfully")
        except Exception as e:
            self._logger.error(f"Error closing pipeline: {e}")

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close()
        except:
            pass  # Ignore errors during cleanup


__all__ = ["LlamaCppServerPipeline"]
