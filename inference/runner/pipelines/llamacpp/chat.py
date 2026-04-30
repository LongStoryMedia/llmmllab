"""
LangChain ChatOpenAI adapter for llama.cpp integration.

This provides a simple adapter that creates a ChatOpenAI instance connected
to our llama.cpp server and exposes it for use with composer agents.
"""

import json
from typing import Any, Dict, Iterator, List, Optional, Type
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import BaseMessage, AIMessage, AIMessageChunk
from langchain_core.outputs import ChatResult, ChatGenerationChunk
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from models import Model
from runner.pipelines.base import BasePipeline
from runner.server_manager import LlamaCppServerManager
from config import LOG_LEVEL
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="LangChainChatOpenAIPipeline")


class ReasoningAwareAIMessageChunk(AIMessageChunk):
    """Extended AIMessageChunk that captures reasoning content."""

    def __init__(self, reasoning_content: str = "", **kwargs):
        super().__init__(**kwargs)
        self.reasoning_content = reasoning_content


class ReasoningChatOpenAI(ChatOpenAI):
    """Custom ChatOpenAI that captures reasoning_content from delta responses."""

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: dict | None,
    ) -> ChatGenerationChunk | None:
        """Override to capture reasoning_content from delta responses."""
        # Get the standard generation chunk first
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )

        if generation_chunk is None:
            return None

        # Check if any choice has reasoning_content in the delta
        choices = chunk.get("choices", [])
        if choices and len(choices) > 0:
            choice = choices[0]
            delta = choice.get("delta", {})
            reasoning_content = delta.get("reasoning_content", "")
            finish_reason = choice.get("finish_reason")

            if finish_reason:
                logger.debug(
                    "Stream finished",
                    extra={"finish_reason": finish_reason},
                )

            if reasoning_content and isinstance(
                generation_chunk.message, AIMessageChunk
            ):
                # Create enhanced chunk with reasoning content
                enhanced_message: ReasoningAwareAIMessageChunk = generation_chunk.message  # type: ignore[assignment]
                enhanced_message.reasoning_content = reasoning_content
                generation_chunk.message = enhanced_message

        return generation_chunk


class ChatLlamaCppPipeline(BasePipeline):
    """
    Simple adapter that creates a ChatOpenAI instance connected to llama.cpp server.

    This maintains compatibility with our existing pipeline architecture while
    providing access to LangChain's built-in tool calling support.
    """

    def __init__(
        self,
        model: Model,
        grammar: Optional[Type[BaseModel]] = None,
        metadata: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(model, grammar, metadata)
        self.user_config = kwargs.get("user_config", None)
        self._logger = llmmllogger.bind(
            component=self.__class__.__name__, model=model.name
        )

        # Create server manager
        self.server_manager = LlamaCppServerManager(
            model=model,
            user_config=self.user_config,
        )

        # Initialize ChatOpenAI instance
        self.chat_model: Optional[ReasoningChatOpenAI] = None
        self.started = False
        self.metadata = metadata or {}

        # Initialize server and ChatOpenAI
        self._initialize_persistent_server()

    def _initialize_persistent_server(self):
        """Initialize llama.cpp server and create ChatOpenAI instance."""
        try:
            self._logger.info(f"Starting server for model {self.model.name}")
            assert self.server_manager is not None
            # Start the llama.cpp server
            self.started = self.server_manager.start()
            if not self.started:
                raise RuntimeError(
                    f"Failed to start server for model {self.model.name}"
                )

            # Create ChatOpenAI instance pointing to our llama.cpp server
            self._initialize_chat_openai()

            self._logger.info(
                f"LangChain ChatOpenAI pipeline ready for {self.model.name}"
            )

        except Exception as e:
            self._logger.error(f"Failed to initialize server and ChatOpenAI: {e}")
            raise

    def _initialize_chat_openai(self):
        """Initialize ChatOpenAI instance to connect to llama.cpp server."""
        try:
            assert self.server_manager is not None
            # Get the base URL from server manager
            base_url = self.server_manager.get_api_endpoint("")  # Gets /v1 endpoint

            # Extract model parameters from profile
            # params = self._build_chat_model_params()

            # Create ChatOpenAI instance with debug logging
            # NOTE: streaming is enabled even with tools bound.  Previously
            # disable_streaming="tool_calling" was set to work around
            # "Invalid diff: now finding less tool calls!" errors from
            # LangChain's streaming tool-call diff tracker with GLM 4.5.
            # That is no longer needed and disabling streaming causes
            # empty-response bugs: when the model returns only reasoning
            # tokens (stripped by --reasoning-budget 0), the non-streaming
            # path receives content="" which is silently dropped, producing
            # an empty assistant turn that breaks Claude Code.

            # Resolve max_tokens: profile uses -1 for "unlimited", but the
            # OpenAI SDK requires a positive int or omission.  llama.cpp
            # defaults to ctx_size when max_tokens is not sent, which is what
            # we want.
            profile_max = (
                self.model.parameters.max_tokens if self.model.parameters else None
            )
            max_tokens = profile_max if (profile_max and profile_max > 0) else None

            self.chat_model = ReasoningChatOpenAI(
                base_url=base_url,
                api_key=lambda: "not-needed",  # llama.cpp server doesn't require auth
                model="local-model",  # Standard llama.cpp model name
                max_retries=0,  # No SDK-level retries — completion_service handles retry logic
                timeout=600,  # 10 minutes — must exceed prompt-processing time for large contexts
                temperature=(
                    self.model.parameters.temperature if self.model.parameters else None
                )
                or 0.7,
                max_completion_tokens=max_tokens,
                top_p=(self.model.parameters.top_p if self.model.parameters else None)
                or 0.9,
                # Frequency penalty discourages the model from repeating the
                # same tokens.  This is the OpenAI-API equivalent of
                # llama.cpp's repeat_penalty and is applied per-request.
                frequency_penalty=0.3,
                # t_max_predict_ms: server-side generation timeout (4 min).
                # llama.cpp kills the generation after this many ms, even if
                # the HTTP connection is still alive.  This is the last line
                # of defence against zombie requests that survive all other
                # timeout layers.
                # extra_body lands in the HTTP JSON body without OpenAI SDK
                # validation, so llama.cpp-specific fields pass through.
                extra_body={"t_max_predict_ms": 240_000},
                streaming=True,
                verbose=LOG_LEVEL.lower() == "trace",
                metadata={
                    "model_name": self.model.name,
                    "task": self.model.task.value,
                    **(self.metadata or {}),
                },
            )

            self._logger.info(f"ChatOpenAI initialized with base_url: {base_url}")

        except Exception as e:
            self._logger.error(f"Failed to initialize ChatOpenAI: {e}")
            raise

    def get_chat_model(self) -> ReasoningChatOpenAI:
        """Get the underlying ReasoningCaptureChatOpenAI instance for direct LangChain use."""
        if not self.chat_model:
            raise RuntimeError("ChatOpenAI not initialized")
        return self.chat_model

    def shutdown(self):
        """Shutdown the llama.cpp server."""
        if self.started and hasattr(self, "server_manager"):
            self._logger.info(f"Shutting down server for {self.model.name}")
            self.server_manager.stop()
            self.started = False

    def bind_metadata(self, metadata: dict):
        """Bind additional metadata to the pipeline.

        Existing metadata keys will be overwritten if they exist in the new metadata.
        """
        if not self.chat_model:
            raise RuntimeError("ChatOpenAI not initialized")
        if not self.metadata:
            self.metadata = {}

        # Use update() which overwrites existing keys with same names
        self.metadata.update(metadata)

        if not self.chat_model.metadata:
            self.chat_model.metadata = {}  # type: ignore[assignment]
        self.chat_model.metadata.update(metadata)

        return self.chat_model

    def __del__(self):
        """Cleanup when pipeline is destroyed."""
        self.shutdown()

    @property
    def _llm_type(self) -> str:
        return "langchain_chatopenai_llamacpp"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        assert self.server_manager is not None
        return {
            "model_name": self.model.name,
            "server_port": self.server_manager.port,
            "pipeline_type": "langchain_chatopenai",
        }

    # Pass messages through raw - no sanitization
    def _sanitize_messages(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """Pass messages through without modification."""
        return messages

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: List[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs,
    ) -> ChatResult:
        """Generate chat completions given input messages."""
        if not self.chat_model:
            raise RuntimeError("ChatOpenAI not initialized")

        messages = self._sanitize_messages(messages)

        self._logger.debug(
            f"Generating with messages: {json.dumps([m.model_dump() for m in messages], indent=4)}"
        )

        # Use protected method with type ignore for compatibility
        return self.chat_model._generate(  # type: ignore[attr-defined]
            messages=messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        )

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: List[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream chat completions given input messages."""
        if not self.chat_model:
            raise RuntimeError("ChatOpenAI not initialized")

        messages = self._sanitize_messages(messages)

        # Use protected method with type ignore for compatibility
        return self.chat_model._stream(  # type: ignore[attr-defined]
            messages=messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        )

    def bind_tools(self, tools: list, **kwargs):
        """Bind tools to the chat model with support for additional parameters like tool_choice.

        Accepts LangChain BaseTool instances or OpenAI-format tool dicts.
        """
        if not self.chat_model:
            raise RuntimeError("ChatOpenAI not initialized")
        return self.chat_model.bind_tools(tools, **kwargs)
