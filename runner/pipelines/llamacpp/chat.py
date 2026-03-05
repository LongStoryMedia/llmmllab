"""
LangChain ChatOpenAI adapter for llama.cpp integration.

This provides a simple adapter that creates a ChatOpenAI instance connected
to our llama.cpp server and exposes it for use with composer agents.
"""

import json
import os
from typing import Any, Dict, Iterator, List, Optional, Type
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import BaseMessage, AIMessageChunk
from langchain_core.outputs import ChatResult, ChatGenerationChunk
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from runner.models import Model, ModelProfile, ModelProfileType
from runner.pipelines.base import BasePipeline
from runner.server_manager import LlamaCppServerManager
from runner.utils.logging import llmmllogger

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
        profile: ModelProfile,
        grammar: Optional[Type[BaseModel]] = None,
        metadata: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(model, profile, grammar, metadata)
        self.user_config = kwargs.get("user_config", None)
        self._logger = llmmllogger.bind(
            component=self.__class__.__name__, model=model.name
        )

        # Create server manager
        self.server_manager = LlamaCppServerManager(
            model=model,
            profile=profile,
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
            # disable_streaming="tool_calling" makes LangChain fall back to
            # a single non-streaming API call whenever tools are bound.
            # This avoids "Invalid diff: now finding less tool calls!"
            # errors from LangChain's streaming tool-call diff tracker,
            # which are triggered by llama.cpp's GLM 4.5 chat format
            # producing chunk sequences that LangChain cannot reconcile.
            # Trade-off: text responses are not token-streamed when tools
            # are bound (Copilot always sends tools), but tool calling
            # is reliable.  Content still arrives via on_chat_model_end.

            # Resolve max_tokens: profile uses -1 for "unlimited", but the
            # OpenAI SDK requires a positive int or omission.  llama.cpp
            # defaults to ctx_size when max_tokens is not sent, which is what
            # we want.
            profile_max = self.profile.parameters.max_tokens
            max_tokens = profile_max if (profile_max and profile_max > 0) else None

            self.chat_model = ReasoningChatOpenAI(
                base_url=base_url,
                api_key=lambda: "not-needed",  # llama.cpp server doesn't require auth
                model="local-model",  # Standard llama.cpp model name
                max_retries=1,
                timeout=self.server_manager.startup_timeout,
                temperature=self.profile.parameters.temperature or 0.7,
                max_tokens=max_tokens,  # type: ignore[assignment]
                top_p=self.profile.parameters.top_p or 0.9,
                disable_streaming="tool_calling",
                verbose=os.getenv("LOG_LEVEL", "WARNING").lower() == "trace",
                # NOTE: reasoning_effort and seed are intentionally omitted.
                # reasoning_effort is an OpenAI o1/o3-only parameter that
                # llama.cpp does not support.  seed is omitted because -1
                # is not a valid value for the OpenAI SDK and llama.cpp
                # defaults to random when unset.
                extra_body={
                    # Disable thinking mode via the Jinja chat template.
                    # GLM-4.7-Flash's template checks `enable_thinking` and
                    # emits <|assistant|></think> (no thinking) when false, vs
                    # <|assistant|><think> (thinking enabled) when true/default.
                    #
                    # With thinking enabled, the model wastes tokens planning
                    # instead of acting — it generates a brief internal plan
                    # then hits EOS without producing tool_calls or content.
                    # Disabling it forces the model to generate content/tool
                    # calls directly, dramatically improving reliability.
                    #
                    # The key must be `chat_template_kwargs` (NOT a top-level
                    # `thinking` or `enable_thinking`) — only this form is
                    # forwarded by llama.cpp to the Jinja template renderer.
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                metadata={
                    "model_profile": self.profile.name,
                    "task": ModelProfileType(self.profile.type).name,
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

        self._logger.debug(
            f"Streaming with messages: {json.dumps([m.model_dump() for m in messages], indent=4)}"
        )

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
