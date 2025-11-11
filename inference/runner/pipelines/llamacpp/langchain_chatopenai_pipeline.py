"""
LangChain ChatOpenAI adapter for llama.cpp integration.

This provides a simple adapter that creates a ChatOpenAI instance connected
to our llama.cpp server and exposes it for use with composer agents.
"""

import json
from typing import Any, Dict, Iterator, List, Optional, Type
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import BaseMessage, AIMessageChunk
from langchain_core.outputs import ChatResult, ChatGenerationChunk
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from models import Model, ModelProfile
from runner.pipelines.base import BasePipeline
from runner.server_manager import LlamaCppServerManager
from utils.logging import llmmllogger, serialize_event_data


logger = llmmllogger.bind(component="LangChainChatOpenAIPipeline")


class ReasoningAwareAIMessageChunk(AIMessageChunk):
    """Extended AIMessageChunk that captures reasoning content."""

    def __init__(self, reasoning_content: str = "", **kwargs):
        super().__init__(**kwargs)
        self.reasoning_content = reasoning_content


class ReasoningCaptureChatOpenAI(ChatOpenAI):
    """Custom ChatOpenAI that captures reasoning_content from delta responses."""

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: dict | None,
    ) -> ChatGenerationChunk | None:
        """Override to capture reasoning_content from delta responses."""
        # logger.debug(serialize_event_data(chunk))
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

            if reasoning_content and isinstance(
                generation_chunk.message, AIMessageChunk
            ):
                # Create enhanced chunk with reasoning content
                enhanced_message: ReasoningAwareAIMessageChunk = generation_chunk.message  # type: ignore[assignment]
                enhanced_message.reasoning_content = reasoning_content
                generation_chunk.message = enhanced_message

        return generation_chunk


class LangChainChatOpenAIPipeline(BasePipeline):
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
        **kwargs,
    ):
        super().__init__(model, profile, grammar)
        self.user_config = kwargs.get("user_config", None)
        self._logger = llmmllogger.bind(
            component=self.__class__.__name__, model=model.name
        )

        # Create server manager
        self.server_manager = LlamaCppServerManager(
            model=self.model,
            profile=self.profile,
            user_config=self.user_config,
        )

        # Initialize ChatOpenAI instance
        self.chat_model: Optional[ReasoningCaptureChatOpenAI] = None
        self._server_started = False

        # Initialize server and ChatOpenAI
        self._initialize_persistent_server()

    def _initialize_persistent_server(self):
        """Initialize llama.cpp server and create ChatOpenAI instance."""
        try:
            self._logger.info(f"Starting server for model {self.model.name}")

            # Start the llama.cpp server
            success = self.server_manager.start()
            if not success:
                raise RuntimeError(
                    f"Failed to start server for model {self.model.name}"
                )

            self._server_started = True

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
            # Get the base URL from server manager
            base_url = self.server_manager.get_api_endpoint("")  # Gets /v1 endpoint

            # Extract model parameters from profile
            params = self._build_chat_model_params()

            # Create ChatOpenAI instance with debug logging
            self.chat_model = ReasoningCaptureChatOpenAI(
                base_url=base_url,
                api_key=lambda: "dummy",  # Use callable to satisfy type requirements
                model="local-model",  # Standard llama.cpp model name
                max_retries=3,
                timeout=self.server_manager.startup_timeout,
                # Note: use_responses_api=True not supported by llama.cpp
                **params,
            )

            self._logger.info(f"ChatOpenAI initialized with base_url: {base_url}")

        except Exception as e:
            self._logger.error(f"Failed to initialize ChatOpenAI: {e}")
            raise

    def _build_chat_model_params(self) -> Dict[str, Any]:
        """Build ChatOpenAI parameters from model profile."""
        params = {}

        profile_params = self.profile.parameters
        if not profile_params:
            return params

        # Map profile parameters to ChatOpenAI parameters
        if (
            hasattr(profile_params, "temperature")
            and profile_params.temperature is not None
        ):
            params["temperature"] = profile_params.temperature

        if (
            hasattr(profile_params, "max_tokens")
            and profile_params.max_tokens is not None
        ):
            params["max_tokens"] = profile_params.max_tokens

        if hasattr(profile_params, "top_p") and profile_params.top_p is not None:
            params["top_p"] = profile_params.top_p

        # Only add parameters that actually exist on ModelParameters
        # Skip frequency_penalty, presence_penalty, n_predict, etc. if not available

        if hasattr(profile_params, "seed") and profile_params.seed is not None:
            params["seed"] = profile_params.seed

        return params

    def get_chat_model(self) -> ReasoningCaptureChatOpenAI:
        """Get the underlying ReasoningCaptureChatOpenAI instance for direct LangChain use."""
        if not self.chat_model:
            raise RuntimeError("ChatOpenAI not initialized")
        return self.chat_model

    def shutdown(self):
        """Shutdown the llama.cpp server."""
        if self._server_started and self.server_manager:
            self._logger.info(f"Shutting down server for {self.model.name}")
            self.server_manager.stop()
            self._server_started = False

    def __del__(self):
        """Cleanup when pipeline is destroyed."""
        self.shutdown()

    @property
    def _llm_type(self) -> str:
        return "langchain_chatopenai_llamacpp"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": self.model.name,
            "model_path": self.server_manager.get_gguf_path(),
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

    def bind_tools(self, tools: list[BaseTool], **kwargs):
        """Bind tools to the chat model with support for additional parameters like tool_choice."""
        if not self.chat_model:
            raise RuntimeError("ChatOpenAI not initialized")
        return self.chat_model.bind_tools(tools, **kwargs)
