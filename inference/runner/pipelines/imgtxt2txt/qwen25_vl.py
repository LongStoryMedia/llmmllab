"""
Pipeline for Qwen 2.5 Vision Language GGUF models.
Clean implementation with only essential methods for public API.
"""

import os
import logging
import datetime
from typing import List, Optional, Dict, Any, cast
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Qwen25VLChatHandler
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph

from models import (
    ChatResponse,
    Message,
    MessageContent,
    MessageContentType,
    MessageRole,
    Model,
    ModelProfile,
)
from ..llamacpp.base_llamacpp import BaseLlamaCppCore

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"


class Qwen25VLPipeline(BaseLlamaCppCore):
    """
    Pipeline class for Qwen 2.5 Vision Language GGUF models using llama-cpp-python.
    Uses the Qwen25VLChatHandler for proper multimodal support.
    Clean implementation with only essential methods.
    """

    def __init__(self, model: Model, profile: ModelProfile):
        """Initialize a Qwen25VLGGUFPipe instance."""
        # Initialize with ChatResponse as the expected return type for multimodal
        super().__init__(
            model,
            profile,
            expected_return_type=ChatResponse,
            model_size_category="large",
        )
        self.logger = logging.getLogger(__name__)
        self.llm: Optional[Llama] = None

        # Validate required model details
        if not (model.details and model.model):
            raise ValueError("Model definition requires model details.")

        self.logger.info(f"Initialized Qwen 2.5 VL GGUF pipeline: {model.name}")

    def _get_gguf_path(self) -> str:
        """Get GGUF file path."""
        return (
            self.model.details.gguf_file
            if self.model.details.gguf_file
            else self.model.model
        )

    def _initialize_llama_cpp_direct(self) -> None:
        """Initialize the Llama model with multimodal support."""
        if self.llm is not None:
            return

        gguf_path = self._get_gguf_path()
        mmproj_path = "/models/qwen2.5-vl-32b-instruct/mmproj-Qwen_Qwen2.5-VL-32B-Instruct-bf16.gguf"

        # Validate file paths
        if not os.path.exists(gguf_path):
            raise FileNotFoundError(f"GGUF model file not found: {gguf_path}")
        if not os.path.exists(mmproj_path):
            raise FileNotFoundError(f"MMProj file not found: {mmproj_path}")

        self.logger.info(f"Loading GGUF model from: {gguf_path}")
        self.logger.info(f"Loading mmproj from: {mmproj_path}")

        try:
            chat_handler = Qwen25VLChatHandler(clip_model_path=mmproj_path)
            self.llm = Llama(
                model_path=gguf_path,
                chat_handler=chat_handler,
                n_gpu_layers=-1,
                n_threads=4,
                verbose=True,
                logits_all=True,  # enforced
                embedding=False,
                n_ctx=96000,
                type_k=1,
                type_v=1,
                n_batch=256,
                n_ubatch=128,
                flash_attn=True,
                tensor_split=[0.5, 0.25, 0.25],
                f16_kv=True,
                use_mlock=False,
                use_mmap=True,
                numa=True,
            )
            self.logger.info("Successfully loaded Qwen 2.5 VL model")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Failed to load Qwen2.5-VL model: {e}") from e

    def _format_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert internal messages to OpenAI format."""
        formatted_messages = []
        for message in messages:
            role = message.role.value.lower()
            content_list = []

            for content_item in message.content:
                if content_item.type == MessageContentType.TEXT:
                    content_list.append({"type": "text", "text": content_item.text})
                elif content_item.type == MessageContentType.IMAGE:
                    if hasattr(content_item, "url") and content_item.url:
                        content_list.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": content_item.url},
                            }
                        )

            formatted_messages.append({"role": role, "content": content_list})
        return formatted_messages

    async def process_messages(
        self,
        messages: List[Message],
        session_id: Optional[str] = None,  # type: ignore
        tools: Optional[List[BaseTool]] = None,
        is_tool_generation: bool = False,
    ) -> ChatResponse:
        """Process messages and return ChatResponse."""
        # Multimodal pipelines don't currently use tools or tool generation flags
        _ = tools, is_tool_generation  # Acknowledge unused parameters

        start_time = datetime.datetime.now(tz=datetime.timezone.utc)

        # Initialize model if needed
        if self.llm is None:
            self._initialize_llama_cpp_direct()

        assert self.llm is not None, "Failed to initialize LLM"

        # Convert messages to OpenAI format
        formatted_messages = self._format_messages(messages)

        # Prepare generation parameters
        params = self.profile.parameters
        temperature = params.temperature if params and params.temperature else 0.7
        top_p = params.top_p if params and params.top_p else 0.95
        top_k = params.top_k if params and params.top_k else 40
        max_tokens = params.num_predict if params and params.num_predict else 1024

        try:
            # Generate response (non-streaming for simplicity)
            completion = self.llm.create_chat_completion(
                messages=cast(
                    Any, formatted_messages
                ),  # Type cast to handle llama-cpp types
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
                stream=False,
            )

            # Extract response content - handle as dictionary
            response_text = ""
            if isinstance(completion, dict):
                choices = completion.get("choices", [])
                if choices and len(choices) > 0:
                    choice = choices[0]
                    message_data = choice.get("message", {})
                    response_text = message_data.get("content", "")

            # Create response message
            response_message = Message(
                role=MessageRole.ASSISTANT,
                content=[
                    MessageContent(
                        type=MessageContentType.TEXT,
                        text=response_text or "No response generated",
                    )
                ],
            )

            end_time = datetime.datetime.now(tz=datetime.timezone.utc)
            total_duration = (end_time - start_time).total_seconds() * 1000

            return ChatResponse(
                message=response_message,
                done=True,
                finish_reason="stop",
                total_duration=total_duration,
                created_at=end_time,
            )

        except Exception as e:
            self.logger.error(f"Error processing messages: {e}")
            error_message = Message(
                role=MessageRole.ASSISTANT,
                content=[
                    MessageContent(
                        type=MessageContentType.TEXT,
                        text=f"Error processing multimodal request: {str(e)}",
                    )
                ],
            )

            return ChatResponse(
                message=error_message,
                done=True,
                finish_reason="error",
                total_duration=0,
                created_at=datetime.datetime.now(tz=datetime.timezone.utc),
            )

    def create_graph(
        self, tools: Optional[List[BaseTool]] = None
    ) -> CompiledStateGraph:
        """Create a simple state graph for multimodal processing."""
        # Placeholder - implement if LangGraph integration needed
        raise NotImplementedError(
            "LangGraph integration not implemented for multimodal pipeline"
        )
