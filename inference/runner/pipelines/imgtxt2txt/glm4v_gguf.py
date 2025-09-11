"""
Pipeline for GLM-4.1V GGUF models.
Clean implementation with only essential methods for public API.
"""

import base64
import datetime
import logging
import requests
from typing import List, Optional, Dict, Any, cast
from llama_cpp import Llama
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


class GLM4VGGUFPipe(BaseLlamaCppCore):
    """
    Pipeline class for GLM-4.1V-9B GGUF model using llama-cpp-python.
    Clean implementation with only essential methods.
    """

    def __init__(self, model: Model, profile: ModelProfile):
        """Initialize a GLM4VGGUFPipe instance."""
        # Initialize with ChatResponse as the expected return type for multimodal
        super().__init__(
            model,
            profile,
            expected_return_type=ChatResponse,
            model_size_category="large",
        )
        self.logger = logging.getLogger(__name__)

        # Validate required model details
        if not (model.details and model.model):
            raise ValueError("Model definition requires model details.")

        self.logger.info(f"Initialized GLM-4.1V GGUF pipeline: {model.name}")

    def _get_gguf_path(self) -> str:
        """Get GGUF file path."""
        return (
            self.model.details.gguf_file
            if self.model.details.gguf_file
            else self.model.model
        )

    def _initialize_llama_cpp_direct(self) -> None:
        """Initialize the Llama model using llama-cpp-python directly."""
        if self.llm is not None:
            return

        gguf_path = self._get_gguf_path()
        self.logger.info(f"Loading GGUF model from: {gguf_path}")

        try:
            self.llm = Llama(
                model_path=gguf_path,
                n_ctx=64000,
                n_gpu_layers=-1,
                n_threads=4,
                seed=42,
                n_batch=256,
                f16_kv=True,
                verbose=True,
            )
            self.logger.info("Successfully loaded GLM-4.1V model")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Failed to load GLM-4.1V model: {e}") from e

    def _convert_image_to_base64_data_uri(self, image_url: str) -> str:
        """Convert an image URL to a base64 data URI format."""
        try:
            if image_url.startswith("http"):
                response = requests.get(image_url, timeout=30)
                response.raise_for_status()
                image_data = response.content

                content_type = response.headers.get("content-type", "")
                if "png" in content_type.lower():
                    format_type = "png"
                elif "jpeg" in content_type.lower() or "jpg" in content_type.lower():
                    format_type = "jpeg"
                elif "webp" in content_type.lower():
                    format_type = "webp"
                else:
                    format_type = "png"
            else:
                with open(image_url, "rb") as f:
                    image_data = f.read()

                if image_url.lower().endswith(".png"):
                    format_type = "png"
                elif image_url.lower().endswith((".jpg", ".jpeg")):
                    format_type = "jpeg"
                elif image_url.lower().endswith(".webp"):
                    format_type = "webp"
                else:
                    format_type = "png"

            base64_encoded = base64.b64encode(image_data).decode("utf-8")
            return f"data:image/{format_type};base64,{base64_encoded}"

        except Exception as e:
            self.logger.error(f"Error converting image to base64: {e}")
            raise

    def _format_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert internal messages to llama-cpp format."""
        formatted_messages = []
        for message in messages:
            role = message.role.value.lower()
            content_list = []

            for content_item in message.content:
                if content_item.type == MessageContentType.TEXT:
                    content_list.append({"type": "text", "text": content_item.text})
                elif content_item.type == MessageContentType.IMAGE and content_item.url:
                    data_uri = self._convert_image_to_base64_data_uri(content_item.url)
                    content_list.append({"type": "image", "url": data_uri})

            formatted_messages.append({"role": role, "content": content_list})
        return formatted_messages

    async def process_messages(
        self,
        messages: List[Message],
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

        # Convert messages to llama-cpp format
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
                messages=cast(Any, formatted_messages),
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
                stream=False,
            )

            # Extract response content
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
        raise NotImplementedError(
            "LangGraph integration not implemented for multimodal pipeline"
        )
