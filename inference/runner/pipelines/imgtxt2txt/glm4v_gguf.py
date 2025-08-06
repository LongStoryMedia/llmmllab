import base64
import requests
from typing import Any, Dict, Generator
import gc
import logging
import torch
import datetime
from llama_cpp import Llama  # type: ignore # pylint: disable=E0401

from models import (
    ChatReq,
    ChatResponse,
    Message,
    MessageContent,
    MessageContentType,
    MessageRole,
    Model,
)

from ..base_pipeline import BasePipeline
from ..helpers import get_role


class GLM4VGGUFPipe(BasePipeline):
    """
    Pipeline class for GLM-4.1V-9B-Thinking GGUF model using llama-cpp-python.
    Combines image processing capabilities with efficient GGUF quantized inference.
    """

    def __init__(self, model_definition: Model):
        """Initialize a GLM4VGGUFPipe instance."""
        super().__init__()

        self.model_def: Model = model_definition
        self.logger: logging.Logger = logging.getLogger(__name__)
        self.logger.info(
            f"Loading GLM-4.1V GGUF model: {model_definition.name} (ID: {model_definition.id})"
        )

        # Load the GGUF model using llama-cpp-python
        self.model = Llama(
            # Path to the GGUF file
            model_path=(
                self.model_def.details.gguf_file
                if self.model_def.details.gguf_file
                else self.model_def.model
            ),
            n_ctx=64000,  # Default context size if not provided
            n_gpu_layers=-1,  # Use all available GPU layers
            n_threads=4,
            seed=42,
            n_batch=256,
            f16_kv=True,
            verbose=True,
        )

    def __del__(self) -> None:
        """
        Clean up resources used by the GLM4VGGUFPipe.
        """
        try:
            if hasattr(self, "model_def") and hasattr(self.model_def, "name"):
                self.logger.info(
                    f"Cleaning up resources for GLM-4.1V GGUF model: {self.model_def.name}"
                )

            # Free the model resources
            if hasattr(self, "model") and self.model is not None:
                self.model = None

            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except (RuntimeError, AttributeError, ValueError, TypeError) as e:
            print(f"Error cleaning up GLM4VGGUFPipe resources: {str(e)}")

    def _convert_image_to_base64_data_uri(self, image_url: str) -> str:
        """
        Convert an image URL to a base64 data URI format required by llama-cpp-python.

        Args:
            image_url (str): The URL or path to the image

        Returns:
            str: Base64 encoded data URI
        """
        try:
            if image_url.startswith("http"):
                # Download image from URL
                response = requests.get(image_url, timeout=30)
                response.raise_for_status()
                image_data = response.content

                # Determine format from content-type or URL
                content_type = response.headers.get("content-type", "")
                if "png" in content_type.lower():
                    format_type = "png"
                elif "jpeg" in content_type.lower() or "jpg" in content_type.lower():
                    format_type = "jpeg"
                elif "webp" in content_type.lower():
                    format_type = "webp"
                else:
                    # Default to png
                    format_type = "png"

            else:
                # Local file
                with open(image_url, "rb") as f:
                    image_data = f.read()

                # Determine format from file extension
                if image_url.lower().endswith(".png"):
                    format_type = "png"
                elif image_url.lower().endswith((".jpg", ".jpeg")):
                    format_type = "jpeg"
                elif image_url.lower().endswith(".webp"):
                    format_type = "webp"
                else:
                    format_type = "png"

            # Convert to base64
            base64_encoded = base64.b64encode(image_data).decode("utf-8")

            # Create data URI
            data_uri = f"data:image/{format_type};base64,{base64_encoded}"

            self.logger.info(
                f"Converted image to base64 data URI (format: {format_type}, size: {len(base64_encoded)} chars)"
            )
            return data_uri

        except Exception as e:
            self.logger.error(f"Error converting image to base64: {str(e)}")
            raise

    def _format_message_content(self, content: MessageContent) -> Dict[str, Any]:
        """
        Format message content for llama-cpp-python.

        Args:
            content (MessageContent): The message content to format

        Returns:
            Dict[str, Any]: Formatted content for llama-cpp
        """
        if (
            content.type == MessageContentType.IMAGE
            and hasattr(content, "url")
            and content.url
        ):
            data_uri = self._convert_image_to_base64_data_uri(content.url)
            return {"type": "image", "url": data_uri}
        elif content.type == MessageContentType.TEXT:
            return {"type": "text", "text": content.text}
        else:
            raise ValueError(f"Unsupported content type: {content.type}")

    def run(self, req: ChatReq) -> Generator[ChatResponse, Any, None]:
        """
        Process the input request and generate a response using the GLM-4.1V GGUF model.

        Args:
            req (ChatReq): The chat request containing messages, model parameters, and other settings.

        Yields:
            Generator[ChatResponse, Any, None]: A generator yielding chat response chunks.
        """
        load_time = 0.0  # For backward compatibility
        start_time = datetime.datetime.now(tz=datetime.timezone.utc)

        # Extract messages and options from request
        messages = req.messages
        # Access options directly from req.options when needed

        # Format messages for llama-cpp with proper image handling
        formatted_messages = []
        for message in messages:
            formatted_content = [
                self._format_message_content(content) for content in message.content
            ]
            formatted_messages.append(
                {"role": get_role(message.role), "content": formatted_content}
            )

        self.logger.info(
            f"Formatted messages for GGUF: {len(formatted_messages)} messages"
        )

        # Extract generation parameters
        temperature = (
            req.options.temperature
            if req.options and req.options.temperature is not None
            else 0.7
        )
        top_p = (
            req.options.top_p if req.options and req.options.top_p is not None else 0.95
        )
        top_k = (
            req.options.top_k if req.options and req.options.top_k is not None else 40
        )
        max_tokens = (
            req.options.num_predict
            if req.options and req.options.num_predict is not None
            else 1024
        )

        self.logger.info(
            f"Generation parameters: temperature={temperature}, top_p={top_p}, top_k={top_k}, max_tokens={max_tokens}"
        )

        try:
            # Process with llama-cpp's chat API
            prompt_eval_start = datetime.datetime.now(tz=datetime.timezone.utc)
            if self.model is None:
                raise RuntimeError("Model is not loaded properly.")

            completion = self.model.create_chat_completion(
                messages=formatted_messages,
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=int(top_k),
                max_tokens=int(max_tokens),
                stream=True,
            )

            prompt_eval_end = datetime.datetime.now(tz=datetime.timezone.utc)
            prompt_eval_duration = (
                # ms
                prompt_eval_end
                - prompt_eval_start
            ).total_seconds() * 1000

            most_recent_message = messages[-1] if messages else None
            prompt_tokens = self.model.n_tokens
            generated_tokens = 0
            full_text = ""

            for chunk in completion:
                # Extract text from the chunk
                if (
                    isinstance(chunk, dict)
                    and "choices" in chunk
                    and len(chunk["choices"]) > 0
                ):
                    choice = chunk["choices"][0]
                    if "delta" in choice and "content" in choice["delta"]:
                        content = choice["delta"]["content"]
                        generated_tokens += 1
                        full_text += content if content else ""

                        # Create message and response
                        message = Message(
                            role=MessageRole.ASSISTANT,
                            content=[
                                MessageContent(
                                    type=MessageContentType.TEXT, text=content
                                )
                            ],
                            id=most_recent_message.id if most_recent_message else 999,
                            conversation_id=(
                                most_recent_message.conversation_id
                                if most_recent_message
                                else 999
                            ),
                            created_at=datetime.datetime.now(tz=datetime.timezone.utc),
                        )

                        current_time = datetime.datetime.now(tz=datetime.timezone.utc)
                        total_duration = (
                            current_time - start_time
                        ).total_seconds() * 1000

                        response = ChatResponse(
                            message=message,
                            done=False,
                            finish_reason=None,
                            total_duration=total_duration,
                            load_duration=load_time,
                            prompt_eval_duration=prompt_eval_duration,
                            eval_duration=(
                                current_time - prompt_eval_end
                            ).total_seconds()
                            * 1000,
                            prompt_eval_count=prompt_tokens,
                            eval_count=generated_tokens,
                            model=self.model_def.model,
                            created_at=current_time,
                        )

                        yield response

                # Handle completion
                if (
                    isinstance(chunk, dict)
                    and "choices" in chunk
                    and len(chunk["choices"]) > 0
                    and chunk["choices"][0].get("finish_reason")
                ):
                    finish_reason = chunk["choices"][0].get("finish_reason")

                    message = Message(
                        role=MessageRole.ASSISTANT,
                        content=[
                            MessageContent(type=MessageContentType.TEXT, text=full_text)
                        ],
                        id=most_recent_message.id if most_recent_message else 999,
                        conversation_id=(
                            most_recent_message.conversation_id
                            if most_recent_message
                            else 999
                        ),
                        created_at=datetime.datetime.now(tz=datetime.timezone.utc),
                    )

                    current_time = datetime.datetime.now(tz=datetime.timezone.utc)
                    total_duration = (current_time - start_time).total_seconds() * 1000

                    response = ChatResponse(
                        message=message,
                        done=True,
                        finish_reason=finish_reason,
                        total_duration=total_duration,
                        load_duration=load_time,
                        prompt_eval_duration=prompt_eval_duration,
                        eval_duration=(current_time - prompt_eval_end).total_seconds()
                        * 1000,
                        prompt_eval_count=prompt_tokens,
                        eval_count=generated_tokens,
                        model=self.model_def.model,
                        created_at=current_time,
                    )

                    yield response

        except Exception as e:
            self.logger.error(
                f"Error running GLM-4.1V GGUF model: {str(e)}", exc_info=True
            )

            error_message = Message(
                role=MessageRole.ASSISTANT,
                content=[
                    MessageContent(
                        type=MessageContentType.TEXT,
                        text=f"Error processing request: {str(e)}",
                    )
                ],
                id=messages[-1].id if messages else 999,
                conversation_id=messages[-1].conversation_id if messages else 999,
                created_at=datetime.datetime.now(tz=datetime.timezone.utc),
            )

            end_time = datetime.datetime.now(tz=datetime.timezone.utc)
            total_duration = (end_time - start_time).total_seconds() * 1000

            error_response = ChatResponse(
                message=error_message,
                done=True,
                finish_reason="error",
                total_duration=total_duration,
                load_duration=load_time,
                prompt_eval_duration=0,
                eval_duration=0,
                prompt_eval_count=0,
                eval_count=0,
                model=self.model_def.model,
                created_at=datetime.datetime.now(tz=datetime.timezone.utc),
            )

            yield error_response
            raise
