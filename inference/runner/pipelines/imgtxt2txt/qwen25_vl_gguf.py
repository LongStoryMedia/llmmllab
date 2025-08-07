import os
import logging
import torch
import datetime
from typing import Any, Generator
from llama_cpp import Llama  # type: ignore # pylint: disable=E0401
from llama_cpp.llama_chat_format import Qwen25VLChatHandler  # type: ignore # pylint: disable=E0401

from models import (
    ChatResponse,
    Message,
    MessageContent,
    MessageContentType,
    MessageRole,
    Model,
    ChatReq,
)
from ..base_pipeline import BasePipeline

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"


class Qwen25VLGGUFPipe(BasePipeline):
    """
    Pipeline class for Qwen 2.5 Vision Language GGUF models using llama-cpp-python.
    Uses the Qwen25VLChatHandler for proper multimodal support.
    """

    def __init__(self, model_definition: Model):
        """Initialize a Qwen25VLGGUFPipe instance."""
        super().__init__()
        self.model_def: Model = model_definition
        self.logger: logging.Logger = logging.getLogger(__name__)
        self.logger.info(
            f"Loading Qwen 2.5 VL GGUF model: {model_definition.name} (ID: {model_definition.id})"
        )

        # Get file paths
        gguf_path = (
            self.model_def.details.gguf_file
            if self.model_def.details.gguf_file
            else self.model_def.model
        )
        mmproj_path = "/models/qwen2.5-vl-32b-instruct/mmproj-Qwen_Qwen2.5-VL-32B-Instruct-bf16.gguf"

        if not mmproj_path:
            raise ValueError("mmproj_file is required for multimodal models")

        self.logger.info(f"Loading GGUF model from: {gguf_path}")
        self.logger.info(f"Loading mmproj from: {mmproj_path}")

        # Check if files exist
        if not os.path.exists(gguf_path):
            raise FileNotFoundError(f"GGUF model file not found: {gguf_path}")
        if not os.path.exists(mmproj_path):
            raise FileNotFoundError(f"MMProj file not found: {mmproj_path}")

        # Create the Qwen2.5-VL chat handler with mmproj
        self.logger.info("Initializing Qwen2.5-VL chat handler...")
        try:
            chat_handler = Qwen25VLChatHandler(clip_model_path=mmproj_path)
            self.logger.info("Successfully created Qwen2.5-VL chat handler")
        except (ImportError, ValueError, RuntimeError) as e:
            self.logger.error(f"Failed to create chat handler: {e}")
            raise RuntimeError(
                f"Failed to initialize Qwen2.5-VL chat handler: {e}"
            ) from e

        # Load the model with the chat handler
        self.logger.info("Loading Llama model with Qwen2.5-VL chat handler...")
        try:
            self.model = Llama(
                model_path=gguf_path,
                chat_handler=chat_handler,
                n_gpu_layers=-1,
                n_threads=4,
                verbose=True,
                logits_all=False,
                embedding=False,
                n_ctx=96000,
                type_k=1,  # f16 keys instead of f32
                type_v=1,  # f16 values instead of f32
                n_batch=256,
                n_ubatch=128,
                flash_attn=True,
                tensor_split=[0.5, 0.25, 0.25],
                f16_kv=True,
                use_mlock=False,
                use_mmap=True,
                numa=True,
            )
            self.logger.info("Successfully loaded Qwen 2.5 VL model with chat handler")
        except (ImportError, RuntimeError, ValueError, OSError) as e:
            self.logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Failed to load Qwen2.5-VL model: {e}") from e

    def __del__(self) -> None:
        """Clean up resources used by the Qwen25VLGGUFPipe."""
        try:
            if hasattr(self, "model_def") and hasattr(self.model_def, "name"):
                self.logger.info(
                    f"Cleaning up resources for Qwen 2.5 VL GGUF model: {self.model_def.name}"
                )

            # Free the model resources
            if hasattr(self, "model") and self.model is not None:
                self.model = None

            # Force garbage collection
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except (RuntimeError, AttributeError) as e:
            print(f"Error cleaning up Qwen25VLGGUFPipe resources: {str(e)}")
        except (ValueError, TypeError, IOError) as e:
            print(f"Unexpected error cleaning up Qwen25VLGGUFPipe resources: {str(e)}")

    def run(self, req: ChatReq) -> Generator[ChatResponse, Any, None]:
        """
        Process the input request and generate a response using the Qwen 2.5 VL GGUF model.

        Args:
            req (ChatReq): The chat request containing messages, model parameters, and other settings.

        Yields:
            ChatResponse: Streaming response chunks.
        """
        load_time = 0.0  # For backward compatibility
        start_time = datetime.datetime.now(tz=datetime.timezone.utc)

        # Extract messages and params from the request
        messages = req.messages
        params = req.options

        # Convert to OpenAI format messages
        formatted_messages = []
        for message in messages:
            role = message.role.value.lower()  # Convert enum to string
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

        self.logger.info(f"Formatted {len(formatted_messages)} messages for processing")
        self.logger.debug(f"Messages: {formatted_messages}")

        # Extract generation parameters
        temperature = (
            params.temperature if params and params.temperature is not None else 0.7
        )
        top_p = params.top_p if params and params.top_p is not None else 0.95
        top_k = params.top_k if params and params.top_k is not None else 40
        max_tokens = (
            params.num_predict if params and params.num_predict is not None else 1024
        )

        self.logger.info(
            f"Generation parameters: temperature={temperature}, top_p={top_p}, top_k={top_k}, max_tokens={max_tokens}"
        )

        try:
            prompt_eval_start = datetime.datetime.now(tz=datetime.timezone.utc)
            if self.model is None:
                raise RuntimeError("Model is not loaded")

            # Use the high-level chat completion API with the chat handler
            completion = self.model.create_chat_completion(
                messages=formatted_messages,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
                stream=True,  # Always use streaming in the new interface
            )

            prompt_eval_end = datetime.datetime.now(tz=datetime.timezone.utc)
            prompt_eval_duration = (
                prompt_eval_end - prompt_eval_start
            ).total_seconds() * 1000

            most_recent_message = messages[-1] if messages else None
            generated_tokens = 0

            for chunk in completion:
                if (
                    isinstance(chunk, dict)
                    and "choices" in chunk
                    and len(chunk["choices"]) > 0
                ):
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content", "")

                    if content:
                        generated_tokens += 1

                        response_message = Message(
                            role=MessageRole.ASSISTANT,
                            content=[
                                MessageContent(
                                    type=MessageContentType.TEXT, text=content, url=None
                                )
                            ],
                            id=most_recent_message.id if most_recent_message else 999,
                            conversation_id=(
                                most_recent_message.conversation_id
                                if most_recent_message
                                else 999
                            ),
                            created_at=datetime.datetime.now(tz=datetime.timezone.utc),
                            tool_calls=None,
                            thinking=None,
                        )

                        finish_reason = chunk["choices"][0].get("finish_reason", None)
                        is_done = finish_reason is not None

                        current_time = datetime.datetime.now(tz=datetime.timezone.utc)
                        total_duration = (
                            current_time - start_time
                        ).total_seconds() * 1000

                        response = ChatResponse(
                            message=response_message,
                            done=is_done,
                            finish_reason=finish_reason if is_done else None,
                            total_duration=total_duration,
                            load_duration=load_time,
                            prompt_eval_duration=prompt_eval_duration,
                            eval_duration=total_duration
                            - prompt_eval_duration
                            - load_time,
                            context=None,
                            prompt_eval_count=getattr(self.model, "n_tokens", 0),
                            eval_count=generated_tokens,
                            model=self.model_def.model,
                            created_at=datetime.datetime.now(tz=datetime.timezone.utc),
                        )

                        yield response

                        if is_done:
                            break

        except (RuntimeError, ValueError, AttributeError, KeyError, OSError) as e:
            self.logger.error(
                f"Error running Qwen 2.5 VL GGUF model: {str(e)}", exc_info=True
            )

            # Create error response
            error_message = Message(
                role=MessageRole.ASSISTANT,
                content=[
                    MessageContent(
                        type=MessageContentType.TEXT,
                        text=f"Error processing multimodal request: {str(e)}",
                        url=None,
                    )
                ],
                id=messages[-1].id if messages else 999,
                conversation_id=messages[-1].conversation_id if messages else 999,
                created_at=datetime.datetime.now(tz=datetime.timezone.utc),
                tool_calls=None,
                thinking=None,
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
                context=None,
            )

            yield error_response
            raise
