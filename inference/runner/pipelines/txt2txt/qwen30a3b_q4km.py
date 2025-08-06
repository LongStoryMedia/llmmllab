import datetime
import logging
import torch
from transformers import AutoTokenizer
from typing import Generator, Any
from llama_cpp import Llama  # type: ignore # pylint: disable=E0401

from models import (
    MessageContent,
    MessageContentType,
    MessageRole,
    Model,
    Message,
    ChatResponse,
    ChatReq,
)
from ..base_pipeline import BasePipeline
from ..helpers import get_role


class QwenGGUFPipe(BasePipeline):
    """
    Pipeline class for Qwen GGUF models using the ctransformers library.
    Achieves very fast model loading and efficient inference.
    """

    def __init__(self, model_definition: Model):
        """Initialize a QwenGGUFPipe instance."""
        super().__init__()
        self.model_def: Model = model_definition
        self.logger: logging.Logger = logging.getLogger(__name__)

        # Ensure model details for GGUF are provided
        if not (
            model_definition.details
            and model_definition.model
            and model_definition.details.parent_model
        ):
            raise ValueError(
                "Model definition for GGUFPipe must include details for 'gguf_file' and 'parent_model'."
            )

        # Determine n_gpu_layers based on available resources
        import os

        # Log model info for debugging
        self.logger.info(f"Model ID: {self.model_def.id}")

        gguf = (
            model_definition.details.gguf_file
            if model_definition.details.gguf_file
            else model_definition.model
        )

        # Check file size
        file_size = os.path.getsize(gguf)
        if file_size < 1_000_000:  # Less than 1MB is suspicious for a quantized model
            raise ValueError(
                f"GGUF file is too small ({file_size} bytes), likely a placeholder: {gguf}"
            )

        # Log the file path we're actually using
        self.logger.info(
            f"Using GGUF file path: {gguf} (size: {file_size/1_000_000:.2f} MB)"
        )

        # Load the GGUF model using llama-cpp-python
        # This is extremely fast due to memory-mapping

        self.model = Llama(
            model_path=gguf,  # Path to the GGUF file
            n_ctx=40960,  # Context length
            n_gpu_layers=-1,  # Offload layers to GPU (-1 means all)
            n_threads=4,  # Number of threads to use
            use_mlock=True,  # Use mlock to keep model in memory
            verbose=True,  # Enable detailed loading info to debug GPU usage
            n_batch=512,  # Larger batch size for better GPU utilization
            # Offload key/query/value tensors to GPU for more efficient processing
            offload_kqv=True,
            flash_attn=True,
            tensor_split=[0.5, 0.25, 0.25],
        )

        # Load the original Hugging Face tokenizer for accurate chat templating
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_def.details.parent_model
        )
        self.logger.info(
            f"Qwen GGUF model '{self.model_def.name}' loaded successfully."
        )

    def __del__(self) -> None:
        """
        Clean up resources used by the QwenGGUFPipe.
        """
        try:
            print(
                f"QwenGGUFPipe for {self.model_def.name if hasattr(self, 'model_def') else 'unknown'}: Cleanup initiated"
            )
            if hasattr(self, "model"):
                # llama-cpp-python models should have their resources cleaned up
                del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except (AttributeError, RuntimeError) as e:
            print(f"Error cleaning up QwenGGUFPipe resources: {str(e)}")
        except (ValueError, TypeError, IOError) as e:
            print(f"Unexpected error cleaning up QwenGGUFPipe resources: {str(e)}")

    def run(self, req: ChatReq) -> Generator[ChatResponse, Any, None]:
        """
        Process the chat request and generate a response using the Qwen GGUF model.

        Args:
            req (ChatReq): The chat request containing messages, model parameters, and other settings.

        Yields:
            ChatResponse: Streaming response chunks.
        """
        messages = req.messages
        params = req.options
        start_time = datetime.datetime.now(tz=datetime.timezone.utc)
        message_dicts = []

        self.logger.info(
            f"Running Qwen GGUF model with {len(messages)} messages\n {messages}"
        )

        for m in messages:
            cntnt = []
            for c in m.content:
                if c.text is not None:
                    cntnt.append(c.text)
            message_dicts.append(
                {"role": get_role(m.role), "content": "\n".join(cntnt)}
            )

        self.logger.info(f"Running Qwen GGUF model with messages: {message_dicts}")
        most_recent_message = messages[-1] if messages else None

        prompt_eval_start_time = datetime.datetime.now(tz=datetime.timezone.utc)

        # Apply the chat template to create the final prompt
        text = self.tokenizer.apply_chat_template(
            message_dicts,
            tokenize=False,
            add_generation_prompt=True,
            # Enables Qwen's special thinking mode - always enabled
            enable_thinking=True,
        )

        prompt_eval_time = (
            datetime.datetime.now(tz=datetime.timezone.utc) - prompt_eval_start_time
        ).total_seconds()
        # Estimate prompt tokens
        prompt_eval_count = len(self.tokenizer.encode(text))
        eval_count = 0
        # Get generation parameters
        max_tokens = params.num_ctx if params and params.num_ctx is not None else 4096
        final_content = ""

        try:
            full_text = ""
            # Get the properly typed parameters
            temperature = float(
                params.temperature if params and params.temperature is not None else 0.6
            )
            top_p = float(params.top_p if params and params.top_p is not None else 0.95)
            top_k = int(params.top_k if params and params.top_k is not None else 20)
            repeat_penalty = float(
                params.repeat_penalty
                if params and params.repeat_penalty is not None
                else 1.1
            )
            stop_tokens = (
                params.stop
                if params and params.stop is not None and len(params.stop) > 0
                else ["<|im_end|>", "<|endoftext|>"]
            )

            # Use llama-cpp's streaming generator
            streamer = self.model.create_chat_completion(
                messages=message_dicts,
                stream=True,  # Always stream in the new interface
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
                stop=stop_tokens,  # Stop tokens for Qwen
            )

            for chunk in streamer:
                # Access the text from the chunk dictionary - llama-cpp-python returns a dict
                try:
                    if (
                        isinstance(chunk, dict)
                        and "choices" in chunk
                        and len(chunk["choices"]) > 0
                    ):
                        new_text = chunk["choices"][0]["delta"].get("content", "")
                    else:
                        # Fall back to string representation if structure is unexpected
                        self.logger.warning(f"Unexpected chunk format: {chunk}")
                        new_text = str(chunk)
                except (KeyError, AttributeError, ValueError) as e:
                    self.logger.warning(f"Error processing chunk: {str(e)}")
                    new_text = ""
                except (TypeError, IOError) as e:
                    self.logger.warning(f"Unexpected error processing chunk: {str(e)}")
                    new_text = ""
                if isinstance(new_text, str):
                    full_text += new_text

                yield ChatResponse(
                    done=False,
                    message=Message(
                        role=MessageRole.ASSISTANT,
                        content=[
                            MessageContent(
                                type=MessageContentType.TEXT, text=new_text, url=None
                            )
                        ],
                        tool_calls=None,
                        thinking=None,
                        id=most_recent_message.id if most_recent_message else -1,
                        conversation_id=(
                            most_recent_message.conversation_id
                            if most_recent_message
                            else -1
                        ),
                        created_at=datetime.datetime.now(tz=datetime.timezone.utc),
                    ),
                    model=self.model_def.model,
                    created_at=datetime.datetime.now(tz=datetime.timezone.utc),
                    context=None,
                    finish_reason=None,
                    total_duration=None,
                    load_duration=None,
                    prompt_eval_count=None,
                    prompt_eval_duration=None,
                    eval_count=None,
                    eval_duration=None,
                )

            # Set up variables for thinking content extraction
            # We'll store thinking content here if found

            # Post-process to separate thinking content from the final answer
            final_content = full_text
            think_tag = "</think>"
            if think_tag in final_content:
                parts = final_content.split(think_tag, 1)
                if len(parts) > 1:
                    # The thinking part includes the <think> tag, let's clean it up
                    thinking_parts = parts[0].split("<think>", 1)
                    if len(thinking_parts) > 1:
                        thinking = thinking_parts[-1].strip()
                    final_content = parts[1].strip()
                else:
                    final_content = parts[0].strip()

            eval_count = len(self.tokenizer.encode(full_text))

        except (RuntimeError, ValueError, KeyError) as e:
            self.logger.error(f"Error running Qwen GGUF model: {str(e)}")
            raise
        except (ImportError, AttributeError, TypeError, IndexError) as e:
            self.logger.error(f"Unexpected error running Qwen GGUF model: {str(e)}")
            raise RuntimeError("Unexpected error in Qwen GGUF model") from e
        finally:
            end_time = datetime.datetime.now(tz=datetime.timezone.utc)
            total_duration = (end_time - start_time).total_seconds() * 1000

            # Create the final response
            res = ChatResponse(
                done=True,
                finish_reason="stop",
                message=Message(
                    role=MessageRole.ASSISTANT,
                    content=[
                        MessageContent(
                            type=MessageContentType.TEXT, text=final_content, url=None
                        )
                    ],
                    tool_calls=None,
                    thinking=None,
                    id=most_recent_message.id if most_recent_message else -1,
                    conversation_id=(
                        most_recent_message.conversation_id
                        if most_recent_message
                        else -1
                    ),
                    created_at=datetime.datetime.now(tz=datetime.timezone.utc),
                ),
                model=self.model_def.model,
                created_at=datetime.datetime.now(tz=datetime.timezone.utc),
                context=None,
                total_duration=total_duration,
                load_duration=0.0,  # No load time in this implementation
                prompt_eval_count=prompt_eval_count,
                prompt_eval_duration=prompt_eval_time,
                eval_count=eval_count,
                eval_duration=total_duration - prompt_eval_time,
            )
            yield res
