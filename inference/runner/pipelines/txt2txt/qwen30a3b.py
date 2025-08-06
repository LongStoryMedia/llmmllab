import datetime
import logging
from threading import Thread
import torch
from transformers import AutoTokenizer, Qwen3MoeForCausalLM

from ..helpers import get_content, get_dtype, get_role
from models import (
    MessageContent,
    MessageContentType,
    MessageRole,
    Message,
    ChatResponse,
    Model,
    ChatReq,
)
from typing import Any, Generator
from transformers.generation.streamers import TextIteratorStreamer
from ..base_pipeline import BasePipeline


class Qwen30A3BPipe(BasePipeline):
    """
    Pipeline class for Qwen3-30B-A3B model supporting chat functionality with thinking capabilities.
    """

    def __init__(self, model_definition: Model):
        """Initialize the pipeline for Qwen models."""
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.model_def = model_definition

        # Setup quantization and model loading parameters
        quantization_config = self._setup_quantization_config()
        dtype = get_dtype(model_definition)
        self.logger.info(f"Using dtype: {dtype}")

        self.model = Qwen3MoeForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_definition.model,
            torch_dtype=dtype,
            quantization_config=quantization_config,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_definition.model)

        self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=True)

    def __del__(self) -> None:
        """
        Clean up resources used by the Qwen30A3BPipe.
        This method releases GPU memory by freeing GPU cache.
        """
        try:
            if hasattr(self, "model_def") and hasattr(self.model_def, "name"):
                # Use print instead of logger which may be gone during deletion
                print(f"Qwen30A3BPipe for {self.model_def.name}: Cleanup initiated")

            # Force garbage collection and clear CUDA cache
            if hasattr(self, "model") and self.model is not None:
                del self.model
            if hasattr(self, "tokenizer") and self.tokenizer is not None:
                del self.tokenizer

            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except (RuntimeError, AttributeError, ValueError, TypeError) as e:
            # Use a direct print as logger might be gone during deletion
            print(f"Error cleaning up Qwen30A3BPipe resources: {str(e)}")

    def run(self, req: ChatReq) -> Generator[ChatResponse, Any, None]:
        """
        Process the chat request and generate a response using the Qwen3-30B-A3B model.

        Args:
            req (ChatReq): The chat request containing messages, model parameters, and other settings.

        Yields:
            ChatResponse: Streaming response chunks.
        """
        load_time = 0.0  # For backward compatibility
        start_time = datetime.datetime.now(tz=datetime.timezone.utc)
        messages = req.messages
        params = req.options

        # Convert Message objects to format compatible with Qwen's chat template
        message_dicts = []
        for message in messages:
            content_text = ""
            for msg in message.content:
                content = get_content(msg)
                # Only concatenate if it's a string (text content)
                if content["text"] and isinstance(content["text"], str):
                    content_text += content["text"]
            message_dicts.append(
                {"role": get_role(message.role), "content": content_text}
            )

        self.logger.info(f"Running Qwen3-30B-A3B model with messages: {message_dicts}")
        most_recent_message = messages[-1] if messages else None
        stop_reason = "stop"  # Default stop reason
        prompt_eval_count = 0
        prompt_eval_time = 0.0
        eval_count = 0
        full_text = ""
        thinking_content = ""
        final_content = ""

        try:
            # Apply chat template - enable thinking mode by default
            text = self.tokenizer.apply_chat_template(
                message_dicts,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,  # Enables thinking mode
            )

            # Tokenize the input
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)  # type: ignore

            # Get generation parameters from params if provided
            temperature = (
                params.temperature if params and params.temperature is not None else 0.6
            )  # Recommended for thinking mode
            top_p = (
                params.top_p if params and params.top_p is not None else 0.95
            )  # Recommended for thinking mode
            top_k = (
                params.top_k if params and params.top_k is not None else 20
            )  # Recommended
            max_tokens = (
                params.num_predict
                if params and params.num_predict is not None
                else 4096
            )

            # Create text streamer
            streamer = TextIteratorStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True
            )

            # Generation parameters
            generation_kwargs = {
                "streamer": streamer,
                "max_new_tokens": max_tokens,
                "do_sample": True,  # Always use sampling for thinking mode
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": 1.1,
            }

            # Start generation in a separate thread
            generation_result = []

            def generate_and_store_result():
                nonlocal prompt_eval_count, prompt_eval_time
                prompt_eval_start_time = datetime.datetime.now(tz=datetime.timezone.utc)

                # Generate the output
                output = self.model.generate(  # type: ignore
                    **model_inputs, **generation_kwargs
                )  # type: ignore

                prompt_eval_end_time = datetime.datetime.now(tz=datetime.timezone.utc)
                prompt_eval_time = (
                    prompt_eval_end_time - prompt_eval_start_time
                ).total_seconds()
                prompt_eval_count = (
                    prompt_eval_end_time - prompt_eval_start_time
                ).total_seconds()

                # Store output for later processing
                generation_result.append(output)

            # Start the generation thread
            thread = Thread(target=generate_and_store_result)
            thread.start()

            # Stream tokens as they're generated
            strbuilder = []
            for new_text in streamer:
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
                strbuilder.append(new_text)

            # Wait for generation to complete
            thread.join()
            full_text = "".join(strbuilder)

            # Process the output to separate thinking and final content
            try:
                # Get raw output tokens
                output_ids = generation_result[0][0][
                    len(model_inputs.input_ids[0]) :
                ].tolist()

                # Try to find the </think> token (token ID 151668)
                try:
                    # Find the last occurrence of </think> token
                    index = len(output_ids) - output_ids[::-1].index(151668)
                    thinking_content = self.tokenizer.decode(
                        output_ids[:index], skip_special_tokens=True
                    ).strip("\n")
                    final_content = self.tokenizer.decode(
                        output_ids[index:], skip_special_tokens=True
                    ).strip("\n")
                except ValueError:
                    # If </think> token is not found, assume everything is final content
                    index = 0
                    thinking_content = ""
                    final_content = full_text

                # Update token count for metrics
                eval_count = len(output_ids)
            except (ValueError, IndexError, AttributeError) as e:
                self.logger.warning(f"Error parsing thinking content: {str(e)}")
                thinking_content = ""
                final_content = full_text

        except (RuntimeError, ValueError, AttributeError, IndexError) as e:
            self.logger.error(f"Error running Qwen3-30B-A3B model: {str(e)}")
            stop_reason = "error"
            raise
        finally:
            end_time = datetime.datetime.now(tz=datetime.timezone.utc)
            total_duration = (
                end_time - start_time
            ).total_seconds() * 1000  # Convert to milliseconds
            self.logger.info(
                f"Qwen3-30B-A3B model run completed in {total_duration} ms"
            )

            # Add thinking content as a metadata field if available
            metadata = {}
            if thinking_content:
                metadata["thinking"] = thinking_content

            res = ChatResponse(
                done=True,
                finish_reason=stop_reason,
                message=Message(
                    role=MessageRole.ASSISTANT,
                    content=[
                        MessageContent(
                            type=MessageContentType.TEXT,
                            text=(
                                final_content
                                if final_content
                                else "Generation completed, but no message was generated."
                            ),
                            url=None,
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
                load_duration=load_time,
                prompt_eval_count=prompt_eval_count,
                prompt_eval_duration=prompt_eval_time,
                eval_count=eval_count,
                eval_duration=total_duration - load_time - prompt_eval_time,
            )
            yield res
