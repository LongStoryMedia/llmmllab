import datetime
import logging
from threading import Thread
import torch
from transformers import AutoTokenizer, MixtralForCausalLM

from ..helpers import get_content, get_dtype, get_role
from models import ChatReq, MessageContent, MessageContentType, MessageRole, Model, Message, ChatResponse
from typing import Any, Generator, List
from transformers.generation.streamers import TextIteratorStreamer
from ..base_pipeline import BasePipeline


class Mixtral8x7bInstructPipe(BasePipeline):
    """
    Pipeline class for Mixtral-8x7B-Instruct models supporting chat functionality.
    """

    def __init__(self, model_definition: Model):
        """Initialize a Mixtral8x7bInstructPipe instance."""
        self.model_def: Model = model_definition
        self.logger: logging.Logger = logging.getLogger(__name__)
        self.logger.info(f"Loading Mixtral-8x7B-Instruct model: {model_definition.name} (ID: {model_definition.id})")

        # Setup quantization and model loading parameters
        # quantization_config = self._setup_quantization_config()
        dtype = get_dtype(model_definition)
        self.logger.info(f"Using dtype: {dtype}")

        self.model = MixtralForCausalLM.from_pretrained(
            model_definition.model,
            # torch_dtype=dtype,
            # quantization_config=quantization_config,
            device_map="auto",
            # attn_implementation="flash_attention_2",
            # use_cache=True,  # From config
            load_in_4bit=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_definition.model,
            # use_fast=True,
            # padding_side="left",  # Better for chat models
            # trust_remote_code=True,  # Ensures we use the model's specific tokenizer implementation
        )

        # Uncomment these lines if you want to use torch.compile
        # self.model.generate = torch.compile(self.model.generate, mode="reduce-overhead", fullgraph=True)
        #
        # if self.model.generation_config is not None:
        #     self.model.generation_config.cache_implementation = "static"
        #     torch._dynamo.config.capture_scalar_outputs = True
        #     self.model.forward = torch.compile(self.model.forward, mode="reduce-overhead", fullgraph=True)

    def __del__(self) -> None:
        """
        Clean up resources used by the Mixtral8x7bInstructPipe.
        This method releases GPU memory by freeing GPU cache.
        """
        try:
            if hasattr(self, 'model_def') and hasattr(self.model_def, 'name'):
                # Use print instead of logger which may be gone during deletion
                print(f"Mixtral8x7bInstructPipe for {self.model_def.name}: Cleanup initiated")

            # Force garbage collection and clear CUDA cache
            if hasattr(self, 'model') and self.model is not None:
                del self.model
            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                del self.tokenizer

        except Exception as e:
            # Use a direct print as logger might be gone during deletion
            print(f"Error cleaning up Mixtral8x7bInstructPipe resources: {str(e)}")

    def run(self, req: ChatReq, load_time: float) -> Generator[ChatResponse, Any, None]:
        """
        Run the Mixtral-8x7B-Instruct model to process text messages.

        Args:
            req (ChatReq): The chat request containing messages to process.
            load_time (float): The time taken to load the model.
        Yields:
            ChatResponse: The response from the model.
        """
        messages = req.messages
        start_time = datetime.datetime.now(tz=datetime.timezone.utc)
        # Convert Message objects to format compatible with Mixtral chat template
        message_dicts = []
        for message in messages:
            content_text = ""
            for msg in message.content:
                content_part = get_content(msg)
                # Only concatenate if it's a string (text content)
                if isinstance(content_part, str):
                    content_text += content_part
            message_dicts.append({"role": get_role(message.role), "content": content_text})
        self.logger.info(f"Running Mixtral-8x7B-Instruct model with messages: {message_dicts}")
        most_recent_message = messages[-1] if messages else None
        stop_reason = "stop"  # Default stop reason
        prompt_eval_count = 0
        prompt_eval_time = 0.0
        eval_count = 0
        full_text = ""
        try:
            with torch.inference_mode():
                # Apply chat template and tokenize
                # Using the specific Mixtral instruction format as described in the docs:
                # <s> [INST] Instruction [/INST] Model answer</s> [INST] Follow-up instruction [/INST]
                chat_template_input = self.tokenizer.apply_chat_template(
                    message_dicts,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt"
                ).to(self.model.device)

                eval_count = chat_template_input.shape[1]
                streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

                # Get generation parameters from request if provided
                temperature = req.options.temperature if req.options and req.options.temperature else 0.7
                top_p = req.options.top_p if req.options and req.options.top_p else 0.9
                top_k = req.options.top_k if req.options and req.options.top_k else 50
                max_tokens = req.options.num_predict if req.options and req.options.num_predict else 20

                # Separate generation parameters from input tensors
                generation_kwargs = {
                    "streamer": streamer,
                    "max_new_tokens": max_tokens,
                    "do_sample": True,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "repetition_penalty": 1.1,  # Slightly increased to reduce repetitions
                    "num_beams": 1,
                    "num_return_sequences": 1,
                    "use_cache": True,
                    "return_dict_in_generate": True,
                    "output_scores": True,
                }

                # Start generation in a separate thread
                # We'll store the result of model.generate in a list to access it after the thread finishes
                generation_result = []

                def generate_and_store_result(**generation_kwargs):
                    nonlocal prompt_eval_count, prompt_eval_time
                    prompt_eval_start_time = datetime.datetime.now(tz=datetime.timezone.utc)
                    # The model.generate call returns the full output object
                    output = self.model.generate(chat_template_input, **generation_kwargs)
                    prompt_eval_end_time = datetime.datetime.now(tz=datetime.timezone.utc)
                    prompt_eval_time = (prompt_eval_end_time - prompt_eval_start_time).total_seconds()

                    # Store duration in seconds as count (consistent with the original implementation)
                    prompt_eval_count = (prompt_eval_end_time - prompt_eval_start_time).total_seconds()

                    # Only append the output once
                    generation_result.append(output)

                thread = Thread(target=generate_and_store_result, kwargs=generation_kwargs)
                thread.start()

                strbuilder = []
                for new_text in streamer:
                    yield ChatResponse(
                        done=False,
                        message=Message(
                            role=MessageRole.ASSISTANT,
                            content=[
                                MessageContent(
                                    type=MessageContentType.TEXT,
                                    text=new_text
                                )
                            ],
                            id=most_recent_message.id if most_recent_message else -1,
                            conversation_id=most_recent_message.conversation_id if most_recent_message else -1,
                            created_at=datetime.datetime.now(tz=datetime.timezone.utc),
                        ),
                        model=self.model_def.model,
                        created_at=datetime.datetime.now(tz=datetime.timezone.utc),
                    )
                    strbuilder.append(new_text)

                thread.join()
                full_text = "".join(strbuilder)
                # Now, access the full output from generation_result
                if generation_result:
                    full_output = generation_result[0]
                    # Check if full_output has sequences and get the token count
                    if hasattr(full_output, 'sequences') and len(full_output.sequences) > 0:
                        # Count the generated tokens (excluding input tokens)
                        eval_count = full_output.sequences[0].size(0) - chat_template_input.shape[1]

        except Exception as e:
            self.logger.error(f"Error running Mixtral-8x7B-Instruct model: {str(e)}")
            stop_reason = "error"
            raise
        finally:
            end_time = datetime.datetime.now(tz=datetime.timezone.utc)
            total_duration = (end_time - start_time).total_seconds() * 1000  # Convert to milliseconds
            self.logger.info(f"Mixtral-8x7B-Instruct model run completed in {total_duration} ms")
            res = ChatResponse(
                done=True,
                finish_reason=stop_reason,
                message=Message(
                    role=MessageRole.ASSISTANT,
                    content=[
                        MessageContent(
                            type=MessageContentType.TEXT,
                            text=full_text if full_text else "Generation completed, but no message was generated."
                        )
                    ],
                    id=most_recent_message.id if most_recent_message else -1,
                    conversation_id=most_recent_message.conversation_id if most_recent_message else -1,
                    created_at=datetime.datetime.now(tz=datetime.timezone.utc),
                ),
                model=self.model_def.model,
                created_at=datetime.datetime.now(tz=datetime.timezone.utc),
                total_duration=total_duration,
                load_duration=load_time,
                prompt_eval_count=prompt_eval_count,
                prompt_eval_duration=prompt_eval_time,
                eval_count=eval_count,
                eval_duration=total_duration - load_time - prompt_eval_time,
            )
            yield res
