import datetime
import logging
from threading import Thread
from venv import create
from numpy import full
from sympy import Q
import torch
from transformers import AutoTokenizer

from models import ChatReq, MessageContent, MessageContentType, MessageRole, Model, Message, ChatResponse
from typing import Any, Generator, List
from transformers.models.glm4v import Glm4vForConditionalGeneration, Glm4vProcessor
from transformers.generation.streamers import TextIteratorStreamer
from ..base_pipeline import BasePipeline
from ..helpers import get_content, get_dtype, get_role


class GLM4VPipe(BasePipeline):
    """
    Pipeline class for GLM4V models supporting chat functionality.
    """

    def __init__(self, model_definition: Model):
        """Initialize a GLM4VPipe instance."""
        self.model_def: Model = model_definition
        self.logger: logging.Logger = logging.getLogger(__name__)
        self.logger.info(
            "Loading GLM4V model: %s (ID: %s)", model_definition.name, model_definition.id)

        # Setup quantization and model loading parameters
        quantization_config = self._setup_quantization_config()
        dtype = get_dtype(model_definition)
        self.logger.info(f"Using dtype: {dtype}")

        self.model: Glm4vForConditionalGeneration = Glm4vForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=model_definition.model,
            torch_dtype=dtype,
            quantization_config=quantization_config,
            device_map="auto",
            temperature=0.7,
            top_p=0.9,
            attn_implementation="eager",
        )
        self.processor = Glm4vProcessor.from_pretrained(
            model_definition.model,
            # use_fast=True,
            attn_implementation="eager",
        )
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
            model_definition.model,
            # use_fast=True,
            attn_implementation="eager",
        )

        # self.model.generate = torch.compile(self.model.generate, mode="reduce-overhead", fullgraph=True)

        # if self.model.generation_config is not None:
        #     self.model.generation_config.cache_implementation = "static"
        #     torch._dynamo.config.capture_scalar_outputs = True
        #     self.model.forward = torch.compile(self.model.forward, mode="reduce-overhead", fullgraph=True)

    def __del__(self) -> None:
        """
        Clean up resources used by the GLM4VPipe.
        This method releases GPU memory by freeing GPU cache.
        """
        try:
            if hasattr(self, 'model_def') and hasattr(self.model_def, 'name'):
                self.logger.info(
                    f"GLM4VPipe for {self.model_def.name}: Cleanup initiated")

            # Force garbage collection and clear CUDA cache
            if self.model is not None:
                del self.model
            if self.processor is not None:
                del self.processor
            if self.tokenizer is not None:
                del self.tokenizer

        except Exception as e:
            # Use a direct print as logger might be gone during deletion
            print(f"Error cleaning up GLM4VPipe resources: {str(e)}")

    def run(self, req: ChatReq, load_time: float) -> Generator[ChatResponse, Any, None]:
        """
        Run the GLM4V model to process multimodal messages (text and images).

        Args:
            messages (List[Message]): The list of messages to process.
            load_time (float): The time taken to load the model.
        Yields:
            ChatResponse: The response from the model.
        """
        messages = req.messages
        start_time = datetime.datetime.now(tz=datetime.timezone.utc)
        # Convert Message objects to dictionaries
        message_dicts = [{"role": get_role(message.role), "content": [get_content(
            msg) for msg in message.content]} for message in messages]
        self.logger.info(f"Running GLM4V model with messages: {message_dicts}")
        most_recent_message = messages[-1] if messages else None
        stop_reason = "stop"  # Default stop reason
        prompt_eval_count = 0
        prompt_eval_time = 0.0
        eval_count = 0
        full_text = ""
        p = self.processor if isinstance(
            self.processor, Glm4vProcessor) else self.processor[0]
        try:
            with torch.inference_mode():
                inputs = p.apply_chat_template(
                    message_dicts,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(self.model.device)  # type: ignore
                eval_count = inputs["input_ids"].shape[1]
                streamer = TextIteratorStreamer(
                    self.tokenizer, skip_prompt=True, skip_special_tokens=True)

                # Separate generation parameters from input tensors
                generation_kwargs = {
                    "streamer": streamer,
                    "max_new_tokens": 100000,
                    "do_sample": True,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 50,
                    "repetition_penalty": 1.0,
                    "num_beams": 1,
                    "num_return_sequences": 1,
                    "use_cache": True,
                    "return_dict_in_generate": True,
                    "output_scores": True,
                    "output_attentions": True,  # Optional: include attentions if needed
                    "output_hidden_states": True,  # Optional: include hidden states if needed
                }

                # Start generation in a separate thread
                # We'll store the result of model.generate in a list to access it after the thread finishes
                generation_result = []

                def generate_and_store_result(**generation_kwargs):
                    nonlocal prompt_eval_count, prompt_eval_time
                    prompt_eval_start_time = datetime.datetime.now(
                        tz=datetime.timezone.utc)
                    # The model.generate call returns the full output object
                    output = self.model.generate(**inputs, **generation_kwargs)
                    prompt_eval_end_time = datetime.datetime.now(
                        tz=datetime.timezone.utc)
                    prompt_eval_time = (
                        prompt_eval_end_time - prompt_eval_start_time).total_seconds()

                    # Store duration in seconds as count (this is likely a naming issue)
                    prompt_eval_count = (
                        prompt_eval_end_time - prompt_eval_start_time).total_seconds()

                    # Only append the output once
                    generation_result.append(output)

                thread = Thread(target=generate_and_store_result,
                                kwargs=generation_kwargs)
                thread.start()

                # Remove this duplicate thread creation which causes errors
                # thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
                # thread.start()

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
                            created_at=datetime.datetime.now(
                                tz=datetime.timezone.utc),
                        ),
                        model=self.model_def.model,
                        created_at=datetime.datetime.now(
                            tz=datetime.timezone.utc),
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
                        eval_count = full_output.sequences[0].size(
                            0) - inputs["input_ids"].shape[1]

                # generated_ids = self.model.generate(**inputs, max_new_tokens=5, return_dict_in_generate=True, output_scores=True, streamer=streamer)
                # output_text = self.processor.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
                # print(output_text)
                # return output_text

        except Exception as e:
            self.logger.error(f"Error running GLM4V model: {str(e)}")
            stop_reason = "error"
            raise
        finally:
            end_time = datetime.datetime.now(tz=datetime.timezone.utc)
            # Convert to milliseconds
            total_duration = (end_time - start_time).total_seconds() * 1000
            self.logger.info(
                f"GLM4V model run completed in {total_duration} ms")
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
