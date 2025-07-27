import datetime
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.streamers import TextIteratorStreamer
import threading

from ..helpers import get_content, get_role
from models import ChatReq, MessageContent, MessageContentType, MessageRole, Model, Message, ChatResponse
from typing import Any, Generator, List, final
from ..base_pipeline import BasePipeline


class MixtralTransformersPipe(BasePipeline):
    """
    Pipeline class for Mixtral models using HuggingFace Transformers.
    Provides model loading and inference for Mixtral models.
    """

    def __init__(self, model_definition: Model):
        """Initialize a MixtralTransformersPipe instance."""
        self.model_def: Model = model_definition
        self.logger: logging.Logger = logging.getLogger(__name__)

        # Get the details dict to properly access attributes
        details_dict = model_definition.details.model_dump()

        # Log the entire details dict for debugging
        self.logger.info(f"Model details: {details_dict}")

        # Use the parent model for loading with transformers
        parent_model = details_dict.get('parent_model')
        if not parent_model:
            raise ValueError("Model definition for MixtralTransformersPipe must include 'parent_model' in details.")

        self.logger.info(f"Loading Mixtral model via transformers: {parent_model}")

        # Determine device based on available resources
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Log GPU information
        if self.device == "cuda":
            self.logger.info(f"CUDA is available with {torch.cuda.device_count()} device(s)")
            self.logger.info(f"GPU 0: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            self.logger.warning("CUDA is NOT available! Using CPU only, which will be much slower.")

        # Load tokenizer first
        self.tokenizer = AutoTokenizer.from_pretrained(parent_model)

        # Load model with appropriate precision based on available hardware
        self.model = AutoModelForCausalLM.from_pretrained(
            parent_model,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            device_map="auto"
        )

        self.logger.info(f"Mixtral Transformers model '{self.model_def.name}' loaded successfully.")

    def __del__(self) -> None:
        """
        Clean up resources used by the MixtralTransformersPipe.
        """
        try:
            print(f"MixtralTransformersPipe for {self.model_def.name}: Cleanup initiated")
            if hasattr(self, 'model'):
                del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error cleaning up MixtralTransformersPipe resources: {str(e)}")

    def run(self, req: ChatReq, load_time: float) -> Generator[ChatResponse, Any, None]:
        """
        Run the Mixtral model to process text messages.
        """
        messages = req.messages
        start_time = datetime.datetime.now(tz=datetime.timezone.utc)
        message_dicts = []
        eval_count = 0

        self.logger.info(f"Running Mixtral model with {len(messages)} messages")

        for m in messages:
            cntnt = []
            for c in m.content:
                if c.text is not None:
                    cntnt.append(c.text)
            message_dicts.append({
                "role": get_role(m.role),
                "content": "\n".join(cntnt)
            })

        self.logger.info(f"Running Mixtral model with messages: {message_dicts}")
        most_recent_message = messages[-1] if messages else None

        prompt_eval_start_time = datetime.datetime.now(tz=datetime.timezone.utc)

        # Apply the chat template to create the final prompt
        text = self.tokenizer.apply_chat_template(
            message_dicts,
            tokenize=False,
            add_generation_prompt=True
        )

        prompt_eval_time = (datetime.datetime.now(tz=datetime.timezone.utc) - prompt_eval_start_time).total_seconds()
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        prompt_eval_count = len(input_ids[0])  # Exact prompt tokens

        # Get generation parameters
        max_tokens = req.options.num_predict if req.options and req.options.num_predict is not None else 2048
        temperature = float(req.options.temperature if req.options and req.options.temperature is not None else 0.7)
        top_p = float(req.options.top_p if req.options and req.options.top_p is not None else 0.9)
        top_k = int(req.options.top_k if req.options and req.options.top_k is not None else 40)
        repetition_penalty = float(req.options.repeat_penalty if req.options and req.options.repeat_penalty is not None else 1.1)
        stop_tokens = req.options.stop if req.options and req.options.stop is not None and len(req.options.stop) > 0 else None

        # Create a streamer for generating tokens
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        # Prepare generation config
        generation_kwargs = dict(
            inputs=input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            streamer=streamer,
            do_sample=temperature > 0.0,
        )

        # Start generation in a separate thread
        threading.Thread(target=self.model.generate, kwargs=generation_kwargs).start()

        # Stream tokens as they're generated
        final_content = ""
        try:
            for new_text in streamer:
                final_content += new_text
                yield ChatResponse(
                    done=False,
                    message=Message(
                        role=MessageRole.ASSISTANT,
                        content=[MessageContent(type=MessageContentType.TEXT, text=new_text)],
                        id=most_recent_message.id if most_recent_message else -1,
                        conversation_id=most_recent_message.conversation_id if most_recent_message else -1,
                        created_at=datetime.datetime.now(tz=datetime.timezone.utc),
                    ),
                    model=self.model_def.model,
                    created_at=datetime.datetime.now(tz=datetime.timezone.utc),
                )

            eval_count = len(self.tokenizer.encode(final_content))

        except Exception as e:
            self.logger.error(f"Error running Mixtral model: {str(e)}")
            raise
        finally:
            end_time = datetime.datetime.now(tz=datetime.timezone.utc)
            total_duration = (end_time - start_time).total_seconds() * 1000

            # Create the final response
            res = ChatResponse(
                done=True,
                finish_reason="stop",
                message=Message(
                    role=MessageRole.ASSISTANT,
                    content=[MessageContent(type=MessageContentType.TEXT, text=final_content)],
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
