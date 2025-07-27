import datetime
import logging
from cv2 import repeat
import torch
from llama_cpp import Llama
from transformers import AutoTokenizer, Qwen3MoeForCausalLM
from zmq import has

from ..helpers import get_content, get_role
from models import ChatReq, MessageContent, MessageContentType, MessageRole, Model, Message, ChatResponse
from typing import Any, Generator, List, final
from ..base_pipeline import BasePipeline


class QwenGGUFPipe(BasePipeline):
    """
    Pipeline class for Qwen GGUF models using the ctransformers library.
    Achieves very fast model loading and efficient inference.
    """

    def __init__(self, model_definition: Model):
        """Initialize a QwenGGUFPipe instance."""
        self.model_def: Model = model_definition
        self.logger: logging.Logger = logging.getLogger(__name__)

        # Ensure model details for GGUF are provided
        if not (model_definition.details and model_definition.model and model_definition.details.parent_model):
            raise ValueError("Model definition for GGUFPipe must include details for 'gguf_file' and 'parent_model'.")

        # Determine n_gpu_layers based on available resources
        import os   

        # Check for CUDA availability
        cuda_available = torch.cuda.is_available()
        cuda_device_count = torch.cuda.device_count() if cuda_available else 0

        # Use all GPU layers if CUDA is available, otherwise use CPU
        # n_gpu_layers = -1 if cuda_available else 0  # -1 means all layers

        # Log GPU information
        if cuda_available:
            self.logger.info(f"CUDA is available with {cuda_device_count} device(s)")
            self.logger.info(f"GPU 0: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            self.logger.warning("CUDA is NOT available! Using CPU only, which will be much slower.")

        # Get number of CPU cores for threading (only relevant if using CPU)
        # n_threads = min(8, max(4, multiprocessing.cpu_count() // 2)) if n_gpu_layers == 0 else 4

        # Log model info for debugging
        self.logger.info(f"Model ID: {self.model_def.id}")

        # Check file size
        file_size = os.path.getsize(model_definition.details.gguf_file)
        if file_size < 1_000_000:  # Less than 1MB is suspicious for a quantized model
            raise ValueError(f"GGUF file is too small ({file_size} bytes), likely a placeholder: {model_definition.details.gguf_file}")

        # Log the file path we're actually using
        self.logger.info(f"Using GGUF file path: {model_definition.details.gguf_file} (size: {file_size/1_000_000:.2f} MB)")

        # Load the GGUF model using llama-cpp-python
        # This is extremely fast due to memory-mapping
        self.model = Llama(
            model_path=model_definition.details.gguf_file,  # Path to the GGUF file
            n_ctx=40960,                # Context length
            n_gpu_layers=-1,  # Offload layers to GPU (-1 means all)
            n_threads=4,        # Number of threads to use
            use_mlock=True,             # Use mlock to keep model in memory
            verbose=True,               # Enable detailed loading info to debug GPU usage
            seed=42,                    # Set seed for reproducibility
            n_batch=512,                # Larger batch size for better GPU utilization
            offload_kqv=True            # Offload key/query/value tensors to GPU for more efficient processing
        )

        # Load the original Hugging Face tokenizer for accurate chat templating
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_def.details.parent_model
        )
        self.logger.info(f"Qwen GGUF model '{self.model_def.name}' loaded successfully.")

    def __del__(self) -> None:
        """
        Clean up resources used by the QwenGGUFPipe.
        """
        try:
            print(f"QwenGGUFPipe for {self.model_def.name}: Cleanup initiated")
            if hasattr(self, 'model'):
                # llama-cpp-python models should have their resources cleaned up
                del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error cleaning up QwenGGUFPipe resources: {str(e)}")

    def run(self, req: ChatReq, load_time: float) -> Generator[ChatResponse, Any, None]:
        """
        Run the Qwen GGUF model to process text messages.
        """
        messages = req.messages
        start_time = datetime.datetime.now(tz=datetime.timezone.utc)
        message_dicts = []

        self.logger.info(f"Running Qwen GGUF model with {len(messages)} messages\n {messages}")

        for m in messages:
            cntnt = []
            for c in m.content:
                if c.text is not None:
                    cntnt.append(c.text)
            message_dicts.append({
                "role": get_role(m.role),
                "content": "\n".join(cntnt)
            })

        self.logger.info(f"Running Qwen GGUF model with messages: {message_dicts}")
        most_recent_message = messages[-1] if messages else None

        prompt_eval_start_time = datetime.datetime.now(tz=datetime.timezone.utc)

        # Apply the chat template to create the final prompt
        text = self.tokenizer.apply_chat_template(
            message_dicts,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True  # Enables Qwen's special thinking mode
        )

        prompt_eval_time = (datetime.datetime.now(tz=datetime.timezone.utc) - prompt_eval_start_time).total_seconds()
        prompt_eval_count = len(self.tokenizer.encode(text))  # Estimate prompt tokens
        eval_count = 0
        # Get generation parameters
        max_tokens = req.options.num_ctx if req.options and req.options.num_ctx is not None else 4096
        final_content = ""
        thinking_content = ""

        try:
            full_text = ""
            # Get the properly typed parameters
            temperature = float(req.options.temperature if req.options and req.options.temperature is not None else 0.6)
            top_p = float(req.options.top_p if req.options and req.options.top_p is not None else 0.95)
            top_k = int(req.options.top_k if req.options and req.options.top_k is not None else 20)
            repeat_penalty = float(req.options.repeat_penalty if req.options and req.options.repeat_penalty is not None else 1.1)
            stop_tokens = req.options.stop if req.options and req.options.stop is not None and len(req.options.stop) > 0 else ["<|im_end|>", "<|endoftext|>"]

            # Use llama-cpp's streaming generator
            streamer = self.model.create_completion(
                prompt=text,
                stream=True,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
                stop=stop_tokens  # Stop tokens for Qwen
            )

            for chunk in streamer:
                # Access the text from the chunk dictionary - llama-cpp-python returns a dict
                try:
                    if isinstance(chunk, dict) and "choices" in chunk and len(chunk["choices"]) > 0:
                        new_text = chunk["choices"][0]["text"]
                    else:
                        # Fall back to string representation if structure is unexpected
                        self.logger.warning(f"Unexpected chunk format: {chunk}")
                        new_text = str(chunk)
                except Exception as e:
                    self.logger.warning(f"Error processing chunk: {str(e)}")
                    new_text = ""

                full_text += new_text
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

            # Post-process to separate thinking content from the final answer
            # This is more robust than searching for a specific token ID
            final_content = full_text
            think_tag = "</think>"
            if think_tag in final_content:
                parts = final_content.split(think_tag, 1)
                # The thinking part includes the <think> tag, let's clean it up
                thinking_content = parts[0].split("<think>", 1)[-1].strip()
                final_content = parts[1].strip()

            eval_count = len(self.tokenizer.encode(full_text))

        except Exception as e:
            self.logger.error(f"Error running Qwen GGUF model: {str(e)}")
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
