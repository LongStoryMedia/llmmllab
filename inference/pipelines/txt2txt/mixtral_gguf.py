import datetime
import logging
import torch
from llama_cpp import Llama
from transformers import AutoTokenizer

from ..helpers import get_content, get_role
from models import ChatReq, MessageContent, MessageContentType, MessageRole, Model, Message, ChatResponse
from typing import Any, Generator, List, final
from ..base_pipeline import BasePipeline


class MixtralGGUFPipe(BasePipeline):
    """
    Pipeline class for Mixtral GGUF models using the llama-cpp-python library.
    Provides fast model loading and efficient inference for quantized Mixtral models.
    """

    def __init__(self, model_definition: Model):
        """Initialize a MixtralGGUFPipe instance."""
        self.model_def: Model = model_definition
        self.logger: logging.Logger = logging.getLogger(__name__)

        # Get the details dict to properly access attributes
        details_dict = model_definition.details.model_dump()

        # Log the entire details dict for debugging
        self.logger.info(f"Model details: {details_dict}")

        # Access gguf_file from the details dictionary
        gguf_file = details_dict.get('gguf_file')
        self.logger.info(f"Loading Mixtral GGUF model via llama-cpp-python: {gguf_file}")

        # Ensure model details for GGUF are provided
        if not (self.model_def.details and self.model_def.model and details_dict.get('parent_model')):
            raise ValueError("Model definition for MixtralGGUFPipe must include details for 'gguf_file' and 'parent_model'.")

        # Get the GGUF file path - ensure it's a valid file path
        if gguf_file:
            gguf_file_path = gguf_file
        else:
            # Fallback to model name if no explicit file path
            gguf_file_path = model_definition.model

        # Determine n_gpu_layers based on available resources
        import os
        import multiprocessing

        # Check for CUDA availability
        cuda_available = torch.cuda.is_available()
        cuda_device_count = torch.cuda.device_count() if cuda_available else 0

        # Use all GPU layers if CUDA is available, otherwise use CPU
        n_gpu_layers = -1 if cuda_available else 0  # -1 means all layers

        # Log GPU information
        if cuda_available:
            self.logger.info(f"CUDA is available with {cuda_device_count} device(s)")
            self.logger.info(f"GPU 0: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            self.logger.warning("CUDA is NOT available! Using CPU only, which will be much slower.")

        # Get number of CPU cores for threading (only relevant if using CPU)
        n_threads = min(8, max(4, multiprocessing.cpu_count() // 2)) if n_gpu_layers == 0 else 4

        # If MIXTRAL_GGUF_CTX_SIZE env var is set, use it for context size
        context_size = int(os.environ.get("MIXTRAL_GGUF_CTX_SIZE", "8192"))

        # Verify the GGUF file exists and is valid
        import os.path
        if gguf_file_path is None:
            raise ValueError("Model definition must specify 'gguf_file' for Mixtral GGUF models.")

        # Log model info for debugging
        self.logger.info(f"Model ID: {self.model_def.id}")
        self.logger.info(f"Model path from config: {gguf_file_path}")
        self.logger.info(f"Model path absolute: {os.path.abspath(gguf_file_path)}")
        self.logger.info(f"Path exists: {os.path.exists(gguf_file_path)}")

        # Check if path exists and is a valid file
        if not os.path.exists(gguf_file_path):
            raise FileNotFoundError(f"GGUF file does not exist: {gguf_file_path}")

        if not os.path.isfile(gguf_file_path):
            raise ValueError(f"Path exists but is not a file: {gguf_file_path}")

        # Check file size
        file_size = os.path.getsize(gguf_file_path)
        if file_size < 1_000_000:  # Less than 1MB is suspicious for a quantized model
            raise ValueError(f"GGUF file is too small ({file_size} bytes), likely a placeholder: {gguf_file_path}")

        # Log the file path we're actually using
        self.logger.info(f"Using GGUF file path: {gguf_file_path} (size: {file_size/1_000_000:.2f} MB)")

        # Load the GGUF model using llama-cpp-python
        # This is extremely fast due to memory-mapping
        try:
            # First try with standard parameters
            self.model = Llama(
                model_path=gguf_file_path,  # Path to the GGUF file
                n_ctx=8192,                 # Reduced context length to avoid memory issues
                n_gpu_layers=n_gpu_layers,  # Offload layers to GPU (-1 means all)
                n_threads=n_threads,        # Number of threads to use
                use_mlock=True,             # Use mlock to keep model in memory
                verbose=True,               # Enable detailed loading info to debug GPU usage
                seed=42,                    # Set seed for reproducibility
                n_batch=512                 # Larger batch size for better GPU utilization
            )
        except Exception as e:
            # If first attempt fails, try with more conservative parameters
            self.logger.warning(f"First attempt to load model failed: {str(e)}")
            self.logger.info("Trying again with more conservative parameters...")

            try:
                self.model = Llama(
                    model_path=gguf_file_path,
                    n_ctx=4096,             # Smaller context
                    n_gpu_layers=1,         # Only offload 1 layer to GPU to test
                    n_threads=n_threads,
                    use_mlock=True,
                    verbose=True,
                    seed=42
                )
            except Exception as e2:
                self.logger.error(f"Second attempt failed: {str(e2)}")
                raise ValueError(f"Could not load GGUF model after multiple attempts. Original error: {str(e)}, Second error: {str(e2)}")

        # Log model loading details
        gpu_info = "all layers on GPU" if n_gpu_layers == -1 else f"{n_gpu_layers} layers on GPU"
        self.logger.info(f"Loaded model with context size: {context_size}, threads: {n_threads}, {gpu_info}")

        # Load the original Hugging Face tokenizer for accurate chat templating
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_def.details.parent_model
        )
        self.logger.info(f"Mixtral GGUF model '{self.model_def.name}' loaded successfully.")

    def __del__(self) -> None:
        """
        Clean up resources used by the MixtralGGUFPipe.
        """
        try:
            print(f"MixtralGGUFPipe for {self.model_def.name}: Cleanup initiated")
            if hasattr(self, 'model'):
                # llama-cpp-python models should have their resources cleaned up
                del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error cleaning up MixtralGGUFPipe resources: {str(e)}")

    def run(self, req: ChatReq, load_time: float) -> Generator[ChatResponse, Any, None]:
        """
        Run the Mixtral GGUF model to process text messages.
        """
        messages = req.messages
        start_time = datetime.datetime.now(tz=datetime.timezone.utc)
        message_dicts = []

        self.logger.info(f"Running Mixtral GGUF model with {len(messages)} messages")

        for m in messages:
            cntnt = []
            for c in m.content:
                if c.text is not None:
                    cntnt.append(c.text)
            message_dicts.append({
                "role": get_role(m.role),
                "content": "\n".join(cntnt)
            })

        self.logger.info(f"Running Mixtral GGUF model with messages: {message_dicts}")
        most_recent_message = messages[-1] if messages else None

        prompt_eval_start_time = datetime.datetime.now(tz=datetime.timezone.utc)

        # Apply the chat template to create the final prompt
        text = self.tokenizer.apply_chat_template(
            message_dicts,
            tokenize=False,
            add_generation_prompt=True
        )

        prompt_eval_time = (datetime.datetime.now(tz=datetime.timezone.utc) - prompt_eval_start_time).total_seconds()
        prompt_eval_count = len(self.tokenizer.encode(text))  # Estimate prompt tokens
        eval_count = 0

        # Get generation parameters
        max_tokens = req.options.num_ctx if req.options and req.options.num_ctx is not None else 2048
        final_content = ""

        try:
            full_text = ""
            # Get the properly typed parameters
            temperature = float(req.options.temperature if req.options and req.options.temperature is not None else 0.7)
            top_p = float(req.options.top_p if req.options and req.options.top_p is not None else 0.9)
            top_k = int(req.options.top_k if req.options and req.options.top_k is not None else 40)
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
                stop=stop_tokens  # Stop tokens for Mixtral
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

            # Final content is the complete generated text
            final_content = full_text
            eval_count = len(self.tokenizer.encode(full_text))

        except Exception as e:
            self.logger.error(f"Error running Mixtral GGUF model: {str(e)}")
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
