"""
Chat service implementation for the gRPC server.
"""

from services.hardware_manager import hardware_manager
from config import logger
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from ..proto import inference_pb2, inference_pb2_grpc
import grpc
import os
import sys
import logging
from typing import Dict, Iterator, List

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class ChatService:
    """
    Service for handling chat and text generation requests.
    """

    def __init__(self):
        self.logger = logger
        self.models: Dict[str, AutoModelForCausalLM] = {}
        self.tokenizers: Dict[str, AutoTokenizer] = {}

    def _load_model_if_needed(self, model_name: str):
        """
        Load the model if it's not already loaded.
        """
        if model_name in self.models and model_name in self.tokenizers:
            return self.models[model_name], self.tokenizers[model_name]

        self.logger.info(f"Loading model: {model_name}")

        # Configure quantization for memory efficiency
        try:
            from transformers.utils.quantization_config import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        except ImportError:
            quantization_config = None

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load the model
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                quantization_config=quantization_config,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
        except Exception as e:
            # Fall back to standard loading if quantization fails
            self.logger.error(f"Error loading model with quantization: {e}")
            self.logger.info("Falling back to standard loading...")

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )

        # Save the model and tokenizer
        self.models[model_name] = model
        self.tokenizers[model_name] = tokenizer

        return model, tokenizer

    def _format_chat_messages(self, messages: List[inference_pb2.ChatMessage], tokenizer):
        """
        Format the chat messages for the model.
        """
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({
                "role": msg.role,
                "content": msg.content,
            })
        return formatted_messages

    def chat_stream(self, request: inference_pb2.ChatRequest, context):
        """
        Stream chat responses back to the client.
        """
        try:
            # Load the model if needed
            model_name = request.model_config.name if request.model_config.name else "Qwen/Qwen3-30B-A3B"
            model, tokenizer = self._load_model_if_needed(model_name)

            # Format the chat messages
            messages = self._format_chat_messages(request.messages, tokenizer)

            # Log the request
            self.logger.info(f"Chat request from user {request.user_id}, conversation {request.conversation_id}")
            self.logger.debug(f"Messages: {messages}")

            # Prepare the model input
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True
            )

            # Generate streaming response
            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            # Get generation config from the request
            temperature = request.model_config.temperature if request.model_config.temperature > 0 else 0.7
            max_tokens = request.model_config.max_tokens if request.model_config.max_tokens > 0 else 1000
            top_p = request.model_config.top_p if request.model_config.top_p > 0 else 0.95

            # Get memory usage
            device_id = model.hf_device_map["model.embed_tokens"] if hasattr(model, "hf_device_map") else 0
            device_stats = hardware_manager.get_device_memory(device_id)
            memory_usage = inference_pb2.MemoryUsage(
                allocated_memory=device_stats.allocated_bytes,
                free_memory=device_stats.free_bytes,
                utilization=device_stats.utilization_pct / 100.0,
            )

            # Stream the generated text
            accumulated_text = ""
            for token in model.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                max_new_tokens=max_tokens,
                top_p=top_p,
                streamer=None,  # Will manually yield each token
                use_cache=True,
            ):
                # Decode the token
                token_str = tokenizer.decode(token, skip_special_tokens=True)

                # Add to accumulated text and yield
                accumulated_text += token_str

                yield inference_pb2.ChatResponse(
                    content=token_str,
                    is_complete=False,
                    is_error=False,
                    memory_usage=memory_usage
                )

            # Final message marking completion
            yield inference_pb2.ChatResponse(
                content="",
                is_complete=True,
                is_error=False,
                memory_usage=memory_usage
            )

        except Exception as e:
            self.logger.error(f"Error in chat_stream: {e}")
            yield inference_pb2.ChatResponse(
                content="",
                is_complete=True,
                is_error=True,
                error_message=str(e),
            )
        finally:
            # Clean up resources if needed
            torch.cuda.empty_cache()

    def generate_stream(self, request: inference_pb2.GenerateRequest, context):
        """
        Stream generated text back to the client.
        """
        try:
            # Load the model if needed
            model_name = request.model_config.name if request.model_config.name else "Qwen/Qwen3-30B-A3B"
            model, tokenizer = self._load_model_if_needed(model_name)

            # Log the request
            self.logger.info(f"Generate request from user {request.user_id}")
            self.logger.debug(f"Prompt: {request.prompt}")

            # Prepare the model input
            inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)

            # Get generation config from the request
            temperature = request.model_config.temperature if request.model_config.temperature > 0 else 0.7
            max_tokens = request.model_config.max_tokens if request.model_config.max_tokens > 0 else 1000
            top_p = request.model_config.top_p if request.model_config.top_p > 0 else 0.95

            # Get memory usage
            device_id = model.hf_device_map["model.embed_tokens"] if hasattr(model, "hf_device_map") else 0
            device_stats = hardware_manager.get_device_memory(device_id)
            memory_usage = inference_pb2.MemoryUsage(
                allocated_memory=device_stats.allocated_bytes,
                free_memory=device_stats.free_bytes,
                utilization=device_stats.utilization_pct / 100.0,
            )

            # Stream the generated text
            for token in model.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                max_new_tokens=max_tokens,
                top_p=top_p,
                streamer=None,  # Will manually yield each token
                use_cache=True,
            ):
                # Decode the token
                token_str = tokenizer.decode(token, skip_special_tokens=True)

                yield inference_pb2.GenerateResponse(
                    content=token_str,
                    is_complete=False,
                    is_error=False,
                    memory_usage=memory_usage
                )

            # Final message marking completion
            yield inference_pb2.GenerateResponse(
                content="",
                is_complete=True,
                is_error=False,
                memory_usage=memory_usage
            )

        except Exception as e:
            self.logger.error(f"Error in generate_stream: {e}")
            yield inference_pb2.GenerateResponse(
                content="",
                is_complete=True,
                is_error=True,
                error_message=str(e),
            )
        finally:
            # Clean up resources if needed
            torch.cuda.empty_cache()
