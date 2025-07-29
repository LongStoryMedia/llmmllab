from abc import ABC, abstractmethod
from typing import List, Any

import torch
from .helpers import get_dtype
from models import ChatReq, Message, Model


class BasePipeline(ABC):
    """
    Base abstract class for all pipeline implementations.
    All concrete pipeline classes should inherit from this class
    and implement the required run method and __del__ method.
    """

    def __init__(self):
        self.model_def: Model

    @abstractmethod
    def run(self, req: ChatReq, load_time: float) -> Any:
        """
        Process the input messages and generate a response using the loaded model.

        Args:
            messages (List[Message]): The list of messages to process.
            load_time (float): The time taken to load the model.

        Returns:
            Any: The generated output, which could be text, image, or other media
                 depending on the specific pipeline implementation.
        """
        raise NotImplementedError("Subclasses must implement the run method.")

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response for a single prompt.

        Args:
            prompt (str): The text prompt to process
            **kwargs: Additional arguments to pass to the model

        Returns:
            str: The generated text response
        """
        from models import Message, MessageContent, MessageRole, ChatReq, MessageContentType, ModelParameters

        # Convert prompt to ChatReq format
        message = Message(
            role=MessageRole.USER,
            content=[MessageContent(
                type=MessageContentType.TEXT, text=prompt)],
            conversation_id=0  # Use default conversation ID
        )

        # Create request with default parameters
        model_id = self.model_def.id if self.model_def and self.model_def.id else "unknown_model"
        req = ChatReq(
            model=model_id,
            messages=[message],
            stream=False,
            options=ModelParameters(**kwargs) if kwargs else None
        )

        # Call the run method and get response
        response = self.run(req, 0)

        # Handle different response types
        if isinstance(response, str):
            return response

        # Handle generator/streaming response
        if hasattr(response, '__iter__') and hasattr(response, '__next__'):
            # Consume generator and concatenate chunks
            full_text = ""
            try:
                for chunk in response:
                    # Handle ChatResponse objects from QwenGGUFPipe
                    if hasattr(chunk, 'message') and not isinstance(chunk, str):
                        message = getattr(chunk, 'message')
                        if hasattr(message, 'content') and getattr(message, 'content'):
                            content_list = getattr(message, 'content')
                            for content_item in content_list:
                                if hasattr(content_item, 'text') and getattr(content_item, 'text') is not None:
                                    full_text += getattr(content_item, 'text')
                    elif isinstance(chunk, str):
                        full_text += chunk
                return full_text
            except Exception as e:
                # Log the error but continue
                import logging
                logging.getLogger(__name__).warning(
                    f"Error consuming generator: {e}")
                return ""

        # If response is more complex, try to extract text content
        try:
            if hasattr(response, "text"):
                return response.text
            if hasattr(response, "message") and hasattr(response.message, "content"):
                for content in response.message.content:
                    if hasattr(content, "text") and content.text:
                        return content.text
            if hasattr(response, "content") and isinstance(response.content, list):
                for content in response.content:
                    if hasattr(content, "text") and content.text:
                        return content.text
        except Exception as e:
            # Log the error but continue
            import logging
            logging.getLogger(__name__).warning(
                f"Error extracting text from response: {e}")

        # Return string representation as fallback
        return str(response)

    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Generate responses for a batch of prompts.

        Args:
            prompts (List[str]): List of text prompts to process
            **kwargs: Additional arguments to pass to the model

        Returns:
            List[str]: List of generated text responses
        """
        responses = []
        for prompt in prompts:
            # Call generate for each prompt
            response = self.generate(prompt, **kwargs)

            # Handle case where response is a generator (streaming response)
            if hasattr(response, '__iter__') and hasattr(response, '__next__'):
                # Consume generator and concatenate chunks
                full_text = ""
                try:
                    for chunk in response:
                        # Handle ChatResponse objects from QwenGGUFPipe
                        if hasattr(chunk, 'message') and not isinstance(chunk, str):
                            message = getattr(chunk, 'message')
                            if hasattr(message, 'content') and getattr(message, 'content'):
                                content_list = getattr(message, 'content')
                                for content_item in content_list:
                                    if hasattr(content_item, 'text') and getattr(content_item, 'text') is not None:
                                        full_text += getattr(content_item,
                                                             'text')
                        elif isinstance(chunk, str):
                            full_text += chunk
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).error(
                        f"Error consuming generator: {e}")
                responses.append(full_text)
            else:
                # Response is already processed by generate()
                responses.append(response)

        return responses

    @abstractmethod
    def __del__(self) -> None:
        """
        Clean up resources used by the pipeline.
        This method should release GPU memory by moving models to CPU.
        It will be called automatically when the pipeline is about to be destroyed.
        """
        raise NotImplementedError(
            "Subclasses must implement the __del__ method.")

    def _setup_quantization_config(self):
        """
        Set up the quantization configuration based on the model details.

        Args:
            model (Model): The model configuration.

        Returns:
            Dict: The quantization configuration parameters.
        """
        from transformers.utils.quantization_config import BitsAndBytesConfig
        quantization_config = {
            "load_in_8bit": False,
            "load_in_4bit": False,
            "llm_int8_threshold": 6.0,
            "llm_int8_skip_modules": None,
            "llm_int8_enable_fp32_cpu_offload": False,
            "llm_int8_has_fp16_weight": False,
            "bnb_4bit_compute_dtype": None,
            "bnb_4bit_quant_type": "fp4",
            "bnb_4bit_use_double_quant": False,
            "bnb_4bit_quant_storage": torch.uint8,
        }

        if self.model_def.details is not None and self.model_def.details.quantization_level is not None:
            if self.model_def.details.quantization_level.lower().startswith(("q4", "int4", "nf4", "fp4", "q2", "int2")):
                quantization_config.update({
                    "load_in_4bit": True,
                    "bnb_4bit_quant_type": "fp4" if self.model_def.details.quantization_level.lower() == "fp4" else "nf4",
                    "bnb_4bit_compute_dtype": get_dtype(self.model_def),
                    "bnb_4bit_use_double_quant": True if self.model_def.details.quantization_level.lower().startswith(("q2", "int2")) else False,
                })
            elif self.model_def.details.quantization_level.lower().startswith(("q8", "int8")):
                quantization_config.update({
                    "load_in_8bit": True,
                })

        return BitsAndBytesConfig(**quantization_config) if quantization_config else {}
