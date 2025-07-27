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

    @abstractmethod
    def __del__(self) -> None:
        """
        Clean up resources used by the pipeline.
        This method should release GPU memory by moving models to CPU.
        It will be called automatically when the pipeline is about to be destroyed.
        """
        raise NotImplementedError("Subclasses must implement the __del__ method.")

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
