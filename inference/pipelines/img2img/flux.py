import logging

from ..helpers import get_dtype
from models.model import Model
from models import Message
from typing import Optional, List, Any
import torch
from diffusers.pipelines.flux.pipeline_flux_kontext import FluxKontextPipeline
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.quantizers.quantization_config import BitsAndBytesConfig
from diffusers.hooks.group_offloading import apply_group_offloading
from diffusers.models.attention_processor import AttnProcessor
from diffusers.quantizers import PipelineQuantizationConfig
from ..base_pipeline import BasePipeline


class FluxKontextPipe(BasePipeline):
    def __init__(self, model: Model):
        """
        Initialize a FluxKontextPipe instance and load the pipeline.

        Args:
            model (Model): The model configuration to load.
        """
        self.model = model

        logging.getLogger(__name__).info(f"Loading Flux pipeline for model: {model.name} (ID: {model.id}, dtype: {torch.bfloat16})")
        quantization_config = self._setup_quantization_config(model)
        transformer_kwargs = {
            "torch_dtype": torch.bfloat16,
            "subfolder": "transformer",
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
            )
        }

        if quantization_config is not None:
            transformer_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
            )

        # Get number of available CUDA devices
        num_gpus = torch.cuda.device_count()
        logging.getLogger(__name__).info(f"Found {num_gpus} CUDA devices for Flux pipeline")

        # For multi-GPU setups, we'll use different strategies but need to keep device_map as a string
        device_map = "balanced"

        # Load the transformer model with appropriate device mapping
        transformer = FluxTransformer2DModel.from_pretrained(
            model.model,
            device_map=device_map,
            attn_processor=AttnProcessor(),
            **transformer_kwargs,
        )

        qc = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "load_in_8bit": False,
            },
        )

        # Load the full pipeline with proper device mapping
        self.pipeline = FluxKontextPipeline.from_pretrained(
            model.name,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            transformer=transformer,
            offload_folder="./offload",
            attn_implementation="eager",
            attn_processor=AttnProcessor(),
            quantization_config=qc,
        )

        # Enable memory optimizations
        self.pipeline.enable_vae_slicing()
        self.pipeline.enable_vae_tiling()

        # Clean memory before model usage
        torch.cuda.empty_cache()

        # Enable memory optimizations based on available GPUs
        if num_gpus > 1:
            # For multi-GPU setup, use model CPU offload instead of direct multi-GPU mapping
            # This is more compatible with the FluxKontextPipeline implementation
            logging.getLogger(__name__).info("Configuring multi-GPU memory optimization")
            torch.cuda.synchronize()  # Ensure CUDA operations are synchronized

            # Use sequential CPU offload for controlled memory management across GPUs
            logging.getLogger(__name__).info("Enabling sequential CPU offload for multi-GPU setup")
            # self.pipeline.enable_sequential_cpu_offload()

            # Set CUDA memory management options to avoid fragmentation
            torch.cuda.set_per_process_memory_fraction(0.9)  # Reserve some memory to avoid OOM errors
        else:
            # For single GPU, use model CPU offload to optimize memory usage
            logging.getLogger(__name__).info("Enabling model CPU offload for single GPU")
            self.pipeline.enable_model_cpu_offload()

    def _setup_quantization_config(self, model: Model) -> Optional[BitsAndBytesConfig]:
        """
        Set up the quantization configuration based on the model details.

        Args:
            model (Model): The model configuration.

        Returns:
            Optional[BitsAndBytesConfig]: The quantization configuration or None.
        """
        if model.details is not None and model.details.quantization_level is not None:
            if model.details.quantization_level.lower().startswith(("q4", "int4", "nf4")):
                return BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                )
            elif model.details.quantization_level.lower().startswith(("q8", "int8")):
                return BitsAndBytesConfig(
                    load_in_8bit=True,
                )
        return None

    def run(self, messages: List[Message]) -> Any:
        """
        Process the input messages and generate an image using the Flux Kontext pipeline.

        Args:
            messages (List[Message]): The list of messages to process.

        Returns:
            Any: The generated image.
        """
        if not self.pipeline:
            raise RuntimeError("Pipeline not initialized. Call load() first.")

        # Extract prompt and image from messages
        prompt = ""
        image = None

        for message in messages:
            if message.role == "user":
                if isinstance(message.content, str):
                    prompt = message.content
                elif isinstance(message.content, list):
                    # Handle multimodal input where content is a list of parts
                    for part in message.content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            prompt = part.get("text", "")
                        elif isinstance(part, dict) and part.get("type") == "image":
                            # This assumes the image is available as a URL or local path
                            # You might need to adjust this based on your actual Message implementation
                            image_url = part.get("url", "")
                            if image_url:
                                # Load image from URL - you might need to implement this
                                from diffusers.utils.loading_utils import load_image
                                image = load_image(image_url)

        if not image:
            raise ValueError("No image provided in messages")

        # Generate image with the pipeline
        result = self.pipeline(prompt=prompt, image=image)
        return result

    def __del__(self) -> None:
        """
        Clean up resources used by the FluxKontextPipe.
        This method releases GPU memory by moving models to CPU.
        """
        try:
            if hasattr(self, 'pipeline') and self.pipeline is not None:
                # Move the pipeline to CPU to free GPU memory
                self.pipeline.to('cpu')
                if hasattr(self, 'model') and hasattr(self.model, 'name'):
                    print(f"FluxKontextPipe for {self.model.name}: Resources moved to CPU during cleanup")
        except Exception as e:
            # Use a direct print as logger might be gone during deletion
            print(f"Error cleaning up FluxKontextPipe resources: {str(e)}")
