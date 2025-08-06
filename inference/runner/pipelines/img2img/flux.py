import logging

from models.model import Model
from models import ChatReq
from typing import Optional, Any
import torch

# Import with different name to avoid linting error when module can't be resolved
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.quantizers.quantization_config import BitsAndBytesConfig
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
        super().__init__()
        self.model = model
        self.model_def = model

        logging.getLogger(__name__).info(
            f"Loading Flux pipeline for model: {model.name} (ID: {model.id}, dtype: {torch.bfloat16})"
        )
        quantization_config = self._setup_quantization_config()
        transformer_kwargs = {
            "torch_dtype": torch.bfloat16,
            "subfolder": "transformer",
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
            ),
        }

        if quantization_config is not None:
            transformer_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
            )

        # Get number of available CUDA devices
        num_gpus = torch.cuda.device_count()
        logging.getLogger(__name__).info(
            f"Found {num_gpus} CUDA devices for Flux pipeline"
        )

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
        try:
            self.pipeline = FluxPipeline.from_pretrained(
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
        except (ImportError, ModuleNotFoundError) as e:
            logging.getLogger(__name__).error(
                f"Failed to load FluxKontextPipeline: {e}"
            )
            raise RuntimeError(f"Failed to load FluxKontextPipeline: {e}") from e

        # Enable memory optimizations
        self.pipeline.enable_vae_slicing()
        self.pipeline.enable_vae_tiling()

        # Clean memory before model usage
        torch.cuda.empty_cache()

        # Enable memory optimizations based on available GPUs
        if num_gpus > 1:
            # For multi-GPU setup, use model CPU offload instead of direct multi-GPU mapping
            # This is more compatible with the FluxKontextPipeline implementation
            logging.getLogger(__name__).info(
                "Configuring multi-GPU memory optimization"
            )
            torch.cuda.synchronize()  # Ensure CUDA operations are synchronized

            # Use sequential CPU offload for controlled memory management across GPUs
            logging.getLogger(__name__).info(
                "Enabling sequential CPU offload for multi-GPU setup"
            )
            # self.pipeline.enable_sequential_cpu_offload()

            # Set CUDA memory management options to avoid fragmentation
            torch.cuda.set_per_process_memory_fraction(
                0.9
            )  # Reserve some memory to avoid OOM errors
        else:
            # For single GPU, use model CPU offload to optimize memory usage
            logging.getLogger(__name__).info(
                "Enabling model CPU offload for single GPU"
            )
            self.pipeline.enable_model_cpu_offload()

    def _setup_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """
        Set up the quantization configuration based on the model details.

        Returns:
            Optional[BitsAndBytesConfig]: The quantization configuration or None.
        """
        return super()._setup_quantization_config()

    def run(self, req: ChatReq) -> Any:
        """
        Process the input messages and generate an image using the Flux Kontext pipeline.

        Args:
            req (ChatReq): The chat request containing messages and parameters.

        Returns:
            Any: The generated image.
        """
        if not self.pipeline:
            raise RuntimeError("Pipeline not initialized. Call load() first.")

        messages = req.messages

        # Extract prompt and image from messages
        prompt = ""
        image = None
        for message in messages:
            if message.role == "user":
                if hasattr(message, "content") and message.content:
                    for content in message.content:
                        if hasattr(content, "text") and content.text:
                            prompt = content.text
                        if (
                            hasattr(content, "type")
                            and content.type == "image"
                            and hasattr(content, "url")
                            and content.url
                        ):
                            # Load image from URL or path
                            from diffusers.utils.loading_utils import load_image

                            image = load_image(content.url)
        if not image:
            raise ValueError("No image provided in messages")
        result = self.pipeline(prompt=prompt, image=image)
        return result

    def __del__(self) -> None:
        """
        Clean up resources used by the FluxKontextPipe.
        """
        try:
            if hasattr(self, "pipeline") and self.pipeline is not None:
                del self.pipeline
        except (RuntimeError, AttributeError, ValueError, TypeError) as e:
            print(f"Error cleaning up FluxKontextPipe resources: {str(e)}")
