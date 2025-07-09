from diffusers.schedulers.scheduling_euler_ancestral_discrete import EulerAncestralDiscreteScheduler
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_instruct_pix2pix import StableDiffusionInstructPix2PixPipeline
import torch
from ..helpers import get_dtype, get_precision
from models.model import Model


class SDXLRefinerPipe:
    @staticmethod
    def load(model: Model) -> StableDiffusionInstructPix2PixPipeline:
        """        
        Load the Stable Diffusion 3 pipeline with quantization and memory optimizations.

        Args:
            model (Model): The model configuration to load.
        Returns:
            StableDiffusion3Pipeline: The loaded pipeline with quantization and optimizations.
        """
        # Load the full pipeline
        pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model.model,
            device_map="balanced",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant=get_precision(model),  # Use the precision from the model details
            safety_checker=None,  # Disable safety checker for now
            attn_implementation="eager",
        )
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)

        # pipeline.to(hardware_manager.device)

        return pipeline
