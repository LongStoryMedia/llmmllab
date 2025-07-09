from ..helpers import get_dtype, get_precision
from models.model import Model
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline


class SDXLPipe:
    @staticmethod
    def load(model: Model) -> StableDiffusionXLPipeline:
        """        
        Load the Stable Diffusion 3 pipeline with quantization and memory optimizations.

        Args:
            model (Model): The model configuration to load.
        Returns:
            StableDiffusion3Pipeline: The loaded pipeline with quantization and optimizations.
        """
        # Load the full pipeline
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            model.model,
            device_map="balanced",
            torch_dtype=get_dtype(model),
            use_safetensors=True,
            variant=get_precision(model),  # Use the precision from the model details
            safety_checker=None,  # Disable safety checker for now
        )
        # pipeline.to(hardware_manager.device)

        return pipeline
