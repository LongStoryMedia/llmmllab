import torch
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img import StableDiffusionXLImg2ImgPipeline
from diffusers.utils.loading_utils import load_image
from ..helpers import get_dtype, get_precision
from models.model import Model

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipe = pipe.to("cuda")
url = "https://huggingface.co/datasets/patrickvonplaten/images/resolve/main/aa_xl/000000009.png"

init_image = load_image(url).convert("RGB")
prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt, image=init_image).images  # type: ignore


class SDXLRefinerPipe:
    @staticmethod
    def load(model: Model) -> StableDiffusionXLImg2ImgPipeline:
        """        
        Load the Stable Diffusion 3 pipeline with quantization and memory optimizations.

        Args:
            model (Model): The model configuration to load.
        Returns:
            StableDiffusion3Pipeline: The loaded pipeline with quantization and optimizations.
        """
        # Load the full pipeline
        pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            model.model,
            device_map="balanced",
            torch_dtype=get_dtype(model),
            use_safetensors=True,
            variant=get_precision(model),  # Use the precision from the model details
            safety_checker=None,  # Disable safety checker for now
        )
        # pipeline.to(hardware_manager.device)

        return pipeline
