from datetime import datetime, timezone
from math import pi
import config
from models.model import Model
from models.inference_queue_message import InferenceQueueMessage
from models.image_generation_response import ImageGenerateResponse
import os
import io
import time
import base64
import logging
import uuid
import gc
import asyncio
from typing import Optional, Union, Callable
import torch
import numpy as np
from PIL import Image

# Import config to use configuration values
import config
from models.inference_queue_message import InferenceQueueMessage
from services.model_service import model_service
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import StableDiffusion3Pipeline
from diffusers.quantizers.quantization_config import BitsAndBytesConfig
from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline


def sd35_pipe(model_name: str):
    # Configure quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
    )

    # Load transformer model with quantization
    transformer = SD3Transformer2DModel.from_pretrained(
        model_name,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
    )

    # Load the full pipeline
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        model_name,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    )

    # Apply memory optimization techniques
    pipeline.enable_model_cpu_offload()
    pipeline.enable_attention_slicing()

    return pipeline


def sdxl_pipe(model_name: str):
    """
    Load the SDXL pipeline with quantization and optimizations.

    Args:
        model_name: Name of the model to load.

    Returns:
        The loaded StableDiffusionPipeline.
    """
    # Load the full pipeline
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        model_name,
        negative_prompt="anime, cartoon, sketch, drawing, lowres, bad anatomy, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username",
        device_map="balanced",
        torch_dtype=torch.float16,
        use_safetensors=True,
        # variant="fp16",
        safety_checker=None,  # Disable safety checker for now
    )
    # pipeline.to(hardware_manager.device)

    return pipeline


def flux_pipe(model: Model):
    """
    Load the Flux pipeline with quantization and optimizations.

    Args:
        model_name: Name of the model to load.

    Returns:
        The loaded StableDiffusionPipeline.
    """
    # Load the full pipeline
    pipeline = FluxPipeline.from_pretrained(
        model.name,
        # device_map="balanced",
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        # variant="fp16",
        # safety_checker=None,  # Disable safety checker for now
    )
    # pipeline.enable_model_cpu_offload()
    pipeline.enable_sequential_cpu_offload()
    # pipeline.enable_vae_slicing()

    # # pipeline.to(hardware_manager.device)
    # if model.lora_weights is not None:
    #     for lora_weight in enumerate(model.lora_weights):
    #         lw = model_service.get
    #         pipeline.load_lora_weights(
    #             lora_weight.name
    #         )
    pipeline.load_lora_weights(
        'Heartsync/Flux-NSFW-uncensored',
        weight_name='lora.safetensors',
        adapter_name="uncensored",
    )

    return pipeline


class ImageGenerator:
    """Service for generating images using diffusion models."""

    def __init__(self):
        """Initialize the image generator."""
        self.logger = logging.getLogger(__name__)
        self.model_service = model_service
        # Use config.IMAGE_DIR if defined, otherwise fallback to environment variable or default
        self.output_dir = config.IMAGE_DIR if hasattr(
            config, 'IMAGE_DIR') else os.environ.get('IMAGE_OUTPUT_DIR', 'output')

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Track if we're currently processing
        self.is_generating = False
        self.last_generation_time = 0

        # Model cache
        self.pipeline = None
        self.current_model_name = None

        # Queue management
        self.max_concurrent_generations = 1  # Adjust based on GPU capacity
        self.generation_semaphore = asyncio.Semaphore(self.max_concurrent_generations)
        self.generation_queue = []
        self.callbacks = {}

    def load_model(self):
        """
        Load the diffusion model if not already loaded or if a different model is requested.

        Args:
            model_name: Name of the model to load. If None, uses active model.

        Returns:
            The loaded pipeline.
        """
        model_name = model_service.get_active_model().name

        # # Return cached model if already loaded
        # if self.pipeline is not None and self.current_model_name == model_name:
        #     self.logger.info(f"Using cached model: {model_name}")
        #     return self.pipeline

        self.logger.info(f"Loading model: {model_name}")

        # Clean up previous model if exists
        if self.pipeline is not None:
            del self.pipeline
            torch.cuda.empty_cache()
            gc.collect()

        current_model = model_service.get_active_model()

        if not current_model:
            raise ValueError(f"Model '{model_name}' not found in model service.")

        self.logger.info(f"Current model details: {current_model}")

        # Check if the model is SD3 or SDXL
        if current_model.pipeline == "StableDiffusion3Pipeline":
            # Load SD3 pipeline
            self.logger.info(f"Loading Stable Diffusion 3 model: {model_name}")
            self.pipeline = sd35_pipe(model_name)
        elif current_model.pipeline == "StableDiffusionXLPipeline":
            # Load SDXL pipeline
            self.logger.info(f"Loading Stable Diffusion XL model: {model_name}")
            self.pipeline = sdxl_pipe(model_name)
        elif current_model.pipeline == "FluxPipeline":
            # Load Flux pipeline
            self.logger.info(f"Loading Flux model: {model_name}")
            self.pipeline = flux_pipe(current_model)
        else:
            raise ValueError(f"Unsupported pipeline type: {current_model.pipeline}")

        # Remember which model we've loaded
        self.current_model_name = model_name

        return self.pipeline

    async def generate_async(self, prompt: str, callback: Optional[Callable] = None, **kwargs) -> None:
        """
        Queue an image generation request to be processed asynchronously.

        Args:
            prompt: Text prompt describing the desired image.
            callback: Optional callback function to call with the result.
            **kwargs: Additional parameters for the diffusion pipeline.
        """
        # Add to queue
        request_id = kwargs.pop("request_id", str(uuid.uuid4()))
        if callback:
            self.callbacks[request_id] = callback

        # Add task to our queue
        task = {
            "id": request_id,
            "prompt": prompt,
            "kwargs": kwargs
        }
        self.generation_queue.append(task)

        # Process queue if not already processing
        if not self.is_generating:
            asyncio.create_task(self._process_queue())

    async def _process_queue(self):
        """Process queued image generation requests."""
        if self.is_generating:
            return

        self.is_generating = True

        try:
            while self.generation_queue:
                # Get the next task
                task = self.generation_queue.pop(0)

                # Acquire semaphore to limit concurrent generations
                async with self.generation_semaphore:
                    # Run the generation in a separate thread to not block the event loop
                    result = await asyncio.to_thread(
                        self._generate_image,
                        task["prompt"],
                        **task["kwargs"]
                    )

                    # Call the callback if one was provided
                    if task["id"] in self.callbacks:
                        callback = self.callbacks[task["id"]]
                        del self.callbacks[task["id"]]
                        callback(result)
        finally:
            self.is_generating = False

    def _generate_image(self, prompt: str, **kwargs) -> Image.Image:
        """Internal method to generate images synchronously."""
        start_time = time.time()
        self.logger.info(f"Starting image generation for prompt: '{prompt}'")

        # Ensure model is loaded
        pipeline = self.load_model()

        # Set default parameters if not provided, ensuring they're never None
        width = int(kwargs.get('width', 1024) or 1024)  # Default to 1024 if None or 0
        height = int(kwargs.get('height', 1024) or 1024)  # Default to 1024 if None or 0
        num_inference_steps = int(kwargs.get('num_inference_steps', 20) or 20)  # Default to 20 if None or 0
        guidance_scale = float(kwargs.get('guidance_scale', 7.0) or 7.0)  # Default to 7.0 if None
        negative_prompt = kwargs.get('negative_prompt', "")

        # Log the final parameter values being used
        self.logger.info(f"Final image generation parameters: prompt='{prompt}', width={width}, height={height}, steps={num_inference_steps}, guidance_scale={guidance_scale}, negative_prompt='{negative_prompt}'")

        # Generate the image
        with torch.inference_mode():
            result = pipeline(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
            )

        end_time = time.time()
        generation_time = end_time - start_time
        self.logger.info(f"Image generation completed in {generation_time:.2f}s")
        self.last_generation_time = generation_time

        return result.images[0]  # type: ignore

    @torch.inference_mode()
    def generate(self, prompt: str, **kwargs) -> Image.Image:
        """
        Generate an image based on the given prompt synchronously.

        Args:
            prompt: Text prompt describing the desired image.
            **kwargs: Additional parameters for the diffusion pipeline.

        Returns:
            The generated image.
        """
        return self._generate_image(prompt, **kwargs)

    def save_image(self, image: Image.Image, filename: Optional[str] = None) -> str:
        """
        Save the generated image to disk.

        Args:
            image: The image to save.
            filename: Optional filename, if not provided a UUID will be generated.

        Returns:
            The path to the saved image.
        """
        if not filename:
            filename = f"{uuid.uuid4().hex}.png"

        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            filename += '.png'

        filepath = os.path.join(self.output_dir, filename)

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        # Save the image
        image.save(filepath)
        self.logger.info(f"Saved image to {filepath}")

        return filepath

    def process_and_save_generated_image(self, image: Union[Image.Image, np.ndarray, torch.Tensor]) -> ImageGenerateResponse:
        request_id = uuid.uuid4().hex
        try:
            if not image:
                raise ValueError("Model returned empty image")
            img: Optional[Image.Image] = None

            # Convert to PIL Image if necessary
            if isinstance(image, np.ndarray):
                # Convert numpy array to PIL Image
                img = Image.fromarray(
                    (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8))
            elif torch.is_tensor(image):
                # Convert tensor to PIL Image
                if image.ndim == 3:
                    if image.shape[0] in [1, 3, 4]:  # [C, H, W]
                        image = image.permute(1, 2, 0)  # Convert to [H, W, C]
                    image_np = image.detach().cpu().numpy()
                    if image_np.max() <= 1.0:
                        image_np = (image_np * 255).astype(np.uint8)
                    img = Image.fromarray(image_np.astype(np.uint8))
            else:
                # Assume image is already a PIL Image
                img = image if isinstance(image, Image.Image) else None

            if img is None:
                raise ValueError("Could not convert image to PIL format")

            # Save the image with the same request_id as filename
            filename = f"{request_id}.png"
            file_path = os.path.join(config.IMAGE_DIR, filename)
            img.save(file_path)

            # Convert PIL image to base64 string for immediate display
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_data = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")

            # Construct download URL
            download_path = f"/download/{filename}"

            # Clean up GPU memory
            torch.cuda.empty_cache()

            print(f"Image generation complete for request {request_id}")
            print(f"Image saved to {file_path}")

            # Create ImageGenerateResponse object with the image data and download URL
            return ImageGenerateResponse(
                image=img_data,
                download=download_path
            )

        except Exception as e:
            import traceback
            print(f"Error in image processing: {type(e).__name__}: {str(e)}")
            print(traceback.format_exc())

            # Create and return InferenceQueueMessage with error response as payload
            return ImageGenerateResponse(
                image="",
                download=""
            )


# Create a singleton instance
image_generator = ImageGenerator()
