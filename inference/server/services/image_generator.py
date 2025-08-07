import json
import server.config as config
from runner.pipelines.factory import PipelineFactory
from models.model import Model
from models.inference_queue_message import InferenceQueueMessage
from models.image_generation_request import ImageGenerateRequest
from models.image_generation_response import ImageGenerateResponse
import os
import io
import time
import base64
import logging
import uuid
import asyncio
from typing import Optional, Union
import torch
import numpy as np
from PIL import Image
from diffusers.utils.loading_utils import load_image

# Import config to use configuration values
import server.config as config
from models.inference_queue_message import InferenceQueueMessage
from services.model_service import model_service
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    StableDiffusion3Pipeline,
)
from diffusers.quantizers.quantization_config import BitsAndBytesConfig
from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline,
)
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
        "Heartsync/Flux-NSFW-uncensored",
        weight_name="lora.safetensors",
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
        self.output_dir = (
            config.IMAGE_DIR
            if hasattr(config, "IMAGE_DIR")
            else os.environ.get("IMAGE_OUTPUT_DIR", "output")
        )

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

    @torch.inference_mode()
    def generate(self, image_request: ImageGenerateRequest) -> Image.Image:
        """
        Generate an image based on the given prompt synchronously.

        Args:
            prompt: Text prompt describing the desired image.
            **kwargs: Additional parameters for the diffusion pipeline.

        Returns:
            The generated image.
        """
        start_time = time.time()
        # Extract the prompt from the message
        if not image_request.prompt:
            raise ValueError("Prompt is required for image generation")

        if not image_request.model:
            self.logger.warning("No model specified, using default model")
            image_request.model = config.DEFAULT_MODEL_ID

        self.logger.info(f"Generating image with prompt: {image_request.prompt}")
        pipeline = PipelineFactory().get_pipeline(image_request.model)

        # Set default parameters if not provided, ensuring they're never None
        width = int(image_request.width or 1024)  # Default to 1024 if None or 0
        height = int(image_request.height or 1024)  # Default to 1024 if None or 0
        inference_steps = int(image_request.inference_steps or 20)
        guidance_scale = float(
            image_request.guidance_scale or 7.0
        )  # Default to 7.0 if None
        negative_prompt = image_request.negative_prompt

        # Log the final parameter values being used
        self.logger.info(
            f"Final image generation parameters: prompt='{image_request.prompt}', width={width}, height={height}, steps={inference_steps}, guidance_scale={guidance_scale}, negative_prompt='{negative_prompt}'"
        )

        # Generate the image
        with torch.inference_mode():
            result = pipeline(
                prompt=image_request.prompt,
                height=height,
                width=width,
                num_inference_steps=inference_steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
            )  # type: ignore

        end_time = time.time()
        generation_time = end_time - start_time
        self.logger.info(f"Image generation completed in {generation_time:.2f}s")
        self.last_generation_time = generation_time

        return result.images[0]  # type: ignore

    @torch.inference_mode()
    def edit(self, msg: InferenceQueueMessage) -> Image.Image:
        """
        Edit an image based on the given prompt and image.

        Args:
            request: Either an InferenceQueueMessage or an ImageGenerateRequest with image editing parameters

        Returns:
            The edited image.
        """
        start_time = time.time()
        kwargs, image_request = self.get_kwargs_from_message(msg)

        # Extract the prompt from the message
        prompt = kwargs.pop("prompt", None) or image_request.prompt
        if not prompt:
            raise ValueError("Prompt is required for image editing")

        # Validate image URL/path is provided
        if image_request.filename is None:
            raise ValueError("Filename is required for image editing")
        image_source = os.path.join(
            config.IMAGE_DIR, "originals", image_request.filename
        )

        model_id = "black-forest-labs-flux.1-kontext-dev"

        self.logger.info(f"Editing image with prompt: {prompt}")
        pipeline = PipelineFactory().get_pipeline(model_id)

        # Set default parameters if not provided, ensuring they're never None
        width = int(kwargs.get("width", 1024) or 1024)  # Default to 1024 if None or 0
        height = int(kwargs.get("height", 1024) or 1024)  # Default to 1024 if None or 0
        inference_steps = kwargs.get("inference_steps")
        num_inference_steps = int(inference_steps or 20)  # Default to 20 if None or 0
        guidance_scale = float(
            kwargs.get("guidance_scale", 7.0) or 7.0
        )  # Default to 7.0 if None
        negative_prompt = kwargs.get("negative_prompt", "")

        # Log the final parameter values being used
        self.logger.info(
            f"Final image editing parameters: prompt='{prompt}', width={width}, height={height}, steps={num_inference_steps}, guidance_scale={guidance_scale}, negative_prompt='{negative_prompt}'"
        )

        # Load the image from URL or path with appropriate headers if needed
        try:
            self.logger.info(f"Loading image from: {image_source}")
            init_image = load_image(image_source).convert("RGB")
            # init_image = load_image_with_headers(image_source, headers=headers).convert("RGB")
            self.logger.info("Image loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load image: {e}")
            raise ValueError(f"Failed to load image from {image_source}: {e}")

        # Generate the edited image
        with torch.inference_mode():
            self.logger.info("Starting image editing...")
            torch.cuda.synchronize()  # Ensure CUDA operations are synchronized
            result = pipeline(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                # negative_prompt=negative_prompt,
                image=init_image,  # Use the loaded image for editing
            )  # type: ignore

        end_time = time.time()
        generation_time = end_time - start_time
        self.logger.info(f"Image editing completed in {generation_time:.2f}s")
        self.last_generation_time = generation_time

        return result.images[0]  # type: ignore

    def save_image(self, image: Image.Image, filename: Optional[str] = None) -> str:
        """
        Save the generated image to disk.

        Args:
            image: The image to save.
            filename: Optional filename, if not provided a UUID will be generated.

        Returns:
            The path to the saved image.
        """
        if not filename or filename.strip() == "":
            filename = f"{uuid.uuid4().hex}.png"

        if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            filename += ".png"

        filepath = os.path.join(self.output_dir, filename)

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        # Save the image
        image.save(filepath)
        self.logger.info(f"Saved image to {filepath}")

        return filepath

    def process_and_save_image(
        self, image: Union[Image.Image, np.ndarray, torch.Tensor]
    ) -> ImageGenerateResponse:
        request_id = uuid.uuid4().hex
        try:
            if not image:
                raise ValueError("Model returned empty image")
            img: Optional[Image.Image] = None

            # Convert to PIL Image if necessary
            if isinstance(image, np.ndarray):
                # Convert numpy array to PIL Image
                img = Image.fromarray(
                    (image * 255).astype(np.uint8)
                    if image.max() <= 1.0
                    else image.astype(np.uint8)
                )
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
            img.save(img_byte_arr, format="PNG")
            img_data = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")

            # Construct download URL
            download_path = f"/download/{filename}"

            # Clean up GPU memory
            torch.cuda.empty_cache()

            print(f"Image generation complete for request {request_id}")
            print(f"Image saved to {file_path}")

            # Create ImageGenerateResponse object with the image data and download URL
            return ImageGenerateResponse(image=img_data, download=download_path)

        except Exception as e:
            import traceback

            print(f"Error in image processing: {type(e).__name__}: {str(e)}")
            print(traceback.format_exc())

            # Create and return InferenceQueueMessage with error response as payload
            return ImageGenerateResponse(image="", download="")

    def get_kwargs_from_message(
        self, msg: InferenceQueueMessage
    ) -> tuple[dict, ImageGenerateRequest]:
        """
        Extract keyword arguments from the InferenceQueueMessage payload.

        Args:
            msg: The InferenceQueueMessage containing the payload.

        Returns:
            A tuple containing a dictionary of keyword arguments and the ImageGenerateRequest.
        """
        kwargs = {}
        image_request: Optional[ImageGenerateRequest] = None
        # Parse parameters from the message payload
        if isinstance(msg, InferenceQueueMessage) and msg.payload:
            # Convert the payload to an ImageGenerateRequest if it's not already
            if isinstance(msg.payload, ImageGenerateRequest):
                # If payload is already an ImageGenerateRequest instance
                image_request = msg.payload
                kwargs = image_request.__dict__.copy()
            elif hasattr(msg.payload, "__dict__"):
                # If payload is an object with __dict__ attribute but not ImageGenerateRequest
                kwargs = msg.payload.__dict__.copy()
                # Create an ImageGenerateRequest instance from the dict
                try:
                    image_request = ImageGenerateRequest(**kwargs)
                    kwargs = image_request.__dict__.copy()
                except Exception as e:
                    self.logger.error(
                        f"Failed to convert payload to ImageGenerateRequest: {e}"
                    )
            elif isinstance(msg.payload, dict):
                # If payload is a dictionary
                kwargs = msg.payload.copy()
                # Create an ImageGenerateRequest instance from the dict
                try:
                    image_request = ImageGenerateRequest(**kwargs)
                    kwargs = image_request.__dict__.copy()
                except Exception as e:
                    self.logger.error(
                        f"Failed to convert dict payload to ImageGenerateRequest: {e}"
                    )
            else:
                # Try to parse as JSON if it's a string
                try:
                    if isinstance(msg.payload, str):
                        payload_dict = json.loads(msg.payload)
                        kwargs = payload_dict
                        # Create an ImageGenerateRequest instance
                        try:
                            image_request = ImageGenerateRequest(**payload_dict)
                            kwargs = image_request.__dict__.copy()
                        except Exception as e:
                            self.logger.error(
                                f"Failed to convert JSON payload to ImageGenerateRequest: {e}"
                            )
                    else:
                        self.logger.error(
                            f"Unexpected payload type: {type(msg.payload)}"
                        )
                        kwargs = {}
                except (json.JSONDecodeError, TypeError) as e:
                    self.logger.error(f"Failed to parse payload: {e}")
                    kwargs = {}
        else:
            kwargs = {}

        if image_request is None:
            # If no image request was found, log an error and raise an exception
            self.logger.error(
                "No valid image generation request found in the message payload"
            )
            raise ValueError("Invalid image generation request")

        return kwargs, image_request


# Create a singleton instance
image_generator = ImageGenerator()
