"""
Image service implementation for the gRPC server.
"""

from services.model_service import model_service
from services.image_generator import image_generator
from services.hardware_manager import hardware_manager
from config import logger, IMAGE_DIR
import torch
from grpc_server.proto import inference_pb2, inference_pb2_grpc
import grpc
import os
import sys
import logging
import uuid
import base64
from io import BytesIO
from PIL import Image as PILImage
from typing import Dict, Iterator, List, Optional

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class ImageService:
    """
    Service for handling image generation and editing requests.
    """

    def __init__(self):
        self.logger = logger

    def generate_image(self, request: inference_pb2.ImageGenerateRequest, context):
        """
        Generate an image based on the request parameters.
        """
        try:
            # Log the request
            self.logger.info(f"Image generation request from user {request.user_id}, conversation {request.conversation_id}")
            self.logger.debug(f"Prompt: {request.prompt}")

            # Generate a unique request ID
            request_id = str(uuid.uuid4())

            # Create a background task for image generation
            self._generate_image_task(
                request_id=request_id,
                user_id=request.user_id,
                conversation_id=request.conversation_id,
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                model_name=request.model_name,
                width=request.width,
                height=request.height,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                seed=request.seed,
                scheduler=request.scheduler,
            )

            # Return the request ID
            return inference_pb2.ImageGenerateResponse(
                request_id=request_id,
                is_error=False,
            )

        except Exception as e:
            self.logger.error(f"Error in generate_image: {e}")
            return inference_pb2.ImageGenerateResponse(
                request_id="",
                is_error=True,
                error_message=str(e),
            )

    def edit_image(self, request: inference_pb2.ImageEditRequest, context):
        """
        Edit an image based on the request parameters.
        """
        try:
            # Log the request
            self.logger.info(f"Image editing request from user {request.user_id}, conversation {request.conversation_id}")
            self.logger.debug(f"Prompt: {request.prompt}")

            # Generate a unique request ID
            request_id = str(uuid.uuid4())

            # Handle the image data
            image_data = None
            if request.image_data:
                # Decode base64 image data
                image_data = BytesIO(base64.b64decode(request.image_data))
            elif request.image_url:
                # Download image from URL
                # This would be implemented based on your specific requirements
                pass

            if not image_data:
                raise ValueError("No image data provided")

            # Create a background task for image editing
            self._edit_image_task(
                request_id=request_id,
                user_id=request.user_id,
                conversation_id=request.conversation_id,
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                model_name=request.model_name,
                image_data=image_data,
                width=request.width,
                height=request.height,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                seed=request.seed,
                scheduler=request.scheduler,
            )

            # Return the request ID
            return inference_pb2.ImageEditResponse(
                request_id=request_id,
                is_error=False,
            )

        except Exception as e:
            self.logger.error(f"Error in edit_image: {e}")
            return inference_pb2.ImageEditResponse(
                request_id="",
                is_error=True,
                error_message=str(e),
            )

    def check_image_status(self, request: inference_pb2.ImageStatusRequest, context):
        """
        Check the status of an image generation/editing request.
        """
        try:
            # Get the request ID
            request_id = request.request_id

            # Check if the image exists
            file_path = os.path.join(IMAGE_DIR, f"{request_id}.png")
            if os.path.exists(file_path):
                # Image is ready
                image = PILImage.open(file_path)
                img_byte_arr = BytesIO()
                image.save(img_byte_arr, format='PNG')
                img_data = base64.b64encode(img_byte_arr.getvalue())

                return inference_pb2.ImageStatusResponse(
                    status=inference_pb2.ImageStatusResponse.Status.COMPLETE,
                    image_data=img_data,
                    download_url=f"/download/{request_id}.png",
                )
            else:
                # Image is still processing
                return inference_pb2.ImageStatusResponse(
                    status=inference_pb2.ImageStatusResponse.Status.PROCESSING,
                )

        except Exception as e:
            self.logger.error(f"Error in check_image_status: {e}")
            return inference_pb2.ImageStatusResponse(
                status=inference_pb2.ImageStatusResponse.Status.ERROR,
                error_message=str(e),
            )

    def _generate_image_task(self, request_id, user_id, conversation_id, prompt, negative_prompt,
                             model_name, width, height, num_inference_steps, guidance_scale, seed, scheduler):
        """
        Background task for image generation.
        """
        try:
            # Ensure the image directory exists
            os.makedirs(IMAGE_DIR, exist_ok=True)

            # Set default values for missing parameters
            if not model_name:
                model_name = model_service.get_default_model().source
            if width <= 0:
                width = 1024
            if height <= 0:
                height = 1024
            if num_inference_steps <= 0:
                num_inference_steps = 50
            if guidance_scale <= 0:
                guidance_scale = 7.5

            # Generate the image
            result = image_generator.generate_image(
                prompt=prompt,
                negative_prompt=negative_prompt,
                model_name=model_name,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                scheduler=scheduler,
            )

            # Save the image
            output_path = os.path.join(IMAGE_DIR, f"{request_id}.png")
            result.save(output_path)

            self.logger.info(f"Image generation completed for request {request_id}, saved to {output_path}")

        except Exception as e:
            self.logger.error(f"Error in _generate_image_task: {e}")

    def _edit_image_task(self, request_id, user_id, conversation_id, prompt, negative_prompt,
                         model_name, image_data, width, height, num_inference_steps,
                         guidance_scale, seed, scheduler):
        """
        Background task for image editing.
        """
        try:
            # Ensure the image directory exists
            os.makedirs(IMAGE_DIR, exist_ok=True)

            # Set default values for missing parameters
            if not model_name:
                model_name = model_service.get_default_model().source
            if width <= 0:
                width = 1024
            if height <= 0:
                height = 1024
            if num_inference_steps <= 0:
                num_inference_steps = 50
            if guidance_scale <= 0:
                guidance_scale = 7.5

            # Load the input image
            input_image = PILImage.open(image_data).convert("RGB")

            # Edit the image
            result = image_generator.edit_image(
                image=input_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                model_name=model_name,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                scheduler=scheduler,
            )

            # Save the image
            output_path = os.path.join(IMAGE_DIR, f"{request_id}.png")
            result.save(output_path)

            self.logger.info(f"Image editing completed for request {request_id}, saved to {output_path}")

        except Exception as e:
            self.logger.error(f"Error in _edit_image_task: {e}")
