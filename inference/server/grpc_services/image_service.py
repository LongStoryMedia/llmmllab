"""
Image service implementation for the gRPC server.
"""

from server.protos.image_generation_response_pb2 import ImageGenerateResponse
from server.protos.image_generation_request_pb2 import ImageGenerateRequest
from server.services.model_service import model_service
from server.services.image_generator import image_generator
from server.services.hardware_manager import hardware_manager
from server.config import logger, IMAGE_DIR
import models
import torch
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
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class ImageService:
    """
    Service for handling image generation and editing requests.
    """

    def __init__(self):
        self.logger = logger

    def GenerateImage(self, request: ImageGenerateRequest, context):
        """
        Generate an image based on the request parameters.
        """
        try:
            # Log the request
            self.logger.debug(f"Prompt: {request.prompt}")

            # convert image_generation_request_pb2.ImageGenerateRequest to models.ImageGenerateRequest
            image_request = models.ImageGenerateRequest(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                model=request.model,
                width=request.width,
                height=request.height,
                inference_steps=request.inference_steps,
                guidance_scale=request.guidance_scale,
            )

            # Schedule the image generation with properly processed parameters
            img = image_generator.generate(image_request)

            if img is None:
                raise ValueError(
                    "Image generation failed - returned None"
                )  # Process and save the generated image
            return image_generator.process_and_save_image(img)

        except Exception as e:
            self.logger.error(f"Error in generate_image: {e}")
            return ImageGenerateResponse()

    def EditImage(self, request: ImageGenerateRequest, context):
        """
        Edit an image based on the request parameters.
        """
        try:
            # Log the request
            self.logger.debug(f"Prompt: {request.prompt}")

            # convert image_generation_request_pb2.ImageGenerateRequest to models.ImageGenerateRequest
            image_request = models.ImageGenerateRequest(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                model=request.model,
                width=request.width,
                height=request.height,
                inference_steps=request.inference_steps,
                guidance_scale=request.guidance_scale,
            )

            # Schedule the image generation with properly processed parameters
            img = image_generator.generate(image_request)

            if img is None:
                raise ValueError(
                    "Image generation failed - returned None"
                )  # Process and save the generated image
            return image_generator.process_and_save_image(img)

        except Exception as e:
            self.logger.error(f"Error in generate_image: {e}")
            return ImageGenerateResponse()
