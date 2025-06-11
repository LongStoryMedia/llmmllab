import torch
from PIL import Image

from services.model_service import model_service
from services.hardware_manager import hardware_manager


class ImageGenerator:
    """Service for generating images using loaded diffusion models."""

    def __init__(self):
        # We'll use model_service for accessing the active model
        self.model_service = model_service
        # Use hardware_manager for resource optimization
        self.hardware_manager = hardware_manager

    def generate(self, prompt: str, **kwargs) -> Image.Image:
        """Generate an image based on the prompt."""
        active_pipeline = None
        try:
            # Load the model on demand
            print("Loading model on demand for image generation...")
            self.model_service.load_active_model()
            active_pipeline = self.model_service.active_pipeline

            if not active_pipeline:
                raise RuntimeError(
                    "Failed to load the model pipeline. Please check model configuration.")

            # Check available memory and log status
            print(
                f"Before generation: {self.hardware_manager.get_memory_status_str()}")

            # Prepare generation parameters
            generation_params = {
                "prompt": prompt,
                "num_inference_steps": 30,
                "guidance_scale": 7.5,
            }

            # Override with any provided kwargs
            generation_params.update(kwargs)

            # Optimize parameters based on available hardware resources
            generation_params = self.hardware_manager.optimize_generation_params(
                generation_params)

            print(f"Starting generation with params: {generation_params}")

            # Generate the image
            if hasattr(active_pipeline, "__call__") and callable(active_pipeline):
                result = active_pipeline(**generation_params)

                # Check if the result has the expected structure
                if not hasattr(result, 'images') or not result.images or len(result.images) == 0:
                    print(f"Pipeline returned unexpected structure: {result}")
                    raise RuntimeError("Pipeline did not return any images")

                # Store the generated image for return
                generated_image = result.images[0]

                # Return the image
                return generated_image
            raise RuntimeError("Active pipeline is not callable")

        except torch.cuda.OutOfMemoryError as e:
            print(f"CUDA out of memory error: {e}")

            # Try to free memory
            self.hardware_manager.clear_memory()

            # Try again with reduced parameters
            try:
                # Use much smaller dimensions and fewer steps
                compact_params = {
                    "prompt": prompt,
                    "num_inference_steps": 20,
                    "guidance_scale": 7.0,
                    "height": 512,
                    "width": 512
                }
                print(f"Retrying with minimal parameters: {compact_params}")
                if hasattr(active_pipeline, "__call__") and callable(active_pipeline):
                    result = active_pipeline(**compact_params)

                    return result.images[0]
                raise RuntimeError(
                    "Active pipeline is not callable after retrying with minimal parameters")
            except Exception as retry_err:
                # Clean up and re-raise
                self.hardware_manager.clear_memory()
                raise RuntimeError(
                    f"Failed even with minimal parameters: {retry_err}")

        except Exception as e:
            print(
                f"Error during image generation: {type(e).__name__}: {str(e)}")
            # Clean up memory
            self.hardware_manager.clear_memory()
            # Re-throw with clearer message
            raise RuntimeError(f"Failed to generate image: {str(e)}")

        finally:
            # Always unload the model and free memory after generation
            print("Unloading model to free memory...")
            self.model_service.unload_active_model()
            self.hardware_manager.clear_memory()
            print(
                f"After generation and cleanup: {self.hardware_manager.get_memory_status_str()}")


# Create a singleton instance
generator = ImageGenerator()
