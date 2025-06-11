import os
import json
import time
import torch
from typing import Dict, List, Optional
from accelerate import Accelerator
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline

from config import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_SOURCE,
    MODELS_CONFIG_PATH,
    ENABLE_MEMORY_EFFICIENT_ATTENTION,
    ENABLE_VAE_SLICING,
    ENABLE_MODEL_CPU_OFFLOAD,
    ENABLE_SEQUENTIAL_CPU_OFFLOAD,
    USE_FP16_PRECISION
)
from services.lora_service import lora_service


class ModelService:
    """Service for managing Stable Diffusion models."""

    def __init__(self):
        # Initialize accelerator
        self.accelerator = Accelerator()
        self.device = self.accelerator.device

        # Dictionary to store loaded models
        self.models: Dict[str, dict] = {}
        self.active_model_id: str = DEFAULT_MODEL_ID
        self.active_pipeline: Optional[DiffusionPipeline] = None

        # Initialize from config or create default
        self._load_models_config()

    def _load_models_config(self):
        """Load models configuration from file or create default."""
        try:
            if os.path.exists(MODELS_CONFIG_PATH):
                with open(MODELS_CONFIG_PATH, 'r') as f:
                    config = json.load(f)
                    self.models = config.get('models', {})
                    self.active_model_id = config.get(
                        'active_model_id', DEFAULT_MODEL_ID)
            else:
                # Create a default configuration
                self.models = {
                    DEFAULT_MODEL_ID: {
                        'id': DEFAULT_MODEL_ID,
                        'name': 'Default Stable Diffusion',
                        'source': DEFAULT_MODEL_SOURCE,
                        'description': 'Default Stable Diffusion model',
                        'is_active': True
                    }
                }
                self._save_models_config()
        except Exception as e:
            print(f"Error loading models config: {e}")
            # Create a default configuration
            self.models = {
                DEFAULT_MODEL_ID: {
                    'id': DEFAULT_MODEL_ID,
                    'name': 'Default Stable Diffusion',
                    'source': DEFAULT_MODEL_SOURCE,
                    'description': 'Default Stable Diffusion model',
                    'is_active': True
                }
            }
            self._save_models_config()

    def _save_models_config(self):
        """Save models configuration to file."""
        config = {
            'models': self.models,
            'active_model_id': self.active_model_id
        }
        os.makedirs(os.path.dirname(MODELS_CONFIG_PATH), exist_ok=True)
        with open(MODELS_CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=2)

    def load_active_model(self):
        """Load the active model pipeline."""
        if self.active_model_id not in self.models:
            # Fall back to default if active model doesn't exist
            self.active_model_id = DEFAULT_MODEL_ID
            if DEFAULT_MODEL_ID not in self.models:
                # Add default model if it doesn't exist
                self.models[DEFAULT_MODEL_ID] = {
                    'id': DEFAULT_MODEL_ID,
                    'name': 'Default Stable Diffusion',
                    'source': DEFAULT_MODEL_SOURCE,
                    'description': 'Default Stable Diffusion model',
                    'is_active': True
                }
                self._save_models_config()

        try:
            model_source = self.models[self.active_model_id]['source']
            print(f"Starting load of model from: {model_source}")

            # Set cache directory - first check environment variable, then use default
            cache_dir = os.environ.get("HF_HOME", "/root/.cache/huggingface")

            # Configure download options with memory optimizations
            load_options = {
                "torch_dtype": torch.float16,  # if USE_FP16_PRECISION else torch.float32,
                "use_safetensors": True,
                "variant": "fp16",  # if USE_FP16_PRECISION else None,
                "cache_dir": cache_dir
            }

            # Add memory optimization parameters
            if ENABLE_MEMORY_EFFICIENT_ATTENTION:
                print("Enabling memory efficient attention")
                load_options["attention_mechanism"] = "xformers"

            # Check if model is already cached
            model_id = model_source.replace("/", "--")
            if not model_id.startswith("models--"):
                model_id = f"models--{model_id}"

            # Improved cache detection - check for specific files and snapshots
            cached_model_path = os.path.join(cache_dir, "hub", model_id)
            snapshots_dir = os.path.join(cached_model_path, "snapshots")
            refs_file = os.path.join(cached_model_path, "refs", "main")

            # Only consider the model cached if both snapshots directory exists and refs file exists
            model_exists = (os.path.exists(snapshots_dir) and
                            os.path.exists(refs_file) and
                            os.path.isdir(snapshots_dir) and
                            len(os.listdir(snapshots_dir)) > 0)

            print(f"Checking for cached model at: {cached_model_path}")
            print(
                f"Model {'exists' if model_exists else 'does not exist'} in cache")
            print(f"Snapshots dir exists: {os.path.exists(snapshots_dir)}")
            print(f"Refs file exists: {os.path.exists(refs_file)}")

            # Always try without local_files_only first to ensure model loads
            try:
                print("Attempting to load model...")
                load_options["local_files_only"] = False

                # Let the library auto-detect the correct pipeline type
                print("Loading with generic DiffusionPipeline")
                self.active_pipeline = DiffusionPipeline.from_pretrained(
                    model_source,
                    **load_options
                )
                print(
                    f"Successfully loaded model with pipeline type: {self.active_pipeline.__class__.__name__}")

            except Exception as e:
                print(f"Error loading with generic pipeline: {e}")

                # Try specific pipeline classes as fallbacks
                print("Falling back to specific pipeline classes")
                # Ensure we can download if needed
                load_options["local_files_only"] = False

                try_pipeline_classes = [
                    "StableDiffusionXLPipeline",
                    "StableDiffusionPipeline"
                ]

                for pipeline_class_name in try_pipeline_classes:
                    try:
                        print(f"Trying with {pipeline_class_name}")
                        import importlib
                        module = importlib.import_module("diffusers")
                        pipeline_class = getattr(
                            module, pipeline_class_name)

                        self.active_pipeline = pipeline_class.from_pretrained(
                            model_source,
                            **load_options
                        )
                        print(
                            f"Successfully loaded with {pipeline_class_name}")
                        break
                    except Exception as class_error:
                        print(
                            f"Failed with {pipeline_class_name}: {class_error}")

                if not self.active_pipeline:
                    raise RuntimeError(
                        f"Failed to load model with any pipeline class")

            # Move to device after loading
            if self.active_pipeline:
                print(f"Moving model to device: {self.device}")

                # Apply memory optimizations based on configuration
                if ENABLE_SEQUENTIAL_CPU_OFFLOAD:
                    print("Enabling sequential CPU offloading for low memory")
                    from accelerate import cpu_offload
                    for component in [self.active_pipeline.unet, self.active_pipeline.vae, self.active_pipeline.text_encoder]:
                        if component is not None:
                            cpu_offload(component, self.device)
                elif ENABLE_MODEL_CPU_OFFLOAD:
                    print("Enabling model CPU offloading for low memory")
                    self.active_pipeline.enable_model_cpu_offload()
                else:
                    # Standard device placement
                    self.active_pipeline = self.active_pipeline.to(self.device)
                    self.active_pipeline.enable_attention_slicing()
                # self.active_pipeline.enable_sequential_cpu_offload()
                # self.active_pipeline.enable_model_cpu_offload()
                    # self.active_pipeline.enable_xformers_memory_efficient_attention()
                # self.active_pipeline.set_progress_bar_config(disable=True)  # Disable progress bar
                # self.active_pipeline.safety_checker = None  # Disable safety checker
                # self.active_pipeline.feature_extractor = None  # Disable feature extractor
                # self.active_pipeline.vae.enable_tiling = True  # Enable tiling for VAE

                # Enable VAE slicing if configured (reduces memory during inference)
                if ENABLE_VAE_SLICING and hasattr(self.active_pipeline, 'enable_vae_slicing'):
                    print("Enabling VAE slicing for memory efficiency")
                    self.active_pipeline.enable_vae_slicing()

                # Empty cache to free up memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Apply active LoRA weights if any
                self._apply_active_loras()

                print(
                    f"Successfully loaded model: {self.active_model_id} from {model_source}")
            else:
                raise RuntimeError(
                    f"Failed to initialize pipeline for model: {self.active_model_id}")

        except Exception as e:
            print(f"Error loading model {self.active_model_id}: {str(e)}")
            print(f"Detailed error: {type(e).__name__}: {str(e)}")

            # Try to load default model as a fallback
            if self.active_model_id != DEFAULT_MODEL_ID:
                print(f"Falling back to default model: {DEFAULT_MODEL_ID}")
                self.active_model_id = DEFAULT_MODEL_ID
                self.load_active_model()
            else:
                # If we're already trying to load the default model and it fails, raise the error
                print("Failed to load default model. No fallback available.")
                raise RuntimeError(f"Failed to load model: {str(e)}")

    def _apply_active_loras(self):
        """Apply all active LoRA weights to the current pipeline."""
        if not self.active_pipeline:
            return

        active_loras = lora_service.get_active_loras()
        if not active_loras:
            print("No active LoRAs to apply")
            return

        # Check if PEFT is installed
        try:
            import peft
            print(f"PEFT library found (version {peft.__version__})")
        except ImportError:
            print("ERROR: PEFT library is not installed. Cannot apply LoRA weights.")
            print("Please install PEFT with: pip install -U peft")
            return

        import torch
        print(f"Applying {len(active_loras)} active LoRA weights")

        for lora in active_loras:
            try:
                lora_source = lora['source']
                # Default weight if not specified
                lora_weight = lora.get('weight', 0.75)

                print(
                    f"Loading LoRA: {lora['name']} (weight: {lora_weight}) from {lora_source}")

                # Apply the LoRA weights directly using the simplified pattern
                try:
                    self.active_pipeline.load_lora_weights(lora_source)

                    # If the model supports adapter weights, set the weight
                    if hasattr(self.active_pipeline, "fuse_lora"):
                        # Some models use fuse_lora with a scale parameter
                        print(f"Fusing LoRA with scale {lora_weight}")
                        self.active_pipeline.fuse_lora(lora_weight)
                    elif hasattr(self.active_pipeline, "set_adapter_strength"):
                        print(f"Setting adapter strength to {lora_weight}")
                        self.active_pipeline.set_adapter_strength(lora_weight)

                    print(f"Successfully applied LoRA: {lora['name']}")
                except Exception as load_err:
                    if "PEFT backend is required for this method" in str(load_err):
                        print(
                            f"ERROR: PEFT backend is required to load LoRA weights.")
                        print("Please install PEFT with: pip install -U peft")
                        break
                    else:
                        raise

            except Exception as e:
                print(f"Error applying LoRA {lora['name']}: {e}")

        # Clear CUDA cache after loading LoRAs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_models(self) -> List[dict]:
        """Get list of all available models."""
        return [model for model in self.models.values()]

    def get_model(self, model_id: str) -> Optional[dict]:
        """Get details of a specific model."""
        return self.models.get(model_id)

    def add_model(self, name: str, source: str, description: Optional[str] = None) -> dict:
        """Add a new model to the configuration."""
        from uuid import uuid4

        # Generate a unique Id for the model
        model_id = str(uuid4())

        # Add the model to the configuration
        self.models[model_id] = {
            'id': model_id,
            'name': name,
            'source': source,
            'description': description,
            'is_active': False
        }

        # Save the updated configuration
        self._save_models_config()

        return self.models[model_id]

    def remove_model(self, model_id: str) -> bool:
        """Remove a model from the configuration."""
        if model_id == DEFAULT_MODEL_ID:
            # Don't allow removing the default model
            return False

        if model_id in self.models:
            # If removing the active model, switch to default
            if model_id == self.active_model_id:
                self.active_model_id = DEFAULT_MODEL_ID

            # Remove the model from configuration
            del self.models[model_id]
            self._save_models_config()
            return True

        return False

    def set_active_model(self, model_id: str) -> bool:
        """Set a model as the active model."""
        if model_id in self.models:
            print(f"Attempting to activate model: {model_id}")
            self.active_model_id = model_id

            # Update is_active flags
            for mid in self.models:
                self.models[mid]['is_active'] = (mid == model_id)

            # Load the model
            try:
                self._save_models_config()
                return True
            except Exception as e:
                # If loading fails, revert to previous model
                print(f"Failed to load model {model_id}: {str(e)}")
                print("Reverting to previous model")
                for mid in self.models:
                    if mid != model_id and self.models[mid].get('is_active', False):
                        self.active_model_id = mid
                        break
                return False

        return False

    def unload_active_model(self):
        """Unload the active model to free up memory."""
        if not self.active_pipeline:
            return  # Nothing to unload

        try:
            print(f"Unloading model: {self.active_model_id}")

            # Move model to CPU first (helps with cleaner CUDA memory release)
            if torch.cuda.is_available() and hasattr(self.active_pipeline, "to"):
                self.active_pipeline.to("cpu")

            # Safely clean up model components to help with garbage collection
            for component in ["unet", "vae", "text_encoder", "scheduler"]:
                if component in self.active_pipeline.__dict__:
                    self.active_pipeline.__dict__[component] = None

            # Delete the pipeline itself
            del self.active_pipeline
            self.active_pipeline = None

            # Run garbage collector
            import gc
            gc.collect()

            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"Model {self.active_model_id} successfully unloaded")

        except Exception as e:
            print(f"Error unloading model {self.active_model_id}: {str(e)}")


# Create a singleton instance
model_service = ModelService()
