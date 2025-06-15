import os
import json
import time
import torch
from typing import Dict, List, Optional
from accelerate import Accelerator
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from config import (
    DEFAULT_LORA_WEIGHT,
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_SOURCE,
    MODELS_CONFIG_PATH,
    ENABLE_MEMORY_EFFICIENT_ATTENTION,
    ENABLE_VAE_SLICING,
    ENABLE_MODEL_CPU_OFFLOAD,
    ENABLE_SEQUENTIAL_CPU_OFFLOAD,
    USE_FP16_PRECISION,
    FORCE_FP32_ON_CPU
)
from models.model_details import ModelDetails
from models.model import Model
from services.lora_service import lora_service
from services.hardware_manager import hardware_manager


class ModelService:
    """Service for managing Stable Diffusion models."""

    def __init__(self):
        # Initialize accelerator
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.default_model = Model(
            name='Default Stable Diffusion',
            model=DEFAULT_MODEL_SOURCE,  # Use the actual model source path instead of the ID
            modified_at='',
            size=0,
            digest='',
            details=ModelDetails(
                parent_model='',
                format='gguf',
                family='llama',
                families=[],
                parameter_size='7.2B',
                quantization_level='Q4_0',
                specialization='TextToImage'
            )  # You may want to provide a ModelDetails instance here
        )

        # Dictionary to store loaded models
        self.models: Dict[str, Model] = {}
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
                    models_dict = config.get('models', {})

                    # Convert dictionary back to Model objects
                    self.models = {}
                    for model_id, model_data in models_dict.items():
                        # Convert model_details dict to ModelDetails object first
                        if 'details' in model_data and isinstance(model_data['details'], dict):
                            model_data['details'] = ModelDetails(
                                **model_data['details'])
                        # Create Model object from dictionary
                        self.models[model_id] = Model(**model_data)

                    self.active_model_id = config.get(
                        'active_model_id', DEFAULT_MODEL_ID)
            else:
                # Create a default configuration
                self.models = {
                    DEFAULT_MODEL_ID: self.default_model
                }
                self._save_models_config()
        except Exception as e:
            print(f"Error loading models config: {e}")
            # Create a default configuration
            self.models = {
                DEFAULT_MODEL_ID: self.default_model
            }
            self._save_models_config()

    def _save_models_config(self):
        """Save models configuration to file."""
        # Convert Model objects to dictionaries for JSON serialization
        serializable_models = {}
        for model_id, model in self.models.items():
            # Use model_dump() for newer Pydantic or dict() for older versions
            try:
                model_dict = model.model_dump()  # Pydantic v2
            except AttributeError:
                model_dict = model.dict()  # Pydantic v1
            serializable_models[model_id] = model_dict

        config = {
            'models': serializable_models,
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
                self.models[DEFAULT_MODEL_ID] = self.default_model
                self._save_models_config()

        try:
            active_model = self.models[self.active_model_id]
            model_source = active_model.model
            print(f"Starting load of model from: {model_source}")

            # Set cache directory - first check environment variable, then use default
            cache_dir = os.environ.get("HF_HOME", "/root/.cache/huggingface")

            # Check if we're running on CPU
            is_cpu_device = self.device.type == 'cpu'
            if is_cpu_device:
                print("Running on CPU device - will use FP32 precision")

            # Configure download options with memory optimizations
            load_options = {
                # Only use float16 if we're on GPU and USE_FP16_PRECISION is True
                "torch_dtype": torch.float32 if is_cpu_device or FORCE_FP32_ON_CPU else (torch.float16 if USE_FP16_PRECISION else torch.float32),
                "use_safetensors": True,
                "variant": "fp16" if USE_FP16_PRECISION and not is_cpu_device else None,
                "cache_dir": cache_dir
            }

            # Check if model is already cached - handle different possible formats for the cache path
            # Split the model path by '/' to get repository owner and name
            path_parts = model_source.split('/')

            # Create different possible cache path formats and check them
            possible_cache_paths = []

            if len(path_parts) >= 2:
                # Format: models--owner--name
                owner = path_parts[0]
                name = path_parts[1]
                possible_cache_paths.append(f"models--{owner}--{name}")

                # For paths with more components
                if len(path_parts) > 2:
                    # Try concatenating all parts with dashes
                    all_parts = '--'.join(path_parts)
                    possible_cache_paths.append(f"models--{all_parts}")

            # Also try the original format
            model_id = model_source.replace("/", "--")
            if not model_id.startswith("models--"):
                possible_cache_paths.append(f"models--{model_id}")

            # Check each possible path
            model_exists = False
            cached_model_path = None

            for path in possible_cache_paths:
                test_path = os.path.join(cache_dir, "hub", path)
                snapshots_dir = os.path.join(test_path, "snapshots")
                refs_file = os.path.join(test_path, "refs", "main")

                if (os.path.exists(snapshots_dir) and
                    os.path.exists(refs_file) and
                    os.path.isdir(snapshots_dir) and
                        len(os.listdir(snapshots_dir)) > 0):
                    model_exists = True
                    cached_model_path = test_path
                    break

            if cached_model_path:
                print(f"Found cached model at: {cached_model_path}")
                print(f"Model exists in cache")
            else:
                print(f"Model not found in any expected cache locations")
                for path in possible_cache_paths:
                    test_path = os.path.join(cache_dir, "hub", path)
                    print(f"Checked path: {test_path} - Not found")

            # Always try without local_files_only first to ensure model loads
            try:
                print("Attempting to load model...")
                # Only use local files if we found it in cache
                load_options["local_files_only"] = model_exists

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

                # Get hardware stats to make intelligent decisions
                hardware_manager.update_all_memory_stats()
                free_memory_gb = hardware_manager.memory_stats.get(
                    hardware_manager.current_device_idx, {}).get('free_gb', 0)

                print(
                    f"Available GPU memory before model loading: {free_memory_gb:.2f} GB")

                # Make memory handling decisions based on actual available memory
                # Only offload if actually needed (less than 4GB available)
                should_use_sequential_offload = is_cpu_device or (
                    float(free_memory_gb) < 2.0 and ENABLE_SEQUENTIAL_CPU_OFFLOAD)
                should_use_regular_offload = is_cpu_device or (
                    float(free_memory_gb) < 4.0 and ENABLE_MODEL_CPU_OFFLOAD)

                # Apply memory optimizations based on configuration and available memory
                if should_use_sequential_offload:
                    print(
                        f"Enabling sequential CPU offloading due to low memory: {free_memory_gb:.2f}GB available")
                    from accelerate import cpu_offload
                    for component in [self.active_pipeline.unet, self.active_pipeline.vae, self.active_pipeline.text_encoder]:
                        if component is not None:
                            cpu_offload(component, self.device)
                elif should_use_regular_offload:
                    print(
                        f"Enabling model CPU offloading due to lower memory: {free_memory_gb:.2f}GB available")
                    self.active_pipeline.enable_model_cpu_offload()
                else:
                    # Standard device placement - enough memory for full GPU utilization
                    print(
                        f"Sufficient memory for full GPU model: {free_memory_gb:.2f}GB available")
                    self.active_pipeline = self.active_pipeline.to(self.device)
                    self.active_pipeline.enable_attention_slicing()
                    print("Using full GPU with attention slicing for efficiency")

                # Add memory optimization parameters
                if ENABLE_MEMORY_EFFICIENT_ATTENTION and not is_cpu_device:
                    print(
                        "Attempting to enable memory efficient attention with xformers")
                    try:
                        self.active_pipeline.enable_xformers_memory_efficient_attention()
                        print(
                            "Successfully enabled xformers memory efficient attention")
                    except (ImportError, ModuleNotFoundError):
                        print(
                            "Warning: xformers not available, falling back to default attention mechanism")
                    except Exception as e:
                        print(f"Warning: Failed to enable xformers: {e}")
                        print("Falling back to default attention mechanism")

                # Enable VAE slicing if configured (reduces memory during inference)
                if ENABLE_VAE_SLICING and hasattr(self.active_pipeline, 'enable_vae_slicing'):
                    print("Enabling VAE slicing for memory efficiency")
                    self.active_pipeline.enable_vae_slicing()

                # Empty cache to free up memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Check available memory after model loading
                hardware_manager.update_all_memory_stats()
                free_memory_after_gb = hardware_manager.memory_stats.get(
                    hardware_manager.current_device_idx, {}).get('free_gb', 0)
                print(
                    f"Available GPU memory after model loading: {free_memory_after_gb:.2f} GB")
                print(
                    f"Memory used by model: {float(free_memory_gb) - float(free_memory_after_gb):.2f} GB")

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
                # Get the correct source path for the LoRA
                lora_source = lora.model  # Now correctly contains the repo path, not the UUID

                # Default weight if not specified
                lora_weight = lora.details.weight or DEFAULT_LORA_WEIGHT

                print(
                    f"Loading LoRA: {lora.name} (weight: {lora_weight}) from {lora_source}")

                # Apply the LoRA weights directly using the simplified pattern
                try:
                    # Verify we're using a valid repository path and not a UUID
                    if lora_source and "/" in lora_source:
                        self.active_pipeline.load_lora_weights(lora_source)

                        # If the model supports adapter weights, set the weight
                        if hasattr(self.active_pipeline, "fuse_lora"):
                            # Some models use fuse_lora with a scale parameter
                            print(f"Fusing LoRA with scale {lora_weight}")
                            self.active_pipeline.fuse_lora(lora_weight)
                        elif hasattr(self.active_pipeline, "set_adapter_strength"):
                            print(f"Setting adapter strength to {lora_weight}")
                            self.active_pipeline.set_adapter_strength(
                                lora_weight)

                        print(f"Successfully applied LoRA: {lora.name}")
                    else:
                        # This is a UUID or invalid path - log clearly what went wrong
                        print(
                            f"ERROR: Invalid LoRA path: '{lora_source}'. LoRA paths should be in the format 'owner/model-name'")
                        print(
                            f"Please update the LoRA configuration for '{lora.name}' to use a valid Hugging Face repository path")
                except Exception as load_err:
                    if "PEFT backend is required for this method" in str(load_err):
                        print(
                            f"ERROR: PEFT backend is required to load LoRA weights.")
                        print("Please install PEFT with: pip install -U peft")
                        break
                    else:
                        raise

            except Exception as e:
                print(f"Error applying LoRA {lora.name}: {e}")

        # Clear CUDA cache after loading LoRAs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_models(self) -> List[Model]:
        """Get list of all available models."""
        return [model for model in self.models.values()]

    def get_model(self, model_id: str) -> Optional[Model]:
        """Get details of a specific model."""
        return self.models.get(model_id)

    def add_model(self, name: str, source: str, description: Optional[str] = None) -> Model:
        """Add a new model to the configuration."""

        # Add the model to the configuration
        self.models[source] = Model(
            name=name,
            model=source,
            modified_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            size=0,  # Size can be updated later
            digest='',  # Digest can be updated later
            details=ModelDetails(
                parent_model='',
                format='gguf',  # Default format, can be updated later
                family='llama',  # Default family, can be updated later
                families=[],
                parameter_size='7.2B',  # Default size, can be updated later
                quantization_level='Q4_0',  # Default quantization level
                specialization=None,  # Can be set later if needed
                description=description or ''  # Optional description
            )
        )

        # Save the updated configuration
        self._save_models_config()

        return self.models[source]

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

            # Load the model
            try:
                self._save_models_config()
                return True
            except Exception as e:
                # If loading fails, revert to previous model
                print(f"Failed to load model {model_id}: {str(e)}")
                print("Reverting to default model")
                self.active_model_id = DEFAULT_MODEL_ID
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
