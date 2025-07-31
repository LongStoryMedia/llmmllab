import datetime
import json
import logging
import os
import time
import threading
from typing import Dict, List, Optional, Tuple
import inference.server.config as config
from models import Model, LoraWeight, ModelDetails
from .base_pipeline import BasePipeline

# Basic Formula: A rough estimate for VRAM usage (in GB) is: Number of Parameters (in billions) * (Precision / 8) * 1.2.
# Inference Rule of Thumb (Half Precision): For LLMs in half precision (FP16), estimate approximately 2GB of GPU memory per 1B parameters.
# Inference Estimate: VRAM (in GB) = 2x model parameters (in billions) + 1x context length (in thousands).
# Training Estimate: VRAM (in GB) = 40x model parameters (in billions).


class PipelineCacheEntry:
    """
    A class to store BasePipeline cache entries with timeout information.
    """

    def __init__(self, pipeline: BasePipeline, timestamp: Optional[float] = None):
        self.pipeline = pipeline
        self.last_accessed = timestamp if timestamp is not None else time.time()


class PipelineFactory:
    """
    Factory class to create pipelines based on model configurations.

    This class implements a caching mechanism that keeps pipeline instances in memory for a configurable
    period (default 5 minutes). After the timeout period, pipelines that haven't been accessed are
    automatically cleaned up - their resources are moved to CPU and then removed from memory to free up GPU
    resources. The cache timeout can be configured using the set_cache_timeout method.

    Key features:
    - Automatic caching of BasePipeline instances by model ID
    - Configurable timeout period (default: 5 minutes)
    - Automatic cleanup of expired pipelines
    - Resource management for GPU memory
    """

    # Cache for loaded pipelines (only BasePipeline instances)
    _pipelines: Dict[str, PipelineCacheEntry] = {}
    _available_models: Dict[str, Model] = {}

    # Cache configuration
    _cache_timeout = 300  # Default cache timeout: 5 minutes (in seconds)
    _cleanup_thread = None
    _cleanup_lock = threading.RLock()  # Lock for thread-safe cache access

    def __init__(self):
        """
        Initialize the PipelineFactory with empty model and tokenizer caches.
        Loads all available models from the models.json file and starts the cleanup thread.
        """
        self.logger = logging.getLogger(__name__)
        self._load_available_models()
        # Start the cleanup thread
        self._start_cleanup_thread()

    def _load_available_models(self):
        """
        Load all models from the model service into the available_models dictionary.
        """
        try:
            models_file = "/app/models.json"
            if not os.path.exists(models_file):
                self.logger.error(f"Models config file not found: {models_file}")
                return

            with open(models_file, "r") as f:
                models_data: List[dict] = json.load(f)

            for model_data in models_data:
                # Create Model instance
                lora_weights: List[dict] = model_data.get("lora_weights", [])
                loras: List[LoraWeight] = []
                if len(lora_weights) > 0:
                    for lora in lora_weights:
                        # Load LoRA weights if available
                        if lora is not None:
                            lora_weight_instance = LoraWeight(
                                id=lora.get("id", ""),
                                name=lora.get("name", ""),
                                weight_name=lora.get("weight_name", ""),
                                adapter_name=lora.get("adapter_name", ""),
                                parent_model=lora.get("parent_model", ""),
                            )
                            loras.append(lora_weight_instance)

                details_dict: dict = model_data.get("details", {})
                details = ModelDetails(
                    parent_model=details_dict.get("parent_model", ""),
                    format=details_dict.get("format", ""),
                    family=details_dict.get("family", ""),
                    families=details_dict.get("families", []),
                    parameter_size=details_dict.get("parameter_size", ""),
                    quantization_level=details_dict.get("quantization_level", ""),
                    specialization=details_dict.get("specialization", ""),
                    dtype=details_dict.get("dtype", "bf16"),
                    precision=details_dict.get("precision", "fp16"),
                    weight=details_dict.get("weight", 1.0),
                    gguf_file=details_dict.get("gguf_file", None),
                    description=details_dict.get("description", None),
                )

                model = Model(
                    id=model_data.get("id"),
                    name=model_data["name"],
                    model=model_data["model"],
                    modified_at=model_data["modified_at"],
                    size=model_data["size"],
                    digest=model_data["digest"],
                    pipeline=model_data.get("pipeline"),
                    lora_weights=loras,
                    details=details,
                    task=model_data.get("task", "TextToText"),
                )

                self._available_models[model_data["id"]] = model

            self.logger.info(f"Loaded {len(self._available_models)} models from config")
        except Exception as e:
            self.logger.error(f"Error loading models config: {e}")

        self.logger.info(
            f"Loaded {len(self._available_models)} models into available models dictionary"
        )

    def get_pipeline(self, model_id: str) -> Tuple[BasePipeline, float]:
        """
        Get the appropriate pipeline for the given model.
        If a pipeline for this model is already cached and not expired,
        returns the cached pipeline and updates its last accessed timestamp.
        Otherwise, creates a new pipeline and caches it.

        Args:
            model_id (str): The ID of the model to create a pipeline for.

        Returns:
            BasePipeline: The created pipeline instance or None if the model type is unsupported.
        """
        start_time = datetime.datetime.now(tz=datetime.timezone.utc)

        # Check if we already have this pipeline cached
        with self._cleanup_lock:
            if model_id in self._pipelines:
                self.logger.info(f"Using cached pipeline for model: {model_id}")
                # Update the last accessed timestamp
                self._pipelines[model_id].last_accessed = time.time()
                end_time = datetime.datetime.now(tz=datetime.timezone.utc)
                return (
                    self._pipelines[model_id].pipeline,
                    (end_time - start_time).total_seconds() * 1000,
                )

        # Check if model exists in our available models dictionary
        self.logger.info(f"Available models: {list(self._available_models.keys())}")
        if not self._available_models:
            self.logger.error(
                "Available models dictionary is empty. Trying to reload from model service."
            )
            # Try to load models if dictionary is empty
            # if model_service.models is None:
            #     raise RuntimeError("Model service is not initialized or models are not loaded.")
            # self._available_models = {
            #     model_id: model for model_id, model in model_service.models.items()
            # }

        if model_id not in self._available_models:
            raise RuntimeError(
                f"Model with ID '{model_id}' not found in available models."
            )

        model = self._available_models[model_id]
        self.logger.info(f"Creating pipeline for model: {model.name} (ID: {model.id})")

        pipe = self.create_pipeline(model)

        if pipe is None:
            raise RuntimeError(f"Failed to create pipeline for model {model.name}.")

        # Cache the pipeline for future use
        with self._cleanup_lock:
            self._pipelines[model_id] = PipelineCacheEntry(pipe)

        end_time = datetime.datetime.now(tz=datetime.timezone.utc)
        return (pipe, (end_time - start_time).total_seconds() * 1000)

    def create_pipeline(self, model: Model) -> Optional[BasePipeline]:
        """
        Factory method to create and return the appropriate pipeline based on the model configuration.
        Creates a BasePipeline-derived instance for the given model.

        Args:
            model (Model): The model configuration to create the pipeline for.

        Returns:
            BasePipeline: The created pipeline instance or None if the model type is unsupported.
        """
        pipe = None

        # Clear memory before loading new model
        # hardware_manager.clear_memory()

        if model.task.endswith("TextToText"):
            if model.pipeline == "GLM4VPipeline":
                self.logger.info(f"Creating GLM4V pipeline for model {model.name}")
                from .imgtxt2txt.glm4v import GLM4VPipe

                pipe = GLM4VPipe(model)
            elif model.pipeline == "GLM4VGGUFPipeline":
                self.logger.info(f"Creating GLM4V GGUF pipeline for model {model.name}")
                from .imgtxt2txt.glm4v_gguf import GLM4VGGUFPipe

                pipe = GLM4VGGUFPipe(model)
            elif model.pipeline == "Qwen25VLGGUFPipeline":
                self.logger.info(
                    f"Creating Qwen 2.5 VL GGUF pipeline for model {model.name}"
                )
                from .imgtxt2txt.qwen25_vl_gguf import Qwen25VLGGUFPipe

                pipe = Qwen25VLGGUFPipe(model)
            elif model.pipeline == "Qwen30A3BQ4KMPipe":
                self.logger.info(f"Creating Qwen GGUF pipeline for model {model.name}")
                from .txt2txt.qwen30a3b_q4km import QwenGGUFPipe

                pipe = QwenGGUFPipe(model)

        elif model.task == "TextToImage":
            if model.pipeline == "StableDiffusion3Pipeline":
                self.logger.info(f"Creating SD3 pipeline for model {model.name}")
                from .txt2img.sd3 import SD3Pipe

                pipe = SD3Pipe(model)
            elif model.pipeline == "StableDiffusionXLPipeline":
                self.logger.info(f"Creating SDXL pipeline for model {model.name}")
                from .txt2img.sdxl import SDXLPipe

                pipe = SDXLPipe(model)
            elif model.pipeline == "FluxPipeline":
                self.logger.info(f"Creating Flux pipeline for model {model.name}")
                from .txt2img.flux import FluxPipe

                pipe = FluxPipe(model)

        elif model.task == "ImageToImage":
            if model.pipeline == "StableDiffusionXLImg2ImgPipeline":
                self.logger.info(
                    f"Creating SDXL Img2Img pipeline for model {model.name}"
                )
                from .img2img.sdxl import SDXLRefinerPipe

                pipe = SDXLRefinerPipe(model)
            elif model.pipeline == "FluxKontextPipeline":
                self.logger.info(
                    f"Creating FluxKontext pipeline for model {model.name}"
                )
                from .img2img.flux import FluxKontextPipe

                pipe = FluxKontextPipe(model)
            elif model.pipeline == "StableDiffusionInstructPix2PixPipeline":
                self.logger.info(f"Creating Pix2Pix pipeline for model {model.name}")
                from .img2img.p2p import Pix2PixPipe

                pipe = Pix2PixPipe(model)

        elif model.task == "Embedding":
            if model.pipeline == "NomicEmbedTextPipe":
                self.logger.info(
                    f"Creating Nomic Embed Text pipeline for model {model.name}"
                )
                from .emb import NomicEmbedTextPipe

                pipe = NomicEmbedTextPipe(model)

        if pipe is None:
            self.logger.error(
                f"Unsupported pipeline type '{model.pipeline}' for model {model.name}"
            )
            return None

        return pipe

    def get_available_models(self) -> Dict[str, Model]:
        """
        Get a dictionary of all available models.

        Returns:
            Dict[str, Model]: Dictionary of model IDs to model objects
        """
        if not self._available_models:
            logger = logging.getLogger(__name__)
            logger.warning(
                "Available models dictionary is empty. Trying to reload from model service."
            )
            # Try to load models if dictionary is empty
            # if model_service.models is not None:
            #     self._available_models = {
            #         model_id: model for model_id, model in model_service.models.items()
            #     }
        return self._available_models

    def is_model_available(self, model_id: str) -> bool:
        """
        Check if a model with the given ID is available.

        Args:
            model_id (str): The ID of the model to check for.

        Returns:
            bool: True if the model is available, False otherwise.
        """
        return model_id in self.get_available_models()

    def set_cache_timeout(self, timeout_seconds: int) -> None:
        """
        Set the timeout duration for the pipeline cache.
        Pipelines that have not been accessed for this duration will be removed from
        the cache, moved to CPU, and their GPU memory will be freed.

        Args:
            timeout_seconds (int): Duration in seconds before cached pipelines are removed.
                                  Default is 300 seconds (5 minutes).
        """
        self._cache_timeout = timeout_seconds

    def _start_cleanup_thread(self):
        """
        Start a background thread to periodically clean up expired cache entries.
        """
        if self._cleanup_thread is None or not self._cleanup_thread.is_alive():
            self._cleanup_thread = threading.Thread(
                target=self._cleanup_cache_task, daemon=True
            )
            self._cleanup_thread.start()

    def _cleanup_cache_task(self):
        """
        Background task that periodically checks and removes expired cache entries.
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting pipeline cache cleanup thread")

        while True:
            time.sleep(60)  # Check every minute

            try:
                self._cleanup_expired_entries()
            except Exception as e:
                logger.error(f"Error in cache cleanup task: {str(e)}")

    def _cleanup_expired_entries(self) -> None:
        """
        Remove expired entries from the pipeline cache.
        """
        logger = logging.getLogger(__name__)
        current_time = time.time()
        models_to_remove = []

        with self._cleanup_lock:
            # Find expired entries
            for model_id, entry in self._pipelines.items():
                if current_time - entry.last_accessed > self._cache_timeout:
                    models_to_remove.append(model_id)

            # Remove expired entries
            for model_id in models_to_remove:
                logger.info(
                    f"Removing expired pipeline for model {model_id} from cache"
                )
                pipe_entry = self._pipelines.pop(
                    model_id, None
                )  # Clean up the pipeline's resources
                if pipe_entry and pipe_entry.pipeline:
                    self._cleanup_pipeline_resources(pipe_entry.pipeline)

            # Clear memory if we removed anything
            # if len(models_to_remove) > 0:
            #     from services.hardware_manager import hardware_manager
            #     hardware_manager.clear_memory(aggressive=True)

    def _cleanup_pipeline_resources(self, pipeline: "BasePipeline") -> None:
        """
        Clean up resources used by a BasePipeline instance by calling its __del__ method.
        This helps ensure GPU memory is properly freed when pipelines are removed from cache.

        Args:
            pipeline: The BasePipeline instance to clean up.
        """
        try:
            # Call the pipeline's __del__ method to clean up resources
            if pipeline is not None:
                self.logger.debug(
                    f"Calling __del__ method on {type(pipeline).__name__}"
                )
                pipeline.__del__()
        except Exception as e:
            self.logger.warning(f"Unexpected error during pipeline cleanup: {str(e)}")

    def clear_cache(self, model_id: Optional[str] = None) -> None:
        """
        Manually clear the pipeline cache and clean up resources.
        If model_id is provided, only that specific model's pipeline will be removed.
        Otherwise, all cached pipelines will be removed.

        Args:
            model_id (Optional[str]): The ID of the specific model to remove from cache.
                                      If None, all pipelines will be removed.
        """
        logger = logging.getLogger(__name__)

        with self._cleanup_lock:
            if model_id is not None:
                # Remove specific pipeline
                logger.info(
                    f"Manually removing pipeline for model {model_id} from cache"
                )
                pipe_entry = self._pipelines.pop(
                    model_id, None
                )  # Clean up the pipeline's resources
                if pipe_entry and pipe_entry.pipeline:
                    self._cleanup_pipeline_resources(pipe_entry.pipeline)
            else:
                # Remove all pipelines
                logger.info("Manually clearing all pipelines from cache")
                model_ids = list(self._pipelines.keys())

                for m_id in model_ids:
                    pipe_entry = self._pipelines.pop(m_id, None)
                    if pipe_entry and pipe_entry.pipeline:
                        self._cleanup_pipeline_resources(pipe_entry.pipeline)

        # Clear memory
        # from services.hardware_manager import hardware_manager
        # hardware_manager.clear_memory(aggressive=True)


pipeline_factory = PipelineFactory()
