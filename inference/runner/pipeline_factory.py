"""
Pipeline factory that routes local vs remote providers and delegates caching
to the PipelineCache API.
"""

from typing import Dict, Optional, Type, Union
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel
from models import (
    Model,
    ModelProvider,
    ModelTask,
    PipelinePriority,
)
from runner.pipelines.base import BasePipeline
from utils.logging import llmmllogger
from .pipeline_cache import pipeline_cache
from .utils.model_loader import ModelLoader


class PipelineFactory:
    """
    Factory for creating pipelines.

    Handles:
    - Routing local providers through the cached pipeline API
    - Creating transient pipelines for remote/API providers
    - Delegating all cache management to PipelineCache
    """

    def __init__(self, models_map: Dict[str, Model]):
        self.logger = llmmllogger.bind(component="PipelineFactory")

        self._available_models: Dict[str, Model] = ModelLoader().get_available_models()

        # Use the module-global cache singleton
        self.cache = pipeline_cache

        # Set self.models to the loaded models, with models_map as fallback
        self.models: Dict[str, Model] = (
            self._available_models if self._available_models else (models_map or {})
        )

        self.logger.info("PipelineFactory initialized with PipelineCache")

    def get_pipeline(
        self,
        model: Model,
        priority: PipelinePriority = PipelinePriority.NORMAL,
        grammar: Optional[Type[BaseModel]] = None,
        metadata: Optional[dict] = {},
    ) -> Union[BasePipeline, Embeddings]:
        model_id = model.name
        self.logger.debug(
            f"Requesting pipeline for model_id: {model_id}, priority: {priority}, grammar: {grammar}, metadata: {metadata}"
        )

        provider = getattr(model, "provider", None)

        # Local providers -> cached path via PipelineCache
        if provider in {
            ModelProvider.LLAMA_CPP,
            ModelProvider.STABLE_DIFFUSION_CPP,
        }:
            self.logger.info(
                f"Using LOCAL cached path for {model_id} (provider: {provider})"
            )
            return self.cache.get(
                model, priority, self.create_pipeline, grammar, metadata
            )

        # Remote / API providers -> create transient each call, no caching
        self.logger.info(
            f"Using REMOTE non-cached path for {model_id} (provider: {provider})"
        )
        pipeline = self.create_pipeline(model)
        if not pipeline:
            raise RuntimeError(
                f"Failed to create pipeline for model '{model.name}' (provider: {provider})"
            )
        self.logger.debug(
            f"Created transient pipeline for remote provider {provider} ({model.name})"
        )
        return pipeline

    def get_embedding_pipeline(
        self,
        model: Model,
        priority: PipelinePriority = PipelinePriority.NORMAL,
        metadata: Optional[dict] = None,
    ) -> Embeddings:
        model_id = model.name

        if model.task != "TextToEmbeddings":
            raise ValueError(
                f"Model '{model.name}' is not an embedding model (task: {model.task})"
            )

        if getattr(model, "provider", None) in {
            ModelProvider.LLAMA_CPP,
            ModelProvider.STABLE_DIFFUSION_CPP,
        }:

            def create_embed(m, _g=None, md=None):
                return self._create_embedding_pipeline(m, md)

            return self.cache.get(model, priority, create_embed, None, metadata)  # type: ignore

        # Remote / API providers -> create transient each call, no caching
        pipeline = self._create_embedding_pipeline(model)
        if not pipeline:
            raise RuntimeError(
                f"Failed to create embedding pipeline for model '{model.name}' (provider: {getattr(model, 'provider', 'unknown')})"
            )
        return pipeline

    def unlock_pipeline(self, model: Model) -> bool:
        return self.cache.unlock(model.name)

    def clear_cache(self, model: Model) -> None:
        self.cache.clear(model.name)

    def _get_model_by_id(self, model_id: str) -> Optional[Model]:
        if not self._available_models:
            self.logger.error("Available models dictionary is empty")
            return None
        if model_id not in self._available_models:
            self.logger.error(
                f"Model '{model_id}' not found. Available: {list(self._available_models.keys())}"
            )
            return None
        return self._available_models[model_id]

    def get_model_by_task(self, task: ModelTask) -> Optional[Model]:
        """Find the first available model matching the given task type."""
        for model in self._available_models.values():
            if model.task == task:
                return model
        self.logger.error(
            f"No model found for task '{task}'. Available: {[(m.name, m.task) for m in self._available_models.values()]}"
        )
        return None

    def create_pipeline(
        self,
        model: Model,
        grammar: Optional[Type[BaseModel]] = None,
        metadata: Optional[dict] = {},
    ) -> Optional[Union[BasePipeline, Embeddings]]:
        try:
            if (
                model.task == ModelTask.TEXTTOTEXT
                or model.task == ModelTask.VISIONTEXTTOTEXT
            ):
                self.logger.info(
                    f"Routing to _create_text_pipeline for model {model.name}"
                )
                return self._create_text_pipeline(model, grammar, metadata)
            if model.task == ModelTask.TEXTTOEMBEDDINGS:
                self.logger.info(
                    f"Routing to _create_embedding_pipeline for {model.name}"
                )
                return self._create_embedding_pipeline(model, metadata)
            if model.task == ModelTask.TEXTTOIMAGE:
                self.logger.info(f"Routing to _create_image_pipeline for {model.name}")
                return self._create_image_pipeline(model, metadata)
            if model.task == ModelTask.IMAGETOIMAGE:
                self.logger.info(
                    f"Routing to _create_image_to_image_pipeline for {model.name}"
                )
                return self._create_image_to_image_pipeline(model, metadata)
            self.logger.error(f"Unsupported task type: {model.task}")
            raise RuntimeError(f"Unsupported task type: {model.task}")
        except Exception as e:
            self.logger.error(f"Error creating pipeline for {model.name}: {e}")
            raise

    def _create_text_pipeline(
        self,
        model: Model,
        grammar: Optional[Type[BaseModel]] = None,
        metadata: Optional[dict] = {},
    ) -> BasePipeline:
        self.logger.info(
            f"Creating text pipeline for model: {model.name}, pipeline: {model.pipeline}, provider: {model.provider}"
        )

        match model.provider:
            case ModelProvider.LLAMA_CPP:
                from .pipelines.llamacpp.chat import (  # pylint: disable=import-outside-toplevel
                    ChatLlamaCppPipeline,
                )

                return ChatLlamaCppPipeline(model, grammar, metadata)
            case ModelProvider.OPENAI:
                from langchain_openai import (  # pylint: disable=import-outside-toplevel
                    ChatOpenAI,
                )
                from pydantic import SecretStr
                from config import OPENAI_API_KEY

                return ChatOpenAI(  # type: ignore[return-value]
                    model=model.name,
                    api_key=SecretStr(OPENAI_API_KEY),
                )
            case ModelProvider.ANTHROPIC:
                from langchain_anthropic import (  # pylint: disable=import-outside-toplevel
                    ChatAnthropic,
                )
                from pydantic import SecretStr
                from config import ANTHROPIC_API_KEY

                return ChatAnthropic(  # type: ignore[return-value]
                    model_name=model.name,  # type: ignore
                    api_key=SecretStr(ANTHROPIC_API_KEY),
                )
            case _:
                raise ValueError(f"Unsupported text provider: {model.provider}")

    def _create_embedding_pipeline(
        self,
        model: Model,
        metadata: Optional[dict] = {},
    ) -> Optional[Embeddings]:
        from .pipelines.llamacpp.embed import (  # pylint: disable=import-outside-toplevel
            EmbedLlamaCppPipeline,
        )

        return EmbedLlamaCppPipeline(model, metadata=metadata)

    def _create_image_pipeline(
        self,
        model: Model,
        metadata: Optional[dict] = {},
    ) -> Optional[BasePipeline]:
        if model.pipeline == "FluxPipeline":
            try:
                from .pipelines.txt2img.flux import (  # pylint: disable=import-outside-toplevel
                    FluxPipe,
                )

                return FluxPipe(model)  # pylint: disable=abstract-class-instantiated
            except Exception as e:
                self.logger.error(f"Failed to initialize FluxPipe: {e}")
                return None
        return None

    def _create_image_to_image_pipeline(
        self,
        model: Model,
        metadata: Optional[dict] = {},
    ) -> Optional[BasePipeline]:
        if model.pipeline == "FluxKontextPipeline":
            try:
                from .pipelines.img2img.flux import (  # pylint: disable=import-outside-toplevel
                    FluxKontextPipe,
                )

                return FluxKontextPipe(  # pylint: disable=abstract-class-instantiated
                    model
                )
            except Exception as e:
                self.logger.error(f"Failed to initialize FluxKontextPipe: {e}")
                return None

        if model.task == ModelTask.IMAGETO3D:
            try:
                from .pipelines.img23d.hunyuan3d import (  # pylint: disable=import-outside-toplevel
                    Hunyuan3DImageTo3DPipeline,
                )

                return Hunyuan3DImageTo3DPipeline(model)  # type: ignore
            except Exception as e:
                self.logger.error(
                    f"Failed to initialize Hunyuan3DImageTo3DPipeline: {e}"
                )

        return None


# Create global factory instance
pipeline_factory = PipelineFactory({})
