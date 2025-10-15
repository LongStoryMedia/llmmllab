"""
Chat Model Factory for creating BaseChatModel instances.
Provides clean interface for retrieving chat models from the pipeline factory.
"""

import logging
from typing import Optional

from langchain_core.language_models import BaseChatModel
from models import Model, ModelProfile, CircuitBreakerConfig
from .pipeline_factory import pipeline_factory


class ChatModelFactory:
    """Factory for creating BaseChatModel instances from the pipeline factory."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("ChatModelFactory initialized")

    def create_chat_model(
        self,
        model: Model,
        profile: ModelProfile,
        circuit_breaker: Optional[CircuitBreakerConfig] = None,
    ) -> Optional[BaseChatModel]:
        """
        Create a BaseChatModel instance from the pipeline factory.

        Args:
            model: Model configuration
            profile: Model profile for runtime settings
            circuit_breaker: Optional circuit breaker configuration

        Returns:
            BaseChatModel instance or None if creation fails
        """
        try:
            # Use the pipeline factory to create the chat model
            pipeline = pipeline_factory.get_pipeline(profile)
            
            # Ensure it's a chat model
            if isinstance(pipeline, BaseChatModel):
                return pipeline
            else:
                self.logger.error(f"Pipeline for {model.name} is not a BaseChatModel: {type(pipeline)}")
                return None

        except Exception as e:
            self.logger.error(f"Failed to create chat model for {model.name}: {e}")
            return None


# Global chat model factory instance
chat_model_factory = ChatModelFactory()