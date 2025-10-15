"""
Chat model factory for creating BaseChatModel implementations.
"""

import logging
from typing import Dict, Optional

from langchain_core.language_models import BaseChatModel

from models import Model, ModelProfile, CircuitBreakerConfig
from models.default_configs import DEFAULT_CIRCUIT_BREAKER_CONFIG


class ChatModelFactory:
    """
    Factory for creating BaseChatModel implementations.

    This factory returns BaseChatModel instances that can be used directly
    with LangChain's create_agent() and other agent construction utilities.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("ChatModelFactory initialized")

    def create_chat_model(
        self,
        model: Model,
        profile: ModelProfile,
        user_circuit_breaker: Optional[CircuitBreakerConfig] = None,
    ) -> Optional[BaseChatModel]:
        """Create a BaseChatModel implementation for the given model."""
        try:
            self.logger.info(
                f"Creating chat model for {model.name} (task: {model.task})"
            )

            # Only handle text-to-text tasks for chat models
            if not model.task.endswith("TextToText"):
                self.logger.error(f"Unsupported task type for chat model: {model.task}")
                return None

            return self._create_text_chat_model(model, profile, user_circuit_breaker)

        except Exception as e:
            self.logger.error(f"Error creating chat model for {model.name}: {e}")

            # Log specific error types for better debugging
            if "unknown model architecture" in str(e):
                self.logger.error(
                    f"Model {model.name} uses unsupported architecture - consider updating llama.cpp or using a different model"
                )
            elif "Failed to create llama_context" in str(e):
                self.logger.error(
                    f"Model {model.name} failed to load - may be corrupted or incompatible"
                )
            elif "validation error" in str(e).lower():
                self.logger.error(f"Model {model.name} configuration validation failed")

            return None

    def _create_text_chat_model(
        self,
        model: Model,
        profile: ModelProfile,
        user_circuit_breaker: Optional[CircuitBreakerConfig] = None,
    ) -> Optional[BaseChatModel]:
        """Create a text-to-text chat model (BaseChatModel implementation)."""
        self.logger.info(
            f"Creating text chat model for model: {model.name}, pipeline: {model.pipeline}"
        )

        if model.pipeline == "Qwen3Pipe":
            self.logger.info("Creating Qwen chat model")
            from .pipelines.txt2txt.qwen3moe import Qwen3Moe

            try:
                chat_model = Qwen3Moe(model, profile)
                self.logger.info("Successfully created Qwen3Moe")
                return chat_model
            except Exception as e:
                self.logger.error(f"Qwen3Moe creation failed: {e}")
                raise

        elif model.pipeline == "LlamaChatSummPipe":
            self.logger.info("Creating Llama Chat Summary model")
            from .pipelines.txt2txt.llamachatsum import LlamaChatSummPipe
            
            try:
                chat_model = LlamaChatSummPipe(model, profile)
                self.logger.info("Successfully created LlamaChatSummPipe")
                return chat_model
            except Exception as e:
                self.logger.error(f"LlamaChatSummPipe creation failed: {e}")
                raise

        elif model.pipeline == "OpenAiGptOssPipe":
            self.logger.info("Creating OpenAI GPT OSS model")
            from .pipelines.txt2txt.openai_gpt_oss import OpenAIGptOssPipeline
            
            try:
                chat_model = OpenAIGptOssPipeline(model, profile)
                self.logger.info("Successfully created OpenAIGptOssPipeline")
                return chat_model
            except Exception as e:
                self.logger.error(f"OpenAIGptOssPipeline creation failed: {e}")
                raise

        elif model.pipeline == "Qwen25VLGGUFPipeline":
            self.logger.info("Creating Qwen25VL model")
            from .pipelines.imgtxt2txt.qwen25_vl import Qwen25VLPipeline
            
            try:
                chat_model = Qwen25VLPipeline(model, profile)
                self.logger.info("Successfully created Qwen25VLPipeline")
                return chat_model
            except Exception as e:
                self.logger.error(f"Qwen25VLPipeline creation failed: {e}")
                raise

        self.logger.warning(
            f"No chat model implementation for pipeline: {model.pipeline}"
        )
        return None


# Create global chat model factory instance
chat_model_factory = ChatModelFactory()
