"""
Composer interface to runner service.

This module provides the interface for the composer app to communicate
with the runner service. Currently uses HTTP, but can be ported to gRPC
in the future for better performance and scalability.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel

from composer.models import (
    Message,
    MessageRole,
    MessageContentType,
    MessageContent,
    SearchResult,
    SearchTopicSynthesis,
)
from composer.utils.logging import llmmllogger


class RunnerResponse(BaseModel):
    """Standard response wrapper for runner service calls."""

    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None


class RunnerService:
    """
    Service interface for communicating with the runner application.

    The composer app should use this interface to:
    - Execute pipelines (text, image, embeddings)
    - Get model information
    - Manage model execution
    """

    def __init__(self, base_url: str = "http://runner:8000"):
        """
        Initialize runner service client.

        Args:
            base_url: Base URL of the runner service
        """
        self.base_url = base_url
        self.logger = llmmllogger.bind(module="ComposerRunner")

    async def execute_pipeline(
        self,
        pipeline_name: str,
        model_name: str,
        input_data: Dict[str, Any],
        **kwargs
    ) -> RunnerResponse:
        """
        Execute a pipeline using the runner service.

        Args:
            pipeline_name: Name of the pipeline to execute
            model_name: Name of the model to use
            input_data: Input data for the pipeline
            **kwargs: Additional pipeline-specific arguments

        Returns:
            RunnerResponse containing the pipeline output
        """
        try:
            # TODO: Implement HTTP call to runner service
            # For now, return a placeholder
            self.logger.debug(
                f"Executing pipeline {pipeline_name} with model {model_name}"
            )
            return RunnerResponse(success=True, data=input_data)
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            return RunnerResponse(success=False, error=str(e))

    async def get_model_info(self, model_id: str) -> RunnerResponse:
        """
        Get information about a specific model.

        Args:
            model_id: ID of the model to query

        Returns:
            RunnerResponse containing model information
        """
        try:
            # TODO: Implement HTTP call to runner service
            self.logger.debug(f"Getting info for model {model_id}")
            return RunnerResponse(success=True, data={})
        except Exception as e:
            self.logger.error(f"Failed to get model info: {e}")
            return RunnerResponse(success=False, error=str(e))

    async def health_check(self) -> bool:
        """
        Check if the runner service is healthy.

        Returns:
            True if the runner service is healthy, False otherwise
        """
        try:
            # TODO: Implement health check endpoint
            return True
        except Exception:
            return False
