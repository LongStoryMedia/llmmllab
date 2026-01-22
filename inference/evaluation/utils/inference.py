"""
Inference engine for benchmarks.
Provides a clean, no-fallback way to interact with model inference backends.

This module follows the principle of failing fast and clearly:
- No mock implementations or fallbacks
- Explicit error propagation
- Clear error messages

If the inference fails, it will raise appropriate exceptions that should be
handled by the calling code if desired. This ensures that any failures in the
evaluation pipeline are visible and not masked by mock data.
"""

import os
import datetime
from typing import Dict, Any
import logging

# Import the required modules directly
from runner import pipeline_factory
from models import MessageContent, Message, MessageRole, default_model_profiles
from models import MessageContentType


class InferenceEngine:
    """Handles model inference for benchmarks."""

    def __init__(self):
        """Initialize the inference engine."""
        self.logger = logging.getLogger("evaluation.utils.inference")

    # Mock inference method removed as per requirement to fail clearly instead of mocking

    async def run_single_inference(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int = 10000,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Run inference using the pipeline factory with no fallback.

        Args:
            model_id: The model identifier
            prompt: The input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Dict with response and metrics

        Raises:
            Exception: Any exception from the pipeline processing
        """
        # Set benchmark mode for evaluation runs
        os.environ["BENCHMARK_MODE"] = "true"

        # Create message for inference
        message = Message(
            role=MessageRole.USER,
            content=[MessageContent(type=MessageContentType.TEXT, text=prompt)],
            id=999,
            conversation_id=999,
            created_at=datetime.datetime.now(tz=datetime.timezone.utc),
        )

        # Get default model profile and configure for benchmarks
        model_profile = default_model_profiles.DEFAULT_PRIMARY_PROFILE
        model_profile.model_name = model_id
        model_profile.parameters.num_ctx = max_tokens

        # Get pipeline
        pipe = pipeline_factory.get_pipeline(model_profile)

        # Initialize response variables
        full_response = ""
        final_metrics = {}

        try:
            async for response in pipe.run([message]):
                if (
                    response.message
                    and response.message.content
                    and len(response.message.content) > 0
                ):
                    content = response.message.content[0].text or ""
                    # print(content, end="", flush=True)
                    full_response += content

                if response.done:
                    final_metrics = {
                        "total_duration": getattr(response, "total_duration", 0),
                        "load_duration": getattr(response, "load_duration", 0),
                        "eval_count": getattr(response, "eval_count", 0),
                    }
                    break

        except Exception as e:
            self.logger.error(f"Inference error: {str(e)}")
            raise RuntimeError(f"Failed to run inference: {str(e)}") from e

        # Add response to metrics
        final_metrics["response"] = full_response
        return final_metrics
