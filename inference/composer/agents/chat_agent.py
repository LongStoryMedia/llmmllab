"""
Chat Agent for LLM chat model operations.
Provides core business logic for chat completions, streaming, and tool integration.
"""

from runner import PipelineFactory
from models import (
    ChatResponse,
    ModelProfile,
    PipelinePriority,
)
from .base_agent import BaseAgent


class ChatAgent(BaseAgent[ChatResponse]):
    """
    Chat Agent for LLM chat model operations with streaming and tool support.

    Provides core business logic for chat completions, handling both streaming
    and non-streaming execution, tool integration, and response processing.
    Supports model profile configuration and circuit breaker integration.
    """

    def __init__(
        self,
        pipeline_factory: PipelineFactory,
        profile: ModelProfile,
        priority: PipelinePriority = PipelinePriority.MEDIUM,
    ):
        """
        Initialize chat agent with dependency injection.

        Args:
            pipeline_factory: Factory for creating chat pipelines
            profile: Model profile for chat operations
            node_metadata: Node execution metadata for tracking
            priority: Pipeline execution priority
            stream: Whether to enable streaming responses by default
        """
        super().__init__(pipeline_factory, profile, "ChatAgent")
        self.priority = priority
