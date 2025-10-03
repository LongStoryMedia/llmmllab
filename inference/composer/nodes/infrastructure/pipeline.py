"""
Pipeline Node for LangGraph workflows.
Wraps LLM pipeline execution for chat model operations within workflows.
"""

# No typing imports needed for this module

from models import ChatResponse, LangChainMessage, ModelProfileType
from utils.model_profile import get_model_profile
from composer.graph.state import WorkflowState
from composer.monitoring.logging import composer_logger
from composer.core.errors import NodeExecutionError


class PipelineNode:
    """
    Wraps chat-model execution as a graph node.

    Handles both streaming and non-streaming execution based on configuration.
    Retrieves model profiles internally from shared data layer using user_id.
    """

    def __init__(
        self, pipeline_factory, profile_type: ModelProfileType, stream: bool = False
    ):
        """
        Initialize pipeline node.

        Args:
            pipeline_factory: Factory for creating pipeline instances
            profile_type: Model profile type (Primary, Analysis, etc.)
            stream: Whether to enable streaming responses
        """
        self.pipeline_factory = pipeline_factory
        self.profile_type = profile_type
        self.stream = stream
        self.logger = composer_logger.logger.bind(component="PipelineNode")

    async def __call__(self, state: WorkflowState) -> WorkflowState:
        """
        Execute pipeline node with grammar-constrained generation.

        Args:
            state: Current workflow state

        Returns:
            Updated workflow state with response
        """
        try:
            # Retrieve user configuration from shared data layer
            user_id = getattr(state, "user_id", None)
            if not user_id:
                raise NodeExecutionError("User ID required for pipeline execution")

            # Lazy imports to avoid circular dependencies
            try:
                # Get model profile for this task type using user_id from state
                model_profile = await get_model_profile(user_id, self.profile_type)
            except ImportError as ie:
                self.logger.warning(f"Model profile utility not available: {ie}")
                model_profile = None

            self.logger.info(
                "Executing pipeline node",
                user_id=user_id,
                profile_type=self.profile_type.value,
                streaming=self.stream,
                model=model_profile.model_name if model_profile else "unknown",
            )

            # Create pipeline instance (placeholder for actual implementation)
            # TODO: Implement proper pipeline factory integration
            if self.pipeline_factory:
                pipeline = await self.pipeline_factory.get_pipeline(
                    model_profile, ChatResponse, streaming=self.stream
                )

                if self.stream:
                    # For streaming nodes: this will be handled by LangGraph streaming
                    # For now, just process non-streaming
                    response = await pipeline.invoke({"messages": state.messages})
                else:
                    # For non-streaming: complete response
                    response = await pipeline.invoke({"messages": state.messages})

                # Add response to state messages
                assistant_message = LangChainMessage(
                    type="ai",
                    content=getattr(response, "content", "Response generated"),
                    tool_calls=getattr(response, "tool_calls", None),
                )
                state.messages.append(assistant_message)
            else:
                # Fallback when pipeline factory not available
                fallback_message = LangChainMessage(
                    type="ai",
                    content="Pipeline factory not configured - this is a placeholder response.",
                )
                state.messages.append(fallback_message)

            return state

        except Exception as e:
            self.logger.error(
                "Pipeline node execution failed",
                user_id=getattr(state, "user_id", "unknown"),
                error=str(e),
                profile_type=self.profile_type.value,
            )
            raise NodeExecutionError(f"Pipeline execution failed: {e}") from e