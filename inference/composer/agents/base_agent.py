"""
Base Agent class providing common functionality for all workflow agents.
Provides node metadata injection, logging setup, and common error handling patterns.
"""

from typing import Optional, Any, Dict
from abc import ABC, abstractmethod

from models import NodeMetadata, ModelProfile
from runner import PipelineFactory
from utils.logging import llmmllogger
from composer.core.errors import NodeExecutionError


class BaseAgent(ABC):
    """
    Base class for all workflow agents providing common functionality.

    This base class provides:
    - Node metadata injection for workflow tracking
    - Consistent logging setup with component binding
    - Common error handling patterns
    - Shared initialization patterns

    All agent classes should inherit from this base class to ensure consistent
    behavior across the workflow system.
    """

    def __init__(
        self,
        pipeline_factory: PipelineFactory,
        profile: ModelProfile,
        node_metadata: NodeMetadata,
        component_name: Optional[str] = None,
    ):
        """
        Initialize base agent with required dependencies.

        Args:
            pipeline_factory: Factory for creating pipelines
            profile: Model profile for agent operations
            node_metadata: Node metadata for workflow tracking
            component_name: Optional component name for logging. If not provided,
                          uses the class name.
        """
        # Set up component-specific logging
        component = component_name or self.__class__.__name__
        self.logger = llmmllogger.logger.bind(component=component)

        # Store required dependencies
        self.pipeline_factory = pipeline_factory
        self.profile = profile
        self._node_metadata = node_metadata

        # Update logger context with node information
        self.logger = self.logger.bind(
            node_name=node_metadata.node_name,
            node_id=node_metadata.node_id,
            node_type=node_metadata.node_type,
            user_id=node_metadata.user_id,
            conversation_id=node_metadata.conversation_id,
        )

        # Additional metadata for debugging and tracking
        self._execution_context: Dict[str, Any] = {}

        self.logger.debug(
            f"Initialized {component}",
            node_name=node_metadata.node_name,
            model_name=profile.model_name,
        )

    @abstractmethod
    async def execute_pipeline(self, stream: bool = False, **kwargs) -> Any:
        """
        Execute the agent's pipeline with streaming option and custom parameters.

        This abstract method must be implemented by derived agent classes to define
        their specific pipeline execution logic. The method should handle both
        streaming and non-streaming execution modes.

        Args:
            stream: Whether to enable streaming mode (defaults to False)
            **kwargs: Additional parameters specific to the derived agent's
                     pipeline requirements (e.g., messages, user_id, tools, etc.)

        Returns:
            Pipeline execution result - type depends on the specific agent implementation

        Raises:
            NodeExecutionError: If pipeline execution fails
        """
        pass

    def inject_node_metadata(self, metadata: NodeMetadata) -> None:
        """
        Update node metadata for workflow execution tracking.

        This method allows workflow nodes to update execution context and metadata
        during agent execution for debugging, logging, and tracking purposes.
        Since metadata is provided at initialization, this method updates the
        existing metadata with new information.

        Args:
            metadata: NodeMetadata object containing updated execution context
        """
        self._node_metadata = metadata

        # Update logger context with new node information
        self.logger = self.logger.bind(
            node_name=metadata.node_name,
            node_id=metadata.node_id,
            node_type=metadata.node_type,
            user_id=metadata.user_id,
            conversation_id=metadata.conversation_id,
        )

        self.logger.debug(
            "Node metadata updated",
            node_name=metadata.node_name,
            node_type=metadata.node_type,
            execution_time=metadata.execution_time.isoformat(),
        )

    def get_node_metadata(self) -> Optional[NodeMetadata]:
        """
        Get the currently injected node metadata.

        Returns:
            NodeMetadata object if metadata has been injected, None otherwise
        """
        return self._node_metadata

    def update_execution_context(self, **context: Any) -> None:
        """
        Update execution context with additional metadata.

        This method allows agents to add additional context information
        that may be useful for debugging or tracking purposes.

        Args:
            **context: Key-value pairs to add to execution context
        """
        self._execution_context.update(context)
        self.logger.debug("Execution context updated", **context)

    def get_execution_context(self) -> Dict[str, Any]:
        """
        Get the current execution context.

        Returns:
            Dictionary containing execution context metadata
        """
        return self._execution_context.copy()

    def _log_operation_start(self, operation: str, **kwargs) -> None:
        """
        Log the start of an operation with context.

        Args:
            operation: Name of the operation being started
            **kwargs: Additional context to log
        """
        context = {
            "operation": operation,
            **kwargs,
        }

        # Add node metadata context if available
        if self._node_metadata:
            context.update(
                {
                    "node_name": self._node_metadata.node_name,
                    "user_id": self._node_metadata.user_id,
                    "conversation_id": self._node_metadata.conversation_id,
                }
            )

        self.logger.info(f"Starting {operation}", **context)

    def _log_operation_success(self, operation: str, **kwargs) -> None:
        """
        Log successful completion of an operation.

        Args:
            operation: Name of the operation that completed
            **kwargs: Additional context to log
        """
        context = {
            "operation": operation,
            **kwargs,
        }

        self.logger.info(f"Completed {operation}", **context)

    def _log_operation_error(self, operation: str, error: Exception, **kwargs) -> None:
        """
        Log operation failure with error details.

        Args:
            operation: Name of the operation that failed
            error: Exception that occurred
            **kwargs: Additional context to log
        """
        context = {
            "operation": operation,
            "error": str(error),
            "error_type": type(error).__name__,
            **kwargs,
        }

        # Add node metadata context if available
        if self._node_metadata:
            context.update(
                {
                    "node_name": self._node_metadata.node_name,
                    "user_id": self._node_metadata.user_id,
                    "conversation_id": self._node_metadata.conversation_id,
                }
            )

        self.logger.error(f"Failed {operation}", **context)

    def _handle_node_error(self, operation: str, error: Exception, **context) -> None:
        """
        Handle and wrap errors in NodeExecutionError with consistent logging.

        Args:
            operation: Name of the operation that failed
            error: Original exception
            **context: Additional context for logging
        """
        self._log_operation_error(operation, error, **context)

        # Create descriptive error message
        error_msg = f"{operation} failed: {error}"

        # Include node context if available
        if self._node_metadata:
            error_msg = f"[{self._node_metadata.node_name}] {error_msg}"

        raise NodeExecutionError(error_msg) from error

    def _get_user_context(self) -> Dict[str, Any]:
        """
        Get user context from node metadata.

        Returns:
            Dictionary containing user and conversation context
        """
        if not self._node_metadata:
            return {}

        return {
            "user_id": self._node_metadata.user_id,
            "conversation_id": self._node_metadata.conversation_id,
        }
