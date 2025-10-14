"""
Base Agent class providing common functionality for all workflow agents.
Provides node metadata injection, logging setup, and common error handling patterns.
"""

from typing import Optional, Any, Dict, TypeVar, Generic
from abc import ABC, abstractmethod

from models import NodeMetadata, ModelProfile
from runner import PipelineFactory
from utils.logging import llmmllogger
from composer.core.errors import NodeExecutionError

T = TypeVar("T")


class BaseAgent(ABC, Generic[T]):
    """
    Base class for all workflow agents providing common functionality.

    This base class provides:
    - Node metadata injection for workflow tracking
    - Consistent logging setup with component binding
    - Common error handling patterns
    - Shared initialization patterns
    - Generic typing for pipeline execution results

    All agent classes should inherit from this base class to ensure consistent
    behavior across the workflow system.

    Generic Type Parameter:
        T: The return type of the execute_pipeline method, specified by derived classes.
           Examples:
           - ChatAgent(BaseAgent[ChatResponse])
           - ClassifierAgent(BaseAgent[List[IntentAnalysis]])
           - EmbeddingAgent(BaseAgent[List[List[float]]])
           - SummarizationAgent(BaseAgent[str])

    Implementation Status:
        ✅ ChatAgent: Fully updated with new constructor and execute_pipeline
        ✅ ClassifierAgent: Fully updated - BaseAgent[List[IntentAnalysis]], execute_pipeline, constructor
        ✅ EmbeddingAgent: Fully updated - BaseAgent[List[List[float]]], execute_pipeline, constructor
        ✅ SummarizationAgent: Fully updated - BaseAgent[str], execute_pipeline, constructor
        ✅ EngineeringAgent: Fully updated - BaseAgent[str], execute_pipeline, constructor
        ✅ MemoryAgent: Fully updated - BaseAgent[List[Memory]], execute_pipeline, constructor

    Migration Pattern Applied:
        All agents now follow the consistent pattern:
        1. Generic type specification: BaseAgent[ReturnType]
        2. Constructor: super().__init__(pipeline_factory, profile, node_metadata, component_name)
        3. Abstract method: execute_pipeline(self, stream: bool = False, **kwargs) -> ReturnType
        4. Delegation: execute_pipeline delegates to existing core business methods

    Special Cases:
        - SummarizationAgent & MemoryAgent: Include additional dependencies in constructor
        - All agents maintain backward compatibility for their existing core methods
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
    async def execute_pipeline(self, stream: bool = False, **kwargs) -> T:
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
            Pipeline execution result - type T is specified by the derived agent class

        Raises:
            NodeExecutionError: If pipeline execution fails
        """

    def update_metadata(self, **kwargs) -> None:
        """
        Update node metadata and logger context with additional information.

        Args:
            **kwargs: Key-value pairs to update in node metadata and logger context
        """
        for key, value in kwargs.items():
            if hasattr(self._node_metadata, key):
                setattr(self._node_metadata, key, value)
                self.logger = self.logger.bind(**{key: value})
                self.logger.debug(f"Updated node metadata: {key}={value}")
            else:
                self.logger.warning(
                    f"Attempted to update unknown metadata field: {key}"
                )

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
