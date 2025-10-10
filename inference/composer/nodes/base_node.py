"""
Base node for all composer workflow nodes.

Provides common functionality including user ID validation, logger initialization,
user configuration access patterns, and standardized error handling.
"""

from abc import ABC, abstractmethod
from typing import Any

from composer.graph.state import WorkflowState
from composer.monitoring.logging import composer_logger
from composer.core.errors import NodeExecutionError


class BaseNode(ABC):
    """
    Base class for all workflow nodes in the composer system.

    Provides standardized initialization, validation, configuration access,
    and error handling patterns that are common across all nodes.
    """

    def __init__(self, node_name: str, **kwargs):
        """
        Initialize base node with common setup.

        Args:
            node_name: Name of the node for logging and error reporting
            **kwargs: Additional initialization parameters for subclasses
        """
        self.node_name = node_name
        self.logger = composer_logger.logger.bind(component=node_name)

        # Allow subclasses to handle additional initialization
        self._initialize_node(**kwargs)

    @abstractmethod
    def _initialize_node(self, pipeline_factory=None, **kwargs) -> None:
        """
        Hook for subclass-specific initialization.

        Override this method in subclasses to handle additional initialization
        parameters passed to the constructor.
        """
        # Base implementation does nothing - subclasses override as needed
        return

    @abstractmethod
    async def __call__(self, state: WorkflowState) -> WorkflowState:
        """Execute the node operation."""
        raise NotImplementedError

    def _validate_user_id(self, state: WorkflowState) -> str:
        """
        Validate and extract user ID from workflow state.

        Args:
            state: Workflow state to validate

        Returns:
            Validated user ID

        Raises:
            NodeExecutionError: If user ID is missing or invalid
        """
        user_id = getattr(state, "user_id", None)
        if not user_id:
            raise NodeExecutionError(f"User ID required for {self.node_name}")
        return user_id

    def _ensure_user_config_initialized(self, state: WorkflowState) -> WorkflowState:
        """
        Ensure user configuration is properly initialized in state.

        Args:
            state: Workflow state to validate

        Returns:
            State with validated user configuration (may be default if not set)

        Note:
            This method logs warnings if user_config is missing but doesn't fail.
            Individual nodes can decide if missing config should be an error.
        """
        if not state.user_config:
            user_id = getattr(state, "user_id", "unknown")
            self.logger.warning(
                f"No user configuration found in workflow state for {self.node_name}",
                user_id=user_id,
            )
        return state

    def _handle_error(
        self,
        state: WorkflowState,
        error: Exception,
        operation: str,
        fail_workflow: bool = False,
    ) -> WorkflowState:
        """
        Handle errors consistently across all nodes.

        Args:
            state: Current workflow state
            error: Exception that occurred
            operation: Description of the operation that failed
            fail_workflow: Whether to raise the error (True) or continue (False)

        Returns:
            Updated workflow state with error details

        Raises:
            NodeExecutionError: If fail_workflow is True
        """
        user_id = getattr(state, "user_id", "unknown")

        error_message = f"{operation} failed in {self.node_name}: {str(error)}"

        self.logger.error(
            error_message,
            user_id=user_id,
            node_name=self.node_name,
            operation=operation,
            error_type=type(error).__name__,
        )

        # Add error to state for circuit breaker and recovery
        state.error_details.append(error_message)

        if fail_workflow:
            raise NodeExecutionError(f"{self.node_name}: {operation} failed", error)

        return state

    def _validate_state_requirements(
        self,
        state: WorkflowState,
        require_messages: bool = False,
        require_user_config: bool = False,
        require_intent_classification: bool = False,
    ) -> None:
        """
        Validate common state requirements.

        Args:
            state: Workflow state to validate
            require_messages: Whether messages are required
            require_user_config: Whether user configuration is required
            require_intent_classification: Whether intent classification is required

        Raises:
            NodeExecutionError: If required state is missing
        """
        if require_messages and not state.messages:
            raise NodeExecutionError(f"Messages required for {self.node_name}")

        if require_user_config and not state.user_config:
            raise NodeExecutionError(
                f"User configuration required for {self.node_name}"
            )

        if require_intent_classification and not state.intent_classification:
            raise NodeExecutionError(
                f"Intent classification required for {self.node_name}"
            )

    def _log_node_execution_start(
        self, state: WorkflowState, **additional_context
    ) -> None:
        """
        Log the start of node execution with standard context.

        Args:
            state: Current workflow state
            **additional_context: Additional logging context
        """
        context = {
            "user_id": getattr(state, "user_id", "unknown"),
            "node_name": self.node_name,
            "has_messages": bool(state.messages),
            "has_user_config": bool(state.user_config),
            "has_intent_classification": bool(state.intent_classification),
            **additional_context,
        }

        self.logger.info(f"Starting {self.node_name} execution", **context)

    def _log_node_execution_complete(
        self, state: WorkflowState, **additional_context
    ) -> None:
        """
        Log the completion of node execution with standard context.

        Args:
            state: Current workflow state
            **additional_context: Additional logging context
        """
        context = {
            "user_id": getattr(state, "user_id", "unknown"),
            "node_name": self.node_name,
            **additional_context,
        }

        self.logger.info(f"Completed {self.node_name} execution", **context)
