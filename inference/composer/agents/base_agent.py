"""
Base Agent class providing common functionality for all workflow agents.
Provides node metadata injection, logging setup, and common error handling patterns.
"""

from typing import (
    Optional,
    Any,
    Dict,
    TypeVar,
    Generic,
    AsyncIterator,
    List,
    cast,
    Callable,
    Awaitable,
)
from abc import ABC, abstractmethod

from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate

from langchain.agents import create_agent, AgentExecutor
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables.config import RunnableConfig

from models import (
    MessageRole,
    NodeMetadata,
    ModelProfile,
    ChatResponse,
    LangChainMessage,
    PipelinePriority,
)
from runner import PipelineFactory, GrammarInput, MessageInput
from runner.chat_model_factory import chat_model_factory
from runner.embedding_model_factory import embedding_model_factory
from utils.logging import llmmllogger
from utils.response import create_streaming_chunk, create_error_response
from composer.core.errors import NodeExecutionError
from composer.utils.conversion import (
    normalize_message_input,
    convert_messages_to_langchain,
)

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

    async def stream(
        self,
        messages: MessageInput,
        user_id: str,
        tools: Optional[List[BaseTool]] = None,
        circuit_breaker: Optional[Any] = None,
        priority: PipelinePriority = PipelinePriority.MEDIUM,
        grammar: Optional[GrammarInput] = None,
    ) -> AsyncIterator[ChatResponse]:
        """
        Stream pipeline execution with node metadata injection.

        This method abstracts the common pattern of streaming pipeline execution
        and automatically injects node metadata into streaming chunks to provide
        context about what type of content is being generated.

        Args:
            messages: Input messages for the pipeline
            user_id: User identifier
            tools: Optional tools for the pipeline
            circuit_breaker: Optional circuit breaker configuration
            priority: Pipeline execution priority

        Yields:
            ChatResponse: Streaming chunks with injected node metadata
        """
        # Lazy imports to avoid circular dependency
        from runner import stream_pipeline  # pylint: disable=import-outside-toplevel

        try:
            self._log_operation_start(
                "stream_pipeline",
                user_id=user_id,
                message_count=len(messages),
                has_tools=bool(tools),
                node_name=self._node_metadata.node_name,
                node_type=self._node_metadata.node_type,
            )

            yield create_streaming_chunk(
                text="",
                role=MessageRole.OBSERVER,
                done=False,
            ).model_copy(update={"channels": self._node_metadata.model_dump()})

            # Execute pipeline with streaming
            with self.pipeline_factory.pipeline(
                self.profile, ChatResponse, priority, circuit_breaker
            ) as pipeline:
                chunk_count = 0

                async for chunk in stream_pipeline(
                    messages,
                    pipeline,
                    cast(List[BaseTool], tools) if tools else None,
                    grammar=grammar,
                ):
                    chunk.channels = self._node_metadata.model_dump()
                    yield chunk

                # Yield end chunk with node metadata

                yield create_streaming_chunk(
                    text="",
                    role=MessageRole.ASSISTANT,
                    done=True,
                ).model_copy(update={"channels": self._node_metadata.model_dump()})

                self._log_operation_success(
                    "stream_pipeline",
                    user_id=user_id,
                    chunk_count=chunk_count,
                    node_name=self._node_metadata.node_name,
                )

        except Exception as e:
            yield create_error_response(str(e))

            self._handle_node_error(
                "stream_pipeline",
                e,
                user_id=user_id,
                message_count=len(messages),
            )

    async def run(
        self,
        messages: MessageInput,
        user_id: str,
        tools: Optional[List[BaseTool]] = None,
        circuit_breaker: Optional[Any] = None,
        priority: PipelinePriority = PipelinePriority.MEDIUM,
        grammar: Optional[GrammarInput] = None,
    ) -> ChatResponse:
        """
        Run pipeline execution with node metadata injection.

        This method abstracts the common pattern of pipeline execution
        and automatically injects node metadata into the response.

        Args:
            messages: Input messages for the pipeline
            user_id: User identifier
            tools: Optional tools for the pipeline
            circuit_breaker: Optional circuit breaker configuration
            priority: Pipeline execution priority

        Returns:
            ChatResponse: Response with injected node metadata
        """
        # Lazy imports to avoid circular dependency
        from runner import run_pipeline  # pylint: disable=import-outside-toplevel

        normalized_messages = convert_messages_to_langchain(
            normalize_message_input(messages)
        )

        try:
            self._log_operation_start(
                "run_pipeline",
                user_id=user_id,
                message_count=len(messages),
                has_tools=bool(tools),
                node_name=self._node_metadata.node_name,
                node_type=self._node_metadata.node_type,
            )

            # Execute pipeline
            with self.pipeline_factory.pipeline(
                self.profile, ChatResponse, priority, circuit_breaker
            ) as pipeline:
                response = await run_pipeline(
                    normalized_messages,
                    pipeline,
                    cast(List[BaseTool], tools) if tools else None,
                    grammar=grammar,
                )

                response.channels = self._node_metadata.model_dump()

                return response

        except Exception as e:
            self._handle_node_error(
                "run_pipeline",
                e,
                user_id=user_id,
                message_count=len(messages),
            )
            return create_error_response(str(e))

    async def embed(
        self,
        messages: MessageInput,
        user_id: str,
        circuit_breaker: Optional[Any] = None,
        priority: PipelinePriority = PipelinePriority.MEDIUM,
    ) -> List[List[float]]:
        """
        Run pipeline execution with node metadata injection.

        This method abstracts the common pattern of pipeline execution
        and automatically injects node metadata into the response.

        Args:
            messages: Input messages for the pipeline
            user_id: User identifier
            tools: Optional tools for the pipeline
            circuit_breaker: Optional circuit breaker configuration
            priority: Pipeline execution priority

        Returns:
            ChatResponse: Response with injected node metadata
        """
        # Lazy imports to avoid circular dependency
        from runner import embed_pipeline  # pylint: disable=import-outside-toplevel

        try:
            self._log_operation_start(
                "run_pipeline",
                user_id=user_id,
                message_count=len(messages),
                node_name=self._node_metadata.node_name,
                node_type=self._node_metadata.node_type,
            )

            # Execute pipeline
            with self.pipeline_factory.pipeline(
                self.profile, ChatResponse, priority, circuit_breaker
            ) as pipeline:
                response = await embed_pipeline(messages, pipeline)

                response.channels = self._node_metadata.model_dump()

                return response

        except Exception as e:
            self._handle_node_error(
                "embed_pipeline",
                e,
                user_id=user_id,
                message_count=len(messages),
            )
            return create_error_response(str(e))


#     async def run_generic_pipeline_with_metadata(
#         self,
#         pipeline_executor: Callable[..., Awaitable[Any]],
#         operation_name: str,
#         **kwargs,
#     ) -> Any:
#         """
#         Run any pipeline execution function with metadata tracking and logging.

#         This method provides a consistent interface for agents that don't return ChatResponse
#         but still want the benefits of metadata tracking, logging, and error handling.

#         Args:
#             pipeline_executor: Async function that executes the pipeline
#             operation_name: Name of the operation for logging
#             **kwargs: Arguments to pass to the pipeline executor

#         Returns:
#             The result from pipeline_executor, potentially wrapped with metadata
#         """
#         try:
#             self._log_operation_start(
#                 operation_name,
#                 node_name=self._node_metadata.node_name,
#                 node_type=self._node_metadata.node_type,
#                 **kwargs,
#             )

#             # Execute the pipeline
#             result = await pipeline_executor(**kwargs)

#             self._log_operation_success(
#                 operation_name,
#                 node_name=self._node_metadata.node_name,
#                 has_result=bool(result),
#             )

#             return result

#         except Exception as e:
#             self._handle_node_error(operation_name, e, **kwargs)
#             raise

#     def run_pipeline_with_context_manager(
#         self,
#         return_type: type,
#         priority: PipelinePriority = PipelinePriority.MEDIUM,
#         circuit_breaker: Optional[Any] = None,
#     ):
#         """
#         Get a pipeline context manager with consistent metadata tracking.

#         This provides the same context manager pattern used by agents but with
#         enhanced logging that includes node metadata.

#         Args:
#             return_type: Expected return type for the pipeline
#             priority: Pipeline execution priority
#             circuit_breaker: Optional circuit breaker configuration

#         Returns:
#             Pipeline context manager with enhanced logging
#         """
#         return PipelineContextWithMetadata(
#             self.pipeline_factory,
#             self.profile,
#             return_type,
#             priority,
#             circuit_breaker,
#             self._node_metadata,
#             self.logger,
#         )


# class PipelineContextWithMetadata:
#     """
#     Context manager wrapper that adds metadata tracking to pipeline operations.

#     This provides the same interface as the regular pipeline factory context manager
#     but adds enhanced logging with node metadata.
#     """

#     def __init__(
#         self,
#         pipeline_factory: PipelineFactory,
#         profile: ModelProfile,
#         return_type: type,
#         priority: PipelinePriority,
#         circuit_breaker: Optional[Any],
#         node_metadata: NodeMetadata,
#         logger: Any,
#     ):
#         self.pipeline_factory = pipeline_factory
#         self.profile = profile
#         self.return_type = return_type
#         self.priority = priority
#         self.circuit_breaker = circuit_breaker
#         self.node_metadata = node_metadata
#         self.logger = logger
#         self._pipeline_context = None

#     def __enter__(self):
#         """Enter the context manager."""
#         self.logger.info(
#             "Starting pipeline context",
#             node_name=self.node_metadata.node_name,
#             node_type=self.node_metadata.node_type,
#             return_type=self.return_type.__name__ if self.return_type else "unknown",
#             priority=(
#                 self.priority.value
#                 if hasattr(self.priority, "value")
#                 else str(self.priority)
#             ),
#         )

#         self._pipeline_context = self.pipeline_factory.pipeline(
#             self.profile, self.return_type, self.priority, self.circuit_breaker
#         )
#         return self._pipeline_context.__enter__()

#     def __exit__(self, exc_type, exc_val, exc_tb):
#         """Exit the context manager."""
#         if exc_type:
#             self.logger.error(
#                 "Pipeline context failed",
#                 node_name=self.node_metadata.node_name,
#                 error=str(exc_val),
#                 error_type=exc_type.__name__,
#             )
#         else:
#             self.logger.info(
#                 "Pipeline context completed",
#                 node_name=self.node_metadata.node_name,
#                 node_type=self.node_metadata.node_type,
#             )

#         if self._pipeline_context:
#             return self._pipeline_context.__exit__(exc_type, exc_val, exc_tb)
