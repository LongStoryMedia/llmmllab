"""
Base Agent class providing common functionality for all workflow agents.
Provides node metadata injection, logging setup, and common error handling patterns.
"""

from typing import (
    Optional,
    Any,
    Dict,
    Self,
    List,
)

import numpy as np

from langchain.agents.middleware import AgentMiddleware
from langchain.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings

from models import (
    NodeMetadata,
)

from utils.logging import llmmllogger
from utils.message_conversion import (
    normalize_message_input,
    MessageInput,
    extract_text_from_message,
)


class EmbeddingAgent:
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
        model: Embeddings,
        component_name: Optional[str] = None,
        middleware: Optional[List[AgentMiddleware]] = None,
    ):
        """
        Initialize base agent with required dependencies.

        Args:
            model: Embeddings model for agent operations
            component_name: Optional component name for logging. If not provided,
                          uses the class name.
        """
        # Set up component-specific logging
        component = component_name or self.__class__.__name__
        self.logger = llmmllogger.bind(component=component)

        # Store required dependencies
        self.model = model

        # Additional metadata for debugging and tracking
        self._execution_context: Dict[str, Any] = {}

        # Persistent pipeline reference - prevents garbage collection
        self._pipeline: Optional[BaseChatModel | Embeddings] = None

        # Track if we have locked a pipeline that needs cleanup
        self._pipeline_locked = False

        self.agent_id = f"{id(self):x}"
        # Middleware list passed to create_agent for behaviors like TodoListMiddleware
        self.middleware: List[AgentMiddleware] = middleware or []

        self.logger.debug(f"Initialized {component}")

        self._node_metadata = NodeMetadata(
            node_name="UNSET",
            node_id="UNSET",
            node_type="UNSET",
        )

    def bind_node_metadata(self, metadata: NodeMetadata) -> Self:
        """
        Bind new node metadata to the agent for workflow tracking.

        Args:
            metadata: New node metadata to bind
        """
        self._node_metadata = metadata
        self.logger = self.logger.bind(
            node_name=metadata.node_name,
            node_id=metadata.node_id,
            node_type=metadata.node_type,
            user_id=metadata.user_id,
        )
        self.logger.debug(
            "Bound new node metadata to agent",
            node_name=metadata.node_name,
            node_type=metadata.node_type,
        )
        return self

    @property
    def current_pipeline(self) -> Optional[BaseChatModel | Embeddings]:
        """Get the current pipeline instance if available."""
        return self._pipeline

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

    async def compute_similarity(
        self, embedding1: List[float], embedding2: List[float]
    ) -> float:
        """
        Compute cosine similarity between two embedding vectors.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score between -1 and 1
        """
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)

            # Compute cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = dot_product / (norm1 * norm2)
            return float(similarity)

        except Exception as e:
            self.logger.error(f"Similarity computation failed: {e}")
            return 0.0

    async def embed(self, messages: MessageInput) -> List[List[float]]:
        """
        Run embedding execution using embedding model factory.

        Creates embeddings using the EmbeddingModelFactory to get the appropriate
        Embeddings implementation, then processes the input messages.

        Args:
            messages: Input messages for embedding
            user_id: User identifier
            priority: Pipeline execution priority (affects model selection)

        Returns:
            List[List[float]]: Embedding vectors for the input messages
        """
        try:
            self._log_operation_start(
                "embedding_factory_run",
                node_name=self._node_metadata.node_name,
                node_type=self._node_metadata.node_type,
            )

            # Convert messages to text list
            normalized_messages = normalize_message_input(messages)
            text_list = []

            for message in normalized_messages:
                if message.content:
                    text_list.append(extract_text_from_message(message))

            if not text_list:
                return []

            # Enforce size cap per text to prevent server crashes
            # If any single text exceeds limit, truncate it
            MAX_CHARS_PER_TEXT = 2000
            truncated_text_list = []
            for text in text_list:
                if len(text) > MAX_CHARS_PER_TEXT:
                    self.logger.warning(
                        f"Truncating embedding input from {len(text)} to {MAX_CHARS_PER_TEXT} chars",
                        node_name=self._node_metadata.node_name,
                    )
                    truncated_text_list.append(text[:MAX_CHARS_PER_TEXT])
                else:
                    truncated_text_list.append(text)

            # Generate embeddings for all texts at once (kept simple to avoid crashes)
            try:
                embeddings = await self.model.aembed_documents(truncated_text_list)
            except Exception as e:
                self._handle_node_error("embedding_factory_run", e)
                return []

            self._log_operation_success(
                "embedding_factory_run",
                embedding_count=len(embeddings),
                node_name=self._node_metadata.node_name,
            )

            return embeddings

        except Exception as e:
            self._handle_node_error(
                "embedding_factory_run",
                e,
            )
            # Return empty embeddings on error
            return []
