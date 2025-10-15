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

from langchain.agents import create_agent
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
        Stream agent execution with node metadata injection.

        Creates a LangChain agent using create_agent() with BaseChatModel from factory,
        then streams the agent execution results with node metadata injection.

        Args:
            messages: Input messages for the agent
            user_id: User identifier
            tools: Optional tools for the agent
            circuit_breaker: Optional circuit breaker configuration
            priority: Pipeline execution priority (affects model selection)

        Yields:
            ChatResponse: Streaming chunks with injected node metadata
        """
        try:
            self._log_operation_start(
                "create_agent_stream",
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

            # Get the model configuration from pipeline factory
            with self.pipeline_factory.pipeline(
                self.profile, ChatResponse, priority, circuit_breaker
            ) as pipeline_config:
                # Get BaseChatModel from chat model factory
                chat_model = chat_model_factory.create_chat_model(
                    pipeline_config.model, self.profile, circuit_breaker
                )

                if not chat_model:
                    raise NodeExecutionError(
                        f"Failed to create chat model for {pipeline_config.model.name}"
                    )

                # Create LangChain agent using create_agent()
                system_prompt = getattr(self.profile, "system_prompt", None)
                agent = create_agent(
                    model=chat_model, tools=tools or [], system_prompt=system_prompt
                )

                # Convert messages to LangChain format
                normalized_messages = convert_messages_to_langchain(
                    normalize_message_input(messages)
                )

                # Stream agent execution
                chunk_count = 0
                async for chunk in agent.astream(
                    {"messages": normalized_messages}, stream_mode="messages"
                ):
                    # Convert agent chunk to ChatResponse
                    if hasattr(chunk, "content") and chunk.content:
                        chat_chunk = create_streaming_chunk(
                            text=str(chunk.content),
                            role=MessageRole.ASSISTANT,
                            done=False,
                        )
                        chat_chunk.channels = self._node_metadata.model_dump()
                        chunk_count += 1
                        yield chat_chunk

                # Yield end chunk with node metadata
                yield create_streaming_chunk(
                    text="",
                    role=MessageRole.ASSISTANT,
                    done=True,
                ).model_copy(update={"channels": self._node_metadata.model_dump()})

                self._log_operation_success(
                    "create_agent_stream",
                    user_id=user_id,
                    chunk_count=chunk_count,
                    node_name=self._node_metadata.node_name,
                )

        except Exception as e:
            yield create_error_response(str(e))

            self._handle_node_error(
                "create_agent_stream",
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
        Run agent execution with node metadata injection.

        Creates a LangChain agent using create_agent() with BaseChatModel from factory,
        then executes the agent and returns the result with node metadata.

        Args:
            messages: Input messages for the agent
            user_id: User identifier
            tools: Optional tools for the agent
            circuit_breaker: Optional circuit breaker configuration
            priority: Pipeline execution priority (affects model selection)

        Returns:
            ChatResponse: Response with injected node metadata
        """
        normalized_messages = convert_messages_to_langchain(
            normalize_message_input(messages)
        )

        try:
            self._log_operation_start(
                "create_agent_run",
                user_id=user_id,
                message_count=len(messages),
                has_tools=bool(tools),
                node_name=self._node_metadata.node_name,
                node_type=self._node_metadata.node_type,
            )

            # Get the model configuration from pipeline factory
            with self.pipeline_factory.pipeline(
                self.profile, ChatResponse, priority, circuit_breaker
            ) as pipeline_config:
                # Get BaseChatModel from chat model factory
                chat_model = chat_model_factory.create_chat_model(
                    pipeline_config.model, self.profile, circuit_breaker
                )

                if not chat_model:
                    raise NodeExecutionError(
                        f"Failed to create chat model for {pipeline_config.model.name}"
                    )

                # Create LangChain agent using create_agent()
                system_prompt = getattr(self.profile, "system_prompt", None)
                agent = create_agent(
                    model=chat_model, tools=tools or [], system_prompt=system_prompt
                )

                # Execute agent
                result = await agent.ainvoke({"messages": normalized_messages})

                # Convert agent result to ChatResponse
                if "messages" in result and result["messages"]:
                    last_message = result["messages"][-1]
                    response = ChatResponse(
                        text=(
                            str(last_message.content)
                            if hasattr(last_message, "content")
                            else ""
                        ),
                        role=MessageRole.ASSISTANT,
                        done=True,
                    )
                else:
                    response = ChatResponse(
                        text="Agent completed without output",
                        role=MessageRole.ASSISTANT,
                        done=True,
                    )

                response.channels = self._node_metadata.model_dump()
                return response

        except Exception as e:
            self._handle_node_error(
                "create_agent_run",
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
        Run embedding execution using embedding model factory.

        Creates embeddings using the EmbeddingModelFactory to get the appropriate
        Embeddings implementation, then processes the input messages.

        Args:
            messages: Input messages for embedding
            user_id: User identifier
            circuit_breaker: Optional circuit breaker configuration
            priority: Pipeline execution priority (affects model selection)

        Returns:
            List[List[float]]: Embedding vectors for the input messages
        """
        try:
            self._log_operation_start(
                "embedding_factory_run",
                user_id=user_id,
                message_count=len(messages),
                node_name=self._node_metadata.node_name,
                node_type=self._node_metadata.node_type,
            )

            # Get the model configuration from pipeline factory
            with self.pipeline_factory.pipeline(
                self.profile,
                ChatResponse,
                priority,
            ) as pipeline_config:
                # Get Embeddings model from embedding model factory
                embedding_model = embedding_model_factory.create_embedding_model(
                    pipeline_config.model, self.profile
                )

                if not embedding_model:
                    raise NodeExecutionError(
                        f"Failed to create embedding model for {pipeline_config.model.name}"
                    )

                # Convert messages to text list
                normalized_messages = normalize_message_input(messages)
                text_list = []

                for message in normalized_messages:
                    if hasattr(message, "content"):
                        if isinstance(message.content, list):
                            # Handle multi-content messages
                            for content in message.content:
                                if hasattr(content, "text") and content.text:
                                    text_list.append(content.text)
                        else:
                            text_list.append(str(message.content))
                    elif isinstance(message, str):
                        text_list.append(message)

                if not text_list:
                    return []

                # Generate embeddings
                embeddings = await embedding_model.aembed_documents(text_list)

                self._log_operation_success(
                    "embedding_factory_run",
                    user_id=user_id,
                    embedding_count=len(embeddings),
                    node_name=self._node_metadata.node_name,
                )

                return embeddings

        except Exception as e:
            self._handle_node_error(
                "embedding_factory_run",
                e,
                user_id=user_id,
                message_count=len(messages),
            )
            # Return empty embeddings on error
            return []
