"""
Base Agent class providing common functionality for all workflow agents.
Provides node metadata injection, logging setup, and common error handling patterns.
"""

import datetime
from typing import (
    Optional,
    Any,
    Dict,
    TypeVar,
    Generic,
    AsyncIterator,
    List,
    cast,
)
from abc import ABC
from numpy import isin
from pydantic import BaseModel
from langchain.agents.structured_output import ProviderStrategy
from langchain.agents import create_agent, AgentState
from langchain.chat_models import BaseChatModel
from langchain.embeddings.base import Embeddings
from langchain_core.tools import BaseTool
from langchain_core.messages import BaseMessage
from langgraph.graph.state import CompiledStateGraph

from models import (
    MessageRole,
    NodeMetadata,
    ModelProfile,
    ChatResponse,
    PipelinePriority,
    Message,
)
from runner import PipelineFactory
from utils.logging import llmmllogger
from utils.response import create_streaming_chunk, create_error_response
from utils.message_conversion import (
    normalize_message_input,
    messages_to_lc_messages,
    lc_message_to_message,
    MessageInput,
    extract_text_from_message,
)
from composer.core.errors import NodeExecutionError


T = TypeVar("T")


def get_message_count(messages: MessageInput) -> int:
    """Helper function to safely get message count from MessageInput."""
    if isinstance(messages, str):
        return 1
    elif isinstance(messages, Message):
        return 1
    elif isinstance(messages, list):
        return len(messages)
    else:
        # Fallback for unknown types
        return 1


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

        # Persistent LangChain agent - initialized once and reused for all operations
        self._agent: Optional[CompiledStateGraph] = None
        
        # Track if we have locked a pipeline that needs cleanup
        self._pipeline_locked = False

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

    def cleanup(self) -> None:
        """
        Clean up resources used by this agent, including unlocking any pipeline.
        
        This method should be called when the agent is no longer needed to ensure
        that locked pipelines are properly released for other components to use.
        """
        if self._pipeline_locked:
            try:
                self.logger.info(f"🔓 Cleaning up agent pipeline for model {self.profile.model_name}")
                success = self.pipeline_factory.unlock_pipeline(self.profile)
                if success:
                    self.logger.info(f"✅ Successfully unlocked pipeline for {self.profile.model_name}")
                    self._pipeline_locked = False
                else:
                    self.logger.warning(f"⚠️ Failed to unlock pipeline for {self.profile.model_name}")
            except Exception as e:
                self.logger.error(f"❌ Error during pipeline cleanup for {self.profile.model_name}: {e}")
        
        # Reset agent state
        self._agent = None
        self.logger.debug("Agent cleanup completed")

    def _get_or_create_agent(
        self,
        system_prompt,
        tools: Optional[List[BaseTool]] = None,
        priority: PipelinePriority = PipelinePriority.MEDIUM,
        grammar: Optional[type[BaseModel]] = None,
    ) -> CompiledStateGraph[AgentState, Any, Any]:
        """
        Get the persistent agent or create it if it doesn't exist.

        Args:
            llm: The language model to use
            tools: List of tools to bind to the agent
            system_prompt: System prompt for the agent
            grammar: Grammar constraints for structured output

        Returns:
            The persistent LangChain agent
        """
        if self._agent is None:
            self.logger.debug("Creating persistent LangChain agent")
            # Get the model configuration from pipeline factory

            if tools:
                system_prompt += (
                    "\n\nYou have access to the following tools:\n"
                    + "\n".join(
                        [f"- {tool.name}: {tool.description}" for tool in tools]
                    )
                    + "\n\nUse them wisely to assist the user.\n\n"
                    + """TOOL CALLING FORMAT:
When you need to call a tool, you MUST use this EXACT JSON format wrapped in <tool_call> tags:
<tool_call>{"name": "tool_name", "arguments": "{\"param\": \"value\"}"}</tool_call>
NEVER fabricate or hallucinate tool results. ALWAYS call the actual tool when you need information.
The arguments field MUST be a JSON string (double-quoted), not a JSON object.
"""
                )

            current_date = datetime.datetime.now().strftime("%Y-%m-%d")

            system_prompt += f"""
TEMPORAL CONTEXT:
The current date is {current_date}. While this is likely past your training data, you can use this information to provide better responses. If the user asks for the date or time, respond with this date.
"""
            if "web_search" in (tool.name for tool in (tools or [])):
                system_prompt += "If the user asks for current events or recent information, use the web_search tool to find up-to-date information."

            chat_model = self.pipeline_factory.get_pipeline(
                self.profile, priority, grammar
            )
            
            # Mark that we have locked a pipeline that needs cleanup
            self._pipeline_locked = True
            self.logger.debug(f"🔒 Locked pipeline for {self.profile.model_name}")

            llm = cast(BaseChatModel, chat_model)
            self._agent = create_agent(
                model=llm,
                tools=tools or [],
                system_prompt=system_prompt,
                response_format=ProviderStrategy(grammar) if grammar else None,
                name=self._node_metadata.node_name,
            )

        return self._agent

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

    def _separate_system_prompt(
        self, messages: MessageInput
    ) -> tuple[str, List[Message]]:
        """
        Extract system prompt from messages if present.

        Args:
            messages: Input messages for the agent

        returns:
            str: Extracted system prompt
        """
        msgs = normalize_message_input(messages)
        convo = []

        system_prompt = self.profile.system_prompt

        for msg in msgs:
            if msg.role == MessageRole.SYSTEM:
                system_prompt += f"\n\n{extract_text_from_message(msg)}"
            else:
                convo.append(msg)

        return system_prompt, convo

    async def stream(
        self,
        messages: MessageInput,
        tools: Optional[List[BaseTool]] = None,
        priority: PipelinePriority = PipelinePriority.MEDIUM,
        grammar: Optional[type[BaseModel]] = None,
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
                message_count=get_message_count(messages),
                has_tools=bool(tools),
                node_name=self._node_metadata.node_name,
                node_type=self._node_metadata.node_type,
            )

            yield create_streaming_chunk(
                text="",
                role=MessageRole.OBSERVER,
                done=False,
            ).model_copy(update={"channels": self._node_metadata.model_dump()})

            system_prompt, convo = self._separate_system_prompt(messages)

            # Use persistent agent - creates once and reuses for state continuity
            agent = self._get_or_create_agent(system_prompt, tools, priority, grammar)

            # Convert messages to LangChain format
            normalized_messages = messages_to_lc_messages(convo)
            npt = {"messages": normalized_messages}

            # Stream agent execution with recursion limit
            chunk_count = 0
            async for chunk in agent.astream(
                npt, stream_mode="messages", subgraphs=True  # type: ignore
            ):
                msg_chunk = {}
                metadata = {}

                self.logger.debug(f"Processing streaming chunk: {chunk}")

                # stream_mode "messages" returns AIMessageChunk objects with metadata
                if isinstance(chunk, tuple) and len(chunk) >= 2:
                    msg_chunk, metadata = chunk
                elif isinstance(chunk, BaseMessage):
                    msg_chunk = chunk

                if isinstance(msg_chunk, BaseMessage):
                    msg = lc_message_to_message(msg_chunk)

                    chat_chunk = ChatResponse(done=False, message=msg)
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
                chunk_count=chunk_count,
                node_name=self._node_metadata.node_name,
            )

        except Exception as e:
            yield create_error_response(str(e))

            self._handle_node_error(
                "create_agent_stream",
                e,
                message_count=get_message_count(messages),
            )

    async def run(
        self,
        messages: MessageInput,
        tools: Optional[List[BaseTool]] = None,
        priority: PipelinePriority = PipelinePriority.MEDIUM,
        grammar: Optional[type[BaseModel]] = None,
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

        try:
            self._log_operation_start(
                "create_agent_run",
                message_count=get_message_count(messages),
                has_tools=bool(tools),
                node_name=self._node_metadata.node_name,
                node_type=self._node_metadata.node_type,
            )
            system_prompt, convo = self._separate_system_prompt(messages)

            # Use persistent agent - creates once and reuses for state continuity
            agent = self._get_or_create_agent(system_prompt, tools, priority, grammar)

            # Convert messages to LangChain format
            normalized_messages = messages_to_lc_messages(convo)
            # Execute agent with normalized messages
            result = await agent.ainvoke(
                {"messages": normalized_messages},  # type: ignore
                grammar=grammar,
                tools=tools,
            )

            # Convert agent result to ChatResponse
            last_message = result["messages"][-1]
            assert isinstance(last_message, BaseMessage)
            msg = lc_message_to_message(last_message)
            response = ChatResponse(
                done=True,
                message=msg,
            )
            response.channels = self._node_metadata.model_dump()
            return response

        except Exception as e:
            self._handle_node_error(
                "create_agent_run",
                e,
                message_count=get_message_count(messages),
            )
            return create_error_response(str(e))

    async def embed(
        self,
        messages: MessageInput,
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
                message_count=get_message_count(messages),
                node_name=self._node_metadata.node_name,
                node_type=self._node_metadata.node_type,
            )

            # Get the model configuration from pipeline factory
            with self.pipeline_factory.pipeline(
                self.profile,
                priority,
            ) as embedding_model:
                if not embedding_model:
                    raise NodeExecutionError("Failed to create embedding model")
                llm = cast(Embeddings, embedding_model)
                # Convert messages to text list
                normalized_messages = normalize_message_input(messages)
                text_list = []

                for message in normalized_messages:
                    if message.content:
                        text_list.append(extract_text_from_message(message))

                if not text_list:
                    return []

                # Generate embeddings
                embeddings = await llm.aembed_documents(text_list)

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
                message_count=get_message_count(messages),
            )
            # Return empty embeddings on error
            return []
