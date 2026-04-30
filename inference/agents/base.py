"""
Base Agent class providing common functionality for all workflow agents.
Provides node metadata injection, logging setup, and common error handling patterns.
"""

import datetime
import logging
from typing import (
    Optional,
    Self,
    List,
)
from pydantic import BaseModel
from langchain.agents.structured_output import ProviderStrategy
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import BaseMessage, AIMessage

from models import (
    MessageContent,
    MessageContentType,
    MessageRole,
    NodeMetadata,
    ChatResponse,
    Message,
)
from utils.logging import llmmllogger, serialize_event_data
from utils.message_conversion import (
    normalize_message_input,
    messages_to_lc_messages,
    lc_message_to_message,
    MessageInput,
    extract_text_from_message,
)
from utils.grammar_generator import parse_structured_output

# Get the 'asyncio' logger
asyncio_logger = logging.getLogger("asyncio")
# Set the logging level to WARNING or higher (e.g., ERROR, CRITICAL)
# This will prevent INFO and DEBUG messages from being displayed when run_sync is used.
asyncio_logger.setLevel(logging.WARNING)


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


class BaseAgent:
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
        model: BaseChatModel,
        system_prompt: str = "",
        num_ctx: int = 100000,
        component_name: Optional[str] = None,
        middleware: Optional[List[AgentMiddleware]] = None,
        tools: Optional[List[BaseTool]] = None,
    ):
        """
        Initialize base agent with required dependencies.

        Args:
            model: Base chat model for agent operations
            system_prompt: System prompt for the agent
            num_ctx: Context window size for summarization threshold
            component_name: Optional component name for logging. If not provided,
                          uses the class name.
        """
        # Set up component-specific logging
        component = component_name or self.__class__.__name__
        self.logger = llmmllogger.bind(component=component)

        # Store required dependencies
        self.model = model
        self.system_prompt = system_prompt
        self.num_ctx = num_ctx

        self.agent_id = f"{id(self):x}"
        # Middleware list passed to create_agent for behaviors like TodoListMiddleware
        self.middleware: List[AgentMiddleware] = middleware or []
        self.tools: List[BaseTool] = tools or []

        self.logger.debug(f"Initialized {component}")

        self._node_metadata = NodeMetadata(
            node_name="UNSET",
            node_id="UNSET",
            node_type=self.__class__.__name__,
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

    async def _get_or_create_agent(
        self,
        system_prompt,
        tools: Optional[List[BaseTool]] = None,
        grammar: Optional[type[BaseModel]] = None,
        middleware: Optional[List[AgentMiddleware]] = None,
        metadata: Optional[NodeMetadata] = None,
    ):
        """
        Get the persistent agent or create it if it doesn't exist.

        For performance and server reuse, we cache the pipeline but not the agent,
        since agent configuration (system prompt, tools, grammar) varies by call.
        The pipeline (LLM server) should be reused across different agent configurations.

        Args:
            system_prompt: System prompt for the agent
            tools: List of tools to bind to the agent
            priority: Pipeline priority
            grammar: Grammar constraints for structured output

        Returns:
            The LangChain agent or ChatOpenAI model (depending on pipeline type)
        """
        # Always create new agent for different configurations, but reuse pipeline
        # This allows different system prompts, tools, and grammars while maintaining server reuse

        self.logger.debug("Creating LangChain agent (pipeline will be reused)")
        agent = create_agent(
            model=self.model,
            tools=tools or [],
            system_prompt=system_prompt,
            response_format=ProviderStrategy(grammar) if grammar else None,
            name=(
                metadata.node_name
                if metadata is not None
                else self._node_metadata.node_name
            ),
            middleware=middleware or [],
        )

        return agent

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

        system_prompt = self.system_prompt

        for msg in msgs:
            if msg.role == MessageRole.SYSTEM:
                system_prompt += f"\n\n{extract_text_from_message(msg)}"
            else:
                convo.append(msg)

        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        system_prompt += f"""
The current date is {current_date}."""

        return system_prompt, convo

    async def run(
        self,
        messages: MessageInput,
        tools: Optional[List[BaseTool]] = None,
        grammar: Optional[type[BaseModel]] = None,
        middleware: Optional[List[AgentMiddleware]] = None,
        metadata: Optional[NodeMetadata] = None,
    ) -> ChatResponse:
        """
        Run agent execution with node metadata injection.

        Creates a LangChain agent using create_agent() with BaseChatModel from factory,
        then executes the agent and returns the result with node metadata.

        Args:
            messages: Input messages for the agent
            tools: Optional tools for the agent
            grammar: Optional grammar constraints for structured output
            middleware: Optional middleware for the agent
            metadata: Optional node metadata for workflow tracking

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
            # Deduplicate tools by name since StructuredTool objects are not hashable
            combined_tools = (self.tools or []) + (tools or [])
            seen_tool_names = set()
            unique_tools = []
            for tool in combined_tools:
                if tool.name not in seen_tool_names:
                    unique_tools.append(tool)
                    seen_tool_names.add(tool.name)

            # Deduplicate middleware by class since middleware objects might not be hashable
            combined_middleware = (self.middleware or []) + (middleware or [])
            seen_middleware_types = set()
            unique_middleware = []
            for mw in combined_middleware:
                mw_type = type(mw).__name__
                if mw_type not in seen_middleware_types:
                    unique_middleware.append(mw)
                    seen_middleware_types.add(mw_type)

            agent = await self._get_or_create_agent(
                system_prompt,
                unique_tools,
                grammar,
                unique_middleware,
                metadata,
            )

            if agent is None:
                self.logger.error("🚨 Agent is None after _get_or_create_agent call!")
                raise ValueError("Agent creation failed - agent is None")

            # Convert messages to LangChain format
            normalized_messages = messages_to_lc_messages(convo)
            self.logger.debug(f"Running agent with {len(normalized_messages)} messages")
            result = await agent.ainvoke({"messages": normalized_messages})  # type: ignore

            if isinstance(result, dict):
                if "structured_response" in result and grammar:
                    result = result["structured_response"]
                    if not isinstance(result, BaseMessage):
                        result = AIMessage(content=grammar.model_dump_json(result))
                elif "messages" in result:
                    msgs = result["messages"]
                    if isinstance(msgs, list) and len(msgs) > 0:
                        # get the last message and assert that it's ai
                        result = msgs[-1]
                        if hasattr(result, "role") and result.role != "ai":
                            self.logger.warning(
                                "🚨 Last message role is not 'ai' - unexpected result format"
                            )

            assert isinstance(result, BaseMessage), "Agent result is not a BaseMessage"
            msg = lc_message_to_message(result)
            response = ChatResponse(
                done=True,
                message=msg,
                metadata=self._node_metadata,
            )

            return response

        except Exception as e:
            self._handle_node_error(
                "create_agent_run",
                e,
                message_count=get_message_count(messages),
            )
            # Re-raise timeout errors so the workflow executor catches them
            # and yields an error response with content.  Without this, the
            # timeout is swallowed, the executor sees no LLM events, and
            # produces an empty response — which triggers the completion
            # service to retry (creating a cascade of wasted requests).
            from openai import APITimeoutError

            if isinstance(e, (APITimeoutError, TimeoutError)):
                raise
            return ChatResponse(
                done=True,
                message=Message(
                    role=MessageRole.ASSISTANT,
                    content=[
                        MessageContent(
                            type=MessageContentType.TEXT,
                            text=f"Error during agent execution: {str(e)}",
                        )
                    ],
                ),
                metadata=self._node_metadata,
            )

    async def run_structured[T: BaseModel](
        self,
        message_input: MessageInput,
        grammar: type[T],
        tools: Optional[List[BaseTool]] = None,
        middleware: Optional[List[AgentMiddleware]] = None,
        metadata: Optional[NodeMetadata] = None,
    ) -> T:
        """
        Run agent execution with node metadata injection.

        Creates a LangChain agent using create_agent() with BaseChatModel from factory,
        then executes the agent and returns the result with node metadata.

        Args:
            messages: Input messages for the agent
            user_id: User identifier
            tools: Optional tools for the agent
            priority: Pipeline execution priority (affects model selection)

        Returns:
            ChatResponse: Response with injected node metadata
        """

        try:
            self._log_operation_start(
                "structured_agent_run",
                message_count=get_message_count(message_input),
                node_name=self._node_metadata.node_name,
                node_type=self._node_metadata.node_type,
            )
            system_prompt, convo = self._separate_system_prompt(message_input)

            # Use persistent agent - creates once and reuses for state continuity
            agent = await self._get_or_create_agent(
                system_prompt,
                list((self.tools or []) + (tools or [])),
                grammar,
                list((self.middleware or []) + (middleware or [])),
                metadata,
            )

            if agent is None:
                self.logger.error("🚨 Agent is None after _get_or_create_agent call!")
                raise ValueError("Agent creation failed - agent is None")

            # Convert messages to LangChain format
            normalized_messages = messages_to_lc_messages(convo)
            self.logger.debug(f"Running agent with {len(normalized_messages)} messages")
            result = await agent.ainvoke({"messages": normalized_messages})  # type: ignore
            self.logger.debug(
                f"Agent run result ({type(result)}): {serialize_event_data(result)}"
            )

            if isinstance(result, dict):
                if "structured_response" in result and grammar:
                    result = result["structured_response"]
                    if isinstance(result, grammar):
                        self.logger.debug("Structured response matches grammar type")
                        return result
                elif "messages" in result:
                    msgs = result["messages"]
                    if isinstance(msgs, list) and len(msgs) > 0:
                        # get the last message and assert that it's ai
                        result = msgs[-1]
                        if hasattr(result, "role") and result.role != "ai":
                            self.logger.warning(
                                "🚨 Last message role is not 'ai' - unexpected result format"
                            )

            assert isinstance(result, BaseMessage), "Agent result is not a BaseMessage"

            msg = lc_message_to_message(result)
            return parse_structured_output(extract_text_from_message(msg), grammar)

        except Exception as e:
            self._handle_node_error(
                "create_agent_run",
                e,
                message_count=get_message_count(message_input),
            )
            raise RuntimeError(f"Structured agent execution failed: {e}") from e
