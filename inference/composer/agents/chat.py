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
from langchain_core.messages import BaseMessage

from models import (
    MessageContent,
    MessageContentType,
    MessageRole,
    ModelProfile,
    NodeMetadata,
    ChatResponse,
    Message,
)
from utils.grammar_generator import parse_structured_output
from utils.logging import llmmllogger, serialize_event_data
from utils.message_conversion import (
    normalize_message_input,
    messages_to_lc_messages,
    lc_message_to_message,
    MessageInput,
    extract_text_from_message,
)


# Get the 'asyncio' logger
asyncio_logger = logging.getLogger("asyncio")

# Set the logging level to WARNING or higher (e.g., ERROR, CRITICAL)
# This will prevent INFO and DEBUG messages from being displayed when run_sync is used.
asyncio_logger.setLevel(logging.WARNING)


class TitleResponse(BaseModel):
    title: str


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


class ChatAgent:
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
        profile: ModelProfile,
        component_name: Optional[str] = None,
        middleware: Optional[List[AgentMiddleware]] = None,
        tools: Optional[List[BaseTool]] = None,
    ):
        """
        Initialize base agent with required dependencies.

        Args:
            model: Base chat model for agent operations
            profile: Model profile for agent operations
            node_metadata: Node metadata for workflow tracking
            component_name: Optional component name for logging. If not provided,
                          uses the class name.
        """
        # Set up component-specific logging
        component = component_name or self.__class__.__name__
        self.logger = llmmllogger.bind(component=component)

        # Store required dependencies
        self.model = model
        self.profile = profile

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

        system_prompt = self.profile.system_prompt

        for msg in msgs:
            if msg.role == MessageRole.SYSTEM:
                system_prompt += f"\n\n{extract_text_from_message(msg)}"
            else:
                convo.append(msg)

        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        system_prompt += f"""
TEMPORAL CONTEXT:
The current date is {current_date}.
While this is likely past your training data, you can use this information to provide better responses. If the user asks for the date or time, respond with this date.

DO NOT EVER PROVIDE ANWERS WITH LOW CONFIDENCE. IT IS BETTER TO ADMIT YOU DON'T KNOW THAN TO MAKE UP ANSWERS. ALWAYS ATTEMPT TO USE TOOLS TO FIND THE ANSWER IF YOU ARE UNSURE.

TOOL USE:
Do not make up results - always use tools to get accurate information, or organize a way to obtain them.
If you intend to use any tools, ensure you follow the tool usage guidelines provided in the system prompt.
If there are not results from tool usage, you must attempt to call the tool again as it is likely that the format is incorrect.
If you believe you have made a tool call, double-check the message history to confirm there was a tool response included.
"""

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
            agent = await self._get_or_create_agent(
                system_prompt,
                list(set((self.tools or []) + (tools or []))),
                grammar,
                list(set((self.middleware or []) + (middleware or []))),
                metadata,
            )

            if agent is None:
                self.logger.error("🚨 Agent is None after _get_or_create_agent call!")
                raise ValueError("Agent creation failed - agent is None")

            # Convert messages to LangChain format
            normalized_messages = messages_to_lc_messages(convo)
            self.logger.debug(f"Running agent with {len(normalized_messages)} messages")
            result = await agent.ainvoke({"messages": normalized_messages})  # type: ignore
            assert isinstance(result, BaseMessage)
            self.logger.debug(
                f"Agent run result ({type(result)}): {serialize_event_data(result)}"
            )
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

    async def generate_title(
        self,
        messages: List[Message],
    ) -> str:
        """
        Generate a concise, descriptive title for a conversation based on its messages.

        Args:
            messages: List of conversation messages to analyze
            circuit_breaker: Optional circuit breaker configuration

        Returns:
            str: Generated conversation title (2-6 words)

        Raises:
            IntentAnalysisError: When title generation fails
        """

        try:
            # Only collect last 5 User/Assistant messages, and concatenate consecutive messages of the same role
            filtered = [
                m
                for m in messages
                if m.role in (MessageRole.USER, MessageRole.ASSISTANT)
            ]
            last_msgs = filtered[-5:] if len(filtered) > 5 else filtered

            # Concatenate consecutive messages of the same role
            conversation_blocks = []
            current_role = None
            current_text = ""
            for msg in last_msgs:
                text = extract_text_from_message(msg)
                if not text.strip():
                    continue
                role = (
                    MessageRole.USER
                    if msg.role == MessageRole.USER
                    else MessageRole.ASSISTANT
                )
                if role == current_role:
                    current_text += f" {text}"  # Concatenate with space
                else:
                    if current_text and current_role:
                        conversation_blocks.append(
                            f"{current_role.value}: {current_text.strip()}"
                        )
                    current_role = role
                    current_text = text
            if current_text:
                conversation_blocks.append(f"{current_role}: {current_text.strip()}")

            conversation_text = "\n".join(conversation_blocks)

            if not conversation_text.strip():
                return "New Conversation"
            title_prompt = f"""
/no_think
Generate a concise, descriptive title for this conversation. The title should:
- Be 2-6 words maximum
- Capture the main topic or purpose
- Be clear and professional
- Not include quotes or special characters
- Be suitable as a conversation label

Conversation:
{conversation_text}
"""

            result = await self.run(
                title_prompt,
                grammar=TitleResponse,
            )

            txt = (
                extract_text_from_message(result.message)
                if result and result.message
                else ""
            )
            assert txt.strip(), "Empty title generation response"

            intents = parse_structured_output(txt, TitleResponse)
            return intents.title

        except Exception as e:
            self.logger.error(
                "Title generation failed", error=str(e), context="title_generation"
            )
            # Provide fallback title instead of raising error
            return "Conversation"

    async def summarize_conversation(
        self,
        messages: List[Message],
        max_length: Optional[int] = None,
    ) -> Message:
        """
        Create primary summary of conversation messages.

        Args:
            messages: Conversation messages to summarize
            user_id: User identifier for model profile retrieval
            max_length: Optional maximum summary length
            tools: Optional tools available to the agent
            grammar: Optional grammar constraints for structured output

        Returns:
            Comprehensive primary conversation summary
        """
        try:
            self.logger.info(
                "Generating primary conversation summary",
                messages_count=len(messages),
            )

            prompt = await self._create_summary_prompt(
                [
                    f"### [{msg.role}]:\n{extract_text_from_message(msg)}\n\n---\n\n"
                    for msg in messages
                ],
                max_length,
            )

            res = await self.run(prompt)
            assert res.message is not None, "No message returned from summarization"
            summary_text = f"Here is a summary of the conversation to date:\n\n{extract_text_from_message(res.message)}"

            summary = Message(
                role=MessageRole.ASSISTANT,
                content=[
                    MessageContent(
                        type=MessageContentType.TEXT,
                        text=summary_text,
                    )
                ],
            )

            self.logger.info(
                "Generated primary conversation summary",
                summary_length=len(summary_text),
            )

            return summary

        except Exception as e:
            self.logger.error(
                "Failed to generate primary conversation summary",
                error=str(e),
            )
            raise RuntimeError(f"Primary conversation summarization failed: {e}") from e

    async def _create_summary_prompt(
        self,
        messages: List[str],
        max_length: Optional[int],
    ) -> str:
        """Create specialized prompt for primary conversation summarization."""
        length_constraint = (
            f"Keep the summary under {max_length} words." if max_length else ""
        )

        return f"""<role>
Context Extraction Assistant
</role>

<primary_objective>
Your sole objective in this task is to extract the highest quality/most relevant context from the conversation history below.
</primary_objective>

<objective_information>
You're nearing the total number of input tokens you can accept, so you must extract the highest quality/most relevant pieces of information from your conversation history.
This context will then overwrite the conversation history presented below. Because of this, ensure the context you extract is only the most important information to your overall goal.
</objective_information>

<instructions>
The conversation history below will be replaced with the context you extract in this step. Because of this, you must do your very best to extract and record all of the most important context from the conversation history.
You want to ensure that you don't repeat any actions you've already completed, so the context you extract from the conversation history should be focused on the most important information to your overall goal.
- Trace the evolution of topics and ideas throughout the conversation
- Identify key decision points and their rationale
- Highlight agreements, disagreements, and resolution processes
- Capture the flow of reasoning and argumentation
- Focus on logical progression and development of concepts
{length_constraint}
</instructions>

The user will message you with the full message history you'll be extracting context from, to then replace. Carefully read over it all, and think deeply about what information is most important to your overall goal that should be saved:

With all of this in mind, please carefully read over the entire conversation history, and extract the most important and relevant context to replace it so that you can free up space in the conversation history.
Respond ONLY with the extracted context. Do not include any additional information, or text before or after the extracted context.

<messages>
Messages to summarize:
{messages}
</messages>
"""
