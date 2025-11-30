"""
Base Agent class providing common functionality for all workflow agents.
Provides node metadata injection, logging setup, and common error handling patterns.
"""

import datetime
import logging
from typing import (
    Optional,
    Any,
    Dict,
    Self,
    TypeVar,
    Generic,
    List,
    cast,
)
from abc import ABC
from pydantic import BaseModel
from langchain.agents.structured_output import ProviderStrategy
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.chat_models import BaseChatModel
from langchain.embeddings.base import Embeddings
from langchain_core.tools import BaseTool
from langchain_core.messages import BaseMessage

from models import (
    MessageContent,
    MessageContentType,
    MessageRole,
    ModelProvider,
    NodeMetadata,
    ModelProfile,
    ChatResponse,
    PipelinePriority,
    Message,
    Summary,
    SummaryStyle,
)
from runner import PipelineFactory
from runner.pipelines.base import BasePipeline
from utils import parse_structured_output
from utils.logging import llmmllogger, serialize_event_data
from utils.response import create_error_response
from utils.message_conversion import (
    normalize_message_input,
    messages_to_lc_messages,
    lc_message_to_message,
    MessageInput,
    extract_text_from_message,
)
from composer.core.errors import NodeExecutionError

from .grammar_responses import TitleResponse


# Get the 'asyncio' logger
asyncio_logger = logging.getLogger("asyncio")

# Set the logging level to WARNING or higher (e.g., ERROR, CRITICAL)
# This will prevent INFO and DEBUG messages from being displayed when run_sync is used.
asyncio_logger.setLevel(logging.WARNING)

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
        component_name: Optional[str] = None,
        middleware: Optional[List[AgentMiddleware]] = None,
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

        # Additional metadata for debugging and tracking
        self._execution_context: Dict[str, Any] = {}

        # Persistent pipeline reference - prevents garbage collection
        self._pipeline: Optional[BaseChatModel | Embeddings] = None

        # Track if we have locked a pipeline that needs cleanup
        self._pipeline_locked = False

        self.agent_id = f"{id(self):x}"
        # Middleware list passed to create_agent for behaviors like TodoListMiddleware
        self.middleware: List[AgentMiddleware] = middleware or []

        self.logger.debug(
            f"Initialized {component}",
            model_name=profile.model_name,
        )

        self._node_metadata = NodeMetadata(
            node_name="UNSET",
            node_id="UNSET",
            node_type="Base",
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
            conversation_id=metadata.conversation_id,
        )
        self.logger.debug(
            "Bound new node metadata to agent",
            node_name=metadata.node_name,
            node_type=metadata.node_type,
        )
        return self

    def cleanup(self) -> None:
        """
        Clean up resources used by this agent.

        Only unlocks pipeline but does NOT force it out of cache, allowing
        other components to reuse it. The pipeline remains cached based on
        the intelligent eviction strategy in LocalPipelineCacheManager.
        """
        if self._pipeline_locked:
            try:
                self.logger.info(
                    f"🔓 Unlocking agent pipeline for model {self.profile.model_name} (keeping in cache for reuse)"
                )
                success = self.pipeline_factory.unlock_pipeline(self.profile)
                if success:
                    self.logger.info(
                        f"✅ Successfully unlocked pipeline for {self.profile.model_name} - available for reuse"
                    )
                    self._pipeline_locked = False
                else:
                    self.logger.warning(
                        f"⚠️ Failed to unlock pipeline for {self.profile.model_name}"
                    )
            except Exception as e:
                self.logger.error(
                    f"❌ Error during pipeline unlock for {self.profile.model_name}: {e}"
                )

        # Keep pipeline reference for potential reuse within same agent
        # Only clear it when agent is truly destroyed
        self.logger.debug("Agent cleanup completed - pipeline remains cached for reuse")

    def __del__(self):
        """Automatic cleanup when agent is garbage collected."""
        try:
            self.cleanup()
        except Exception:
            # Silently ignore cleanup errors during destruction
            pass

    async def get_pipeline(
        self,
        priority: PipelinePriority = PipelinePriority.MEDIUM,
        grammar: Optional[type[BaseModel]] = None,
        metadata: Optional[NodeMetadata] = None,
    ) -> Optional[BaseChatModel | Embeddings]:
        """
        Get the current pipeline instance if available.

        Returns:
            The current pipeline instance or None if not created.
        """
        if self._pipeline is None:
            self._pipeline = self.pipeline_factory.get_pipeline(
                self.profile,
                priority,
                grammar,
                self._node_metadata.model_dump(),
            )

            if not self._pipeline.started:  # type: ignore
                from composer import (  # pylint: disable=import-outside-toplevel
                    clear_workflow_cache,
                )

                self.logger.error("🚨 Pipeline is not started after creation!")
                assert self._node_metadata.user_id is not None
                await clear_workflow_cache(self._node_metadata.user_id)

            # Mark that we have locked a pipeline that needs cleanup
            self._pipeline_locked = True
            self.logger.debug(
                f"🔒 Agent {self.agent_id} locked new pipeline for {self.profile.model_name}"
            )

        if metadata is not None and isinstance(self._pipeline, BasePipeline):
            self._pipeline.bind_metadata(metadata.model_dump())
        return self._pipeline

    async def _get_or_create_agent(
        self,
        system_prompt,
        tools: Optional[List[BaseTool]] = None,
        priority: PipelinePriority = PipelinePriority.MEDIUM,
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
        pipeline = await self.get_pipeline(priority, grammar, metadata)
        if pipeline is None:
            self.logger.error("🚨 Pipeline is None after get_pipeline call!")
            raise ValueError("Pipeline creation failed - pipeline is None")

        llm = cast(BaseChatModel, pipeline)
        agent = create_agent(
            model=llm,
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

    async def _is_llama_model(self) -> bool:
        """
        Detect if the current model is llama.cpp based.
        Used to determine whether to use file-friendly or base64 format for multimodal content.
        """
        pipe = await self.get_pipeline()
        if pipe and isinstance(pipe, BasePipeline):
            if pipe.model.provider == ModelProvider.LLAMA_CPP:
                return True

        model_name = self.profile.model_name.lower()

        # Check for common llama.cpp model patterns
        llama_indicators = [
            "llama",
            "mistral",
            "qwen",
            "phi",
            "gemma",
            "codellama",
            "alpaca",
            "vicuna",
            "orca",
            "wizard",
            "dolphin",
            "openchat",
            "starling",
        ]

        # Also check for .gguf file extension (common llama.cpp format)
        if ".gguf" in model_name or "ggml" in model_name:
            return True

        # Check if model name contains llama indicators
        return any(indicator in model_name for indicator in llama_indicators)

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
        priority: PipelinePriority = PipelinePriority.MEDIUM,
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
                tools,
                priority,
                grammar,
                list(set((self.middleware or []) + (middleware or []))),
                metadata,
            )

            if agent is None:
                self.logger.error("🚨 Agent is None after _get_or_create_agent call!")
                raise ValueError("Agent creation failed - agent is None")

            # Convert messages to LangChain format
            # Detect if we're using llama.cpp and need file-friendly format
            use_llama_format = await self._is_llama_model()
            normalized_messages = messages_to_lc_messages(convo, use_llama_format)

            self.logger.debug(f"Running agent with {len(normalized_messages)} messages")

            result = await agent.ainvoke({"messages": normalized_messages})  # type: ignore

            # Convert agent result to ChatResponse
            if isinstance(result, BaseMessage):
                last_message = result
            elif isinstance(result, dict) and "messages" in result:
                last_message = result["messages"][-1]
            else:
                last_message = result

            assert isinstance(last_message, BaseMessage)
            self.logger.debug(
                f"Agent run result ({type(last_message)}): {serialize_event_data(last_message)}"
            )
            msg = lc_message_to_message(last_message)

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
        conversation_id: int,
        max_length: Optional[int] = None,
        style: SummaryStyle = SummaryStyle.CONCISE,
    ) -> Summary:
        """
        Create primary summary of conversation messages.

        Args:
            messages: Conversation messages to summarize
            user_id: User identifier for model profile retrieval
            max_length: Optional maximum summary length
            style: Summary style preference
            tools: Optional tools available to the agent
            grammar: Optional grammar constraints for structured output

        Returns:
            Comprehensive primary conversation summary
        """
        try:
            self.logger.info(
                "Generating primary conversation summary",
                messages_count=len(messages),
                style=style,
            )

            prompt = await self._create_primary_conversation_prompt(
                [
                    f"### [{msg.role}]:\n{extract_text_from_message(msg)}\n\n---\n\n"
                    for msg in messages
                ],
                style,
                max_length,
            )

            metadata = self._node_metadata.model_copy()
            metadata.node_name = "PrimaryConversationSummaryAgent"
            metadata.node_type = "Summarization"

            res = await self.run(prompt, metadata=metadata)
            await self.get_pipeline(
                metadata=self._node_metadata
            )  # Reset pipeline metadata
            assert res.message is not None, "No message returned from summary agent"
            summary_text = extract_text_from_message(res.message)

            summary = Summary(
                id=0,  # ID to be set when storing
                content=summary_text,
                level=1,
                conversation_id=conversation_id,
                created_at=datetime.datetime.now(datetime.timezone.utc),
                source_ids=[msg.id for msg in messages if msg.id is not None],
            )

            from db import storage  # pylint: disable=import-outside-toplevel

            # Store primary conversation summary
            await storage.get_service(storage.summary).create_summary(summary)

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
            raise NodeExecutionError(
                f"Primary conversation summarization failed: {e}"
            ) from e

    async def _create_primary_conversation_prompt(
        self, messages: List[str], style: SummaryStyle, max_length: Optional[int]
    ) -> str:
        """Create specialized prompt for primary conversation summarization."""
        style_instruction = self._get_style_instruction(style)
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
STYLE: {style_instruction}
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

    def _get_style_instruction(self, style: SummaryStyle) -> str:
        """Get style-specific instruction for primary summaries."""
        style_instructions = {
            SummaryStyle.CONCISE: "Provide a focused yet comprehensive analysis.",
            SummaryStyle.DETAILED: "Provide extensive detail and comprehensive coverage.",
            SummaryStyle.BULLET_POINTS: "Use structured bullet points for comprehensive analysis.",
        }
        return style_instructions.get(
            style, "Provide a balanced comprehensive analysis."
        )
