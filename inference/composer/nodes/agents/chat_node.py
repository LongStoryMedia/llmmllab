"""
Chat Node for LangGraph workflows.
Uses ChatAgent for LLM chat completions within workflow execution.
"""

# No additional model imports needed
from composer.graph.state import WorkflowState
from composer.core.errors import NodeExecutionError
from composer.utils.state import assemble_context_messages
from composer.utils.conversion import convert_messages_to_langchain
from composer.agents.chat_agent import ChatAgent
from runner import PipelineFactory
from models import ModelProfile, PipelinePriority
from utils.logging import llmmllogger


class ChatNode:
    """
    Chat Node for LangGraph workflows using ChatAgent.

    Handles chat completions within workflow execution, supporting streaming,
    tool integration, and metadata tracking. Replaces the PipelineNode with
    a cleaner separation of concerns between agent logic and workflow integration.
    """

    def __init__(
        self,
        pipeline_factory: PipelineFactory,
        agent: ChatAgent,
        tool_registry=None,  # Optional for backward compatibility
    ):
        """
        Initialize chat node with dependencies for ChatAgent creation.

        Args:
            pipeline_factory: Factory for creating chat pipelines
            agent: ChatAgent instance for handling chat operations
            tool_registry: Optional tool registry for tool conversion
        """
        self.pipeline_factory = pipeline_factory
        self.agent = agent
        self.tool_registry = tool_registry
        self.logger = llmmllogger.logger.bind(component="ChatNode")

    async def __call__(self, state: WorkflowState) -> WorkflowState:
        """
        Execute chat node with ChatAgent.

        Args:
            state: Current workflow state

        Returns:
            Updated workflow state with chat response
        """
        try:
            # Validate required state
            if not state.user_id:
                raise NodeExecutionError("User ID required for chat execution")

            if not state.user_config:
                raise NodeExecutionError("User config required for chat execution")
            # Assemble context messages
            context_messages = assemble_context_messages(state)
            if not context_messages:
                raise NodeExecutionError(
                    "No context messages available for chat completion"
                )

            self.logger.info(
                "Executing chat completion",
                user_id=state.user_id,
                conversation_id=state.conversation_id,
                message_count=len(context_messages),
                tool_count=len(state.available_tools) if state.available_tools else 0,
                streaming=True,
            )

            # Execute chat completion with conversion
            assistant_message = await self.agent.chat_completion_with_conversion(
                messages=convert_messages_to_langchain(context_messages),
                tools=(
                    self.tool_registry.convert_tools_to_langchain(state.available_tools)
                    if self.tool_registry and state.available_tools
                    else None
                ),
                # Use agent's default stream setting
                stream=None,
            )

            # Add response to state messages
            state.messages.append(assistant_message)

            # Extract and surface tool calls for downstream nodes
            tool_calls = self.agent.extract_tool_calls(assistant_message)
            state.tool_calls = tool_calls

            self.logger.info(
                "Chat completion successful",
                user_id=state.user_id,
                conversation_id=state.conversation_id,
                has_tool_calls=bool(tool_calls),
                tool_calls_count=len(tool_calls) if tool_calls else 0,
                message_added=True,
            )

            return state

        except Exception as e:
            self.logger.error(
                "Chat node execution failed",
                user_id=getattr(state, "user_id", "unknown"),
                conversation_id=getattr(state, "conversation_id", None),
                error=str(e),
            )
            raise NodeExecutionError(f"Chat execution failed: {e}") from e
