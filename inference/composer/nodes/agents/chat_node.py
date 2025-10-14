"""
Chat Node for LangGraph workflows.
Uses ChatAgent for LLM chat completions within workflow execution.
"""

from typing import List, cast

from langchain.tools import BaseTool

# No additional model imports needed
from composer.graph.state import WorkflowState
from composer.core.errors import NodeExecutionError
from composer.utils.state import assemble_context_messages
from composer.nodes.base_node import BaseNode
from composer.agents.chat_agent import ChatAgent


class ChatNode(BaseNode):
    """
    Chat Node for LangGraph workflows using ChatAgent.

    Handles chat completions within workflow execution, supporting streaming,
    tool integration, and metadata tracking. Replaces the PipelineNode with
    a cleaner separation of concerns between agent logic and workflow integration.
    """

    def __init__(
        self,
        chat_agent: ChatAgent,
        node_name: str = "ChatNode",
    ):
        """
        Initialize chat node with injected ChatAgent.

        Args:
            chat_agent: Injected ChatAgent for chat operations
            node_name: Optional custom name for this node
        """
        super().__init__(node_name=node_name)
        self.chat_agent = chat_agent

    async def execute(self, state: WorkflowState) -> WorkflowState:
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

            # Create and inject node metadata
            metadata = self.create_node_metadata(
                state=state,
                model_name=self.chat_agent.profile.model_name if self.chat_agent.profile else None,
                profile_type=getattr(self.chat_agent.profile, 'profile_type', None),
                priority=self.chat_agent.priority.value if self.chat_agent.priority else None,
                streaming=self.chat_agent.stream,
                tool_count=len(state.available_tools) if state.available_tools else 0,
            )
            self.chat_agent.inject_node_metadata(metadata)

            # Assemble context messages
            context_messages = assemble_context_messages(state)
            if not context_messages:
                raise NodeExecutionError("No context messages available for chat completion")

            self.logger.info(
                "Executing chat completion",
                user_id=state.user_id,
                conversation_id=state.conversation_id,
                message_count=len(context_messages),
                tool_count=len(state.available_tools) if state.available_tools else 0,
                streaming=self.chat_agent.stream,
            )

            # Execute chat completion with conversion
            assistant_message = await self.chat_agent.chat_completion_with_conversion(
                messages=context_messages,
                user_id=state.user_id,
                tools=cast(List[BaseTool], state.available_tools) if state.available_tools else None,
                circuit_breaker=state.user_config.circuit_breaker,
                # Use agent's default stream setting
                stream=None,
            )

            # Add response to state messages
            state.messages.append(assistant_message)

            # Extract and surface tool calls for downstream nodes
            tool_calls = self.chat_agent.extract_tool_calls(assistant_message)
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

    async def __call__(self, state: WorkflowState) -> WorkflowState:
        """
        Synchronous wrapper for compatibility with PipelineNode interface.
        
        Note: This is a compatibility shim. Prefer using execute() directly
        for proper async handling.
        """
        # Simply delegate to execute method since both are now async
        return await self.execute(state)