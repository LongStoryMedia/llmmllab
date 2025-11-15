"""
Tools Agent Subgraph - Standard LangChain Agent Pattern

Following the exact pattern from LangChain documentation:
https://docs.langchain.com/oss/python/langgraph/workflows-agents#agents

Standard architecture:
1. agent: ChatOpenAI with tools bound via bind_tools()
2. tools: Standard ToolNode that handles execution
3. Built-in tools_condition for routing
4. No manual extraction or conversion - LangChain handles everything
"""

import asyncio
from typing import Dict, Any

from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command
from composer.graph.state import WorkflowState, ToolsState, assemble_context_messages
from composer.agents.chat_agent import ChatAgent
from composer.tools.registry import ToolRegistry
from models import Message, PipelinePriority
from utils.message_conversion import (
    lc_messages_to_messages,
    message_to_lc_message,
    messages_to_lc_messages,
)
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="ToolsAgentSubgraph")


def should_continue_tool_calls(message: Message) -> bool:
    """Determine if the agent should continue making tool calls based on the message."""
    # Example logic: continue if there are tool calls remaining
    if hasattr(message, "tool_calls") and message.tool_calls:
        return True
    return False


class ToolsAgentSubgraph:
    """
    Standard LangChain agent subgraph following official documentation pattern.
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        chat_agent: ChatAgent,
    ):
        """Initialize agent subgraph with tools."""
        self.tool_registry = tool_registry
        self.chat_agent = chat_agent
        self.graph = None
        self._build_graph()

        logger.info("ToolsAgentSubgraph initialized with standard LangChain pattern")

    def _build_graph(self) -> None:
        """Build standard agent following LangChain documentation exactly."""
        try:
            # Get tools for ToolNode
            tools_dict = self.tool_registry.get_all_executable_tools()
            tools_list = list(tools_dict.values()) if tools_dict else []

            logger.info(f"🔧 Building standard agent with {len(tools_list)} tools")

            # Standard StateGraph following LangChain pattern
            builder = StateGraph(ToolsState)

            # Add agent node that uses ChatOpenAI with bound tools
            builder.add_node("agent", self._agent_node)

            # Add standard ToolNode for tool execution
            if tools_list:
                tool_node = ToolNode(tools_list)
                builder.add_node("tools", tool_node)

                # Standard conditional routing using LangChain's tools_condition
                builder.add_conditional_edges(
                    "agent",
                    tools_condition,  # Standard LangChain routing
                )
                builder.add_edge("tools", "agent")

            # Start with agent
            builder.add_edge(START, "agent")

            # Compile with standard settings
            self.graph = builder.compile()

            logger.info("✅ Standard LangChain agent subgraph built successfully")

        except Exception as e:
            logger.error(f"Failed to build standard agent subgraph: {e}")
            raise

    async def _agent_node(self, state: ToolsState) -> ToolsState:
        """Standard agent node using ChatOpenAI with bound tools."""
        try:
            # Get tools and bind them to ChatOpenAI
            tools_dict = self.tool_registry.get_all_executable_tools()
            tools_list = list(tools_dict.values()) if tools_dict else []
            # Invoke ChatOpenAI - this handles tool calling automatically
            logger.info("📤 Invoking ChatOpenAI with standard LangChain pattern")
            # async for chunk in self.chat_agent.stream(
            #     lc_messages_to_messages(state.messages),
            #     tools=tools_list,
            #     priority=PipelinePriority.HIGH,
            # ):
            #     logger.debug(f"📥 Agent Received chunk: {chunk.model_dump_json()}")

            response = await self.chat_agent.run(
                messages=lc_messages_to_messages(state.messages),
                tools=tools_list,
                priority=PipelinePriority.HIGH,
            )

            logger.info(f"📨 ChatOpenAI response: {type(response)}")
            if response.message:
                if response.message.tool_calls:
                    logger.info(
                        f"🔧 Generated {len(response.message.tool_calls)} tool calls"
                    )
                # Add response to messages
                state.messages.append(message_to_lc_message(response.message))

            # Inject shared pipeline for tool reuse
            if self.chat_agent.current_pipeline and not state.shared_pipeline:
                state.shared_pipeline = self.chat_agent.current_pipeline
                logger.debug("💾 Injected shared pipeline into state")

            return state

        except Exception as e:
            logger.error(f"Agent node error: {e}", exc_info=True)
            return state

    def transform_to_tools_state(self, main_state: WorkflowState) -> ToolsState:
        """Transform main WorkflowState to minimal ToolsState for agent subgraph."""
        assert main_state.user_config
        assert main_state.user_id
        assert main_state.conversation_id
        assert main_state.messages

        return ToolsState(
            messages=messages_to_lc_messages(assemble_context_messages(main_state)),
            user_id=main_state.user_id,
            conversation_id=main_state.conversation_id,
            user_config=main_state.user_config,
            tool_call_count=0,
        )

    def transform_to_main_state(
        self, agent_result: Dict[str, Any], main_state: WorkflowState
    ) -> Dict[str, Any]:
        """Transform agent subgraph results back to main WorkflowState updates."""
        updates = {}

        # Add new messages from agent execution
        if agent_result.get("messages"):
            main_messages = getattr(main_state, "messages", [])
            agent_messages = agent_result["messages"]

            # Find messages that weren't in the original main state
            original_count = len(main_messages)
            new_messages = []

            for i, msg in enumerate(agent_messages):
                if i >= original_count:  # This is a new message from agent
                    # Convert to Message format for main state
                    logger.info(f"🔄 Converting {type(msg).__name__} to Message")
                    # Convert BaseMessage to Message using existing utilities
                    messages_list = lc_messages_to_messages([msg])
                    if messages_list:
                        message_obj = messages_list[0]
                        message_obj.conversation_id = getattr(
                            main_state, "conversation_id", None
                        )
                        logger.info(
                            f"🔄 Created Message with role='{message_obj.role}'"
                        )
                        new_messages.append(message_obj)

            if new_messages:
                updates["messages"] = main_messages + new_messages

        return updates

    async def execute(self, main_state: WorkflowState) -> Command:
        """Execute the agent subgraph and return Command with state updates."""
        try:
            if not self.graph:
                logger.error("Agent subgraph not initialized")
                return Command(update={})

            # Transform to agent state
            tools_state = self.transform_to_tools_state(main_state)

            # Execute the agent subgraph without timeout constraints
            logger.info("🔄 Executing agent subgraph with ainvoke")

            try:
                result = await self.graph.ainvoke(tools_state)
                logger.info("🔄 Agent subgraph ainvoke completed successfully")
            except Exception as e:
                logger.error(f"❌ Agent subgraph ainvoke execution failed: {e}")
                return Command(update={})

            # Transform results back to main state updates
            logger.info(
                f"🔄 Agent subgraph completed with {len(result.get('messages', []))} messages"
            )
            updates = self.transform_to_main_state(result, main_state)

            logger.info(f"🔄 Agent subgraph returning {len(updates)} state updates")
            if "messages" in updates:
                logger.info(
                    f"🔄 Returning {len(updates['messages']) - len(main_state.messages)} new messages"
                )

            return Command(update=updates)

        except Exception as e:
            logger.error(f"Agent subgraph execution failed: {e}", exc_info=True)
            return Command(update={})
