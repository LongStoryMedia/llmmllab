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
from composer.graph.state import WorkflowState, assemble_context_messages
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
            builder = StateGraph(WorkflowState)

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

    async def _agent_node(self, state: WorkflowState) -> WorkflowState:
        """Standard agent node using ChatOpenAI with bound tools."""
        try:
            # Get tools and bind them to ChatOpenAI
            tools_dict = self.tool_registry.get_all_executable_tools()
            tools_list = list(tools_dict.values()) if tools_dict else []
            # Invoke ChatOpenAI - this handles tool calling automatically
            logger.info("📤 Invoking ChatOpenAI with standard LangChain pattern")
            
            response = await self.chat_agent.run(
                messages=state.messages,  # state.messages is already List[Message]
                tools=tools_list,
                priority=PipelinePriority.HIGH,
            )

            logger.info(f"📨 ChatOpenAI response: {type(response)}")
            if response.message:
                if response.message.tool_calls:
                    logger.info(
                        f"🔧 Generated {len(response.message.tool_calls)} tool calls"
                    )
                # Add response to messages - append to WorkflowState.messages list
                if not state.messages:
                    state.messages = []
                state.messages.append(response.message)

            return state

        except Exception as e:
            logger.error(f"Agent node error: {e}", exc_info=True)
            return state


