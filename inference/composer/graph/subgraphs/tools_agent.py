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

from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict
from typing import Annotated, List
from composer.graph.state import WorkflowState
from composer.agents.chat_agent import ChatAgent
from composer.tools.registry import ToolRegistry
from models import Message, PipelinePriority
from utils.message_conversion import messages_to_lc_messages, lc_messages_to_messages, message_to_lc_message
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="ToolsAgentSubgraph")


class ToolsAgentState(TypedDict):
    """Minimal state for LangChain ToolNode compatibility."""
    messages: Annotated[List[BaseMessage], add_messages]


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
            builder = StateGraph(ToolsAgentState)

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

    async def _agent_node(self, state: ToolsAgentState) -> ToolsAgentState:
        """Standard agent node using ChatOpenAI with bound tools."""
        try:
            # Get tools and bind them to ChatOpenAI
            tools_dict = self.tool_registry.get_all_executable_tools()
            tools_list = list(tools_dict.values()) if tools_dict else []
            # Invoke ChatOpenAI - this handles tool calling automatically
            logger.info("📤 Invoking ChatOpenAI with standard LangChain pattern")

            # Convert LangChain messages to our Message format for the chat agent
            our_messages = lc_messages_to_messages(state["messages"])

            response = await self.chat_agent.run(
                messages=our_messages,
                tools=tools_list,
                priority=PipelinePriority.HIGH,
            )

            logger.info(f"📨 ChatOpenAI response: {type(response)}")
            if response.message:
                if response.message.tool_calls:
                    logger.info(
                        f"🔧 Generated {len(response.message.tool_calls)} tool calls"
                    )
                # Convert our Message back to LangChain format and add to state
                lc_message = message_to_lc_message(response.message)
                state["messages"].append(lc_message)

            return state

        except Exception as e:
            logger.error(f"Agent node error: {e}", exc_info=True)
            return state

    async def execute(self, main_state: WorkflowState) -> Command:
        """Execute the agent subgraph and return Command with state updates."""
        try:
            if not self.graph:
                logger.error("Agent subgraph not initialized")
                return Command(update={})

            # Convert WorkflowState.messages to LangChain format for the subgraph
            lc_messages = messages_to_lc_messages(main_state.messages)
            tools_state = ToolsAgentState(messages=lc_messages)

            # Execute the agent subgraph with LangChain-compatible state
            result = await self.graph.ainvoke(tools_state)

            # Convert result back to our Message format
            result_messages = lc_messages_to_messages(result["messages"])

            # Return updated messages from the result
            logger.info("🔄 Agent subgraph completed")
            return Command(update={"messages": result_messages})

        except Exception as e:
            logger.error(f"Agent subgraph execution failed: {e}", exc_info=True)
            return Command(update={})
