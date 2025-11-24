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

from typing import List

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages.utils import count_tokens_approximately
from composer.graph import WorkflowState
from composer.agents.chat_agent import ChatAgent
from composer.tools.registry import ToolRegistry
from models import NodeMetadata, PipelinePriority
from utils import extract_text_from_message
from utils.logging import llmmllogger, serialize_event_data
from .summarization_middleware import (
    SummarizationMiddleware,
    DEFAULT_SUMMARY_PROMPT,
    SUMMARY_PREFIX,
)

logger = llmmllogger.bind(component="ToolsAgentSubgraph")


def should_continue_tool_calls(state: WorkflowState) -> str:
    """Determine if the agent should continue making tool calls based on the last message."""
    # Get the last message from state
    if not state.messages:
        return "end"

    last_message = state.messages[-1]

    # Check if the last message has tool calls
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"


class ToolsAgentSubgraph:
    """
    Standard LangChain agent subgraph following official documentation pattern.
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        chat_agent: ChatAgent,
        node_metadata: NodeMetadata,
    ):
        """Initialize agent subgraph with tools."""
        self.tool_registry = tool_registry
        self.chat_agent = chat_agent.bind_node_metadata(node_metadata)
        self.graph = None
        self._build_graph()

        logger.info("ToolsAgentSubgraph initialized with standard LangChain pattern")

    def _build_graph(self) -> None:
        """Build standard agent following LangChain documentation exactly."""
        try:
            # Get user-specific tools for ToolNode
            tools_dict = self.tool_registry.get_all_executable_tools()
            tools_list = list(tools_dict.values()) if tools_dict else []

            logger.info(f"🔧 Building standard agent with {len(tools_list)} user tools")

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
                    "agent", should_continue_tool_calls, {"tools": "tools", "end": END}
                )
                builder.add_edge("tools", "agent")
            else:
                logger.warning("No tools available for user")

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
            # Get user config from state (should be available from context assembly)
            assert state.current_user_message
            assert state.user_config

            if state.user_config:
                # Initialize ToolRegistry with all tools for this workflow execution
                await self.tool_registry.initialize_for_workflow(
                    user_query=extract_text_from_message(state.current_user_message),
                    user_config=state.user_config,
                )
                logger.info("🔧 ToolRegistry initialized for workflow execution")
            else:
                logger.info(
                    "⚠️ Skipping ToolRegistry initialization - missing config or storage"
                )

            # Get all available tools from the registry (static + previous dynamic + new dynamic)
            tools_dict = self.tool_registry.get_all_executable_tools()
            tools_list = list(tools_dict.values()) if tools_dict else []

            # Invoke ChatOpenAI - this handles tool calling automatically
            logger.info(
                f"📤 Invoking ChatOpenAI with {len(tools_list)} tools available"
            )

            logger.debug(f"tools: {tools_dict.keys()}")

            n_ctx = self.chat_agent.profile.parameters.batch_size or 4096
            logger.debug(f"Model context length: {n_ctx}")
            max_tokens_before_summary = int(n_ctx * 0.95)
            logger.debug(f"Max tokens before summary: {max_tokens_before_summary}")

            summarizer = SummarizationMiddleware(
                agent=self.chat_agent,
                max_tokens_before_summary=10000,
                messages_to_keep=10,
                summary_prompt=DEFAULT_SUMMARY_PROMPT,
                summary_prefix=SUMMARY_PREFIX,
                token_counter=count_tokens_approximately,
            )

            # Perform async summarization prior to model invocation.
            state.messages = await summarizer.maybe_summarize(state.messages)  # type: ignore[assignment]

            response = await self.chat_agent.run(
                messages=state.messages,
                tools=tools_list,
                priority=PipelinePriority.HIGH,
                middleware=[],
            )

            # Persist todos extracted in ChatResponse (already converted in BaseAgent)
            if response.todos:
                try:
                    from db import storage

                    todos_to_store = response.todos
                    if storage.initialized and storage.todo:
                        svc = storage.get_service(storage.todo)
                        saved = []
                        for td in todos_to_store:
                            saved_item = await svc.add_todo(td)
                            if saved_item:
                                saved.append(saved_item)
                        state.generated_todos.extend(saved if saved else todos_to_store)
                    else:
                        state.generated_todos.extend(todos_to_store)
                    logger.info(
                        f"📝 Stored {len(state.generated_todos)} todos from middleware result"
                    )
                except Exception as db_err:
                    logger.error(f"Failed storing middleware todos: {db_err}")
                    state.generated_todos.extend(response.todos)

            logger.info(f"📨 ChatOpenAI response: {type(response)}")
            logger.debug(f"Response content: {serialize_event_data(response)}")

            if response.message:
                if response.message.tool_calls:
                    logger.info(
                        f"🔧 Generated {len(response.message.tool_calls)} tool calls"
                    )
                # Convert our Message back to LangChain format and add to state
                state.messages.append(response.message)

            return state

        except Exception as e:
            logger.error(f"Agent node error: {e}", exc_info=True)
            return state
