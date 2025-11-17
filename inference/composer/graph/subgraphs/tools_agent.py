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

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from composer.graph import WorkflowState
from composer.agents.chat_agent import ChatAgent
from composer.tools.registry import ToolRegistry
from models import PipelinePriority, TodoItem
from utils.logging import llmmllogger, serialize_event_data

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
                    "agent", should_continue_tool_calls, {"tools": "tools", "end": END}
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
                messages=state.messages,
                tools=tools_list,
                priority=PipelinePriority.HIGH,
            )

            # Extract todos from middleware if present on agent
            todos_created = []
            try:
                if hasattr(self.chat_agent, "middleware"):
                    for mw in self.chat_agent.middleware:  # type: ignore[attr-defined]
                        # TodoListMiddleware exposes .todo_list attribute
                        if hasattr(mw, "todo_list") and mw.todo_list:  # type: ignore
                            for raw in mw.todo_list:  # type: ignore[index]
                                # raw is likely a dict with keys 'description' or 'task'
                                task_text = (
                                    raw.get("task")
                                    or raw.get("description")
                                    or raw.get("text")
                                    or "Untitled Task"
                                )
                                # Simple priority heuristic
                                priority = "medium"
                                lowered = task_text.lower()
                                if any(p in lowered for p in ["urgent", "asap", "immediately"]):
                                    priority = "urgent"
                                # Create TodoItem
                                todo = TodoItem(
                                    user_id=state.user_id,
                                    conversation_id=state.conversation_id,
                                    title=task_text[:60],
                                    description=task_text,
                                    status="not-started",
                                    priority=priority,
                                )
                                todos_created.append(todo)
            except Exception as mw_err:
                logger.warning(f"Failed extracting middleware todos: {mw_err}")

            # Persist todos and append to state
            if todos_created:
                try:
                    from db import storage
                    if storage.initialized and storage.todo:
                        svc = storage.get_service(storage.todo)
                        saved = []
                        for td in todos_created:
                            saved_item = await svc.add_todo(td)
                            if saved_item:
                                saved.append(saved_item)
                        state.generated_todos.extend(saved if saved else todos_created)
                    else:
                        state.generated_todos.extend(todos_created)
                    logger.info(f"📝 Stored {len(state.generated_todos)} todos from middleware")
                except Exception as db_err:
                    logger.error(f"Failed storing middleware todos: {db_err}")
                    state.generated_todos.extend(todos_created)

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
