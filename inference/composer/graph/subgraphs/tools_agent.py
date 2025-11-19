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
from models import PipelinePriority
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
            # Initialize ToolRegistry with comprehensive tool management for this request
            user_query = ""
            if state.messages:
                # Get user query from the last user message
                for msg in reversed(state.messages):
                    # Handle different message types and extract content as string
                    if hasattr(msg, 'content'):
                        content = msg.content
                        if isinstance(content, str):
                            user_query = content
                            break
                        elif isinstance(content, list) and content:
                            # Handle multimodal content - get first text content
                            for item in content:
                                if hasattr(item, 'text') and isinstance(item.text, str):
                                    user_query = item.text
                                    break
                            if user_query:
                                break

            # Get user config from state (should be available from context assembly)
            user_config = getattr(state, 'user_config', None)
            
            # Get dynamic tool storage if available in state
            dynamic_tool_storage = getattr(state, 'dynamic_tool_storage', None)
            
            if user_config and dynamic_tool_storage:
                # Initialize ToolRegistry with all tools for this workflow execution
                await self.tool_registry.initialize_for_workflow(
                    dynamic_tool_storage=dynamic_tool_storage,
                    user_query=user_query, 
                    user_config=user_config
                )
                logger.info("🔧 ToolRegistry initialized for workflow execution")
            else:
                logger.info("⚠️ Skipping ToolRegistry initialization - missing config or storage")

            # Get all available tools from the registry (static + previous dynamic + new dynamic)
            tools_dict = self.tool_registry.get_all_executable_tools()
            tools_list = list(tools_dict.values()) if tools_dict else []

            # Invoke ChatOpenAI - this handles tool calling automatically
            logger.info(f"📤 Invoking ChatOpenAI with {len(tools_list)} tools available")

            response = await self.chat_agent.run(
                messages=state.messages,
                tools=tools_list,
                priority=PipelinePriority.HIGH,
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
