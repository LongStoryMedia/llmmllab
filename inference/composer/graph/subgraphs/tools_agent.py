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
from composer.tools.registry import ToolRegistryManager
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
        tool_registry_manager: ToolRegistryManager,
        chat_agent: ChatAgent,
    ):
        """Initialize agent subgraph with tools."""
        self.tool_registry_manager = tool_registry_manager
        self.chat_agent = chat_agent
        self.graph = None
        self._build_graph()

        logger.info("ToolsAgentSubgraph initialized with standard LangChain pattern")

    def _build_graph(self) -> None:
        """Build standard agent following LangChain documentation exactly."""
        try:
            # We'll build the graph structure without specific tools
            # Tools will be loaded dynamically per user during execution
            logger.info("🔧 Building standard agent with dynamic tool loading")

            # Standard StateGraph following LangChain pattern
            builder = StateGraph(WorkflowState)

            # Add agent node that uses ChatOpenAI with bound tools
            builder.add_node("agent", self._agent_node)

            # Add custom tool execution node that handles user-specific tools
            builder.add_node("tools", self._tool_execution_node)

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
            # Get user-specific tools
            user_registry = await self.tool_registry_manager.get_user_registry(state.user_id)
            tools_dict = user_registry.get_all_executable_tools()
            tools_list = list(tools_dict.values()) if tools_dict else []
            
            # Invoke ChatOpenAI - this handles tool calling automatically
            logger.info("📤 Invoking ChatOpenAI with standard LangChain pattern")

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
    
    async def _tool_execution_node(self, state: WorkflowState) -> WorkflowState:
        """Custom tool execution node that uses user-specific tools."""
        try:
            # Get user-specific tools
            user_registry = await self.tool_registry_manager.get_user_registry(state.user_id)
            tools_dict = user_registry.get_all_executable_tools()
            tools_list = list(tools_dict.values()) if tools_dict else []
            
            if tools_list:
                # Create a ToolNode with user-specific tools and execute
                tool_node = ToolNode(tools_list)
                return await tool_node.ainvoke(state)
            else:
                logger.warning("No tools available for user", user_id=state.user_id)
                return state
                
        except Exception as e:
            logger.error(f"Tool execution error: {e}", exc_info=True)
            return state
