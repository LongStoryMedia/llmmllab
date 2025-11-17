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
from models.dynamic_tool import DynamicTool
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
                    # Detect dynamic tool creation results
                    try:
                        from db import storage
                        if storage.initialized and storage.dynamic_tool:
                            dyn_service = storage.get_service(storage.dynamic_tool)
                            for tc in response.message.tool_calls:
                                if tc.name == "create_dynamic_tool" and tc.result_data:
                                    raw = tc.result_data.get("content")
                                    if not raw or not isinstance(raw, str):
                                        continue
                                    import json, textwrap
                                    spec = None
                                    try:
                                        spec = json.loads(raw)
                                    except Exception:
                                        cleaned = raw.strip()
                                        if cleaned.startswith("`"):
                                            cleaned = cleaned.strip("` ")
                                        try:
                                            spec = json.loads(cleaned)
                                        except Exception:
                                            logger.warning("Failed to parse dynamic tool spec JSON", content_preview=cleaned[:120])
                                            continue
                                    required = ["name", "description", "args_schema"]
                                    if all(k in spec for k in required):
                                        # Build minimal executable code placeholder
                                        function_name = spec["name"]
                                        code = textwrap.dedent(
                                            f"""async def {function_name}(**kwargs):\n    \"\"\"Dynamically generated tool stub. Replace implementation.\"\"\"\n    return \"Dynamic tool '{function_name}' executed with args: \" + str(kwargs)\n"""
                                        )
                                        dyn_tool = DynamicTool(
                                            user_id=state.user_id,
                                            name=spec["name"],
                                            description=spec.get("description", ""),
                                            args_schema=spec.get("args_schema"),
                                            return_direct=spec.get("return_direct", False),
                                            tags=spec.get("tags", []),
                                            metadata=spec.get("metadata", {}),
                                            handle_tool_error=False,
                                            handle_validation_error=False,
                                            response_format=spec.get("response_format", "content"),
                                            code=code,
                                            function_name=function_name,
                                        )
                                        try:
                                            saved = await dyn_service.create_tool(dyn_tool)
                                            logger.info("🧩 Dynamic tool persisted", tool_name=saved.name, tool_id=saved.id)
                                            # Register lightweight Tool model for future static loading reuse
                                            from models import Tool
                                            reg_tool = Tool(
                                                name=saved.name,
                                                description=saved.description,
                                                args_schema=saved.args_schema,
                                                return_direct=saved.return_direct,
                                                tags=saved.tags,
                                                metadata=saved.metadata,
                                                handle_tool_error=saved.handle_tool_error,
                                                handle_validation_error=saved.handle_validation_error,
                                                response_format=saved.response_format,
                                            )
                                            await self.tool_registry.register_dynamic_tool_instance(
                                                tool_id=f"{state.user_id}_{saved.name}",
                                                tool_instance=reg_tool,
                                                user_id=state.user_id,
                                            )
                                        except Exception as persist_err:
                                            logger.error("Failed to persist/register dynamic tool", error=str(persist_err))
                    except Exception as dyn_err:
                        logger.error("Dynamic tool handling error", error=str(dyn_err))
                # Convert our Message back to LangChain format and add to state
                state.messages.append(response.message)

            return state

        except Exception as e:
            logger.error(f"Agent node error: {e}", exc_info=True)
            return state
