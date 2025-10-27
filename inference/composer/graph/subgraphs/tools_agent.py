"""
Tools Agent Subgraph - LangGraph middleware-based agent workflow.

This subgraph implements proper LangGraph agent patterns using built-in middleware
for tool routing instead of manual conditional logic. Uses tools_condition from
langgraph.prebuilt to handle agent cycling, preventing duplicate executions and
ensuring proper termination conditions.

Key Benefits:
1. Built-in middleware - uses LangGraph's tools_condition for proper routing
2. Prevents duplicate execution - proper termination logic prevents agent loops
3. Minimal state - ToolsState with only essential fields to minimize context usage
4. ToolRuntime pattern - all tools use modern ToolRuntime[ToolsState] injection
5. State isolation - agent operations don't bloat main workflow state

Architecture:
- ToolsState: Minimal state optimized for agent operations
- chat_agent: LLM node that can make tool calls using available tools
- tool_executor: ToolNode that executes tools with ToolRuntime[ToolsState] access
- tools_condition: Built-in LangGraph middleware for proper agent loop control
- Middleware boundaries: controlled data flow to/from main workflow
"""

from typing import Dict, Any
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

from models import NodeMetadata, PipelinePriority
from composer.graph.state import WorkflowState, ToolsState
from composer.agents.chat_agent import ChatAgent
from composer.tools.registry import ToolRegistry
from runner import PipelineFactory
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="ToolsAgentSubgraph")


@dataclass
class ToolsContext:
    """Context schema for tools runtime - provides state access for ToolRuntime injection."""

    state: ToolsState

    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access to state for compatibility."""
        return getattr(self.state, key, None)


class ToolsAgentSubgraph:
    """
    Complete agent subgraph with chat_agent + tool_node cycling workflow.

    Uses proper dependency injection pattern like the main graph builder,
    importing ChatAgent and ToolExecutorNode with their required dependencies.
    """

    def __init__(
        self,
        pipeline_factory: PipelineFactory,
        tool_registry: ToolRegistry,
    ):
        """Initialize subgraph with dependency injection."""
        self.pipeline_factory = pipeline_factory
        self.tool_registry = tool_registry
        self.graph = None

        # Create node metadata for the subgraph agents
        self.subgraph_metadata = NodeMetadata(
            node_name="tools_agent_subgraph",
            node_id="tools_agent_subgraph",
            node_type="subgraph",
            user_id="system",  # Will be updated at runtime
            conversation_id=0,  # Will be updated at runtime
        )

        self._build_graph()

    def _create_chat_agent(self, user_id: str, conversation_id: int) -> ChatAgent:
        """Create ChatAgent instance with proper dependency injection."""
        from models.default_model_profiles import DEFAULT_PRIMARY_PROFILE

        # Update metadata with runtime context
        runtime_metadata = NodeMetadata(
            node_name="subgraph_chat_agent",
            node_id="subgraph_chat_agent",
            node_type="agent",
            user_id=user_id,
            conversation_id=conversation_id,
        )

        return ChatAgent(
            pipeline_factory=self.pipeline_factory,
            profile=DEFAULT_PRIMARY_PROFILE,
            node_metadata=runtime_metadata,
            priority=PipelinePriority.MEDIUM,
        )

    def _create_tool_node(self) -> ToolNode:
        """
        Create LangGraph ToolNode with proper tools list and ToolRuntime injection.

        LangChain will automatically inject ToolRuntime for tools with `runtime: ToolRuntime` parameter.
        This is the correct pattern - no manual ToolRuntime creation needed.
        """
        try:
            # Get executable tools from registry
            executable_tools = self.tool_registry.get_all_executable_tools()
            tools_dict: dict[str, Any] = executable_tools if executable_tools else {}

            if not tools_dict:
                logger.warning("No tools available for ToolNode creation")
                return ToolNode([])  # Empty tool node

            # Convert to list of tool functions for ToolNode
            tools_list = list(tools_dict.values())

            logger.info(
                f"🛠️ Creating ToolNode with {len(tools_list)} tools: {list(tools_dict.keys())}"
            )

            # Create ToolNode - LangChain will handle ToolRuntime injection automatically
            return ToolNode(tools_list)

        except Exception as e:
            logger.error(f"Failed to create ToolNode: {e}")
            return ToolNode([])  # Return empty tool node on error

    def _build_graph(self) -> None:
        """Build the complete agent subgraph using proper dependency injection."""
        try:
            # Build graph with StateGraph pattern like main builder
            # Add context_schema to enable ToolRuntime context propagation to subgraphs
            builder = StateGraph(ToolsState, context_schema=ToolsContext)

            # Add chat agent node - will be created at runtime with proper context
            builder.add_node("chat_agent", self._chat_agent_wrapper)

            # Add tool executor node - using LangGraph's ToolNode with automatic ToolRuntime injection
            tool_node = self._create_tool_node()
            builder.add_node("tool_executor", tool_node)

            # Sophisticated routing for intelligent agent cycling with tools
            def should_execute_tools(state: ToolsState):
                """
                Intelligent tool execution router with limiting middleware.

                Implements tool call limiting and planning middleware patterns:
                - Limits total tool calls per session
                - Prevents rapid-fire identical calls
                - Implements planning-based decision making
                """
                messages = state.get("messages", [])
                if not messages:
                    logger.info("🔀 Subgraph: No messages, finishing")
                    return "__end__"

                last_message = messages[-1]

                # Check if last message has tool calls
                if (
                    isinstance(last_message, AIMessage)
                    and hasattr(last_message, "tool_calls")
                    and last_message.tool_calls
                ):

                    # Log tool call grouping information
                    tool_calls = last_message.tool_calls
                    grouped_calls = {}
                    for tc in tool_calls:
                        tool_name = tc.get("name", "unknown")
                        if tool_name not in grouped_calls:
                            grouped_calls[tool_name] = []
                        grouped_calls[tool_name].append(tc)

                    # Log grouping details
                    for tool_name, calls in grouped_calls.items():
                        if len(calls) > 1:
                            logger.info(
                                f"🛠️ Agent routing: Detected {len(calls)} grouped {tool_name} calls"
                            )
                            if tool_name == "web_search":
                                queries = [
                                    tc.get("args", {}).get("query", "")[:30]
                                    for tc in calls
                                ]
                                logger.info(f"🛠️ Web search topics: {queries}")

                    # Tool call limiting middleware - count various metrics
                    ai_with_tools_count = 0
                    total_tool_calls = 0
                    recent_tool_calls = 0
                    tool_call_types = set()

                    for i, msg in enumerate(messages):
                        if (
                            isinstance(msg, AIMessage)
                            and hasattr(msg, "tool_calls")
                            and msg.tool_calls
                        ):
                            ai_with_tools_count += 1
                            total_tool_calls += len(msg.tool_calls)

                            # Count recent calls (last 5 messages)
                            if i >= len(messages) - 5:
                                recent_tool_calls += len(msg.tool_calls)

                            # Track tool types
                            for tc in msg.tool_calls:
                                tool_call_types.add(tc.get("name", "unknown"))

                    # Apply limiting middleware rules (relaxed to allow natural multiple tool calls)

                    # Rule 1: Total AI messages with tools limit (increased)
                    if ai_with_tools_count >= 8:
                        logger.info(
                            f"🔀 Subgraph: Tool limit reached - too many AI messages with tools ({ai_with_tools_count})"
                        )
                        return "__end__"

                    # Rule 2: Total tool calls limit (increased)
                    if total_tool_calls >= 25:
                        logger.info(
                            f"🔀 Subgraph: Tool limit reached - too many total tool calls ({total_tool_calls})"
                        )
                        return "__end__"

                    # Rule 3: Recent tool calls limit - only prevent extreme rapid-fire
                    if recent_tool_calls >= 12:
                        logger.info(
                            f"🔀 Subgraph: Tool limit reached - too many recent tool calls ({recent_tool_calls})"
                        )
                        return "__end__"

                    # Rule 4: Web search result availability check
                    if "web_search" in tool_call_types:
                        # Check if we already have successful web search results
                        web_search_results_count = 0
                        for msg in messages:
                            if isinstance(
                                msg, ToolMessage
                            ) and "Web search completed successfully" in str(
                                msg.content
                            ):
                                web_search_results_count += 1

                        # If we have multiple successful searches and agent wants to search again, check for redundancy
                        if web_search_results_count >= 2 and ai_with_tools_count >= 3:
                            logger.info(
                                f"🔀 Subgraph: Web search results already available ({web_search_results_count} successful searches), encouraging synthesis instead of more searches"
                            )
                            # Don't immediately end, but make this the final tool iteration
                            if ai_with_tools_count >= 4:
                                return "__end__"

                    # Rule 5: Planning middleware - allow more calls of same type but with limits
                    if len(tool_call_types) == 1 and total_tool_calls >= 10:
                        logger.info(
                            f"🔀 Subgraph: Planning limit - too many calls of same tool type ({list(tool_call_types)})"
                        )
                        return "__end__"

                    logger.info(
                        f"🔀 Subgraph: Tool execution approved - AI msgs: {ai_with_tools_count}, total calls: {total_tool_calls}, recent: {recent_tool_calls}, types: {len(tool_call_types)}"
                    )
                    return "tools"

                # No tool calls, finish
                logger.info("🔀 Subgraph: No tool calls found, finishing")
                return "__end__"

            def should_continue_agent_loop(state: ToolsState):
                """
                Intelligent agent loop continuation with planning middleware.

                Implements sophisticated planning patterns:
                - Analyzes conversation flow and completion signals
                - Prevents endless loops while allowing natural reasoning
                - Recognizes when agent has sufficient information
                """
                messages = state.get("messages", [])
                if not messages:
                    logger.info("🔀 Subgraph: No messages after tools, finishing")
                    return "__end__"

                # Analyze recent conversation pattern
                recent_messages = messages[-8:]  # Analyze last 8 messages
                tool_message_count = sum(
                    1 for msg in recent_messages if isinstance(msg, ToolMessage)
                )
                ai_message_count = sum(
                    1 for msg in recent_messages if isinstance(msg, AIMessage)
                )

                # Planning middleware: Analyze conversation completion signals

                # Signal 1: Tool results are available for processing
                if tool_message_count > 0:
                    # Check conversation length for natural stopping points
                    total_messages = len(messages)

                    # Signal 2: Prevent excessive cycling (length-based)
                    if total_messages >= 20:
                        logger.info(
                            f"🔀 Subgraph: Conversation too long ({total_messages} messages), finishing"
                        )
                        return "__end__"

                    # Signal 3: Too many AI responses in recent context
                    if ai_message_count >= 4:
                        logger.info(
                            f"🔀 Subgraph: Too many recent AI messages ({ai_message_count}), finishing"
                        )
                        return "__end__"

                    # Signal 4: Check if agent shows completion intent
                    last_ai_msg = None
                    for msg in reversed(messages):
                        if isinstance(msg, AIMessage):
                            last_ai_msg = msg
                            break

                    # If last AI message has no tool calls, agent likely completed its task
                    if last_ai_msg and (
                        not hasattr(last_ai_msg, "tool_calls")
                        or not last_ai_msg.tool_calls
                    ):
                        # Additional check: look for completion phrases in content
                        content = getattr(last_ai_msg, "content", "").lower()
                        completion_signals = [
                            "based on the search results",
                            "in summary",
                            "in conclusion",
                            "to summarize",
                            "here's what i found",
                            "the research shows",
                            "summary of",
                            "developments in",
                            "latest developments",
                            "can't provide more details",
                            "limited to the titles and snippets",
                            "for detailed insights",
                            "further targeted investigation",
                        ]

                        if any(signal in content for signal in completion_signals):
                            logger.info(
                                "🔀 Subgraph: Agent shows completion intent, finishing"
                            )
                            return "__end__"

                        # If content is substantial (agent provided comprehensive response)
                        if len(content) > 300:
                            logger.info(
                                "🔀 Subgraph: Agent provided comprehensive response, finishing"
                            )
                            return "__end__"

                    # Signal 5: Tool diversity check - if recent tools are repetitive, consider stopping
                    recent_tool_names = []
                    for msg in reversed(messages[-6:]):
                        if (
                            isinstance(msg, AIMessage)
                            and hasattr(msg, "tool_calls")
                            and msg.tool_calls
                        ):
                            for tc in msg.tool_calls:
                                recent_tool_names.append(tc.get("name", "unknown"))

                    if len(recent_tool_names) >= 3 and len(set(recent_tool_names)) == 1:
                        logger.info(
                            f"🔀 Subgraph: Repetitive tool usage detected ({recent_tool_names[0]}), finishing"
                        )
                        return "__end__"

                    logger.info(
                        f"🔀 Subgraph: Continuing agent loop - messages: {total_messages}, recent AI: {ai_message_count}, tool diversity: {len(set(recent_tool_names))}"
                    )
                    return "chat_agent"

                # Default to finishing if no clear continuation signal
                logger.info("🔀 Subgraph: No continuation signals, finishing")
                return "__end__"

            # Add conditional routing for intelligent agent cycling
            builder.add_conditional_edges(
                "chat_agent",
                should_execute_tools,
                {
                    "tools": "tool_executor",
                    "__end__": END,
                },
            )

            # Tool executor uses intelligent routing to decide next step
            builder.add_conditional_edges(
                "tool_executor",
                should_continue_agent_loop,
                {
                    "chat_agent": "chat_agent",  # Continue agent loop
                    "__end__": END,  # Finish subgraph
                },
            )

            # Start with chat agent
            builder.add_edge(START, "chat_agent")

            # Compile the graph with safeguards and middleware
            self.graph = builder.compile()
            logger.info(
                "Intelligent agent subgraph built with sophisticated routing and tool execution"
            )

        except Exception as e:
            logger.error(f"Failed to build agent subgraph: {e}")
            raise

    async def _chat_agent_wrapper(self, state: ToolsState) -> Dict[str, Any]:
        """Wrapper that creates ChatAgent at runtime and executes it."""
        try:
            # Extract user context from state
            user_id = state.get("user_id", "system")
            conversation_id = state.get("conversation_id", 0)

            # Create ChatAgent with runtime context
            chat_agent = self._create_chat_agent(user_id, conversation_id)

            # Convert LangChain core messages to our LangChainMessage format
            messages = state["messages"]
            from models import LangChainMessage

            langchain_messages = []
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    langchain_messages.append(
                        LangChainMessage(
                            content=msg.content,
                            type="human",
                            additional_kwargs=getattr(msg, "additional_kwargs", {}),
                            response_metadata=getattr(msg, "response_metadata", {}),
                        )
                    )
                elif isinstance(msg, AIMessage):
                    # Handle tool calls properly
                    tool_calls = None
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        tool_calls = [
                            {
                                "name": tc.get("name", ""),
                                "args": tc.get("args", {}),
                                "id": tc.get("id", ""),
                                "type": "tool_call",
                            }
                            for tc in msg.tool_calls
                        ]

                    langchain_messages.append(
                        LangChainMessage(
                            content=msg.content,
                            type="ai",
                            tool_calls=tool_calls,
                            additional_kwargs=getattr(msg, "additional_kwargs", {}),
                            response_metadata=getattr(msg, "response_metadata", {}),
                        )
                    )
                elif isinstance(msg, ToolMessage):
                    langchain_messages.append(
                        LangChainMessage(
                            content=msg.content,
                            type="tool",
                            id=msg.tool_call_id,
                            additional_kwargs=getattr(msg, "additional_kwargs", {}),
                            response_metadata=getattr(msg, "response_metadata", {}),
                        )
                    )
                else:
                    # Already in LangChainMessage format
                    langchain_messages.append(msg)

            # Get tools from registry for the agent
            executable_tools = self.tool_registry.get_all_executable_tools()
            tools_list = list(executable_tools.values()) if executable_tools else None

            # Execute chat completion with tools
            response_msg = await chat_agent.chat_completion_with_conversion(
                messages=langchain_messages, tools=tools_list
            )

            # Convert response back to LangChain core AIMessage format for LangGraph
            tool_calls = []
            if hasattr(response_msg, "tool_calls") and response_msg.tool_calls:
                for tc in response_msg.tool_calls:
                    # Use LangGraph's expected tool call format
                    tool_calls.append(
                        {
                            "name": tc.get("name", ""),
                            "args": tc.get("args", {}),
                            "id": tc.get("id", f"call_{len(tool_calls)}"),
                            "type": "tool_call",
                        }
                    )

            # Create AIMessage compatible with LangGraph ToolNode
            ai_message = AIMessage(
                content=response_msg.content or "",
                tool_calls=tool_calls,
                additional_kwargs=getattr(response_msg, "additional_kwargs", {}),
                response_metadata=getattr(response_msg, "response_metadata", {}),
            )

            # Return new message in state update format
            return {"messages": [ai_message]}

        except Exception as e:
            logger.error(f"Chat agent wrapper failed: {e}")
            import traceback

            traceback.print_exc()
            # Return error message
            error_msg = AIMessage(content=f"Agent error: {str(e)}")
            return {"messages": [error_msg]}

    # Removed _should_continue - using LangGraph's built-in tools_condition instead

    def transform_to_tools_state(self, main_state: WorkflowState) -> ToolsState:
        """Transform main WorkflowState to minimal ToolsState for agent subgraph."""
        # Get recent messages for agent context and convert to LangChain core messages
        recent_messages = getattr(main_state, "messages", [])[-10:]
        langchain_messages = []

        for msg in recent_messages:
            if hasattr(msg, "type") and hasattr(msg, "content"):
                # Convert custom LangChainMessage to proper LangChain core message
                if msg.type == "human":
                    langchain_messages.append(HumanMessage(content=msg.content))
                elif msg.type == "ai":
                    # Check if this AI message has tool calls and convert properly
                    tool_calls = []
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            if isinstance(tc, dict):
                                tool_calls.append(
                                    {
                                        "name": tc.get("name", ""),
                                        "args": tc.get("args", {}),
                                        "id": tc.get("id", "unknown"),
                                        "type": "tool_call",
                                    }
                                )
                            else:
                                # Handle other tool call formats
                                tool_calls.append(
                                    {
                                        "name": getattr(tc, "name", ""),
                                        "args": getattr(tc, "args", {}),
                                        "id": getattr(tc, "id", "unknown"),
                                        "type": "tool_call",
                                    }
                                )

                    langchain_messages.append(
                        AIMessage(
                            content=msg.content,
                            tool_calls=tool_calls if tool_calls else [],
                        )
                    )
                elif msg.type == "tool":
                    langchain_messages.append(
                        ToolMessage(
                            content=msg.content,
                            tool_call_id=getattr(msg, "id", None) or "unknown",
                        )
                    )
                else:
                    # Default to human message for unknown types
                    langchain_messages.append(HumanMessage(content=str(msg.content)))
            else:
                # Already a proper LangChain message, use as-is
                langchain_messages.append(msg)

        # Pass full user_config object for tool access (tools need full config objects)
        user_config = getattr(main_state, "user_config", None)

        return {
            "messages": langchain_messages,
            "user_id": getattr(main_state, "user_id", ""),
            "conversation_id": getattr(main_state, "conversation_id", 0),
            "user_config": user_config,
            "system_config": None,  # Not available in WorkflowState
            "current_date": getattr(main_state, "current_date", ""),
            "tool_call_count": 0,
        }

    def transform_to_main_state(
        self, agent_result: Dict[str, Any], main_state: WorkflowState
    ) -> Dict[str, Any]:
        """Transform agent subgraph results back to main WorkflowState updates."""
        from models import LangChainMessage

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
                    if isinstance(msg, (AIMessage, ToolMessage)):
                        # Convert to LangChainMessage format for main state
                        logger.info(
                            f"🔄 transform_to_main_state: Converting {type(msg).__name__} with type='{msg.type}' to LangChainMessage"
                        )
                        lang_chain_msg = LangChainMessage(
                            content=msg.content,
                            type=msg.type,
                            name=getattr(msg, "name", None),
                            id=getattr(msg, "id", None)
                            or getattr(msg, "tool_call_id", None),
                            additional_kwargs=getattr(msg, "additional_kwargs", {}),
                            response_metadata=getattr(msg, "response_metadata", {}),
                        )
                        logger.info(
                            f"🔄 transform_to_main_state: Created LangChainMessage with type='{lang_chain_msg.type}'"
                        )
                        new_messages.append(lang_chain_msg)

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

            # Execute the agent subgraph with higher recursion limit for intelligent cycling
            result = await self.graph.ainvoke(
                tools_state,
                config={
                    "recursion_limit": 20
                },  # Allow intelligent agent cycling with tools
            )

            # Transform results back to main state updates
            logger.info(
                f"🔄 ToolsAgentSubgraph: Calling transform_to_main_state with result containing {len(result.get('messages', []))} messages"
            )
            updates = self.transform_to_main_state(result, main_state)

            logger.info(
                f"🔄 ToolsAgentSubgraph: Agent subgraph completed with {len(updates)} state updates"
            )
            if "messages" in updates:
                logger.info(
                    f"🔄 ToolsAgentSubgraph: Returning {len(updates['messages']) - len(main_state.messages)} new messages"
                )
            return Command(update=updates)

        except Exception as e:
            logger.error(f"Agent subgraph execution failed: {e}", exc_info=True)
            return Command(update={})


class _LazyToolsAgentSubgraph:
    """Lazy initializer for tools agent subgraph with dependency injection."""

    def __init__(self):
        self._subgraph = None

    def _ensure_initialized(self):
        """Initialize the subgraph if not already done."""
        if self._subgraph is None:
            # Import here to avoid circular imports
            from runner.pipeline_factory import pipeline_factory
            from composer.tools.registry import ToolRegistry

            # Create registry - this should be improved to use proper DI in the future
            tool_registry = ToolRegistry(pipeline_factory)

            self._subgraph = ToolsAgentSubgraph(pipeline_factory, tool_registry)
        return self._subgraph

    async def execute(self, main_state: WorkflowState):
        """Execute the subgraph (lazy initialization)."""
        subgraph = self._ensure_initialized()
        return await subgraph.execute(main_state)


# Global instance for backward compatibility
tools_agent_subgraph = _LazyToolsAgentSubgraph()
