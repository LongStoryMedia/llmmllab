"""
Tools Agent Subgraph - Simple LangChain Agent Pattern

Following the exact pattern from LangChain documentation:
https://docs.langchain.com/oss/python/langgraph/workflows-agents#agents

Simple architecture:
1. chat_agent: LLM node that can call tools
2. tool_executor: ToolNode that executes tools
3. Built-in tools_condition for routing
4. No custom logic - let LangChain handle everything
"""

import json
from typing import Dict, Any, List

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain.tools import BaseTool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode  # Keep for type hints if needed
from langgraph.types import Command

from composer.graph.state import WorkflowState, ToolsState, assemble_context_messages
from composer.agents.chat_agent import ChatAgent
from composer.tools.registry import ToolRegistry
from utils.message_conversion import (
    messages_to_lc_messages,
    message_to_lc_message,
    lc_messages_to_messages,
    lc_message_to_message,
)
from utils.tool_call_types import (
    LangChainToolCall,
    extract_tool_call_requests,
    has_tool_calls,
)
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="ToolsAgentSubgraph")


class ToolsAgentSubgraph:
    """
    Simple agent subgraph following LangChain quickstart pattern.
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

        logger.info("ToolsAgentSubgraph initialized")

    def _create_tool_node(self):
        """
        Create custom ToolNode with proper state injection for ToolRuntime.

        LangChain's standard ToolNode doesn't pass the full state to ToolRuntime,
        causing tools like memory_retrieval to fail with "Missing user_id in state".
        This custom implementation ensures proper state injection.
        """

        class StateInjectedToolNode:
            """Custom ToolNode that properly injects full state into ToolRuntime."""

            def __init__(self, tool_registry: ToolRegistry):
                self.tool_registry = tool_registry

            async def __call__(self, state: ToolsState) -> ToolsState:
                """Execute tools with proper state injection."""

                if not state.messages:
                    return state

                last_message = state.messages[-1]

                # Check if last message has tool calls
                if not isinstance(last_message, AIMessage) or not has_tool_calls(
                    last_message
                ):
                    return state

                # Get executable tools
                executable_tools = self.tool_registry.get_all_executable_tools()
                if not executable_tools:
                    logger.warning("No executable tools available")
                    return state

                # Execute each tool call with proper state injection
                tool_messages = []
                if not last_message.tool_calls:
                    return state

                for tool_call in last_message.tool_calls:
                    # Handle both dictionary and object formats from OpenAI
                    if hasattr(tool_call, 'function'):
                        # OpenAI object format (ChatCompletionMessageFunctionToolCall)
                        tool_name = tool_call.function.name
                        tool_args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
                        tool_call_id = tool_call.id
                    elif isinstance(tool_call, dict):
                        # Dictionary format
                        tool_name = tool_call.get("name") or tool_call.get("function", {}).get("name")
                        tool_args = tool_call.get("args", {})
                        if not tool_args and "function" in tool_call:
                            args_str = tool_call["function"].get("arguments", "{}")
                            tool_args = json.loads(args_str) if args_str else {}
                        tool_call_id = tool_call.get("id")
                    else:
                        logger.warning(f"Unknown tool call format: {type(tool_call)}")
                        continue

                    if tool_name in executable_tools:
                        tool = executable_tools[tool_name]

                        # Create proper ToolRuntime instance
                        from langchain.tools import ToolRuntime
                        from langchain_core.runnables.config import RunnableConfig
                        
                        # Create minimal runtime - we mainly need state and tool_call_id
                        runtime = ToolRuntime(
                            state=state,  # ToolsState object
                            context={},  # Empty context for now
                            config=RunnableConfig(),  # Empty config 
                            stream_writer=None,  # Not needed for our tools
                            tool_call_id=tool_call_id,
                            store=None  # Not needed for our tools
                        )

                        try:
                            logger.info(
                                f"🔧 Executing tool '{tool_name}' with full state injection"
                            )

                            # Add timeout to tool execution to prevent hanging
                            import asyncio

                            try:
                                result = await asyncio.wait_for(
                                    tool.ainvoke({**tool_args, "runtime": runtime}),
                                    timeout=state.user_config.tool.tool_timeout,
                                )
                            except asyncio.TimeoutError:
                                logger.error(
                                    f"⏰ Tool '{tool_name}' execution timed out after {state.user_config.tool.tool_timeout} seconds"
                                )
                                result = f"❌ Tool execution timed out after {state.user_config.tool.tool_timeout} seconds"

                            tool_messages.append(
                                ToolMessage(
                                    content=str(result),
                                    tool_call_id=tool_call_id,
                                    name=tool_name,
                                )
                            )

                            logger.info(f"🔧 Tool '{tool_name}' executed successfully")

                        except Exception as e:
                            logger.error(
                                f"Tool '{tool_name}' execution failed: {e}",
                                exc_info=True,
                            )
                            tool_messages.append(
                                ToolMessage(
                                    content=f"❌ Tool execution failed: {str(e)}",
                                    tool_call_id=tool_call_id,
                                    name=tool_name,
                                )
                            )
                    else:
                        logger.warning(f"Tool '{tool_name}' not found in registry")
                        tool_messages.append(
                            ToolMessage(
                                content=f"❌ Tool '{tool_name}' not available",
                                tool_call_id=tool_call_id,
                                name=tool_name or "unknown",
                            )
                        )

                # Return updated state with tool messages
                state.messages.extend(tool_messages)
                return state

        try:
            logger.info("🛠️ Creating custom ToolNode with full state injection")
            return StateInjectedToolNode(self.tool_registry)

        except Exception as e:
            logger.error(f"Failed to create custom ToolNode: {e}")

            # Return a minimal fallback
            class EmptyToolNode:
                async def __call__(self, state: ToolsState) -> ToolsState:
                    return state

            return EmptyToolNode()

    def _build_graph(self) -> None:
        """Build simple agent following LangChain quickstart pattern."""
        try:
            # Simple StateGraph following LangChain docs exactly
            builder = StateGraph(ToolsState)

            # Add chat agent node
            builder.add_node("chat_agent", self._chat_agent_node)

            # Add tool executor node - must be named "tools" for tools_condition
            tool_node = self._create_tool_node()
            builder.add_node("tools", tool_node)

            # Custom routing condition for tool calls with safety limits
            def should_continue_to_tools(state: ToolsState) -> str:
                """Check if we should route to tools or end with safety limits."""
                if not state.messages:
                    return "__end__"

                # Count total interactions to prevent infinite loops
                total_messages = len(state.messages)
                max_messages = 50  # Safety limit
                if total_messages > max_messages:
                    logger.warning(
                        f"🛑 Stopping: reached message limit ({total_messages}/{max_messages})"
                    )
                    return "__end__"

                # Count tool call iterations in recent messages
                recent_messages = state.messages[-20:]  # Look at last 20 messages
                tool_call_count = sum(
                    1
                    for msg in recent_messages
                    if hasattr(msg, "tool_calls") and getattr(msg, "tool_calls", None)
                )
                max_tool_iterations = 10  # Safety limit
                if tool_call_count > max_tool_iterations:
                    logger.warning(
                        f"🛑 Stopping: reached tool call limit ({tool_call_count}/{max_tool_iterations})"
                    )
                    return "__end__"

                last_message = state.messages[-1]
                # Check if message has tool calls
                if hasattr(last_message, "tool_calls") and getattr(
                    last_message, "tool_calls", None
                ):
                    tool_calls = getattr(last_message, "tool_calls", [])
                    logger.info(
                        f"🔧 Routing to tools: {len(tool_calls)} tool calls to execute"
                    )
                    return "tools"

                logger.info("✅ No tool calls found, ending workflow")
                return "__end__"

            builder.add_conditional_edges(
                "chat_agent",
                should_continue_to_tools,
                {
                    "tools": "tools",
                    "__end__": END,
                },
            )

            # Simple continuation after tools
            builder.add_edge("tools", "chat_agent")

            # Start with chat agent
            builder.add_edge(START, "chat_agent")

            # Compile with reasonable recursion limit
            self.graph = builder.compile()

            logger.info("Simple tools agent subgraph built following LangChain pattern")

        except Exception as e:
            logger.error(f"Failed to build agent subgraph: {e}")
            raise

    def _extract_tool_call_requests_from_message(
        self, msg: BaseMessage
    ) -> List[LangChainToolCall]:
        """
        Extract tool call requests from a message with strong typing.

        Returns:
            List of LangChain tool call requests (what AI wants to call)
        """
        return extract_tool_call_requests(msg)

    def _extract_previous_tool_call_requests(
        self, messages: List[BaseMessage]
    ) -> List[LangChainToolCall]:
        """Extract all previous tool call requests from conversation history."""
        previous_requests = []
        for msg in messages:
            tool_call_requests = self._extract_tool_call_requests_from_message(msg)
            previous_requests.extend(tool_call_requests)

        logger.debug(
            f"Extracted {len(previous_requests)} previous tool call requests: {[req['name'] for req in previous_requests]}"
        )
        return previous_requests

    def _is_duplicate_tool_call_request(
        self,
        current_request: LangChainToolCall,
        previous_requests: List[LangChainToolCall],
    ) -> bool:
        """
        Check if a tool call request is a duplicate of a previous one.

        Only considers exact duplicates (same tool name AND same arguments).
        Different arguments to the same tool are allowed for legitimate use cases like:
        - Multiple web searches with different queries
        - Reading multiple URLs with read_web_content
        - Multiple API calls with different parameters
        """
        duplicate_count = 0
        for prev_request in previous_requests:
            if (
                prev_request["name"] == current_request["name"]
                and prev_request["args"] == current_request["args"]
            ):
                duplicate_count += 1

        # Allow 1 duplicate (so 2 total calls with same args), block after that
        # This handles cases where the AI might legitimately retry a failed call
        return duplicate_count >= 2

    def _optimize_vision_content(self, messages: List) -> List:
        """Simple pass-through - vision optimization disabled for now."""
        # TODO: Implement proper vision optimization that prevents processing at pipeline level
        # Current approach was causing more issues than it solved
        return messages

    async def _chat_agent_node(self, state: ToolsState) -> ToolsState:
        """Simple LangChain agent node."""
        try:
            # Get available tools
            tools_dict = self.tool_registry.get_all_executable_tools()
            tools_list = list(tools_dict.values()) if tools_dict else None

            # Use the ChatAgent's chat completion method
            response = await self.chat_agent.chat_completion(
                messages=lc_messages_to_messages(state.messages),
                tools=tools_list,
                stream=False,
            )

            # Convert response message to LangChain BaseMessage format
            if response and response.message:
                state.messages.append(message_to_lc_message(response.message))

            # Return updated state following LangChain agent pattern
            return state

        except Exception as e:
            logger.error(f"Error in chat agent node: {e}")
            # Fallback: return state unchanged
            return state

    # Removed _should_continue - using LangGraph's built-in tools_condition instead

    def transform_to_tools_state(self, main_state: WorkflowState) -> ToolsState:
        """Transform main WorkflowState to minimal ToolsState for agent subgraph."""
        # Get recent messages for agent context and convert to LangChain core messages
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
                    if isinstance(msg, (AIMessage, ToolMessage)):
                        # Convert to Message format for main state
                        logger.info(
                            f"🔄 transform_to_main_state: Converting {type(msg).__name__} to Message"
                        )
                        # Convert BaseMessage to Message using existing utilities
                        messages_list = lc_messages_to_messages([msg])
                        if messages_list:
                            message_obj = messages_list[0]
                            message_obj.conversation_id = getattr(
                                main_state, "conversation_id", None
                            )
                            logger.info(
                                f"🔄 transform_to_main_state: Created Message with role='{message_obj.role}'"
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

            # Execute the agent subgraph with timeout and iteration limit
            import asyncio

            # Set timeout for graph execution (5 minutes max)
            timeout_seconds = 300

            try:
                result = await asyncio.wait_for(
                    self.graph.ainvoke(tools_state), timeout=timeout_seconds
                )
            except asyncio.TimeoutError:
                logger.error(
                    f"❌ Agent subgraph execution timed out after {timeout_seconds} seconds"
                )
                return Command(update={})

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
