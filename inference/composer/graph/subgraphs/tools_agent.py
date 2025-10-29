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

from typing import Dict, Any, List

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain.tools import BaseTool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command

from models import LangChainMessage, NodeMetadata
from composer.graph.state import WorkflowState, ToolsState
from composer.agents.chat_agent import ChatAgent
from composer.tools.registry import ToolRegistry
from composer.utils.conversion import (
    convert_base_langchain_to_messages,
    convert_messages_to_langchain,
)
from composer.utils.tool_call_types import (
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
        """Initialize subgraph with dependency injection."""
        self.tool_registry = tool_registry
        self.chat_agent = chat_agent
        self.graph = None
        self._build_graph()

    def _create_tool_node(self) -> ToolNode:
        """
        Create LangGraph ToolNode with proper tools list and ToolRuntime injection.

        LangChain will automatically inject ToolRuntime for tools with `runtime: ToolRuntime` parameter.
        This is the correct pattern - no manual ToolRuntime creation needed.
        """
        try:
            # Get executable tools from registry
            executable_tools = self.tool_registry.get_all_executable_tools()
            tools_dict: dict[str, BaseTool] = (
                executable_tools if executable_tools else {}
            )

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
        """Build simple agent following LangChain quickstart pattern."""
        try:
            # Simple StateGraph following LangChain docs exactly
            builder = StateGraph(ToolsState)

            # Add chat agent node
            builder.add_node("chat_agent", self._chat_agent_wrapper)

            # Add tool executor node - must be named "tools" for tools_condition
            tool_node = self._create_tool_node()
            builder.add_node("tools", tool_node)

            # EXACTLY like the LangChain quickstart - use built-in tools_condition
            builder.add_conditional_edges(
                "chat_agent",
                tools_condition,  # Use built-in routing - expects "tools" node
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
        self, msg: BaseMessage | LangChainMessage
    ) -> List[LangChainToolCall]:
        """
        Extract tool call requests from a message with strong typing.

        Returns:
            List of LangChain tool call requests (what AI wants to call)
        """
        if isinstance(msg, BaseMessage):
            return extract_tool_call_requests(msg)

        # Handle our LangChainMessage format
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            validated_calls = []
            for tc in msg.tool_calls:
                if isinstance(tc, dict) and "name" in tc and "args" in tc:
                    validated_calls.append(
                        LangChainToolCall(
                            name=tc["name"], args=tc["args"], id=tc.get("id")
                        )
                    )
            return validated_calls

        return []

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
        """Check if a tool call request is a duplicate of a previous one."""
        for prev_request in previous_requests:
            if (
                prev_request["name"] == current_request["name"]
                and prev_request["args"] == current_request["args"]
            ):
                return True
        return False

    async def _chat_agent_wrapper(self, state: ToolsState) -> Dict[str, Any]:
        """
        Chat agent wrapper that properly manages conversation state.

        CRITICAL: The agent must see the full conversation history including
        tool results to make proper decisions about whether to continue with
        more tools or provide a final answer.

        ANTI-RECURSION: Prevents duplicate tool calls to avoid infinite loops.
        """
        try:
            # Get all messages from state - this includes conversation history + tool results
            messages = state["messages"]

            # Log conversation state for debugging
            logger.debug(f"Agent wrapper processing {len(messages)} messages")
            for i, msg in enumerate(messages[-3:]):  # Log last 3 messages
                msg_type = getattr(msg, "type", type(msg).__name__)
                msg_has_tool_calls = (
                    has_tool_calls(msg) if isinstance(msg, BaseMessage) else False
                )
                logger.debug(
                    f"  Message {len(messages)-3+i}: {msg_type}, tool_calls={msg_has_tool_calls}"
                )

            # Extract previous tool call requests to prevent duplicates
            previous_tool_requests = self._extract_previous_tool_call_requests(messages)
            logger.debug(
                f"Found {len(previous_tool_requests)} previous tool call requests in conversation"
            )

            # Convert to our format while preserving ALL conversation history
            langchain_messages = convert_messages_to_langchain(
                convert_base_langchain_to_messages(messages)
            )

            # Get tools from registry for the agent
            executable_tools = self.tool_registry.get_all_executable_tools()
            tools_list = list(executable_tools.values()) if executable_tools else None

            # Execute chat completion with full conversation history and tools
            # The agent will see: [user_message, previous_ai_message, tool_results, ...]
            response_msg = await self.chat_agent.chat_completion_with_conversion(
                messages=langchain_messages,  # Full conversation including tool results
                tools=tools_list,
            )

            # Convert response back to LangChain core AIMessage format for LangGraph
            tool_call_requests = []
            filtered_requests = []
            duplicates_blocked = 0
            response_tool_requests = self._extract_tool_call_requests_from_message(
                response_msg
            )
            total_tool_requests = len(response_tool_requests)

            if response_tool_requests:
                for request in response_tool_requests:
                    # Check if this is a duplicate tool call request
                    if self._is_duplicate_tool_call_request(
                        request, previous_tool_requests
                    ):
                        logger.warning(
                            f"🔄 BLOCKED duplicate tool call request: {request['name']} with args {request['args']}"
                        )
                        duplicates_blocked += 1
                        continue

                    # Keep the validated request
                    tool_call_requests.append(request)
                    filtered_requests.append(request)

            # Critical fix: If we have many tool requests but blocked some duplicates OR
            # if we have excessive tool usage (indicating potential loop), force final answer
            excessive_tool_usage = (
                len(previous_tool_requests) > 8
            )  # More than 8 previous tool requests indicates a loop
            should_force_final_answer = (
                duplicates_blocked > 0
                or excessive_tool_usage
                or (
                    total_tool_requests > 0 and len(tool_call_requests) == 0
                )  # All tool requests were blocked
            )

            if should_force_final_answer:
                reason = []
                if duplicates_blocked > 0:
                    reason.append(
                        f"blocked {duplicates_blocked} duplicate tool requests"
                    )
                if excessive_tool_usage:
                    reason.append(
                        f"detected excessive tool usage ({len(previous_tool_requests)} previous requests)"
                    )
                if total_tool_requests > 0 and len(tool_call_requests) == 0:
                    reason.append("all tool requests were blocked as duplicates")

                logger.info(f"🛡️ Forcing final answer - {', '.join(reason)}")

                final_content = (
                    response_msg.content
                    or "Based on the information I have already gathered from previous tool calls, I can provide you with a comprehensive response."
                )
                # Create final answer message with no tool calls - this will cause tools_condition to route to END
                ai_message = AIMessage(
                    content=final_content,
                    tool_calls=[],  # No tool calls - force final answer and END routing
                    additional_kwargs=getattr(response_msg, "additional_kwargs", {}),
                    response_metadata=getattr(response_msg, "response_metadata", {}),
                )
            else:
                # Create AIMessage compatible with LangGraph ToolNode
                ai_message = AIMessage(
                    content=response_msg.content or "",
                    tool_calls=tool_call_requests,  # Use validated requests
                    additional_kwargs=getattr(response_msg, "additional_kwargs", {}),
                    response_metadata=getattr(response_msg, "response_metadata", {}),
                )

            # Log what the agent decided to do
            if tool_call_requests:
                logger.info(
                    f"Agent decided to make {len(tool_call_requests)} tool call requests: {[req['name'] for req in tool_call_requests]}"
                )
            else:
                logger.info(
                    "Agent decided to provide final answer - no more tool calls"
                )

            # Return ONLY the new AI message - LangGraph will append it to conversation
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

            # Execute the agent subgraph with LangChain defaults
            result = await self.graph.ainvoke(tools_state)

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
