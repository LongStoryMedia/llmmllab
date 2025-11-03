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
from langgraph.prebuilt import ToolNode  # Keep for type hints if needed
from langgraph.types import Command

from models import LangChainMessage
from composer.graph.state import WorkflowState, ToolsState
from composer.agents.chat_agent import ChatAgent
from composer.tools.registry import ToolRegistry
from composer.utils.conversion import (
    convert_base_langchain_to_messages,
    convert_messages_to_langchain,
    to_lc_message,
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
            
            def __init__(self, tool_registry):
                self.tool_registry = tool_registry
                
            async def __call__(self, state: ToolsState) -> ToolsState:
                """Execute tools with proper state injection."""
                messages = state.get("messages", [])
                
                if not messages:
                    return state
                
                last_message = messages[-1]
                
                # Check if last message has tool calls
                if not (hasattr(last_message, "tool_calls") and last_message.tool_calls):
                    return state
                
                # Get executable tools
                executable_tools = self.tool_registry.get_all_executable_tools()
                if not executable_tools:
                    logger.warning("No executable tools available")
                    return state
                
                # Execute each tool call with proper state injection
                tool_messages = []
                
                for tool_call in last_message.tool_calls:
                    tool_name = tool_call.get("name")
                    tool_args = tool_call.get("args", {})
                    tool_call_id = tool_call.get("id", "unknown")
                    
                    if tool_name in executable_tools:
                        tool = executable_tools[tool_name]
                        
                        # Create ToolRuntime with full state - this is the key fix!
                        class ToolRuntimeImpl:
                            def __init__(self, state_dict, call_id):
                                self.state = state_dict  # Full ToolsState with user_id, etc.
                                self.tool_call_id = call_id
                        
                        runtime = ToolRuntimeImpl(state, tool_call_id)
                        
                        try:
                            logger.info(f"🔧 Executing tool '{tool_name}' with full state injection")
                            logger.debug(f"Tool state includes: {list(state.keys())}")
                            logger.debug(f"User ID in tool state: {state.get('user_id')}")
                            
                            # Call tool with runtime injection
                            if hasattr(tool, '_arun'):
                                result = await tool._arun(runtime=runtime, **tool_args)
                            elif hasattr(tool, 'ainvoke'):
                                # For newer LangChain tools
                                result = await tool.ainvoke({**tool_args, "runtime": runtime})
                            else:
                                # Fallback synchronous execution
                                result = tool._run(runtime=runtime, **tool_args)
                            
                            tool_messages.append(ToolMessage(
                                content=str(result),
                                tool_call_id=tool_call_id,
                                name=tool_name
                            ))
                            
                            logger.info(f"🔧 Tool '{tool_name}' executed successfully")
                            
                        except Exception as e:
                            logger.error(f"Tool '{tool_name}' execution failed: {e}", exc_info=True)
                            tool_messages.append(ToolMessage(
                                content=f"❌ Tool execution failed: {str(e)}",
                                tool_call_id=tool_call_id,
                                name=tool_name
                            ))
                    else:
                        logger.warning(f"Tool '{tool_name}' not found in registry")
                        tool_messages.append(ToolMessage(
                            content=f"❌ Tool '{tool_name}' not available",
                            tool_call_id=tool_call_id,
                            name=tool_name or "unknown"
                        ))
                
                # Return updated state with tool messages
                updated_messages = messages + tool_messages
                return {**state, "messages": updated_messages}
        
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

            # Custom routing condition for tool calls (since we use custom ToolNode)
            def should_continue_to_tools(state: ToolsState) -> str:
                """Check if we should route to tools or end."""
                messages = state.get("messages", [])
                if not messages:
                    return "__end__"
                
                last_message = messages[-1]
                # Check if message has tool calls using hasattr for safety
                if hasattr(last_message, "tool_calls") and getattr(last_message, "tool_calls", None):
                    return "tools"
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
        from langchain_core.messages import AIMessage

        # Get messages from state
        messages = state["messages"]

        try:
            # Get available tools
            tools_dict = self.tool_registry.get_all_executable_tools()
            tools_list = list(tools_dict.values()) if tools_dict else None

            # Convert LangChain BaseMessages to our internal Message format first
            internal_messages = convert_base_langchain_to_messages(messages)
            # Then convert to LangChainMessage format that chat_completion expects
            langchain_format_messages = convert_messages_to_langchain(internal_messages)

            # Use the ChatAgent's chat completion method
            response = await self.chat_agent.chat_completion(
                messages=langchain_format_messages, tools=tools_list, stream=False
            )

            # Convert response message to LangChain BaseMessage format
            if response and response.message:
                langchain_response = to_lc_message(response.message)
            else:
                langchain_response = AIMessage(content="No response generated")

            # Ensure all messages are BaseMessage instances
            updated_messages = list(messages) + [langchain_response]

            # Return updated state following LangChain agent pattern
            return {**state, "messages": updated_messages}

        except Exception as e:
            logger.error(f"Error in chat agent node: {e}")
            # Fallback: return state unchanged
            return state

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

        # Get user_id with validation
        user_id = getattr(main_state, "user_id", None)
        if not user_id:
            logger.warning(
                f"WorkflowState missing user_id - this will cause tool failures. "
                f"user_id={user_id}, conversation_id={getattr(main_state, 'conversation_id', 'missing')}"
            )

        return {
            "messages": langchain_messages,
            "user_id": user_id
            or "",  # Still use empty string for backward compatibility, but log the issue
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
