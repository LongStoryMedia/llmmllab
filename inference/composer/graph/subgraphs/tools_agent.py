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
import asyncio
from typing import Dict, Any, List

from langchain_core.messages import AIMessage, ToolMessage, BaseMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from composer.graph.state import WorkflowState, ToolsState, assemble_context_messages
from composer.agents.chat_agent import ChatAgent
from composer.tools.registry import ToolRegistry
from utils.message_conversion import (
    messages_to_lc_messages,
    message_to_lc_message,
    lc_messages_to_messages,
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
        Create standard LangChain ToolNode with automatic ToolRuntime injection.
        
        LangChain's standard ToolNode automatically injects ToolRuntime into tools
        that have runtime parameters, which is the correct pattern according to docs.
        """
        try:
            # Get executable tools from registry
            executable_tools = self.tool_registry.get_all_executable_tools()
            if not executable_tools:
                logger.warning("No executable tools available for ToolNode")
                
                # Return minimal fallback node
                class EmptyToolNode:
                    async def __call__(self, state):
                        return state
                return EmptyToolNode()
            
            # Create list of tools for ToolNode
            tools_list = list(executable_tools.values())
            
            logger.info(f"�️ Creating standard LangChain ToolNode with {len(tools_list)} tools")
            
            # Use LangChain's standard ToolNode which handles ToolRuntime injection automatically
            return ToolNode(tools_list)
            
        except Exception as e:
            logger.error(f"Failed to create ToolNode: {e}")
            
            # Return minimal fallback
            class ToolNodeFallback:
                async def __call__(self, state):
                    return state
            return ToolNodeFallback()

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
                logger.info(f"🔍 State has {len(state.messages)} messages, checking last message (idx {len(state.messages)-1})")
                # Check if message has tool calls using our extraction utility
                from utils.tool_call_extraction import extract_tool_calls_from_langchain_message
                tool_calls = extract_tool_calls_from_langchain_message(last_message)
                
                # Debug logging
                logger.debug(f"🔍 Routing debug: message type={type(last_message)}, content preview={str(last_message.content)[:100] if hasattr(last_message, 'content') else 'no content'}")
                logger.debug(f"🔍 Routing debug: extracted {len(tool_calls)} tool calls")
                logger.debug(f"🔍 Routing debug: tool_calls list = {tool_calls}")
                logger.debug(f"🔍 Routing debug: bool(tool_calls) = {bool(tool_calls)}")
                for i, tc in enumerate(tool_calls):
                    logger.debug(f"🔍 Tool call {i}: name={getattr(tc, 'name', 'unknown')}, success={getattr(tc, 'success', 'unknown')}")
                
                if tool_calls:
                    logger.info(
                        f"🔧 Routing to tools: {len(tool_calls)} tool calls to execute"
                    )
                    return "tools"
                else:
                    logger.warning(f"❌ Tool calls list failed boolean check: len={len(tool_calls)}, type={type(tool_calls)}, repr={repr(tool_calls)}")
                    
                    # DEBUG: Check the raw message attributes for debugging
                    if hasattr(last_message, "additional_kwargs"):
                        logger.info(f"� Debug additional_kwargs: {last_message.additional_kwargs}")
                    if hasattr(last_message, "tool_calls") and isinstance(last_message, AIMessage):
                        logger.info(f"🔍 Debug tool_calls attribute: {last_message.tool_calls}")
                    if hasattr(last_message, "response_metadata"):
                        logger.info(f"🔍 Debug response_metadata: {last_message.response_metadata}")
                    
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
        """Enhanced chat agent node with LangChain ChatOpenAI integration."""
        try:
            # Get available tools
            tools_dict = self.tool_registry.get_all_executable_tools()
            tools_list = list(tools_dict.values()) if tools_dict else None

            # Check if we have tool results in conversation that should be used for final answer
            has_tool_results = any(
                hasattr(msg, 'type') and getattr(msg, 'type', '') == 'tool'
                for msg in state.messages[-10:]  # Check recent messages
            )
            
            # Count recent failed tool calls to detect when model should stop calling tools
            recent_tool_errors = sum(
                1 for msg in state.messages[-5:]
                if hasattr(msg, 'content') and 'Error:' in str(getattr(msg, 'content', ''))
            )
            
            # Modify messages to include explicit instruction if model should provide final answer
            messages_for_completion = state.messages.copy()
            if has_tool_results and recent_tool_errors >= 2:
                # Add instruction to use existing tool results instead of calling more tools
                from langchain_core.messages import SystemMessage
                final_instruction = SystemMessage(
                    content=(
                        "IMPORTANT: You have already gathered information from tools. "
                        "Some recent tool calls failed, but you have previous successful tool results. "
                        "Based on the information you already have from successful tool calls, "
                        "provide a direct, helpful answer to the user's question. "
                        "DO NOT make any more tool calls. Use the information you already gathered."
                    )
                )
                messages_for_completion.append(final_instruction)

            # Ensure chat agent has a pipeline by calling ensure_pipeline_created first
            await self.chat_agent.ensure_pipeline_created()
            
            # Now check for ChatOpenAI capability
            pipeline = self.chat_agent.current_pipeline
            logger.info(f"🔍 ChatOpenAI check: pipeline={pipeline}, type={type(pipeline)}")
            logger.info(f"🔍 ChatOpenAI check: has_get_chat_model={hasattr(pipeline, 'get_chat_model') if pipeline else 'No pipeline'}")
            
            if pipeline and hasattr(pipeline, 'get_chat_model'):
                try:
                    logger.info("🚀 Using direct ChatOpenAI approach for tool calling")
                    chat_model = pipeline.get_chat_model()  # type: ignore
                    
                    # Bind tools to ChatOpenAI if available
                    if tools_list:
                        logger.info(f"🔧 Binding {len(tools_list)} tools to ChatOpenAI")
                        chat_model = chat_model.bind_tools(tools_list)
                    
                    # Invoke ChatOpenAI directly - this handles tool calling natively
                    logger.info("📤 About to invoke ChatOpenAI with tool calling...")
                    logger.info(f"📤 Input messages count: {len(messages_for_completion)}")
                    for i, msg in enumerate(messages_for_completion):
                        logger.info(f"📤 Input message {i}: type={type(msg).__name__}, content_preview={str(getattr(msg, 'content', ''))[:100]}")
                    
                    response_message = await chat_model.ainvoke(messages_for_completion)
                    
                    # COMPREHENSIVE response debugging
                    logger.info("📨 ChatOpenAI Response Analysis:")
                    logger.info(f"📨 Response type: {type(response_message)}")
                    logger.info(f"📨 Response dir: {[attr for attr in dir(response_message) if not attr.startswith('_')]}")
                    
                    # Check all possible attributes
                    for attr in ['content', 'tool_calls', 'additional_kwargs', 'response_metadata', 'usage_metadata']:
                        if hasattr(response_message, attr):
                            value = getattr(response_message, attr)
                            logger.info(f"📨 Response.{attr}: {value}")
                        else:
                            logger.info(f"📨 Response.{attr}: NOT PRESENT")
                    
                    # Special focus on tool_calls
                    if hasattr(response_message, 'tool_calls'):
                        tool_calls = getattr(response_message, 'tool_calls')
                        logger.info(f"📨 tool_calls type: {type(tool_calls)}")
                        logger.info(f"📨 tool_calls length: {len(tool_calls) if tool_calls else 'None'}")
                        if tool_calls:
                            for i, tc in enumerate(tool_calls):
                                logger.info(f"📨 tool_call {i}: type={type(tc)}, value={tc}")
                    
                    # Check for any function calling indicators in content
                    content = getattr(response_message, 'content', '')
                    if content and ('function' in content.lower() or 'tool' in content.lower() or '{' in content):
                        logger.info(f"📨 Content might contain function calls: {content}")
                    
                    logger.info(f"📨 Full response object: {response_message}")
                    
                    # Add response to state
                    state.messages.append(response_message)
                    
                    # Inject the agent's pipeline into state for tools to reuse
                    if self.chat_agent.current_pipeline and not state.shared_pipeline:
                        state.shared_pipeline = self.chat_agent.current_pipeline
                        logger.debug("💾 Injected shared pipeline into state for tool reuse")
                    
                    logger.info("✅ ChatOpenAI tool calling successful")
                    return state
                
                except Exception as e:
                    logger.warning(f"ChatOpenAI direct approach failed: {e}, falling back to traditional method")

            # Fallback to traditional chat completion approach
            logger.info("🔄 Using traditional chat completion approach")
            response = await self.chat_agent.chat_completion(
                messages=lc_messages_to_messages(messages_for_completion),
                tools=tools_list,
                stream=False,
            )

            # Convert response message to LangChain BaseMessage format
            if response and response.message:
                state.messages.append(message_to_lc_message(response.message))

            # Inject the agent's pipeline into state for tools to reuse
            if self.chat_agent.current_pipeline and not state.shared_pipeline:
                state.shared_pipeline = self.chat_agent.current_pipeline
                logger.debug("💾 Injected shared pipeline into state for tool reuse")

            # Return updated state following LangChain agent pattern
            return state

        except Exception as e:
            logger.error(f"Error in chat agent node: {e}")
                        # DEBUG: Add detailed pipeline creation logging
            import traceback

            call_stack = traceback.extract_stack()
            call_info = " → ".join(
                [f"{frame.filename}:{frame.lineno}\n" for frame in call_stack]
            )

            logger.error(f"Pipeline creation call stack: {call_info}")
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
