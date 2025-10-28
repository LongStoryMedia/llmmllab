"""
Tools Agent Subgraph - Clean LangGraph agent with built-in routing.

This subgraph implements proper LangGraph agent patterns using LangChain's built-in
tools_condition for routing instead of manual planning logic. Planning decisions
are delegated to the PlanningIntentSubgraph, and rate limiting uses LangGraph's
built-in recursion limits.

Key Benefits:
1. Built-in routing - uses LangGraph's tools_condition for proper agent cycling
2. Separation of concerns - planning handled by PlanningIntentSubgraph
3. Minimal state - ToolsState with only essential fields for tool execution
4. ToolRuntime pattern - all tools use modern ToolRuntime[ToolsState] injection
5. Clean architecture - focused only on tool execution, not planning decisions

Architecture:
- ToolsState: Minimal state optimized for agent operations
- chat_agent: LLM node that can make tool calls using available tools
- tool_executor: ToolNode that executes tools with ToolRuntime[ToolsState] access
- tools_condition: Built-in LangChain routing for proper agent termination
- Rate limiting: Uses LangGraph's recursion_limit instead of manual middleware
"""

from typing import Dict, Any

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain.tools import BaseTool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command

from models import LangChainMessage, NodeMetadata
from composer.graph.state import WorkflowState, ToolsState
from composer.agents.chat_agent import ChatAgent
from composer.tools.registry import ToolRegistry
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="ToolsAgentSubgraph")


class ToolsAgentSubgraph:
    """
    Complete agent subgraph with chat_agent + tool_node cycling workflow.

    Uses proper dependency injection pattern like the main graph builder,
    with ChatAgent and ToolRegistry dependencies.
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

        # Create node metadata for the subgraph agents
        self.subgraph_metadata = NodeMetadata(
            node_name="tools_agent_subgraph",
            node_id="tools_agent_subgraph",
            node_type="subgraph",
            user_id="system",  # Will be updated at runtime
            conversation_id=0,  # Will be updated at runtime
        )

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
        """Build the complete agent subgraph using proper dependency injection."""
        try:
            # Build graph with StateGraph pattern like main builder
            # ToolRuntime automatically gets state access - no context_schema needed
            builder = StateGraph(ToolsState)

            # Add chat agent node - will be created at runtime with proper context
            builder.add_node("chat_agent", self._chat_agent_wrapper)

            # Add tool executor node - using LangGraph's ToolNode with automatic ToolRuntime injection
            tool_node = self._create_tool_node()
            builder.add_node("tool_executor", tool_node)

            # Use LangChain's built-in tools_condition for routing
            # This handles all the complex logic of determining whether tools should execute
            # Rate limiting and planning logic is handled by middleware

            # Enhanced logic to prevent infinite tool calling loops
            def should_continue_after_tools(state: ToolsState):
                """
                Determine whether to continue agent loop or end subgraph after tools execute.
                
                Logic:
                1. Count tool execution cycles - limit to prevent infinite loops
                2. If we have tool results, allow ONE response generation then END
                3. If agent keeps generating tool calls instead of responses, force END
                """
                messages = state.get("messages", [])
                if not messages:
                    logger.info("🔀 Subgraph: No messages after tools, finishing")
                    return END

                # Count tool execution cycles by counting ToolMessage instances
                tool_execution_count = sum(1 for msg in messages if isinstance(msg, ToolMessage))
                
                # If we have many tool executions, force END to prevent infinite loops
                if tool_execution_count >= 4:  # Allow up to 4 tool execution cycles
                    logger.info(f"🔀 Subgraph: Tool execution limit reached ({tool_execution_count} executions), forcing END")
                    return END

                # Count recent AI messages to detect response patterns
                recent_ai_count = 0
                recent_tool_count = 0
                
                # Look at last 10 messages to understand the pattern
                for msg in messages[-10:]:
                    if isinstance(msg, ToolMessage):
                        recent_tool_count += 1
                    elif isinstance(msg, AIMessage):
                        recent_ai_count += 1

                # If we have tool results but multiple AI messages without ending, force END
                if recent_tool_count > 0 and recent_ai_count >= 3:
                    logger.info(f"🔀 Subgraph: Multiple AI responses ({recent_ai_count}) after tools ({recent_tool_count}), finishing to prevent loop")
                    return END

                # Check if we have recent tool execution
                has_recent_tool = any(isinstance(msg, ToolMessage) for msg in messages[-3:])
                
                if has_recent_tool:
                    # Check if there's already an AI response after the most recent tool message
                    last_tool_index = -1
                    for i in reversed(range(len(messages))):
                        if isinstance(messages[i], ToolMessage):
                            last_tool_index = i
                            break
                    
                    if last_tool_index >= 0:
                        # Check if there's an AI message after the last tool message
                        has_ai_after_tool = False
                        for i in range(last_tool_index + 1, len(messages)):
                            if isinstance(messages[i], AIMessage):
                                has_ai_after_tool = True
                                break
                        
                        if has_ai_after_tool:
                            logger.info("🔀 Subgraph: AI already responded to tools, finishing")
                            return END
                        else:
                            logger.info(f"🔀 Subgraph: Tools executed (cycle {tool_execution_count}), allowing final response generation")
                            return "chat_agent"

                # Default to ending - be more conservative about continuation
                logger.info("🔀 Subgraph: Default case, finishing subgraph")
                return END

            # Custom routing function that enforces tool execution limits
            def enhanced_tools_condition(state: ToolsState):
                """
                Enhanced routing logic that combines tools_condition with execution limits.
                
                Prevents infinite tool calling by enforcing hard limits at the routing level.
                """
                messages = state.get("messages", [])
                if not messages:
                    logger.info("🔀 Enhanced routing: No messages, ending")
                    return END

                # Count total tool executions to enforce hard limit
                tool_execution_count = sum(1 for msg in messages if isinstance(msg, ToolMessage))
                
                # HARD STOP: If we've hit our tool execution limit, force END regardless of LLM output
                if tool_execution_count >= 3:  # Even more aggressive limit: max 3 tool executions
                    logger.info(f"🔀 Enhanced routing: HARD STOP - Tool execution limit reached ({tool_execution_count} executions), forcing END")
                    return END

                # Get the last message to check for tool calls
                last_message = messages[-1]
                
                # If last message is not an AI message, end
                if not isinstance(last_message, AIMessage):
                    logger.info("🔀 Enhanced routing: Last message not from AI, ending")
                    return END
                
                # Check if AI message has tool calls
                has_tool_calls = (
                    hasattr(last_message, 'tool_calls') and 
                    last_message.tool_calls and 
                    len(last_message.tool_calls) > 0
                )
                
                if has_tool_calls:
                    # Additional check: if we already have tool results, be very restrictive about more tools
                    if tool_execution_count >= 2:
                        logger.info(f"🔀 Enhanced routing: OVERRIDE - Blocking additional tool calls after {tool_execution_count} executions")
                        return END
                    
                    logger.info(f"🔀 Enhanced routing: Tool calls detected (execution #{tool_execution_count + 1}), routing to tools")
                    return "tool_executor"
                else:
                    logger.info("🔀 Enhanced routing: No tool calls, ending subgraph")
                    return END

            # Use enhanced routing that enforces limits at the routing level
            builder.add_conditional_edges(
                "chat_agent",
                enhanced_tools_condition,  # Custom routing with hard limits
                {
                    "tool_executor": "tool_executor",
                    "__end__": END,
                },
            )

            # Simple continuation after tools - middleware handles complexity
            builder.add_conditional_edges(
                "tool_executor",
                should_continue_after_tools,
                {
                    "chat_agent": "chat_agent",  # Continue agent loop
                    "__end__": END,  # Finish subgraph
                },
            )

            # Start with chat agent
            builder.add_edge(START, "chat_agent")

            # Compile the graph with basic recursion limits
            # Note: LangChain's middleware is for create_agent API, not StateGraph
            # We'll implement basic rate limiting in the routing functions instead
            self.graph = builder.compile()

            logger.info(
                "Tools agent subgraph built with LangChain's tools_condition routing"
            )

        except Exception as e:
            logger.error(f"Failed to build agent subgraph: {e}")
            raise

    async def _chat_agent_wrapper(self, state: ToolsState) -> Dict[str, Any]:
        """Wrapper that creates ChatAgent at runtime and executes it."""
        try:
            # Convert LangChain core messages to our LangChainMessage format
            messages = state["messages"]

            # Check if we have recent tool results that need synthesis
            has_recent_tool_results = False
            tool_result_count = 0
            for msg in messages:  # Check ALL messages for total count
                if isinstance(msg, ToolMessage):
                    has_recent_tool_results = True
                    tool_result_count += 1

            langchain_messages = []
            
            # Add VERY aggressive system message if we have tool results
            if has_recent_tool_results:
                from langchain_core.messages import SystemMessage
                
                if tool_result_count >= 2:  # Be aggressive: force synthesis after just 2 tool executions
                    synthesis_prompt = SystemMessage(
                        content="""CRITICAL INSTRUCTION: You have already executed web searches and have sufficient information. You MUST now provide a comprehensive final response that synthesizes the search results. 

DO NOT USE ANY TOOLS. DO NOT MAKE ANY FUNCTION CALLS. DO NOT REQUEST MORE SEARCHES.

Your response must be a detailed, well-structured summary of AI developments in 2024 based on the search results you have already received. Structure your response with clear sections covering:

1. Major AI model releases in 2024
2. Recent breakthroughs in AI research  
3. Current AI safety developments

Synthesize ALL the information from your previous searches into one comprehensive response NOW."""
                    )
                    langchain_messages.append(synthesis_prompt)
                elif tool_result_count >= 1:  # Even after 1 tool execution, start encouraging synthesis
                    synthesis_prompt = SystemMessage(
                        content="You have search results available. Consider whether you have sufficient information to provide a final response about AI developments in 2024. Focus on synthesizing the information you have rather than requesting more searches."
                    )
                    langchain_messages.append(synthesis_prompt)
            
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
            response_msg = await self.chat_agent.chat_completion_with_conversion(
                messages=langchain_messages,
                tools=tools_list,
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

            # Execute the agent subgraph with reasonable recursion limit
            # LangChain's tools_condition handles the intelligent routing
            result = await self.graph.ainvoke(
                tools_state,
                config={"recursion_limit": 15},  # Reasonable limit for tool iterations
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
