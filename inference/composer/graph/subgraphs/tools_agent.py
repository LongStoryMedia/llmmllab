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
from composer.utils.conversion import (
    convert_base_langchain_to_messages,
    convert_messages_to_langchain,
)
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="ToolsAgentSubgraph")


class ToolsAgentSubgraph:
    """
    Simple agent subgraph following LangChain quickstart pattern.
    """

    def __init__(
        self,
        chat_agent: BaseAgent,
        tool_registry: ToolRegistry,
        config: AgentConfig,
    ):
        """Initialize with direct agent and tool registry."""
        from langgraph.prebuilt import create_agent
        
        self.chat_agent = chat_agent
        self.tool_registry = tool_registry
        self.config = config
        
        # Create persistent LangChain agent to maintain conversation state
        self.langchain_agent = None
        self.graph: Optional[CompiledGraph] = None
        self._build_graph()
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

            # Add tool executor node (must be named 'tools' for tools_condition to work)
            tool_node = self._create_tool_node()
            builder.add_node("tools", tool_node)

            # EXACTLY like the LangChain quickstart - use built-in tools_condition
            builder.add_conditional_edges(
                "chat_agent",
                tools_condition,  # Use built-in routing - no custom logic
            )

            # Simple continuation after tools
            builder.add_edge("tools", "chat_agent")

            # Start with chat agent
            builder.add_edge(START, "chat_agent")

            # Compile graph - no recursion limit needed since we maintain state properly
            self.graph = builder.compile()
            
            # Initialize persistent LangChain agent after graph is compiled
            self._initialize_persistent_agent()

            logger.info("Simple tools agent subgraph built following LangChain pattern")

        except Exception as e:
            logger.error(f"Failed to build agent subgraph: {e}")
            raise

    def _initialize_persistent_agent(self) -> None:
        """Initialize persistent LangChain agent to maintain conversation state."""
        try:
            # Import at method level to avoid circular dependencies
            from langchain_core.language_models.chat_models import BaseChatModel
            from typing import cast
            
            # Get tools for the agent
            executable_tools = self.tool_registry.get_all_executable_tools()
            tools_list = list(executable_tools.values()) if executable_tools else []
            
            # Get model from chat agent's pipeline factory
            with self.chat_agent.pipeline_factory.pipeline(
                self.chat_agent.profile, 
                self.chat_agent.priority
            ) as chat_model:
                if not chat_model:
                    raise ValueError("Failed to create chat model for persistent agent")
                    
                llm = cast(BaseChatModel, chat_model)
                
                # Create persistent agent with system prompt
                system_prompt = self.chat_agent.profile.system_prompt or ""
                
                # Create StateGraph-based agent directly (like our main pattern)
                from langgraph.graph import StateGraph, START, END
                from langchain_core.messages import AIMessage
                
                def llm_node(state):
                    """LLM node that processes messages and decides on tool calls."""
                    messages = state["messages"]
                    
                    # Add system message if needed
                    if system_prompt and (not messages or messages[0].type != "system"):
                        from langchain_core.messages import SystemMessage
                        messages = [SystemMessage(content=system_prompt)] + messages
                    
                    # Bind tools and invoke
                    model_with_tools = llm.bind_tools(tools_list)
                    response = model_with_tools.invoke(messages)
                    return {"messages": [response]}
                
                # Build the persistent agent graph
                agent_builder = StateGraph(ToolsState)
                agent_builder.add_node("llm", llm_node)
                agent_builder.add_node("tools", ToolNode(tools_list))
                
                agent_builder.add_edge(START, "llm")
                agent_builder.add_conditional_edges("llm", tools_condition)
                agent_builder.add_edge("tools", "llm")
                
                self.langchain_agent = agent_builder.compile()
                
                logger.info(f"✅ Created persistent LangChain agent with {len(tools_list)} tools")
                
        except Exception as e:
            logger.error(f"Failed to initialize persistent agent: {e}")
            # Fallback to None - use original method
            self.langchain_agent = None

    async def _chat_agent_wrapper(self, state: ToolsState) -> Dict[str, Any]:
        """Chat agent wrapper using persistent LangChain agent to maintain state."""
        try:
            # Use persistent agent if available (maintains conversation state)
            if self.langchain_agent:
                # Direct invocation of persistent agent with current state
                result = await self.langchain_agent.ainvoke(state)
                # The persistent agent returns the proper state format already
                return result
            
            # Fallback to original method if persistent agent failed to initialize
            messages = state["messages"]
            langchain_messages = convert_messages_to_langchain(
                convert_base_langchain_to_messages(messages)
            )

            # Get tools from registry for the agent
            executable_tools = self.tool_registry.get_all_executable_tools()
            tools_list = list(executable_tools.values()) if executable_tools else None

            # Execute chat completion with tools (creates new agent each time - problematic)
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
