"""
Tools Agent Subgraph - Complete agent workflow with chat_agent + tool_node cycling.

This subgraph implements the proper LangGraph agent pattern with both chat_agent and 
tool_node, allowing the full agent workflow to execute internally with minimal state.
The agent cycles between LLM calls and tool execution until completion, then returns
results to the main workflow via middleware-controlled ingress/egress.

Key Benefits:
1. Complete agent workflow - chat_agent <-> tool_node cycling within subgraph
2. Minimal state - ToolsState with only essential fields to minimize context usage
3. Proper tool integration - uses ToolNode with ToolRuntime pattern
4. Middleware control - clean ingress/egress boundaries with main workflow
5. State isolation - agent operations don't bloat main workflow state

Architecture:
- ToolsState: Minimal state optimized for agent operations
- chat_agent: LLM node that can make tool calls using available tools
- tool_node: ToolNode that executes tools with ToolRuntime[ToolsState] access
- Conditional routing: should_continue logic for agent cycling
- Middleware boundaries: controlled data flow to/from main workflow
"""

from typing import Dict, List, Any, Optional, Literal
from typing_extensions import TypedDict, Annotated

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, create_react_agent
from langgraph.types import Command
from langgraph.graph.message import add_messages

from models import MessageRole, MessageContent, MessageContentType, NodeMetadata, PipelinePriority
from composer.graph.state import WorkflowState
from composer.agents.chat_agent import ChatAgent
from composer.tools.registry import ToolRegistry
from runner import PipelineFactory
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="ToolsAgentSubgraph")


class ToolsState(TypedDict):
    """
    Minimal state for agent subgraph with chat_agent + tool_node workflow.
    
    Contains only essential data for the agent to operate efficiently while
    minimizing context window usage. The agent cycles between chat_agent and
    tool_node until completion, then returns results via Command.
    """
    # Message thread for agent conversation (using LangChain core messages for proper serialization)
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Essential context for tool operations
    user_id: str
    conversation_id: int
    
    # User configuration (full object for tool access)  
    user_config: Optional[Any]  # UserConfig object, avoiding circular import
    system_config: Optional[Dict[str, Any]]
    
    # Current operation tracking
    current_date: str
    tool_call_count: int


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
            conversation_id=0   # Will be updated at runtime
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
            conversation_id=conversation_id
        )
        
        return ChatAgent(
            pipeline_factory=self.pipeline_factory,
            profile=DEFAULT_PRIMARY_PROFILE,
            node_metadata=runtime_metadata,
            priority=PipelinePriority.MEDIUM
        )
    
    async def _tool_executor_wrapper(self, state: ToolsState) -> Dict[str, Any]:
        """Tool executor wrapper using LangGraph's ToolNode."""
        try:
            # Get executable tools from registry
            executable_tools = self.tool_registry.get_all_executable_tools()
            tools_list = list(executable_tools.values()) if executable_tools else []
            
            if not tools_list:
                logger.warning("No tools available for execution")
                return state
            
            # Create ToolNode with available tools
            tool_node = ToolNode(tools_list)
            
            # Execute tools using ToolNode
            result = await tool_node.ainvoke(state)
            
            return result
            
        except Exception as e:
            logger.error(f"Tool executor wrapper failed: {e}")
            # Return current state on error
            return state
    
    def _build_graph(self) -> None:
        """Build the complete agent subgraph using proper dependency injection."""
        try:
            # Build graph with StateGraph pattern like main builder
            builder = StateGraph(ToolsState)
            
            # Add chat agent node - will be created at runtime with proper context
            builder.add_node("chat_agent", self._chat_agent_wrapper)
            
            # Add tool executor node - using wrapper for ToolNode
            builder.add_node("tool_executor", self._tool_executor_wrapper)
            
            # Add conditional routing between chat agent and tool executor
            builder.add_conditional_edges(
                "chat_agent",
                self._should_continue,
                {
                    "continue": "tool_executor", 
                    "end": END
                }
            )
            
            # Tool executor always goes back to chat agent for potential follow-up
            builder.add_edge("tool_executor", "chat_agent")
            
            # Start with chat agent
            builder.add_edge(START, "chat_agent")
            
            # Compile the graph
            self.graph = builder.compile()
            logger.info("Agent subgraph built with proper dependency injection")
            
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
            
            # Use messages directly - ChatAgent should handle LangChain core messages
            messages = state["messages"]
            
            # For now, we need to convert to our format but this should be simplified
            from models import LangChainMessage
            langchain_messages = []
            
            for msg in messages:
                if isinstance(msg, (HumanMessage, AIMessage, ToolMessage)):
                    # Convert to LangChainMessage format
                    langchain_msg = LangChainMessage(
                        content=msg.content,
                        type=msg.type,
                        additional_kwargs=getattr(msg, 'additional_kwargs', {}),
                        response_metadata=getattr(msg, 'response_metadata', {})
                    )
                    langchain_messages.append(langchain_msg)
                else:
                    # Already in correct format or compatible
                    langchain_messages.append(msg)
            
            # Get tools from registry for the agent
            executable_tools = self.tool_registry.get_all_executable_tools()
            tools_list = list(executable_tools.values()) if executable_tools else None
            
            # Execute chat completion with tools
            response_msg = await chat_agent.chat_completion_with_conversion(
                messages=langchain_messages,
                tools=tools_list
            )
            
            # Convert response back to LangChain core message format for ToolsState
            # AIMessage already imported at module level
            
            # Extract tool calls if present
            tool_calls = []
            if hasattr(response_msg, 'tool_calls') and response_msg.tool_calls:
                for tc in response_msg.tool_calls:
                    tool_calls.append({
                        "name": tc.get("name", ""),
                        "args": tc.get("args", {}),
                        "id": tc.get("id", ""),
                        "type": "tool_call"
                    })
            
            # Create AIMessage compatible with LangGraph
            ai_message = AIMessage(
                content=response_msg.content,
                tool_calls=tool_calls if tool_calls else [],
                additional_kwargs=getattr(response_msg, 'additional_kwargs', {}),
                response_metadata=getattr(response_msg, 'response_metadata', {})
            )
            
            # Return new message in state update format
            return {"messages": [ai_message]}
            
        except Exception as e:
            logger.error(f"Chat agent wrapper failed: {e}")
            # Return error message
            error_msg = AIMessage(content=f"Agent error: {str(e)}")
            return {"messages": [error_msg]}
    
    def _should_continue(self, state: ToolsState) -> Literal["continue", "end"]:
        """Determine if agent should continue to tools or end."""
        messages = state["messages"]
        if not messages:
            return "end"
            
        last_message = messages[-1]
        
        # Check if last message has tool calls
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "continue"
        
        return "end"
    

    

    

    
    def transform_to_tools_state(self, main_state: WorkflowState) -> ToolsState:
        """Transform main WorkflowState to minimal ToolsState for agent subgraph."""
        # Get recent messages for agent context and convert to LangChain core messages
        recent_messages = getattr(main_state, "messages", [])[-10:]
        langchain_messages = []
        
        for msg in recent_messages:
            if hasattr(msg, 'type') and hasattr(msg, 'content'):
                # Convert custom LangChainMessage to proper LangChain core message
                if msg.type == "human":
                    langchain_messages.append(HumanMessage(content=msg.content))
                elif msg.type == "ai":
                    # Check if this AI message has tool calls
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        langchain_messages.append(AIMessage(
                            content=msg.content,
                            tool_calls=msg.tool_calls
                        ))
                    else:
                        langchain_messages.append(AIMessage(content=msg.content))
                elif msg.type == "tool":
                    langchain_messages.append(ToolMessage(
                        content=msg.content,
                        tool_call_id=getattr(msg, 'id', None) or "unknown"
                    ))
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
            "tool_call_count": 0
        }
    
    def transform_to_main_state(self, agent_result: Dict[str, Any], main_state: WorkflowState) -> Dict[str, Any]:
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
                        lang_chain_msg = LangChainMessage(
                            content=msg.content,
                            type=msg.type,
                            name=getattr(msg, 'name', None),
                            id=getattr(msg, 'id', None) or getattr(msg, 'tool_call_id', None),
                            additional_kwargs=getattr(msg, 'additional_kwargs', {}),
                            response_metadata=getattr(msg, 'response_metadata', {})
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
            
            # Execute the agent subgraph
            result = await self.graph.ainvoke(tools_state)
            
            # Transform results back to main state updates
            updates = self.transform_to_main_state(result, main_state)
            
            logger.info(f"Agent subgraph completed with {len(updates)} state updates")
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