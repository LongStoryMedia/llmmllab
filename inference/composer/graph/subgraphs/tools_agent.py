"""
Tools Agent Subgraph for efficient tool execution with minimal state.

This subgraph provides a lightweight environment for executing tools with a simplified
state structure to minimize context window usage and prevent schema bloat. It uses
the new ToolRuntime pattern instead of InjectedState to access necessary context.

Key Benefits:
1. Minimal state - only essential data for tool execution
2. State isolation - tool execution doesn't affect main workflow state directly  
3. Clean tool schemas - no massive state injection in tool definitions
4. Easy integration - returns results to main workflow via Command pattern

Architecture:
- ToolsState: Minimal state with only required fields
- Tool execution node: Handles tool calls with ToolRuntime access
- Result aggregation: Collects tool outputs for return to main workflow
- State transformation: Converts between main WorkflowState and ToolsState
"""

import asyncio
import inspect
from typing import Dict, List, Any, Optional, Sequence
from typing_extensions import TypedDict
from dataclasses import dataclass

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langgraph.prebuilt import ToolNode

from models import MessageRole, MessageContent, MessageContentType, Tool
from composer.graph.state import WorkflowState
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="ToolsAgentSubgraph")


class ToolsState(TypedDict):
    """
    Minimal state for tool execution subgraph.
    
    Contains only the essential data needed for tool execution to minimize
    context window usage and prevent schema bloat.
    """
    # Essential tool execution data
    messages: List[BaseMessage]  # Only recent messages needed for tool context
    user_id: str  # User identifier for tool operations
    conversation_id: int  # Conversation context
    
    # User configuration subset (only tool-relevant settings)
    web_search_config: Optional[Dict[str, Any]]  # Web search preferences
    memory_config: Optional[Dict[str, Any]]  # Memory retrieval settings
    
    # Tool execution results
    tool_results: List[Dict[str, Any]]  # Collected tool outputs
    

@dataclass
class ToolExecutionContext:
    """Context data for tool execution that doesn't need to be in state."""
    user_id: str
    conversation_id: int
    web_search_config: Dict[str, Any]
    memory_config: Dict[str, Any]


class ToolsAgentSubgraph:
    """
    Subgraph for efficient tool execution with minimal state overhead.
    
    This subgraph handles tool execution in an isolated environment with a
    simplified state structure to prevent context window bloat while maintaining
    full tool functionality via ToolRuntime access patterns.
    """
    
    def __init__(self):
        self.graph = None
        self._build_graph()
    
    def _build_graph(self) -> None:
        """Build the tools execution subgraph."""
        builder = StateGraph(ToolsState)
        
        # Add the tool execution node
        builder.add_node("execute_tools", self._execute_tools_node)
        
        # Simple linear flow
        builder.add_edge(START, "execute_tools")
        builder.add_edge("execute_tools", END)
        
        # Compile the subgraph
        self.graph = builder.compile()
        logger.info("Tools agent subgraph compiled successfully")
    
    async def _execute_tools_node(self, state: ToolsState) -> Dict[str, Any]:
        """
        Execute tools using the existing tool pattern with state injection.
        
        This node works with the current InjectedState pattern by creating a
        compatible execution environment for tools.
        """
        try:
            # Get the tools that need to be executed from the messages
            # Handle both LangChain core messages and LangChainMessage format
            tool_messages = []
            for msg in state["messages"]:
                has_tool_calls = False
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    has_tool_calls = True
                elif hasattr(msg, 'type') and msg.type == 'ai' and hasattr(msg, 'tool_calls') and msg.tool_calls:
                    has_tool_calls = True
                
                if has_tool_calls:
                    tool_messages.append(msg)
            
            if not tool_messages:
                logger.warning("No tool calls found in messages")
                return {"tool_results": []}
            
            # Extract tool calls from the latest AI message
            latest_ai_message = tool_messages[-1]
            tool_calls = latest_ai_message.tool_calls
            
            if not tool_calls:
                logger.warning("No tool calls in latest AI message")
                return {"tool_results": []}
            
            logger.info(f"Executing {len(tool_calls)} tool calls", tool_names=[call["name"] for call in tool_calls])
            
            # Get available tools from the registry
            from composer.tools.registry import ToolRegistry
            from runner.pipeline_factory import pipeline_factory
            
            registry = ToolRegistry(pipeline_factory)
            executable_tools = registry.get_all_executable_tools()
            
            if not executable_tools:
                logger.error("No executable tools available")
                return {"tool_results": []}
            
            # Execute each tool call
            tool_results = []
            new_messages = list(state["messages"])  # Copy existing messages
            
            for call in tool_calls:
                tool_name = call.get("name")
                args = call.get("args") or call.get("arguments") or {}
                tool_call_id = call.get("id", f"call_{tool_name}")
                
                if tool_name not in executable_tools:
                    logger.error(f"Tool '{tool_name}' not found in available tools")
                    tool_results.append({
                        "tool_call_id": tool_call_id,
                        "name": tool_name,
                        "content": f"Error: Tool '{tool_name}' not available",
                        "status": "failed"
                    })
                    continue
                
                tool = executable_tools[tool_name]
                
                try:
                    # Check tool function signature to determine execution method
                    import inspect
                    
                    # Get the actual function to inspect its signature
                    # Check both 'func' and 'coroutine' attributes for LangGraph tools
                    tool_func = getattr(tool, 'func', None) or getattr(tool, 'coroutine', None)
                    if tool_func and (inspect.isfunction(tool_func) or inspect.iscoroutinefunction(tool_func)):
                        sig = inspect.signature(tool_func)
                        param_names = list(sig.parameters.keys())
                        
                        # Check if this tool expects injected parameters
                        if 'tool_call_id' in param_names and 'state' in param_names:
                            # This is a LangGraph tool with injection - call directly with injected params
                            minimal_state = self._create_minimal_workflow_state(state)
                            
                            # Call the function directly with injected parameters
                            if inspect.iscoroutinefunction(tool_func):
                                # Async function - call directly
                                result = await tool_func(
                                    tool_call_id=tool_call_id,
                                    state=minimal_state,
                                    **args
                                )
                            else:
                                # Sync function - use thread
                                result = await asyncio.create_task(
                                    asyncio.to_thread(
                                        tool_func,
                                        tool_call_id=tool_call_id,
                                        state=minimal_state,
                                        **args
                                    )
                                )
                            
                            # Handle Command returns (like web_search)
                            if hasattr(result, 'update') and result.update:
                                # Apply command updates to our minimal state
                                for key, value in result.update.items():
                                    if hasattr(minimal_state, key):
                                        # Special handling for messages - convert LangChain messages to LangChainMessage schema
                                        if key == 'messages' and isinstance(value, list):
                                            from composer.utils.langchain_compat import _coerce_to_langchain_message_dict
                                            converted_messages = []
                                            for msg in value:
                                                converted_messages.append(_coerce_to_langchain_message_dict(msg))
                                            setattr(minimal_state, key, converted_messages)
                                        else:
                                            setattr(minimal_state, key, value)
                                        
                                # Extract the actual result content
                                result_content = "Tool completed and results added to state"
                            else:
                                result_content = str(result)
                        else:
                            # Regular LangChain tool - use the standard execution pattern
                            from langchain_core.runnables import RunnableConfig
                            
                            tool_config = RunnableConfig()
                            
                            if hasattr(tool, "_arun"):
                                result_content = await tool._arun(config=tool_config, **args)
                            else:
                                # Fallback to sync _run
                                run_fn = getattr(tool, "_run", None) or getattr(tool, "run", None)
                                if run_fn is None:
                                    raise RuntimeError(f"Tool '{tool_name}' has no runnable method")
                                result_content = run_fn(**args)
                    else:
                        # Fallback: try different execution methods
                        if hasattr(tool, "_arun"):
                            # Try with injection parameters first
                            try:
                                minimal_state = self._create_minimal_workflow_state(state)
                                result_content = await tool._arun(
                                    tool_call_id=tool_call_id,
                                    state=minimal_state,
                                    **args
                                )
                            except TypeError:
                                # If that fails, try without injection
                                from langchain_core.runnables import RunnableConfig
                                tool_config = RunnableConfig()
                                result_content = await tool._arun(config=tool_config, **args)
                        else:
                            run_fn = getattr(tool, "_run", None) or getattr(tool, "run", None)
                            if run_fn is None:
                                raise RuntimeError(f"Tool '{tool_name}' has no runnable method")
                            result_content = run_fn(**args)
                    
                    # Create tool message
                    tool_message = ToolMessage(
                        content=str(result_content),
                        tool_call_id=tool_call_id,
                        name=tool_name
                    )
                    new_messages.append(tool_message)
                    
                    tool_results.append({
                        "tool_call_id": tool_call_id,
                        "name": tool_name,
                        "content": str(result_content),
                        "status": "success"
                    })
                    
                    logger.info(f"Successfully executed tool '{tool_name}'")
                    
                except Exception as e:
                    logger.error(f"Tool '{tool_name}' execution failed: {e}", exc_info=True)
                    
                    error_message = ToolMessage(
                        content=f"Error executing {tool_name}: {str(e)}",
                        tool_call_id=tool_call_id,
                        name=tool_name
                    )
                    new_messages.append(error_message)
                    
                    tool_results.append({
                        "tool_call_id": tool_call_id,
                        "name": tool_name,
                        "content": f"Error: {str(e)}",
                        "status": "failed"
                    })
            
            logger.info(f"Successfully executed {len(tool_results)} tools")
            return {"tool_results": tool_results, "messages": new_messages}
            
        except Exception as e:
            logger.error(f"Tool execution failed: {e}", exc_info=True)
            return {
                "tool_results": [{
                    "error": str(e),
                    "status": "failed"
                }]
            }
    
    def _create_minimal_workflow_state(self, tools_state: ToolsState) -> 'WorkflowState':
        """
        Create a minimal WorkflowState for tool injection from ToolsState.
        
        This allows tools that expect InjectedState to work with our subgraph.
        """
        # Import WorkflowState here to avoid circular imports
        from composer.graph.state import WorkflowState
        from models import LangChainMessage, UserConfig
        from models.default_configs import (
            DEFAULT_SUMMARIZATION_CONFIG,
            DEFAULT_MEMORY_CONFIG, 
            DEFAULT_WEB_SEARCH_CONFIG,
            DEFAULT_PREFERENCES_CONFIG,
            DEFAULT_MODEL_PROFILE_CONFIG,
            DEFAULT_REFINEMENT_CONFIG,
            DEFAULT_IMAGE_GENERATION_CONFIG,
            DEFAULT_CIRCUIT_BREAKER_CONFIG,
            DEFAULT_GPU_CONFIG,
            DEFAULT_WORKFLOW_CONFIG,
            DEFAULT_TOOL_CONFIG,
            DEFAULT_CONTEXT_WINDOW_CONFIG
        )
        
        # Create a minimal state object with required fields
        minimal_state = WorkflowState()
        
        # Convert LangChain core messages to LangChainMessage format
        converted_messages = []
        for msg in tools_state["messages"]:
            if isinstance(msg, (HumanMessage, AIMessage, ToolMessage)):
                # Convert to LangChainMessage format
                lang_chain_msg = LangChainMessage(
                    content=msg.content,
                    type=msg.type,
                    additional_kwargs=getattr(msg, 'additional_kwargs', {}),
                    response_metadata=getattr(msg, 'response_metadata', {})
                )
                converted_messages.append(lang_chain_msg)
            else:
                # Already in correct format
                converted_messages.append(msg)
        
        # Set essential fields from tools_state
        minimal_state.messages = converted_messages
        minimal_state.user_id = tools_state["user_id"]
        minimal_state.conversation_id = tools_state["conversation_id"]
        
        # Create a minimal but valid UserConfig using defaults and subsets
        web_search_config = tools_state.get("web_search_config", {})
        memory_config = tools_state.get("memory_config", {})
        
        # Merge with defaults
        merged_web_search = {**DEFAULT_WEB_SEARCH_CONFIG.model_dump(), **web_search_config}
        merged_memory = {**DEFAULT_MEMORY_CONFIG.model_dump(), **memory_config}
        
        minimal_user_config = UserConfig(
            user_id=tools_state["user_id"],
            summarization=DEFAULT_SUMMARIZATION_CONFIG,
            memory=type(DEFAULT_MEMORY_CONFIG)(**merged_memory),
            web_search=type(DEFAULT_WEB_SEARCH_CONFIG)(**merged_web_search),
            preferences=DEFAULT_PREFERENCES_CONFIG,
            model_profiles=DEFAULT_MODEL_PROFILE_CONFIG,
            refinement=DEFAULT_REFINEMENT_CONFIG,
            image_generation=DEFAULT_IMAGE_GENERATION_CONFIG,
            circuit_breaker=DEFAULT_CIRCUIT_BREAKER_CONFIG,
            gpu_config=DEFAULT_GPU_CONFIG,
            workflow=DEFAULT_WORKFLOW_CONFIG,
            tool=DEFAULT_TOOL_CONFIG,
            context_window=DEFAULT_CONTEXT_WINDOW_CONFIG
        )
        
        minimal_state.user_config = minimal_user_config
        
        # Set other required fields to defaults
        minimal_state.current_date = ""
        minimal_state.things_to_remember = []
        minimal_state.web_search_results = []
        minimal_state.tool_calls = []
        
        return minimal_state
    
    def transform_to_tools_state(self, main_state: WorkflowState) -> ToolsState:
        """
        Transform main WorkflowState to minimal ToolsState.
        
        Extracts only the essential data needed for tool execution to minimize
        context window usage.
        """
        # Get only recent messages (last 10 to keep context minimal)
        recent_messages = getattr(main_state, "messages", [])[-10:]
        
        # Extract user config subsets
        user_config = getattr(main_state, "user_config", None)
        web_search_config = {}
        memory_config = {}
        
        if user_config:
            web_search_config = getattr(user_config, "web_search", {})
            memory_config = getattr(user_config, "memory", {})
        
        return ToolsState(
            messages=recent_messages,
            user_id=getattr(main_state, "user_id", ""),
            conversation_id=getattr(main_state, "conversation_id", 0),
            web_search_config=web_search_config,
            memory_config=memory_config,
            tool_results=[]
        )
    
    def transform_to_main_state(self, tools_state: ToolsState, main_state: WorkflowState) -> Dict[str, Any]:
        """
        Transform ToolsState results back to main WorkflowState updates.
        
        Returns a state update dictionary that can be applied to the main workflow.
        """
        from models import LangChainMessage
        
        updates = {}
        
        # Add tool messages to main state
        if tools_state.get("messages"):
            # Only add new tool messages
            main_messages = getattr(main_state, "messages", [])
            tools_messages = tools_state["messages"]
            
            # Find new messages (tool responses) and convert to LangChainMessage format
            new_messages = []
            for msg in tools_messages:
                if isinstance(msg, ToolMessage) and msg not in main_messages:
                    # Convert ToolMessage to LangChainMessage format
                    lang_chain_msg = LangChainMessage(
                        content=msg.content,
                        type="tool",
                        name=getattr(msg, 'name', None),
                        id=getattr(msg, 'tool_call_id', None),
                        additional_kwargs=getattr(msg, 'additional_kwargs', {}),
                        response_metadata=getattr(msg, 'response_metadata', {})
                    )
                    new_messages.append(lang_chain_msg)
                elif not isinstance(msg, ToolMessage) and msg not in main_messages:
                    # For non-tool messages, add as-is if they're already LangChainMessage
                    new_messages.append(msg)
            
            if new_messages:
                updates["messages"] = main_messages + new_messages
        
        # Add tool results to things_to_remember if they have useful content
        tool_results = tools_state.get("tool_results", [])
        if tool_results:
            things_to_remember = getattr(main_state, "things_to_remember", [])
            for result in tool_results:
                if result.get("status") == "success" and result.get("content"):
                    things_to_remember.append({
                        "type": "tool_result",
                        "tool_name": result.get("name", "unknown"),
                        "content": result["content"],
                        "timestamp": getattr(main_state, "current_date", "")
                    })
            updates["things_to_remember"] = things_to_remember
        
        return updates
    
    async def execute(self, main_state: WorkflowState) -> Command:
        """
        Execute the tools subgraph and return a Command with state updates.
        
        This is the main entry point for using the subgraph from the main workflow.
        """
        try:
            # Transform to minimal tools state
            tools_state = self.transform_to_tools_state(main_state)
            
            # Execute the subgraph
            result = await self.graph.ainvoke(tools_state)
            
            # Transform results back to main state updates
            updates = self.transform_to_main_state(result, main_state)
            
            logger.info(f"Tools subgraph completed with {len(updates)} state updates")
            return Command(update=updates)
            
        except Exception as e:
            logger.error(f"Tools subgraph execution failed: {e}", exc_info=True)
            # Return empty update on failure
            return Command(update={})


# Global instance for use in main workflow
tools_agent_subgraph = ToolsAgentSubgraph()