"""
Standard LangGraph nodes for basic workflow operations.
Implements PipelineNode, ToolExecutorNode, RAGNode per Phase 2 requirements.
"""

import asyncio
from typing import List, Any, Optional, Dict, AsyncIterator, Union

from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolNode

from models import Message, ChatResponse, LangChainMessage, AvailableTool, ModelProfileType

# Lazy imports to avoid circular dependencies
# from db import storage - imported when needed
# from utils.model_profile_utils import get_model_profile_for_task - imported when needed

from composer.graph.state import WorkflowState
from composer.monitoring.logging import composer_logger
from composer.core.errors import NodeExecutionError


class PipelineNode:
    """
    Wraps chat-model execution as a graph node.
    
    Handles both streaming and non-streaming execution based on configuration.
    Retrieves model profiles internally from shared data layer using user_id.
    """

    def __init__(self, 
                 pipeline_factory, 
                 profile_type: ModelProfileType, 
                 stream: bool = False):
        """
        Initialize pipeline node.
        
        Args:
            pipeline_factory: Factory for creating pipeline instances
            profile_type: Model profile type (Primary, Analysis, etc.)
            stream: Whether to enable streaming responses
        """
        self.pipeline_factory = pipeline_factory
        self.profile_type = profile_type
        self.stream = stream
        self.logger = composer_logger.bind(component="PipelineNode")

    async def __call__(self, state: WorkflowState) -> WorkflowState:
        """
        Execute pipeline node with grammar-constrained generation.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state with response
        """
        try:
            # Retrieve user configuration from shared data layer
            user_id = getattr(state, 'user_id', None)
            if not user_id:
                raise NodeExecutionError("User ID required for pipeline execution")
            
            # Lazy imports to avoid circular dependencies
            try:
                from db import storage
                from utils.model_profile_utils import get_model_profile_for_task
                
                uc = await storage.get_service(storage.user_config).get_user_config(user_id)
                
                # Get model profile for this task type
                model_profile = get_model_profile_for_task(
                    uc.model_profiles, self.profile_type, user_id
                )
            except ImportError as ie:
                self.logger.warning(f"Database not available: {ie}")
                model_profile = None
                uc = None
            
            self.logger.info(
                "Executing pipeline node",
                user_id=user_id,
                profile_type=self.profile_type.value,
                streaming=self.stream,
                model=model_profile.model_name if model_profile else "unknown"
            )

            # Create pipeline instance (placeholder for actual implementation)
            # TODO: Implement proper pipeline factory integration
            if self.pipeline_factory:
                pipeline = await self.pipeline_factory.get_pipeline(
                    model_profile, 
                    ChatResponse, 
                    streaming=self.stream
                )
                
                if self.stream:
                    # For streaming nodes: this will be handled by LangGraph streaming
                    # For now, just process non-streaming
                    response = await pipeline.invoke({"messages": state.messages})
                else:
                    # For non-streaming: complete response
                    response = await pipeline.invoke({"messages": state.messages})
                    
                # Add response to state messages
                assistant_message = LangChainMessage(
                    role="assistant",
                    content=getattr(response, 'content', 'Response generated'),
                    tool_calls=getattr(response, 'tool_calls', None)
                )
                state.messages.append(assistant_message)
            else:
                # Fallback when pipeline factory not available
                fallback_message = LangChainMessage(
                    role="assistant",
                    content="Pipeline factory not configured - this is a placeholder response."
                )
                state.messages.append(fallback_message)

            return state

        except Exception as e:
            self.logger.error(
                "Pipeline node execution failed",
                user_id=getattr(state, 'user_id', 'unknown'),
                error=str(e),
                profile_type=self.profile_type.value
            )
            raise NodeExecutionError(f"Pipeline execution failed: {e}") from e


class ToolExecutorNode:
    """
    Executes tool calls produced by the previous agent or tool node.
    
    Uses LangGraph's ToolNode for reliable tool execution with proper error handling.
    """

    def __init__(self, tools: List[BaseTool]):
        """
        Initialize tool executor node.
        
        Args:
            tools: List of available tools for execution
        """
        self.tools = {tool.name: tool for tool in tools}
        self.tool_node = ToolNode(tools)
        self.logger = composer_logger.bind(component="ToolExecutorNode")

    async def __call__(self, state: WorkflowState) -> WorkflowState:
        """
        Execute tool calls from the last message.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state with tool results
        """
        try:
            if not state.messages:
                return state

            last_message = state.messages[-1]
            
            # Check if last message has tool calls
            if not (hasattr(last_message, 'tool_calls') and last_message.tool_calls):
                return state

            self.logger.info(
                "Executing tool calls",
                user_id=getattr(state, 'user_id', 'unknown'),
                tool_count=len(last_message.tool_calls),
                tools=[call.get('name', 'unknown') for call in last_message.tool_calls]
            )

            # Execute tools using LangGraph ToolNode
            tool_results = await self.tool_node.ainvoke({"messages": [last_message]})
            
            # Add tool results to state messages
            if 'messages' in tool_results:
                state.messages.extend(tool_results['messages'])

            # Update tool execution status in state
            if not hasattr(state, 'tool_executions'):
                state.tool_executions = []
                
            state.tool_executions.extend([
                {
                    'tool_name': call.get('name', 'unknown'),
                    'status': 'completed',
                    'timestamp': asyncio.get_event_loop().time()
                }
                for call in last_message.tool_calls
            ])

            return state

        except Exception as e:
            self.logger.error(
                "Tool execution failed",
                user_id=getattr(state, 'user_id', 'unknown'),
                error=str(e),
                tools=list(self.tools.keys())
            )
            
            # Add error message to state
            error_message = LangChainMessage(
                role="assistant",
                content=f"Tool execution failed: {str(e)}"
            )
            state.messages.append(error_message)
            
            return state


class RAGNode:
    """
    Retrieval-Augmented Generation node.
    
    Embeds latest user message and performs retrieval augmentation based on
    user configuration retrieved from shared data layer.
    """

    def __init__(self, user_id: str):
        """
        Initialize RAG node.
        
        Args:
            user_id: User identifier for configuration retrieval
        """
        self.user_id = user_id
        self.logger = composer_logger.bind(component="RAGNode")

    async def __call__(self, state: WorkflowState) -> WorkflowState:
        """
        Perform retrieval augmentation on the latest user message.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state with search results
        """
        try:
            if not state.messages:
                return state

            # Get user query from latest message
            latest_message = state.messages[-1]
            if latest_message.role != "user":
                return state

            query = latest_message.content
            
            # Retrieve user configuration from shared data layer  
            try:
                from db import storage
                uc = await storage.get_service(storage.user_config).get_user_config(self.user_id)
                search_config = uc.workflow_preferences.search_config
            except ImportError:
                # Fallback configuration when database not available
                search_config = type('Config', (), {
                    'max_sources': 5,
                    'similarity_threshold': 0.7
                })()
            
            self.logger.info(
                "Performing RAG retrieval",
                user_id=self.user_id,
                query_length=len(query),
                search_depth=search_config.depth,
                max_sources=search_config.max_sources
            )

            # Perform vector search using existing memory retrieval
            try:
                memory_service = storage.get_service(storage.memory)
            except (NameError, AttributeError):
                # Database not available - skip memory search
                state.search_results = "Memory search unavailable - database not configured."
                return state
            
            # Search conversation memory
            memories = await memory_service.search_memories(
                user_id=self.user_id,
                query=query,
                limit=search_config.max_sources,
                similarity_threshold=search_config.similarity_threshold
            )

            # Format search results
            if memories:
                search_results = "\n\n".join([
                    f"Memory {i+1}: {memory.content}"
                    for i, memory in enumerate(memories)
                ])
                
                state.search_results = search_results
                
                # Add context to messages for the model
                context_message = LangChainMessage(
                    role="system",
                    content=f"Retrieved context:\n\n{search_results}"
                )
                state.messages.insert(-1, context_message)  # Insert before user message
            else:
                state.search_results = "No relevant context found."

            return state

        except Exception as e:
            self.logger.error(
                "RAG node execution failed",
                user_id=self.user_id,
                error=str(e)
            )
            
            # Continue without RAG on error
            state.search_results = f"RAG retrieval failed: {str(e)}"
            return state


class CircuitProtectedNode:
    """
    Wrapper node that provides circuit breaker protection for any node.
    
    Implements fault tolerance and graceful degradation patterns per Phase 2 requirements.
    """

    def __init__(self, wrapped_node: Any, circuit_config: Optional[Dict] = None):
        """
        Initialize circuit protected node.
        
        Args:
            wrapped_node: The node to wrap with circuit breaker protection
            circuit_config: Circuit breaker configuration
        """
        self.wrapped_node = wrapped_node
        self.circuit_config = circuit_config or {
            'failure_threshold': 5,
            'recovery_timeout': 30,
            'success_threshold': 2
        }
        
        # Circuit breaker state
        self.failure_count = 0
        self.last_failure_time = None
        self.circuit_open = False
        
        self.logger = composer_logger.bind(component="CircuitProtectedNode")

    async def __call__(self, state: WorkflowState) -> WorkflowState:
        """
        Execute wrapped node with circuit breaker protection.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state or fallback response
        """
        # Check if circuit is open
        if self.circuit_open:
            if self._should_attempt_reset():
                self.logger.info("Attempting circuit breaker reset")
                try:
                    result = await self.wrapped_node(state)
                    self._record_success()
                    return result
                except Exception as e:
                    self._record_failure()
                    return self._fallback_response(state, e)
            else:
                return self._fallback_response(state, Exception("Circuit breaker open"))

        # Normal execution
        try:
            result = await self.wrapped_node(state)
            self._record_success()
            return result
            
        except Exception as e:
            self._record_failure()
            
            if self.failure_count >= self.circuit_config['failure_threshold']:
                self.circuit_open = True
                self.logger.warning(
                    "Circuit breaker opened",
                    failure_count=self.failure_count,
                    node_type=type(self.wrapped_node).__name__
                )
            
            return self._fallback_response(state, e)

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt circuit reset."""
        if not self.last_failure_time:
            return True
            
        recovery_timeout = self.circuit_config['recovery_timeout']
        return (asyncio.get_event_loop().time() - self.last_failure_time) > recovery_timeout

    def _record_success(self):
        """Record successful execution."""
        self.failure_count = max(0, self.failure_count - 1)
        if self.failure_count == 0:
            self.circuit_open = False

    def _record_failure(self):
        """Record failed execution."""
        self.failure_count += 1
        self.last_failure_time = asyncio.get_event_loop().time()

    def _fallback_response(self, state: WorkflowState, error: Exception) -> WorkflowState:
        """Generate fallback response when node fails."""
        self.logger.error(
            "Node execution failed, using fallback",
            node_type=type(self.wrapped_node).__name__,
            error=str(error),
            circuit_open=self.circuit_open
        )
        
        # Add fallback message
        fallback_message = LangChainMessage(
            role="assistant",
            content=f"I'm experiencing technical difficulties. Please try again later. Error: {str(error)[:100]}..."
        )
        state.messages.append(fallback_message)
        
        return state