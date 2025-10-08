"""
Example integration of context assembly utility in workflow nodes.

This demonstrates how to use the assemble_context_messages utility function
in real workflow scenarios following the context extension architecture.
"""

from composer.graph.state import WorkflowState, assemble_context_messages
from models import MessageRole, LangChainMessage
# Note: These imports would need to be adjusted based on actual project structure
# from server.services.pipeline_factory import pipeline_factory
# from server.services.pipeline import run_pipeline
import logging

logger = logging.getLogger(__name__)


class ContextAwareWorkflowNode:
    """
    Example workflow node that demonstrates proper context assembly usage.
    
    This node shows how to integrate the context extension architecture
    into workflow execution using the assemble_context_messages utility.
    """
    
    def __init__(self, node_name: str):
        self.node_name = node_name
        self.logger = logger.getChild(node_name)
    
    async def _mock_pipeline_run(self, messages, profile_id):
        """Mock pipeline run for demonstration purposes."""
        class MockResponse:
            def __init__(self):
                self.message = f"Mock response from {profile_id} with {len(messages)} context messages"
        
        return MockResponse()
    
    async def chat_node(self, state: WorkflowState) -> WorkflowState:
        """
        Standard chat workflow node with full context assembly.
        
        This demonstrates the most common usage pattern where all context
        components are included for optimal conversation continuity.
        """
        try:
            self.logger.info("Assembling context for chat workflow")
            
            # Assemble complete context following context extension architecture
            context_messages = assemble_context_messages(
                state,
                max_context_tokens=8000,  # Adjust based on model capacity
                include_search_results=True,  # External Search RAG
                include_memories=True,        # Memory Search RAG  
                include_summaries=True        # In-Context Summarization
            )
            
            self.logger.info(
                f"Context assembled successfully: {len(context_messages)} total messages, "
                f"{len([m for m in context_messages if m.role == MessageRole.SYSTEM])} system messages, "
                f"{len([m for m in context_messages if m.role == MessageRole.USER])} user messages"
            )
            
            # Get user configuration and model profile
            if not state.user_config:
                raise ValueError("User configuration required for chat workflow")
            
            # Use default profile for conversational interactions
            # Note: Adjust based on actual UserConfig structure
            profile_id = "default_chat_profile"
            
            # Create pipeline and run with assembled context
            # Note: Replace with actual pipeline implementation
            response = await self._mock_pipeline_run(context_messages, profile_id)
            
            # Add response to state messages as LangChainMessage
            if response.message:
                # Convert response to LangChainMessage format for state storage
                response_message = LangChainMessage(
                    content=response.message,
                    type="ai"
                )
                state.messages.append(response_message)
                
                self.logger.info("Chat response generated successfully")
            
            return state
            
        except Exception as e:
            self.logger.error(f"Chat workflow failed: {e}", exc_info=True)
            state.error_details.append(f"Chat node error: {str(e)}")
            return state
    
    async def research_node(self, state: WorkflowState) -> WorkflowState:
        """
        Research workflow node emphasizing search results and memories.
        
        This demonstrates selective context assembly for research tasks
        where external information and historical context are prioritized.
        """
        try:
            self.logger.info("Assembling context for research workflow")
            
            # Research-focused context assembly
            research_context = assemble_context_messages(
                state,
                max_context_tokens=12000,     # Larger context for detailed research
                include_search_results=True,  # Critical for research accuracy
                include_memories=True,        # Historical research context
                include_summaries=False       # Skip summaries for detailed analysis
            )
            
            self.logger.info(
                f"Research context assembled: {len(research_context)} total messages, "
                f"has search results: {bool(state.search_results)}"
            )
            
            # Use research-specific model profile
            if not state.user_config:
                raise ValueError("User configuration required for research workflow")
            
            profile_id = "research_profile"
            
            response = await self._mock_pipeline_run(research_context, profile_id)
            
            if response and hasattr(response, 'message') and response.message:
                response_message = LangChainMessage(
                    content=response.message,
                    type="ai"
                )
                state.messages.append(response_message)
                
                # Store research findings in state for potential summarization
                # Note: Using error list as a placeholder for execution metadata
                state.error_details.append(f"Research completed: {response.message[:100]}...")
                
                self.logger.info("Research findings stored in execution metadata")
                
                self.logger.info("Research response generated successfully")
            
            return state
            
        except Exception as e:
            self.logger.error(f"Research workflow failed: {e}", exc_info=True)
            state.error_details.append(f"Research node error: {str(e)}")
            return state
    
    async def memory_focused_node(self, state: WorkflowState) -> WorkflowState:
        """
        Memory-focused workflow node for personal assistance.
        
        This demonstrates context assembly that prioritizes retrieved memories
        for personalized interactions based on conversation history.
        """
        try:
            self.logger.info("Assembling memory-focused context")
            
            # Memory-centric context assembly
            memory_context = assemble_context_messages(
                state,
                max_context_tokens=6000,
                include_search_results=False,  # Focus on personal context
                include_memories=True,         # Primary context source
                include_summaries=True         # Conversation continuity
            )
            
            # Ensure we have sufficient memory context
            memory_messages = [
                m for m in memory_context 
                if m.role == MessageRole.SYSTEM and "Memory" in (m.content[0].text or "")
            ]
            
            if not memory_messages:
                self.logger.warning("No memory context available for personalized response")
            
            self.logger.info(
                f"Memory context assembled: {len(memory_context)} total messages, "
                f"memory count: {len(memory_messages)}"
            )
            
            # Use chat profile for personalized interactions
            if not state.user_config:
                raise ValueError("User configuration required for memory workflow")
            
            profile_id = "memory_chat_profile"
            
            response = await self._mock_pipeline_run(memory_context, profile_id)
            
            if response.message:
                response_message = LangChainMessage(
                    content=response.message,
                    type="ai"
                )
                state.messages.append(response_message)
                
                self.logger.info("Memory-focused response generated successfully")
            
            return state
            
        except Exception as e:
            self.logger.error(f"Memory workflow failed: {e}", exc_info=True)
            state.error_details.append(f"Memory node error: {str(e)}")
            return state
    
    async def token_optimized_node(self, state: WorkflowState) -> WorkflowState:
        """
        Token-optimized workflow node for resource-constrained scenarios.
        
        This demonstrates context assembly with strict token budgeting
        for scenarios with limited model context capacity.
        """
        try:
            self.logger.info("Assembling token-optimized context")
            
            # Strict token budgeting for resource constraints
            optimized_context = assemble_context_messages(
                state,
                max_context_tokens=2000,      # Strict limit for smaller models
                include_search_results=True,  # Keep essential external context
                include_memories=False,       # Skip memories to save tokens
                include_summaries=True        # Keep summaries for continuity
            )
            
            estimated_tokens = sum(
                len((m.content[0].text or "")) // 4 
                for m in optimized_context
            )
            
            self.logger.info(
                f"Token-optimized context assembled: {len(optimized_context)} total messages, "
                f"estimated tokens: {estimated_tokens}"
            )
            
            # Use lightweight model profile if available
            if not state.user_config:
                raise ValueError("User configuration required for optimized workflow")
            
            profile_id = "lightweight_profile"
            
            response = await self._mock_pipeline_run(optimized_context, profile_id)
            
            if response.message:
                response_message = LangChainMessage(
                    content=response.message,
                    type="ai"
                )
                state.messages.append(response_message)
                
                self.logger.info("Token-optimized response generated successfully")
            
            return state
            
        except Exception as e:
            self.logger.error(f"Token-optimized workflow failed: {e}", exc_info=True)
            state.error_details.append(f"Optimized node error: {str(e)}")
            return state


def create_context_aware_nodes():
    """
    Factory function to create context-aware workflow nodes.
    
    Returns:
        Dictionary of workflow node functions ready for LangGraph integration
    """
    node_instance = ContextAwareWorkflowNode("context_aware")
    
    return {
        "chat_node": node_instance.chat_node,
        "research_node": node_instance.research_node,
        "memory_node": node_instance.memory_focused_node,
        "optimized_node": node_instance.token_optimized_node,
    }


# Example usage in LangGraph workflow definition
async def example_workflow_integration():
    """
    Example of integrating context-aware nodes into a LangGraph workflow.
    """
    from langgraph.graph import StateGraph
    
    # Create workflow nodes
    nodes = create_context_aware_nodes()
    
    # Build LangGraph workflow
    workflow = StateGraph(WorkflowState)
    
    # Add context-aware nodes
    workflow.add_node("chat", nodes["chat_node"])
    workflow.add_node("research", nodes["research_node"])
    workflow.add_node("memory", nodes["memory_node"])
    workflow.add_node("optimized", nodes["optimized_node"])
    
    # Add edges based on routing logic
    workflow.add_edge("chat", "memory")
    workflow.add_edge("research", "optimized")
    
    # Set entry point
    workflow.set_entry_point("chat")
    
    # Compile workflow
    app = workflow.compile()
    
    return app


if __name__ == "__main__":
    # Example of running a context-aware workflow
    print("Context-aware workflow node examples created successfully!")
    print("\nKey benefits:")
    print("✅ Automatic context extension architecture integration")
    print("✅ Flexible context component selection")
    print("✅ Token budgeting for resource optimization")
    print("✅ Proper error handling and logging")
    print("✅ LangGraph-compatible state management")