"""
Memory workflow implementation using LangGraph.
Orchestrates embedding generation, memory retrieval, and storage operations.
"""

from typing import Optional

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from composer.graph.state import WorkflowState
from composer.nodes.memory import EmbeddingNode
from composer.nodes.memory import MemoryNode
from composer.monitoring.logging import composer_logger


async def build_memory_workflow(
    user_id: str, 
    pipeline_factory, 
    store_memories: bool = True
) -> CompiledStateGraph:
    """
    Build memory workflow subgraph for embedding generation and memory operations.
    
    This workflow:
    1. Generates embeddings for query text
    2. Searches for similar memories 
    3. Optionally stores new memories
    4. Augments state with memory context
    
    Args:
        user_id: User identifier for configuration
        pipeline_factory: Factory for creating embedding pipelines
        store_memories: Whether to store new memories after processing
        
    Returns:
        Compiled memory workflow graph
    """
    logger = composer_logger.logger.bind(component="MemoryWorkflow")
    
    try:
        logger.info(
            "Building memory workflow",
            user_id=user_id,
            store_memories=store_memories
        )

        # Initialize nodes
        embedding_node = EmbeddingNode(pipeline_factory)
        memory_node = MemoryNode()

        # Create workflow graph
        workflow = StateGraph(WorkflowState)

        # Add nodes to workflow
        workflow.add_node("generate_embeddings", embedding_node.generate_query_embeddings)
        workflow.add_node("retrieve_memories", memory_node)
        
        if store_memories:
            workflow.add_node("store_memories", memory_node.store_memories)

        # Define workflow edges
        workflow.set_entry_point("generate_embeddings")
        workflow.add_edge("generate_embeddings", "retrieve_memories")
        
        if store_memories:
            workflow.add_edge("retrieve_memories", "store_memories")
            workflow.add_edge("store_memories", END)
        else:
            workflow.add_edge("retrieve_memories", END)

        # Compile workflow
        compiled_workflow = workflow.compile()
        
        logger.info(
            "Memory workflow compiled successfully",
            user_id=user_id,
            nodes=["generate_embeddings", "retrieve_memories"] + (["store_memories"] if store_memories else [])
        )

        return compiled_workflow

    except Exception as e:
        logger.error(
            "Memory workflow compilation failed",
            user_id=user_id,
            error=str(e)
        )
        raise


async def build_embedding_only_workflow(
    user_id: str,
    pipeline_factory,
    model_name: Optional[str] = None
) -> CompiledStateGraph:
    """
    Build embedding-only workflow for generating embeddings without memory operations.
    
    Args:
        user_id: User identifier
        pipeline_factory: Factory for creating embedding pipelines
        model_name: Optional specific embedding model
        
    Returns:
        Compiled embedding workflow graph
    """
    logger = composer_logger.logger.bind(component="EmbeddingWorkflow")
    
    try:
        logger.info(
            "Building embedding-only workflow",
            user_id=user_id,
            model_name=model_name
        )

        # Initialize embedding node
        embedding_node = EmbeddingNode(pipeline_factory, model_name)

        # Create simple workflow graph
        workflow = StateGraph(WorkflowState)
        workflow.add_node("generate_embeddings", embedding_node)
        workflow.set_entry_point("generate_embeddings")
        workflow.add_edge("generate_embeddings", END)

        # Compile workflow
        compiled_workflow = workflow.compile()
        
        logger.info(
            "Embedding workflow compiled successfully",
            user_id=user_id,
            model_name=model_name
        )

        return compiled_workflow

    except Exception as e:
        logger.error(
            "Embedding workflow compilation failed",
            user_id=user_id,
            error=str(e)
        )
        raise


def should_enable_memory_workflow(state: WorkflowState) -> bool:
    """
    Determine if memory workflow should be enabled based on state and user config.
    
    Args:
        state: Current workflow state
        
    Returns:
        True if memory workflow should be enabled
    """
    try:
        # Check if user has memory enabled in their configuration
        user_id = getattr(state, "user_id", None)
        if not user_id:
            return False

        # Check state metadata for memory preferences
        memory_enabled = state.execution_metadata.get("memory_enabled", True)
        
        # Check if we have messages to process
        has_messages = bool(state.messages)
        
        return memory_enabled and has_messages

    except Exception:
        # Default to disabled if we can't determine
        return False


def create_memory_routing_condition():
    """
    Create a routing condition function for memory workflow integration.
    
    Returns:
        Function that determines if memory workflow should be used
    """
    async def should_use_memory(state: WorkflowState) -> str:
        """Determine routing for memory workflow."""
        if should_enable_memory_workflow(state):
            return "memory_workflow"
        else:
            return "skip_memory"
    
    return should_use_memory


# Export key functions
__all__ = [
    "build_memory_workflow",
    "build_embedding_only_workflow", 
    "should_enable_memory_workflow",
    "create_memory_routing_condition"
]