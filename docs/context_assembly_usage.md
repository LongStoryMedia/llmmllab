# Context Assembly Utility Usage Guide

## Overview

The `assemble_context_messages()` utility function in `composer/graph/state.py` provides a standardized way to construct Message objects from WorkflowState for pipeline consumption. This function implements the three-pronged context extension architecture described in the [Context Extension documentation](context_extension.md).

## Function Signature

```python
def assemble_context_messages(
    state: WorkflowState,
    max_context_tokens: Optional[int] = None,
    include_search_results: bool = True,
    include_memories: bool = True,
    include_summaries: bool = True,
) -> List[Message]:
```

## Parameters

- **state**: The WorkflowState containing all context components
- **max_context_tokens**: Optional token budget for context limiting
- **include_search_results**: Whether to include external search results (External Search RAG)
- **include_memories**: Whether to include retrieved memories (Memory Search RAG)
- **include_summaries**: Whether to include conversation summaries (In-Context Summarization)

## Usage Examples

### Basic Usage

```python
from composer.graph.state import WorkflowState, assemble_context_messages
from models import Message

# Assume you have a populated WorkflowState
state: WorkflowState = get_workflow_state()

# Assemble complete context with all strategies
context_messages = assemble_context_messages(state)

# Send to pipeline
pipeline_response = await run_pipeline(context_messages, pipeline)
```

### Memory-Only Context

```python
# Include only retrieved memories, exclude search results and summaries
memory_context = assemble_context_messages(
    state,
    include_search_results=False,
    include_summaries=False,
    include_memories=True
)
```

### Token-Budgeted Context

```python
# Limit context to approximately 4000 tokens
budgeted_context = assemble_context_messages(
    state,
    max_context_tokens=4000
)
```

### Selective Context for Specific Use Cases

```python
# For a search-focused workflow
search_context = assemble_context_messages(
    state,
    include_search_results=True,
    include_memories=False,
    include_summaries=True,
    max_context_tokens=2000
)

# For a memory-intensive conversation
memory_context = assemble_context_messages(
    state,
    include_search_results=False,
    include_memories=True,
    include_summaries=True
)
```

## Context Assembly Order

The function assembles context in the following priority order:

1. **Conversation Summaries** (Hierarchical Summarization)
   - Higher-level summaries first (broader context)
   - Formatted as system messages with level indicators

2. **Retrieved Memories** (Memory Search RAG)
   - Sorted by similarity score (most relevant first)
   - Limited to top 5 memories
   - Formatted as system messages with similarity scores

3. **External Search Results** (External Search RAG)
   - Consolidated search results as single system message
   - Only included if search results are available in state

4. **Recent Conversation Messages**
   - Converted from LangChainMessage to Message format
   - Maintains chronological order
   - Preserves original message roles

## Integration with Pipelines

### Standard Pipeline Integration

```python
async def run_chat_workflow(state: WorkflowState) -> WorkflowState:
    """Example workflow node using context assembly."""
    
    # Assemble context following context extension architecture
    context_messages = assemble_context_messages(
        state,
        max_context_tokens=8000,  # Adjust based on model capacity
        include_search_results=True,
        include_memories=True,
        include_summaries=True
    )
    
    # Get user configuration and model profile
    user_config = state.user_config
    profile = await get_model_profile(user_config.model_profiles.chat_profile_id)
    
    # Create pipeline and run with assembled context
    with pipeline_factory.pipeline(profile, str) as pipeline:
        response = await run_pipeline(context_messages, pipeline)
    
    # Add response to state messages
    if response.message:
        state.messages.append(response.message)
    
    return state
```

### Specialized Workflow Integration

```python
async def run_research_workflow(state: WorkflowState) -> WorkflowState:
    """Research workflow emphasizing external search and memories."""
    
    # Prioritize search results and memories for research tasks
    research_context = assemble_context_messages(
        state,
        max_context_tokens=12000,  # Larger context for research
        include_search_results=True,  # Critical for research
        include_memories=True,       # Historical research context
        include_summaries=False      # Skip summaries for detailed research
    )
    
    # Run research-specific pipeline
    profile = await get_model_profile(user_config.model_profiles.research_profile_id)
    with pipeline_factory.pipeline(profile, str) as pipeline:
        response = await run_pipeline(research_context, pipeline)
    
    return state
```

## Error Handling

The function includes robust error handling for missing or malformed state components:

```python
# Safe handling of missing components
try:
    context_messages = assemble_context_messages(state)
except Exception as e:
    logger.error(f"Context assembly failed: {e}")
    # Fallback to basic message context
    context_messages = [msg for msg in state.messages if isinstance(msg, Message)]
```

## Performance Considerations

### Token Budgeting

- Use `max_context_tokens` to prevent context overflow
- Token estimation uses 4 characters ≈ 1 token approximation
- Recent messages are prioritized when budget is exceeded

### Memory Optimization

- Memory retrieval is limited to top 5 by similarity
- Summaries are sorted by level for optimal context hierarchy
- External search results are consolidated into single message

### Selective Inclusion

- Use boolean flags to exclude unnecessary context components
- Reduces token usage and improves response relevance
- Customize based on workflow requirements

## Best Practices

1. **Always use this function** when sending messages to pipelines
2. **Set appropriate token limits** based on model context windows
3. **Customize inclusion flags** based on workflow needs
4. **Monitor context token usage** to optimize performance
5. **Handle exceptions gracefully** with fallback context strategies

## Architecture Compliance

This utility function ensures compliance with the context extension architecture by:

- **External Search RAG**: Including search results as system messages
- **Memory Search RAG**: Including similarity-ranked memories
- **In-Context Summarization**: Including hierarchical conversation summaries
- **Token Management**: Providing budgeting to prevent context overflow
- **Consistent Formatting**: Standardizing message formats for pipeline consumption

This approach ensures that every pipeline invocation has access to the full context extension capabilities while maintaining performance and relevance.
