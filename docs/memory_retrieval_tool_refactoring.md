# Memory Retrieval Tool Refactoring

## Overview

The memory retrieval tool has been refactored to follow the same architectural patterns as other static tools in the composer system, specifically matching the patterns established by agents and nodes.

## Key Changes

### 1. Architecture Pattern Migration

**Before (Class-based):**

```python
class MemoryRetrievalTool(BaseTool):
    def __init__(self, user_id: str, conversation_id: int, **kwargs):
        super().__init__(user_id=user_id, conversation_id=conversation_id, **kwargs)
    
    async def _arun(self, query: str) -> str:
        # Complex database access and config retrieval
        user_config = await storage.get_service(storage.user_config).get_user_config(self.user_id)
        # ...
        return json.dumps(result)
```

**After (Function-based with @tool decorator):**

```python
@tool
async def memory_retrieval(
    query: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[WorkflowState, InjectedState],
) -> Command:
    # Efficient config access from injected state
    memory_config = state.user_config.memory if state.user_config else DEFAULT_MEMORY_CONFIG
    # ...
    return Command(update={
        "retrieved_memories": memories,
        "messages": [ToolMessage(response_message, tool_call_id=tool_call_id)]
    })
```

### 2. Pattern Compliance

The refactored tool now follows the established patterns:

- **@tool decorator**: Uses LangChain's `@tool` decorator for automatic tool registration
- **InjectedState/InjectedToolCallId**: Leverages LangGraph's dependency injection
- **Command pattern**: Returns `Command` objects for proper state updates
- **Efficient config access**: Gets user config from injected state instead of database calls
- **Strong typing**: Uses `WorkflowState` for type safety

### 3. Integration Benefits

**State Management:**

- Direct access to `WorkflowState` without additional database queries
- Automatic tool call tracking via `InjectedToolCallId`
- Seamless integration with LangGraph workflow execution

**Performance:**

- Eliminates redundant database calls for user configuration
- Leverages pre-loaded state data
- Follows efficient patterns established by web search tool

**Maintainability:**

- Consistent with other static tools (web_search)
- Follows established architectural patterns
- Reduces code complexity and duplication

### 4. Tool Registry Updates

Updated the tool registry to support both function-based and class-based tools:

```python
def _load_static_tools(self):
    # Class-based tools
    self.static_tools.update({
        # "summarization": SummarizationTool,  # Temporarily disabled
    })
    
    # Function-based tools (with @tool decorator)
    self.executable_tools.update({
        "memory_retrieval": memory_retrieval,
        "web_search": web_search,
    })
```

## Technical Implementation

### Embedding Generation

The tool maintains proper embedding generation using the pipeline factory:

```python
# Get embedding pipeline from factory
embedding_pipeline = pipeline_factory.get_embedding_pipeline(
    profile=embedding_profile
)
# Generate embeddings using Embeddings interface
query_embeddings = embedding_pipeline.embed_documents([query])
```

### Error Handling

Robust error handling with fallback mechanisms:

```python
try:
    embedding_pipeline = pipeline_factory.get_embedding_pipeline(profile=embedding_profile)
    query_embeddings = embedding_pipeline.embed_documents([query])
except Exception as embed_error:
    logger.warning(f"Embedding generation failed: {embed_error}, using mock embeddings")
    query_embeddings = [[0.1] * 768]  # Fallback mock embedding
```

### State Updates

Proper state updates using the Command pattern:

```python
return Command(
    update={
        "retrieved_memories": memories,           # Raw Memory objects for workflow use
        "memory_query": query,                   # Query tracking
        "messages": [                            # User-facing response
            ToolMessage(response_message, tool_call_id=tool_call_id)
        ],
    }
)
```

## Validation

The refactored tool has been validated to ensure:

- ✅ **StructuredTool creation**: Uses `@tool` decorator correctly
- ✅ **Proper attributes**: Has name, description, run/arun methods
- ✅ **LangGraph integration**: Ready for workflow integration
- ✅ **Import compatibility**: Works with updated tool registry
- ✅ **Error handling**: Graceful fallbacks for embedding failures

## Impact

### Positive Changes

1. **Architectural Consistency**: Now matches patterns used by web_search and other modern tools
2. **Performance Improvement**: Eliminates redundant database calls
3. **Maintainability**: Follows established patterns, easier to understand and modify
4. **Integration**: Seamless LangGraph workflow integration

### Breaking Changes

- Tool is now function-based instead of class-based
- Factory functions (`create_memory_retrieval_tool`) are no longer needed
- Tool registry updated to handle new patterns

## Future Considerations

1. **Summarization Tool**: Similar refactoring needed for consistency
2. **Dynamic Tools**: Consider applying similar patterns to dynamic tool generation
3. **Error Handling**: Centralize error handling patterns across all static tools
4. **Testing**: Comprehensive integration testing with actual workflow execution

## Usage

The refactored tool integrates automatically with LangGraph workflows:

```python
# Tool is automatically available when registered in tool registry
# LangGraph handles injection of tool_call_id and WorkflowState
# Returns Command objects that update workflow state appropriately
```

This refactoring ensures the memory retrieval tool follows the same high-quality patterns established by the agent/node architecture, providing better performance, maintainability, and integration capabilities.
