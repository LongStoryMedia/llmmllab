# LangChain/LangGraph Migration Plan

## Current Issues

Our current implementation is fighting against LangGraph best practices:

1. **Using deprecated `InjectedState` instead of `ToolRuntime`**
2. **Complex schema filtering that shouldn't be necessary**
3. **Manual Command content extraction that defeats the purpose**
4. **Creating fake `WorkflowState` instead of injecting proper subgraph state**
5. **Not leveraging built-in middleware for agent loops**

## Migration Path

### Phase 1: Tool Runtime Migration (Immediate)

**Current Pattern:**
```python
async def web_search(
    query: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[WorkflowState, InjectedState],
) -> Command:
```

**New Pattern:**
```python
from langchain_core.tools import ToolRuntime

async def web_search(
    query: str,
    runtime: ToolRuntime[ToolsState],
) -> Command:
    # Access state via runtime.state
    # Access tool_call_id via runtime.tool_call_id
    # ToolRuntime parameters are NOT visible to the model
```

**Benefits:**
- No schema filtering needed
- Clean parameter injection
- Proper LangGraph integration
- Command objects work as intended

### Phase 2: Subgraph State Injection (Immediate)

**Current:** Inject fake `WorkflowState` into tools
**New:** Inject proper `ToolsState` from subgraph

```python
# In tools_agent.py subgraph
class ToolsState(TypedDict):
    messages: List[BaseMessage]
    user_id: str
    conversation_id: int
    web_search_config: Dict[str, Any]
    # ... other tool-specific config

# Tools access this clean state instead of massive WorkflowState
```

### Phase 3: Built-in Middleware (Near-term)

**Current:** Manual agent cycling with conditional edges
**New:** LangGraph built-in middleware for agent loops

```python
from langgraph.middleware import AgentLoopMiddleware

# Use built-in middleware for agent->tool cycling
# Much cleaner than manual conditional routing
```

### Phase 4: Memory Store Migration (Medium-term)

**Current:** Custom memory storage in database
**New:** LangGraph built-in memory stores

- **Persistent Memory**: https://docs.langchain.com/oss/python/langgraph/persistence
- **Long-term Memory**: https://docs.langchain.com/oss/python/langchain/long-term-memory
- **Time Travel**: https://docs.langchain.com/oss/python/langgraph/use-time-travel

### Phase 5: MCP Server Integration (Future)

**Goal:** Better tool distribution via Model Context Protocol
- **MCP Docs**: https://docs.langchain.com/oss/python/langchain/mcp
- Distribute tools across services
- Better scalability and separation of concerns

## Implementation Steps

### Step 1: Update Web Search Tool
```python
from langchain_core.tools import ToolRuntime

@tool
async def web_search(
    query: str,
    runtime: ToolRuntime[ToolsState],
) -> Command:
    """Search the web for information."""
    state = runtime.state
    tool_call_id = runtime.tool_call_id
    
    # Use state.web_search_config directly
    # Return Command with updates to ToolsState
    return Command(update={
        "web_search_results": results,
        "messages": [ToolMessage(content, tool_call_id=tool_call_id)]
    })
```

### Step 2: Remove Schema Filtering
- Delete `schema_filter.py` complexity
- Remove wrapper functions
- Let ToolRuntime handle parameter injection cleanly

### Step 3: Fix Command Handling
- Stop extracting content from Commands
- Let LangGraph handle ToolMessage creation
- Use Commands for proper state updates

### Step 4: Use Built-in Agent Patterns
- Replace manual conditional routing
- Use LangGraph's agent middleware
- Follow DeepAgents patterns for subgraphs

## Benefits of Migration

1. **Simpler Code**: Remove complex workarounds
2. **Better Performance**: Built-in optimizations
3. **Future-proof**: Following LangGraph evolution
4. **Better Debugging**: Standard patterns and tools
5. **Less Maintenance**: Fewer custom solutions

## Testing Strategy

1. **Tool by Tool**: Migrate one tool at a time
2. **Backward Compatibility**: Keep old patterns until migration complete
3. **E2E Testing**: Verify each step works
4. **Performance Testing**: Ensure no regressions

## Documentation

- Update tool documentation to reflect ToolRuntime usage
- Create migration guide for future tool development
- Document state injection patterns for subgraphs