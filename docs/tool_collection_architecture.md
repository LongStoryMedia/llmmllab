## Updated Master Workflow Architecture

### Flow Overview

The master workflow now includes dedicated tool collection, ensuring tools are available for all subgraphs:

```
Input → Intent Analysis → Tool Collection → Router → Subgraph(s) → Coordinator → Output
```

### Node Responsibilities

#### 1. Intent Analysis Node
- Analyzes user input to determine intent and complexity
- Provides routing intelligence for workflow selection
- **Output**: `state.intent_classification` with intent analysis results

#### 2. Tool Collection Node (**NEW**)
- Collects available tools based on intent analysis results
- Uses ToolRegistry to gather both static and dynamic tools
- Replaces tool collection logic removed from core service
- **Input**: `state.intent_classification` from intent analysis
- **Output**: `state.required_tools` with collected tools

#### 3. Router Node
- Determines execution strategy (single/parallel/series)
- Routes to appropriate subgraph(s) based on intent or explicit workflow type
- **Input**: Intent and tools from previous nodes
- **Output**: Routing decisions in state

#### 4. Subgraph Nodes
- Individual workflows (Chat, Research, Creative, Multi-Agent) as compiled subgraphs
- **Access**: All collected tools via `state.required_tools`
- **Execution**: Isolated contexts with shared state

#### 5. Coordinator Node
- Orchestrates complex execution strategies
- Handles result aggregation and final response formatting
- **Input**: Results from executed subgraph(s)
- **Output**: Final processed response

### Tool Collection Implementation

```python
def _create_tool_collection_node(self, user_id: str):
    """Create tool collection node that gathers available tools based on intent analysis."""
    
    async def collect_tools(state):
        # Get intent from previous node
        intent = getattr(state, 'intent_classification', None)
        
        # Initialize tool registry and collect tools
        tool_registry = ToolRegistry()
        tools = await tool_registry.get_tools_for_context(intent, user_id)
        
        # Store tools in state for subgraphs to use
        state.required_tools = tools
        
        return state
```

### Key Features

#### Comprehensive Tool Collection
- **Static Tools**: Standard tools based on intent (web search, memory retrieval, etc.)
- **Dynamic Tools**: Generated or retrieved tools for specialized needs
- **Smart Selection**: ToolRegistry determines relevant tools based on intent analysis

#### Robust Error Handling
- Graceful degradation with empty tool list on errors
- Comprehensive logging for debugging and monitoring
- Continues workflow execution even if tool collection fails

#### State Management
- Uses existing `WorkflowState.required_tools` field
- Tools available to all subsequent nodes and subgraphs
- Maintains consistency across the entire workflow execution

### Benefits

1. **Restored Functionality**: Replaces tool collection logic removed from core service
2. **Separation of Concerns**: Dedicated node for tool-specific logic
3. **Shared Access**: All subgraphs can access collected tools via shared state
4. **Maintainable Architecture**: Clear, single-purpose node with well-defined responsibilities
5. **Extensible Design**: Easy to enhance tool collection logic without affecting other components

### Integration with Existing Systems

The tool collection node seamlessly integrates with:
- **ToolRegistry**: Leverages existing tool management infrastructure
- **Intent Classification**: Uses intent analysis results for smart tool selection
- **Subgraph Execution**: Provides tools to all workflow types
- **Error Handling**: Maintains workflow stability with graceful error recovery

This architecture ensures that tool collection is handled consistently and efficiently across all workflow types while maintaining the clean separation of concerns established in the intelligent routing system.