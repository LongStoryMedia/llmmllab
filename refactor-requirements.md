# LLM ML Lab - Refactor Requirements

## High-Level Architecture Changes

### Component Responsibility Separation

The platform requires comprehensive architectural refactoring to establish clear component boundaries:

- **Runner**: Pure LLM interface (local/remote model execution, embeddings, streaming, grammar-constrained output)
- **Composer**: Workflow orchestration (LangGraph graphs, agent nodes, state management, multi-step processing)

### Current Architecture Issues

1. **Mixed Responsibilities**: Runner currently handles both LLM execution AND graph orchestration
2. **Tight Coupling**: LangGraph code embedded throughout runner pipelines via `create_graph()` methods
3. **Complex Inheritance**: `BaseLangGraphPipeline` and subclasses blur component boundaries
4. **Unclear Interfaces**: Difficult to understand what functionality belongs where

### Target Architecture

```
UI ← REST API ← Composer (LangGraph Orchestration) ← Runner (Simple LLM Interface)
                    ↑                                      ↑
              Agent Nodes & Graphs                  Pure Model Execution
              State Management                      Grammar Constraints  
              Multi-step Workflows                  Streaming & Embeddings
```

## Detailed Migration Strategy

### Phase 1: Runner Simplification & LangGraph Extraction

**Remove LangGraph Orchestration from Runner:**

Based on codebase analysis, the following files contain `create_graph()` methods that must be extracted:

1. **Base Pipeline Classes:**
   - `runner/pipelines/base.py` - Remove `LangGraphCapable` protocol and abstract `create_graph()` method
   - `runner/pipelines/base_langgraph.py` - **ENTIRE FILE** to be moved to composer (contains primary graph orchestration)

2. **Specific Pipeline Implementations:**
   - `runner/pipelines/txt2txt/llamachatsum.py` (lines 422-500) - Extract graph creation logic
   - `runner/pipelines/txt2txt/qwen3moe.py` (lines 300-400) - Extract multi-step orchestration  
   - `runner/pipelines/txt2txt/openai_gpt_oss.py` - Extract any graph orchestration patterns

3. **Pipeline Infrastructure:**
   - Remove LangGraph imports from all pipeline files
   - Eliminate `StateGraph`, `CompiledStateGraph`, `MemorySaver` dependencies
   - Remove graph compilation, timeout handling, and state management

**Runner Interface After Refactor:**
```python
# Clean, simple LLM execution functions
async def run_pipeline(
    model_profile: ModelProfile, 
    messages: List[Message], 
    tools: Optional[List[BaseTool]] = None,
    grammar: Optional[GrammarInput] = None
) -> ChatResponse

async def stream_pipeline(
    model_profile: ModelProfile, 
    messages: List[Message], 
    tools: Optional[List[BaseTool]] = None,
    grammar: Optional[GrammarInput] = None
) -> AsyncIterator[ChatResponse]

async def embed_pipeline(
    model_profile: ModelProfile, 
    input_text: str
) -> EmbeddingResponse
```

**Components to Remove from Runner:**
- `BaseLangGraphPipeline` class hierarchy
- All `create_graph()` method implementations
- LangGraph state management (`_create_graph_with_timeout`, circuit breaker logic)
- Graph compilation and execution infrastructure
- Agent node definitions and routing logic

### Phase 2: Composer Enhancement & Graph Migration

**Move Orchestration Infrastructure to Composer:**

1. **Migrate Base Graph Infrastructure:**
   - Move `runner/pipelines/base_langgraph.py` → `composer/orchestration/base_graph.py`
   - Adapt graph creation patterns for composer-centric architecture
   - Implement agent nodes that interface with simplified runner functions

2. **Graph Creation Migration:**
   - Extract graph creation logic from each pipeline implementation
   - Create pipeline-specific orchestrators in composer (e.g., `composer/orchestration/llamachatsum_orchestrator.py`)
   - Convert complex pipeline logic into agent node workflows

3. **State Management Enhancement:**
   - Centralize all workflow state management in composer
   - Implement persistent state for multi-step processes
   - Handle memory and context across agent node transitions

**Composer Responsibilities After Migration:**
- LangGraph StateGraph creation and compilation
- Agent node implementation for all complex workflows
- Multi-step processing orchestration (summarization, research, analysis)
- Workflow state persistence and memory management
- Tool integration through graph nodes
- Circuit breaker and timeout handling for complex workflows

### Phase 3: Agent Node Implementation

**Agent Node Pattern for LLM Interaction:**
```python
# Clean composer agent nodes calling simplified runner
async def llm_agent_node(state: WorkflowState) -> WorkflowState:
    """Agent node that calls runner for LLM execution."""
    
    # Call simplified runner interface
    response = await run_pipeline(
        model_profile=state.model_profile,
        messages=state.messages,
        tools=state.available_tools,
        grammar=state.output_grammar
    )
    
    # Update workflow state
    state.messages.append(response.message)
    state.last_response = response
    
    return state

async def streaming_agent_node(state: WorkflowState) -> WorkflowState:
    """Agent node for streaming LLM responses."""
    
    async for chunk in stream_pipeline(
        model_profile=state.model_profile,
        messages=state.messages,
        tools=state.available_tools,
        grammar=state.output_grammar
    ):
        # Handle streaming chunks in workflow
        await state.stream_handler(chunk)
    
    return state
```

**Multi-Step Workflow Example:**
```python
# Complex workflow orchestration in composer
def create_research_workflow() -> CompiledStateGraph:
    """Create multi-step research workflow using agent nodes."""
    
    workflow = StateGraph(ResearchState)
    
    # Agent nodes calling simplified runner functions
    workflow.add_node("analyze_query", analyze_query_node)
    workflow.add_node("search_web", web_search_node) 
    workflow.add_node("summarize", summarize_node)
    workflow.add_node("generate_response", llm_agent_node)
    
    # Workflow routing and orchestration
    workflow.add_edge(START, "analyze_query")
    workflow.add_conditional_edges("analyze_query", route_next_step)
    workflow.add_edge("search_web", "summarize")
    workflow.add_edge("summarize", "generate_response")
    workflow.add_edge("generate_response", END)
    
    return workflow.compile()
```

### Phase 4: Grammar Integration Preservation

**Maintain Grammar-Constrained Output:**
- Preserve all grammar generation utilities (`utils/grammar_generator.py`)
- Ensure grammar parameters flow through simplified runner interface
- Maintain Pydantic model to GBNF conversion functionality
- Keep structured output parsing and validation

**Grammar Flow After Refactor:**
```
Composer Workflow → Agent Node → Runner (with grammar) → Structured LLM Output → Agent Node → Workflow State
```

## Implementation Priority & Migration Order

### Priority 1 (Critical): Runner Simplification
1. **Extract `BaseLangGraphPipeline`**: Move entire class hierarchy to composer
2. **Remove `create_graph()` Methods**: Extract from all pipeline implementations
3. **Simplify Runner Interface**: Clean function signatures without LangGraph dependencies
4. **Update Pipeline Base Classes**: Remove LangGraph abstractions and complex inheritance

### Priority 2 (High): Composer Infrastructure  
1. **Migrate Graph Creation**: Move orchestration logic from runner to composer
2. **Implement Agent Nodes**: Create nodes that call simplified runner functions
3. **State Management**: Centralize workflow state in composer
4. **Tool Integration**: Route tool calls through agent nodes instead of pipeline methods

### Priority 3 (Medium): Integration & Testing
1. **Update Server Endpoints**: Modify to use composer for orchestration instead of runner graphs
2. **Preserve Grammar Support**: Ensure grammar constraints work through new architecture
3. **Performance Optimization**: Optimize agent node communication with runner
4. **Comprehensive Testing**: Validate all workflows through new architecture

### Priority 4 (Low): Advanced Features
1. **Enhanced Orchestration**: Add advanced composer features (parallel processing, conditional routing)
2. **Monitoring & Observability**: Add workflow monitoring and debugging capabilities
3. **Configuration Management**: Optimize configuration for new architecture

## Files Requiring Changes

### Runner Component (Simplification)
- `runner/pipelines/base.py` - Remove `LangGraphCapable` protocol and graph abstractions
- `runner/pipelines/base_langgraph.py` - **DELETE ENTIRELY** (move to composer)
- `runner/pipelines/txt2txt/llamachatsum.py` - Extract graph creation (lines 422-500)
- `runner/pipelines/txt2txt/qwen3moe.py` - Extract multi-step orchestration (lines 300-400)  
- `runner/pipelines/txt2txt/openai_gpt_oss.py` - Remove any graph orchestration patterns
- `runner/pipelines/llamacpp/base_llamacpp.py` - Preserve grammar integration, remove graph logic
- `runner/__init__.py` - Simplify exports to basic LLM execution functions

### Composer Component (Enhancement)
- `composer/orchestration/` - **NEW DIRECTORY** for graph creation and management
- `composer/orchestration/base_graph.py` - **NEW FILE** (migrated from runner base_langgraph.py)
- `composer/orchestration/llamachatsum_orchestrator.py` - **NEW FILE** for chat summarization workflows
- `composer/orchestration/qwen3moe_orchestrator.py` - **NEW FILE** for multi-step processing
- `composer/agents/` - **NEW DIRECTORY** for agent node implementations
- `composer/agents/llm_agent.py` - **NEW FILE** for basic LLM interaction nodes
- `composer/agents/tool_agent.py` - **NEW FILE** for tool calling agent nodes
- `composer/__init__.py` - Export workflow creation and execution functions

### Integration Layer
- `server/` - Update endpoints to use composer for workflow orchestration
- Update tool integration to work through composer agent nodes
- Modify configuration to support new architecture boundaries
- Update documentation to reflect new component responsibilities

### Shared Components (Preserved)
- `utils/grammar_generator.py` - **PRESERVE** all grammar functionality
- `models/` - **PRESERVE** all Pydantic models and type definitions
- `db/` - **PRESERVE** all database interfaces and storage
- Schema definitions - **PRESERVE** all YAML schemas and generated models

## Success Criteria

1. **Clean Separation**: Runner contains no LangGraph imports or graph creation logic
2. **Simple Interface**: Runner provides basic `run_pipeline`, `stream_pipeline`, `embed_pipeline` functions
3. **Comprehensive Orchestration**: Composer handles all complex workflows through agent nodes
4. **Grammar Preservation**: Grammar-constrained output continues to work seamlessly
5. **Performance Maintained**: No performance degradation from architectural changes
6. **Test Coverage**: All existing functionality validated through new architecture