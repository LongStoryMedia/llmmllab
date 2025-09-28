# Comprehensive Requirements Document: Refactoring Agentic System to LangGraph-Based Architecture

## Executive Summary

This document outlines a complete refactoring strategy to transform the current agentic system from a mixed implementation with manual orchestration into a unified LangGraph-based architecture. The refactoring introduces a new "Composer" component that centralizes graph construction, tool management, and pipeline orchestration while maintaining backward compatibility and system flexibility.

## 1. Current State Analysis

### 1.1 Architecture Problems Identified

- **Fragmented Orchestration**: Manual coordination between server components for operations like title generation, summarization, and memory retrieval
- **Tool Management Complexity**: Dynamic tools built in server component with no centralized management
- **Pipeline Coupling**: Direct coupling between server and runner through factory pattern
- **Inconsistent State Management**: Mixed approaches between LangGraph and manual state handling
- **Resource Management Issues**: Memory cleanup and pipeline lifecycle management scattered across components
- **Error Handling Gaps**: Inconsistent error propagation and recovery mechanisms

### 1.2 Current Component Responsibilities

- **Server**: HTTP request handling, user configuration, model profiles, manual orchestration
- **Runner**: Pipeline factory, execution functions (stream/run/embed), resource management
- **Evaluation**: Model evaluation (currently isolated)
- **Tools**: Mixed between server (dynamic generation) and inline implementations

### 1.3 Key Pain Points

- Circuit breaker configuration flows through multiple layers without clear ownership
- Tool generation logic embedded in server rather than dedicated component
- RAG operations manually coordinated in ConversationContext
- No clear separation between orchestration logic and execution logic
- Embedding operations treated separately from other pipeline operations

## 2. Target Architecture

### 2.1 Component Structure

```
┌─────────────────────────────────────────────────────┐
│                   Server Layer                       │
│  (HTTP handlers, auth, request/response mapping)    │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│                 Composer Layer                       │
│  (Graph construction, tool management, orchestration)│
│  ┌────────────┐ ┌────────────┐ ┌─────────────┐     │
│  │Graph       │ │Tool        │ │State        │     │
│  │Builder     │ │Registry    │ │Manager      │     │
│  └────────────┘ └────────────┘ └─────────────┘     │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│                  Runner Layer                        │
│  (Pipeline factory, execution, resource management)  │
└──────────────────────────────────────────────────────┘
```

### 2.2 Composer Component Design

The Composer acts as the orchestration layer that:

- Constructs LangGraph workflows based on conversation context
- Manages tool lifecycle (static and dynamic)
- Coordinates pipeline execution through graph nodes
- Handles state management and checkpointing

## 3. Detailed Requirements

### 3.1 Composer Component Requirements

#### 3.1.1 Core Structure

```python
# composer/__init__.py
class ComposerService:
    """Main composer service coordinating all subcomponents"""
    
    def __init__(self):
        self.graph_builder = GraphBuilder()
        self.tool_registry = ToolRegistry()
        self.state_manager = StateManager()
        self.workflow_cache = WorkflowCache()
    
    async def compose_workflow(
        self,
        conversation_ctx: ConversationContext,
        workflow_type: WorkflowType
    ) -> CompiledGraph:
        """Main entry point for workflow composition"""
        pass
```

### 3.1.2 Graph Builder Module

```python
# composer/graph_builder.py
class GraphBuilder:
    """Constructs LangGraph workflows dynamically"""
    
    def build_chat_workflow(
        self,
        tools: List[BaseTool],
        config: WorkflowConfig
    ) -> StateGraph:
        """Build standard chat completion workflow"""
        
    def build_rag_workflow(
        self,
        retrieval_config: RetrievalConfig,
        tools: List[BaseTool]
    ) -> StateGraph:
        """Build RAG-enhanced workflow"""
        
    def build_multi_agent_workflow(
        self,
        agents: List[AgentConfig]
    ) -> StateGraph:
        """Build multi-agent workflow"""
```

### 3.1.3 Tool Registry Module

```python
# composer/tools/registry.py
class ToolRegistry:
    """Centralized tool management"""
    
    def __init__(self):
        self.static_tools = {}
        self.dynamic_tools = {}
        self.tool_generators = {}
    
    def register_static_tool(self, tool: BaseTool) -> None:
        """Register a static tool"""
        
    async def generate_dynamic_tool(
        self,
        context: ConversationContext,
        spec: DynamicToolSpec
    ) -> BaseTool:
        """Generate and register dynamic tool"""
        
    def get_tools_for_context(
        self,
        context: ConversationContext
    ) -> List[BaseTool]:
        """Get all applicable tools for context"""
```

### 3.1.4 State Manager Module

```python
# composer/state/manager.py
class StateManager:
    """Manages workflow state and checkpointing"""
    
    def create_initial_state(
        self,
        messages: List[Message],
        context: Dict[str, Any]
    ) -> WorkflowState:
        """Create initial workflow state"""
        
    def checkpoint_state(
        self,
        state: WorkflowState,
        checkpoint_id: str
    ) -> None:
        """Save state checkpoint"""
        
    def restore_state(
        self,
        checkpoint_id: str
    ) -> WorkflowState:
        """Restore from checkpoint"""
```

### 3.2 Node Definitions

#### 3.2.1 Standard Nodes

```python
# composer/nodes/standard.py

class PipelineNode:
    """Wraps pipeline execution in a graph node"""
    
    def __init__(self, pipeline_factory, profile_selector):
        self.pipeline_factory = pipeline_factory
        self.profile_selector = profile_selector
    
    async def __call__(self, state: WorkflowState) -> WorkflowState:
        profile = self.profile_selector(state)
        pipeline = self.pipeline_factory.get_pipeline(profile, ...)
        result = await run_pipeline(state.messages, pipeline)
        state.messages.append(result.message)
        return state

class ToolExecutorNode:
    """Executes tool calls from previous node"""
    
    async def __call__(self, state: WorkflowState) -> WorkflowState:
        last_message = state.messages[-1]
        if hasattr(last_message, 'tool_calls'):
            results = await self.execute_tools(last_message.tool_calls)
            state.messages.extend(results)
        return state

class RAGNode:
    """Performs RAG operations"""
    
    async def __call__(self, state: WorkflowState) -> WorkflowState:
        embeddings = await self.create_embeddings(state.current_message)
        
        # Parallel RAG operations
        results = await asyncio.gather(
            self.memory_retrieval(embeddings),
            self.web_search(state.current_message),
            self.summarize(state.messages)
        )
        
        state.context.update({
            'memories': results[0],
            'search_results': results[1],
            'summary': results[2]
        })
        return state
```

#### 3.2.2 Specialized Nodes

```python
# composer/nodes/specialized.py

class TitleGenerationNode:
    """Generates conversation titles"""
    
    async def __call__(self, state: WorkflowState) -> WorkflowState:
        if not state.conversation.title or len(state.messages) == 1:
            profile = state.user_config.model_profiles.formatting_profile
            pipeline = self.pipeline_factory.get_pipeline(profile, str)
            title = await run_pipeline(
                state.messages + ["Generate title (max 5 words)"],
                pipeline
            )
            state.conversation.title = title
        return state

class DynamicToolGenerationNode:
    """Generates dynamic tools when needed"""
    
    async def __call__(self, state: WorkflowState) -> WorkflowState:
        analyzer = SmartIntentAnalyzer()
        analysis = analyzer.analyze_intent(state.current_message)
        
        if analysis.needs_dynamic_tool:
            tool = await self.tool_registry.generate_dynamic_tool(
                state.context, 
                analysis.tool_spec
            )
            state.available_tools.append(tool)
        return state
```

### 3.3 Workflow Definitions

#### 3.3.1 Standard Chat Workflow

```python
# composer/workflows/chat.py

def build_chat_workflow(config: ChatConfig) -> StateGraph:
    workflow = StateGraph(WorkflowState)
    
    # Add nodes
    workflow.add_node("rag_enrichment", RAGNode())
    workflow.add_node("dynamic_tools", DynamicToolGenerationNode())
    workflow.add_node("agent", PipelineNode(
        pipeline_factory,
        lambda s: s.user_config.model_profiles.primary_profile
    ))
    workflow.add_node("tools", ToolExecutorNode())
    
    # Define edges
    workflow.add_edge(START, "rag_enrichment")
    workflow.add_edge("rag_enrichment", "dynamic_tools")
    workflow.add_edge("dynamic_tools", "agent")
    
    # Conditional routing after agent
    def route_after_agent(state):
        if state.messages[-1].tool_calls:
            return "tools"
        return END
    
    workflow.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", END: END}
    )
    
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()
```

#### 3.3.2 Research Workflow

```python
# composer/workflows/research.py

def build_research_workflow(config: ResearchConfig) -> StateGraph:
    workflow = StateGraph(ResearchState)
    
    # Research-specific nodes
    workflow.add_node("query_expansion", QueryExpansionNode())
    workflow.add_node("parallel_search", ParallelSearchNode())
    workflow.add_node("source_validation", SourceValidationNode())
    workflow.add_node("synthesis", SynthesisNode())
    workflow.add_node("fact_check", FactCheckNode())
    
    # Linear flow for research
    workflow.add_edge(START, "query_expansion")
    workflow.add_edge("query_expansion", "parallel_search")
    workflow.add_edge("parallel_search", "source_validation")
    workflow.add_edge("source_validation", "synthesis")
    workflow.add_edge("synthesis", "fact_check")
    workflow.add_edge("fact_check", END)
    
    return workflow.compile()
```

### 3.4 Migration Strategy

#### 3.4.1 Phase 1: Foundation (Week 1-2)

1. **Create Composer structure**

```bash
composer/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── service.py
│   └── config.py
├── graph/
│   ├── __init__.py
│   ├── builder.py
│   └── cache.py
├── tools/
│   ├── __init__.py
│   ├── registry.py
│   ├── static/
│   └── dynamic/
├── nodes/
│   ├── __init__.py
│   ├── standard.py
│   ├── specialized.py
│   └── rag.py
├── state/
│   ├── __init__.py
│   └── manager.py
└── workflows/
    ├── __init__.py
    ├── chat.py
    ├── research.py
    └── multi_agent.py
```

2. **Extract tool logic from server**
   - Move `integration.py` → `composer/tools/integration.py`
   - Move `rag_tools.py` → `composer/tools/static/rag.py`
   - Move `dynamic_tool.py` → `composer/tools/dynamic/executor.py`

3. **Create base node implementations**
   - Implement `PipelineNode` wrapping existing pipeline execution
   - Implement `ToolExecutorNode` using existing tool execution logic
   - Create `RAGNode` from current `ConversationContext.process_rag_operations`

#### 3.4.2 Phase 2: Integration (Week 3-4)

1. **Create workflow builders**

```python
# composer/graph/builder.py
class GraphBuilder:
    def __init__(self, pipeline_factory, tool_registry):
        self.pipeline_factory = pipeline_factory
        self.tool_registry = tool_registry
    
    def build_from_context(
        self,
        conversation_ctx: ConversationContext
    ) -> CompiledGraph:
        # Determine workflow type from intent
        if conversation_ctx.intent.deep_research:
            return self.build_research_workflow(conversation_ctx)
        elif conversation_ctx.intent.image_generation:
            return self.build_creative_workflow(conversation_ctx)
        else:
            return self.build_chat_workflow(conversation_ctx)
```

2. **Update completion handler**

```python
# server/handlers/completion.py
async def agent_chat_completion(
    conversation_ctx: ConversationContext,
    background_tasks: BackgroundTasks
):
    # Old way (to be replaced)
    # tools = await get_tools(conversation_ctx)
    # pipeline = pipeline_factory.get_pipeline(...)
    # async for chunk in stream_pipeline(messages, pipeline, tools):
    
    # New way
    composer = ComposerService()
    workflow = await composer.compose_workflow(
        conversation_ctx,
        WorkflowType.CHAT
    )
    
    async for chunk in workflow.astream(
        initial_state,
        config={"configurable": {"thread_id": conversation_ctx.conversation.id}}
    ):
        yield serialize_to_json(chunk)
```

3. **Implement state management**

```python
# composer/state/manager.py
class WorkflowState(TypedDict):
    messages: List[Message]
    context: Dict[str, Any]
    available_tools: List[BaseTool]
    current_message: Optional[Message]
    conversation: Conversation
    user_config: UserConfig
    metadata: Dict[str, Any]
```

#### 3.4.3 Phase 3: Refactoring (Week 5-6)

1. **Eliminate manual orchestration**
   - Remove direct pipeline calls from `ConversationContext`
   - Move title generation to `TitleGenerationNode`
   - Move search formatting to `QueryFormattingNode`

2. **Centralize tool management**

```python
# composer/tools/registry.py
class ToolRegistry:
    def __init__(self):
        self._static_tools = {
            'web_search': WebSearchTool,
            'memory_retrieval': MemoryRetrievalTool,
            'summarization': SummarizationTool,
        }
        self._dynamic_cache = TTLCache(maxsize=100, ttl=3600)
    
    async def get_tools_for_workflow(
        self,
        context: ConversationContext,
        workflow_type: WorkflowType
    ) -> List[BaseTool]:
        tools = []
        
        # Add static tools
        for name, tool_class in self._static_tools.items():
            if self._should_include_tool(name, context):
                tools.append(tool_class(context))
        
        # Check for dynamic tools
        if self._needs_dynamic_tools(context):
            dynamic_tool = await self._generate_or_retrieve_dynamic_tool(context)
            if dynamic_tool:
                tools.append(dynamic_tool)
        
        return tools
```

3. **Implement workflow caching**

```python
# composer/graph/cache.py
class WorkflowCache:
    def __init__(self):
        self._cache = {}
        self._lock = asyncio.Lock()
    
    def get_cache_key(
        self,
        user_config: UserConfig,
        workflow_type: WorkflowType,
        tools: List[BaseTool]
    ) -> str:
        tool_signature = "|".join(sorted(t.name for t in tools))
        return f"{user_config.user_id}:{workflow_type}:{tool_signature}"
    
    async def get_or_create(
        self,
        key: str,
        builder_fn: Callable
    ) -> CompiledGraph:
        async with self._lock:
            if key not in self._cache:
                self._cache[key] = await builder_fn()
            return self._cache[key]
```

#### 3.4.4 Phase 4: Optimization (Week 7-8)

1. **Implement streaming optimizations**

```python
# composer/streaming/processor.py
class StreamProcessor:
    def __init__(self):
        self.buffer = []
        self.thinking_phase = True
    
    async def process_stream(
        self,
        workflow: CompiledGraph,
        initial_state: WorkflowState
    ) -> AsyncIterator[ChatResponse]:
        async for event in workflow.astream_events(
            initial_state,
            version="v2"
        ):
            if event["event"] == "on_chat_model_stream":
                chunk = self._process_chunk(event["data"])
                if chunk:
                    yield chunk
```

2. **Add circuit breaker integration**

```python
# composer/nodes/protected.py
class CircuitProtectedNode:
    def __init__(self, node, circuit_config: CircuitBreakerConfig):
        self.node = node
        self.circuit_config = circuit_config
        self.failure_count = 0
        self.last_failure_time = None
    
    async def __call__(self, state: WorkflowState) -> WorkflowState:
        if self._is_circuit_open():
            raise CircuitOpenError("Circuit breaker is open")
        
        try:
            result = await asyncio.wait_for(
                self.node(state),
                timeout=self.circuit_config.base_timeout
            )
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
```

3. **Implement multi-agent support**

```python
# composer/workflows/multi_agent.py
class AgentNode:
    def __init__(self, agent_config: AgentConfig):
        self.name = agent_config.name
        self.profile = agent_config.profile
        self.tools = agent_config.tools
    
    async def __call__(self, state: WorkflowState) -> WorkflowState:
        # Agent-specific logic
        agent_context = self._create_agent_context(state)
        pipeline = pipeline_factory.get_pipeline(self.profile, ChatResponse)
        
        response = await run_pipeline(
            agent_context.messages,
            pipeline,
            self.tools
        )
        
        state.agent_responses[self.name] = response
        return state

def build_multi_agent_workflow(agents: List[AgentConfig]) -> StateGraph:
    workflow = StateGraph(MultiAgentState)
    
    # Add supervisor
    workflow.add_node("supervisor", SupervisorNode())
    
    # Add agents
    for agent in agents:
        workflow.add_node(agent.name, AgentNode(agent))
    
    # Supervisor routes to agents
    def route_to_agent(state):
        return state.next_agent
    
    workflow.add_conditional_edges(
        "supervisor",
        route_to_agent,
        {agent.name: agent.name for agent in agents}
    )
    
    # Agents report back to supervisor
    for agent in agents:
        workflow.add_edge(agent.name, "supervisor")
    
    return workflow.compile()
```

### 3.5 Testing Requirements

#### 3.5.1 Unit Tests

```python
# tests/composer/test_nodes.py
async def test_pipeline_node():
    """Test pipeline node execution"""
    node = PipelineNode(mock_factory, lambda s: s.profile)
    state = WorkflowState(messages=[test_message])
    result = await node(state)
    assert len(result.messages) == 2

# tests/composer/test_workflows.py
async def test_chat_workflow():
    """Test complete chat workflow"""
    workflow = build_chat_workflow(ChatConfig())
    result = await workflow.ainvoke(initial_state)
    assert result.messages[-1].role == MessageRole.ASSISTANT
```

#### 3.5.2 Integration Tests

```python
# tests/integration/test_composer_integration.py
async def test_end_to_end_chat():
    """Test full chat flow through composer"""
    composer = ComposerService()
    context = create_test_context()
    
    workflow = await composer.compose_workflow(
        context,
        WorkflowType.CHAT
    )
    
    async for chunk in workflow.astream(initial_state):
        assert chunk is not None
```

#### 3.5.3 Performance Tests

```python
# tests/performance/test_workflow_performance.py
async def test_workflow_caching():
    """Test that workflows are properly cached"""
    composer = ComposerService()
    
    # First call should create workflow
    start = time.time()
    workflow1 = await composer.compose_workflow(context, WorkflowType.CHAT)
    creation_time = time.time() - start
    
    # Second call should use cache
    start = time.time()
    workflow2 = await composer.compose_workflow(context, WorkflowType.CHAT)
    cache_time = time.time() - start
    
    assert cache_time < creation_time / 10
    assert workflow1 is workflow2
```

### 3.6 Monitoring and Observability

#### 3.6.1 Metrics

```python
# composer/monitoring/metrics.py
class ComposerMetrics:
    workflow_creation_time = Histogram('composer_workflow_creation_seconds')
    node_execution_time = Histogram('composer_node_execution_seconds')
    tool_generation_count = Counter('composer_tool_generation_total')
    workflow_cache_hits = Counter('composer_cache_hits_total')
    workflow_cache_misses = Counter('composer_cache_misses_total')
```

#### 3.6.2 Logging

```python
# composer/monitoring/logging.py
class WorkflowLogger:
    def log_workflow_start(self, workflow_id: str, workflow_type: str):
        logger.info(f"Starting workflow", extra={
            "workflow_id": workflow_id,
            "workflow_type": workflow_type,
            "timestamp": datetime.now()
        })
    
    def log_node_execution(self, node_name: str, duration: float):
        logger.debug(f"Node executed", extra={
            "node": node_name,
            "duration_ms": duration * 1000
        })
```

### 3.7 Configuration Management

```python
# composer/config.py
@dataclass
class ComposerConfig:
    enable_workflow_caching: bool = True
    workflow_cache_ttl: int = 3600
    max_parallel_tools: int = 5
    enable_multi_agent: bool = False
    default_timeout: float = 60.0
    
    # Node-specific configs
    rag_config: RAGConfig = field(default_factory=RAGConfig)
    tool_config: ToolConfig = field(default_factory=ToolConfig)
    
    @classmethod
    def from_env(cls) -> 'ComposerConfig':
        """Load configuration from environment variables"""
        return cls(
            enable_workflow_caching=os.getenv('COMPOSER_ENABLE_CACHE', 'true').lower() == 'true',
            workflow_cache_ttl=int(os.getenv('COMPOSER_CACHE_TTL', '3600')),
            max_parallel_tools=int(os.getenv('COMPOSER_MAX_PARALLEL_TOOLS', '5'))
        )
```

### 3.8 Error Handling Strategy

```python
# composer/errors.py
class ComposerError(Exception):
    """Base exception for composer errors"""
    pass

class WorkflowConstructionError(ComposerError):
    """Failed to construct workflow"""
    pass

class NodeExecutionError(ComposerError):
    """Node execution failed"""
    def __init__(self, node_name: str, original_error: Exception):
        self.node_name = node_name
        self.original_error = original_error
        super().__init__(f"Node '{node_name}' failed: {original_error}")

class ToolGenerationError(ComposerError):
    """Failed to generate dynamic tool"""
    pass

# composer/core/error_handler.py
class ErrorHandler:
    def __init__(self, config: ErrorConfig):
        self.config = config
        self.error_counts = defaultdict(int)
    
    async def handle_node_error(
        self,
        error: Exception,
        node_name: str,
        state: WorkflowState
    ) -> WorkflowState:
        self.error_counts[node_name] += 1
        
        if self.error_counts[node_name] > self.config.max_retries:
            # Add error to state and continue
            state.errors.append({
                "node": node_name,
                "error": str(error),
                "timestamp": datetime.now()
            })
            return state
        
        # Retry with exponential backoff
        await asyncio.sleep(2 ** self.error_counts[node_name])
        raise RetryableError(f"Retrying {node_name}")
```

## 4. Implementation Checklist

### Phase 1: Foundation Setup

- [ ] Create composer directory structure
- [ ] Set up basic configuration management
- [ ] Extract tool logic from server component
- [ ] Create base node implementations
- [ ] Set up logging and monitoring framework

### Phase 2: Core Implementation

- [ ] Implement GraphBuilder with basic workflows
- [ ] Create ToolRegistry with static tool registration
- [ ] Implement StateManager with checkpointing
- [ ] Build standard nodes (Pipeline, Tool, RAG)
- [ ] Create workflow cache system

### Phase 3: Integration

- [ ] Update completion handler to use composer
- [ ] Migrate ConversationContext orchestration to nodes
- [ ] Implement dynamic tool generation in composer
- [ ] Create workflow selection logic
- [ ] Add circuit breaker protection to nodes

### Phase 4: Advanced Features

- [ ] Implement multi-agent workflow support
- [ ] Add research workflow with specialized nodes
- [ ] Create streaming optimizations
- [ ] Implement workflow composition from config
- [ ] Add performance monitoring and metrics

### Phase 5: Testing and Documentation

- [ ] Write comprehensive unit tests
- [ ] Create integration test suite
- [ ] Performance testing and optimization
- [ ] Document workflow creation process
- [ ] Create migration guide for existing code

### Phase 6: Deployment

- [ ] Gradual rollout with feature flags
- [ ] Monitor performance metrics
- [ ] Gather feedback and iterate
- [ ] Complete migration of legacy code
- [ ] Archive deprecated components

## 5. Success Criteria

### Code Quality

- Reduction in cyclomatic complexity by 40%
- Clear separation of concerns between components
- Consistent error handling across all workflows

### Performance

- Workflow creation time < 100ms with caching
- Node execution overhead < 5ms per node
- Memory usage reduced by 20% through better resource management

### Maintainability

- New workflows can be created in < 1 hour
- Tool additions require no server code changes
- Clear documentation for all components

### Reliability

- 99.9% uptime for composer service
- Graceful degradation when nodes fail
- Automatic recovery from transient errors

## 6. Risk Mitigation

### Backward Compatibility

- Maintain existing APIs during migration
- Feature flag for gradual rollout
- Comprehensive testing of legacy flows

### Performance Regression

- Benchmark critical paths before/after
- Profile node execution times
- Optimize hot paths identified in profiling

### Complexity Introduction

- Keep workflows simple initially
- Document patterns and anti-patterns
- Regular architecture reviews

This requirements document provides a comprehensive roadmap for refactoring your agentic system to a cleaner, more maintainable LangGraph-based architecture. The phased approach ensures minimal disruption while systematically addressing the current pain points and technical debt.
