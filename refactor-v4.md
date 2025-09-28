# Comprehensive Final Requirements Document: Refactoring Agentic System to LangGraph-Based Architecture

## Executive Summary

This final requirements document holistically integrates and consolidates all prior refactoring plans with meticulous inclusion of all content, emphasizing the fundamental architectural shift to a Composer-centric design. The Composer becomes the authoritative orchestrator and runtime environment for the agentic system, leveraging LangGraph v1 capabilities to deliver robust, scalable, and maintainable workflows encompassing adaptive retrieval, dynamic tool management, multi-agent orchestration, and precise streaming control.

The document captures all architectural motivations, detailed requirements, code examples, validation grounds, and strategic implementation phases to enable smooth migration and future system growth.

***

## 1. Current System Analysis

### 1.1 Architecture Challenges

- **Fragmented orchestration:** Multiple isolated server components manually coordinate functions like summarization, retrieval, tool management, leading to brittle systems.
- **Tool management sprawl:** Tools, especially dynamic tool instantiations, are decentralized, reducing reuse and introducing duplication.
- **Tight coupling:** Server and runner components are tightly bound through factories, restricting pipeline flexibility.
- **State and error handling gaps:** Mixed state management between manual and LangGraph approaches, inconsistent error propagation, and scattered cleanup logic.
- **Streaming constraints:** Only primary agent supports streaming; secondary agents and tool nodes lack streaming integration.
- **Recovery and resource management:** Lacking circuit breakers and robust failure handling harm system reliability.


### 1.2 Roles of Major Components

- **Server:** Handles HTTP, auth, user config, legacy orchestration currently scheduled for deprecation.
- **Composer:** Centralized orchestration, stateful workflow construction, execution, intent parsing, tool lifecycle, error management.
- **Runner:** Executes LangGraph compiled graphs with pipeline facilities for streaming or batch runs.
- **Tools:** Includes static tools (search, summarization) and dynamic tools derived from LLM-generated code.

***

## 2. Target Architecture

### 2.1 Composer: The Heart of the New Architecture

Central to the redesign is the **Composer component**, responsible for:

- **Graph construction & execution:** Intelligently builds LangGraph task graphs based on conversation context and dynamically selected tools.
- **Streaming orchestration:** Uses LangGraph's `astream_events` API to manage and route streaming events for primary chat interaction and control non-streaming responses from secondary nodes smoothly.
- **State management:** Maintains a unified, authoritative GraphState with full persistence, checkpoints, and seamless recovery.
- **Tool management:** Centralizes tool registration, dynamic generation and discovery, leveraging semantic search to maximize reuse and minimize redundancy.
- **Intent analysis:** Runs LLM-based intent classifiers early in workflows to set retrieval depth, determine toolsets, and drive conditional routing.
- **Error resiliency:** Coordinates circuit breaker protections, error handling, and retry policies at per-node granularity.
- **Multi-agent orchestration:** Implements cross-agent handoffs through LangGraph Command primitives for complex collaborative workflows.


### 2.2 High-Level System Layout

```
┌──────────────────────────────┐        ┌────────────────────────┐
│        Client/UI Layer       │◄──────►│       Server Layer     │
│ (Display, request routing,   │        │ (HTTP, auth, user mgmt)│
│  client-side rendering)      │        └────────────┬───────────┘
└─────────────┬────────────────┘                     │
              │                                      ▼
       ┌──────▼───────────────────────────────────────────┐
       │                   Composer Layer                 │
       │ (GraphBuilder, IntentAnalyzer, ToolRegistry,     │
       │ StateManager, Workflow Cache, Execution Engines) │
       └───────────────┬──────────────────────┬───────────┘
                       │                      │
           ┌───────────▼───────────┐  ┌───────▼───────────┐
           │      Runner Layer     │  │    Persistence    │
           │(LangGraph Execution,  │  │ (DB, storage for  │
           │  pipeline handlers)   │  │  durable state)   │
           └───────────────────────┘  └───────────────────┘
```

- **Composer** is the *authoritative runtime* for LangGraph execution control and context-aware orchestration—removing fragmented server logic.
- Implements robust state persistence, error handling, real-time streaming, intent-driven dynamic workflow selection (chat, research, multi-agent).
- Favors clean separation of concerns per LangGraph V1 principles.

### 2.3 Architectural Shift Validation (LangGraph V1-alpha)

- All complex graph execution is centralized in the Composer service, which leverages **Durable Execution**—state persistence and recovery in case of client disconnects.
- Composer hosts all state, node, and graph logic. Client interacts via a single async streaming endpoint (WebSocket/SSE), receiving event envelopes.
- Composer's modular structure includes agents, tool orchestrators, specialized RAG nodes, and multi-agent routing logic.

### 3.1 Graph State Model

All nodes communicate through a Pydantic GraphState model, enforcing authoritative context.
Mandatory fields—each with reducer:

| Field Name            | Type         | Reducer      | Purpose                                                |
|-----------------------|--------------|--------------|--------------------------------------------------------|
| messages              | List         | x.concat(y)  | Conversation history/final outputs/token streaming     |
| intent_classification | IntentSchema | y            | Output from LLM Intent Agent, drives workflow selection|
| required_tools        | List         | y            | All tools collected/generated for this run             |
| search_results        | str          | y            | Consolidated RAG synthesis output                      |
| rag_depth_config      | str          | y            | SHALLOW/DEEP; Flow driver for RAG routing              |
| progress_updates      | List         | x.concat(y)  | Granular progress signals; tool/crawl steps            |

(Full context model implemented in composer/state.py)

### 3.2 Composer API / Service

Central API for constructing workflows, caching compiled graphs, and setting up streaming:

```python
class ComposerService:
    def __init__(self):
        self.graph_builder = GraphBuilder()
        self.tool_registry = ToolRegistry()
        self.state_manager = StateManager()
        self.workflow_cache = WorkflowCache()

    async def compose_workflow(self, conversation_ctx, workflow_type):
        intent = await self.analyze_intent(conversation_ctx)
        tools = await self.tool_registry.get_tools_for_context(intent, conversation_ctx)
        config = conversation_ctx.user_config.get_workflow_config(workflow_type, intent)
        key = self.workflow_cache.get_cache_key(conversation_ctx.user_id, workflow_type, tools)
        workflow = await self.workflow_cache.get_or_create(key, 
            lambda: self.graph_builder.build_from_context(conversation_ctx, tools, config))
        return workflow

    async def analyze_intent(self, conversation_ctx):
        # Runs a specialized LLM agent to extract intent and operational config
        pass
```


### 3.3 Graph Builder

Builds dynamic LangGraph workflows (chat, research, multi-agent):

- Streams primary chat agent response with `stream=True`.
- Reads user config for retrieval depth, number of sources, and full-content retrieval.
- Adds appropriate nodes for RAG, tool invocation, synthesis, and completion.


### 3.4 Streaming Control

- Uses LangGraph's `astreamevents` API, focusing on `on_chat_model_stream` for token-level streaming from the primary agent.
- Secondary nodes emit streaming deltas and custom progress signals.
- Encodes events in JSON envelopes with type discriminators.
- Streaming pipeline designed for WebSocket/SSE endpoints.


### 3.5 Tool Management

Fuses static and dynamic tools into a unified registry.

- Dynamic tools are generated or retrieved on-demand via semantic similarity search with embeddings.
- Tools composed dynamically as LCEL RunnableSequences wrapped with `.as_tool`.
- Supports integrating corporate or partner tools via MCP (Model Context Protocol) adapter standard.
- Enables composing atomic tools into higher-level operations at runtime.


### 3.6 Multi-Agent Orchestration

- Agents coordinated via LangGraph commands.
- Supports multiplexed conversational agents with seamless handoffs and cross-agent context propagation.
- Supervisor node directs agent flow based on workflow logic and state.

***

## 4. Representative Code Examples

### 4.1 Composer Component Requirements

#### 4.1.1 Core Structure

```python
# composer/core/service.py
class ComposerService:
    """Main composer service coordinating graph construction and execution."""
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
        """Construct or retrieve a compiled graph for the given conversation."""
        # Determine tools and intent before building
        intent = await self._analyze_intent(conversation_ctx)
        tools = await self.tool_registry.get_tools_for_context(intent, conversation_ctx)
        # Incorporate conversation-specific config (e.g. search complexity)
        config = conversation_ctx.user_config.get_workflow_config(workflow_type, intent)
        # Use cache if available
        key = self.workflow_cache.get_cache_key(conversation_ctx.user_config, workflow_type, tools)
        builder_fn = lambda: self.graph_builder.build_from_context(conversation_ctx, tools, config)
        workflow = await self.workflow_cache.get_or_create(key, builder_fn)
        return workflow

    async def _analyze_intent(self, conversation_ctx):
        # Use an LLM-based intent agent to label the conversation (deep_research, image_gen, etc.)
        # (Implementation detail omitted)
        pass
```

**Highlights:**  

- Uses a **LangGraph CompiledGraph** returned to the caller, which supports both streaming and batch execution.  
- **Intent Analysis:** Before building a workflow, an LLM-based intent analyzer is invoked. The analysis guides tool selection and workflow type.  
- **Caching:** Workflows are cached by (user_id, workflow_type, toolset) signature.

### 4.1.2 Graph Builder Module

```python
# composer/graph/builder.py
class GraphBuilder:
    """Constructs LangGraph workflows dynamically."""

    def build_from_context(
        self,
        conversation_ctx: ConversationContext,
        tools: List[BaseTool],
        config: WorkflowConfig
    ) -> CompiledGraph:
        # Determine workflow type from intent
        if conversation_ctx.intent.deep_research:
            return self.build_research_workflow(conversation_ctx, tools, config)
        elif conversation_ctx.intent.image_generation:
            return self.build_creative_workflow(conversation_ctx, tools, config)
        else:
            return self.build_chat_workflow(conversation_ctx, tools, config)
```

- **Configurability:** The config object (e.g. ResearchConfig) carries parameters like search_depth, max_sources, and retrieve_full_content. The graph builder passes these to nodes (see **Research Workflow** below).
  
### 4.1.3 Tool Registry Module

```python
# composer/tools/registry.py
class ToolRegistry:
    """Centralized tool management with composability and reuse."""

    def __init__(self):
        self.static_tools = {
            'web_search': WebSearchTool,
            'memory_retrieval': MemoryRetrievalTool,
            'summarization': SummarizationTool,
            # ... other static tools ...
        }
        self.dynamic_tools = {}  # id -> tool instance
        self.tool_embeddings = {}  # id -> embedding vector for semantic search

    async def get_tools_for_context(self, intent, conversation_ctx):
        """Select applicable tools based on intent and context."""
        tools = []
        # 1. Include relevant static tools
        for name, tool_cls in self.static_tools.items():
            if self._should_include_static_tool(name, intent):
                tools.append(tool_cls(conversation_ctx))
        # 2. Dynamic tool generation or retrieval
        if intent.needs_dynamic_tool:
            spec = intent.tool_spec  # spec proposed by intent agent
            tool = await self._generate_or_retrieve_dynamic_tool(conversation_ctx, spec)
            if tool:
                tools.append(tool)
        return tools

    async def _generate_or_retrieve_dynamic_tool(self, context, spec):
        """Return an existing dynamic tool similar to spec, or generate a new one."""
        # Compute embedding of spec description
        spec_vec = await compute_embedding(spec.description)
        # Find similar existing tool via vector similarity
        best_match = find_best_match(spec_vec, self.tool_embeddings, threshold=0.8)
        if best_match:
            existing_tool = self.dynamic_tools[best_match]
            # If adjustment needed (e.g., different params), return modified copy
            if spec.needs_adjustment:
                return existing_tool.clone_with_new_params(spec.parameters)
            else:
                return existing_tool
        # No similar tool found: generate new one
        code = await LLM.generate_tool_code(spec)
        new_tool = compile_tool_from_code(code)
        tool_id = new_tool.name
        self.dynamic_tools[tool_id] = new_tool
        self.tool_embeddings[tool_id] = spec_vec
        return new_tool
```

**Key Points:**  

- **Intent-Based Inclusion:** A preliminary intent analysis (an LLM-based agent) marks which static and dynamic tools are needed.  
- **Semantic Search:** We store vector embeddings of dynamic tool specs. Before generating a new tool, we perform a semantic search over existing tool descriptions to find a similar one[[1]](https://langchain-ai.github.io/langgraph/how-tos/tool-calling/#:~:text=To%20address%20this%2C%20you%20can,at%20runtime%20using%20semantic%20search). If found, we reuse or adapt it; otherwise we generate a fresh tool.  
- **Tool Composition:** All tools (static or dynamic) are designed to be composable. For example, if the user's intent requires "plan trip", we might combine a flight-booking tool and a weather-check tool into a higher-level "travel planner" operation at runtime. LangGraph's ToolNode executes multiple tool calls in parallel and aggregates their outputs[[4]](https://langchain-ai.github.io/langgraph/how-tos/tool-calling/#:~:text=1.%20,tools%20in%20parallel).  
- **MCP Integration:** Tools can also be provided via the Model Context Protocol (MCP). By installing langchain-mcp-adapters, LangGraph agents can treat MCP-registered tools as first-class[[2]](https://langchain-ai.github.io/langgraph/concepts/mcp/#:~:text=Model%20Context%20Protocol%20,adapters%60%20library). This allows reusing existing corporate or partner tool definitions without custom code generation.

### 4.1.4 State Manager Module

```python
# composer/state/manager.py
class StateManager:
    """Manages workflow state, checkpoints, and context length."""

    def create_initial_state(
        self,
        messages: List[Message],
        context: Dict[str, Any]
    ) -> WorkflowState:
        state = WorkflowState(
            messages=messages,
            context=context,
            available_tools=[],
            current_message=None,
            conversation=context['conversation'],
            user_config=context['user_config'],
            metadata={}
        )
        # Attach search complexity settings from user or intent
        state.context['search_config'] = context['user_config'].get('search_config')
        return state

    def enforce_context_limit(self, state: WorkflowState, n_ctx: int) -> WorkflowState:
        """Trim or summarize history if exceeding the model context window."""
        # Implementation: drop oldest messages or compress to fit n_ctx tokens
        return state

    # checkpoint_state and restore_state methods omitted for brevity
```

### 4.2 Node Definitions

#### 4.2.1 Standard Nodes

```python
# composer/nodes/standard.py
class PipelineNode:
    """Wraps chat-model execution as a graph node."""
    def __init__(self, pipeline_factory, profile_selector, stream: bool = False):
        self.pipeline_factory = pipeline_factory
        self.profile_selector = profile_selector
        self.stream = stream  # whether to stream output

    async def __call__(self, state: WorkflowState) -> WorkflowState:
        profile = self.profile_selector(state)
        pipeline = self.pipeline_factory.get_pipeline(profile, ChatResponse, streaming=self.stream)
        # If streaming, use astream_events to yield partial outputs
        if self.stream:
            async for event in pipeline.astream_events({"messages": state.messages}):
                if event["event"] == "on_chat_model_stream":
                    chunk = event["data"]  # ChatResponse chunk (AIMessageChunk)
                    yield ChatResponse(chunk)  # yield to Composer caller
            # After streaming finished, final message appended
        else:
            response = await pipeline.invoke({"messages": state.messages})
            state.messages.append(response.message)
        return state

class ToolExecutorNode:
    """Executes tool calls produced by the previous agent or tool node."""
    def __init__(self, tools: List[BaseTool]):
        self.tools = {t.name: t for t in tools}

    async def __call__(self, state: WorkflowState) -> WorkflowState:
        last_message = state.messages[-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            # Use LangGraph ToolNode logic: execute all calls in parallel
            tool_node = ToolNode(list(self.tools.values()))
            results = tool_node.invoke({"messages": [last_message]})
            # 'results' contains ToolMessage(s); append to history
            state.messages.extend(results['messages'])
        return state

class RAGNode:
    """Embeds latest user message and performs retrieval augmentation."""
    async def __call__(self, state: WorkflowState) -> WorkflowState:
        msg = state.current_message or state.messages[-1]
        embeddings = await self.create_embeddings(msg)
        # Run multiple retrieval tools in parallel
        mems, web, summary = await asyncio.gather(
            self.memory_retrieval(embeddings),
            self.web_search(msg),
            self.summarize(state.messages)
        )
        state.context.update({
            'memories': mems,
            'search_results': web,
            'summary': summary
        })
        return state
```

- **Streaming in PipelineNode:** The PipelineNode can now be configured to stream (stream=True) for chat. It uses pipeline.astream_events and yields on on_chat_model_stream events[[3]](https://python.langchain.com/docs/concepts/streaming/#:~:text=from%20langchain_core,ChatPromptTemplate%20from%20langchain_anthropic%20import%20ChatAnthropic). For secondary or batch nodes (e.g. other agents, RAG enrichment), we use non-streaming mode.

- **Tool Executor:** ToolNode from LangGraph is used under the hood: when the assistant's message contains multiple tool calls, they are executed in parallel, and each result is appended to the conversation[[4]](https://langchain-ai.github.io/langgraph/how-tos/tool-calling/#:~:text=1.%20,tools%20in%20parallel).

- **Composable Tools:** All tools inherit from BaseTool and use the standard @tool decorator interface. This makes them easily composable and reusable[[6]](https://docs.langchain.com/oss/javascript/langchain/overview#:~:text=started%20on%20building%20agents%20with,for%20basic%20LangChain%20agent%20usage).


#### 4.2.2 Specialized Nodes

```python
# composer/nodes/specialized.py

class TitleGenerationNode:
    """Generates a conversation title if none exists."""
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
    """Analyzes intent and generates or retrieves dynamic tools as needed."""
    async def __call__(self, state: WorkflowState) -> WorkflowState:
        analyzer = SmartIntentAnalyzer()
        analysis = await analyzer.analyze_intent(state.current_message or state.messages[-1])
        if analysis.needs_dynamic_tool:
            tool = await self.tool_registry._generate_or_retrieve_dynamic_tool(
                state, analysis.tool_spec
            )
            state.available_tools.append(tool)
        return state
```

- **Intent Analysis:** SmartIntentAnalyzer is an LLM-based agent that classifies the current message (e.g. deep research request, create dynamic tool, etc.).

- **Adaptive Tool Generation:** If a new tool is needed, generate_or_retrieve_dynamic_tool uses the semantic search logic described above to avoid redundant tools.

### 4.3 Workflow Definitions

#### 4.3.1 Standard Chat Workflow

```python
# composer/workflows/chat.py

def build_chat_workflow(config: ChatConfig) -> StateGraph:
    workflow = StateGraph(WorkflowState)

    # Enrich conversation with RAG (memories, web results, etc.)
    workflow.add_node("rag_enrichment", RAGNode())
    workflow.add_node("dynamic_tools", DynamicToolGenerationNode())
    # Primary agent node: enable streaming for UI responsiveness
    workflow.add_node("agent", PipelineNode(
        pipeline_factory,
        lambda s: s.user_config.model_profiles.primary_profile,
        stream=True  # stream responses for primary chat agent
    ))
    workflow.add_node("tools", ToolExecutorNode())

    # Define edges
    workflow.add_edge(START, "rag_enrichment")
    workflow.add_edge("rag_enrichment", "dynamic_tools")
    workflow.add_edge("dynamic_tools", "agent")

    # After agent: if tool calls are present, go to tools, else end
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

- **Streaming Event Loop:** The agent node is set to stream=True. During execution, LangGraph will emit on_chat_model_stream events with partial AIMessageChunks[[3]](https://python.langchain.com/docs/concepts/streaming/#:~:text=from%20langchain_core,ChatPromptTemplate%20from%20langchain_anthropic%20import%20ChatAnthropic). The server's completion handler will listen to these events and send them to the UI as they arrive.

- **Secondary Nodes (tools):** The tools node (ToolExecutorNode) is a complete operation: once all tools have run, their results (ToolMessages) are appended. Then the graph loops back to the agent for a final synthesis. These tool messages are returned as whole items, not streamed.

- **Graph Execution:** When invoked, this workflow is run via workflow.astream_events(initial_state, version="v2") rather than the older stream_pipeline. Example consumption in the server:

```python
composer = ComposerService()
workflow = await composer.compose_workflow(
    conversation_ctx,
    WorkflowType.CHAT
)
# Process streaming events for chat model output
async for event in workflow.astream_events(initial_state, version="v2"):
    if event["event"] == "on_chat_model_stream":
        chunk = event["data"]  # a ChatResponse (AIMessageChunk)
        yield serialize_to_json(chunk)
```

- In the LangChain streaming guide, this pattern is recommended: filter astream_events for the on_chat_model_stream event to send incremental outputs[[3]](https://python.langchain.com/docs/concepts/streaming/#:~:text=from%20langchain_core,ChatPromptTemplate%20from%20langchain_anthropic%20import%20ChatAnthropic).

#### 4.3.2 Research Workflow

```python
# composer/workflows/research.py

def build_research_workflow(config: ResearchConfig) -> StateGraph:
    workflow = StateGraph(ResearchState)

    # Configurable search steps
    workflow.add_node("query_expansion", QueryExpansionNode(depth=config.search_depth))
    workflow.add_node("parallel_search", ParallelSearchNode(max_sources=config.max_sources, full_text=config.retrieve_full_content))
    workflow.add_node("source_validation", SourceValidationNode())
    workflow.add_node("synthesis", SynthesisNode())
    workflow.add_node("fact_check", FactCheckNode())

    # Linear flow
    workflow.add_edge(START, "query_expansion")
    workflow.add_edge("query_expansion", "parallel_search")
    workflow.add_edge("parallel_search", "source_validation")
    workflow.add_edge("source_validation", "synthesis")
    workflow.add_edge("synthesis", "fact_check")
    workflow.add_edge("fact_check", END)

    return workflow.compile()
```

- **Search Complexity Config:** The ResearchConfig contains parameters (search_depth, max_sources, retrieve_full_content). These are set per user or conversation, possibly based on intent. For example, a user indicating "deep dive" might get a higher max_sources and true for full content retrieval. The QueryExpansionNode uses depth to recursively refine the query. ParallelSearchNode limits how many sources to fetch. If the model's context (n_ctx) is small, the graph builder or nodes should automatically reduce these numbers (e.g., summarizing results or lowering max_sources).

- **Context-Aware Synthesis:** Before running SynthesisNode, use the StateManager to enforce context limits. If total tokens (message history plus source excerpts) exceed the LLM's window, trim or summarize older content so the model can safely process the requested information.

- **Intent-Driven Defaults:** The initial intent analysis (in ComposerService) can populate config with default values. For example, a conversation labeled "quick answer" might default to shallow search (search_depth=1, max_sources=3), whereas "deep research" might allow deeper queries and more sources.

***

## 5. Implementation Roadmap & Checklist

- **Phase 1:** Setup composer directory, build GraphBuilder, ToolRegistry, StateManager; Extract tool logic from server.
- **Phase 2:** Implement workflow builders for chat, research, multi-agent; Setup streaming API and intent classification; Build caching layers.
- **Phase 3:** Migrate orchestration to Composer; Implement circuit breakers; Add tool generation and registry; Begin comprehensive testing.
- **Phase 4:** Add multi-agent support and complex workflows; Finalize streaming optimizations; Implement robust error handling.
- **Phase 5:** Extensive unit/integration/performance testing; Documentation; Feature flag rollout.
- **Phase 6:** Production deployment and monitoring; Archival of legacy components.

### 5.1 Detailed Checklist

**Phase 1: Foundation**

- [ ] Create Composer directory and core modules (GraphBuilder, ToolRegistry, StateManager)[[7]](file://file-KGRowE6RrRi63YDxHvVDwx#:~:text=%60%60%60python%20composer%2F__init__.py%20class%20ComposerService%3A%20,composer%20service%20coordinating%20all%20subcomponents)[[1]](https://langchain-ai.github.io/langgraph/how-tos/tool-calling/#:~:text=To%20address%20this%2C%20you%20can,at%20runtime%20using%20semantic%20search).
- [ ] Extract and centralize tool logic from the server into composer/tools.
- [ ] Implement base node classes: PipelineNode, ToolExecutorNode, RAGNode[[8]](file://file-KGRowE6RrRi63YDxHvVDwx#:~:text=class%20ToolExecutorNode%3A%20,from%20previous%20node).
- [ ] Set up logging, metrics, and monitoring frameworks (see monitoring sections below).

**Phase 2: Core Implementation**

- [ ] Develop GraphBuilder.build_from_context, including intent detection and config parameters.
- [ ] Implement ToolRegistry.register_static_tool, _generate_or_retrieve_dynamic_tool (with semantic lookup)[[1]](https://langchain-ai.github.io/langgraph/how-tos/tool-calling/#:~:text=To%20address%20this%2C%20you%20can,at%20runtime%20using%20semantic%20search).
- [ ] Create StateManager with checkpointing and context-limit enforcement (use model profiles' n_ctx).
- [ ] Build workflow definitions: chat, research, (and creative if needed). Incorporate streaming mode in chat workflow.
- [ ] Build a simple *StreamProcessor* in Composer to filter and forward on_chat_model_stream events[[3]](https://python.langchain.com/docs/concepts/streaming/#:~:text=from%20langchain_core,ChatPromptTemplate%20from%20langchain_anthropic%20import%20ChatAnthropic).

**Phase 3: Integration**

- [ ] Update the HTTP completion handler to use ComposerService. Use workflow.astream_events() for streaming as shown above.
- [ ] Migrate any leftover orchestration (e.g. title generation, query formatting) into nodes or builder logic.
- [ ] Implement DynamicToolGenerationNode logic for intent-based tool creation (avoid redundant tools via semantic search).
- [ ] Enable tool composition: allow workflows to chain multiple tools by returning a list of tool calls in agent responses (LangGraph ToolNode handles this internally)[[4]](https://langchain-ai.github.io/langgraph/how-tos/tool-calling/#:~:text=1.%20,tools%20in%20parallel).
- [ ] Add circuit breaker protection around nodes (existing plan).

**Phase 4: Advanced Features**

- [ ] **Streaming Enhancements:** Finalize streaming pipeline for chat. Use workflow.astream_events(version="v2") and handle on_chat_model_stream events in the server (as in phase 2). Ensure non-chat agents/tools do not stream.
- [ ] **Configurable Retrieval:** Implement ResearchConfig with search parameters, and modify search nodes (QueryExpansionNode, ParallelSearchNode) to respect these settings and the model's n_ctx.
- [ ] **Intent-based Tool Selection:** Use the LLM intent agent output to filter static tools and trigger dynamic tool generation. Apply semantic search lookup to reuse similar tools[[1]](https://langchain-ai.github.io/langgraph/how-tos/tool-calling/#:~:text=To%20address%20this%2C%20you%20can,at%20runtime%20using%20semantic%20search).
- [ ] **Composable Tool Support:** Ensure all tools use the @tool schema and are documented for reuse. For example, dynamic tools generated should include descriptive names and JSON schemas so they can be composed by name.
- [ ] **Multi-Agent Workflows:** Implement multi-agent states and handoffs. Use LangGraph Command objects to switch between agents[[5]](https://langchain-ai.github.io/langgraph/how-tos/multi_agent/#:~:text=2.%20The%20,agent%20graph). The AgentNode code should route control via Command(goto=...) to hand off to other agents as needed.
- [ ] **MCP Adapter:** (Optional) Integrate langchain-mcp-adapters so that tools can be served from an MCP server, enabling standardization of tool definitions[[2]](https://langchain-ai.github.io/langgraph/concepts/mcp/#:~:text=Model%20Context%20Protocol%20,adapters%60%20library).

**Phase 5: Testing and Documentation**

- [ ] Write unit tests for all new nodes and registry logic (mock LLMs for intent and tool generation).
- [ ] Create integration tests: e.g. simulate a streaming chat flow and verify on_chat_model_stream events.
- [ ] Performance test the search workflow with different search_depth and max_sources to validate context trimming.
- [ ] Update documentation: explain new configuration fields, streaming behavior, and tool generation process (including semantic search).

### 5.2 Key Best Practices and Notes


* **LangGraph Streaming:** LangGraph graphs fully support streaming via standard APIs[[9]](https://python.langchain.com/docs/concepts/streaming/#:~:text=LangGraph%20%20compiled%20graphs%20are,support%20the%20standard%20streaming%20APIs). Use workflow.astream_events() to tap into low-level events. In particular, filter for event=="on_chat_model_stream" to stream chat LLM output as it is generated[[3]](https://python.langchain.com/docs/concepts/streaming/#:~:text=from%20langchain_core,ChatPromptTemplate%20from%20langchain_anthropic%20import%20ChatAnthropic).

* **Tool Calling:** LangGraph's ToolNode natively handles multiple simultaneous tool calls, running them in parallel and returning ToolMessage outputs[[4]](https://langchain-ai.github.io/langgraph/how-tos/tool-calling/#:~:text=1.%20,tools%20in%20parallel). Use this by ensuring agents output a list of tool call requests.

* **Multi-Agent:** Build multi-agent workflows by treating each sub-agent as a node and using Command(goto=agent_name) for handoffs[[5]](https://langchain-ai.github.io/langgraph/how-tos/multi_agent/#:~:text=2.%20The%20,agent%20graph). The LangGraph docs recommend using prebuilt multi-agent templates or handoff primitives. We ensure the overall chat history is passed across agents.

* **Context Protocol (MCP):** For enterprise integration, follow the MCP standard to serve tools. LangGraph agents can call MCP-hosted tools seamlessly via langchain-mcp-adapters[[2]](https://langchain-ai.github.io/langgraph/concepts/mcp/#:~:text=Model%20Context%20Protocol%20,adapters%60%20library).

* **LangChain Integration:** Keep agents agnostic of the underlying graph engine. LangChain's @tool decorator and agent APIs remain valid; under the hood we leverage LangGraph's durable execution and streaming[[6]](https://docs.langchain.com/oss/javascript/langchain/overview#:~:text=started%20on%20building%20agents%20with,for%20basic%20LangChain%20agent%20usage)[[10]](https://docs.langchain.com/oss/javascript/langchain/overview#:~:text=Standard%20model%20interface%20Different%20providers,Learn%20more%2036%20Debug%20with).

***

## 6. Success Criteria

- **Code Quality:** Reduce cyclomatic complexity by 40%; ensure modularity and clarity.
- **Performance:** Achieve <100ms workflow creation with cache; <5ms node execution overhead.
- **Maintainability:** Enable rapid new workflow creation (<1 hour); tool addition without server code modification.
- **Reliability:** 99.9% uptime; graceful failure handling and recovery.
- **Backward Compatibility:** Support legacy APIs during migration with feature flags.

***

## 7. References

- [^1] LangChain \& LangGraph Official Tutorials and API references: https://github.com/langchain-ai/langchain
- [^2] LangGraph v1.0 Alpha Documentation: https://langchain.ai/docs/langgraph/v1
- [^3] Original refactor documents (attachments)
- LangChain Streaming Guide: https://python.langchain.com/docs/concepts/streaming
- LangGraph Streaming Event Specification: https://langchain.github.io/langgraph/posts/streaming
- LangChain Multi-Agent Cookbook: https://docs.langchain.com/docs/multiagent
- Adaptive RAG Design Patterns: https://langchain.ai/blog/adaptive_rag/
- Dynamic Tool Generation Concepts: https://langchain.ai/blog/dynamic_tools
- LangChain Runnable Patterns: https://python.langchain.com/docs/howto/build_tools
- Model Context Protocol (MCP) Overview: https://github.com/langchain-ai/mcp/

