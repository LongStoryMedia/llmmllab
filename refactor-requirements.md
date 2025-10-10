# Comprehensive Final Requirements Document: Refactoring Agentic System to LangGraph-Based Architecture

## Executive Summary

This final requirements document holistically integrates and consolidates all prior refactoring plans with meticulous inclusion of all content, emphasizing the fundamental architectural shift to a **Composer-centric design**. The Composer becomes the authoritative orchestrator and runtime environment for the agentic system.

The analysis confirms that the specified refactoring requirements—incorporating selective streaming, implementing configurable Retrieval-Augmented Generation (RAG) depth, and establishing a system for dynamic, composable tool management—are entirely feasible and maintain high system integrity. These requirements align precisely with the core design principles of the **LangGraph V1 alpha framework**, which is engineered for production-grade agent orchestration and advanced workflow control.

The foundation of system integrity is provided by LangGraph V1 alpha's features, particularly its **Durable Execution** capability, which provides a built-in agent runtime ensuring state continuity and reliability. This resilience is essential because the architecture mandates moving the actual execution of the complex agent graph into the remote composer service. Furthermore, LangGraph V1 emphasizes **Execution Control**, enabling "fine-tuned control over execution." This programmatic control is the prerequisite for implementing the conditional logic that governs dynamic tool selection and the flexible routing necessary for configurable RAG depth.

Leveraging LangGraph v1 capabilities and **grammar-constrained structured output**, this refactoring will deliver robust, scalable, and maintainable workflows encompassing adaptive retrieval, dynamic tool management, multi-agent orchestration, and precise streaming control. The integration of llamacpp grammars ensures type-safe, reliable LLM output across all agent interactions, eliminating JSON parsing errors and providing guaranteed structure validation. The document captures all architectural motivations, detailed requirements, code examples, validation grounds, and strategic implementation phases to enable a smooth migration and future system growth.

-----

## 1\. Current System Analysis

### 1.1 Architecture Challenges

- **Fragmented orchestration:** Multiple isolated server components manually coordinate functions like summarization, retrieval, and tool management, leading to brittle systems.
- **Tool management sprawl:** Tools, especially dynamic tool instantiations, are decentralized, reducing reuse and introducing duplication.
- **Tight coupling:** Server and runner components are tightly bound through factories, restricting pipeline flexibility.
- **State and error handling gaps:** Mixed state management between manual and LangGraph approaches, inconsistent error propagation, and scattered cleanup logic.
- **Streaming constraints:** Only the primary agent supports streaming; secondary agents and tool nodes lack streaming integration.
- **Recovery and resource management:** Lacking circuit breakers and robust failure handling harms system reliability.

### 1.2 Critical Architectural Migration: LangGraph Orchestration from Runner to Composer

The current architecture suffers from a fundamental design violation: **LangGraph orchestration logic is embedded throughout the runner component**, creating tight coupling between pure LLM execution and complex workflow orchestration. This architectural flaw must be addressed through a comprehensive migration that clearly separates concerns between runner and composer components.

#### 1.2.1 Current Runner Architecture Problems

**Embedded LangGraph Orchestration:**

- Runner pipelines contain `create_graph()` methods with extensive orchestration logic (780+ lines in `openai_gpt_oss.py`)
- Graph creation, agent node implementation, tool orchestration, and state management are deeply embedded in runner
- Pipeline classes implement `LangGraphCapable` protocol with graph construction responsibilities
- Agent nodes, tool handlers, and conditional routing logic reside in runner instead of composer

**Architectural Violations:**

- **Tight Coupling:** Runner cannot be used as a simple LLM interface due to embedded orchestration
- **Mixed Responsibilities:** Pipeline execution mixed with workflow orchestration in single components  
- **Code Duplication:** Similar graph patterns repeated across multiple pipeline implementations
- **Testing Complexity:** Unit testing pipelines requires mocking extensive LangGraph infrastructure
- **Scalability Issues:** Adding new workflow patterns requires modifying runner pipeline code

**Specific Components Requiring Migration:**

```python
# Current problematic architecture in runner/pipelines/
class BasePipelineCore(LangGraphCapable):
    @abstractmethod
    async def create_graph(self) -> CompiledGraph:  # ❌ WRONG: Orchestration in runner
        """Creates LangGraph workflow - should be in composer"""
        pass

class OpenAIGPTOSSPipeline(BasePipelineCore):
    async def create_graph(self) -> CompiledGraph:  # ❌ 780+ lines of orchestration
        # Extensive agent node definitions
        # Tool orchestration logic  
        # State management
        # Conditional routing
        # All should be in composer, not runner
```

#### 1.2.2 Target Migration Architecture

**Composer Responsibility (NEW):**

- **Graph Construction:** All `create_graph()` logic moved to `composer/workflows/`
- **Agent Node Implementation:** All node functions relocated to `composer/nodes/`
- **Tool Orchestration:** Centralized tool management and routing in composer
- **State Management:** Unified GraphState handling in composer runtime
- **Workflow Patterns:** Reusable workflow templates in composer for different use cases

**Runner Responsibility (SIMPLIFIED):**

- **Pure LLM Interface:** Simple, clean interface for LLM interactions (local or remote)
- **Grammar Processing:** Handle grammar-constrained structured output for individual LLM calls
- **Streaming Support:** Token-level streaming for real-time response delivery  
- **Pipeline Abstraction:** Lightweight pipeline wrappers for different LLM providers
- **No Orchestration:** Zero workflow, graph, or multi-step logic

**Clean Separation Pattern:**

```python
# Target: Simplified runner interface
class BaseLLMPipeline:
    """Pure LLM interface without orchestration."""
    
    async def generate(self, messages: List[Message], 
                      grammar: Optional[GrammarInput] = None) -> ChatResponse:
        """Simple LLM generation with optional grammar constraints."""
    
    async def stream_generate(self, messages: List[Message],
                             grammar: Optional[GrammarInput] = None) -> AsyncIterator[ChatResponse]:
        """Streaming LLM generation with grammar support."""

# Target: Composer handles all orchestration
class WorkflowOrchestrator:
    """Centralized graph construction and execution in composer."""
    
    async def create_chat_workflow(self, context: ConversationContext) -> CompiledGraph:
        """Build chat workflow using composer patterns."""
    
    async def execute_workflow(self, workflow: CompiledGraph, 
                              state: GraphState) -> AsyncIterator[WorkflowEvent]:
        """Execute workflow with streaming support."""
```

#### 1.2.3 Migration Impact Analysis

**Files Requiring Major Changes:**

- `runner/pipelines/base.py` - Remove `LangGraphCapable`, simplify to pure LLM interface
- `runner/pipelines/txt2txt/openai_gpt_oss.py` - Extract 780+ lines of orchestration to composer
- `runner/pipelines/txt2txt/base_langgraph.py` - Move `_create_graph_with_timeout` to composer
- All pipeline implementations with `create_graph()` methods

**Files Requiring Creation:**

- `composer/workflows/chat_workflow.py` - Chat workflow patterns from runner extraction
- `composer/workflows/research_workflow.py` - Research workflow patterns  
- `composer/nodes/` - All agent node implementations extracted from runner
- `composer/runtime/orchestrator.py` - Central workflow execution engine

**Interface Changes:**

- Runner interfaces (`run_pipeline`, `stream_pipeline`) simplified to pure LLM calls
- Composer interfaces enhanced with workflow construction and execution methods
- Clear Protocol definitions for runner-composer communication
- Grammar support maintained in runner for individual LLM interactions

### 1.3 Roles of Major Components (Post-Migration)

- **Server:** Handles HTTP, auth, user config, and legacy orchestration currently scheduled for deprecation.
- **Composer:** Centralized orchestration, stateful workflow construction, execution, intent parsing, tool lifecycle, and error management. **NOW INCLUDES:** All LangGraph graph construction, agent node implementation, and workflow orchestration previously embedded in runner.
- **Runner:** **SIMPLIFIED TO:** Pure LLM interface for local and remote model interactions with grammar-constrained structured output support. **REMOVES:** All LangGraph orchestration, graph creation, and multi-step workflow logic.
- **Tools:** Includes static tools (search, summarization) and dynamic tools derived from LLM-generated code.

-----

## 2. Target Architecture

### 2.1 Composer: The Heart of the New Architecture

Central to the redesign is the **Composer component**. The refactoring dictates a crucial architectural shift: the composer project must transition from merely defining the graph structure to serving as the **primary, authoritative execution runtime**. By relying on LangGraph V1's promise of durable execution, the composer can manage lengthy, multi-step operations—such as complex web crawls, iterative RAG, or the creation of new tools—without the risk of external state loss, even if the client connection is interrupted temporarily. This architectural decision significantly enhances system resilience by decoupling the UI's front-end responsiveness from the backend's computational load.

The Composer is responsible for:

- **Graph construction & execution:** Intelligently builds LangGraph task graphs based on conversation context and dynamically selected tools.
- **Streaming orchestration:** Uses LangGraph's `astream_events` API to manage and route streaming events for primary chat interaction and control non-streaming responses from secondary nodes smoothly.
- **State management:** Maintains a unified, authoritative GraphState with full persistence, checkpoints, and seamless recovery.
- **Tool management:** Centralizes tool registration, dynamic generation and discovery through **Engineering Agent** with grammar-constrained structured output, leveraging semantic search to maximize reuse and minimize redundancy.
- **Intent analysis:** Runs grammar-constrained LLM intent classifiers with guaranteed structured output to set retrieval depth, determine toolsets, and drive conditional routing.
- **Structured Output Control:** Implements llamacpp grammar constraints across all agent nodes to ensure type-safe, validated LLM responses with automatic Pydantic model parsing.
- **Error resiliency:** Coordinates circuit breaker protections, error handling, and retry policies at per-node granularity.
- **Multi-agent orchestration:** Implements cross-agent handoffs through LangGraph `Command` primitives for complex collaborative workflows.

### 2.2 High-Level System Layout


```plaintext
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
- Implements robust state persistence, error handling, real-time streaming, and intent-driven dynamic workflow selection (chat, research, multi-agent).
- Favors clean separation of concerns per LangGraph V1 principles.

### 2.3 Runner: Simplified Pure LLM Interface (Post-Migration)

The migration transforms the runner from a complex orchestration component to a **clean, focused LLM interface**. This architectural simplification enables better testability, reusability, and maintainability while providing a stable foundation for composer orchestration.

**Simplified Runner Responsibilities:**

- **Pure Model Interface:** Direct interaction with local (llama.cpp) and remote (OpenAI, Anthropic) LLM providers
- **Grammar-Constrained Generation:** Support for GBNF grammar constraints ensuring structured output validation
- **Streaming Support:** Token-level streaming for real-time response delivery without orchestration complexity
- **Pipeline Abstraction:** Lightweight wrappers for different model providers and configurations
- **Type Safety:** Integration with grammar generator for Pydantic model validation

**Removed Responsibilities (Migrated to Composer):**

- ❌ **Graph Construction:** All `create_graph()` methods removed from pipeline classes
- ❌ **Agent Orchestration:** Agent nodes, tool coordination, and multi-step workflows
- ❌ **State Management:** Complex GraphState handling and persistence
- ❌ **Conditional Routing:** Workflow decision logic and branching
- ❌ **Tool Integration:** Dynamic tool discovery, creation, and execution coordination

**Post-Migration Runner Interface:**

```python
# Simplified runner interface - pure LLM interaction
class SimplifiedRunnerInterface:
    """Clean LLM interface without orchestration complexity."""
    
    async def generate(
        self, 
        messages: List[Message], 
        grammar: Optional[GrammarInput] = None,
        stream: bool = False
    ) -> Union[ChatResponse, AsyncIterator[ChatResponse]]:
        """Generate response with optional grammar constraints."""
    
    async def validate_grammar(self, grammar: GrammarInput) -> bool:
        """Validate grammar constraints before generation."""
    
    async def health_check(self) -> ModelHealthStatus:
        """Check model availability and performance."""

# Used by composer nodes for individual LLM calls
class ComposerLLMNode:
    """Composer node using simplified runner interface."""
    
    def __init__(self, runner_client: SimplifiedRunnerInterface):
        self.runner = runner_client
    
    async def __call__(self, state: GraphState) -> GraphState:
        # Single LLM call with grammar constraints
        response = await self.runner.generate(
            messages=state.messages,
            grammar=IntentAnalysis,  # Pydantic model for structured output
            stream=False
        )
        # Update state with validated response
        state.intent_classification = response.structured_output
        return state
```

**Migration Benefits:**

- **Testability:** Runner components can be unit tested without LangGraph mocking
- **Reusability:** Simplified interface can be used by other services beyond composer  
- **Performance:** Reduced overhead from removing orchestration complexity
- **Maintainability:** Clear separation of concerns between LLM execution and workflow orchestration
- **Scalability:** Independent scaling of LLM resources vs orchestration logic

### 2.4 Architectural Shift Validation (LangGraph V1-alpha)

All complex graph execution is centralized in the Composer service, which leverages **Durable Execution**—state persistence and recovery in case of client disconnects. The Composer service will exclusively host the complex LangGraph application, managing the central graph state, executing all node functions, and enforcing dynamic conditional flow. Consequently, all core LangGraph logic, including the foundational state definitions, node implementations, and the final compiled graph artifact (`app.compile()`), must be securely housed within a dedicated structure, specifically `composer/agent_runtime/`. Communication with the client is managed through a specialized, single asynchronous endpoint designed to handle the required selective data streaming (WebSocket/SSE), receiving event envelopes.

-----

## 3. Implementation Architecture

### 3.1 Unified Graph State Schema Design

A foundational requirement for a sophisticated, durable LangGraph system is the definition of a single, centralized `GraphState` schema. This state model acts as the common interface for all nodes, ensuring data consistency and enabling conditional routing by maintaining context across execution steps. This state is typically implemented as a Pydantic model located in `composer/agent_runtime/state.py`.

**Mandatory Elements for GraphState:**

| Field Name | Type | Reducer Function (LangGraph) | Purpose |
| :--- | :--- | :--- | :--- |
| `messages` | `List` | `lambda x, y: x.concat(y)` | Conversation history and final outputs, essential for context and token streaming. |
| `intent_classification` | `IntentAnalysis` | `lambda x, y: y` | Grammar-constrained structured output from Intent Agent using `intent_analysis.yaml` schema. |
| `required_tools` | `List` | `lambda x, y: y` | The curated list of standard and dynamic tools collected for the current execution phase. |
| `search_results` | `str` | `lambda x, y: y` | The consolidated, synthesized output from RAG execution (whether shallow or deep). |
| `rag_depth_config` | `str` | `lambda x, y: y` | Stores the decision ('SHALLOW' or 'DEEP'), which drives the conditional edge for RAG routing. |
| `progress_updates` | `List` | `lambda x, y: x.concat(y)` | User-defined signals used for granular progress tracking during tool or crawl execution. |

(Full context model implemented in composer/state.py)

### 3.2 Execution Control and Asynchronous Streaming Architecture

This requirement mandates precise control over data delivery: streaming real-time tokens from the primary conversational node while relying on completed, structured text outputs from all upstream agent nodes. LangGraph’s native streaming system supports this level of granularity by offering multiple streaming modes.

#### 3.2.1 Implementation of Selective Streaming

To implement selective data delivery, each node within the graph must be configured to return data compatible with a specific LangGraph streaming mode.

The **Primary Chat Generator Node** must be configured to stream its output using the `messages` mode. This mode is specifically designed to deliver LLM tokens in real time, accompanied by necessary metadata, thereby fulfilling the core mandate of streaming the main chat operation back to the UI.

In contrast, intermediate **Agent Nodes**—such as the Intent Classifier, Tool Collector, and RAG Executor—generate intermediate logic and synthesized resources. Their streaming behavior must be configured to utilize either the `updates` mode, which streams state deltas, or the `custom` mode, which allows for the emission of arbitrary user-defined signals. Streaming state deltas or custom progress notifications, such as "Tool Selection Agent Running" or "Executing Deep Web Crawl," provides necessary operational transparency. Without this granular feedback, the user would experience silent waiting periods during the most computationally intensive phases.

#### 3.2.2 Streaming Modes Configuration

| Node Type | Role/Function | Required Streaming Mode (LangGraph) | Data Payload | Impact on UX |
| :--- | :--- | :--- | :--- | :--- |
| **Primary Chat Generator** | Final Conversational LLM Response. | `messages` (LLM tokens + metadata) | Real-time token chunks. | Real-time token display (low perceived latency). |
| **Intent Classification Agent** | Initial decision, intent parsing, tool request schema definition. | `updates` (State Delta) | Update to `intent_classification` and `rag_depth_config`. | Status update ("Analyzing intent...") upon completion. |
| **Engineering Agent (Dynamic Tools)** | Tool search, composition, and creation with grammar constraints. | `updates` and `custom` (Progress signal) | Updates to `required_tools`. Custom signal: "Tool registry accessed (ID: X)." | Transparent tracking of dynamic tool assembly process. |
| **Deep RAG Executor Node** | Executes resource-intensive crawl/synthesize. | `updates` or `custom` | Custom signal: "Fetched 10/100 records". Update to `search_results`. | Granular progress display during high-latency RAG. |

#### 3.2.3 Integration Blueprint: Composer Service and UI

Effective implementation requires a robust transport layer between the composer service and the UI client, such as **WebSockets or Server-Sent Events (SSE)**. The composer executes the LangGraph application, yielding a continuous stream of distinct outputs (tokens, state deltas, custom signals). The service is responsible for serializing these diverse payloads using a consistent envelope, including a type discriminator (e.g., `{"type": "token_chunk", "data":...}`). The UI client then parses this envelope and routes the data accordingly.

### 3.3 Configurable Knowledge Retrieval Pipeline (Adaptive RAG Specification)

This addresses the need for configurable knowledge retrieval, moving away from a fixed, deep RAG operation for every query. This functionality is achieved through an **Adaptive RAG** pattern, relying on LangGraph's powerful ability to route execution flow using conditional edges.

#### 3.3.1 The Intent Agent's Role in RAG Depth Selection

The `ClassifierAgent` is mandated to execute early in the graph flow using **grammar-constrained structured output**. A node within this agent, `decide_search_depth`, analyzes the initial user message using llamacpp grammars derived from the `intent_analysis.yaml` schema. The grammar ensures the LLM output is guaranteed to match the `IntentAnalysis` Pydantic model.

#### 3.3.2 Defining RAG Complexity Levels

1. **Level 1: Shallow RAG:** This path, executed by the `execute_shallow_search` node, involves a direct, single-pass retrieval using only the internal vector store retriever. This operation is designed to be fast and low-cost.
2. **Level 2: Deep RAG:** This path, executed by the `execute_deep_crawl_and_synthesize` node, triggers a more resource-intensive, multi-step sub-graph. This typically includes an initial web search using external APIs (e.g., Tavily API), followed by crawling, indexing of new data, and sophisticated synthesis across disparate sources.

#### 3.3.3 Graph Topology for Adaptive Search Routing

After the `ClassifierAgent` completes, the flow routes to a designated `Router_RAG` node. This router node reads the `rag_depth_config` field from the state. If the state dictates `'SHALLOW'`, the conditional edge directs the flow to the `execute_shallow_search` node. If `'DEEP'`, the edge directs execution to the `execute_deep_crawl_and_synthesize` node. This conditional routing is vital for resource optimization. Both RAG paths conclude by routing to a common merge point for subsequent tool orchestration.

### 3.4 Intent-Driven Dynamic Tool Discovery and Composability

This demands a high degree of agent intelligence to select, modify, compose, and generate executable functions dynamically based on **grammar-constrained intent analysis**, leveraging LangChain Expression Language (LCEL) with structured output validation.

#### 3.4.0 Structured Intent Analysis Implementation

```python
# composer/nodes/classifier_agent.py
from models.intent_analysis import IntentAnalysis
from utils.grammar_generator import GrammarGenerator

class IntentClassifierNode:
    """Grammar-constrained intent analysis node."""
    
    def __init__(self, pipeline_factory):
        self.pipeline_factory = pipeline_factory
        # Generate grammar for intent analysis at initialization
        self.intent_grammar = GrammarGenerator.from_pydantic_model(IntentAnalysis)
    
    async def __call__(self, state: WorkflowState) -> WorkflowState:
        """Analyze intent with guaranteed structured output."""
        
        # Create grammar-constrained pipeline
        structured_pipeline = await self.pipeline_factory.create_structured_pipeline(
            prompt_template=self._get_intent_prompt(),
            output_schema=IntentAnalysis,
            grammar=self.intent_grammar,
            enable_fallback=True
        )
        
        # Execute with type-safe result
        intent_analysis = await structured_pipeline.execute({
            "user_query": state.messages[-1].content,
            "conversation_context": self._format_context(state.messages)
        })
        
        # Update state with validated structured output
        state.intent_classification = intent_analysis
        state.rag_depth_config = intent_analysis.search_complexity.value
        
        return state
    
    def _get_intent_prompt(self) -> str:
        return """Analyze the user's request and classify their intent.
        
        User Query: {user_query}
        Conversation Context: {conversation_context}
        
        Classify the intent including:
        1. Main intent category and complexity
        2. Required search depth (SHALLOW or DEEP)
        3. Tool requirements and capabilities needed
        4. Resource intensity and time sensitivity
        
        Respond with valid JSON matching the IntentAnalysis schema.
        All fields are required and must conform to the specified enums and constraints."""
```

#### 3.4.1 Phase 1: Intent Discovery and Conditional Standard Tool Collection

The `ClassifierAgent` serves as the initial tool requirement analyzer using **grammar-constrained output**. It generates a type-safe `IntentAnalysis` model detailing functional needs with guaranteed structure validation. Based on this structured analysis, **Conditional Standard Tool Collection** occurs: pre-defined, standard tools are registered and conditionally included in the `required_tools` list in the GraphState. When dynamic tool generation is required, the **Engineering Agent** is invoked with grammar constraints to ensure predictable, structured tool creation. The grammar constraints ensure tool requirements are properly validated and formatted.

#### 3.4.2 Phase 2: Dynamic Tool Assessment and Creation Logic

If standard tools are insufficient, the **Engineering Agent** begins an intelligent assessment for dynamic tool generation:

1. The DTA queries a **Tool Registry Vector Database (VDB)**, which stores descriptions and schemas of all existing dynamic tools.
2. It performs a semantic similarity check, comparing the user's functional requirement against the existing tool descriptions.
3. A **grammar-constrained LLM call** using `dynamic_tool_spec.yaml` schema ensures structured tool assessment, culminating in a decisive workflow:
      - **Use Existing:** If similarity score is high (e.g., > 0.9), the existing tool is used.
      - **Modify or Compose:** If similarity is moderate (e.g., 0.6 - 0.9), the agent generates a structured modification specification with guaranteed valid LCEL composition syntax.
      - **Create New:** If similarity is low (e.g., < 0.6), the agent initiates a **structured LLM process** with grammar constraints to generate type-safe code and validated schema for a new tool.

#### 3.4.3 Abstraction Mandate: Utilizing LCEL for Composability

The mandate that tools must be "composable and abstract" is achieved by implementing all functional components as **Runnables** within LCEL. LCEL allows any two runnables to be chained together using the pipe operator (`|`) to form a seamless `RunnableSequence`. This resulting sequence is itself a single, complex runnable.

Crucially, this complex sequence achieves **Abstraction** by utilizing the `.as_tool()` method. This method wraps the entire `RunnableSequence`, assigning it a single high-level name, description, and input/output schema. The main agent LLM, responsible for tool calling, only perceives this abstraction and is unaware of the internal multi-step execution logic, simplifying its reasoning process.

**Tool Abstraction and Composability Design:**

| Abstraction Principle | LangChain Mechanism | Implementation Detail | Benefit |
| :--- | :--- | :--- | :--- |
| **Composability** | LCEL Pipe Operator (`\|`) and`RunnableSequence`. | Tools are defined as runnable objects that pass output of one to input of the next. | Allows rapid assembly of bespoke tools from existing atomic functions. |
| **Abstraction** | `.as_tool()` method. | Attaches a name, description, and schema to a complex LCEL sequence. | Hides complexity from the reasoning LLM, simplifying agent decision-making. |
| **Dynamic Creation** | Grammar-Constrained Engineering Agent + Structured Output Parsing. | Intent Agent identifies requirements; **Engineering Agent** uses grammar-constrained LLM to generate validated function code/LCEL sequence using `dynamic_tool_spec.yaml`. | Enables creation of genuinely new, purpose-built tools with guaranteed syntax validity and type safety through specialized Engineering Agent. |

-----

## 4\. Representative Code Examples

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
        user_id: str,
        conversation_id: int,
        workflow_type: WorkflowType,
        messages: List[Message]
    ) -> CompiledGraph:
        """Construct or retrieve a compiled graph for the given conversation."""
        # Configuration and model profiles retrieved from shared data layer internally
        # Data layer provides 3-tier caching (memory -> redis -> postgres) for efficiency
        # No need to pass config objects - they are retrieved as needed with user_id
        
        # Determine tools and intent before building using structured output
        intent = await self._analyze_intent_structured(messages, user_id)
        tools = await self.tool_registry.get_tools_for_context(intent, user_id)
        # Workflow config retrieved internally from shared data layer
        workflow_config = await self._get_workflow_config(user_id, workflow_type, intent)
        # Use cache if available - cache key based on user_id, not config objects
        key = self.workflow_cache.get_cache_key(user_id, workflow_type, tools)
        builder_fn = lambda: self.graph_builder.build_from_context(user_id, messages, tools)
        workflow = await self.workflow_cache.get_or_create(key, builder_fn)
        return workflow

    async def _analyze_intent_structured(self, messages: List[Message], user_id: str):
        # Use grammar-constrained LLM-based intent agent to label the conversation
        # Configuration and model profiles retrieved internally from shared data layer using user_id
        # Example: uc = await storage.get_service(storage.user_config).get_user_config(user_id)
        # Example: mp = await get_model_profile_for_task(uc.model_profiles, ModelProfileType.Analysis, user_id)
        # (Implementation detail omitted)
        pass
```

**Highlights:**

- Uses a **LangGraph CompiledGraph** returned to the caller, which supports both streaming and batch execution.
- **Intent Analysis:** Before building a workflow, an LLM-based intent analyzer is invoked. The analysis guides tool selection and workflow type.
- **Caching:** Workflows are cached by (user\_id, workflow\_type, toolset) signature.

### 4.1.2 Graph Builder Module

```python
# composer/graph/builder.py
class GraphBuilder:
    """Constructs LangGraph workflows dynamically."""

    async def build_from_context(
        self,
        user_id: str,
        messages: List[Message],
        tools: List[BaseTool]
    ) -> CompiledGraph:
        # Configuration and model profiles retrieved internally from shared data layer
        # Following pattern: uc = await storage.get_service(storage.user_config).get_user_config(user_id)
        # Model profiles accessed via: get_model_profile_for_task(uc.model_profiles, task_type, user_id)
        
        # Retrieve user config and workflow preferences internally
        uc = await storage.get_service(storage.user_config).get_user_config(user_id)
        workflow_config = uc.workflow_preferences  # Retrieved from shared data layer
        intent = workflow_config.intent_classification
        
        if intent.deep_research:
            return await self.build_research_workflow(user_id, tools)
        elif intent.image_generation:
            return await self.build_creative_workflow(user_id, tools)
        else:
            return await self.build_chat_workflow(user_id, tools)
```

- **Configurability:** Configuration parameters like `search_depth`, `max_sources`, and `retrieve_full_content` are retrieved internally by nodes from the shared data layer using user_id. No config objects are passed between components.

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
        }
        self.dynamic_tools = {}  # id -> tool instance
        self.tool_embeddings = {}  # id -> embedding vector for semantic search

    async def get_tools_for_context(self, intent: IntentAnalysis, user_id: str):
        """Select applicable tools based on intent and context."""
        tools = []
        # 1. Include relevant static tools (conditional standard tool collection)
        for name, tool_cls in self.static_tools.items():
            if self._should_include_static_tool(name, intent):
                tools.append(tool_cls(conversation_ctx))
        # 2. Dynamic tool generation or retrieval
        if intent.needs_dynamic_tool:
            spec = intent.tool_spec
            tool = await self._generate_or_retrieve_dynamic_tool(user_id, spec)
            if tool:
                tools.append(tool)
        return tools

    async def _generate_or_retrieve_dynamic_tool(self, context, spec):
        """Return an existing dynamic tool similar to spec, or generate a new one."""
        # Compute embedding of spec description
        spec_vec = await compute_embedding(spec.description)
        # Find similar existing tool via vector similarity (semantic search)
        best_match_id, score = find_best_match(spec_vec, self.tool_embeddings)
        
        # Use Existing
        if score > 0.9:
            return self.dynamic_tools[best_match_id]
        # Modify or Compose
        elif 0.6 <= score <= 0.9:
            existing_tool = self.dynamic_tools[best_match_id]
            if spec.needs_adjustment:
                return existing_tool.clone_with_new_params(spec.parameters)
            else:
                return existing_tool
        # Create New with Engineering Agent Grammar-Constrained Generation
        else:
            # Use Engineering Agent with structured output for guaranteed valid tool specification
            tool_spec = await self._generate_tool_spec_with_engineering_agent(spec)
            new_tool = await self._compile_tool_from_structured_spec(tool_spec)
            tool_id = new_tool.name
            self.dynamic_tools[tool_id] = new_tool
            self.tool_embeddings[tool_id] = spec_vec
            return new_tool
    
    async def _generate_tool_spec_with_engineering_agent(self, spec) -> DynamicToolSpec:
        """Generate structured tool specification using Engineering Agent with grammar constraints."""
        from utils.grammar_generator import GrammarGenerator
        from models.dynamic_tool_spec import DynamicToolSpec
        
        # Generate grammar for tool specification
        grammar = GrammarGenerator.from_pydantic_model(DynamicToolSpec)
        
        # Create grammar-constrained Engineering Agent pipeline
        engineering_pipeline = await self.pipeline_factory.create_structured_pipeline(
            prompt_template=self._get_engineering_agent_prompt(),
            output_schema=DynamicToolSpec,
            grammar=grammar,
            enable_fallback=True
        )
        
        # Generate structured tool specification via Engineering Agent
        return await engineering_pipeline.execute({"requirement": spec.description})
    
    async def _compile_tool_from_structured_spec(self, tool_spec: DynamicToolSpec):
        """Compile tool from validated structured specification."""
        # Validate Python code syntax before compilation
        try:
            compile(tool_spec.implementation, '<dynamic_tool>', 'exec')
        except SyntaxError as e:
            raise ToolGenerationError(f"Generated tool has syntax errors: {e}")
        
        # Create tool from validated specification
        return self._create_tool_instance(tool_spec)
```

**Key Points:**

- **Data Layer Access Pattern:** Configuration and model profiles retrieved using shared data layer:

  ```python
  # Get model profile for specific task
  mp = await get_model_profile(
      user_id,
      ModelProfileType.Primary
  )

  # Get user configuration
  uc = await storage.get_service(storage.user_config).get_user_config(user_id)
  ```

- **Caching Efficiency:** 3-tier caching (memory -> redis -> postgres) makes repeated config retrieval efficient
- **MCP Integration:** Tools can also be provided via the Model Context Protocol (MCP). By installing `langchain-mcp-adapters`, LangGraph agents can treat MCP-registered tools as first-class, reusing corporate or partner tool definitions.

### 4.1.4 State Manager Module

```python
# composer/state/manager.py
class StateManager:
    """Manages workflow state, checkpoints, and context length."""

    async def create_initial_state(
        self,
        messages: List[Message],
        user_id: str
    ) -> WorkflowState:
        # User configuration retrieved internally from shared data layer
        # Example: uc = await storage.get_service(storage.user_config).get_user_config(user_id)
        # 3-tier caching (memory -> redis -> postgres) makes repeated retrieval efficient
        
        state = WorkflowState(
            messages=messages,
            user_id=user_id,
            # user_config not stored in state - retrieved as needed with user_id
            # ... other initial state fields ...
        )
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
        self.stream = stream

    async def __call__(self, state: WorkflowState) -> WorkflowState:
        profile = self.profile_selector(state)
        pipeline = self.pipeline_factory.get_pipeline(profile, ChatResponse, streaming=self.stream)
        if self.stream:
            async for event in pipeline.astream_events({"messages": state.messages}):
                if event["event"] == "on_chat_model_stream":
                    chunk = event["data"]
                    yield ChatResponse(chunk)
        else:
            response = await pipeline.invoke({"messages": state.messages})
            state.messages.append(response.message)
        return state

class ToolExecutorNode:
    """Executes tool calls produced by the previous agent or tool node."""
    def __init__(self, tools: List[BaseTool]):
        self.tool_node = ToolNode(tools)

    async def __call__(self, state: WorkflowState) -> WorkflowState:
        last_message = state.messages[-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            results = self.tool_node.invoke({"messages": [last_message]})
            state.messages.extend(results['messages'])
        return state

class RAGNode:
    """Embeds latest user message and performs retrieval augmentation."""
    async def __call__(self, state: WorkflowState) -> WorkflowState:
        # Implementation of retrieval logic...
        return state
```

#### 4.2.2 Specialized Nodes

```python
# composer/nodes/specialized.py

class TitleGenerationNode:
    """Generates a conversation title if none exists."""
    async def __call__(self, state: WorkflowState) -> WorkflowState:
        # ... implementation ...
        return state

class EngineeringAgentNode:
    """Engineering Agent that generates or retrieves dynamic tools using grammar-constrained structured output."""
    async def __call__(self, state: WorkflowState) -> WorkflowState:
        # ... implementation using ToolRegistry with Engineering Agent ...
        return state
```

### 4.3 Workflow Definitions

#### 4.3.1 Standard Chat Workflow

```python
# composer/workflows/chat.py

async def build_chat_workflow(user_id: str, tools: List[BaseTool]) -> StateGraph:
    workflow = StateGraph(WorkflowState)

    # Configuration retrieved internally from shared data layer using user_id
    # Example: uc = await storage.get_service(storage.user_config).get_user_config(user_id)
    # Example: chat_config = uc.workflow_preferences.chat_config

    workflow.add_node("rag_enrichment", RAGNode(user_id))
    workflow.add_node("engineering_agent", EngineeringAgentNode(user_id))
    # Primary agent node: enable streaming for UI responsiveness
    # Model profile retrieved internally using user_id
    workflow.add_node("agent", PipelineNode(
        user_id,
        ModelProfileType.Primary,
        stream=True  # stream responses for primary chat agent
    ))
    workflow.add_node("tools", ToolExecutorNode(tools))  # Tools passed within composer

    workflow.set_entry_point("rag_enrichment")
    workflow.add_edge("rag_enrichment", "engineering_agent")
    workflow.add_edge("engineering_agent", "agent")

    def route_after_agent(state):
        return "tools" if state.messages[-1].tool_calls else END
    
    workflow.add_conditional_edges("agent", route_after_agent)
    workflow.add_edge("tools", "agent")

    return workflow.compile()
```

- **Graph Execution:** This workflow is run via `workflow.astream_events(initial_state, version="v2")`. The server consumes this stream:

<!-- end list -->

```python
# Server-side consumption
async for event in workflow.astream_events(initial_state, version="v2"):
    if event["event"] == "on_chat_model_stream":
        chunk = event["data"]
        yield serialize_to_json(chunk) # Send to UI
```

#### 4.3.2 Research Workflow

```python
# composer/workflows/research.py

async def build_research_workflow(user_id: str, tools: List[BaseTool]) -> StateGraph:
    workflow = StateGraph(ResearchState)

    # Configuration retrieved internally from shared data layer using user_id
    # Example: uc = await storage.get_service(storage.user_config).get_user_config(user_id)
    # Example: research_config = uc.workflow_preferences.research_config
    
    # Configuration values retrieved internally via user_id in each node
    workflow.add_node("query_expansion", QueryExpansionNode(user_id))  # Gets depth from user config internally
    workflow.add_node("parallel_search", ParallelSearchNode(user_id))  # Gets max_sources, full_text from user config internally
    workflow.add_node("synthesis", SynthesisNode(user_id))

    workflow.set_entry_point("query_expansion")
    workflow.add_edge("query_expansion", "parallel_search")
    workflow.add_edge("parallel_search", "synthesis")
    workflow.add_edge("synthesis", END)

    return workflow.compile()
```

- **Context-Aware Synthesis:** Before running `SynthesisNode`, the `StateManager` is used to enforce context limits. If total tokens exceed the LLM's window, older content is trimmed or summarized.

-----

## 5\. Implementation Roadmap, File Structure & Checklist

### 5.1 Mandatory Project File Structure

A clean, structured file hierarchy is mandatory for managing complexity. Following inference service patterns:

```text
composer/
├── requirements.txt
├── pyproject.toml             # Python package configuration
├── __init__.py
├── app.py                     # FastAPI application entry point
├── config.py                  # Environment-based configuration
└── core/
    ├── __init__.py
    ├── service.py             # Main ComposerService orchestrator
    ├── errors.py              # Error definitions and handling
    └── error_handler.py       # Circuit breaker and retry logic

├── graph/
│   ├── __init__.py
│   ├── builder.py             # GraphBuilder for dynamic workflow construction
│   ├── cache.py               # Workflow caching with TTL
│   └── state.py               # GraphState Pydantic models with reducers

├── nodes/
│   ├── __init__.py
│   ├── standard.py            # PipelineNode, ToolExecutorNode, RAGNode
│   ├── specialized.py         # MIGRATED FROM: runner agent nodes → TitleGenerationNode, EngineeringAgentNode
│   ├── protected.py           # CircuitProtectedNode wrapper
│   └── rag/
│       ├── __init__.py
│       ├── router.py          # MIGRATED FROM: runner conditional logic → Shallow/deep routing
│       └── executor.py        # MIGRATED FROM: runner RAG orchestration → RAG execution nodes

├── tools/
│   ├── __init__.py
│   ├── registry.py            # ToolRegistry with semantic search
│   ├── static/                # Pre-defined tools
│   │   ├── __init__.py
│   │   ├── search.py
│   │   └── summarization.py
│   └── dynamic/
│       ├── __init__.py
│       ├── generator.py       # LLM-based tool generation
│       └── serializer.py      # LCEL serialization helpers

├── workflows/
│   ├── __init__.py
│   ├── chat.py                # MIGRATED FROM: runner create_graph() methods → Centralized chat workflow
│   ├── research.py            # MIGRATED FROM: runner pipeline orchestration → Research workflow with configurable depth
│   └── multi_agent.py         # MIGRATED FROM: runner agent coordination → Multi-agent orchestration

├── streaming/
│   ├── __init__.py
│   ├── processor.py           # Stream event processing
│   └── api.py                 # WebSocket/SSE endpoints

├── agents/
│   ├── __init__.py
│   ├── classifier_agent.py   # Intent analysis and classification
│   ├── engineering_agent.py   # Dynamic tool generation with grammar constraints
│   └── tool_orchestrator.py   # Tool discovery and composition coordination

├── monitoring/
│   ├── __init__.py
│   ├── metrics.py             # Prometheus metrics
│   └── logging.py             # Structured logging

└── schemas/                   # Generated from YAML schemas
    ├── __init__.py
    └── ...                    # Auto-generated Pydantic models
```

### 5.2 Detailed Migration Strategy

#### Phase 1: Foundation Setup (Week 1-2)

**Environment and Dependencies:**

- [ ] Update `inference/requirements.txt` to LangChain/LangGraph V1 alpha
- [ ] Create `composer/` directory in `inference/` following service patterns
- [ ] Set up `composer/pyproject.toml` and `composer/requirements.txt`
- [ ] Configure `composer/config.py` with environment variable patterns

**Core Structure:**

- [ ] Create composer directory structure as specified
- [ ] Implement `GraphState` Pydantic model with LangGraph reducers
- [ ] Set up basic `ComposerService` in `composer/core/service.py`
- [ ] Configure logging with structured format matching inference service

**Tool Migration:**

- [ ] Move `inference/server/tools/integration.py` → `composer/tools/static/`
- [ ] Move `inference/server/tools/dynamic_tool.py` → `composer/tools/dynamic/generator.py`
- [ ] Extract tool generation logic from server handlers
- [ ] Implement `ToolRegistry` with semantic search similar to existing memory store and query (`server/db/sql/memory/search.sql`). Note: a query exists for this at `server/db/sql/tool/search_tools_by_embedding.sql`.

#### Phase 2: Core Node Implementation (Week 3-4)

**Basic Nodes:**

- [ ] Implement `PipelineNode` wrapping existing pipeline execution
- [ ] Create `ToolExecutorNode` using LangGraph's `ToolNode`
- [ ] Build `RAGNode` with configuration retrieved from shared data layer (not ConversationContext)
- [ ] Implement circuit breaker `CircuitProtectedNode` wrapper

**Specialized Nodes:**

- [ ] Create `TitleGenerationNode` from server logic
- [ ] Build `ClassifierAgent` with structured output schema
- [ ] Implement `EngineeringAgentNode` with grammar-constrained tool generation
- [ ] Integrate Engineering Agent with ToolRegistry for dynamic tool creation

**Graph Builder:**

- [ ] Implement `GraphBuilder.build_chat_workflow()`
- [ ] Create workflow caching with TTL
- [ ] Add conditional routing for RAG depth selection

#### Phase 3: Streaming and Advanced Features (Week 5-6)

**Streaming Implementation:**

- [ ] Configure nodes for appropriate streaming modes
- [ ] Implement `StreamProcessor` for event routing
- [ ] Create WebSocket/SSE endpoints in `composer/streaming/api.py`
- [ ] Test selective streaming with token chunks and state updates

**Advanced Workflows:**

- [ ] Implement research workflow with configurable depth
- [ ] Build multi-agent orchestration with `Command` primitives
- [ ] Add intent-driven tool selection and composition
- [ ] Implement semantic search for tool discovery

#### Phase 4: Integration and Migration (Week 7-8)

**Critical Runner-Composer Migration:**

- [ ] **Extract LangGraph Logic from Runner:** Move all `create_graph()` methods from runner pipelines to composer workflows
- [ ] **Migrate Agent Nodes:** Extract agent node implementations from `runner/pipelines/txt2txt/openai_gpt_oss.py` (780+ lines) to `composer/nodes/`
- [ ] **Simplify Runner Interface:** Remove `LangGraphCapable` protocol from `BasePipelineCore`, convert to pure LLM interface
- [ ] **Update Pipeline Abstractions:** Modify `run_pipeline()` and `stream_pipeline()` to handle only individual LLM calls with grammar constraints
- [ ] **Create Workflow Patterns:** Transform runner's embedded orchestration into reusable composer workflow templates
- [ ] **Migrate Tool Orchestration:** Move tool coordination logic from runner to composer tool registry
- [ ] **Update State Management:** Replace runner's mixed state handling with centralized composer GraphState
- [ ] **Protocol Definitions:** Create clean interfaces for runner-composer communication via Protocol classes
- [ ] **Remove Orchestration Dependencies:** Strip LangGraph imports and workflow logic from runner components
- [ ] **Validate Separation:** Ensure runner can be used independently as simple LLM interface without composer

**Server Integration:**

- [ ] Update `inference/server/handlers/completion.py` to use ComposerService
- [ ] Replace manual orchestration in server handlers (ConversationContext not available in composer)
- [ ] Migrate RAG operations from server to composer nodes
- [ ] Migrate dynamic tool generation to Engineering Agent with grammar constraints
- [ ] Update streaming endpoints to consume composer events

**Legacy Migration:**

- [ ] Feature flag existing vs. new workflow systems
- [ ] Gradual migration of conversation types to composer
- [ ] Deprecate old pipeline orchestration logic
- [ ] Remove manual tool generation from server

#### Phase 5: Testing and Validation (Week 9-10)

**Comprehensive Testing:**

- [ ] Unit tests for all nodes with mocked dependencies
- [ ] Integration tests for complete workflows
- [ ] Performance tests for caching and context management
- [ ] Streaming tests for event ordering and delivery
- [ ] Load tests for concurrent workflow execution

**Documentation:**

- [ ] API documentation for composer service
- [ ] Configuration reference with environment variables
- [ ] Migration guide for existing functionality
- [ ] Troubleshooting guide for common issues

#### Phase 6: Production Deployment (Week 11-12)

**Deployment Preparation:**

- [ ] Production configuration with environment variables
- [ ] Monitoring dashboards for composer metrics
- [ ] Alerting rules for circuit breaker states
- [ ] Performance baselines and SLA definitions

**Rollout Strategy:**

- [ ] Canary deployment with feature flags
- [ ] Gradual traffic migration (10% → 50% → 100%)
- [ ] Performance monitoring and rollback procedures
- [ ] Legacy system deprecation timeline
- [ ] Post-migration cleanup and optimization

### 5.2.1 Migration Success Criteria and Validation

**Architecture Separation Validation:**

- [ ] **Runner Independence:** Verify runner can be imported and used without composer dependencies
- [ ] **No LangGraph in Runner:** Confirm zero LangGraph imports in runner module tree
- [ ] **Composer Orchestration:** All graph creation, agent nodes, and workflows reside in composer
- [ ] **Clean Interfaces:** Protocol-based communication between runner and composer with no tight coupling
- [ ] **Grammar Support:** Runner maintains grammar-constrained generation without orchestration complexity

**Functional Equivalence Validation:**

- [ ] **API Compatibility:** All existing endpoints work with same request/response formats  
- [ ] **Streaming Parity:** Streaming behavior identical to pre-migration functionality
- [ ] **Performance Baseline:** Response times within 10% of pre-migration performance
- [ ] **Error Handling:** Error scenarios handled consistently with improved circuit breaker support
- [ ] **Grammar Constraints:** Structured output generation works across all workflow types

**Code Quality Metrics:**

- [ ] **Test Coverage:** >90% coverage for composer components, >95% for runner interfaces
- [ ] **Cyclomatic Complexity:** All functions <10 complexity, classes <15
- [ ] **Import Dependencies:** Clean dependency graph with no circular imports
- [ ] **Code Duplication:** <5% duplication between runner and composer codebases
- [ ] **Documentation:** All public APIs documented with examples and type hints

**Migration Completeness Checklist:**

```python
# Automated validation script: debug/validate_migration_completeness.py
async def validate_runner_simplification():
    """Automated checks for successful runner simplification."""
    
    # Check 1: No orchestration methods in runner
    forbidden_methods = ['create_graph', 'build_workflow', 'orchestrate']
    runner_violations = scan_for_methods(runner.pipelines, forbidden_methods)
    assert len(runner_violations) == 0, f"Orchestration methods found in runner: {runner_violations}"
    
    # Check 2: All LangGraph logic in composer  
    required_composer_components = [
        'composer.workflows.chat.build_chat_workflow',
        'composer.nodes.specialized.IntentClassifierNode', 
        'composer.agents.tool_orchestrator.ToolOrchestratorAgent'
    ]
    for component in required_composer_components:
        assert importable(component), f"Missing composer component: {component}"
    
    # Check 3: Runner interface simplicity
    runner_interface = get_public_api(runner)
    allowed_methods = ['generate', 'stream_generate', 'validate_grammar', 'health_check']
    extra_methods = set(runner_interface) - set(allowed_methods)
    assert len(extra_methods) == 0, f"Extra methods in runner interface: {extra_methods}"
    
    # Check 4: Grammar support maintained
    response = await runner.generate(
        messages=[test_message], 
        grammar=IntentAnalysis
    )
    assert response.structured_output is not None, "Grammar constraints not working"
    
    print("✅ Migration validation passed - runner successfully simplified")

async def validate_composer_orchestration():
    """Validate composer contains all orchestration capabilities."""
    
    # Check 1: Workflow construction
    workflow = await composer.compose_workflow(
        user_id="test", conversation_id=1, 
        workflow_type=WorkflowType.CHAT, messages=[]
    )
    assert isinstance(workflow, CompiledGraph), "Composer cannot create workflows"
    
    # Check 2: Agent node execution
    from composer.nodes.specialized import IntentClassifierNode
    intent_node = IntentClassifierNode(mock_pipeline_factory)
    result = await intent_node(test_state)
    assert result.intent_classification is not None, "Agent nodes not working"
    
    # Check 3: Tool orchestration
    from composer.agents.tool_orchestrator import ToolOrchestratorAgent
    orchestrator = ToolOrchestratorAgent()
    tools = await orchestrator.coordinate_tools(test_context)
    assert len(tools) > 0, "Tool orchestration not working"
    
    print("✅ Composer orchestration validation passed")
```

**Performance Impact Acceptance Criteria:**

- [ ] **Latency:** First token time <2s for simple queries, <10s for complex workflows
- [ ] **Throughput:** Support >100 concurrent conversations with <5% resource increase
- [ ] **Memory Usage:** Composer memory overhead <500MB, runner memory usage <200MB
- [ ] **Error Recovery:** Circuit breaker recovery time <30s, state persistence >99.9%
- [ ] **Cache Efficiency:** Workflow cache hit rate >80%, tool discovery cache >90%

### 5.3 Testing Requirements

#### 5.3.1 Unit Tests

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
    workflow = await build_chat_workflow(user_id, tools)
    result = await workflow.ainvoke(initial_state)
    assert result.messages[-1].role == MessageRole.ASSISTANT

# tests/composer/test_tool_registry.py
async def test_semantic_tool_search():
    """Test tool discovery via semantic similarity"""
    registry = ToolRegistry()
    spec = ToolSpec(description="fetch weather data")
    tool = await registry.find_or_create_tool(spec)
    assert tool.name == "weather_tool"
```

#### 5.3.2 Integration Tests

```python
# tests/integration/test_composer_integration.py
async def test_end_to_end_chat():
    """Test full chat flow through composer"""
    composer = ComposerService()
    
    workflow = await composer.compose_workflow(
        user_id="test_user_123",
        conversation_id=456,
        workflow_type=WorkflowType.CHAT,
        messages=[test_message]
    )
    
    result = await workflow.ainvoke({
        "messages": [test_message]
    })
    
    assert result["messages"][-1].role == MessageRole.ASSISTANT

# tests/integration/test_streaming.py
async def test_selective_streaming():
    """Verify streaming modes work correctly"""
    workflow = await build_chat_workflow(user_id, tools)  # Streaming config retrieved from user preferences
    events = []
    
    async for event in workflow.astream_events(initial_state, version="v2"):
        events.append(event)
    
    # Verify token chunks from chat node
    token_events = [e for e in events if e["event"] == "on_chat_model_stream"]
    assert len(token_events) > 0
    
    # Verify state updates from other nodes
    update_events = [e for e in events if e["event"] == "on_chain_end"]
    assert len(update_events) > 0
```

#### 5.3.3 Performance Tests

```python
# tests/performance/test_workflow_performance.py
async def test_workflow_caching():
    """Test that workflows are properly cached"""
    composer = ComposerService()
    
    start_time = time.time()
    workflow1 = await composer.compose_workflow(
        user_id="test_user", conversation_id=123, 
        workflow_type=WorkflowType.CHAT, messages=[test_message]
    )
    first_creation_time = time.time() - start_time
    
    start_time = time.time()
    workflow2 = await composer.compose_workflow(
        user_id="test_user", conversation_id=123,
        workflow_type=WorkflowType.CHAT, messages=[test_message]
    )
    second_creation_time = time.time() - start_time
    
    # Second call should be much faster (cached)
    assert second_creation_time < first_creation_time / 10
    assert workflow1 is workflow2

async def test_rag_depth_performance():
    """Test RAG performance with different depths"""
    # Configuration retrieved internally from user preferences in shared data layer
    # Test with different user configs that have different research preferences
    
    shallow_workflow = await build_research_workflow(user_id_shallow, tools)
    deep_workflow = await build_research_workflow(user_id_deep, tools)
    
    # Shallow should be faster
    shallow_time = await measure_execution_time(shallow_workflow, test_state)
    deep_time = await measure_execution_time(deep_workflow, test_state)
    
    assert shallow_time < deep_time
    assert shallow_time < 5.0  # Under 5 seconds
```

#### 5.3.3 Migration-Specific Testing Requirements

**Runner Simplification Tests:**

```python
# tests/migration/test_runner_simplification.py
async def test_runner_pure_interface():
    """Verify runner only provides pure LLM interface post-migration."""
    runner = SimplifiedRunnerInterface()
    
    # Should work: simple LLM generation
    response = await runner.generate(
        messages=[Message(role=MessageRole.USER, content="Hello")],
        grammar=None
    )
    assert isinstance(response, ChatResponse)
    
    # Should work: grammar-constrained generation
    structured_response = await runner.generate(
        messages=[Message(role=MessageRole.USER, content="Analyze intent")],
        grammar=IntentAnalysis
    )
    assert structured_response.structured_output is not None
    
    # Should NOT exist: orchestration methods removed
    with pytest.raises(AttributeError):
        _ = runner.create_graph()  # Should be removed
    
    with pytest.raises(AttributeError):
        _ = runner.build_workflow()  # Should be removed

async def test_no_langgraph_dependencies_in_runner():
    """Ensure runner has no LangGraph imports post-migration."""
    import ast
    import inspect
    
    # Analyze runner module imports
    runner_module = importlib.import_module("runner.pipelines.base")
    source = inspect.getsource(runner_module)
    tree = ast.parse(source)
    
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend([alias.name for alias in node.names])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    
    # Assert no LangGraph imports
    langgraph_imports = [imp for imp in imports if "langgraph" in imp.lower()]
    assert len(langgraph_imports) == 0, f"Found LangGraph imports: {langgraph_imports}"

async def test_composer_orchestration_migration():
    """Verify all orchestration logic moved to composer."""
    composer = ComposerService()
    
    # Composer should handle workflow creation
    workflow = await composer.compose_workflow(
        user_id="test_user",
        conversation_id=123,
        workflow_type=WorkflowType.CHAT,
        messages=[test_message]
    )
    assert isinstance(workflow, CompiledGraph)
    
    # Composer should handle agent nodes
    workflow_def = await composer.graph_builder.build_from_context(
        user_id="test_user",
        messages=[test_message],
        tools=[]
    )
    
    # Verify agent nodes are in composer, not runner
    node_names = list(workflow_def.nodes.keys())
    assert "classifier_agent" in node_names
    assert "tool_orchestrator" in node_names
    assert "rag_executor" in node_names

async def test_backward_compatibility():
    """Ensure migration maintains API compatibility."""
    # Legacy runner interfaces should still work but be simplified
    from runner import run_pipeline, stream_pipeline
    
    # Should work with simplified interface
    response = await run_pipeline(
        messages=[test_message],
        pipeline_name="openai-gpt-4",
        grammar=None
    )
    assert isinstance(response, ChatResponse)
    
    # Streaming should work
    async for chunk in stream_pipeline(
        messages=[test_message],
        pipeline_name="openai-gpt-4"
    ):
        assert isinstance(chunk, ChatResponse)
        break  # Just test first chunk
```

**Migration Validation Tests:**

```python
# tests/migration/test_extraction_completeness.py
async def test_all_create_graph_methods_removed():
    """Verify all create_graph methods removed from runner."""
    import pkgutil
    import runner.pipelines
    
    for importer, modname, ispkg in pkgutil.walk_packages(
        runner.pipelines.__path__, runner.pipelines.__name__ + "."
    ):
        if not ispkg:
            module = importlib.import_module(modname)
            # Check all classes in module
            for name, cls in inspect.getmembers(module, inspect.isclass):
                if hasattr(cls, 'create_graph'):
                    raise AssertionError(
                        f"Found create_graph method in {modname}.{name} - should be migrated to composer"
                    )

async def test_langgraph_capable_protocol_removed():
    """Verify LangGraphCapable protocol removed from runner."""
    try:
        from runner.pipelines.base import LangGraphCapable
        raise AssertionError("LangGraphCapable protocol still exists in runner - should be removed")
    except ImportError:
        pass  # Expected - protocol should be removed

async def test_composer_has_orchestration_logic():
    """Verify composer contains all orchestration logic."""
    # Check composer has workflow patterns
    from composer.workflows.chat import build_chat_workflow
    from composer.nodes.specialized import IntentClassifierNode
    from composer.agents.tool_orchestrator import ToolOrchestratorAgent
    
    # Should be able to create workflows
    workflow = await build_chat_workflow("test_user", [])
    assert workflow is not None
    
    # Should have agent nodes
    intent_node = IntentClassifierNode(mock_pipeline_factory)
    assert callable(intent_node)
    
    # Should have tool orchestration
    orchestrator = ToolOrchestratorAgent()
    assert hasattr(orchestrator, 'coordinate_tools')
```

### 5.4 Error Handling and Circuit Breaker Implementation

```python
# composer/core/errors.py
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

class CircuitOpenError(ComposerError):
    """Circuit breaker is open"""
    pass

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
            raise NodeExecutionError(self.node.__class__.__name__, e)
    
    def _is_circuit_open(self) -> bool:
        if self.failure_count < self.circuit_config.failure_threshold:
            return False
        
        if self.last_failure_time is None:
            return False
        
        time_since_failure = time.time() - self.last_failure_time
        return time_since_failure < self.circuit_config.recovery_timeout
    
    def _on_success(self):
        self.failure_count = 0
        self.last_failure_time = None
    
    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
```

### 5.5 Monitoring and Observability

```python
# composer/monitoring/metrics.py
from prometheus_client import Histogram, Counter, Gauge

class ComposerMetrics:
    workflow_creation_time = Histogram('composer_workflow_creation_seconds')
    node_execution_time = Histogram('composer_node_execution_seconds', ['node_type'])
    tool_generation_count = Counter('composer_tool_generation_total', ['tool_type'])
    workflow_cache_hits = Counter('composer_cache_hits_total')
    workflow_cache_misses = Counter('composer_cache_misses_total')
    active_workflows = Gauge('composer_active_workflows')
    streaming_events_total = Counter('composer_streaming_events_total', ['event_type'])

# composer/monitoring/logging.py
import structlog

class WorkflowLogger:
    def __init__(self):
        self.logger = structlog.get_logger("composer")
    
    def log_workflow_start(self, workflow_id: str, workflow_type: str, user_id: str):
        self.logger.info(
            "Workflow started",
            workflow_id=workflow_id,
            workflow_type=workflow_type,
            user_id=user_id,
            timestamp=datetime.now().isoformat()
        )
    
    def log_node_execution(self, node_name: str, duration: float, success: bool):
        self.logger.debug(
            "Node executed",
            node_name=node_name,
            duration_ms=duration * 1000,
            success=success
        )
    
    def log_tool_generation(self, tool_spec: str, method: str, success: bool):
        self.logger.info(
            "Tool generation",
            tool_spec=tool_spec,
            method=method,  # "existing", "modified", "new"
            success=success
        )
```

### 5.6 Configuration Management

```python
# composer/config.py
from dataclasses import dataclass, field
from typing import Optional
import os
from schemas.circuit_breaker_config import CircuitBreakerConfig
from schemas.memory_config import MemoryConfig

@dataclass
class ComposerConfig:
    # Workflow settings
    enable_workflow_caching: bool = True
    workflow_cache_ttl: int = 3600
    max_parallel_tools: int = 5
    enable_multi_agent: bool = False
    default_timeout: float = 60.0
    
    # Memory and performance
    max_context_length: int = 128000
    context_trim_threshold: float = 0.8
    
    # Tool settings
    tool_similarity_threshold: float = 0.9
    tool_modification_threshold: float = 0.6
    enable_tool_generation: bool = True
    
    # Circuit breaker
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    
    # Streaming
    enable_streaming: bool = True
    stream_buffer_size: int = 1024
    
    @classmethod
    def from_env(cls) -> 'ComposerConfig':
        """Load configuration from environment variables"""
        return cls(
            enable_workflow_caching=os.getenv('COMPOSER_ENABLE_CACHE', 'true').lower() == 'true',
            workflow_cache_ttl=int(os.getenv('COMPOSER_CACHE_TTL', '3600')),
            max_parallel_tools=int(os.getenv('COMPOSER_MAX_PARALLEL_TOOLS', '5')),
            enable_multi_agent=os.getenv('COMPOSER_ENABLE_MULTI_AGENT', 'false').lower() == 'true',
            default_timeout=float(os.getenv('COMPOSER_DEFAULT_TIMEOUT', '60.0')),
            max_context_length=int(os.getenv('COMPOSER_MAX_CONTEXT_LENGTH', '128000')),
            tool_similarity_threshold=float(os.getenv('COMPOSER_TOOL_SIMILARITY_THRESHOLD', '0.9')),
            enable_tool_generation=os.getenv('COMPOSER_ENABLE_TOOL_GENERATION', 'true').lower() == 'true',
            enable_streaming=os.getenv('COMPOSER_ENABLE_STREAMING', 'true').lower() == 'true'
        )
```

### 5.7 Structured Output Integration

#### Grammar-Constrained LLM Output System

The refactored system integrates **llamacpp grammar constraints** to ensure reliable, type-safe structured output across all LLM interactions. This eliminates JSON parsing errors and provides guaranteed validation.

##### Core Structured Output Components

```python
# composer/utils/structured_output.py
from typing import Type, TypeVar, Dict, Any
from pydantic import BaseModel
from utils.grammar_generator import GrammarGenerator

T = TypeVar('T', bound=BaseModel)

class StructuredOutputPipeline:
    """Pipeline wrapper for grammar-constrained structured output."""
    
    def __init__(self, pipeline_factory):
        self.pipeline_factory = pipeline_factory
        self._grammar_cache: Dict[str, str] = {}
    
    async def create_structured_pipeline(
        self,
        prompt_template: str,
        output_schema: Type[T],
        enable_fallback: bool = True,
        max_retries: int = 3,
        **kwargs
    ) -> 'StructuredPipeline[T]':
        """Create grammar-constrained pipeline for structured output."""
        
        # Generate or retrieve cached grammar
        schema_key = f"{output_schema.__name__}_{hash(str(output_schema.schema()))}"
        if schema_key not in self._grammar_cache:
            self._grammar_cache[schema_key] = GrammarGenerator.from_pydantic_model(output_schema)
        
        grammar = self._grammar_cache[schema_key]
        
        # Create grammar-constrained pipeline
        pipeline = await self.pipeline_factory.create_grammar_pipeline(
            prompt_template=prompt_template,
            grammar=grammar,
            output_schema=output_schema,
            enable_fallback=enable_fallback,
            max_retries=max_retries,
            **kwargs
        )
        
        return StructuredPipeline(pipeline, output_schema)

class StructuredPipeline:
    """Grammar-constrained pipeline with type-safe execution."""
    
    def __init__(self, pipeline, output_schema: Type[T]):
        self.pipeline = pipeline
        self.output_schema = output_schema
    
    async def execute(self, inputs: Dict[str, Any]) -> T:
        """Execute pipeline with guaranteed structured output."""
        try:
            # Execute with grammar constraints
            result = await self.pipeline.invoke(inputs)
            
            # Validate and parse result
            if isinstance(result, self.output_schema):
                return result
            elif isinstance(result, dict):
                return self.output_schema(**result)
            else:
                # Parse JSON string result
                import json
                data = json.loads(str(result)) if isinstance(result, str) else result
                return self.output_schema(**data)
                
        except Exception as e:
            raise StructuredOutputError(f"Failed to generate structured output: {e}")

class StructuredOutputError(Exception):
    """Exception raised when structured output generation fails."""
    pass
```

##### Integration with Agent Nodes

All agent nodes use structured output through base class integration:

```python
# composer/nodes/base_structured_node.py
from abc import ABC, abstractmethod
from typing import Type, TypeVar, Generic
from pydantic import BaseModel

T = TypeVar('T', bound=BaseModel)

class BaseStructuredNode(ABC, Generic[T]):
    """Base class for nodes with structured output requirements."""
    
    def __init__(self, output_schema: Type[T], pipeline_factory):
        self.output_schema = output_schema
        self.structured_output = StructuredOutputPipeline(pipeline_factory)
    
    @abstractmethod
    def get_prompt_template(self) -> str:
        """Return the prompt template for structured output generation."""
        pass
    
    async def execute_structured(
        self,
        inputs: Dict[str, Any],
        **kwargs
    ) -> T:
        """Execute node with guaranteed structured output."""
        
        pipeline = await self.structured_output.create_structured_pipeline(
            prompt_template=self.get_prompt_template(),
            output_schema=self.output_schema,
            **kwargs
        )
        
        return await pipeline.execute(inputs)
```

##### Structured Output Configuration

```python
# composer/config.py (addition)
@dataclass
class StructuredOutputConfig:
    """Configuration for grammar-constrained output."""
    enable_grammar_constraints: bool = True
    fallback_on_grammar_failure: bool = True
    max_grammar_retries: int = 3
    grammar_cache_size: int = 100
    grammar_validation_strict: bool = False
    
    # Performance settings
    grammar_generation_timeout: float = 10.0
    structured_parsing_timeout: float = 5.0
```

### 5.8 Schema Integration and Code Generation

Following the existing codebase pattern of YAML-driven development with structured output support:

#### Required Schema Definitions

Add to `schemas/` directory:

```yaml
# schemas/composer_config.yaml
$schema: http://json-schema.org/draft-07/schema#
title: ComposerConfig
description: Composer service configuration
type: object
properties:
  enable_workflow_caching:
    type: boolean
    default: true
  workflow_cache_ttl:
    type: integer
    default: 3600
  max_parallel_tools:
    type: integer
    default: 5
  enable_multi_agent:
    type: boolean
    default: false
  circuit_breaker:
    $ref: circuit_breaker_config.yaml

# schemas/workflow_state.yaml  
$schema: http://json-schema.org/draft-07/schema#
title: WorkflowState
description: LangGraph workflow state schema with structured output support
type: object
properties:
  messages:
    type: array
    items:
      $ref: lang_chain_message.yaml
  intent_classification:
    $ref: intent_analysis.yaml
  required_tools:
    type: array
    items:
      $ref: tool.yaml
  progress_updates:
    type: array
    items:
      type: string

# schemas/structured_output_config.yaml
$schema: http://json-schema.org/draft-07/schema#
title: StructuredOutputConfig
description: Configuration for grammar-constrained structured output
type: object
properties:
  enable_grammar_constraints:
    type: boolean
    default: true
  fallback_on_grammar_failure:
    type: boolean
    default: true
  max_grammar_retries:
    type: integer
    default: 3
    minimum: 1
    maximum: 10
  grammar_cache_size:
    type: integer
    default: 100
    minimum: 10
  grammar_validation_strict:
    type: boolean
    default: false
    type: array
    items:
      $ref: available_tool.yaml
  progress_updates:
    type: array
    items:
      type: string

# schemas/composer_request.yaml
$schema: http://json-schema.org/draft-07/schema#  
title: ComposerRequest
description: Request to compose workflow (ConversationContext not available - use shared data layer)
type: object
properties:
  user_id:
    type: string
    description: User identifier for retrieving config from shared data layer
  conversation_id:
    type: integer
    description: Conversation identifier
  messages:
    type: array
    items:
      $ref: message.yaml
  workflow_type:
    type: string
    enum: ["CHAT", "RESEARCH", "MULTI_AGENT", "CREATIVE"]
  config_overrides:
    type: object
required:
  - user_id
  - conversation_id
  - messages
  - workflow_type
```

#### Code Generation Integration

Update `regenerate_models.sh` to include composer schemas:

```bash
# Add to regenerate_models.sh
echo "Generating composer models..."
datamodel-codegen \
    --input schemas/composer_config.yaml \
    --output composer/schemas/composer_config.py

datamodel-codegen \
    --input schemas/workflow_state.yaml \
    --output composer/schemas/workflow_state.py
```

### 5.8 Final Integrity Check: Compliance with LangGraph V1 Concepts

The proposed architecture fully complies with and leverages the core capabilities of the LangGraph V1 framework.

1. **State Management:** The defined `GraphState` correctly employs reducer functions to manage complex state fields, such as concatenating `messages`, ensuring conversation context is consistently maintained across all nodes.
2. **Execution Control:** Conditional execution, achieved by reading `rag_depth_config` and `required_tools` from the state, is implemented through LangGraph's conditional edges, granting precise, programmatic control over the workflow.
3. **Streaming Paradigm:** The architecture correctly mandates the use of distinct streaming modes—`messages` for token output and `updates`/`custom` for state feedback—to deliver the selective streaming functionality required for optimal user experience and operational transparency.
4. **Error Resilience:** Circuit breaker patterns and comprehensive error handling ensure system stability under failure conditions.
5. **Observability:** Comprehensive metrics and structured logging provide full visibility into system performance and behavior.
6. **Schema Compliance:** Integration with existing YAML schema patterns ensures type safety and consistency across services.

-----

## 6\. Success Criteria and Validation

### 6.1 Performance Metrics

- **Workflow Creation:** <100ms with cache hits, <2s for cache misses
- **Node Execution Overhead:** <5ms per node transition
- **Streaming Latency:** <50ms for first token, <10ms between tokens
- **Memory Usage:** 20% reduction through context management and workflow caching
- **Tool Generation:** <3s for new tool creation, <500ms for existing tool retrieval

### 6.2 Code Quality Standards

- **Cyclomatic Complexity:** Reduce by 40% through modular node architecture
- **Test Coverage:** >90% for all composer modules
- **Type Safety:** 100% type hints with Pydantic model validation
- **Documentation:** API documentation for all public interfaces
- **Linting:** Pass all pyright, pylint, and black formatting checks

### 6.3 Maintainability Goals

- **Workflow Creation:** New workflow types implementable in <1 hour
- **Tool Addition:** Static tools added without server code changes
- **Schema Evolution:** Backward-compatible schema updates via YAML versioning
- **Configuration Management:** All settings configurable via environment variables
- **Deployment:** Zero-downtime deployments with feature flags

### 6.4 Reliability Requirements

- **Uptime:** 99.9% availability with graceful degradation
- **Error Recovery:** Circuit breakers prevent cascading failures
- **State Persistence:** Durable execution survives client disconnections
- **Monitoring:** Full observability with metrics, logging, and alerting
- **Testing:** Comprehensive unit, integration, and performance test suites

### 6.5 Codebase Integration

- **Schema Compliance:** Use existing YAML schema generation patterns
- **Configuration:** Follow inference service environment variable patterns
- **Logging:** Integrate with existing structured logging framework
- **Database:** Leverage existing PostgreSQL connection patterns
- **Authentication:** Integrate with existing Auth0/JWT authentication
- **API Versioning:** Follow existing API versioning conventions

-----

## 7\. Codebase Integration Patterns

### 7.1 Existing Infrastructure Integration

**Database Integration:**

- Use existing PostgreSQL connection patterns from `inference/server/config.py`
- Leverage existing database schema in `inference/server/db/sql/`
- Follow connection pooling patterns from server service
- Access user configuration and model profiles through shared data layer using user_id (ConversationContext not available)
- Follow existing pattern: `await storage.get_service(storage.user_config).get_user_config(user_id)`
- Use `get_model_profile_for_task()` utility for model profile retrieval
- Leverage 3-tier caching system for efficient repeated access

**Configuration Management:**

- Follow environment variable patterns from `schemas/config.yaml`
- Use existing configuration validation with Pydantic models
- Integrate with current API versioning system

**Authentication and Authorization:**

- Use existing Auth0/JWT patterns from `inference/server/auth.py`
- Maintain current user management and session handling
- Preserve existing API security patterns

**Service Communication:**

- Follow existing FastAPI patterns from `inference/server/app.py`
- Use current middleware and error handling patterns
- Maintain API versioning conventions

### 7.2 Development Workflow Integration

**Code Generation:**

- Integrate composer schemas into `./regenerate_models.sh`
- Follow existing YAML schema patterns for type safety
- Use current Python and TypeScript model generation

**Testing Patterns:**

- Follow existing test structure in `inference/test/`
- Use current pytest configuration and fixtures  
- Maintain existing test database patterns

**Deployment Integration:**

- Follow Kubernetes deployment patterns from `inference/k8s/`
- Use existing Docker build patterns from `inference/Dockerfile`
- Integrate with current CI/CD pipeline

**Monitoring Integration:**

- Use existing structured logging patterns
- Integrate with current Prometheus metrics collection
- Follow established alerting and dashboard patterns

## 8\. References

### 8.1 LangGraph and LangChain Documentation

- LangChain and LangGraph Enter v1.0 Alpha: A New Era for Agentic AI Development
- LangChain & LangGraph 1.0 alpha releases
- LangGraph - LangChain
- How to stream state updates of your graph (LangGraphJS Docs)
- What's possible with LangGraph streaming - Overview (LangGraph Docs)
- Streaming - LangChain Python Documentation
- Adaptive RAG with local LLMs (LangGraphJS Tutorials)
- Dynamic tool calling in LangGraph agents - LangChain Changelog
- How to chain runnables | LangChain Python Documentation
- How to create tools | LangChain Python Documentation
- Tools | LangChain Concepts

### 8.2 Project-Specific References

- LLM ML Lab Project Architecture (see `.github/copilot-instructions.md`)
- Context Extension System Documentation (`docs/context_extension.md`)
- GPU Configuration Guide (`docs/gpu_configuration.md`)
- Pipeline Implementation Guide (`docs/PIPELINE_IMPLEMENTATION_GUIDE.md`)
- Pipeline API Reference (`docs/PIPELINE_API_REFERENCE.md`)
- Existing Schema Definitions (`schemas/*.yaml`)
- Current Service Architecture (`inference/server/`, `inference/runner/`)

### 8.3 External Framework References

- LangChain & LangGraph Official Tutorials: [https://github.com/langchain-ai/langchain](https://github.com/langchain-ai/langchain)
- LangGraph v1.0 Alpha Documentation: [https://langchain.ai/docs/langgraph/v1](https://langchain.ai/docs/langgraph/v1)
- LangGraph Streaming Event Specification: [https://langchain.github.io/langgraph/posts/streaming](https://langchain.github.io/langgraph/posts/streaming)
- LangChain Multi-Agent Cookbook: [https://docs.langchain.com/docs/multiagent](https://docs.langchain.com/docs/multiagent)
- Adaptive RAG Design Patterns: [https://langchain.ai/blog/adaptive_rag/](https://langchain.ai/blog/adaptive_rag/)
- Dynamic Tool Generation Concepts: [https://langchain.ai/blog/dynamic_tools](https://langchain.ai/blog/dynamic_tools)
- Model Context Protocol (MCP) Overview: [https://github.com/langchain-ai/mcp/](https://github.com/langchain-ai/mcp/)
