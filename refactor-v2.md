

# **Comprehensive Agent System Refactoring Specification (LangGraph V1 Mandate)**

## **I. Executive Summary and V1 Architectural Validation**

### **1.1 Confirmation of Feasibility and Integrity Check**

The analysis confirms that the specified refactoring requirements—incorporating selective streaming, implementing configurable Retrieval-Augmented Generation (RAG) depth, and establishing a system for dynamic, composable tool management—are entirely feasible and maintain high system integrity. These requirements align precisely with the core design principles of the LangGraph V1 alpha framework, which is engineered for production-grade agent orchestration and advanced workflow control.1

The foundation of system integrity is provided by the LangGraph V1 alpha's features, particularly its **Durable Execution** capability. This feature provides a built-in agent runtime that ensures state continuity and reliability.1 This resilience is essential because the architecture mandates moving the actual execution of the complex agent graph into the remote

composer service. Furthermore, LangGraph V1 emphasizes **Execution Control**, enabling developers to design custom workflows and exercise "fine-tuned control over execution".1 This programmatic control is the prerequisite for implementing conditional logic that governs dynamic tool selection and the flexible routing necessary for configurable RAG depth. The requirements do not necessitate any breaking changes in the core LangGraph framework, as they leverage capabilities that are being promoted to 1.0 status due to their stability and widespread use in production systems.2

### **1.2 High-Level Architectural Shift: Moving Execution to the Composer Service**

The refactoring dictates a crucial architectural shift: the composer project must transition from merely defining the graph structure to serving as the primary, authoritative execution runtime. This necessitates robust state persistence within the composer service. By relying on LangGraph V1's promise of durable execution, the composer can manage lengthy, multi-step operations—such as complex web crawls, iterative RAG, or the creation of new tools—without the risk of external state loss, even if the client connection is interrupted temporarily.1

This architectural decision significantly enhances system resilience by decoupling the UI's front-end responsiveness from the backend's computational load. The composer service will exclusively host the complex LangGraph application, managing the central graph state, executing all node functions, and enforcing dynamic conditional flow. Consequently, all core LangGraph logic, including the foundational state definitions, node implementations, and the final compiled graph artifact (app.compile()), must be securely housed within a dedicated structure, specifically recommended to be the composer/agent\_runtime/ directory. Communication with the client is managed through a specialized, single asynchronous endpoint designed to handle the required selective data streaming.

### **1.3 Unified Graph State Schema Design**

A foundational requirement for a sophisticated, durable LangGraph system is the definition of a single, centralized GraphState schema. This state model acts as the common interface for all nodes, ensuring data consistency and enabling conditional routing by maintaining context across execution steps.4 The following elements are mandatory for defining the authoritative

GraphState, typically implemented as a Pydantic model located in composer/agent\_runtime/state.py.

Table: Mandatory Elements for GraphState

| Field Name | Type | Reducer Function (LangGraph) | Purpose |
| :---- | :---- | :---- | :---- |
| messages | List | (x, y) \=\> x.concat(y) | Conversation history and final outputs, essential for context and token streaming.4 |
| intent\_classification | IntentSchema | lambda x, y: y | Structured output from the Intent Agent, directing subsequent RAG and tool decisions. |
| required\_tools | List | lambda x, y: y | The curated list of standard and dynamic tools collected for the current execution phase. |
| search\_results | str | None | lambda x, y: y | The consolidated, synthesized output from RAG execution (whether shallow or deep). |
| rag\_depth\_config | str | lambda x, y: y | Stores the decision ('SHALLOW' or 'DEEP'), which drives the conditional edge for RAG routing. |
| progress\_updates | List | (x, y) \=\> x.concat(y) | User-defined signals used for granular progress tracking during tool or crawl execution.5 |

## **II. Requirement 1: Execution Control and Asynchronous Streaming Architecture**

The first requirement mandates a precise control over data delivery: streaming real-time tokens from the primary conversational node while relying on completed, structured text outputs from all upstream agent nodes. LangGraph’s native streaming system supports this level of granularity by offering multiple streaming modes.5

### **2.1 Implementation of Selective Streaming**

To implement selective data delivery, each node within the graph must be configured to return data compatible with a specific LangGraph streaming mode.

The **Primary Chat Generator Node**, which produces the final response to the user, must be configured to stream its output using the messages mode. This mode is specifically designed to deliver LLM tokens in real time, accompanied by necessary metadata, thereby fulfilling the core mandate of streaming the main chat operation back to the UI.5 This capability is central to managing user experience by reducing perceived latency, as output is displayed progressively.6

In contrast, intermediate **Agent Nodes**—such as the Intent Classifier, Tool Collector, and RAG Executor—generate intermediate logic and synthesized resources. These nodes should return structured data that updates the GraphState (e.g., updating intent\_classification or search\_results). Their streaming behavior must be configured to utilize either the updates mode, which streams state deltas, or the custom mode, which allows for the emission of arbitrary user-defined signals.4 Streaming state deltas or custom progress notifications, such as "Tool Selection Agent Running" or "Executing Deep Web Crawl," provides necessary operational transparency.6 Without this granular feedback, the user would experience silent waiting periods during the most computationally intensive phases (tool discovery, deep RAG). By streaming these intermittent updates, the system significantly improves user experience by actively communicating workflow progress.

### **2.2 Streaming Modes Configuration**

The detailed configuration ensures that output from each node type is optimized for its role, balancing low-latency token streaming with crucial workflow transparency.

Table: Streaming Node Behavior and Output Modes

| Node Type | Role/Function | Required Streaming Mode (LangGraph) | Data Payload | Impact on UX |  |
| :---- | :---- | :---- | :---- | :---- | :---- |
| Primary Chat Generator | Final Conversational LLM Response. | messages (LLM tokens \+ metadata) | Real-time token chunks. | Real-time token display (low perceived latency). |  |
| Intent Classification Agent | Initial decision, intent parsing, tool request schema definition. | updates (State Delta) | Update to intent\_classification and rag\_depth\_config. | Status update ("Analyzing intent...") upon completion. |  |
| Dynamic Tool Agent (DTA) | Tool search, composition, and creation. | updates and custom (Progress signal) | Updates to required\_tools. Custom signal: "Tool registry accessed (ID: X)." | Transparent tracking of dynamic tool assembly process. |  |
| Deep RAG Executor Node | Executes resource-intensive crawl/synthesize. | updates or custom | Custom signal: "Fetched 10/100 records".5 Update to | search\_results. | Granular progress display during high-latency RAG. |

### **2.3 Integration Blueprint: Composer Service and UI**

Effective implementation of selective streaming requires a robust transport layer between the composer service and the UI client. The composer must manage the stream iterator generated by LangGraph's .stream() method via a persistent connection interface, such as WebSockets or Server-Sent Events (SSE).

The workflow involves the client sending the initial prompt to the composer. The composer service then executes the LangGraph application, yielding a continuous stream of distinct outputs (tokens, state deltas, custom signals).5 The service is responsible for serializing these diverse payloads using a consistent envelope, often including a type discriminator (e.g.,

{"type": "token\_chunk", "data":...}). The UI client then parses this envelope and routes the data accordingly: token chunks are streamed immediately to the chat interface, while state updates or custom signals are used to update status banners or progress indicators, ensuring real-time visibility into the workflow's progression.

## **III. Requirement 2: Configurable Knowledge Retrieval Pipeline (Adaptive RAG Specification)**

Requirement 2 addresses the need for configurable knowledge retrieval, moving away from a fixed, deep RAG operation for every query. This functionality is achieved through an Adaptive RAG pattern, relying on LangGraph's powerful ability to route execution flow using conditional edges.7 The depth of the search operation—simple (shallow) or complex (deep)—is determined upstream by the Intent Agent.

### **3.1 The Intent Agent's Role in RAG Depth Selection**

The IntentClassifierAgent is mandated to execute early in the graph flow and serve as the upstream decision-maker for RAG complexity. A node within this agent, decide\_search\_depth, analyzes the initial user message. Its LLM prompt must be specifically designed to output a structured Pydantic object that includes the required search complexity, setting the rag\_depth\_config field in the GraphState to either 'SHALLOW' or 'DEEP'. For instance, if the query relates to recently indexed, internal data, a SHALLOW configuration is selected. Conversely, if the query demands synthesis across multiple dynamic sources or necessitates web crawling (such as "Compare Alpha v1 features to legacy LangChain structure"), a DEEP configuration is required.

### **3.2 Defining RAG Complexity Levels**

Implementing two discrete RAG execution nodes simplifies maintenance, ensures strict adherence to resource control, and facilitates clear performance profiling.

1. **Level 1: Shallow RAG:** This path, executed by the execute\_shallow\_search node, involves a direct, single-pass retrieval using only the internal vector store retriever. This operation is designed to be fast and low-cost, immediately returning the top K results and a swift synthesis, updating the search\_results field in the state.  
2. **Level 2: Deep RAG:** This path, executed by the execute\_deep\_crawl\_and\_synthesize node, triggers a more resource-intensive, multi-step sub-graph. This typically includes an initial web search using external APIs (e.g., Tavily API) 7, followed by crawling, indexing of new data, and sophisticated synthesis across disparate, and potentially novel, sources. Although this results in higher latency and cost, it provides greater accuracy for complex, current-event, or highly comparative queries.

### **3.3 Graph Topology for Adaptive Search Routing**

The implementation of configurable RAG relies entirely on a conditional edge emanating from the Intent Agent's decision. After the IntentClassifierAgent completes, the flow routes to a designated Router\_RAG node. This router node reads the rag\_depth\_config field from the state.

If the state dictates 'SHALLOW', the conditional edge directs the flow to the execute\_shallow\_search node. If the state dictates 'DEEP', the edge directs execution to the complex execute\_deep\_crawl\_and\_synthesize node. This conditional routing is vital for resource optimization. By only triggering the expensive deep processing (crawling and complex synthesis) when explicitly required, the system conserves computational resources and drastically improves the average response time for simpler queries. Both RAG execution paths conclude by routing the workflow to a common merge point, select\_tools\_and\_final\_generation, allowing the subsequent tool orchestration phase to proceed with the appropriate search context.

## **IV. Requirement 3: Intent-Driven Dynamic Tool Discovery and Composability**

Requirement 3 demands a high degree of agent intelligence to select, modify, compose, and generate executable functions dynamically based on user intent. This is accomplished by strategically leveraging LangChain Expression Language (LCEL) alongside robust LLM reasoning.

### **4.1 Phase 1: Intent Discovery and Conditional Standard Tool Collection**

The IntentClassifierAgent serves as the initial tool orchestration manager. Following its analysis of the user message, it outputs a structured Pydantic IntentSchema detailing the user's functional needs, including the required domain (e.g., Financial, Legal), and specific functional requirements (e.g., "Need to perform calculation").

Based on the required domain, the agent initiates the **Conditional Standard Tool Collection**. Pre-defined, standard tools (such as a specific API wrapper or a specialized local function) are registered and conditionally included in the execution graph.8 For example, if the intent falls under the "Technical Support" domain, the pre-built "JIRA ticket creation tool" is collected and added to the

required\_tools list in the GraphState.

### **4.2 Phase 2: Dynamic Tool Assessment and Creation Logic**

If the intent analysis identifies a specific functional need that cannot be met by the conditionally collected standard tools, the Dynamic Tool Agent (DTA) begins an intelligent assessment and creation workflow.

The DTA first queries a **Tool Registry Vector Database (VDB)**, which stores detailed descriptions and input/output schemas of all existing dynamic tools. The DTA performs a semantic similarity check, comparing the user's defined functional requirement against the existing tool descriptions in the VDB. An LLM call is then used to judge the relevance of retrieved schemas.

This process culminates in a decisive workflow that governs tool assembly:

* **Use Existing:** If the similarity score between the required functionality and an existing tool description is high (e.g., above 0.9), the existing tool ID is retrieved and used directly.  
* **Modify or Compose:** If the similarity is moderate (e.g., between 0.6 and 0.9), the agent determines that the existing tool is close but requires modification, or that multiple existing tools must be chained together. This triggers the LCEL composition or modification workflow detailed below.  
* **Create New:** If the similarity score is low (e.g., below 0.6), indicating no sufficient tool exists, the agent initiates an LLM-driven generation process to define the code stub and schema for an entirely new tool.

Once resolved, the final list of standard and dynamic tools is compiled and stored in the required\_tools field in the Graph State.

### **4.3 Abstraction Mandate: Utilizing LCEL for Composability**

The mandate that tools must be "composable and abstract" is structurally achieved by ensuring all functional components are implemented as Runnables within the LangChain Expression Language (LCEL).

LCEL allows any two runnables—be they models, prompts, parsers, or wrapped functions—to be chained together using the pipe operator (|) to form a seamless RunnableSequence.9 This resulting sequence is itself treated as a single, complex runnable, enabling sophisticated

**Composability**. For instance, a high-level tool requiring "Deeply summarized financial trends" can be composed by chaining three atomic runnables: a FetchDataTool followed by a TrendAnalysisTool, which then pipes its output to a SummarizeAndFormatTool.

Crucially, this complex sequence achieves **Abstraction** by utilizing the .as\_tool() method.10 This method wraps the entire

RunnableSequence, assigning it a single high-level name, description, and input/output schema. The main agent LLM, which is responsible for tool calling, only perceives this abstraction and is unaware of the internal multi-step execution logic, simplifying its reasoning process. Tool modification is similarly handled by defining a new LCEL chain that incorporates changes to one or more internal components (e.g., replacing a default output parser with a highly specific Pydantic JsonOutputParser) before wrapping the modified chain again using .as\_tool().11

An essential architectural implication of this dynamic composition is the requirement for the Tool Registry VDB to store not just descriptions, but the underlying LCEL sequence or configuration necessary for dynamic instantiation. The composer must implement a robust serialization framework for these RunnableSequence artifacts, ensuring that dynamically assembled tools can be versioned, retrieved, and reused efficiently.

Table: Tool Abstraction and Composability Design

| Abstraction Principle | LangChain Mechanism | Implementation Detail | Benefit |
| :---- | :---- | :---- | :---- |
| Composability | LCEL Pipe Operator (|) and RunnableSequence.9 | Tools are defined as runnable objects that pass output of one to input of the next. | Allows rapid assembly of bespoke tools from existing atomic functions. |
| Abstraction | .as\_tool() method.10 | Attaches a name, description, and schema to a complex LCEL sequence. | Hides complexity from the reasoning LLM, simplifying agent decision-making. |
| Dynamic Creation | LLM Output Parsing \+ Code Generation. | Intent Agent output dictates schema; LLM generates function code/LCEL sequence. | Enables creation of genuinely new, purpose-built tools on demand. |

## **V. Implementation Roadmap and File Structure Guide**

### **5.1 Step-by-Step Guide for Refactoring the composer Project**

The refactoring process must proceed in logical phases, ensuring that foundational elements are complete before building complex logic.

#### **Phase 1: Foundation and State**

1. **Dependency Update:** Update project dependencies to target LangChain and LangGraph V1 alpha releases.2  
2. **Define Graph State:** Implement the authoritative GraphState Pydantic model (composer/agent\_runtime/state.py), ensuring correct LangGraph reducers are applied for cumulative fields like messages.  
3. **Basic Graph Initialization:** Establish the core LangGraph structure and configure durable execution by connecting the state persistence layer (e.g., database) to the runtime.

#### **Phase 2: Intent and Adaptive RAG**

4. **Implement Intent Agent Node:** Develop the IntentClassifierAgent (composer/agents/intent\_classifier.py). Configure its LLM to output the structured IntentSchema, including the critical rag\_depth\_config.  
5. **Define RAG Nodes:** Implement the functionally distinct execute\_shallow\_search and execute\_deep\_crawl\_and\_synthesize nodes (composer/rag/rag\_nodes.py).  
6. **Implement RAG Router:** Introduce the Router\_RAG conditional edge logic to direct the flow based on the rag\_depth\_config state field.

#### **Phase 3: Dynamic Tooling and Streaming**

7. **Setup Tool Registry:** Establish the external Tool Registry VDB service and the internal service logic for querying and storing tool metadata (composer/tools/dynamic\_registry/registry\_service.py).  
8. **Implement Dynamic Tool Agent (DTA):** Develop the DTA node logic (composer/agents/tool\_orchestrator.py) responsible for registry search, LCEL assembly, and tool creation.  
9. **Standard Tool Collection:** Define and register all standard tools within a centralized directory structure (composer/tools/standard/).  
10. **Implement Streaming Logic:** Configure the final node (Primary Chat Generator) to stream output using the messages mode. Configure all upstream nodes to use updates or custom streaming modes to provide granular workflow feedback.  
11. **API Integration:** Develop the composer streaming endpoint (composer/agent\_runtime/streaming\_api.py) to manage the LangGraph stream iterator and correctly serialize/route the distinct token and state update payloads to the UI client.

### **5.2 Mandatory Project File Structure**

A clean, structured file hierarchy is mandatory for managing the complexity of dynamic agent systems. The following structure ensures logical separation between core LangGraph primitives, agent logic, and tool services.

composer/  
├── requirements.txt  
├── main.py                     \# Primary startup/API configuration  
└── agent\_runtime/  
    ├── \_\_init\_\_.py  
    ├── state.py                \# Definition of GraphState (Pydantic models)  
    ├── graph\_builder.py        \# Logic to define nodes, edges, and compile the graph  
    ├── streaming\_api.py        \# Handles WebSocket/SSE connection and stream iterator  
    └── config.py               \# Runtime configuration (LLM models, API keys)  
      
    ├── agents/  
    │   ├── intent\_classifier.py \# Intent Agent logic, outputs IntentSchema  
    │   └── tool\_orchestrator.py \# Dynamic Tool Agent (DTA) logic  
      
    ├── rag/  
    │   ├── rag\_router.py       \# Conditional logic for shallow/deep routing  
    │   └── rag\_nodes.py        \# Shallow and Deep RAG execution nodes  
      
    └── tools/  
        ├── standard/           \# Collection of pre-defined, static tools  
        │   ├── jira\_tool.py  
        │   └── finance\_tool.py  
        └── dynamic\_registry/   \# Logic for interacting with the Tool Registry VDB  
            ├── registry\_service.py \# API interaction with VDB  
            └── serialization.py    \# LCEL serialization/deserialization helpers

### **5.3 Final Integrity Check: Compliance with LangGraph V1 Concepts**

The proposed architecture fully complies with and leverages the core capabilities of the LangGraph V1 framework. The integrity of the system is validated by the explicit use of critical V1 concepts:

1. **State Management:** The defined GraphState correctly employs reducer functions to manage complex state fields, such as concatenating messages 4, ensuring conversation context is consistently maintained across all nodes.  
2. **Execution Control:** Conditional execution, achieved by reading rag\_depth\_config and required\_tools from the state, is implemented through LangGraph's conditional edges, granting precise, programmatic control over the workflow. This ensures resource-intensive operations are only utilized when justified by the user's intent.  
3. **Streaming Paradigm:** The architecture correctly mandates the use of distinct streaming modes—messages for token output and updates/custom for state feedback 5—to deliver the selective streaming functionality required for optimal user experience and operational transparency. The use of V1's durable execution further supports the robustness of the remote execution model hosted in the  
   composer service.

#### **Works cited**

1. LangChain and LangGraph Enter v1.0 Alpha: A New Era for Agentic AI Development, accessed September 27, 2025, [https://joshuaberkowitz.us/blog/news-1/langchain-and-langgraph-enter-v1-0-alpha-a-new-era-for-agentic-ai-development-940](https://joshuaberkowitz.us/blog/news-1/langchain-and-langgraph-enter-v1-0-alpha-a-new-era-for-agentic-ai-development-940)  
2. LangChain & LangGraph 1.0 alpha releases, accessed September 27, 2025, [https://blog.langchain.com/langchain-langchain-1-0-alpha-releases/](https://blog.langchain.com/langchain-langchain-1-0-alpha-releases/)  
3. LangGraph \- LangChain, accessed September 27, 2025, [https://www.langchain.com/langgraph](https://www.langchain.com/langgraph)  
4. How to stream state updates of your graph, accessed September 27, 2025, [https://langchain-ai.github.io/langgraphjs/how-tos/stream-updates/](https://langchain-ai.github.io/langgraphjs/how-tos/stream-updates/)  
5. What's possible with LangGraph streaming \- Overview, accessed September 27, 2025, [https://langchain-ai.github.io/langgraph/concepts/streaming/](https://langchain-ai.github.io/langgraph/concepts/streaming/)  
6. Streaming \- ️ LangChain, accessed September 27, 2025, [https://python.langchain.com/docs/concepts/streaming/](https://python.langchain.com/docs/concepts/streaming/)  
7. Adaptive RAG with local LLMs, accessed September 27, 2025, [https://langchain-ai.github.io/langgraphjs/tutorials/rag/langgraph\_adaptive\_rag\_local/](https://langchain-ai.github.io/langgraphjs/tutorials/rag/langgraph_adaptive_rag_local/)  
8. Dynamic tool calling in LangGraph agents \- LangChain \- Changelog, accessed September 27, 2025, [https://changelog.langchain.com/announcements/dynamic-tool-calling-in-langgraph-agents](https://changelog.langchain.com/announcements/dynamic-tool-calling-in-langgraph-agents)  
9. How to chain runnables | 🦜️ LangChain, accessed September 27, 2025, [https://python.langchain.com/docs/how\_to/sequence/](https://python.langchain.com/docs/how_to/sequence/)  
10. How to create tools | 🦜️ LangChain, accessed September 27, 2025, [https://python.langchain.com/docs/how\_to/custom\_tools/](https://python.langchain.com/docs/how_to/custom_tools/)  
11. Tools | 🦜️ LangChain, accessed September 27, 2025, [https://python.langchain.com/docs/concepts/tools/](https://python.langchain.com/docs/concepts/tools/)