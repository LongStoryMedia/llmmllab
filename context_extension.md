# Multi-Strategy Context Extension System

## Overview

This system addresses one of the fundamental limitations of Large Language Models (LLMs) - their finite context windows - by implementing a sophisticated three-pronged approach that combines external knowledge retrieval, semantic memory search, and hierarchical summarization. The system enables LLMs to maintain coherent conversations and access relevant information far beyond their native context limitations.

## Core Architecture

The system operates on three complementary strategies:
1. **External Search RAG** - Real-time web knowledge integration
2. **Memory Search RAG** - Semantic retrieval from conversation history
3. **In-Context Summarization** - Hierarchical compression of conversation context

## Data Model Definitions

### Message Structure
Messages form the atomic unit of conversation, containing:
- **Role**: Participant type (user, assistant, system, tool, agent, observer)
- **Content**: Array of typed content blocks (text, image, tool calls, etc.)
- **Metadata**: Unique ID, creation timestamp, conversation association
- **Optional**: Tool calls and thinking processes

### Summary Structure
Summaries represent consolidated information with hierarchical levels:
- **Content**: Compressed textual representation
- **Level**: Position in summarization hierarchy (1, 2, 3, master)
- **Source IDs**: Traceable references to original messages/summaries
- **Metadata**: Creation time, conversation association

### Memory Structure
Memories encapsulate retrievable conversation fragments:
- **Fragments**: Array of role-content pairs with IDs
- **Source**: Origin type (message or summary)
- **Similarity**: Vector search relevance score
- **Metadata**: Creation time, source references

## Strategy 1: External Search RAG

### Purpose
Augments LLM responses with real-time external knowledge, ensuring access to current information and specialized knowledge not present in training data.

### Implementation Flow

#### Keyword-Based Trigger
1. **Query Analysis**: User prompt undergoes keyword extraction
2. **Pattern Matching**: System identifies search-indicating terms:
   - Temporal indicators: "latest", "current", "recent", "today"
   - Information requests: "what is", "tell me about", "find"
   - Specific domains: company names, recent events, technical specs
3. **Search Execution**: If keywords detected, immediate web search initiated

#### LLM-Based Decision Making
1. **Fallback Mechanism**: If no explicit keywords found
2. **SLM Evaluation**: Small Language Model analyzes whether external search would enhance response quality
3. **Binary Decision**: SLM returns affirmative/negative for search necessity
4. **Conditional Search**: Search executed only on affirmative SLM response

#### Search Processing
1. **Multi-Engine Support**: Google, DuckDuckGo, or other configured engines
2. **Result Retrieval**: Configurable number of results (default: 3)
3. **Content Cleaning**: HTML markup removal, text extraction
4. **Context Integration**: Clean results appended to LLM context

### Configuration Parameters
- `search_engines`: List of available search providers
- `max_results`: Maximum search results to include (default: 3)
- `keyword_threshold`: Sensitivity for keyword detection
- `slm_model`: Small model for search decision making

## Strategy 2: Memory Search RAG

### Purpose
Provides semantic access to historical conversation data, enabling the LLM to reference relevant past interactions even when they exceed the current context window.

### Implementation Flow

#### Embedding Generation
1. **Message Processing**: Every user query and LLM response converted to embeddings
2. **Model Selection**: Dedicated embedding model (e.g., sentence-transformers)
3. **Vector Storage**: Embeddings stored in vector database with metadata:
   - Source type (message/summary)
   - Role information
   - Conversation ID
   - Creation timestamp

#### Semantic Search
1. **Query Embedding**: Current user query converted to vector representation
2. **Similarity Search**: Vector database queried for semantically similar content
3. **Threshold Filtering**: Results filtered by configurable similarity threshold (default: 0.7)
4. **Result Ranking**: Results ordered by similarity score

#### Context Pairing Logic
1. **User Message Retrieval**: If similar user message found, corresponding assistant response paired
2. **Assistant Message Retrieval**: If similar assistant response found, preceding user query paired
3. **Summary Retrieval**: Summaries returned independently without pairing
4. **Fragment Assembly**: Results packaged into Memory objects with fragments array

#### Memory Integration
1. **Configurable Retrieval**: Number of memories retrieved is configurable
2. **Context Injection**: Retrieved memories inserted into current context
3. **Relevance Scoring**: Memories ordered by similarity for optimal placement

### Configuration Parameters
- `embedding_model`: Model for vector generation
- `similarity_threshold`: Minimum similarity for retrieval (default: 0.7)
- `max_memories`: Maximum number of memories to retrieve
- `vector_db_config`: Database connection and indexing parameters

## Strategy 3: In-Context Summarization

### Purpose
Maintains conversation coherence while managing context window limitations through hierarchical compression that preserves essential information across multiple abstraction levels.

### Hierarchical Summarization Process

#### Level 1 Summarization
1. **Trigger Condition**: When context chain reaches `n_sum` messages (default: 6)
2. **Window Selection**: Last `sum_window` messages selected for summarization (default: 3)
3. **Summary Generation**: SLM/LLM creates concise summary of selected messages
4. **Storage Operations**:
   - Summary stored in database with source message IDs
   - Summary converted to embedding for vector search
   - Summary added to vector database as memory
5. **Context Update**: Summary replaces `sum_window` messages in context chain

#### Level 2+ Summarization
1. **Trigger Condition**: When `n_sum_sum` summaries of same level accumulate (default: 3)
2. **Summary Aggregation**: Multiple summaries combined into higher-level summary
3. **Hierarchy Tracking**: Level incremented, source IDs maintained
4. **Storage Operations**: Same as Level 1 with appropriate level marking
5. **Context Replacement**: Higher-level summary replaces constituent summaries

#### Master Summary Management
1. **Creation Trigger**: When `sum_window` summaries reach `max_sum_lvl` (default: 3)
2. **Master Generation**: Highest-level summaries combined into master summary
3. **Context Integration**: Master summary replaces source summaries in context
4. **Update Mechanism**: Subsequent max-level summaries update existing master summary

### Summarization Algorithm Details

#### Content Preservation Priorities
1. **Key Information**: Critical facts, decisions, conclusions preserved
2. **Context Continuity**: Conversation flow and topic progression maintained
3. **Semantic Density**: Maximum information per token achieved
4. **Reference Integrity**: Important entity and concept references retained

#### Quality Assurance
1. **Model Selection**: Dedicated summarization model for consistency
2. **Prompt Engineering**: Optimized prompts for different summary levels
3. **Length Control**: Configurable summary length targets
4. **Coherence Validation**: Optional coherence checking between levels

### Configuration Parameters
- `n_sum`: Messages before summarization trigger (default: 6)
- `sum_window`: Messages included in each summary (default: 3)
- `n_sum_sum`: Summaries before next level trigger (default: 3)
- `max_sum_lvl`: Maximum summarization level (default: 3)
- `summary_model`: Model for summary generation
- `summary_length`: Target summary length

## System Integration

### Unified Context Assembly
1. **Priority Order**: External search results → Memory search results → Current context
2. **Token Management**: Dynamic context allocation based on available window
3. **Relevance Weighting**: More relevant content positioned closer to current query
4. **Overflow Handling**: Graceful degradation when context limits approached

### Performance Optimization
1. **Caching**: Frequently accessed embeddings and summaries cached
2. **Async Processing**: Non-blocking operations for search and embedding generation
3. **Batch Operations**: Multiple embeddings generated simultaneously
4. **Index Optimization**: Vector database indices optimized for similarity search

### Error Handling and Fallbacks
1. **Search Failures**: System continues with available information
2. **Embedding Errors**: Graceful degradation to keyword-based matching
3. **Database Unavailability**: Local caching provides limited functionality
4. **Model Failures**: Fallback models for critical operations

## Benefits and Outcomes

### Extended Context Capabilities
- **Infinite Memory**: Access to entire conversation history through semantic search
- **Current Information**: Real-time external knowledge integration
- **Coherent Long Conversations**: Hierarchical summarization maintains context continuity

### Efficiency Gains
- **Reduced Redundancy**: Summaries eliminate repetitive information
- **Optimized Retrieval**: Semantic search finds relevant information quickly
- **Scalable Architecture**: System performance scales with conversation length

### Enhanced User Experience
- **Consistent Personality**: Memory search maintains assistant's communication style
- **Contextual Awareness**: System remembers and references past conversations
- **Accurate Information**: External search provides up-to-date facts

## Implementation Considerations

### Hardware Requirements
- **Vector Database**: Sufficient storage for embedding vectors
- **Compute Resources**: GPU acceleration recommended for embedding generation
- **Network Bandwidth**: Required for external search operations
- **Memory**: Adequate RAM for caching and processing

### Security and Privacy
- **Data Encryption**: All stored conversations and embeddings encrypted
- **Access Control**: User-specific memory isolation
- **Search Privacy**: Anonymized external search queries when possible
- **Data Retention**: Configurable conversation and memory retention policies

### Monitoring and Analytics
- **Performance Metrics**: Search latency, embedding generation time, summary quality
- **Usage Statistics**: Memory retrieval frequency, external search triggers
- **Quality Metrics**: Summary coherence scores, retrieval relevance ratings
- **System Health**: Database performance, model availability, error rates

---


# Multi-Strategy Context Extension System

## Overview

This system addresses one of the fundamental limitations of Large Language Models (LLMs) - their finite context windows - by implementing a sophisticated three-pronged approach that combines external knowledge retrieval, semantic memory search, and hierarchical summarization. The system enables LLMs to maintain coherent conversations and access relevant information far beyond their native context limitations.

The core architecture operates on three complementary strategies:
1.  **External Search RAG** - Real-time web knowledge integration
2.  **Memory Search RAG** - Semantic retrieval from conversation history
3.  **In-Context Summarization** - Hierarchical compression of conversation context

## Overall System Architecture

This diagram shows how the three strategies work together to extend LLM context capabilities. Each strategy operates independently but contributes to a unified context assembly that feeds into the LLM. A user query simultaneously initiates processes in the External Search, Memory Search, and In-Context Summarization modules. The outputs from these strategies are then fed into a central "Context Assembly" component, which integrates the information based on priority and manages token limits. This final, enhanced context is then passed to the Large Language Model to generate an informed response.

<svg viewBox="0 0 1200 800" style="width: 100%; height: auto;">
    <defs>
        <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
            <polygon points="0 0, 10 3.5, 0 7" fill="#2c3e50" />
        </marker>
        <pattern id="diagonalHatch" patternUnits="userSpaceOnUse" width="4" height="4">
            <path d="M-1,1 l2,-2 M0,4 l4,-4 M3,5 l2,-2" stroke="#3498db" stroke-width="1" opacity="0.3"/>
        </pattern>
    </defs>
    
    <rect x="50" y="50" width="120" height="60" class="component" rx="10" style="fill: #3498db; stroke: #2c3e50; stroke-width: 2;"/>
    <text x="110" y="85" class="text" style="font-family: 'Segoe UI', sans-serif; font-size: 12px; fill: #2c3e50; text-anchor: middle;">User Query</text>
    
    <rect x="250" y="20" width="200" height="120" class="external" rx="15" style="fill: #f39c12; stroke: #d68910; stroke-width: 2;"/>
    <text x="350" y="50" class="text" style="font-family: 'Segoe UI', sans-serif; font-size: 12px; fill: #2c3e50; text-anchor: middle; font-weight: bold;">External Search RAG</text>
    
    <rect x="270" y="70" width="80" height="30" class="component" rx="5" style="fill: #3498db; stroke: #2c3e50; stroke-width: 2;"/>
    <text x="310" y="90" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">Keyword Check</text>
    
    <rect x="360" y="70" width="80" height="30" class="component" rx="5" style="fill: #3498db; stroke: #2c3e50; stroke-width: 2;"/>
    <text x="400" y="90" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">SLM Decision</text>
    
    <rect x="270" y="105" width="160" height="25" class="process" rx="5" style="fill: #2ecc71; stroke: #27ae60; stroke-width: 2;"/>
    <text x="350" y="120" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">Web Search &amp; Content Clean</text>
    
    <rect x="250" y="180" width="200" height="120" class="highlight" rx="15" style="fill: #9b59b6; stroke: #8e44ad; stroke-width: 2;"/>
    <text x="350" y="210" class="text" style="font-family: 'Segoe UI', sans-serif; font-size: 12px; fill: #2c3e50; text-anchor: middle; font-weight: bold;">Memory Search RAG</text>
    
    <rect x="270" y="230" width="70" height="30" class="component" rx="5" style="fill: #3498db; stroke: #2c3e50; stroke-width: 2;"/>
    <text x="305" y="250" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">Query Embed</text>
    
    <rect x="350" y="230" width="90" height="30" class="component" rx="5" style="fill: #3498db; stroke: #2c3e50; stroke-width: 2;"/>
    <text x="395" y="250" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">Vector Search</text>
    
    <rect x="270" y="270" width="160" height="25" class="process" rx="5" style="fill: #2ecc71; stroke: #27ae60; stroke-width: 2;"/>
    <text x="350" y="285" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">Memory Fragment Assembly</text>
    
    <rect x="250" y="340" width="200" height="120" class="database" rx="15" style="fill: #e74c3c; stroke: #c0392b; stroke-width: 2;"/>
    <text x="350" y="370" class="text" style="font-family: 'Segoe UI', sans-serif; font-size: 12px; fill: #2c3e50; text-anchor: middle; font-weight: bold;">In-Context Summary</text>
    
    <rect x="270" y="390" width="70" height="30" class="component" rx="5" style="fill: #3498db; stroke: #2c3e50; stroke-width: 2;"/>
    <text x="305" y="410" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">Level Check</text>
    
    <rect x="350" y="390" width="90" height="30" class="component" rx="5" style="fill: #3498db; stroke: #2c3e50; stroke-width: 2;"/>
    <text x="395" y="410" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">Summarize</text>
    
    <rect x="270" y="430" width="160" height="25" class="process" rx="5" style="fill: #2ecc71; stroke: #27ae60; stroke-width: 2;"/>
    <text x="350" y="445" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">Hierarchical Compression</text>
    
    <rect x="550" y="200" width="150" height="80" class="process" rx="15" style="fill: #2ecc71; stroke: #27ae60; stroke-width: 2;"/>
    <text x="625" y="230" class="text" style="font-family: 'Segoe UI', sans-serif; font-size: 12px; fill: #2c3e50; text-anchor: middle; font-weight: bold;">Context Assembly</text>
    <text x="625" y="250" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">Priority Integration</text>
    <text x="625" y="265" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">Token Management</text>
    
    <rect x="800" y="180" width="150" height="120" class="highlight" rx="15" style="fill: #9b59b6; stroke: #8e44ad; stroke-width: 2;"/>
    <text x="875" y="210" class="text" style="font-family: 'Segoe UI', sans-serif; font-size: 12px; fill: #2c3e50; text-anchor: middle; font-weight: bold;">Large Language</text>
    <text x="875" y="230" class="text" style="font-family: 'Segoe UI', sans-serif; font-size: 12px; fill: #2c3e50; text-anchor: middle; font-weight: bold;">Model</text>
    <text x="875" y="260" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">Extended Context</text>
    <text x="875" y="280" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">Enhanced Response</text>
    
    <rect x="1030" y="210" width="120" height="60" class="component" rx="10" style="fill: #3498db; stroke: #2c3e50; stroke-width: 2;"/>
    <text x="1090" y="245" class="text" style="font-family: 'Segoe UI', sans-serif; font-size: 12px; fill: #2c3e50; text-anchor: middle;">Enhanced</text>
    <text x="1090" y="260" class="text" style="font-family: 'Segoe UI', sans-serif; font-size: 12px; fill: #2c3e50; text-anchor: middle;">Response</text>
    
    <rect x="50" y="500" width="100" height="60" class="database" rx="10" style="fill: #e74c3c; stroke: #c0392b; stroke-width: 2;"/>
    <text x="100" y="525" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">Message</text>
    <text x="100" y="540" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">Database</text>
    
    <rect x="180" y="500" width="100" height="60" class="database" rx="10" style="fill: #e74c3c; stroke: #c0392b; stroke-width: 2;"/>
    <text x="230" y="525" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">Vector</text>
    <text x="230" y="540" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">Database</text>
    
    <rect x="310" y="500" width="100" height="60" class="database" rx="10" style="fill: #e74c3c; stroke: #c0392b; stroke-width: 2;"/>
    <text x="360" y="525" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">Summary</text>
    <text x="360" y="540" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">Database</text>
    
    <line x1="170" y1="80" x2="240" y2="80" class="arrow" style="fill: none; stroke: #2c3e50; stroke-width: 2; marker-end: url(#arrowhead);"/>
    <line x1="170" y1="80" x2="240" y2="240" class="arrow" style="fill: none; stroke: #2c3e50; stroke-width: 2; marker-end: url(#arrowhead);"/>
    <line x1="170" y1="80" x2="240" y2="400" class="arrow" style="fill: none; stroke: #2c3e50; stroke-width: 2; marker-end: url(#arrowhead);"/>
    
    <line x1="450" y1="80" x2="540" y2="220" class="arrow" style="fill: none; stroke: #2c3e50; stroke-width: 2; marker-end: url(#arrowhead);"/>
    <line x1="450" y1="240" x2="540" y2="240" class="arrow" style="fill: none; stroke: #2c3e50; stroke-width: 2; marker-end: url(#arrowhead);"/>
    <line x1="450" y1="400" x2="540" y2="260" class="arrow" style="fill: none; stroke: #2c3e50; stroke-width: 2; marker-end: url(#arrowhead);"/>
    
    <line x1="700" y1="240" x2="790" y2="240" class="arrow" style="fill: none; stroke: #2c3e50; stroke-width: 2; marker-end: url(#arrowhead);"/>
    <line x1="950" y1="240" x2="1020" y2="240" class="arrow" style="fill: none; stroke: #2c3e50; stroke-width: 2; marker-end: url(#arrowhead);"/>
    
    <line x1="100" y1="500" x2="310" y2="300" class="arrow" stroke-dasharray="5,5" opacity="0.6" style="fill: none; stroke: #2c3e50; stroke-width: 2; marker-end: url(#arrowhead);"/>
    <line x1="230" y1="500" x2="350" y2="300" class="arrow" stroke-dasharray="5,5" opacity="0.6" style="fill: none; stroke: #2c3e50; stroke-width: 2; marker-end: url(#arrowhead);"/>
    <line x1="360" y1="500" x2="400" y2="300" class="arrow" stroke-dasharray="5,5" opacity="0.6" style="fill: none; stroke: #2c3e50; stroke-width: 2; marker-end: url(#arrowhead);"/>
    
    <text x="200" y="40" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">Query Analysis</text>
    <text x="475" y="180" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">Retrieved</text>
    <text x="475" y="195" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">Content</text>
    <text x="750" y="180" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">Unified</text>
    <text x="750" y="195" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">Context</text>
</svg>

## Data Model Definitions

### Message Structure
Messages form the atomic unit of conversation, containing:
- **Role**: Participant type (user, assistant, system, tool, agent, observer)
- **Content**: Array of typed content blocks (text, image, tool calls, etc.)
- **Metadata**: Unique ID, creation timestamp, conversation association
- **Optional**: Tool calls and thinking processes

### Summary Structure
Summaries represent consolidated information with hierarchical levels:
- **Content**: Compressed textual representation
- **Level**: Position in summarization hierarchy (1, 2, 3, master)
- **Source IDs**: Traceable references to original messages/summaries
- **Metadata**: Creation time, conversation association

### Memory Structure
Memories encapsulate retrievable conversation fragments:
- **Fragments**: Array of role-content pairs with IDs
- **Source**: Origin type (message or summary)
- **Similarity**: Vector search relevance score
- **Metadata**: Creation time, source references

---

## Strategy 1: External Search RAG 🔍

### Purpose
This strategy augments LLM responses with real-time external knowledge, ensuring access to current information and specialized knowledge not present in training data.

### Implementation Flow
This strategy identifies when external knowledge is needed and retrieves relevant web content to augment the LLM's response capabilities.

#### Keyword-Based Trigger
1.  **Query Analysis**: The user prompt undergoes keyword extraction.
2.  **Pattern Matching**: The system identifies search-indicating terms such as temporal indicators ("latest", "current"), information requests ("what is"), and specific domains (company names, recent events).
3.  **Search Execution**: If keywords are detected, an immediate web search is initiated.

#### LLM-Based Decision Making
1.  **Fallback Mechanism**: This is used if no explicit keywords are found.
2.  **SLM Evaluation**: A Small Language Model analyzes whether external search would enhance response quality.
3.  **Binary Decision**: The SLM returns an affirmative/negative response for search necessity.
4.  **Conditional Search**: A search is executed only on an affirmative SLM response.

#### Search Processing
1.  **Multi-Engine Support**: Supports Google, DuckDuckGo, or other configured engines.
2.  **Result Retrieval**: Retrieves a configurable number of results (default: 3).
3.  **Content Cleaning**: Involves HTML markup removal and text extraction.
4.  **Context Integration**: Clean results are appended to the LLM context.

<svg viewBox="0 0 1000 600" style="width: 100%; height: auto;">
    <defs><marker id="arrowhead-2" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="#2c3e50"></polygon></marker></defs>
    <ellipse cx="100" cy="100" rx="80" ry="40" class="component" style="fill: #3498db; stroke: #2c3e50; stroke-width: 2;"></ellipse>
    <text x="100" y="105" class="text" style="font-family: 'Segoe UI', sans-serif; font-size: 12px; fill: #2c3e50; text-anchor: middle;">User Query</text>
    
    <rect x="250" y="50" width="120" height="80" class="process" rx="10" style="fill: #2ecc71; stroke: #27ae60; stroke-width: 2;"></rect>
    <text x="310" y="80" class="text" style="font-family: 'Segoe UI', sans-serif; font-size: 12px; fill: #2c3e50; text-anchor: middle;">Keyword</text>
    <text x="310" y="95" class="text" style="font-family: 'Segoe UI', sans-serif; font-size: 12px; fill: #2c3e50; text-anchor: middle;">Analysis</text>
    <text x="310" y="115" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">• Temporal terms</text>
    <text x="310" y="125" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">• Info requests</text>
    
    <polygon points="450,50 500,90 450,130 400,90" class="highlight" style="fill: #9b59b6; stroke: #8e44ad; stroke-width: 2;"></polygon>
    <text x="450" y="95" class="text" style="font-family: 'Segoe UI', sans-serif; font-size: 12px; fill: #2c3e50; text-anchor: middle;">Keywords?</text>
    
    <rect x="550" y="150" width="120" height="80" class="process" rx="10" style="fill: #2ecc71; stroke: #27ae60; stroke-width: 2;"></rect>
    <text x="610" y="180" class="text" style="font-family: 'Segoe UI', sans-serif; font-size: 12px; fill: #2c3e50; text-anchor: middle;">SLM</text>
    <text x="610" y="195" class="text" style="font-family: 'Segoe UI', sans-serif; font-size: 12px; fill: #2c3e50; text-anchor: middle;">Decision</text>
    <text x="610" y="215" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">Search needed?</text>
    
    <rect x="400" y="280" width="120" height="80" class="external" rx="10" style="fill: #f39c12; stroke: #d68910; stroke-width: 2;"></rect>
    <text x="460" y="310" class="text" style="font-family: 'Segoe UI', sans-serif; font-size: 12px; fill: #2c3e50; text-anchor: middle;">Web Search</text>
    <text x="460" y="330" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">• Google</text>
    <text x="460" y="340" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">• DuckDuckGo</text>
    
    <rect x="600" y="280" width="120" height="80" class="process" rx="10" style="fill: #2ecc71; stroke: #27ae60; stroke-width: 2;"></rect>
    <text x="660" y="305" class="text" style="font-family: 'Segoe UI', sans-serif; font-size: 12px; fill: #2c3e50; text-anchor: middle;">Content</text>
    <text x="660" y="320" class="text" style="font-family: 'Segoe UI', sans-serif; font-size: 12px; fill: #2c3e50; text-anchor: middle;">Processing</text>
    <text x="660" y="340" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">• Clean HTML</text>
    <text x="660" y="350" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">• Extract text</text>
    
    <rect x="780" y="200" width="120" height="80" class="highlight" rx="10" style="fill: #9b59b6; stroke: #8e44ad; stroke-width: 2;"></rect>
    <text x="840" y="230" class="text" style="font-family: 'Segoe UI', sans-serif; font-size: 12px; fill: #2c3e50; text-anchor: middle;">Context</text>
    <text x="840" y="245" class="text" style="font-family: 'Segoe UI', sans-serif; font-size: 12px; fill: #2c3e50; text-anchor: middle;">Integration</text>
    <text x="840" y="265" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">Max 3 results</text>
    
    <rect x="550" y="50" width="120" height="50" class="component" rx="10" style="fill: #3498db; stroke: #2c3e50; stroke-width: 2;"></rect>
    <text x="610" y="80" class="text" style="font-family: 'Segoe UI', sans-serif; font-size: 12px; fill: #2c3e50; text-anchor: middle;">No External</text>
    <text x="610" y="90" class="text" style="font-family: 'Segoe UI', sans-serif; font-size: 12px; fill: #2c3e50; text-anchor: middle;">Search</text>
    
    <line x1="180" y1="100" x2="240" y2="100" class="arrow" style="fill: none; stroke: #2c3e50; stroke-width: 2; marker-end: url(#arrowhead-2);"></line>
    <line x1="370" y1="90" x2="390" y2="90" class="arrow" style="fill: none; stroke: #2c3e50; stroke-width: 2; marker-end: url(#arrowhead-2);"></line>
    <line x1="450" y1="130" x2="550" y2="180" class="arrow" style="fill: none; stroke: #2c3e50; stroke-width: 2; marker-end: url(#arrowhead-2);"></line>
    <line x1="500" y1="90" x2="540" y2="80" class="arrow" style="fill: none; stroke: #2c3e50; stroke-width: 2; marker-end: url(#arrowhead-2);"></line>
    <line x1="610" y1="230" x2="460" y2="270" class="arrow" style="fill: none; stroke: #2c3e50; stroke-width: 2; marker-end: url(#arrowhead-2);"></line>
    <line x1="520" y1="320" x2="590" y2="320" class="arrow" style="fill: none; stroke: #2c3e50; stroke-width: 2; marker-end: url(#arrowhead-2);"></line>
    <line x1="720" y1="320" x2="780" y2="250" class="arrow" style="fill: none; stroke: #2c3e50; stroke-width: 2; marker-end: url(#arrowhead-2);"></line>
    <line x1="670" y1="80" x2="780" y2="230" class="arrow" style="fill: none; stroke: #2c3e50; stroke-width: 2; marker-end: url(#arrowhead-2);"></line>
    
    <text x="200" y="85" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">Input</text>
    <text x="380" y="75" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">Parse</text>
    <text x="470" y="45" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">Yes</text>
    <text x="470" y="145" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">No</text>
    <text x="520" y="40" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">Skip</text>
    <text x="575" y="250" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">Execute</text>
    <text x="550" y="340" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">Results</text>
    <text x="750" y="300" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">Clean &amp; Add</text>
</svg>

### Configuration Parameters
- `search_engines`: List of available search providers
- `max_results`: Maximum search results to include (default: 3)
- `keyword_threshold`: Sensitivity for keyword detection
- `slm_model`: Small model for search decision making

---

## Strategy 2: Memory Search RAG 🧠

### Purpose
This strategy provides semantic access to historical conversation data, enabling the LLM to reference relevant past interactions even when they exceed the current context window.

### Implementation Flow
This strategy provides semantic access to conversation history through vector embeddings, enabling retrieval of relevant past interactions.

#### Embedding Generation
1.  **Message Processing**: Every user query and LLM response is converted to embeddings.
2.  **Model Selection**: A dedicated embedding model (e.g., sentence-transformers) is used.
3.  **Vector Storage**: Embeddings are stored in a vector database with metadata such as source type, role information, conversation ID, and a creation timestamp.

#### Semantic Search
1.  **Query Embedding**: The current user query is converted to a vector representation.
2.  **Similarity Search**: The vector database is queried for semantically similar content.
3.  **Threshold Filtering**: Results are filtered by a configurable similarity threshold (default: 0.7).
4.  **Result Ranking**: Results are ordered by similarity score.

#### Context Pairing Logic
1.  **User Message Retrieval**: If a similar user message is found, the corresponding assistant response is paired with it.
2.  **Assistant Message Retrieval**: If a similar assistant response is found, the preceding user query is paired.
3.  **Summary Retrieval**: Summaries are returned independently without pairing.
4.  **Fragment Assembly**: Results are packaged into Memory objects with a fragments array.

#### Memory Integration
1.  **Configurable Retrieval**: The number of memories retrieved is configurable.
2.  **Context Injection**: Retrieved memories are inserted into the current context.
3.  **Relevance Scoring**: Memories are ordered by similarity for optimal placement.

<svg viewBox="0 0 1000 700" style="width: 100%; height: auto;">
    <defs><marker id="arrowhead-3" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="#2c3e50"></polygon></marker></defs>
    <ellipse cx="100" cy="100" rx="80" ry="40" class="component" style="fill: #3498db; stroke: #2c3e50; stroke-width: 2;"></ellipse>
    <text x="100" y="105" class="text" style="font-family: 'Segoe UI', sans-serif; font-size: 12px; fill: #2c3e50; text-anchor: middle;">User Query</text>
    
    <rect x="250" y="70" width="120" height="60" class="process" rx="10" style="fill: #2ecc71; stroke: #27ae60; stroke-width: 2;"></rect>
    <text x="310" y="95" class="text" style="font-family: 'Segoe UI', sans-serif; font-size: 12px; fill: #2c3e50; text-anchor: middle;">Generate</text>
    <text x="310" y="110" class="text" style="font-family: 'Segoe UI', sans-serif; font-size: 12px; fill: #2c3e50; text-anchor: middle;">Embedding</text>
    
    <rect x="50" y="250" width="150" height="120" class="database" rx="15" style="fill: #e74c3c; stroke: #c0392b; stroke-width: 2;"></rect>
    <text x="125" y="280" class="text" style="font-family: 'Segoe UI', sans-serif; font-size: 12px; fill: #2c3e50; text-anchor: middle; font-weight: bold;">Vector Database</text>
    <text x="125" y="300" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">• Message embeddings</text>
    <text x="125" y="315" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">• Summary embeddings</text>
    <text x="125" y="330" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">• Metadata</text>
    <text x="125" y="345" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">• Similarity indices</text>
    
    <rect x="250" y="200" width="120" height="80" class="highlight" rx="10" style="fill: #9b59b6; stroke: #8e44ad; stroke-width: 2;"></rect>
    <text x="310" y="225" class="text" style="font-family: 'Segoe UI', sans-serif; font-size: 12px; fill: #2c3e50; text-anchor: middle;">Similarity</text>
    <text x="310" y="240" class="text" style="font-family: 'Segoe UI', sans-serif; font-size: 12px; fill: #2c3e50; text-anchor: middle;">Search</text>
    <text x="310" y="260" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">Threshold: 0.7</text>
    
    <polygon points="450,200 500,240 450,280 400,240" class="process" style="fill: #2ecc71; stroke: #27ae60; stroke-width: 2;"></polygon>
    <text x="450" y="235" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">Above</text>
    <text x="450" y="250" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">Threshold?</text>
    
    <rect x="550" y="150" width="140" height="100" class="component" rx="10" style="fill: #3498db; stroke: #2c3e50; stroke-width: 2;"></rect>
    <text x="620" y="175" class="text" style="font-family: 'Segoe UI', sans-serif; font-size: 12px; fill: #2c3e50; text-anchor: middle;">Context Pairing</text>
    <text x="620" y="195" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">• User → Assistant</text>
    <text x="620" y="210" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">• Assistant → User</text>
    <text x="620" y="225" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">• Summary → Independent</text>
    
    <rect x="550" y="300" width="140" height="80" class="process" rx="10" style="fill: #2ecc71; stroke: #27ae60; stroke-width: 2;"></rect>
    <text x="620" y="325" class="text" style="font-family: 'Segoe UI', sans-serif; font-size: 12px; fill: #2c3e50; text-anchor: middle;">Memory</text>
    <text x="620" y="340" class="text" style="font-family: 'Segoe UI', sans-serif; font-size: 12px; fill: #2c3e50; text-anchor: middle;">Assembly</text>
    <text x="620" y="360" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">Fragments array</text>
    
    <rect x="750" y="200" width="120" height="80" class="highlight" rx="10" style="fill: #9b59b6; stroke: #8e44ad; stroke-width: 2;"></rect>
    <text x="810" y="225" class="text" style="font-family: 'Segoe UI', sans-serif; font-size: 12px; fill: #2c3e50; text-anchor: middle;">Context</text>
    <text x="810" y="240" class="text" style="font-family: 'Segoe UI', sans-serif; font-size: 12px; fill: #2c3e50; text-anchor: middle;">Integration</text>
    <text x="810" y="260" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">Relevance order</text>
    
    <rect x="50" y="450" width="120" height="60" class="external" rx="10" style="fill: #f39c12; stroke: #d68910; stroke-width: 2;"></rect>
    <text x="110" y="475" class="text" style="font-family: 'Segoe UI', sans-serif; font-size: 12px; fill: #2c3e50; text-anchor: middle;">Message</text>
    <text x="110" y="490" class="text" style="font-family: 'Segoe UI', sans-serif; font-size: 12px; fill: #2c3e50; text-anchor: middle;">Storage</text>
    
    <rect x="200" y="450" width="120" height="60" class="process" rx="10" style="fill: #2ecc71; stroke: #27ae60; stroke-width: 2;"></rect>
    <text x="260" y="475" class="text" style="font-family: 'Segoe UI', sans-serif; font-size: 12px; fill: #2c3e50; text-anchor: middle;">Embedding</text>
    <text x="260" y="490" class="text" style="font-family: 'Segoe UI', sans-serif; font-size: 12px; fill: #2c3e50; text-anchor: middle;">Generation</text>
    
    <rect x="350" y="450" width="120" height="60" class="database" rx="10" style="fill: #e74c3c; stroke: #c0392b; stroke-width: 2;"></rect>
    <text x="410" y="475" class="text" style="font-family: 'Segoe UI', sans-serif; font-size: 12px; fill: #2c3e50; text-anchor: middle;">Vector</text>
    <text x="410" y="490" class="text" style="font-family: 'Segoe UI', sans-serif; font-size: 12px; fill: #2c3e50; text-anchor: middle;">Storage</text>
    
    <line x1="180" y1="100" x2="240" y2="100" class="arrow" style="fill: none; stroke: #2c3e50; stroke-width: 2; marker-end: url(#arrowhead-3);"></line>
    <line x1="310" y1="130" x2="310" y2="190" class="arrow" style="fill: none; stroke: #2c3e50; stroke-width: 2; marker-end: url(#arrowhead-3);"></line>
    <line x1="200" y1="310" x2="240" y2="240" class="arrow" style="fill: none; stroke: #2c3e50; stroke-width: 2; marker-end: url(#arrowhead-3);"></line>
    <line x1="370" y1="240" x2="390" y2="240" class="arrow" style="fill: none; stroke: #2c3e50; stroke-width: 2; marker-end: url(#arrowhead-3);"></line>
    <line x1="500" y1="200" x2="540" y2="180" class="arrow" style="fill: none; stroke: #2c3e50; stroke-width: 2; marker-end: url(#arrowhead-3);"></line>
    <line x1="500" y1="280" x2="540" y2="320" class="arrow" style="fill: none; stroke: #2c3e50; stroke-width: 2; marker-end: url(#arrowhead-3);"></line>
    <line x1="620" y1="250" x2="620" y2="290" class="arrow" style="fill: none; stroke: #2c3e50; stroke-width: 2; marker-end: url(#arrowhead-3);"></line>
    <line x1="690" y1="200" x2="740" y2="220" class="arrow" style="fill: none; stroke: #2c3e50; stroke-width: 2; marker-end: url(#arrowhead-3);"></line>
    <line x1="690" y1="340" x2="740" y2="260" class="arrow" style="fill: none; stroke: #2c3e50; stroke-width: 2; marker-end: url(#arrowhead-3);"></line>
    
    <line x1="170" y1="480" x2="190" y2="480" class="arrow" style="fill: none; stroke: #2c3e50; stroke-width: 2; marker-end: url(#arrowhead-3);"></line>
    <line x1="320" y1="480" x2="340" y2="480" class="arrow" style="fill: none; stroke: #2c3e50; stroke-width: 2; marker-end: url(#arrowhead-3);"></line>
    <line x1="410" y1="450" x2="150" y2="370" class="arrow" stroke-dasharray="5,5" style="fill: none; stroke: #2c3e50; stroke-width: 2; marker-end: url(#arrowhead-3);"></line>
    
    <text x="200" y="85" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">Embed</text>
    <text x="280" y="170" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">Query</text>
    <text x="380" y="220" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">Filter</text>
    <text x="520" y="165" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">Pass</text>
    <text x="520" y="295" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">Fail</text>
    <text x="715" y="185" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">Paired</text>
    <text x="715" y="285" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">Assembled</text>
    <text x="180" y="535" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">Historical Process</text>
</svg>

### Configuration Parameters
- `embedding_model`: Model for vector generation
- `similarity_threshold`: Minimum similarity for retrieval (default: 0.7)
- `max_memories`: Maximum number of memories to retrieve
- `vector_db_config`: Database connection and indexing parameters

---

## Strategy 3: In-Context Summarization 📚

### Purpose
This strategy maintains conversation coherence while managing context window limitations through hierarchical compression that preserves essential information across multiple abstraction levels.

### Hierarchical Summarization Process

#### Level 1 Summarization
1.  **Trigger Condition**: Occurs when the context chain reaches `n_sum` messages (default: 6).
2.  **Window Selection**: The last `sum_window` messages are selected for summarization (default: 3).
3.  **Summary Generation**: An SLM/LLM creates a concise summary of the selected messages.
4.  **Storage Operations**: The summary is stored in the database with source message IDs, converted to an embedding for vector search, and added to the vector database as a memory.
5.  **Context Update**: The summary replaces the `sum_window` messages in the context chain.

#### Level 2+ Summarization
1.  **Trigger Condition**: Activates when `n_sum_sum` summaries of the same level accumulate (default: 3).
2.  **Summary Aggregation**: Multiple summaries are combined into a higher-level summary.
3.  **Hierarchy Tracking**: The level is incremented, and source IDs are maintained.
4.  **Storage Operations**: Same as Level 1, with appropriate level marking.
5.  **Context Replacement**: The higher-level summary replaces its constituent summaries.

#### Master Summary Management
1.  **Creation Trigger**: Occurs when `sum_window` summaries reach `max_sum_lvl` (default: 3).
2.  **Master Generation**: The highest-level summaries are combined into a master summary.
3.  **Context Integration**: The master summary replaces the source summaries in the context.
4.  **Update Mechanism**: Subsequent max-level summaries update the existing master summary.

<svg viewBox="0 0 1200 900" style="width: 100%; height: auto;">
    <defs><marker id="arrowhead-4" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="#2c3e50"></polygon></marker><pattern id="diagonalHatch-2" patternUnits="userSpaceOnUse" width="4" height="4"><path d="M-1,1 l2,-2 M0,4 l4,-4 M3,5 l2,-2" stroke="#3498db" stroke-width="1" opacity="0.3"></path></pattern></defs>
    <rect x="50" y="50" width="60" height="30" class="component" rx="5" style="fill: #3498db; stroke: #2c3e50; stroke-width: 2;"></rect>
    <text x="80" y="70" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">M1</text>
    
    <rect x="130" y="50" width="60" height="30" class="component" rx="5" style="fill: #3498db; stroke: #2c3e50; stroke-width: 2;"></rect>
    <text x="160" y="70" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">M2</text>
    
    <rect x="210" y="50" width="60" height="30" class="component" rx="5" style="fill: #3498db; stroke: #2c3e50; stroke-width: 2;"></rect>
    <text x="240" y="70" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">M3</text>
    
    <rect x="290" y="50" width="60" height="30" class="component" rx="5" style="fill: #3498db; stroke: #2c3e50; stroke-width: 2;"></rect>
    <text x="320" y="70" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">M4</text>
    
    <rect x="370" y="50" width="60" height="30" class="component" rx="5" style="fill: #3498db; stroke: #2c3e50; stroke-width: 2;"></rect>
    <text x="400" y="70" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">M5</text>
    
    <rect x="450" y="50" width="60" height="30" class="component" rx="5" style="fill: #3498db; stroke: #2c3e50; stroke-width: 2;"></rect>
    <text x="480" y="70" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">M6</text>
    
    <text x="550" y="70" class="text" style="fill: #e74c3c; font-family: 'Segoe UI', sans-serif; font-size: 12px; text-anchor: middle;">n_sum = 6 reached!</text>
    
    <rect x="50" y="150" width="180" height="40" class="process" rx="10" style="fill: #2ecc71; stroke: #27ae60; stroke-width: 2;"></rect>
    <text x="140" y="175" class="text" style="font-family: 'Segoe UI', sans-serif; font-size: 12px; fill: #2c3e50; text-anchor: middle;">Level 1 Summary (M4+M5+M6)</text>
    
    <rect x="50" y="220" width="60" height="30" class="component" rx="5" style="fill: #3498db; stroke: #2c3e50; stroke-width: 2;"></rect>
    <text x="80" y="240" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">M1</text>
    
    <rect x="130" y="220" width="60" height="30" class="component" rx="5" style="fill: #3498db; stroke: #2c3e50; stroke-width: 2;"></rect>
    <text x="160" y="240" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">M2</text>
    
    <rect x="210" y="220" width="60" height="30" class="component" rx="5" style="fill: #3498db; stroke: #2c3e50; stroke-width: 2;"></rect>
    <text x="240" y="240" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">M3</text>
    
    <rect x="290" y="220" width="80" height="30" class="database" rx="5" style="fill: #e74c3c; stroke: #c0392b; stroke-width: 2;"></rect>
    <text x="330" y="240" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">S1-L1</text>
    
    <rect x="390" y="220" width="60" height="30" class="component" rx="5" style="fill: #3498db; stroke: #2c3e50; stroke-width: 2;"></rect>
    <text x="420" y="240" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">M7</text>
    
    <rect x="470" y="220" width="60" height="30" class="component" rx="5" style="fill: #3498db; stroke: #2c3e50; stroke-width: 2;"></rect>
    <text x="500" y="240" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">M8</text>
    
    <rect x="550" y="220" width="60" height="30" class="component" rx="5" style="fill: #3498db; stroke: #2c3e50; stroke-width: 2;"></rect>
    <text x="580" y="240" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">M9</text>
    
    <rect x="650" y="220" width="80" height="30" class="database" rx="5" style="fill: #e74c3c; stroke: #c0392b; stroke-width: 2;"></rect>
    <text x="690" y="240" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">S2-L1</text>
    
    <rect x="750" y="220" width="80" height="30" class="database" rx="5" style="fill: #e74c3c; stroke: #c0392b; stroke-width: 2;"></rect>
    <text x="790" y="240" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">S3-L1</text>
    
    <text x="860" y="240" class="text" style="fill: #e74c3c; font-family: 'Segoe UI', sans-serif; font-size: 12px; text-anchor: middle;">n_sum_sum = 3 reached!</text>
    
    <rect x="650" y="320" width="180" height="40" class="highlight" rx="10" style="fill: #9b59b6; stroke: #8e44ad; stroke-width: 2;"></rect>
    <text x="740" y="345" class="text" style="font-family: 'Segoe UI', sans-serif; font-size: 12px; fill: #2c3e50; text-anchor: middle;">Level 2 Summary (S1+S2+S3)</text>
    
    <rect x="50" y="390" width="60" height="30" class="component" rx="5" style="fill: #3498db; stroke: #2c3e50; stroke-width: 2;"></rect>
    <text x="80" y="410" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">M1</text>
    
    <rect x="130" y="390" width="60" height="30" class="component" rx="5" style="fill: #3498db; stroke: #2c3e50; stroke-width: 2;"></rect>
    <text x="160" y="410" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">M2</text>
    
    <rect x="210" y="390" width="60" height="30" class="component" rx="5" style="fill: #3498db; stroke: #2c3e50; stroke-width: 2;"></rect>
    <text x="240" y="410" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">M3</text>
    
    <rect x="290" y="390" width="80" height="30" class="external" rx="5" style="fill: #f39c12; stroke: #d68910; stroke-width: 2;"></rect>
    <text x="330" y="410" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">S1-L2</text>
    
    <rect x="500" y="500" width="200" height="60" class="highlight" rx="15" style="fill: #9b59b6; stroke: #8e44ad; stroke-width: 2;"></rect>
    <text x="600" y="525" class="text" style="font-family: 'Segoe UI', sans-serif; font-size: 12px; fill: #2c3e50; text-anchor: middle; font-weight: bold;">Level 3 Summary</text>
    <text x="600" y="545" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">max_sum_lvl reached</text>
    
    <rect x="750" y="500" width="200" height="60" class="database" rx="15" style="fill: #e74c3c; stroke: #c0392b; stroke-width: 2;"></rect>
    <text x="850" y="525" class="text" style="font-family: 'Segoe UI', sans-serif; font-size: 12px; fill: #2c3e50; text-anchor: middle; font-weight: bold;">Master Summary</text>
    <text x="850" y="545" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">Ultimate compression</text>
    
    <rect x="50" y="650" width="400" height="200" class="component" rx="15" fill="url(#diagonalHatch-2)" style="fill: #3498db; stroke: #2c3e50; stroke-width: 2;"></rect>
    <text x="250" y="680" class="text" style="font-family: 'Segoe UI', sans-serif; font-size: 12px; fill: #2c3e50; text-anchor: middle; font-weight: bold;">Configuration Parameters</text>
    <text x="70" y="710" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: start;">n_sum = 6 (messages before summarization)</text>
    <text x="70" y="730" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: start;">sum_window = 3 (messages in each summary)</text>
    <text x="70" y="750" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: start;">n_sum_sum = 3 (summaries before next level)</text>
    <text x="70" y="770" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: start;">max_sum_lvl = 3 (maximum hierarchy level)</text>
    <text x="70" y="800" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: start;">summary_model = SLM/LLM for summarization</text>
    <text x="70" y="820" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: start;">summary_length = Target token count</text>
    
    <rect x="500" y="650" width="150" height="80" class="database" rx="10" style="fill: #e74c3c; stroke: #c0392b; stroke-width: 2;"></rect>
    <text x="575" y="675" class="text" style="font-family: 'Segoe UI', sans-serif; font-size: 12px; fill: #2c3e50; text-anchor: middle;">Summary Storage</text>
    <text x="575" y="695" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">• Database record</text>
    <text x="575" y="710" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">• Source IDs tracked</text>
    <text x="575" y="720" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">• Level metadata</text>
    
    <rect x="680" y="650" width="150" height="80" class="highlight" rx="10" style="fill: #9b59b6; stroke: #8e44ad; stroke-width: 2;"></rect>
    <text x="755" y="675" class="text" style="font-family: 'Segoe UI', sans-serif; font-size: 12px; fill: #2c3e50; text-anchor: middle;">Vector Storage</text>
    <text x="755" y="695" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">• Embedding created</text>
    <text x="755" y="710" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">• Added to memory</text>
    <text x="755" y="720" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">• Searchable</text>
    
    <line x1="140" y1="80" x2="140" y2="140" class="arrow" style="fill: none; stroke: #2c3e50; stroke-width: 2; marker-end: url(#arrowhead-4);"></line>
    <line x1="740" y1="250" x2="740" y2="310" class="arrow" style="fill: none; stroke: #2c3e50; stroke-width: 2; marker-end: url(#arrowhead-4);"></line>
    <line x1="600" y1="560" x2="750" y2="530" class="arrow" style="fill: none; stroke: #2c3e50; stroke-width: 2; marker-end: url(#arrowhead-4);"></line>
    <line x1="740" y1="360" x2="575" y2="480" class="arrow" style="fill: none; stroke: #2c3e50; stroke-width: 2; marker-end: url(#arrowhead-4);"></line>
    
    <line x1="600" y1="560" x2="575" y2="640" class="arrow" style="fill: none; stroke: #2c3e50; stroke-width: 2; marker-end: url(#arrowhead-4);"></line>
    <line x1="650" y1="690" x2="670" y2="690" class="arrow" style="fill: none; stroke: #2c3e50; stroke-width: 2; marker-end: url(#arrowhead-4);"></line>
    
    <text x="50" y="30" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: start;">Original Messages</text>
    <text x="50" y="200" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: start;">After L1 Summary</text>
    <text x="50" y="370" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: start;">After L2 Summary</text>
    <text x="580" y="630" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">Storage Process</text>
</svg>

### Configuration Parameters
- `n_sum`: Messages before summarization trigger (default: 6)
- `sum_window`: Messages included in each summary (default: 3)
- `n_sum_sum`: Summaries before next level trigger (default: 3)
- `max_sum_lvl`: Maximum summarization level (default: 3)
- `summary_model`: Model for summary generation
- `summary_length`: Target summary length

---

## System Integration and Data Flow

This diagram illustrates how all three strategies integrate their outputs into a unified context that maximizes the LLM's response quality while respecting token limits.

### Unified Context Assembly
1.  **Priority Order**: External search results → Memory search results → Current context
2.  **Token Management**: Dynamic context allocation based on available window
3.  **Relevance Weighting**: More relevant content positioned closer to current query
4.  **Overflow Handling**: Graceful degradation when context limits are approached

### Performance Optimization
1.  **Caching**: Frequently accessed embeddings and summaries are cached.
2.  **Async Processing**: Non-blocking operations are used for search and embedding generation.
3.  **Batch Operations**: Multiple embeddings are generated simultaneously.
4.  **Index Optimization**: Vector database indices are optimized for similarity search.

### Error Handling and Fallbacks
1.  **Search Failures**: The system continues with available information.
2.  **Embedding Errors**: Graceful degradation to keyword-based matching occurs.
3.  **Database Unavailability**: Local caching provides limited functionality.
4.  **Model Failures**: Fallback models are used for critical operations.

<svg viewBox="0 0 1000 800" style="width: 100%; height: auto;">
    <defs><marker id="arrowhead-5" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="#2c3e50"></polygon></marker></defs>
    <rect x="50" y="50" width="120" height="60" class="external" rx="10" style="fill: #f39c12; stroke: #d68910; stroke-width: 2;"></rect>
    <text x="110" y="75" class="text" style="font-family: 'Segoe UI', sans-serif; font-size: 12px; fill: #2c3e50; text-anchor: middle;">External</text>
    <text x="110" y="90" class="text" style="font-family: 'Segoe UI', sans-serif; font-size: 12px; fill: #2c3e50; text-anchor: middle;">Search Results</text>
    
    <rect x="50" y="150" width="120" height="60" class="highlight" rx="10" style="fill: #9b59b6; stroke: #8e44ad; stroke-width: 2;"></rect>
    <text x="110" y="175" class="text" style="font-family: 'Segoe UI', sans-serif; font-size: 12px; fill: #2c3e50; text-anchor: middle;">Memory</text>
    <text x="110" y="190" class="text" style="font-family: 'Segoe UI', sans-serif; font-size: 12px; fill: #2c3e50; text-anchor: middle;">Search Results</text>
    
    <rect x="50" y="250" width="120" height="60" class="database" rx="10" style="fill: #e74c3c; stroke: #c0392b; stroke-width: 2;"></rect>
    <text x="110" y="275" class="text" style="font-family: 'Segoe UI', sans-serif; font-size: 12px; fill: #2c3e50; text-anchor: middle;">Hierarchical</text>
    <text x="110" y="290" class="text" style="font-family: 'Segoe UI', sans-serif; font-size: 12px; fill: #2c3e50; text-anchor: middle;">Summaries</text>
    
    <rect x="50" y="350" width="120" height="60" class="component" rx="10" style="fill: #3498db; stroke: #2c3e50; stroke-width: 2;"></rect>
    <text x="110" y="375" class="text" style="font-family: 'Segoe UI', sans-serif; font-size: 12px; fill: #2c3e50; text-anchor: middle;">Current</text>
    <text x="110" y="390" class="text" style="font-family: 'Segoe UI', sans-serif; font-size: 12px; fill: #2c3e50; text-anchor: middle;">Conversation</text>
    
    <rect x="250" y="150" width="150" height="120" class="process" rx="15" style="fill: #2ecc71; stroke: #27ae60; stroke-width: 2;"></rect>
    <text x="325" y="180" class="text" style="font-family: 'Segoe UI', sans-serif; font-size: 12px; fill: #2c3e50; text-anchor: middle; font-weight: bold;">Priority Ordering</text>
    <text x="325" y="205" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">1. External results</text>
    <text x="325" y="220" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">2. Memory fragments</text>
    <text x="325" y="235" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">3. Recent summaries</text>
    <text x="325" y="250" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">4. Current context</text>
    
    <rect x="450" y="100" width="150" height="80" class="highlight" rx="10" style="fill: #9b59b6; stroke: #8e44ad; stroke-width: 2;"></rect>
    <text x="525" y="125" class="text" style="font-family: 'Segoe UI', sans-serif; font-size: 12px; fill: #2c3e50; text-anchor: middle;">Token</text>
    <text x="525" y="140" class="text" style="font-family: 'Segoe UI', sans-serif; font-size: 12px; fill: #2c3e50; text-anchor: middle;">Management</text>
    <text x="525" y="165" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">Dynamic allocation</text>
    
    <rect x="450" y="200" width="150" height="100" class="process" rx="10" style="fill: #2ecc71; stroke: #27ae60; stroke-width: 2;"></rect>
    <text x="525" y="225" class="text" style="font-family: 'Segoe UI', sans-serif; font-size: 12px; fill: #2c3e50; text-anchor: middle; font-weight: bold;">Context Assembly</text>
    <text x="525" y="245" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">• Relevance scoring</text>
    <text x="525" y="260" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">• Token optimization</text>
    <text x="525" y="275" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">• Overflow handling</text>
    <text x="525" y="290" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">• Format consistency</text>
    
    <rect x="650" y="150" width="120" height="80" class="component" rx="10" style="fill: #3498db; stroke: #2c3e50; stroke-width: 2;"></rect>
    <text x="710" y="175" class="text" style="font-family: 'Segoe UI', sans-serif; font-size: 12px; fill: #2c3e50; text-anchor: middle;">Quality</text>
    <text x="710" y="190" class="text" style="font-family: 'Segoe UI', sans-serif; font-size: 12px; fill: #2c3e50; text-anchor: middle;">Assurance</text>
    <text x="710" y="210" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">• Coherence check</text>
    <text x="710" y="220" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">• Relevance filter</text>
    
    <rect x="450" y="350" width="200" height="100" class="highlight" rx="15" style="fill: #9b59b6; stroke: #8e44ad; stroke-width: 2;"></rect>
    <text x="550" y="380" class="text" style="font-family: 'Segoe UI', sans-serif; font-size: 12px; fill: #2c3e50; text-anchor: middle; font-weight: bold;">Enhanced Context</text>
    <text x="550" y="405" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">• External knowledge</text>
    <text x="550" y="420" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">• Historical relevance</text>
    <text x="550" y="435" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">• Compressed summaries</text>
    
    <rect x="750" y="300" width="150" height="120" class="external" rx="15" style="fill: #f39c12; stroke: #d68910; stroke-width: 2;"></rect>
    <text x="825" y="335" class="text" style="font-family: 'Segoe UI', sans-serif; font-size: 12px; fill: #2c3e50; text-anchor: middle; font-weight: bold;">Large Language</text>
    <text x="825" y="355" class="text" style="font-family: 'Segoe UI', sans-serif; font-size: 12px; fill: #2c3e50; text-anchor: middle; font-weight: bold;">Model</text>
    <text x="825" y="380" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">Extended context</text>
    <text x="825" y="395" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">Enhanced reasoning</text>
    <text x="825" y="410" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">Informed responses</text>
    
    <rect x="250" y="500" width="200" height="120" class="component" rx="15" style="fill: #3498db; stroke: #2c3e50; stroke-width: 2;"></rect>
    <text x="350" y="530" class="text" style="font-family: 'Segoe UI', sans-serif; font-size: 12px; fill: #2c3e50; text-anchor: middle; font-weight: bold;">Performance Monitoring</text>
    <text x="350" y="555" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">• Retrieval latency</text>
    <text x="350" y="570" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">• Context utilization</text>
    <text x="350" y="585" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">• Summary quality scores</text>
    <text x="350" y="600" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">• User satisfaction metrics</text>
    
    <rect x="500" y="500" width="200" height="120" class="database" rx="15" style="fill: #e74c3c; stroke: #c0392b; stroke-width: 2;"></rect>
    <text x="600" y="530" class="text" style="font-family: 'Segoe UI', sans-serif; font-size: 12px; fill: #2c3e50; text-anchor: middle; font-weight: bold;">Continuous Learning</text>
    <text x="600" y="555" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">• Usage pattern analysis</text>
    <text x="600" y="570" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">• Model improvement</text>
    <text x="600" y="585" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">• Parameter tuning</text>
    <text x="600" y="600" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">• Error correction</text>
    
    <line x1="170" y1="80" x2="240" y2="160" class="arrow" style="fill: none; stroke: #2c3e50; stroke-width: 2; marker-end: url(#arrowhead-5);"></line>
    <line x1="170" y1="180" x2="240" y2="200" class="arrow" style="fill: none; stroke: #2c3e50; stroke-width: 2; marker-end: url(#arrowhead-5);"></line>
    <line x1="170" y1="280" x2="240" y2="240" class="arrow" style="fill: none; stroke: #2c3e50; stroke-width: 2; marker-end: url(#arrowhead-5);"></line>
    <line x1="170" y1="380" x2="240" y2="260" class="arrow" style="fill: none; stroke: #2c3e50; stroke-width: 2; marker-end: url(#arrowhead-5);"></line>
    
    <line x1="400" y1="210" x2="440" y2="140" class="arrow" style="fill: none; stroke: #2c3e50; stroke-width: 2; marker-end: url(#arrowhead-5);"></line>
    <line x1="400" y1="210" x2="440" y2="250" class="arrow" style="fill: none; stroke: #2c3e50; stroke-width: 2; marker-end: url(#arrowhead-5);"></line>
    
    <line x1="600" y1="250" x2="640" y2="190" class="arrow" style="fill: none; stroke: #2c3e50; stroke-width: 2; marker-end: url(#arrowhead-5);"></line>
    <line x1="600" y1="300" x2="550" y2="340" class="arrow" style="fill: none; stroke: #2c3e50; stroke-width: 2; marker-end: url(#arrowhead-5);"></line>
    
    <line x1="650" y1="400" x2="740" y2="360" class="arrow" style="fill: none; stroke: #2c3e50; stroke-width: 2; marker-end: url(#arrowhead-5);"></line>
    
    <line x1="550" y1="450" x2="350" y2="490" class="arrow" style="fill: none; stroke: #2c3e50; stroke-width: 2; marker-end: url(#arrowhead-5);"></line>
    <line x1="450" y1="560" x2="490" y2="560" class="arrow" style="fill: none; stroke: #2c3e50; stroke-width: 2; marker-end: url(#arrowhead-5);"></line>
    <line x1="600" y1="500" x2="525" y2="300" class="arrow" stroke-dasharray="5,5" style="fill: none; stroke: #2c3e50; stroke-width: 2; marker-end: url(#arrowhead-5);"></line>
    
    <text x="200" y="120" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">Input</text>
    <text x="200" y="140" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">Sources</text>
    <text x="420" y="185" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">Prioritize</text>
    <text x="610" y="175" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">Validate</text>
    <text x="680" y="330" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">Process</text>
    <text x="400" y="470" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">Monitor</text>
    <text x="550" y="470" class="text small-text" style="font-family: 'Segoe UI', sans-serif; font-size: 10px; fill: #7f8c8d; text-anchor: middle;">Feedback</text>
</svg>

## Benefits and Outcomes

### Extended Context Capabilities
- **Infinite Memory**: Access to entire conversation history through semantic search
- **Current Information**: Real-time external knowledge integration
- **Coherent Long Conversations**: Hierarchical summarization maintains context continuity

### Efficiency Gains
- **Reduced Redundancy**: Summaries eliminate repetitive information
- **Optimized Retrieval**: Semantic search finds relevant information quickly
- **Scalable Architecture**: System performance scales with conversation length

### Enhanced User Experience
- **Consistent Personality**: Memory search maintains the assistant's communication style.
- **Contextual Awareness**: The system remembers and references past conversations.
- **Accurate Information**: External search provides up-to-date facts.

## Implementation Considerations

### Hardware Requirements
- **Vector Database**: Sufficient storage for embedding vectors
- **Compute Resources**: GPU acceleration recommended for embedding generation
- **Network Bandwidth**: Required for external search operations
- **Memory**: Adequate RAM for caching and processing

### Security and Privacy
- **Data Encryption**: All stored conversations and embeddings are encrypted.
- **Access Control**: User-specific memory isolation
- **Search Privacy**: Anonymized external search queries when possible
- **Data Retention**: Configurable conversation and memory retention policies

### Monitoring and Analytics
- **Performance Metrics**: Search latency, embedding generation time, summary quality
- **Usage Statistics**: Memory retrieval frequency, external search triggers
- **Quality Metrics**: Summary coherence scores, retrieval relevance ratings
- **System Health**: Database performance, model availability, error rates
