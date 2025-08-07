# LangChain + vLLM Integration Plan

## Overview

This document outlines the plan to refactor the current pipeline-based architecture to use:
- **vLLM** for high-performance model serving
- **LangChain** for application framework and prompt management

## Benefits

1. **Performance**
   - vLLM's PagedAttention improves memory efficiency by 2-10x
   - Continuous batching increases throughput by up to 24x
   - Reduced latency through tensor parallelism

2. **Developer Experience**
   - LangChain provides structured abstractions
   - Simplified prompt management
   - Built-in conversation memory
   - Reusable components

3. **Flexibility**
   - Easy to add new models
   - Support for agents and tools
   - Enhanced retrieval and RAG capabilities

## Architecture

```
┌──────────────┐     ┌───────────────────┐     ┌───────────────┐
│              │     │                   │     │               │
│ HTTP/gRPC    │────►│  FastAPI Server   │────►│ LangChain     │
│ Client       │     │  (app.py)         │     │ Framework     │
│              │     │                   │     │               │
└──────────────┘     └───────────────────┘     └───────┬───────┘
                                                       │
                                                       ▼
                     ┌───────────────────┐     ┌───────────────┐
                     │                   │     │               │
                     │  Model Registry   │◄────┤ vLLM Server   │
                     │  (models.json)    │     │ (models)      │
                     │                   │     │               │
                     └───────────────────┘     └───────────────┘
```

## Implementation Steps

### 1. Install Dependencies
```bash
pip install vllm langchain langchain_community langchain_openai
```

### 2. Create vLLM Server
- Implement a standalone vLLM server for model inference
- Configure tensor parallelism based on GPU configuration
- Set up proper quantization for GGUF models
- Expose HTTP/gRPC endpoints

### 3. Create LangChain Integration
- Define model wrappers compatible with LangChain
- Implement chat templates as LangChain prompt templates
- Set up conversation chain for handling messages
- Create custom callbacks for streaming responses

### 4. Update API Layer
- Modify FastAPI endpoints to use LangChain chains
- Implement proper streaming responses
- Maintain backward compatibility with existing clients

### 5. Refactor Pipeline Architecture
- Replace existing pipeline classes with LangChain chains
- Update model factory to use vLLM configurations
- Implement proper memory management

### 6. Update Model Registry
- Add vLLM-specific configuration to models.json
- Include LangChain-specific metadata
- Define prompt templates for each model

### 7. Implement Testing
- Create test suite for LangChain chains
- Test vLLM performance with benchmarks
- Validate streaming functionality

## Migration Strategy

The migration will be done in phases:
1. Set up vLLM server alongside existing infrastructure
2. Introduce LangChain gradually, starting with text models
3. Migrate image generation models
4. Transition to the new architecture in production

## Timeline

- Week 1: Setup vLLM server and basic LangChain integration
- Week 2: Migrate text-to-text models (Mixtral, Qwen)
- Week 3: Migrate multimodal models (GLM4V)
- Week 4: Complete migration and testing
