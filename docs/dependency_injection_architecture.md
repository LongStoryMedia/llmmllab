# Dependency Injection Architecture

## Overview

The LangGraph GraphBuilder now uses a comprehensive dependency injection pattern for all agents and storage services. This eliminates direct imports and creates a clean separation of concerns.

## Architecture

### GraphBuilder Structure
```python
class GraphBuilder:
    def __init__(self, storage):
        # Extract storage services for injection
        self._create_storage_services()
        # Create agents with injected dependencies  
        self._create_agents()
```

### Storage Service Injection
All storage services are extracted from the main storage instance and injected into agents:
- `user_config_storage` - User configuration management
- `conversation_storage` - Conversation persistence
- `message_storage` - Message storage and retrieval  
- `memory_storage` - Memory embeddings and search
- `summary_storage` - Summary creation and management
- `search_storage` - Search result persistence
- `dynamic_tool_storage` - Dynamic tool management

### Agent Dependency Injection

#### IntentClassifierAgent
```python
IntentClassifierAgent(user_config_storage=self.user_config_storage)
```

#### EngineeringAgent  
```python
EngineeringAgent(
    pipeline_factory=self.pipeline_factory,
    user_config_storage=self.user_config_storage
)
```

#### MemoryAgent
```python
MemoryAgent(memory_storage=self.memory_storage)
```

#### EmbeddingAgent
```python
EmbeddingAgent(
    pipeline_factory=self.pipeline_factory, 
    user_config_storage=self.user_config_storage
)
```

#### SummarizationAgent
```python
SummarizationAgent(
    pipeline_factory=self.pipeline_factory,
    summary_storage=self.summary_storage,
    search_storage=self.search_storage,
    user_config_storage=self.user_config_storage
)
```

## Backward Compatibility

All agents maintain backward compatibility with fallback imports:
```python
if self.storage_service:
    service = self.storage_service
else:
    from db import storage
    service = storage.get_service(storage.service_type)
```

## Node Integration

All LangGraph nodes already support dependency injection via constructor parameters:
- `MemorySearchNode(memory_agent=agent, storage=storage)`
- `TitleGenerationNode(pipeline_factory, summarization_agent=agent)`
- `IntentClassifierNode(intent_classifier_agent=agent)`

## Benefits

1. **Clean Separation**: Eliminates direct storage imports from agents
2. **Testability**: Easy to mock and test individual components
3. **Maintainability**: Clear dependency relationships
4. **Flexibility**: Storage services can be swapped or configured per instance
5. **Backward Compatibility**: Existing code continues to work with fallback patterns

## Testing

Dependency injection is validated through automated tests that verify:
- All agents are properly instantiated with injected dependencies
- Storage services are correctly extracted and passed to agents
- Agent attributes contain expected injected services
- No import errors or missing dependencies