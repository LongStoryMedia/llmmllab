# BaseAgent Architecture

## Overview

The `BaseAgent` class provides a unified foundation for all workflow agents in the LLM ML Lab system. It implements common functionality including node metadata injection, consistent logging, error handling patterns, and execution context management.

## Key Features

### 1. Node Metadata Injection

The BaseAgent provides a standardized way to inject workflow execution metadata into agents:

```python
from composer.agents import BaseAgent
from models import NodeMetadata

class MyAgent(BaseAgent):
    def __init__(self):
        super().__init__("MyAgent")
    
    async def my_operation(self, user_id: str):
        # Use base class logging and error handling
        self._log_operation_start("my_operation", user_id=user_id)
        # ... perform work ...
        self._log_operation_success("my_operation", result="success")

# Usage in workflow nodes
agent = MyAgent()
metadata = NodeMetadata(
    node_name="MyWorkflowNode",
    node_id="node-123",
    node_type="MyAgent",
    execution_time=datetime.now(timezone.utc),
    user_id="user-456",
    conversation_id=42
)
agent.inject_node_metadata(metadata)
```

### 2. Consistent Logging

All agents automatically get component-specific logging with metadata context:

```python
class MyAgent(BaseAgent):
    def __init__(self):
        super().__init__("MyCustomAgent")  # Component name for logging
        
    async def do_work(self):
        # Logger automatically includes component context
        self.logger.info("Starting work")
        
        # After metadata injection, logger includes node context too
        # Log entries will include: node_name, user_id, conversation_id
```

### 3. Error Handling Patterns

Standardized error handling with consistent logging and NodeExecutionError wrapping:

```python
async def my_operation(self, user_id: str):
    try:
        self._log_operation_start("my_operation", user_id=user_id)
        # ... do work that might fail ...
        self._log_operation_success("my_operation")
        return result
    except Exception as e:
        # Automatically logs error with context and raises NodeExecutionError
        self._handle_node_error("my_operation", e, user_id=user_id)
```

### 4. Execution Context Management

Track additional metadata throughout operation lifecycle:

```python
async def complex_operation(self):
    self.update_execution_context(
        step="initialization",
        start_time=time.time()
    )
    
    # ... do work ...
    
    self.update_execution_context(
        step="processing",
        items_processed=100
    )
    
    # Get current context for debugging
    context = self.get_execution_context()
    self.logger.info("Current context", **context)
```

## Agent Implementation Pattern

All agents should follow this pattern:

```python
from composer.agents.base_agent import BaseAgent
from composer.core.errors import NodeExecutionError

class MySpecializedAgent(BaseAgent):
    """Specialized agent for specific workflow tasks."""
    
    def __init__(self, dependency1, dependency2):
        """Initialize with component name and dependencies."""
        super().__init__("MySpecializedAgent")
        self.dependency1 = dependency1
        self.dependency2 = dependency2
    
    async def primary_operation(self, user_id: str, **params):
        """Primary agent operation with standardized patterns."""
        try:
            self._log_operation_start("primary_operation", user_id=user_id, **params)
            
            # Update execution context
            self.update_execution_context(operation="primary_operation")
            
            # Do the actual work
            result = await self._do_work(user_id, **params)
            
            self._log_operation_success("primary_operation", result_size=len(result))
            return result
            
        except Exception as e:
            self._handle_node_error("primary_operation", e, user_id=user_id, **params)
    
    async def _do_work(self, user_id: str, **params):
        """Internal work method."""
        # Implementation here
        pass
```

## Current Agent Implementations

All existing agents now inherit from BaseAgent:

- **EmbeddingAgent**: Text-to-vector conversion with metadata tracking
- **MemoryAgent**: Semantic memory operations with execution context
- **ClassifierAgent**: Intent analysis with logging and error handling
- **EngineeringAgent**: Technical response generation with metadata
- **SummarizationAgent**: Content summarization with consistent patterns

## Workflow Integration

Workflow nodes can inject metadata into agents for tracking:

```python
from composer.nodes.base_node import BaseNode
from composer.agents import EmbeddingAgent

class EmbedTextNode(BaseNode):
    def __init__(self):
        super().__init__()
        self.agent = EmbeddingAgent(pipeline_factory, profile)
    
    async def execute(self, state: WorkflowState) -> WorkflowState:
        # Create and inject node metadata
        metadata = self.create_node_metadata(
            user_id=state.user_id,
            conversation_id=state.conversation_id
        )
        self.agent.inject_node_metadata(metadata)
        
        # Agent operations now include workflow context
        embeddings = await self.agent.generate_embeddings(
            texts=state.texts,
            user_id=state.user_id
        )
        
        return state
```

## Benefits

1. **Consistency**: All agents follow the same patterns for logging, error handling, and metadata management
2. **Debugging**: Rich context information available in logs and error messages
3. **Tracking**: Node metadata enables workflow execution tracking and debugging
4. **Maintainability**: Common functionality centralized in base class
5. **Testing**: Standardized testing patterns for all agent functionality

## Testing

The BaseAgent includes comprehensive unit tests covering:

- Metadata injection and retrieval
- Logging setup and context binding
- Error handling with and without metadata
- Execution context management
- Integrated workflow testing

Run tests with:

```bash
kubectl exec -it -n ollama <pod-name> -- /app/v.sh python -m pytest test/test_base_agent.py -v
```
