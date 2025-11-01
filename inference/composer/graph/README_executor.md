# Generic Workflow Executor

The `WorkflowExecutor` module provides a generic, reusable streaming interface for executing any `CompiledStateGraph` with any state type.

## Overview

This module extracts the streaming logic from `ComposerService.execute_workflow` into a generic component that can be used across different graph types and state models, promoting code reuse and consistency.

## Key Features

- **Generic State Support**: Works with any state type (dict, Pydantic models, etc.)
- **Streaming & Batch Modes**: Supports both streaming and batch execution
- **Event Enrichment**: Automatically enriches events with metadata and tool information
- **Error Handling**: Robust error handling with graceful degradation
- **Thread Configuration**: Automatic checkpointing configuration management
- **Extensible**: Easy to extend with custom behavior

## Architecture

```
WorkflowExecutor
├── stream_workflow()      # Streaming execution with event enrichment
├── execute_workflow_batch() # Batch execution mode  
├── create_thread_config() # Checkpointing configuration
└── _enrich_event()       # Event metadata enrichment
```

## Usage Examples

### Basic Streaming with Dict State

```python
from composer.graph.executor import WorkflowExecutor

# Simple dictionary state
state = {
    "messages": ["Hello world"], 
    "user_id": "user_123",
    "step": 0
}

executor = WorkflowExecutor()

async for event in executor.stream_workflow(
    workflow=compiled_graph,
    initial_state=state,
    thread_id="session_123"
):
    print(f"Event: {event}")
```

### Using with Pydantic Models

```python
from pydantic import BaseModel
from composer.graph.executor import WorkflowExecutor

class MyWorkflowState(BaseModel):
    messages: List[str]
    user_id: str
    processing_step: str = "initial"
    
    def model_dump(self) -> Dict[str, Any]:
        return super().model_dump()

state = MyWorkflowState(
    messages=["User input"],
    user_id="user_456",
    processing_step="analysis"
)

executor = WorkflowExecutor()
async for event in executor.stream_workflow(workflow, state):
    handle_event(event)
```

### Convenience Functions

```python
from composer.graph.executor import stream_workflow, execute_workflow

# Streaming convenience function
async for event in stream_workflow(
    workflow=graph, 
    initial_state=state,
    thread_id="thread_789"
):
    process_event(event)

# Batch convenience function  
result = await execute_workflow(
    workflow=graph,
    initial_state=state,
    thread_id="thread_789"
)
```

### Custom Executor with Extended Behavior

```python
class CustomExecutor(WorkflowExecutor):
    def _enrich_event(self, event, context_name):
        # Call parent enrichment
        enriched = super()._enrich_event(event, context_name)
        
        # Add custom metadata
        enriched.setdefault("metadata", {}).update({
            "custom_processor": True,
            "version": "2.0"
        })
        
        return enriched

executor = CustomExecutor(default_context="custom_app")
```

## Integration with ComposerService

The `ComposerService` has been updated to use `WorkflowExecutor` internally:

```python
class ComposerService:
    def __init__(self):
        self.executor = WorkflowExecutor(
            logger=self.logger, 
            default_context="composer_service"
        )
    
    async def execute_workflow(self, workflow, initial_state, stream=True):
        thread_id = f"thread_{initial_state.user_id}_{initial_state.conversation_id}"
        
        if stream:
            async for event in self.executor.stream_workflow(
                workflow, initial_state, thread_id=thread_id
            ):
                yield event
        else:
            result = await self.executor.execute_workflow_batch(
                workflow, initial_state, thread_id=thread_id
            )
            yield {"event": "workflow_complete", "data": result}
```

## Event Enrichment

The executor automatically enriches events with:

- **Tool Information**: Injects `tool_calls` from state into events
- **Node Metadata**: Adds `node_metadata` when available  
- **Timing Information**: Timestamps for chain/tool start/end events
- **Context Information**: Adds workflow context for traceability

## Error Handling

- **Graceful Degradation**: Enrichment errors don't break the stream
- **Comprehensive Logging**: Detailed error logging with context
- **Error Events**: Execution errors generate `workflow_error` events

## State Requirements

State objects must be one of:
1. **Dictionary**: Plain `dict` object
2. **Pydantic Model**: With `model_dump()` method
3. **Legacy Pydantic**: With `dict()` method
4. **Custom Object**: With compatible serialization method

## Thread Configuration

The executor handles checkpointing configuration automatically:

```python
config = executor.create_thread_config(
    thread_id="unique_thread_id",
    additional_config={"custom": "value"}
)
```

## Benefits

1. **Code Reuse**: Single implementation for all workflow streaming
2. **Consistency**: Standardized event enrichment and error handling
3. **Flexibility**: Works with any graph type and state model
4. **Maintainability**: Centralized streaming logic
5. **Extensibility**: Easy to customize for specific needs
6. **Type Safety**: Generic typing with protocol support

## Migration Guide

### From ComposerService.execute_workflow

**Before:**
```python
async for event in composer_service.execute_workflow(workflow, state):
    handle_event(event)
```

**After:**
```python
from composer.graph.executor import stream_workflow

async for event in stream_workflow(workflow, state, thread_id="thread_123"):
    handle_event(event)
```

### Custom Streaming Logic

**Before:**
```python
async for event in workflow.astream_events(state.model_dump(), config=config):
    # Custom enrichment logic here
    yield enriched_event
```

**After:**
```python
executor = WorkflowExecutor()
async for event in executor.stream_workflow(workflow, state, thread_id="123"):
    # Events are pre-enriched
    yield event
```

This generic approach provides a solid foundation for workflow execution across the entire application while maintaining flexibility and extensibility.