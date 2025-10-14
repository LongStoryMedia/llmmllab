# Streaming with Node Metadata

This document explains how to use the new abstracted pipeline execution methods in BaseAgent that automatically inject node metadata into streaming responses.

## Overview

The BaseAgent class now provides two main methods for pipeline execution:

1. `stream_pipeline_with_metadata()` - For streaming responses with metadata injection
2. `run_pipeline_with_metadata()` - For non-streaming responses with metadata injection

These methods automatically inject node metadata into responses/chunks using the `channels` field of `ChatResponse`, providing better observability and context about what type of content is being generated.

## Key Features

- **Automatic metadata injection**: Node information is automatically added to every chunk/response
- **Boundary markers**: Start/end chunks mark the beginning and end of node execution
- **Error handling**: Errors are wrapped with node context
- **Consistent logging**: All agents get consistent logging patterns

## Metadata Structure

The metadata is injected into the `channels` field of `ChatResponse` objects with the following structure:

```python
{
    "node_metadata": {
        "node_name": "Chat Agent",
        "node_id": "chat_agent_001", 
        "node_type": "ChatAgent",
        "user_id": "user123",
        "conversation_id": 456
    },
    "stream_metadata": {  # Only for boundary chunks
        "is_boundary": True,
        "is_start": True,  # or False for end chunks
        "content_type": "stream_start",  # or "stream_end", "stream_error"
        "node_operation": "pipeline_execution"
    },
    "chunk_metadata": {  # Only for content chunks
        "chunk_index": 5,
        "is_boundary": False
    }
}
```

## Usage Examples

### In Agent Implementation

```python
class ChatAgent(BaseAgent[ChatResponse]):
    async def chat_completion(self, messages, user_id, tools=None, stream=None):
        if stream:
            # Streaming with metadata
            return await self._execute_streaming_completion_with_metadata(
                messages, user_id, tools
            )
        else:
            # Non-streaming with metadata
            return await self.run_pipeline_with_metadata(
                messages=messages,
                user_id=user_id,
                tools=tools,
                priority=self.priority
            )
    
    async def stream_chat_completion(self, messages, user_id, tools=None):
        """Direct streaming method for LangGraph integration."""
        async for chunk in self.stream_pipeline_with_metadata(
            messages=messages,
            user_id=user_id,
            tools=tools,
            priority=self.priority
        ):
            yield chunk
```

### In LangGraph Nodes

```python
async def chat_node(state: WorkflowState) -> WorkflowState:
    chat_agent = get_chat_agent()  # Your agent creation logic
    
    # Stream with metadata - each chunk contains node context
    async for chunk in chat_agent.stream_chat_completion(
        messages=state.messages,
        user_id=state.user_id,
        tools=state.available_tools
    ):
        # Check if this is a boundary chunk
        if chunk.channels and chunk.channels.get("stream_metadata", {}).get("is_boundary"):
            if chunk.channels["stream_metadata"]["is_start"]:
                print(f"Starting {chunk.channels['node_metadata']['node_name']}")
            else:
                print(f"Finished {chunk.channels['node_metadata']['node_name']}")
        else:
            # Regular content chunk with metadata
            print(f"Content from {chunk.channels['node_metadata']['node_name']}: {chunk.message.content}")
    
    return state
```

### Processing Streaming Responses

```python
async def process_stream_with_metadata(agent, messages, user_id):
    content_chunks = []
    
    async for chunk in agent.stream_pipeline_with_metadata(
        messages=messages,
        user_id=user_id
    ):
        # Extract metadata
        node_meta = chunk.channels.get("node_metadata", {}) if chunk.channels else {}
        stream_meta = chunk.channels.get("stream_metadata", {}) if chunk.channels else {}
        
        if stream_meta.get("is_boundary"):
            if stream_meta.get("is_start"):
                print(f"🚀 Starting {node_meta.get('node_name')} operation")
            elif stream_meta.get("content_type") == "stream_end":
                print(f"✅ Completed {node_meta.get('node_name')} operation")
            elif stream_meta.get("content_type") == "stream_error":
                print(f"❌ Error in {node_meta.get('node_name')}: {stream_meta.get('error')}")
        else:
            # Regular content chunk
            if chunk.message and chunk.message.content:
                content_chunks.append(chunk)
                print(f"📝 Content chunk {stream_meta.get('chunk_index', '?')} from {node_meta.get('node_name')}")
    
    return content_chunks
```

## Benefits

1. **Better Observability**: Know exactly which node is generating content
2. **Debugging**: Clear boundaries and error context
3. **Monitoring**: Track performance per node type
4. **Client Integration**: Frontend can display different UI for different node types
5. **Analytics**: Better understanding of workflow execution patterns

## LangGraph Integration

The metadata injection works seamlessly with LangGraph's native streaming capabilities. You can use LangGraph's streaming features while getting enhanced metadata about node execution.

For complex workflows, you can track the flow of execution across multiple nodes and understand where time is being spent or where errors occur.

## Migration Guide

To migrate existing agents:

1. Replace direct `stream_pipeline()` calls with `self.stream_pipeline_with_metadata()`
2. Replace direct `run_pipeline()` calls with `self.run_pipeline_with_metadata()`
3. Update streaming accumulation logic to handle boundary chunks
4. Use the metadata for enhanced logging and debugging

The new methods are drop-in replacements that provide the same functionality with added metadata injection.
