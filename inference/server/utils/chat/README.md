# Chat Completion and Agentic Workflow Module

This module provides a simplified yet powerful implementation for chat completions with both standard and agentic workflows.

## Overview

The system implements:

1. **Standard Chat Completions**: Using model-based generation with proper streaming support
2. **Agentic Workflows**: For tool-enhanced chat completions when computational or complex tasks are detected
3. **RAG Integration**: Seamless integration with retrieval augmented generation through enhanced messages

## Key Components

### Main Functions

- `enhanced_chat_completion_logic`: Entry point that decides between standard or agentic workflow
- `generate_streaming_response`: Streams tokens from the model for real-time responses
- `generate_complete_response`: Generates complete non-streaming responses
- `stream_agentic_response`: Simulates streaming for agentic responses that were generated as a whole

### Utility Functions

- `_extract_model_parameters`: Helper to extract parameters from model profiles
- `prepare_enhanced_messages`: (In workflow.py) Prepares messages with RAG context
- `should_use_agentic_workflow`: (In workflow.py) Detects when agentic tools would be beneficial

## Usage

The module is designed to be called from the chat router:

```python
# Example usage in router
result = await enhanced_chat_completion_logic(
    conversation_ctx=context,
    model_profile=model_profile,
    stream=stream_enabled,
    background_tasks=background_tasks
)
return result  # Will be either StreamingResponse or ChatResponse
```

## Streaming Implementation

The streaming implementation follows these steps:

1. Initialize a streaming response with proper event stream format
2. For each token from the model:
   - Append the token to the full response
   - Yield the token as a properly formatted SSE event
3. When complete, send a final "done" message with the full text
4. Store the complete message in the background

## Error Handling

All operations have proper error handling to ensure that:

- Exceptions during generation are caught and logged
- Error messages are returned to the client in the expected format
- Background tasks for message storage handle failures gracefully

## Best Practices

- Always use background tasks for storing messages to avoid delaying responses
- Make sure model parameters are properly extracted and passed to pipelines
- Use proper content types for messages to ensure compatibility
