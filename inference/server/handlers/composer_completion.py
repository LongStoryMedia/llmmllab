"""
Simplified completion handler that delegates to composer.
All chat logic has been moved to the composer module.
"""

from typing import AsyncGenerator
import json
from fastapi import BackgroundTasks
from server.config import logger

# Import composer interface
import composer


async def composer_chat_completion(
    user_id: str, 
    conversation_id: int, 
    background_tasks: BackgroundTasks
) -> AsyncGenerator[str, None]:
    """
    Handle chat completions by delegating to composer interface.
    
    Args:
        user_id: User ID for the request
        conversation_id: Conversation ID for context
        background_tasks: FastAPI background tasks
        
    Yields:
        Server-Sent Events formatted strings
    """
    try:
        # Initialize composer service if needed
        await composer.initialize_composer()
        
        # Compose workflow for user
        workflow = await composer.compose_workflow(user_id)
        
        # Create initial state
        initial_state = await composer.create_initial_state(user_id, conversation_id)
        
        # Execute workflow with streaming
        async for event in composer.execute_workflow(workflow, initial_state, stream=True):
            # Convert composer events to SSE format
            if isinstance(event, dict):
                # Handle different event types
                event_type = event.get("event", "chunk")
                
                if event_type == "on_llm_stream":
                    # Stream token from LLM
                    chunk = event.get("data", {}).get("chunk", {})
                    if chunk:
                        content = chunk.get("content", "") if isinstance(chunk, dict) else str(chunk)
                        if content:
                            yield f"data: {json.dumps({'content': content})}\n\n"
                elif event_type == "on_chain_end":
                    # End of workflow
                    yield f"data: {json.dumps({'type': 'done'})}\n\n"
                else:
                    # Other events - pass through
                    yield f"data: {json.dumps(event)}\n\n"
            else:
                # Handle raw string events
                yield f"data: {json.dumps({'content': str(event)})}\n\n"
                
    except Exception as e:
        logger.error(f"Error in composer chat completion: {e}")
        error_data = json.dumps({"error": str(e), "type": "error"})
        yield f"data: {error_data}\n\n"