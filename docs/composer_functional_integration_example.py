"""
Example of how server completion handler would integrate with composer.
This shows the clean functional interface usage.
"""

import asyncio
from typing import AsyncIterator, Dict, Any, Optional
from fastapi import HTTPException

# Instead of HTTP calls, direct functional imports
from composer import (
    compose_workflow,
    create_initial_state,
    execute_workflow,
    get_composer_config
)
from models import ChatResponse, ConversationCtx
from server.config import logger


async def handle_chat_completion_with_composer(
    conversation_ctx: ConversationCtx,
    stream: bool = True,
    config_overrides: Optional[Dict[str, Any]] = None
) -> AsyncIterator[ChatResponse]:
    """
    Handle chat completion using composer service.
    Clean functional interface - no HTTP overhead.
    """
    try:
        # Step 1: Compose workflow based on conversation context
        # This replaces manual orchestration logic
        workflow = await compose_workflow(
            conversation_ctx=conversation_ctx,
            workflow_type="CHAT",  # Could be determined from intent
            config_overrides=config_overrides
        )
        
        # Step 2: Create initial state from conversation
        initial_state = await create_initial_state(
            conversation_ctx=conversation_ctx,
            workflow_type="CHAT",
            additional_context={"stream_enabled": stream}
        )
        
        # Step 3: Execute workflow with streaming
        async for event in execute_workflow(
            workflow=workflow,
            initial_state=initial_state,
            stream=stream
        ):
            # Handle different event types
            event_type = event.get("event", "")
            
            if event_type == "on_chat_model_stream" and stream:
                # Real-time token streaming from primary chat node
                chunk_data = event.get("data", {})
                yield ChatResponse.create_streaming_chunk(chunk_data)
                
            elif event_type == "on_chain_end":
                # Node completion updates
                node_name = event.get("name", "")
                if node_name == "agent":
                    # Final response from chat node
                    output = event.get("data", {}).get("output", {})
                    if "messages" in output:
                        final_message = output["messages"][-1]
                        yield ChatResponse.create_final_chunk(final_message)
                        
            elif event_type == "on_tool_start":
                # Tool execution started - send progress update
                tool_name = event.get("name", "unknown_tool")
                yield ChatResponse.create_progress_chunk(f"Using {tool_name}...")
                
            elif event_type == "on_tool_end":
                # Tool execution completed
                tool_name = event.get("name", "unknown_tool")
                yield ChatResponse.create_progress_chunk(f"Completed {tool_name}")

    except Exception as e:
        logger.error(f"Composer workflow execution failed: {e}", exc_info=True)
        yield ChatResponse.create_error_chunk(f"Workflow execution failed: {str(e)}")


async def handle_research_workflow_example(
    conversation_ctx: ConversationCtx,
    search_depth: str = "SHALLOW"
) -> AsyncIterator[ChatResponse]:
    """
    Example of research workflow with configurable RAG depth.
    Shows how composer handles different workflow types.
    """
    try:
        # Configure research-specific settings
        config_overrides = {
            "rag_depth": search_depth,
            "max_sources": 10 if search_depth == "DEEP" else 3,
            "enable_web_crawl": search_depth == "DEEP"
        }
        
        # Compose research workflow
        workflow = await compose_workflow(
            conversation_ctx=conversation_ctx,
            workflow_type="RESEARCH", 
            config_overrides=config_overrides
        )
        
        initial_state = await create_initial_state(
            conversation_ctx=conversation_ctx,
            workflow_type="RESEARCH",
            additional_context={"search_depth": search_depth}
        )
        
        # Execute with progress tracking
        async for event in execute_workflow(workflow, initial_state, stream=True):
            event_type = event.get("event", "")
            
            if event_type == "on_chain_start":
                node_name = event.get("name", "")
                if node_name == "intent_classifier":
                    yield ChatResponse.create_progress_chunk("Analyzing research intent...")
                elif node_name == "shallow_search":
                    yield ChatResponse.create_progress_chunk("Performing quick search...")
                elif node_name == "deep_crawl":
                    yield ChatResponse.create_progress_chunk("Starting deep research crawl...")
                    
            elif event_type == "on_chain_end":
                node_name = event.get("name", "")
                if node_name == "synthesis":
                    # Research synthesis completed
                    output = event.get("data", {}).get("output", {})
                    synthesis_result = output.get("search_results", "")
                    yield ChatResponse.create_final_chunk({
                        "content": synthesis_result,
                        "metadata": {"research_depth": search_depth}
                    })

    except Exception as e:
        logger.error(f"Research workflow failed: {e}", exc_info=True)
        yield ChatResponse.create_error_chunk(f"Research failed: {str(e)}")


def get_composer_status() -> Dict[str, Any]:
    """Get composer service status for health checks."""
    try:
        config = get_composer_config()
        return {
            "status": "healthy",
            "composer": config
        }
    except RuntimeError as e:
        return {
            "status": "unhealthy", 
            "error": str(e)
        }


# Integration with existing completion endpoint would be:
"""
@app.post("/chat/completions")
async def chat_completions(request: ChatRequest):
    conversation_ctx = await get_conversation_context_from_request(request)
    
    # Instead of manual orchestration, use composer
    async for chunk in handle_chat_completion_with_composer(
        conversation_ctx=conversation_ctx,
        stream=request.stream,
        config_overrides=request.config
    ):
        yield serialize_to_json(chunk)
"""