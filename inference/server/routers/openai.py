"""
OpenAI API compatibility router.
Implements OpenAI-compatible endpoints for IDE integration.
See: https://platform.openai.com/docs/api-reference
"""

from typing import Any, AsyncIterator, Dict, List, Literal, Optional, Union
from datetime import datetime
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from server.middleware.auth import get_user_id
from db import storage
from models import Message, MessageRole, MessageContent, MessageContentType
from utils.logging import llmmllogger
import composer

logger = llmmllogger.bind(component="openai_router")
router = APIRouter(prefix="/v1", tags=["openai"])


# === OpenAI Request/Response Models ===

class OpenAIFunctionCall(BaseModel):
    """Function call in a message"""
    name: str
    arguments: str  # JSON string


class OpenAIToolCall(BaseModel):
    """Tool call in a message"""
    id: str
    type: str = "function"
    function: OpenAIFunctionCall


class OpenAIChatMessage(BaseModel):
    """OpenAI chat message format"""
    role: Literal["system", "user", "assistant", "tool", "function"]
    content: Optional[Union[str, List[Dict[str, Any]]]] = None  # String or array of content parts
    name: Optional[str] = None
    tool_calls: Optional[List[OpenAIToolCall]] = None
    tool_call_id: Optional[str] = None  # For tool/function responses
    function_call: Optional[OpenAIFunctionCall] = None  # Deprecated but supported


class OpenAIFunctionDefinition(BaseModel):
    """Function definition for tools"""
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class OpenAITool(BaseModel):
    """Tool definition"""
    type: str = "function"
    function: OpenAIFunctionDefinition


class OpenAIResponseFormat(BaseModel):
    """Response format specification"""
    type: Literal["text", "json_object", "json_schema"] = "text"
    json_schema: Optional[Dict[str, Any]] = None


class OpenAIStreamOptions(BaseModel):
    """Streaming options"""
    include_usage: Optional[bool] = False


class OpenAIChatCompletionRequest(BaseModel):
    """Request model for /v1/chat/completions endpoint"""
    model: str
    messages: List[OpenAIChatMessage]
    
    # Sampling parameters
    temperature: Optional[float] = Field(1.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0)
    n: Optional[int] = Field(1, ge=1)
    stream: Optional[bool] = False
    stream_options: Optional[OpenAIStreamOptions] = None
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    
    # Function/tool calling
    tools: Optional[List[OpenAITool]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = "auto"
    parallel_tool_calls: Optional[bool] = True
    
    # Response format
    response_format: Optional[OpenAIResponseFormat] = None
    
    # Additional parameters
    presence_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    logit_bias: Optional[Dict[str, float]] = None
    logprobs: Optional[bool] = False
    top_logprobs: Optional[int] = None
    user: Optional[str] = None
    seed: Optional[int] = None
    
    # Service tier
    service_tier: Optional[Literal["auto", "default"]] = "auto"
    
    # Metadata
    metadata: Optional[Dict[str, Any]] = None


class OpenAICompletionRequest(BaseModel):
    """Request model for /v1/completions endpoint (legacy)"""
    model: str
    prompt: Union[str, List[str], List[int], List[List[int]]]
    
    # Sampling parameters
    temperature: Optional[float] = Field(1.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0)
    n: Optional[int] = Field(1, ge=1)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = 16
    
    # Additional parameters
    presence_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    logit_bias: Optional[Dict[str, float]] = None
    logprobs: Optional[int] = None
    echo: Optional[bool] = False
    suffix: Optional[str] = None
    best_of: Optional[int] = 1
    user: Optional[str] = None
    seed: Optional[int] = None


class OpenAIEmbeddingRequest(BaseModel):
    """Request model for /v1/embeddings endpoint"""
    model: str
    input: Union[str, List[str], List[int], List[List[int]]]
    encoding_format: Optional[Literal["float", "base64"]] = "float"
    dimensions: Optional[int] = None
    user: Optional[str] = None


# === Utility Functions ===

def convert_openai_to_internal_messages(openai_messages: List[OpenAIChatMessage], conversation_id: int) -> List[Message]:
    """Convert OpenAI message format to internal Message format"""
    messages = []
    for msg in openai_messages:
        # Map roles
        if msg.role == "system":
            role = MessageRole.SYSTEM
        elif msg.role == "user":
            role = MessageRole.USER
        elif msg.role == "assistant":
            role = MessageRole.ASSISTANT
        elif msg.role in ["tool", "function"]:
            role = MessageRole.TOOL
        else:
            role = MessageRole.USER  # Fallback
        
        # Handle content (can be string or array)
        content = []
        if isinstance(msg.content, str):
            content.append(MessageContent(
                type=MessageContentType.TEXT,
                text=msg.content
            ))
        elif isinstance(msg.content, list):
            for item in msg.content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        content.append(MessageContent(
                            type=MessageContentType.TEXT,
                            text=item.get("text", "")
                        ))
                    elif item.get("type") == "image_url":
                        image_url = item.get("image_url", {})
                        url = image_url.get("url", "") if isinstance(image_url, dict) else str(image_url)
                        content.append(MessageContent(
                            type=MessageContentType.IMAGE,
                            image_url=url
                        ))
        
        # If no content was extracted, add empty text
        if not content:
            content.append(MessageContent(
                type=MessageContentType.TEXT,
                text=""
            ))
        
        messages.append(Message(
            role=role,
            content=content,
            conversation_id=conversation_id
        ))
    
    return messages


async def stream_composer_to_openai_chat(
    user_id: str,
    conversation_id: int,
    request_id: str,
    model_name: str
) -> AsyncIterator[str]:
    """Stream composer events in OpenAI chat completion format"""
    workflow = await composer.compose_workflow(user_id, None)
    initial_state = await composer.create_initial_state(user_id, conversation_id)
    
    chunk_id = f"chatcmpl-{uuid4().hex[:29]}"
    
    async for event in composer.execute_workflow(initial_state, workflow):
        if event.message and event.message.content:
            text_parts = [c.text for c in event.message.content if c.type == MessageContentType.TEXT and c.text]
            response_text = "".join(text_parts)
            
            # Stream in OpenAI SSE format
            chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": int(datetime.utcnow().timestamp()),
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "delta": {"content": response_text},
                    "finish_reason": None
                }]
            }
            
            import json
            yield f"data: {json.dumps(chunk)}\n\n"
    
    # Final done chunk
    final_chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(datetime.utcnow().timestamp()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    }
    
    import json
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


# === OpenAI Endpoints ===

@router.post("/chat/completions")
async def create_chat_completion(body: OpenAIChatCompletionRequest, request: Request):
    """
    Create a chat completion.
    OpenAI-compatible endpoint for /v1/chat/completions
    """
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    # Create or get conversation
    conversation = await storage.get_service(storage.conversation).create_conversation(user_id)
    conversation_id = conversation.id
    
    # Convert and store all messages
    internal_messages = convert_openai_to_internal_messages(body.messages, conversation_id)
    for msg in internal_messages:
        await storage.get_service(storage.message).add_message(msg)
    
    request_id = str(uuid4())
    
    if body.stream:
        return StreamingResponse(
            stream_composer_to_openai_chat(user_id, conversation_id, request_id, body.model),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    else:
        # TODO: Implement non-streaming response
        raise HTTPException(status_code=501, detail="Non-streaming not yet implemented")


@router.post("/completions")
async def create_completion(body: OpenAICompletionRequest, request: Request):
    """
    Create a completion (legacy endpoint).
    OpenAI-compatible endpoint for /v1/completions
    """
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    # Convert prompt to message format
    if isinstance(body.prompt, str):
        prompts = [body.prompt]
    elif isinstance(body.prompt, list):
        if body.prompt and isinstance(body.prompt[0], (list, int)):
            # Token IDs - not supported
            raise HTTPException(status_code=400, detail="Token ID prompts not supported")
        prompts = body.prompt
    else:
        prompts = [str(body.prompt)]
    
    # For now, only support single prompt
    if len(prompts) > 1:
        raise HTTPException(status_code=400, detail="Multiple prompts not yet supported")
    
    # Create conversation and message
    conversation = await storage.get_service(storage.conversation).create_conversation(user_id)
    conversation_id = conversation.id
    
    user_message = Message(
        role=MessageRole.USER,
        content=[MessageContent(type=MessageContentType.TEXT, text=prompts[0])],
        conversation_id=conversation_id
    )
    await storage.get_service(storage.message).add_message(user_message)
    
    request_id = str(uuid4())
    
    if body.stream:
        # TODO: Implement streaming for completions
        raise HTTPException(status_code=501, detail="Streaming completions not yet implemented")
    else:
        # TODO: Implement non-streaming response
        raise HTTPException(status_code=501, detail="Non-streaming completions not yet implemented")


@router.post("/embeddings")
async def create_embeddings(body: OpenAIEmbeddingRequest, request: Request):
    """
    Create embeddings.
    OpenAI-compatible endpoint for /v1/embeddings
    """
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    # TODO: Implement embeddings via runner
    raise HTTPException(status_code=501, detail="Embeddings not yet implemented")


@router.get("/models")
async def list_models(request: Request):
    """
    List available models.
    OpenAI-compatible endpoint for /v1/models
    """
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        from runner.utils.model_loader import ModelLoader
        
        model_loader = ModelLoader()
        models_dict = model_loader.get_available_models()
        
        # Convert to OpenAI format
        openai_models = []
        for model_name in models_dict.keys():
            openai_models.append({
                "id": model_name,
                "object": "model",
                "created": int(datetime.utcnow().timestamp()),
                "owned_by": "system"
            })
        
        return {
            "object": "list",
            "data": openai_models
        }
    
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_id}")
async def retrieve_model(model_id: str, request: Request):
    """
    Retrieve a model.
    OpenAI-compatible endpoint for /v1/models/{model_id}
    """
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        from runner.utils.model_loader import ModelLoader
        
        model_loader = ModelLoader()
        models_dict = model_loader.get_available_models()
        
        if model_id not in models_dict:
            raise HTTPException(status_code=404, detail="Model not found")
        
        return {
            "id": model_id,
            "object": "model",
            "created": int(datetime.utcnow().timestamp()),
            "owned_by": "system"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving model: {e}")
        raise HTTPException(status_code=500, detail=str(e))
