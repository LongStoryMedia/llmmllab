"""
Ollama API compatibility router.
Implements Ollama-compatible endpoints for IDE integration.
See: https://github.com/ollama/ollama/blob/main/docs/api.md
"""

from typing import Any, AsyncIterator, Dict, List, Optional, Union
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
from runner.utils.model_loader import ModelLoader

logger = llmmllogger.bind(component="ollama_router")
router = APIRouter(prefix="/api", tags=["ollama"])


# === Ollama Request/Response Models ===

class OllamaGenerateRequest(BaseModel):
    """Request model for /api/generate endpoint"""
    model: str
    prompt: str
    suffix: Optional[str] = None
    images: Optional[List[str]] = None
    think: Optional[bool] = None
    format: Optional[Union[str, Dict[str, Any]]] = None  # "json" or JSON schema
    options: Optional[Dict[str, Any]] = None
    system: Optional[str] = None
    template: Optional[str] = None
    stream: Optional[bool] = True
    raw: Optional[bool] = False
    keep_alive: Optional[Union[str, int]] = "5m"
    context: Optional[List[int]] = None  # Deprecated


class OllamaMessage(BaseModel):
    """Ollama chat message format"""
    role: str  # system, user, assistant, tool
    content: str
    thinking: Optional[str] = None  # For thinking models
    images: Optional[List[str]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_name: Optional[str] = None  # For tool responses


class OllamaTool(BaseModel):
    """Ollama tool definition"""
    type: str = "function"
    function: Dict[str, Any]


class OllamaChatRequest(BaseModel):
    """Request model for /api/chat endpoint"""
    model: str
    messages: List[OllamaMessage]
    tools: Optional[List[OllamaTool]] = None
    think: Optional[bool] = None
    format: Optional[Union[str, Dict[str, Any]]] = None
    options: Optional[Dict[str, Any]] = None
    stream: Optional[bool] = True
    keep_alive: Optional[Union[str, int]] = "5m"


class OllamaEmbedRequest(BaseModel):
    """Request model for /api/embed endpoint"""
    model: str
    input: Union[str, List[str]]  # Single string or list of strings
    truncate: Optional[bool] = True
    options: Optional[Dict[str, Any]] = None
    keep_alive: Optional[Union[str, int]] = "5m"
    dimensions: Optional[int] = None


class OllamaShowRequest(BaseModel):
    """Request for /api/show endpoint"""
    model: str
    verbose: Optional[bool] = False


class OllamaCreateRequest(BaseModel):
    """Request for /api/create endpoint"""
    model: str
    from_: Optional[str] = Field(None, alias="from")
    files: Optional[Dict[str, str]] = None  # filename -> SHA256
    adapters: Optional[Dict[str, str]] = None
    template: Optional[str] = None
    license: Optional[Union[str, List[str]]] = None
    system: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    messages: Optional[List[OllamaMessage]] = None
    stream: Optional[bool] = True
    quantize: Optional[str] = None


class OllamaCopyRequest(BaseModel):
    """Request for /api/copy endpoint"""
    source: str
    destination: str


class OllamaDeleteRequest(BaseModel):
    """Request for /api/delete endpoint"""
    model: str


class OllamaPullRequest(BaseModel):
    """Request for /api/pull endpoint"""
    model: str
    insecure: Optional[bool] = False
    stream: Optional[bool] = True


class OllamaPushRequest(BaseModel):
    """Request for /api/push endpoint"""
    model: str
    insecure: Optional[bool] = False
    stream: Optional[bool] = True


# === Utility Functions ===

def convert_ollama_to_internal_messages(ollama_messages: List[OllamaMessage], conversation_id: int) -> List[Message]:
    """Convert Ollama message format to internal Message format"""
    messages = []
    for msg in ollama_messages:
        # Map roles
        if msg.role == "system":
            role = MessageRole.SYSTEM
        elif msg.role == "user":
            role = MessageRole.USER
        elif msg.role == "assistant":
            role = MessageRole.ASSISTANT
        elif msg.role == "tool":
            role = MessageRole.TOOL
        else:
            role = MessageRole.USER  # Fallback
        
        # Create content
        content = [MessageContent(
            type=MessageContentType.TEXT,
            text=msg.content
        )]
        
        # Handle images if present
        if msg.images:
            for img_data in msg.images:
                content.append(MessageContent(
                    type=MessageContentType.IMAGE,
                    image_data=img_data
                ))
        
        messages.append(Message(
            role=role,
            content=content,
            conversation_id=conversation_id
        ))
    
    return messages


async def stream_composer_to_ollama_generate(
    user_id: str,
    conversation_id: int,
    request_id: str,
    model_name: str
) -> AsyncIterator[str]:
    """Stream composer events in Ollama generate format"""
    workflow = await composer.compose_workflow(user_id, None)
    initial_state = await composer.create_initial_state(user_id, conversation_id)
    
    async for event in composer.execute_workflow(initial_state, workflow):
        if event.message and event.message.content:
            # Extract text from content
            text_parts = [c.text for c in event.message.content if c.type == MessageContentType.TEXT and c.text]
            response_text = "".join(text_parts)
            
            # Stream in Ollama format
            yield f'{{"model":"{model_name}","created_at":"{datetime.utcnow().isoformat()}Z","response":"{response_text}","done":false}}\n'
    
    # Final done message
    yield f'{{"model":"{model_name}","created_at":"{datetime.utcnow().isoformat()}Z","response":"","done":true}}\n'


async def stream_composer_to_ollama_chat(
    user_id: str,
    conversation_id: int,
    request_id: str,
    model_name: str
) -> AsyncIterator[str]:
    """Stream composer events in Ollama chat format"""
    workflow = await composer.compose_workflow(user_id, None)
    initial_state = await composer.create_initial_state(user_id, conversation_id)
    
    async for event in composer.execute_workflow(initial_state, workflow):
        if event.message and event.message.content:
            text_parts = [c.text for c in event.message.content if c.type == MessageContentType.TEXT and c.text]
            response_text = "".join(text_parts)
            
            # Stream in Ollama chat format
            yield f'{{"model":"{model_name}","created_at":"{datetime.utcnow().isoformat()}Z","message":{{"role":"assistant","content":"{response_text}"}},"done":false}}\n'
    
    # Final done message
    yield f'{{"model":"{model_name}","created_at":"{datetime.utcnow().isoformat()}Z","message":{{"role":"assistant","content":""}},"done":true}}\n'


# === Ollama Endpoints ===

@router.post("/generate")
async def generate(body: OllamaGenerateRequest, request: Request):
    """
    Generate a completion for a given prompt.
    Ollama-compatible endpoint for /api/generate
    """
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    # Create or get conversation
    conversation = await storage.get_service(storage.conversation).create_conversation(user_id)
    conversation_id = conversation.id
    
    # Convert to internal message format
    user_message = Message(
        role=MessageRole.USER,
        content=[MessageContent(type=MessageContentType.TEXT, text=body.prompt)],
        conversation_id=conversation_id
    )
    
    # Add system message if provided
    if body.system:
        system_message = Message(
            role=MessageRole.SYSTEM,
            content=[MessageContent(type=MessageContentType.TEXT, text=body.system)],
            conversation_id=conversation_id
        )
        await storage.get_service(storage.message).add_message(system_message)
    
    # Store user message
    await storage.get_service(storage.message).add_message(user_message)
    
    request_id = str(uuid4())
    
    if body.stream:
        return StreamingResponse(
            stream_composer_to_ollama_generate(user_id, conversation_id, request_id, body.model),
            media_type="application/x-ndjson"
        )
    else:
        # TODO: Implement non-streaming response
        raise HTTPException(status_code=501, detail="Non-streaming not yet implemented")


@router.post("/chat")
async def chat(body: OllamaChatRequest, request: Request):
    """
    Generate a chat completion.
    Ollama-compatible endpoint for /api/chat
    """
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    # Create or get conversation
    conversation = await storage.get_service(storage.conversation).create_conversation(user_id)
    conversation_id = conversation.id
    
    # Convert and store all messages
    internal_messages = convert_ollama_to_internal_messages(body.messages, conversation_id)
    for msg in internal_messages:
        await storage.get_service(storage.message).add_message(msg)
    
    request_id = str(uuid4())
    
    if body.stream:
        return StreamingResponse(
            stream_composer_to_ollama_chat(user_id, conversation_id, request_id, body.model),
            media_type="application/x-ndjson"
        )
    else:
        # TODO: Implement non-streaming response
        raise HTTPException(status_code=501, detail="Non-streaming not yet implemented")


@router.get("/tags")
async def list_local_models(request: Request):
    """
    List locally available models.
    Ollama-compatible endpoint for /api/tags
    """
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        model_loader = ModelLoader()
        models_dict = model_loader.get_available_models()
        
        # Convert to Ollama format
        ollama_models = []
        for model_name, model_obj in models_dict.items():
            ollama_models.append({
                "name": model_name,
                "model": model_name,
                "modified_at": datetime.utcnow().isoformat() + "Z",
                "size": 0,  # TODO: Get actual size
                "digest": "",  # TODO: Generate digest
                "details": {
                    "parent_model": "",
                    "format": "gguf",
                    "family": model_obj.family if hasattr(model_obj, 'family') else "unknown",
                    "families": [model_obj.family] if hasattr(model_obj, 'family') else [],
                    "parameter_size": model_obj.params if hasattr(model_obj, 'params') else "unknown",
                    "quantization_level": model_obj.quantization if hasattr(model_obj, 'quantization') else "unknown"
                }
            })
        
        return {"models": ollama_models}
    
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/show")
async def show_model_info(body: OllamaShowRequest, request: Request):
    """
    Show information about a model.
    Ollama-compatible endpoint for /api/show
    """
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        model_loader = ModelLoader()
        models_dict = model_loader.get_available_models()
        
        if body.model not in models_dict:
            raise HTTPException(status_code=404, detail="Model not found")
        
        model_obj = models_dict[body.model]
        
        return {
            "modelfile": f"FROM {body.model}",
            "parameters": "",
            "template": "",
            "details": {
                "parent_model": "",
                "format": "gguf",
                "family": model_obj.family if hasattr(model_obj, 'family') else "unknown",
                "families": [model_obj.family] if hasattr(model_obj, 'family') else [],
                "parameter_size": model_obj.params if hasattr(model_obj, 'params') else "unknown",
                "quantization_level": model_obj.quantization if hasattr(model_obj, 'quantization') else "unknown"
            },
            "model_info": {},
            "capabilities": ["completion"]
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error showing model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/create")
async def create_model(body: OllamaCreateRequest, request: Request):
    """
    Create a model.
    Ollama-compatible endpoint for /api/create
    """
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    # Not implemented for this system - would require model file management
    raise HTTPException(status_code=501, detail="Model creation not supported")


@router.post("/copy")
async def copy_model(body: OllamaCopyRequest, request: Request):
    """
    Copy a model.
    Ollama-compatible endpoint for /api/copy
    """
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    # Not implemented - would require model file management
    raise HTTPException(status_code=501, detail="Model copy not supported")


@router.delete("/delete")
async def delete_model(body: OllamaDeleteRequest, request: Request):
    """
    Delete a model.
    Ollama-compatible endpoint for /api/delete
    """
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    # Not implemented - would require model file management
    raise HTTPException(status_code=501, detail="Model deletion not supported")


@router.post("/pull")
async def pull_model(body: OllamaPullRequest, request: Request):
    """
    Pull a model from the library.
    Ollama-compatible endpoint for /api/pull
    """
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    # Not implemented - would require model downloading
    raise HTTPException(status_code=501, detail="Model pull not supported")


@router.post("/push")
async def push_model(body: OllamaPushRequest, request: Request):
    """
    Push a model to the library.
    Ollama-compatible endpoint for /api/push
    """
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    # Not implemented - would require model uploading
    raise HTTPException(status_code=501, detail="Model push not supported")


@router.post("/embed")
async def generate_embeddings(body: OllamaEmbedRequest, request: Request):
    """
    Generate embeddings from a model.
    Ollama-compatible endpoint for /api/embed
    """
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    # TODO: Implement embeddings via runner
    raise HTTPException(status_code=501, detail="Embeddings not yet implemented")


@router.get("/ps")
async def list_running_models(request: Request):
    """
    List currently loaded models.
    Ollama-compatible endpoint for /api/ps
    """
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    # Return empty list - we don't track loaded models currently
    return {"models": []}


@router.get("/version")
async def get_version(request: Request):
    """
    Get Ollama version.
    Ollama-compatible endpoint for /api/version
    """
    return {"version": "0.1.0"}
