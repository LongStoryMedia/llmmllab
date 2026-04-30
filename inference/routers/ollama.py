"""
Ollama API compatibility router.
Implements Ollama-compatible endpoints for IDE integration.
See: https://github.com/ollama/ollama/blob/main/docs/api.md

These endpoints use API key authentication for IDE integration.
"""

import json
from typing import Any, AsyncIterator, Dict, List, Optional, Union
from datetime import datetime
import ipaddress

from fastapi import APIRouter, HTTPException, Request, Depends, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from middleware.auth import get_auth_middleware
from models import (
    Message,
    MessageRole,
    MessageContent,
    MessageContentType,
)
from utils.logging import llmmllogger
import composer_init as composer
from runner.utils.model_loader import ModelLoader
from utils import extract_text_from_message  # Import logging utility

logger = llmmllogger.bind(component="ollama_router")
router = APIRouter(tags=["ollama"])
LOCAL_USER_ID = "local-network-user"  # Special user_id for local network access

# === API Key Authentication ===


async def verify_api_key_access(request: Request) -> str:
    """
    Verify API key access and return user_id.
    Supports both X-API-Key header and Authorization header with API key.
    Bypasses auth for local network requests (logs the IP).
    """
    client_ip = get_client_ip(request)

    # Allow local network access without authentication
    if is_local_network(client_ip):
        logger.info(f"Ollama API access from local IP (bypassing auth): {client_ip}")
        # Return a special local-network user_id
        return LOCAL_USER_ID

    # Check for X-API-Key header (preferred for IDE integration)
    api_key = request.headers.get("X-API-Key")
    if api_key:
        auth_middleware = get_auth_middleware()
        result = await auth_middleware.api_key_validator.validate_api_key(api_key)
        if result:
            logger.debug(
                f"Ollama API access granted with API key for user {result.user_id} from IP {client_ip}"
            )
            return result.user_id
        logger.warning(
            f"Invalid API key provided for Ollama endpoint from IP {client_ip}"
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired API key",
        )

    # Fall back to Authorization: Bearer <api_key> for compatibility
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        api_key = auth_header[7:]  # Remove "Bearer " prefix
        auth_middleware = get_auth_middleware()
        result = await auth_middleware.api_key_validator.validate_api_key(api_key)
        if result:
            logger.debug(
                f"Ollama API access granted with Bearer API key for user {result.user_id} from IP {client_ip}"
            )
            return result.user_id
        logger.warning(
            f"Invalid API key provided in Bearer token for Ollama endpoint from IP {client_ip}"
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired API key",
        )

    logger.warning(f"No API key provided for Ollama endpoint from IP {client_ip}")
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="API key required (use X-API-Key header or Authorization: Bearer <key>)",
    )


# === Local Network Validation (kept for reference, can be removed if not needed) ===

LOCAL_NETWORK_RANGES = [
    ipaddress.ip_network("10.0.0.0/8"),  # Class A private
    ipaddress.ip_network("172.16.0.0/12"),  # Class B private
    ipaddress.ip_network("192.168.0.0/16"),  # Class C private
    ipaddress.ip_network("127.0.0.0/8"),  # Loopback
]


def get_client_ip(request: Request) -> str:
    """Extract client IP address from request, considering X-Forwarded-For header"""
    # Check X-Forwarded-For first (set by reverse proxies)
    x_forwarded_for = request.headers.get("x-forwarded-for")
    if x_forwarded_for:
        # X-Forwarded-For can contain multiple IPs, use the first one
        return x_forwarded_for.split(",")[0].strip()

    # Fall back to direct connection IP
    return request.client.host if request.client else "0.0.0.0"


def is_local_network(ip_str: str) -> bool:
    """Check if IP address is from a local network"""
    try:
        ip = ipaddress.ip_address(ip_str)
        return any(ip in network for network in LOCAL_NETWORK_RANGES)
    except ValueError:
        logger.warning(f"Invalid IP address: {ip_str}")
        return False


def verify_local_network_access(request: Request) -> None:
    """
    Verify that request comes from local network.
    Raises HTTPException with 403 if not from local network.
    """
    client_ip = get_client_ip(request)

    if not is_local_network(client_ip):
        logger.warning(f"Ollama API access attempt from non-local IP: {client_ip}")
        raise HTTPException(
            status_code=403, detail="Ollama API is only accessible from local network"
        )

    logger.debug(f"Ollama API access from local IP: {client_ip}")


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


def convert_ollama_to_internal_messages(
    ollama_messages: List[OllamaMessage], conversation_id: int
) -> List[Message]:
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
        content = [MessageContent(type=MessageContentType.TEXT, text=msg.content)]

        # Handle images if present
        if msg.images:
            for img_data in msg.images:
                content.append(
                    MessageContent(type=MessageContentType.IMAGE, url=img_data)
                )

        messages.append(
            Message(role=role, content=content, conversation_id=conversation_id)
        )

    return messages


# async def stream_composer_to_ollama_generate(
#     user_id: str, conversation_id: int, model_name: str
# ) -> AsyncIterator[str]:
#     """Stream composer events in Ollama generate format"""
#     builder = await composer.get_graph_builder(composer.WorkFlowType.IDE, user_id)
#     workflow = await composer.compose_workflow(user_id, builder, None)
#     initial_state = await builder.create_initial_state(user_id, conversation_id)

#     async for event in composer.execute_workflow(initial_state, workflow):
#         if event.message and event.message.content:
#             # Extract text from content
#             text_parts = [
#                 c.text
#                 for c in event.message.content
#                 if c.type == MessageContentType.TEXT and c.text
#             ]
#             response_text = "".join(text_parts)

#             # Stream in Ollama format
#             yield f'{{"model":"{model_name}","created_at":"{datetime.utcnow().isoformat()}Z","response":"{response_text}","done":false}}\n'

#     # Final done message
#     yield f'{{"model":"{model_name}","created_at":"{datetime.utcnow().isoformat()}Z","response":"","done":true}}\n'


async def stream_composer_to_ollama_chat(
    messages: List[OllamaMessage],
) -> AsyncIterator[str]:
    """Stream composer events in Ollama chat format"""
    builder = await composer.get_graph_builder(composer.WorkFlowType.IDE, LOCAL_USER_ID)
    workflow = await composer.compose_workflow(LOCAL_USER_ID, builder, None)
    initial_state = await builder.create_initial_state(
        LOCAL_USER_ID,
        0,
        convert_ollama_to_internal_messages(messages, conversation_id=0),
    )

    async for event in composer.execute_workflow(initial_state, workflow):
        print(
            extract_text_from_message(event.message) if event.message else "",
            flush=True,
            end="",
        )  # Debug print
        if event.message and event.message.content:
            text_parts = [
                c.text
                for c in event.message.content
                if c.type == MessageContentType.TEXT and c.text
            ]
            response_text = "".join(text_parts)

            # Stream in Ollama chat format
            yield f'{{"model":"urmom","created_at":"{datetime.utcnow().isoformat()}Z","message":{{"role":"assistant","content":"{response_text}"}},"done":false}}\n'

    # Final done message
    yield f'{{"model":"urmomu","created_","created_at":"{datetime.utcnow().isoformat()}Z","message":{{"role":"assistant","content":""}},"done":true}}\n'


# === Ollama Endpoints ===


@router.post("/generate")
@router.post("/chat")
@router.post("/api/chat")
@router.post("/chat/completions")
async def chat(
    body: OllamaChatRequest,
    request: Request,
    user_id: str = Depends(verify_api_key_access),
):
    """
    Generate a chat completion.
    Ollama-compatible endpoint for /api/chat
    Requires API key authentication via X-API-Key header or Authorization: Bearer <key>
    """
    logger.debug(
        f"Received chat request from user {user_id} with model {body.model}\n\n {json.dumps(body.dict(), indent=2)}\n\n {request.headers} \n\n {await request.body()}"
    )
    if body.stream:
        return StreamingResponse(
            stream_composer_to_ollama_chat(body.messages),
            media_type="application/x-ndjson",
        )
    else:
        raise HTTPException(status_code=501, detail="Non-streaming not yet implemented")


@router.get("/api/tags")
async def list_local_models(request: Request):
    """
    List locally available models.
    Ollama-compatible endpoint for /api/tags
    Requires API key authentication via X-API-Key header or Authorization: Bearer <key>
    """
    verify_local_network_access(request)

    try:
        model_loader = ModelLoader()
        models_dict = model_loader.get_available_models()

        # Convert to Ollama format
        ollama_models = []
        for model_name, model_obj in models_dict.items():
            ollama_models.append(
                {
                    "name": model_name,
                    "model": model_name,
                    "modified_at": datetime.utcnow().isoformat() + "Z",
                    "size": model_obj.details.size,
                    "digest": "",
                    "details": {
                        "parent_model": "",
                        "format": "gguf",
                        "family": model_obj.details.family,
                        "families": model_obj.details.families,
                        "parameter_size": model_obj.details.parameter_size,
                        "quantization_level": model_obj.details.quantization_level,
                    },
                    "capabilities": {
                        "supports": {
                            "tool_calls": True,
                            "streaming": True,
                            "structured_outputs": True,
                            "vision": model_obj.details.supports_vision if hasattr(model_obj.details, "supports_vision") else False,  # type: ignore
                        },
                        "type": "chat",
                        "object": "model_capabilities",
                        "limits": {
                            "max_prompt_tokens": model_obj.details.original_ctx,
                        },
                    },
                }
            )

        return {"models": ollama_models}
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/api/show")
async def show_model_info(body: OllamaShowRequest):
    """>
    Show information about a model.
    Ollama-compatible endpoint for /api/show
    Requires API key authentication via X-API-Key header or Authorization: Bearer <key>
    """

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
                "parent_model": model_obj.details.parent_model or "",
                "format": model_obj.details.format,
                "family": model_obj.details.family,
                "families": model_obj.details.families,
                "parameter_size": model_obj.details.parameter_size,
                "quantization_level": model_obj.details.quantization_level or "",
            },
            "model_info": {},
            "capabilities": ["completion"],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error showing model info: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/create")
async def create_model(
    body: OllamaCreateRequest,
    request: Request,
    user_id: str = Depends(verify_api_key_access),
):
    """
    Create a model.
    Ollama-compatible endpoint for /api/create
    Requires API key authentication via X-API-Key header or Authorization: Bearer <key>
    """

    # Not implemented for this system - would require model file management
    raise HTTPException(status_code=501, detail="Model creation not supported")


@router.post("/copy")
async def copy_model(
    body: OllamaCopyRequest,
    request: Request,
    user_id: str = Depends(verify_api_key_access),
):
    """
    Copy a model.
    Ollama-compatible endpoint for /api/copy
    Requires API key authentication via X-API-Key header or Authorization: Bearer <key>
    """

    # Not implemented - would require model file management
    raise HTTPException(status_code=501, detail="Model copy not supported")


@router.delete("/delete")
async def delete_model(
    body: OllamaDeleteRequest,
    request: Request,
    user_id: str = Depends(verify_api_key_access),
):
    """
    Delete a model.
    Ollama-compatible endpoint for /api/delete
    Requires API key authentication via X-API-Key header or Authorization: Bearer <key>
    """

    # Not implemented - would require model file management
    raise HTTPException(status_code=501, detail="Model deletion not supported")


@router.post("/pull")
async def pull_model(
    body: OllamaPullRequest,
    request: Request,
    user_id: str = Depends(verify_api_key_access),
):
    """
    Pull a model from the library.
    Ollama-compatible endpoint for /api/pull
    Requires API key authentication via X-API-Key header or Authorization: Bearer <key>
    """

    # Not implemented - would require model downloading
    raise HTTPException(status_code=501, detail="Model pull not supported")


@router.post("/push")
async def push_model(
    body: OllamaPushRequest,
    request: Request,
    user_id: str = Depends(verify_api_key_access),
):
    """
    Push a model to the library.
    Ollama-compatible endpoint for /api/push
    Requires API key authentication via X-API-Key header or Authorization: Bearer <key>
    """

    # Not implemented - would require model uploading
    raise HTTPException(status_code=501, detail="Model push not supported")


@router.post("/embed")
async def generate_embeddings(request: Request) -> None:
    """
    Generate embeddings from a model.
    Ollama-compatible endpoint for /api/embed
    """
    verify_local_network_access(request)
    raise HTTPException(status_code=501, detail="Embeddings not yet implemented")


@router.get("/ps")
async def list_running_models(request: Request):
    """
    List currently loaded models.
    Ollama-compatible endpoint for /api/ps
    """
    verify_local_network_access(request)

    # Return empty list - we don't track loaded models currently
    return {"models": []}


@router.get("/api/version")
async def get_version() -> Dict[str, str]:
    """
    Get Ollama version.
    Ollama-compatible endpoint for /api/version
    """
    return {"version": "0.14.2"}
