#!/usr/bin/env python3
"""
OpenAI Compatible Router for existing FastAPI application
Integrates vLLM and LangChain with existing chat infrastructure
Enhanced with additional endpoints and model management
"""

import asyncio
import json
import time
import uuid
import base64
import io
from typing import Dict, List, Optional, AsyncGenerator, Union, Any
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.schema.output_parser import BaseOutputParser

# Import your existing models and services
from models.message import Message
from models.message_role import MessageRole
from models.message_content import MessageContent
from models.message_content_type import MessageContentType
from ..services.model_service import model_service
from ..services.hardware_manager import hardware_manager
from ..config import logger


# OpenAI API compatible models
class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Content of the message")


class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="Model to use for completion")
    messages: List[ChatMessage] = Field(..., description="List of messages")
    temperature: Optional[float] = Field(0.7, ge=0, le=2)
    max_tokens: Optional[int] = Field(None, ge=1)
    top_p: Optional[float] = Field(1.0, ge=0, le=1)
    frequency_penalty: Optional[float] = Field(0.0, ge=-2, le=2)
    presence_penalty: Optional[float] = Field(0.0, ge=-2, le=2)
    stream: Optional[bool] = Field(False)
    stop: Optional[Union[str, List[str]]] = Field(None)


class CompletionRequest(BaseModel):
    model: str = Field(..., description="Model to use for completion")
    prompt: Union[str, List[str]] = Field(..., description="Prompt(s) to complete")
    temperature: Optional[float] = Field(0.7, ge=0, le=2)
    max_tokens: Optional[int] = Field(None, ge=1)
    top_p: Optional[float] = Field(1.0, ge=0, le=1)
    frequency_penalty: Optional[float] = Field(0.0, ge=-2, le=2)
    presence_penalty: Optional[float] = Field(0.0, ge=-2, le=2)
    stream: Optional[bool] = Field(False)
    stop: Optional[Union[str, List[str]]] = Field(None)


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]] = Field(..., description="Input text(s) to embed")
    model: str = Field(..., description="Model to use for embedding")
    encoding_format: Optional[str] = Field(
        "float", description="Format of the embedding"
    )
    dimensions: Optional[int] = Field(
        None, description="Number of dimensions in the embedding"
    )
    user: Optional[str] = Field(None, description="User identifier")


class AudioTranscriptionRequest(BaseModel):
    model: str = Field(..., description="Model to use for transcription")
    language: Optional[str] = Field(None, description="Language of the audio")
    prompt: Optional[str] = Field(None, description="Optional text to guide the model")
    response_format: Optional[str] = Field("json", description="Response format")
    temperature: Optional[float] = Field(0, ge=0, le=1)


class AudioTranslationRequest(BaseModel):
    model: str = Field(..., description="Model to use for translation")
    prompt: Optional[str] = Field(None, description="Optional text to guide the model")
    response_format: Optional[str] = Field("json", description="Response format")
    temperature: Optional[float] = Field(0, ge=0, le=1)


class ImageGenerationRequest(BaseModel):
    prompt: str = Field(..., description="Text description of the desired image")
    model: Optional[str] = Field(
        "dall-e-3", description="Model to use for image generation"
    )
    n: Optional[int] = Field(1, ge=1, le=10, description="Number of images to generate")
    quality: Optional[str] = Field("standard", description="Quality of the image")
    response_format: Optional[str] = Field("url", description="Response format")
    size: Optional[str] = Field("1024x1024", description="Size of the generated image")
    style: Optional[str] = Field("vivid", description="Style of the generated image")
    user: Optional[str] = Field(None, description="User identifier")


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


class CompletionChoice(BaseModel):
    index: int
    text: str
    finish_reason: Optional[str] = None


class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Usage


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: EmbeddingUsage


class AudioTranscriptionResponse(BaseModel):
    text: str


class AudioTranslationResponse(BaseModel):
    text: str


class ImageData(BaseModel):
    url: Optional[str] = None
    b64_json: Optional[str] = None
    revised_prompt: Optional[str] = None


class ImageGenerationResponse(BaseModel):
    created: int
    data: List[ImageData]


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "vllm"


class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


# LangChain Output Parser for structured responses
class OpenAIOutputParser(BaseOutputParser[str]):
    """Parse LLM output to string format."""

    def parse(self, text: str) -> str:
        return text.strip()


# Global variables for vLLM engine
vllm_engine: Optional[AsyncLLMEngine] = None
current_model_name: str = ""
available_models: List[Dict[str, Any]] = []

# Create router for OpenAI endpoints
router = APIRouter(prefix="/v1", tags=["openai-compatible"])


def load_models_from_json() -> List[Dict[str, Any]]:
    """Load models from the models.json file."""
    try:
        # Assuming models.json is in the same directory or accessible
        models_file = Path("models.json")
        if models_file.exists():
            with open(models_file, "r") as f:
                return json.load(f)
        else:
            logger.warning("models.json not found, using default models")
            return []
    except Exception as e:
        logger.error(f"Error loading models from JSON: {e}")
        return []


# Load models at startup
available_models = load_models_from_json()


def get_model_by_id(model_id: str) -> Optional[Dict[str, Any]]:
    """Get model information by ID."""
    for model in available_models:
        if model.get("id") == model_id or model.get("name") == model_id:
            return model
    return None


def get_models_by_task(task: str) -> List[Dict[str, Any]]:
    """Get models that support a specific task."""
    return [model for model in available_models if model.get("task") == task]


class VLLMService:
    """Service class to manage vLLM engine lifecycle."""

    def __init__(self):
        self.engine: Optional[AsyncLLMEngine] = None
        self.model_name: str = ""
        self.model_info: Optional[Dict[str, Any]] = None
        self.initialized = False

    async def initialize_engine(self, model_id: str):
        """Initialize the vLLM engine with the specified model."""
        model_info = get_model_by_id(model_id)
        if not model_info:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

        model_path = model_info.get("model", model_id)

        if self.engine and self.model_name == model_path:
            return  # Already initialized with this model

        logger.info(f"Initializing vLLM engine with model: {model_path}")

        try:
            # Clear existing engine if any
            if self.engine:
                del self.engine
                hardware_manager.clear_memory()

            # Create new engine args based on model details
            details = model_info.get("details", {})

            engine_args = AsyncEngineArgs(
                model=model_path,
                tensor_parallel_size=1,
                dtype="auto",
                trust_remote_code=True,
                max_model_len=2048,  # Adjust based on your needs
                gpu_memory_utilization=0.8,
            )

            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            self.model_name = model_path
            self.model_info = model_info
            self.initialized = True

            logger.info(
                f"vLLM engine initialized successfully with model: {model_path}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize vLLM engine: {e}")
            self.engine = None
            self.model_name = ""
            self.model_info = None
            self.initialized = False
            raise HTTPException(
                status_code=500, detail=f"Failed to initialize model: {e}"
            )

    async def get_engine(self) -> AsyncLLMEngine:
        """Get the current engine, initializing if necessary."""
        if not self.initialized or not self.engine:
            # Use first available text-to-text model as default
            text_models = get_models_by_task("TextToText")
            if text_models:
                await self.initialize_engine(text_models[0]["id"])
            else:
                raise HTTPException(
                    status_code=503, detail="No text generation models available"
                )

        if not self.engine:
            raise HTTPException(status_code=503, detail="vLLM engine not available")

        return self.engine

    def is_ready(self) -> bool:
        """Check if the engine is ready."""
        return self.initialized and self.engine is not None


# Global vLLM service instance
vllm_service = VLLMService()


def convert_openai_to_internal_messages(
    openai_messages: List[ChatMessage],
) -> List[Message]:
    """Convert OpenAI format messages to internal Message format."""
    messages = []

    for i, msg in enumerate(openai_messages):
        # Map OpenAI roles to internal MessageRole enum
        role_mapping = {
            "system": MessageRole.SYSTEM,
            "user": MessageRole.USER,
            "assistant": MessageRole.ASSISTANT,
            "tool": MessageRole.TOOL,
        }

        role = role_mapping.get(msg.role.lower(), MessageRole.USER)

        # Create message content
        content = [MessageContent(type=MessageContentType.TEXT, text=msg.content)]

        # Create internal message
        internal_msg = Message(
            id=i,
            role=role,
            content=content,
            conversation_id=0,  # Default conversation ID
            created_at=None,  # Will be set automatically
        )

        messages.append(internal_msg)

    return messages


def convert_internal_to_openai_message(internal_msg: Message) -> ChatMessage:
    """Convert internal Message to OpenAI format."""
    # Extract text content
    text_content = ""
    for content in internal_msg.content:
        if content.type == MessageContentType.TEXT and content.text:
            text_content += content.text

    return ChatMessage(role=internal_msg.role.value, content=text_content)


def create_sampling_params(
    request: Union[ChatCompletionRequest, CompletionRequest],
) -> SamplingParams:
    """Create vLLM SamplingParams from request."""
    stop_sequences = []
    if request.stop:
        if isinstance(request.stop, str):
            stop_sequences = [request.stop]
        else:
            stop_sequences = request.stop

    return SamplingParams(
        temperature=request.temperature if request.temperature is not None else 0.7,
        max_tokens=request.max_tokens or 512,
        top_p=request.top_p if request.top_p is not None else 1.0,
        frequency_penalty=(
            request.frequency_penalty if request.frequency_penalty is not None else 0.0
        ),
        presence_penalty=(
            request.presence_penalty if request.presence_penalty is not None else 0.0
        ),
        stop=stop_sequences if stop_sequences else None,
    )


def messages_to_prompt(messages: List[ChatMessage]) -> str:
    """Convert OpenAI chat messages to a single prompt string."""
    prompt_parts = []

    for message in messages:
        role = message.role.lower()
        content = message.content

        if role == "system":
            prompt_parts.append(f"System: {content}")
        elif role == "user":
            prompt_parts.append(f"Human: {content}")
        elif role == "assistant":
            prompt_parts.append(f"Assistant: {content}")
        else:
            prompt_parts.append(f"{role.title()}: {content}")

    # Add assistant prefix for completion
    prompt_parts.append("Assistant:")
    return "\n".join(prompt_parts)


def estimate_tokens(text: str) -> int:
    """Simple token estimation (rough approximation)."""
    return round(len(text.split()) * 1.3)  # Rough estimate


@router.get("/models")
async def list_models() -> ModelsResponse:
    """List available models."""
    model_infos = []

    for model in available_models:
        model_infos.append(
            ModelInfo(
                id=model["id"],
                created=int(
                    time.mktime(
                        time.strptime(
                            model.get("modified_at", "2025-01-01"), "%Y-%m-%d"
                        )
                    )
                    if model.get("modified_at")
                    else time.time()
                ),
                owned_by="local",
            )
        )

    return ModelsResponse(data=model_infos)


@router.get("/models/{model}")
async def get_model(model: str) -> ModelInfo:
    """Get information about a specific model."""
    model_info = get_model_by_id(model)
    if not model_info:
        raise HTTPException(status_code=404, detail=f"Model {model} not found")

    return ModelInfo(
        id=model_info["id"],
        created=int(
            time.mktime(
                time.strptime(model_info.get("modified_at", "2025-01-01"), "%Y-%m-%d")
            )
            if model_info.get("modified_at")
            else time.time()
        ),
        owned_by="local",
    )


@router.post("/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
) -> Union[ChatCompletionResponse, StreamingResponse]:
    """Create a chat completion, compatible with OpenAI's API."""
    try:
        # Validate model supports text generation
        model_info = get_model_by_id(request.model)
        if not model_info or model_info.get("task") not in [
            "TextToText",
            "VisionTextToText",
        ]:
            raise HTTPException(
                status_code=400,
                detail=f"Model {request.model} does not support text generation",
            )

        # Initialize engine if needed
        if (
            not vllm_service.is_ready()
            or vllm_service.model_info.get("id") != request.model
        ):
            await vllm_service.initialize_engine(request.model)

        engine = await vllm_service.get_engine()

        # Convert messages to prompt
        prompt = messages_to_prompt(request.messages)
        sampling_params = create_sampling_params(request)

        if request.stream:
            return StreamingResponse(
                stream_chat_completion(prompt, sampling_params, request),
                media_type="text/plain",
            )

        # Generate completion
        request_id = str(uuid.uuid4())
        results = []

        async for output in engine.generate(prompt, sampling_params, request_id):
            results.append(output)

        if not results:
            raise HTTPException(status_code=500, detail="No output generated")

        final_output = results[-1]
        generated_text = final_output.outputs[0].text

        # Calculate token usage (approximation)
        prompt_tokens = int(estimate_tokens(prompt))
        completion_tokens = int(estimate_tokens(generated_text))

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=generated_text),
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    except Exception as e:
        logger.error(f"Error in chat completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def stream_chat_completion(
    prompt: str, sampling_params: SamplingParams, request: ChatCompletionRequest
) -> AsyncGenerator[str, None]:
    """Stream chat completion responses."""
    try:
        engine = await vllm_service.get_engine()
        request_id = str(uuid.uuid4())

        async for output in engine.generate(prompt, sampling_params, request_id):
            if output.outputs:
                chunk_data = {
                    "id": f"chatcmpl-{uuid.uuid4().hex}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": output.outputs[0].text},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"

        # Send final chunk
        final_chunk = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"Error in streaming chat completion: {e}")
        error_chunk = {"error": {"message": str(e), "type": "server_error"}}
        yield f"data: {json.dumps(error_chunk)}\n\n"


@router.post("/completions")
async def create_completion(
    request: CompletionRequest,
) -> Union[CompletionResponse, StreamingResponse]:
    """Create a text completion, compatible with OpenAI's API."""
    try:
        # Validate model supports text generation
        model_info = get_model_by_id(request.model)
        if not model_info or model_info.get("task") not in [
            "TextToText",
            "VisionTextToText",
        ]:
            raise HTTPException(
                status_code=400,
                detail=f"Model {request.model} does not support text generation",
            )

        # Initialize engine if needed
        if (
            not vllm_service.is_ready()
            or vllm_service.model_info.get("id") != request.model
        ):
            await vllm_service.initialize_engine(request.model)

        engine = await vllm_service.get_engine()

        prompt = (
            request.prompt if isinstance(request.prompt, str) else request.prompt[0]
        )
        sampling_params = create_sampling_params(request)

        if request.stream:
            return StreamingResponse(
                stream_completion(prompt, sampling_params, request),
                media_type="text/plain",
            )

        # Generate completion
        request_id = str(uuid.uuid4())
        results = []

        async for output in engine.generate(prompt, sampling_params, request_id):
            results.append(output)

        if not results:
            raise HTTPException(status_code=500, detail="No output generated")

        final_output = results[-1]
        generated_text = final_output.outputs[0].text

        # Calculate token usage (approximation)
        prompt_tokens = int(estimate_tokens(prompt))
        completion_tokens = int(estimate_tokens(generated_text))

        return CompletionResponse(
            id=f"cmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=request.model,
            choices=[
                CompletionChoice(index=0, text=generated_text, finish_reason="stop")
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    except Exception as e:
        logger.error(f"Error in completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def stream_completion(
    prompt: str, sampling_params: SamplingParams, request: CompletionRequest
) -> AsyncGenerator[str, None]:
    """Stream completion responses."""
    try:
        engine = await vllm_service.get_engine()
        request_id = str(uuid.uuid4())

        async for output in engine.generate(prompt, sampling_params, request_id):
            if output.outputs:
                chunk_data = {
                    "id": f"cmpl-{uuid.uuid4().hex}",
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "text": output.outputs[0].text,
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"

        # Send final chunk
        final_chunk = {
            "id": f"cmpl-{uuid.uuid4().hex}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{"index": 0, "text": "", "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"Error in streaming completion: {e}")
        error_chunk = {"error": {"message": str(e), "type": "server_error"}}
        yield f"data: {json.dumps(error_chunk)}\n\n"


@router.post("/embeddings")
async def create_embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
    """Create embeddings for the given input."""
    try:
        # Validate model supports embeddings
        model_info = get_model_by_id(request.model)
        if not model_info or model_info.get("task") != "Embedding":
            # Look for embedding models if specific model not found
            embedding_models = get_models_by_task("Embedding")
            if not embedding_models:
                raise HTTPException(
                    status_code=400, detail="No embedding models available"
                )
            model_info = embedding_models[0]  # Use first available embedding model

        # For this implementation, we'll use a placeholder
        # In a real implementation, you'd use the actual embedding model
        inputs = request.input if isinstance(request.input, list) else [request.input]

        embeddings_data = []
        total_tokens = 0

        for i, text in enumerate(inputs):
            # Placeholder embedding - in reality, you'd use the actual model
            # This creates a simple hash-based embedding for demonstration
            import hashlib

            text_hash = hashlib.md5(text.encode()).hexdigest()
            # Convert hex to normalized float values
            embedding = [
                float(int(text_hash[j : j + 2], 16)) / 255.0 - 0.5
                for j in range(0, len(text_hash), 2)
            ]

            # Pad or truncate to desired dimensions
            target_dim = request.dimensions or 768
            if len(embedding) < target_dim:
                embedding.extend([0.0] * (target_dim - len(embedding)))
            else:
                embedding = embedding[:target_dim]

            embeddings_data.append(EmbeddingData(embedding=embedding, index=i))

            total_tokens += estimate_tokens(text)

        return EmbeddingResponse(
            data=embeddings_data,
            model=model_info["id"],
            usage=EmbeddingUsage(prompt_tokens=total_tokens, total_tokens=total_tokens),
        )

    except Exception as e:
        logger.error(f"Error creating embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/audio/transcriptions")
async def create_transcription(
    file: UploadFile = File(...),
    model: str = "whisper-1",
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    response_format: str = "json",
    temperature: float = 0,
) -> AudioTranscriptionResponse:
    """Transcribe audio to text."""
    try:
        # Validate file type
        if not file.content_type.startswith("audio/"):
            raise HTTPException(status_code=400, detail="File must be an audio file")

        # Read audio file
        audio_data = await file.read()

        # Placeholder implementation - in reality, you'd use Whisper or similar
        # For demonstration, return a placeholder transcription
        transcription_text = f"[Placeholder transcription of {file.filename}]"

        if response_format == "text":
            return transcription_text
        else:
            return AudioTranscriptionResponse(text=transcription_text)

    except Exception as e:
        logger.error(f"Error in audio transcription: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/audio/translations")
async def create_translation(
    file: UploadFile = File(...),
    model: str = "whisper-1",
    prompt: Optional[str] = None,
    response_format: str = "json",
    temperature: float = 0,
) -> AudioTranslationResponse:
    """Translate audio to English text."""
    try:
        # Validate file type
        if not file.content_type.startswith("audio/"):
            raise HTTPException(status_code=400, detail="File must be an audio file")

        # Read audio file
        audio_data = await file.read()

        # Placeholder implementation - in reality, you'd use Whisper or similar
        # For demonstration, return a placeholder translation
        translation_text = f"[Placeholder English translation of {file.filename}]"

        if response_format == "text":
            return translation_text
        else:
            return AudioTranslationResponse(text=translation_text)

    except Exception as e:
        logger.error(f"Error in audio translation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/images/generations")
async def create_image(request: ImageGenerationRequest) -> ImageGenerationResponse:
    """Generate images from text prompts."""
    try:
        # Find appropriate image generation model
        image_models = get_models_by_task("TextToImage")
        if not image_models:
            raise HTTPException(
                status_code=400, detail="No image generation models available"
            )

        # Use the requested model or first available
        model_info = (
            get_model_by_id(request.model) if request.model else image_models[0]
        )
        if not model_info or model_info.get("task") != "TextToImage":
            model_info = image_models[0]

        # Placeholder implementation - in reality, you'd use Stable Diffusion, FLUX, etc.
        # For demonstration, return placeholder image data
        images_data = []

        for i in range(request.n):
            if request.response_format == "b64_json":
                # Create a simple placeholder base64 image (1x1 pixel PNG)
                placeholder_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
                images_data.append(
                    ImageData(b64_json=placeholder_b64, revised_prompt=request.prompt)
                )
            else:
                # Return placeholder URL
                images_data.append(
                    ImageData(
                        url=f"https://placeholder.example.com/image_{i}.png",
                        revised_prompt=request.prompt,
                    )
                )

        return ImageGenerationResponse(created=int(time.time()), data=images_data)

    except Exception as e:
        logger.error(f"Error in image generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint for vLLM service."""
    return {
        "status": "healthy" if vllm_service.is_ready() else "initializing",
        "model": vllm_service.model_name or "not loaded",
        "engine_ready": vllm_service.is_ready(),
        "available_models": len(available_models),
        "models_by_task": {
            "TextToText": len(get_models_by_task("TextToText")),
            "VisionTextToText": len(get_models_by_task("VisionTextToText")),
            "TextToImage": len(get_models_by_task("TextToImage")),
            "ImageToImage": len(get_models_by_task("ImageToImage")),
            "Embedding": len(get_models_by_task("Embedding")),
        },
    }


# Initialization function to be called from your main app
async def initialize_vllm_service(model_id: Optional[str] = None):
    """Initialize the vLLM service. Call this from your app startup."""
    try:
        # Load models from JSON
        global available_models
        available_models = load_models_from_json()

        # Initialize with specified model or first available text model
        if model_id:
            await vllm_service.initialize_engine(model_id)
        else:
            text_models = get_models_by_task("TextToText")
            if text_models:
                await vllm_service.initialize_engine(text_models[0]["id"])

        logger.info("vLLM service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize vLLM service: {e}")
        raise


# Cleanup function to be called from your app shutdown
async def cleanup_vllm_service():
    """Cleanup vLLM resources. Call this from your app shutdown."""
    try:
        if vllm_service.engine:
            del vllm_service.engine
            vllm_service.engine = None
            vllm_service.initialized = False
            vllm_service.model_info = None
            hardware_manager.clear_memory()
        logger.info("vLLM service cleaned up successfully")
    except Exception as e:
        logger.error(f"Error cleaning up vLLM service: {e}")


# Additional utility functions for model management
def reload_models():
    """Reload models from JSON file."""
    global available_models
    available_models = load_models_from_json()
    logger.info(f"Reloaded {len(available_models)} models from JSON")


def get_model_capabilities(model_id: str) -> Dict[str, Any]:
    """Get detailed capabilities of a specific model."""
    model_info = get_model_by_id(model_id)
    if not model_info:
        return {}

    details = model_info.get("details", {})
    return {
        "id": model_info.get("id"),
        "name": model_info.get("name"),
        "task": model_info.get("task"),
        "specialization": details.get("specialization"),
        "parameter_size": details.get("parameter_size"),
        "format": details.get("format"),
        "family": details.get("family"),
        "quantization_level": details.get("quantization_level"),
        "dtype": details.get("dtype"),
        "supports_streaming": model_info.get("task")
        in ["TextToText", "VisionTextToText"],
        "supports_embeddings": model_info.get("task") == "Embedding",
        "supports_image_generation": model_info.get("task")
        in ["TextToImage", "ImageToImage"],
        "supports_vision": model_info.get("task") == "VisionTextToText",
    }


@router.get("/models/{model}/capabilities")
async def get_model_capabilities_endpoint(model: str) -> Dict[str, Any]:
    """Get detailed capabilities of a specific model."""
    capabilities = get_model_capabilities(model)
    if not capabilities:
        raise HTTPException(status_code=404, detail=f"Model {model} not found")
    return capabilities


@router.post("/models/reload")
async def reload_models_endpoint():
    """Reload models from the models.json file."""
    try:
        reload_models()
        return {
            "status": "success",
            "message": f"Reloaded {len(available_models)} models",
            "models_count": len(available_models),
        }
    except Exception as e:
        logger.error(f"Error reloading models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/tasks/{task}")
async def get_models_by_task_endpoint(task: str) -> Dict[str, Any]:
    """Get all models that support a specific task."""
    valid_tasks = [
        "TextToText",
        "VisionTextToText",
        "TextToImage",
        "ImageToImage",
        "Embedding",
    ]
    if task not in valid_tasks:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task. Valid tasks are: {', '.join(valid_tasks)}",
        )

    models = get_models_by_task(task)
    return {
        "task": task,
        "models": [
            {
                "id": model["id"],
                "name": model["name"],
                "parameter_size": model.get("details", {}).get("parameter_size"),
                "specialization": model.get("details", {}).get("specialization"),
            }
            for model in models
        ],
        "count": len(models),
    }


# Export the router and service functions
__all__ = [
    "router",
    "initialize_vllm_service",
    "cleanup_vllm_service",
    "vllm_service",
    "reload_models",
    "get_model_capabilities",
    "get_model_by_id",
    "get_models_by_task",
]
