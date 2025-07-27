"""
vLLM server implementation for high-performance model inference.
This module provides a standalone vLLM server that can be used to serve models
for the inference system.
"""

import datetime
import json
import os
import asyncio
import logging
from typing import Dict, List, Optional, Union, AsyncGenerator

import torch
import uvicorn
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create the FastAPI app
app = FastAPI(title="vLLM Model Server")

# Global engine instance
llm_engine = None


class ModelRequest(BaseModel):
    """Model request parameters."""
    model: str = Field(..., description="The model ID to use")
    prompt: str = Field(..., description="The prompt to generate from")
    max_tokens: int = Field(1024, description="Maximum number of tokens to generate")
    temperature: float = Field(0.7, description="Sampling temperature")
    top_p: float = Field(0.95, description="Top-p sampling parameter")
    top_k: int = Field(40, description="Top-k sampling parameter")
    presence_penalty: float = Field(0.0, description="Presence penalty parameter")
    frequency_penalty: float = Field(0.0, description="Frequency penalty parameter")
    stop: Optional[List[str]] = Field(None, description="Stop sequences")
    stream: bool = Field(False, description="Whether to stream the response")


class ModelResponse(BaseModel):
    """Model response structure."""
    id: str = Field(..., description="The request ID")
    model: str = Field(..., description="The model ID used")
    choices: List[Dict] = Field(..., description="The generated outputs")
    usage: Dict = Field(..., description="Token usage information")


@app.on_event("startup")
async def startup_event():
    """Initialize the vLLM engine on startup."""
    global llm_engine

    # Get model configuration from environment or use default
    model_id = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-30B-A3B")
    tensor_parallel_size = int(os.environ.get("VLLM_TP_SIZE", "1"))
    gpu_memory_utilization = float(os.environ.get("VLLM_GPU_UTIL", "0.85"))
    quantization = os.environ.get("VLLM_QUANT", "awq")
    max_model_len = int(os.environ.get("VLLM_MAX_LEN", "8192"))

    # Check for GGUF model
    if model_id.endswith(".gguf"):
        logger.info(f"Loading GGUF model: {model_id}")
        quantization = "gguf"

    # Log GPU information
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        logger.info(f"GPU {i} Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")

    logger.info(f"Initializing vLLM engine with model: {model_id}")
    logger.info(f"Tensor parallel size: {tensor_parallel_size}")
    logger.info(f"Quantization: {quantization}")

    # Create engine arguments
    engine_args = AsyncEngineArgs(
        model=model_id,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        quantization=quantization,  # type: ignore
        max_model_len=max_model_len,
    )

    # Initialize the engine
    llm_engine = AsyncLLMEngine.from_engine_args(engine_args)
    logger.info("vLLM engine initialized successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    global llm_engine
    if llm_engine:
        # No explicit cleanup needed for vLLM engine
        llm_engine = None
    logger.info("vLLM engine shut down")


@app.post("/v1/completions")
async def create_completion(request: ModelRequest):
    """Generate completions for the given prompt."""
    global llm_engine

    if llm_engine is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Engine not initialized"}
        )

    request_id = random_uuid()
    sampling_params = SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        max_tokens=request.max_tokens,
        presence_penalty=request.presence_penalty,
        frequency_penalty=request.frequency_penalty,
        stop=request.stop,
    )

    async def stream_response():
        if llm_engine is None:
            yield "data: [ERROR] Engine not initialized\n\n"
            return
        async for output in llm_engine.generate(prompt=request.prompt, sampling_params=sampling_params, request_id=request_id):
            choice = output.outputs[0]
            chunk = {
                "id": request_id,
                "object": "text_completion.chunk",
                "created": datetime.datetime.now(tz=datetime.timezone.utc).timestamp(),
                "model": request.model,
                "choices": [{
                    "text": choice.text,
                    "index": 0,
                    "finish_reason": choice.finish_reason,
                }],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(stream_response(), media_type="text/event-stream")


@app.post("/v1/chat/completions")
async def create_chat_completion(request: Request):
    """Generate chat completions using the vLLM engine."""
    global llm_engine

    if llm_engine is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Engine not initialized"}
        )

    # Parse the request body
    body = await request.json()
    model_id = body.get("model", "")
    messages = body.get("messages", [])
    stream = body.get("stream", False)

    # Extract sampling parameters
    sampling_params = SamplingParams(
        temperature=body.get("temperature", 0.7),
        top_p=body.get("top_p", 0.95),
        max_tokens=body.get("max_tokens", 1024),
        presence_penalty=body.get("presence_penalty", 0.0),
        frequency_penalty=body.get("frequency_penalty", 0.0),
        stop=body.get("stop", None),
    )

    # Convert chat messages to prompt
    prompt = ""
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            prompt += f"System: {content}\n\n"
        elif role == "user":
            prompt += f"User: {content}\n\n"
        elif role == "assistant":
            prompt += f"Assistant: {content}\n\n"

    prompt += "Assistant: "
    request_id = random_uuid()

    async def stream_chat_response():
        if llm_engine is None:
            yield "data: [ERROR] Engine not initialized\n\n"
            return
        async for output in llm_engine.generate(prompt, sampling_params, request_id=request_id):
            choice = output.outputs[0]
            chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": datetime.datetime.now(tz=datetime.timezone.utc).timestamp(),
                "model": model_id,
                "choices": [{
                    "delta": {"content": choice.text},
                    "index": 0,
                    "finish_reason": choice.finish_reason,
                }],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(stream_chat_response(), media_type="text/event-stream")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if llm_engine is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "message": "Engine not initialized"}
        )

    # Return basic health information
    return {
        "status": "healthy",
        "models_loaded": True,
        "gpu_info": {
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
        }
    }


if __name__ == "__main__":
    # Run the server
    port = int(os.environ.get("VLLM_PORT", "8000"))
    uvicorn.run("vllm_server:app", host="0.0.0.0", port=port, log_level="info")
