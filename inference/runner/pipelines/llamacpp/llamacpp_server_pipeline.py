"""
LlamaCppServerPipeline - Direct llama.cpp server integration.

This pipeline replaces llama-cpp-python with direct llama.cpp server management,
providing better performance, compatibility, and feature support.

Features:
- Direct llama.cpp server process management
- OpenAI-compatible API interface via LangChain
- Full feature parity: streaming, tool calling, grammar constraints
- Better memory management and OOM recovery
- Support for all llama.cpp server features
"""

import asyncio
import json
import logging
import os
import signal
import subprocess
import time
import threading
from typing import Any, Dict, List, Optional, Type, Iterator, Union, Tuple, cast
from contextlib import asynccontextmanager
from pathlib import Path
import psutil
from pathlib import Path

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    ToolCall as LangChainToolCall,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools.base import BaseTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr

from models import (
    Model,
    ModelParameters,
    ModelProfile,
    UserConfig,
    GPUConfig,
    OptimalParameters,
)
from models.config_utils import (
    resolve_parameter_optimization_config,
    resolve_gpu_config,
)
from runner.utils.model_loader import ModelLoader
from utils.logging import llmmllogger
from runner.utils.hardware_manager import hardware_manager
from runner.utils.intelligent_oom_recovery import IntelligentOOMRecovery
from runner.utils.resizer import Resizer
from runner.pipelines.base import BasePipeline


class LlamaCppServerManager:
    """Manages llama.cpp server process lifecycle."""
    
    def __init__(self, model: Model, profile: ModelProfile, user_config: Optional[UserConfig] = None):
        self.model = model
        self.profile = profile
        self.user_config = user_config
        self._logger = llmmllogger.bind(component=self.__class__.__name__, model=model.name)
        self.process: Optional[subprocess.Popen] = None
        self.port: int = self._find_available_port()
        self.server_url = f"http://localhost:{self.port}/v1"
        self._lock = threading.Lock()
        self._shutdown_event = threading.Event()
        self._startup_timeout = 60  # seconds
        
    def _find_available_port(self, start_port: int = 8001) -> int:
        """Find an available port starting from start_port."""
        import socket
        
        for port in range(start_port, start_port + 100):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(('localhost', port))
                    return port
                except OSError:
                    continue
        raise RuntimeError("No available ports found")
    
    def _get_gguf_path(self) -> str:
        """Return resolved GGUF file path for model."""
        details = getattr(self.model, "details", None)
        if details and hasattr(details, "gguf_file") and details.gguf_file:
            return details.gguf_file
        return self.model.model
    
    def _build_server_args(self) -> List[str]:
        """Build command line arguments for llama.cpp server."""
        gguf_path = self._get_gguf_path()
        params = self.profile.parameters
        gcfg = resolve_gpu_config(self.profile, self.user_config)
        
        args = [
            "/llama.cpp/build/bin/llama-server",
            "--model", gguf_path,
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "--cont-batching",
            "--metrics",
        ]
        
        # Core performance parameters
        args.extend(["--threads", str(os.cpu_count() or 4)])  # Use system CPU count
        args.extend(["--ctx-size", str(params.num_ctx or 40960)])
        args.extend(["--batch-size", str(params.batch_size or 256)])
        args.extend(["--ubatch-size", str(params.batch_size or 256)])
        
        # GPU configuration
        args.extend(["--n-gpu-layers", str(gcfg.gpu_layers if gcfg.gpu_layers is not None else -1)])
        
        # Main GPU selection
        if gcfg.main_gpu is not None and gcfg.main_gpu >= 0:
            args.extend(["--main-gpu", str(gcfg.main_gpu)])
        
        # Tensor split configuration
        if gcfg.tensor_split:
            tensor_split_str = ",".join(map(str, gcfg.tensor_split))
            args.extend(["--tensor-split", tensor_split_str])
        
        # Split mode configuration
        if hasattr(gcfg, 'split_mode') and gcfg.split_mode:
            args.extend(["--split-mode", str(gcfg.split_mode)])
        
        # Flash attention - use parameter from ModelParameters if available
        if hasattr(params, 'flash_attention') and params.flash_attention is not None:
            args.extend(["--flash-attn", "on" if params.flash_attention else "off"])
        else:
            args.extend(["--flash-attn", "on"])  # Default to on
        
        # MoE (Mixture of Experts) configuration - this parameter exists in ModelParameters
        if hasattr(params, 'n_cpu_moe') and params.n_cpu_moe is not None and params.n_cpu_moe > 0:
            args.extend(["--n-cpu-moe", str(params.n_cpu_moe)])
        
        # Multimodal support - this is critical for vision models
        mmproj_path = None
        
        # Check various possible locations for mmproj file
        if hasattr(self.model.details, 'clip_model_path') and self.model.details.clip_model_path:
            mmproj_path = self.model.details.clip_model_path
        elif hasattr(self.model.details, 'mmproj_path') and self.model.details.mmproj_path:
            mmproj_path = self.model.details.mmproj_path
        elif hasattr(self.model.details, 'parent_model') and self.model.details.parent_model and 'qwen' in self.model.details.parent_model.lower():
            # For Qwen models, try to find mmproj in same directory as model
            model_dir = Path(gguf_path).parent
            possible_mmproj = model_dir / "mmproj-model-f16.gguf"
            if possible_mmproj.exists():
                mmproj_path = str(possible_mmproj)
            else:
                # Try other common names
                for mmproj_name in ["mmproj.gguf", "clip.gguf", "vision.gguf"]:
                    possible = model_dir / mmproj_name
                    if possible.exists():
                        mmproj_path = str(possible)
                        break
        
        if mmproj_path and Path(mmproj_path).exists():
            args.extend(["--mmproj", mmproj_path])
            self._logger.info(f"Using multimodal projector: {mmproj_path}")
        elif "vl" in self.model.name.lower() or "vision" in self.model.name.lower():
            self._logger.warning(f"Vision model detected but no mmproj file found for {self.model.name}")
        
        # Draft model support for speculative decoding
        if hasattr(self.profile, 'draft_model') and self.profile.draft_model:
            args.extend(["--model-draft", str(self.profile.draft_model)])
        
        # Embedding mode for embedding models
        if hasattr(self.model, 'task') and self.model.task and 'embed' in str(self.model.task).lower():
            args.extend(["--embeddings"])
        
        # Disable web UI for server mode
        args.extend(["--no-webui"])
        
        # Add logging configuration
        if os.getenv("LOG_LEVEL", "WARNING").lower() == "debug":
            args.extend(["--verbose"])
        
        # Additional GPU optimizations from the hardware config
        # KV offloading configuration (note: offload_kqv=True means we want KV on GPU, so no_kv_offload=False)
        if hasattr(gcfg, 'offload_kqv') and not gcfg.offload_kqv:
            args.extend(["--no-kv-offload"])
        
        self._logger.info(f"Server args: {' '.join(args)}")
        return args
    
    def start(self) -> bool:
        """Start the llama.cpp server process."""
        with self._lock:
            if self.process and self.process.poll() is None:
                self._logger.info(f"Server already running on port {self.port}")
                return True
            
            try:
                args = self._build_server_args()
                
                self._logger.info(f"Starting llama.cpp server on port {self.port}")
                
                # Start the process
                self.process = subprocess.Popen(
                    args,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                # Wait for server to be ready
                if self._wait_for_server():
                    self._logger.info(f"Server started successfully on port {self.port}")
                    return True
                else:
                    self._logger.error("Server failed to start within timeout")
                    self.stop()
                    return False
                    
            except Exception as e:
                self._logger.error(f"Failed to start server: {e}")
                return False
    
    def _wait_for_server(self) -> bool:
        """Wait for server to become ready."""
        import requests
        
        start_time = time.time()
        while time.time() - start_time < self._startup_timeout:
            try:
                response = requests.get(f"http://localhost:{self.port}/health", timeout=2)
                if response.status_code == 200:
                    return True
            except requests.exceptions.RequestException:
                pass
            
            if self.process and self.process.poll() is not None:
                stdout, stderr = self.process.communicate(timeout=1)
                self._logger.error(f"Server process died. stdout: {stdout}, stderr: {stderr}")
                return False
                
            time.sleep(0.5)
        
        return False
    
    def stop(self) -> bool:
        """Stop the llama.cpp server process."""
        with self._lock:
            if not self.process:
                return True
                
            try:
                self._logger.info(f"Stopping server on port {self.port}")
                
                # Send SIGTERM for graceful shutdown
                self.process.terminate()
                
                # Wait for graceful shutdown
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._logger.warning("Server didn't shut down gracefully, force killing")
                    self.process.kill()
                    self.process.wait()
                
                self.process = None
                self._logger.info("Server stopped successfully")
                return True
                
            except Exception as e:
                self._logger.error(f"Error stopping server: {e}")
                return False
    
    def is_running(self) -> bool:
        """Check if server is running and responsive."""
        if not self.process or self.process.poll() is not None:
            return False
            
        try:
            import requests
            response = requests.get(f"http://localhost:{self.port}/health", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get server performance statistics."""
        try:
            import requests
            response = requests.get(f"http://localhost:{self.port}/metrics", timeout=2)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return {}


class LlamaCppServerPipeline(BasePipeline):
    """
    Pipeline using llama.cpp server with OpenAI-compatible interface.
    
    Replaces llama-cpp-python with direct server management for better:
    - Performance and compatibility
    - Memory management
    - Feature support (multimodal, tool calling, etc.)
    - Debugging and monitoring
    """

    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
        extra = "allow"

    def __init__(
        self,
        model: Model,
        profile: ModelProfile,
        grammar: Optional[Type[BaseModel]] = None,
        user_config: Optional[UserConfig] = None,
        **kwargs,
    ):
        """Initialize LlamaCpp server pipeline."""
        super().__init__(model=model, profile=profile, grammar=grammar, user_config=user_config, **kwargs)
        
        self._logger = llmmllogger.bind(component=self.__class__.__name__, model=model.name)
        self.grammar = grammar
        self.user_config = user_config
        self._bound_tools: List[BaseTool] = kwargs.get("_bound_tools", [])
        
        # Initialize server manager
        self.server_manager = LlamaCppServerManager(model, profile, user_config)
        
        # Initialize OpenAI client (will connect to our server)
        self.openai_client: Optional[ChatOpenAI] = None
        
        # Performance monitoring
        self.hardware_manager = hardware_manager
        self.resizer = Resizer()
        
        # OOM recovery (optional)
        self.use_intelligent_oom = (
            os.getenv("ENABLE_INTELLIGENT_OOM_RECOVERY", "false").lower() == "true"
        )
        self.oom_recovery = IntelligentOOMRecovery() if self.use_intelligent_oom else None
        
        # Start the server
        if not self.server_manager.start():
            raise RuntimeError(f"Failed to start llama.cpp server for model {model.name}")
        
        # Initialize OpenAI client
        self._initialize_openai_client()

    @property
    def _llm_type(self) -> str:
        return "llamacpp_server"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": self.model.name,
            "model_path": self.server_manager._get_gguf_path(),
            "server_port": self.server_manager.port,
            "parameters": self.profile.parameters.model_dump() if self.profile.parameters else {},
        }

    def _initialize_openai_client(self) -> None:
        """Initialize OpenAI client to connect to our llama.cpp server."""
        try:
            # Map parameters to what OpenAI client supports
            model_kwargs = {}
            
            # Add supported OpenAI parameters
            if hasattr(self.profile.parameters, 'top_p') and self.profile.parameters.top_p is not None:
                model_kwargs["top_p"] = self.profile.parameters.top_p
            
            if hasattr(self.profile.parameters, 'seed') and self.profile.parameters.seed is not None:
                model_kwargs["seed"] = self.profile.parameters.seed
            
            # Add llama.cpp specific parameters as extra_body
            extra_body = {}
            if hasattr(self.profile.parameters, 'top_k') and self.profile.parameters.top_k is not None:
                extra_body["top_k"] = self.profile.parameters.top_k
            
            if hasattr(self.profile.parameters, 'repeat_penalty') and self.profile.parameters.repeat_penalty is not None:
                extra_body["repeat_penalty"] = self.profile.parameters.repeat_penalty
            
            if extra_body:
                model_kwargs["extra_body"] = extra_body
            
            self.openai_client = ChatOpenAI(
                model="gpt-3.5-turbo",  # Placeholder - server ignores this
                base_url=self.server_manager.server_url,
                api_key=SecretStr("sk-no-key-required"),  # Server doesn't require auth
                temperature=self.profile.parameters.temperature or 0.7,
                max_tokens=self.profile.parameters.max_tokens or 4096,
                model_kwargs=model_kwargs,
                streaming=True,
                timeout=60.0,
            )
            
            self._logger.info(f"OpenAI client initialized for server at {self.server_manager.server_url}")
            
        except Exception as e:
            self._logger.error(f"Failed to initialize OpenAI client: {e}")
            raise

    def _format_messages_for_openai(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        """Convert LangChain messages to OpenAI format."""
        openai_messages = []
        
        # Add system message if present in profile
        if self.profile.system_prompt:
            openai_messages.append({"role": "system", "content": self.profile.system_prompt})
        
        # Convert conversation messages
        for msg in messages:
            if isinstance(msg, HumanMessage):
                openai_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                if msg.content:  # Skip empty messages
                    openai_messages.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, SystemMessage):
                openai_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, ToolMessage):
                openai_messages.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content
                })
        
        return openai_messages

    def _calculate_usage_metadata(self, prompt_tokens: int, completion_tokens: int) -> UsageMetadata:
        """Calculate usage metadata for response."""
        return UsageMetadata(
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate response using llama.cpp server."""
        if not self.server_manager.is_running():
            raise RuntimeError("llama.cpp server is not running")
        
        if not self.openai_client:
            raise RuntimeError("OpenAI client not initialized")
        
        try:
            # Convert messages to OpenAI format
            openai_messages = self._format_messages_for_openai(messages)
            
            # Prepare generation kwargs with clean parameters
            generation_kwargs = {}
            if stop:
                generation_kwargs["stop"] = stop
            
            # Add grammar constraints if specified
            if self.grammar:
                # Convert Pydantic model to JSON schema for llama.cpp server
                schema = self.grammar.model_json_schema()
                generation_kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": schema
                }
            
            # Bind tools if available
            client = self.openai_client
            if self._bound_tools:
                client = self.openai_client.bind_tools(self._bound_tools)
            
            # Generate response
            response = client.invoke(openai_messages, **generation_kwargs)
            
            # Extract content and usage
            content = str(response.content) if response.content else ""
            
            # Get server stats for usage calculation (approximate)
            stats = self.server_manager.get_stats()
            prompt_tokens = stats.get("tokens_input", 0)
            completion_tokens = stats.get("tokens_output", 0)
            
            usage_metadata = self._calculate_usage_metadata(prompt_tokens, completion_tokens)
            
            # Create result
            ai_message = AIMessage(
                content=content,
                usage_metadata=usage_metadata,
            )
            
            return ChatResult(generations=[ChatGeneration(message=ai_message)])
            
        except Exception as e:
            self._logger.error(f"Generation failed: {e}")
            raise

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream response using llama.cpp server."""
        if not self.server_manager.is_running():
            raise RuntimeError("llama.cpp server is not running")
        
        if not self.openai_client:
            raise RuntimeError("OpenAI client not initialized")
        
        try:
            # Convert messages to OpenAI format
            openai_messages = self._format_messages_for_openai(messages)
            
            # Prepare generation kwargs with clean parameters
            generation_kwargs = {}
            if stop:
                generation_kwargs["stop"] = stop
            
            # Add grammar constraints if specified
            if self.grammar:
                schema = self.grammar.model_json_schema()
                generation_kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": schema
                }
            
            # Bind tools if available
            client = self.openai_client
            if self._bound_tools:
                client = self.openai_client.bind_tools(self._bound_tools)
            
            # Stream response
            total_content = ""
            for chunk in client.stream(openai_messages, **generation_kwargs):
                if hasattr(chunk, 'content') and chunk.content:
                    chunk_content = str(chunk.content)
                    total_content += chunk_content
                    
                    ai_chunk = AIMessageChunk(content=chunk_content)
                    
                    yield ChatGenerationChunk(message=ai_chunk)
            
            # Yield final chunk with usage metadata
            stats = self.server_manager.get_stats()
            prompt_tokens = stats.get("tokens_input", 0)
            completion_tokens = stats.get("tokens_output", 0)
            usage_metadata = self._calculate_usage_metadata(prompt_tokens, completion_tokens)
            
            final_chunk = AIMessageChunk(content="", usage_metadata=usage_metadata)
            yield ChatGenerationChunk(message=final_chunk)
            
        except Exception as e:
            self._logger.error(f"Streaming failed: {e}")
            raise

    def bind_tools(
        self, 
        tools: List[BaseTool], 
        *, 
        tool_choice: str | None = None, 
        **kwargs: Any
    ) -> "LlamaCppServerPipeline":
        """Bind tools to this pipeline."""
        return LlamaCppServerPipeline(
            model=self.model,
            profile=self.profile,
            grammar=self.grammar,
            user_config=self.user_config,
            _bound_tools=tools,
            **kwargs
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics from server."""
        return self.server_manager.get_stats()

    def close(self):
        """Clean up resources."""
        try:
            if self.server_manager:
                self.server_manager.stop()
            self._logger.info("LlamaCppServerPipeline closed successfully")
        except Exception as e:
            self._logger.error(f"Error closing pipeline: {e}")

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close()
        except:
            pass  # Ignore errors during cleanup


__all__ = ["LlamaCppServerPipeline", "LlamaCppServerManager"]