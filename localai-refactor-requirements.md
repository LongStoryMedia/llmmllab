# LocalAI Integration Requirements: Comprehensive Runner Module Refactoring

## Executive Summary

This requirements document outlines the comprehensive refactoring of the LLM ML Lab runner module to leverage LocalAI as the unified backend for all AI operations. LocalAI provides a complete OpenAI-compatible API ecosystem with automatic backend detection, simplifying the complex model management currently handled by the runner's specialized pipeline implementations.

**Prerequisites:** This refactor assumes all phases detailed in the `refactor-requirements.md` document have been completed, specifically:
- Migration of all LangGraph orchestration from runner to composer
- Simplified runner interface to pure LLM execution
- Established composer service as the authoritative workflow orchestrator

## Current State Analysis

### 1.1 Current Runner Architecture Complexity

The current runner module contains significant complexity that LocalAI can simplify:

**Pipeline Diversity:**
- **Text-to-Text:** Direct llama.cpp integration (`llamacpp/base_llamacpp.py`)
- **Embeddings:** Custom embedding pipelines (`emb/qwen3emb.py`, `emb/nom2.py`)
- **Image Generation:** Specialized image pipelines (`txt2img/flux.py`, `img2img/flux.py`)
- **Vision Models:** Multi-modal implementations (`imgtxt2txt/qwen25_vl.py`)

**Low-Level Infrastructure Management:**
- Custom GPU memory optimization
- Direct GGUF/GGML model file handling
- Manual backend selection and configuration
- Complex pipeline lifecycle management
- Grammar constraint implementation via llama-cpp-python

**Architectural Problems Addressed by LocalAI:**
- **Backend Management:** Manual CUDA/CPU/GPU detection and optimization
- **Model Format Complexity:** Direct handling of GGUF/GGML formats
- **Resource Management:** Custom GPU memory tracking and optimization
- **API Inconsistency:** Different interfaces for different model types

### 1.2 LocalAI Advantages

**Unified Interface:**
- Single OpenAI-compatible REST API for all operations
- Automatic backend detection (NVIDIA CUDA, AMD ROCm, Intel GPU, CPU)
- Built-in model gallery with automatic downloads
- Native support for constrained grammars (GBNF)

**Feature Coverage:**
- Text generation with llama.cpp backend
- Embeddings via sentence-transformers and llama.cpp
- Image generation via Stable Diffusion integration
- Audio processing (TTS/STT) via Whisper
- Vision capabilities (GPT-Vision compatible)
- Function calling and OpenAI tools support

**Operational Benefits:**
- Eliminates custom GPU detection and optimization code
- Reduces model management complexity
- Provides consistent streaming interfaces
- Enables distributed inference capabilities
- Built-in model caching and optimization

## 2. Target Architecture

### 2.1 LocalAI-Centric Runner Architecture

**Core Principle:** Replace complex pipeline implementations with LocalAI HTTP client interfaces, maintaining the simplified runner API established in the primary refactor while delegating all low-level model operations to LocalAI.

```plaintext
┌─────────────────────────────────────────────────────────────┐
│                    Composer Service                         │
│        (LangGraph Orchestration - Already Migrated)         │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                Simplified Runner API                        │
│          (Pure LLM Interface - Post Refactor)               │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   Text Client   │  │ Embedding Client│  │ Image Client │ │
│  │ (LocalAI HTTP)  │  │ (LocalAI HTTP)  │  │(LocalAI HTTP)│ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                   LocalAI Instance                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │    Automatic Backend Detection & Optimization        │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────┐ │   │
│  │  │llama.cpp    │ │Transformers │ │Stable Diffusion │ │   │
│  │  │(GGUF/GGML)  │ │(HuggingFace)│ │   (Images)      │ │   │
│  │  └─────────────┘ └─────────────┘ └─────────────────┘ │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Component Transformation Strategy

**Existing Complex Pipelines → LocalAI HTTP Clients:**

| Current Pipeline | LocalAI Equivalent | Transformation |
|------------------|-------------------|----------------|
| `llamacpp/base_llamacpp.py` | `/v1/chat/completions` | Replace 800+ lines with HTTP client |
| `emb/qwen3emb.py` | `/v1/embeddings` | Replace custom embedding logic |
| `txt2img/flux.py` | `/v1/images/generations` | Replace Stable Diffusion integration |
| `imgtxt2txt/qwen25_vl.py` | `/v1/chat/completions` (vision) | Replace multi-modal implementation |

**Architecture Benefits:**
- **Simplified Codebase:** Reduce runner module from 2000+ lines to ~300 lines
- **Unified Configuration:** Single LocalAI configuration replaces multiple backend configs
- **Improved Reliability:** LocalAI handles GPU optimization, memory management, and error recovery
- **Enhanced Features:** Access to LocalAI's built-in function calling, grammar constraints, and distributed inference

## 3. Implementation Phases

### Phase 1: LocalAI Infrastructure Setup

**3.1.1 LocalAI Deployment Configuration**

Create deployment configuration supporting the platform's Kubernetes architecture:

```yaml
# k8s/localai-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: localai
  namespace: ollama
spec:
  replicas: 1
  selector:
    matchLabels:
      app: localai
  template:
    metadata:
      labels:
        app: localai
    spec:
      containers:
      - name: localai
        image: localai/localai:latest-aio-gpu-nvidia-cuda-12
        ports:
        - containerPort: 8080
        env:
        - name: MODELS_PATH
          value: "/models"
        - name: DEBUG
          value: "true"
        - name: CONTEXT_SIZE
          value: "131072"
        - name: GPU_LAYERS
          value: "-1"  # Use all GPU layers
        resources:
          limits:
            nvidia.com/gpu: 1
        volumeMounts:
        - name: models-storage
          mountPath: /models
      volumes:
      - name: models-storage
        persistentVolumeClaim:
          claimName: localai-models-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: localai-service
  namespace: ollama
spec:
  selector:
    app: localai
  ports:
  - port: 8080
    targetPort: 8080
  type: ClusterIP
```

**3.1.2 Environment Configuration**

Add LocalAI configuration to the existing environment variable system:

```yaml
# schemas/localai_config.yaml
type: object
properties:
  service:
    type: object
    properties:
      host:
        type: string
        default: "localai-service"
      port:
        type: integer
        default: 8080
      protocol:
        type: string
        default: "http"
        enum: ["http", "https"]
      timeout:
        type: integer
        default: 300
      max_retries:
        type: integer
        default: 3
  models:
    type: object
    properties:
      text_generation:
        type: string
        default: "llama-3.2-1b-instruct:q4_k_m"
      embeddings:
        type: string
        default: "text-embedding-ada-002"
      image_generation:
        type: string
        default: "flux.1-dev-ggml"
      vision:
        type: string
        default: "minicpm-v-4_5"
  features:
    type: object
    properties:
      enable_function_calling:
        type: boolean
        default: true
      enable_grammar_constraints:
        type: boolean
        default: true
      enable_distributed_inference:
        type: boolean
        default: false
      max_context_length:
        type: integer
        default: 131072
```

### Phase 2: Core Runner Interface Migration

**3.2.1 LocalAI HTTP Client Implementation**

Create unified LocalAI client replacing all existing pipeline complexity:

```python
# runner/clients/localai_client.py
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional, Union, AsyncGenerator
from pydantic import BaseModel
from pathlib import Path

from models import ChatResponse, Message, MessageContent
from utils.grammar_generator import GrammarGenerator

class LocalAIClient:
    """Unified LocalAI client for all AI operations."""
    
    def __init__(self, base_url: str, timeout: int = 300):
        self.base_url = base_url.rstrip('/')
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        stream: bool = False,
        grammar: Optional[str] = None,
        **kwargs
    ) -> Union[ChatResponse, AsyncGenerator[str, None]]:
        """OpenAI-compatible chat completion with optional grammar constraints."""
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            **kwargs
        }
        
        if grammar:
            payload["grammar"] = grammar
        
        if stream:
            return self._stream_chat_completion(payload)
        else:
            return await self._complete_chat_completion(payload)
    
    async def _stream_chat_completion(self, payload: Dict) -> AsyncGenerator[str, None]:
        """Stream chat completion tokens."""
        async with self.session.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"}
        ) as response:
            async for line in response.content:
                if line:
                    line_str = line.decode('utf-8').strip()
                    if line_str.startswith('data: '):
                        data = line_str[6:]  # Remove 'data: ' prefix
                        if data != '[DONE]':
                            try:
                                import json
                                chunk = json.loads(data)
                                if 'choices' in chunk and len(chunk['choices']) > 0:
                                    delta = chunk['choices'][0].get('delta', {})
                                    content = delta.get('content', '')
                                    if content:
                                        yield content
                            except json.JSONDecodeError:
                                continue
    
    async def _complete_chat_completion(self, payload: Dict) -> ChatResponse:
        """Complete chat completion (non-streaming)."""
        async with self.session.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"}
        ) as response:
            result = await response.json()
            
            # Convert LocalAI response to our ChatResponse format
            content = result['choices'][0]['message']['content']
            return ChatResponse(
                content=content,
                role="assistant",
                finish_reason=result['choices'][0].get('finish_reason', 'stop')
            )
    
    async def create_embeddings(
        self,
        input_text: Union[str, List[str]],
        model: str = "text-embedding-ada-002"
    ) -> List[List[float]]:
        """Generate embeddings using LocalAI."""
        
        payload = {
            "input": input_text,
            "model": model
        }
        
        async with self.session.post(
            f"{self.base_url}/v1/embeddings",
            json=payload,
            headers={"Content-Type": "application/json"}
        ) as response:
            result = await response.json()
            return [item['embedding'] for item in result['data']]
    
    async def generate_image(
        self,
        prompt: str,
        model: str = "flux.1-dev-ggml",
        size: str = "1024x1024",
        n: int = 1
    ) -> List[str]:
        """Generate images using LocalAI."""
        
        payload = {
            "prompt": prompt,
            "model": model,
            "size": size,
            "n": n
        }
        
        async with self.session.post(
            f"{self.base_url}/v1/images/generations",
            json=payload,
            headers={"Content-Type": "application/json"}
        ) as response:
            result = await response.json()
            return [item['url'] for item in result['data']]
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models."""
        async with self.session.get(f"{self.base_url}/v1/models") as response:
            result = await response.json()
            return result.get('data', [])

    async def grammar_constrained_generation(
        self,
        messages: List[Dict[str, str]],
        model: str,
        schema: Union[BaseModel, Dict, str],
        **kwargs
    ) -> Any:
        """Generate structured output using grammar constraints."""
        
        # Generate GBNF grammar from schema
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            grammar = GrammarGenerator.from_pydantic(schema)
        elif isinstance(schema, dict):
            grammar = GrammarGenerator.from_json_schema(schema)
        else:
            grammar = str(schema)
        
        # Use chat completion with grammar
        response = await self.chat_completion(
            messages=messages,
            model=model,
            grammar=grammar,
            **kwargs
        )
        
        # Parse response according to schema
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            return schema.parse_raw(response.content)
        return response.content
```

**3.2.2 Simplified Pipeline Implementations**

Replace complex pipeline classes with LocalAI client wrappers:

```python
# runner/pipelines/localai/text_pipeline.py
from typing import List, Optional, Dict, Any, AsyncGenerator, Union
from models import Message, ChatResponse, ModelProfile, Model
from ..base import BasePipelineCore, PipeReturn, GrammarInput
from ...clients.localai_client import LocalAIClient

class LocalAITextPipeline(BasePipelineCore):
    """Simplified text generation pipeline using LocalAI."""
    
    def __init__(
        self, 
        model: Model, 
        profile: ModelProfile,
        localai_url: str = "http://localai-service:8080"
    ):
        super().__init__(model, profile)
        self.localai_url = localai_url
        self.model_name = model.name  # LocalAI model identifier
    
    async def run_pipeline(
        self,
        messages: List[Message],
        grammar: Optional[GrammarInput] = None,
        **kwargs
    ) -> PipeReturn:
        """Execute text generation pipeline."""
        
        # Convert internal messages to OpenAI format
        openai_messages = [
            {"role": msg.role.value, "content": self._extract_content(msg)}
            for msg in messages
        ]
        
        # Generate grammar if provided
        grammar_str = None
        if grammar:
            from utils.grammar_generator import GrammarGenerator
            grammar_str = GrammarGenerator.generate(grammar)
        
        async with LocalAIClient(self.localai_url) as client:
            response = await client.chat_completion(
                messages=openai_messages,
                model=self.model_name,
                grammar=grammar_str,
                **kwargs
            )
            
            return response.content
    
    async def stream_pipeline(
        self,
        messages: List[Message],
        grammar: Optional[GrammarInput] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream text generation pipeline."""
        
        # Convert internal messages to OpenAI format
        openai_messages = [
            {"role": msg.role.value, "content": self._extract_content(msg)}
            for msg in messages
        ]
        
        # Generate grammar if provided
        grammar_str = None
        if grammar:
            from utils.grammar_generator import GrammarGenerator
            grammar_str = GrammarGenerator.generate(grammar)
        
        async with LocalAIClient(self.localai_url) as client:
            async for token in client.chat_completion(
                messages=openai_messages,
                model=self.model_name,
                stream=True,
                grammar=grammar_str,
                **kwargs
            ):
                yield token
    
    def _extract_content(self, message: Message) -> str:
        """Extract text content from message."""
        if isinstance(message.content, list):
            return " ".join([
                item.text for item in message.content 
                if item.type == "text"
            ])
        return message.content
```

```python
# runner/pipelines/localai/embedding_pipeline.py
from typing import List
from models import Model, ModelProfile
from ..base import EmbeddingPipeline, Embeddings
from ...clients.localai_client import LocalAIClient

class LocalAIEmbeddingPipeline(EmbeddingPipeline):
    """Simplified embedding pipeline using LocalAI."""
    
    def __init__(
        self, 
        model: Model, 
        profile: ModelProfile,
        localai_url: str = "http://localai-service:8080"
    ):
        super().__init__(model, profile)
        self.localai_url = localai_url
        self.model_name = model.name
    
    async def embed_pipeline(self, texts: List[str]) -> Embeddings:
        """Generate embeddings using LocalAI."""
        
        async with LocalAIClient(self.localai_url) as client:
            embeddings = await client.create_embeddings(
                input_text=texts,
                model=self.model_name
            )
            return embeddings
```

### Phase 3: Configuration Migration

**3.3.1 Model Configuration Transformation**

Create LocalAI model configuration system replacing individual pipeline configs:

```python
# runner/config/localai_models.py
from typing import Dict, List
from dataclasses import dataclass
from enum import Enum

class LocalAIBackend(Enum):
    LLAMA_CPP = "llama"
    TRANSFORMERS = "transformers"
    SENTENCE_TRANSFORMERS = "sentencetransformers"
    STABLE_DIFFUSION = "stablediffusion"
    WHISPER = "whisper"

@dataclass
class LocalAIModelConfig:
    """LocalAI model configuration."""
    name: str
    backend: LocalAIBackend
    parameters: Dict[str, any]
    features: List[str]
    context_size: int = 4096
    gpu_layers: int = -1  # Use all GPU layers by default

# Model registry mapping internal model names to LocalAI configurations
LOCALAI_MODEL_REGISTRY = {
    # Text Generation Models
    "qwen3-30b-a3b-q4-k-m": LocalAIModelConfig(
        name="qwen3-30b-instruct:q4_k_m",
        backend=LocalAIBackend.LLAMA_CPP,
        parameters={"model": "qwen3-30b-instruct.q4_k_m.gguf"},
        features=["text-generation", "function-calling", "grammar-constraints"],
        context_size=131072,
        gpu_layers=-1
    ),
    
    "openai-gpt-oss-20b": LocalAIModelConfig(
        name="openai-gpt-oss-20b:q4_k_m",
        backend=LocalAIBackend.LLAMA_CPP,
        parameters={"model": "openai-gpt-oss-20b.q4_k_m.gguf"},
        features=["text-generation", "function-calling", "grammar-constraints"],
        context_size=131072,
        gpu_layers=-1
    ),
    
    # Embedding Models  
    "qwen3-embedding": LocalAIModelConfig(
        name="text-embedding-ada-002",
        backend=LocalAIBackend.SENTENCE_TRANSFORMERS,
        parameters={"model": "all-MiniLM-L6-v2"},
        features=["embeddings"],
        context_size=512,
        gpu_layers=0  # CPU embeddings
    ),
    
    # Image Generation Models
    "flux-1-dev": LocalAIModelConfig(
        name="flux.1-dev-ggml",
        backend=LocalAIBackend.STABLE_DIFFUSION,
        parameters={"model": "flux.1-dev.ggml"},
        features=["image-generation"],
        context_size=0,
        gpu_layers=-1
    ),
    
    # Vision Models
    "qwen25-vl": LocalAIModelConfig(
        name="minicpm-v-4_5",
        backend=LocalAIBackend.LLAMA_CPP,
        parameters={"model": "minicpm-v-4_5.gguf"},
        features=["vision", "text-generation"],
        context_size=32768,
        gpu_layers=-1
    )
}

def get_localai_config(internal_model_name: str) -> LocalAIModelConfig:
    """Get LocalAI configuration for internal model name."""
    return LOCALAI_MODEL_REGISTRY.get(
        internal_model_name,
        LocalAIModelConfig(
            name=internal_model_name,
            backend=LocalAIBackend.LLAMA_CPP,
            parameters={"model": f"{internal_model_name}.gguf"},
            features=["text-generation"],
            context_size=4096,
            gpu_layers=-1
        )
    )
```

**3.3.2 Pipeline Factory Refactoring**

Update pipeline factory to use LocalAI configurations:

```python
# runner/pipeline_factory.py (Updated)
from typing import Optional, Type
from models import Model, ModelProfile
from .pipelines.base import BasePipelineCore, EmbeddingPipeline
from .pipelines.localai.text_pipeline import LocalAITextPipeline
from .pipelines.localai.embedding_pipeline import LocalAIEmbeddingPipeline
from .pipelines.localai.image_pipeline import LocalAIImagePipeline
from .config.localai_models import get_localai_config, LocalAIBackend

def pipeline_factory(
    model: Model,
    profile: ModelProfile,
    expected_return_type: Optional[type] = None,
    localai_url: str = "http://localai-service:8080"
) -> BasePipelineCore:
    """Factory for creating LocalAI-based pipelines."""
    
    # Get LocalAI configuration for the model
    localai_config = get_localai_config(model.name)
    
    # Determine pipeline type based on expected return type and model features
    if expected_return_type and issubclass(expected_return_type, list):
        # Embedding pipeline
        return LocalAIEmbeddingPipeline(
            model=model,
            profile=profile,
            localai_url=localai_url
        )
    elif "image-generation" in localai_config.features:
        # Image generation pipeline
        return LocalAIImagePipeline(
            model=model,
            profile=profile,
            localai_url=localai_url
        )
    else:
        # Default text generation pipeline
        return LocalAITextPipeline(
            model=model,
            profile=profile,
            localai_url=localai_url
        )

# Legacy compatibility - maintain existing interface
PipeReturn = Union[str, List[List[float]], bytes]
```

### Phase 4: Grammar Constraint Integration

**3.4.1 Grammar Generator LocalAI Integration**

Integrate existing grammar generator with LocalAI's GBNF support:

```python
# utils/grammar_generator.py (Enhanced for LocalAI)
from typing import Union, Type, Dict, Any
from pathlib import Path
from pydantic import BaseModel
import json

class GrammarGenerator:
    """Enhanced grammar generator with LocalAI GBNF support."""
    
    @staticmethod
    def generate(grammar_input: Union[str, Path, Type[BaseModel], Dict, None]) -> Optional[str]:
        """Generate GBNF grammar for LocalAI constraints."""
        
        if grammar_input is None:
            return None
        
        if isinstance(grammar_input, str):
            # Direct GBNF grammar string
            return grammar_input
        
        if isinstance(grammar_input, Path):
            # Load GBNF grammar from file
            return grammar_input.read_text()
        
        if isinstance(grammar_input, type) and issubclass(grammar_input, BaseModel):
            # Generate GBNF from Pydantic model
            return GrammarGenerator.from_pydantic(grammar_input)
        
        if isinstance(grammar_input, dict):
            # Generate GBNF from JSON schema
            return GrammarGenerator.from_json_schema(grammar_input)
        
        return None
    
    @staticmethod
    def from_pydantic(model_class: Type[BaseModel]) -> str:
        """Generate GBNF grammar from Pydantic model."""
        
        # Convert Pydantic model to JSON schema
        schema = model_class.model_json_schema()
        return GrammarGenerator.from_json_schema(schema)
    
    @staticmethod
    def from_json_schema(schema: Dict[str, Any]) -> str:
        """Generate GBNF grammar from JSON schema."""
        
        # Simplified GBNF generation - in practice, this would be more comprehensive
        if schema.get('type') == 'object':
            properties = schema.get('properties', {})
            required = set(schema.get('required', []))
            
            rules = ['root ::= "{" ws object-content ws "}"']
            rules.append('object-content ::= (property ("," ws property)*)?')
            
            property_rules = []
            for name, prop_schema in properties.items():
                is_required = name in required
                prop_type = GrammarGenerator._get_property_type(prop_schema)
                
                if is_required:
                    property_rules.append(f'"{name}" ws ":" ws {prop_type}')
                else:
                    property_rules.append(f'("{name}" ws ":" ws {prop_type})?')
            
            rules.append(f'property ::= {" | ".join(property_rules)}')
            rules.extend(GrammarGenerator._get_basic_rules())
            
            return '\n'.join(rules)
        
        return 'root ::= [^]*'  # Fallback: accept any content
    
    @staticmethod
    def _get_property_type(schema: Dict[str, Any]) -> str:
        """Get GBNF type rule for property schema."""
        
        schema_type = schema.get('type', 'string')
        
        if schema_type == 'string':
            return 'string-value'
        elif schema_type == 'integer':
            return 'integer-value'
        elif schema_type == 'number':
            return 'number-value'
        elif schema_type == 'boolean':
            return 'boolean-value'
        elif schema_type == 'array':
            return 'array-value'
        elif schema_type == 'object':
            return 'object-value'
        
        return 'string-value'  # Fallback
    
    @staticmethod
    def _get_basic_rules() -> List[str]:
        """Get basic GBNF rules for JSON components."""
        
        return [
            'ws ::= [ \t\n\r]*',
            'string-value ::= "\"" [^"]* "\""',
            'integer-value ::= "-"? [0-9]+',
            'number-value ::= "-"? [0-9]+ ("." [0-9]+)?',
            'boolean-value ::= "true" | "false"',
            'array-value ::= "[" ws (string-value ("," ws string-value)*)? ws "]"',
            'object-value ::= "{" ws "}"'  # Simplified nested object
        ]
```

### Phase 5: Testing and Validation Framework

**3.5.1 LocalAI Integration Tests**

Create comprehensive test suite for LocalAI integration:

```python
# test/localai/test_integration.py
import pytest
import asyncio
from runner.clients.localai_client import LocalAIClient
from runner.pipelines.localai.text_pipeline import LocalAITextPipeline
from models import Model, ModelProfile, Message, MessageRole

class TestLocalAIIntegration:
    """Integration tests for LocalAI client and pipelines."""
    
    @pytest.fixture
    def localai_client(self):
        return LocalAIClient("http://localhost:8080")
    
    @pytest.fixture
    def sample_model(self):
        return Model(
            name="qwen3-30b-instruct:q4_k_m",
            type="text-generation",
            path="/models/qwen3-30b-instruct.q4_k_m.gguf"
        )
    
    @pytest.fixture
    def sample_profile(self):
        return ModelProfile(
            name="default",
            parameters={
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 2048
            }
        )
    
    @pytest.mark.asyncio
    async def test_localai_connection(self, localai_client):
        """Test LocalAI server connection."""
        
        async with localai_client:
            models = await localai_client.list_models()
            assert isinstance(models, list)
            assert len(models) > 0
    
    @pytest.mark.asyncio
    async def test_chat_completion(self, localai_client):
        """Test chat completion functionality."""
        
        messages = [
            {"role": "user", "content": "Say 'Hello World'"}
        ]
        
        async with localai_client:
            response = await localai_client.chat_completion(
                messages=messages,
                model="qwen3-30b-instruct:q4_k_m"
            )
            
            assert response.content is not None
            assert "Hello" in response.content
    
    @pytest.mark.asyncio
    async def test_streaming_completion(self, localai_client):
        """Test streaming chat completion."""
        
        messages = [
            {"role": "user", "content": "Count from 1 to 5"}
        ]
        
        tokens = []
        async with localai_client:
            async for token in localai_client.chat_completion(
                messages=messages,
                model="qwen3-30b-instruct:q4_k_m",
                stream=True
            ):
                tokens.append(token)
        
        assert len(tokens) > 0
        full_response = "".join(tokens)
        assert any(str(i) in full_response for i in range(1, 6))
    
    @pytest.mark.asyncio
    async def test_grammar_constraints(self, localai_client):
        """Test grammar-constrained generation."""
        
        messages = [
            {"role": "user", "content": "Do you like apples?"}
        ]
        
        # Simple binary choice grammar
        grammar = 'root ::= ("yes" | "no")'
        
        async with localai_client:
            response = await localai_client.chat_completion(
                messages=messages,
                model="qwen3-30b-instruct:q4_k_m",
                grammar=grammar
            )
            
            assert response.content.lower() in ["yes", "no"]
    
    @pytest.mark.asyncio
    async def test_text_pipeline(self, sample_model, sample_profile):
        """Test LocalAI text pipeline integration."""
        
        pipeline = LocalAITextPipeline(
            model=sample_model,
            profile=sample_profile,
            localai_url="http://localhost:8080"
        )
        
        messages = [
            Message(
                role=MessageRole.USER,
                content="What is the capital of France?"
            )
        ]
        
        response = await pipeline.run_pipeline(messages)
        assert isinstance(response, str)
        assert "Paris" in response
    
    @pytest.mark.asyncio
    async def test_embedding_generation(self, localai_client):
        """Test embedding generation."""
        
        texts = ["Hello world", "How are you?"]
        
        async with localai_client:
            embeddings = await localai_client.create_embeddings(
                input_text=texts,
                model="text-embedding-ada-002"
            )
            
            assert len(embeddings) == 2
            assert all(isinstance(emb, list) for emb in embeddings)
            assert all(len(emb) > 0 for emb in embeddings)

# test/localai/test_performance.py
import pytest
import time
import asyncio
from runner.clients.localai_client import LocalAIClient

class TestLocalAIPerformance:
    """Performance tests for LocalAI integration."""
    
    @pytest.mark.asyncio
    async def test_response_time(self):
        """Test response time for standard completion."""
        
        client = LocalAIClient("http://localhost:8080")
        messages = [{"role": "user", "content": "Hello"}]
        
        start_time = time.time()
        async with client:
            response = await client.chat_completion(
                messages=messages,
                model="qwen3-30b-instruct:q4_k_m"
            )
        end_time = time.time()
        
        response_time = end_time - start_time
        assert response_time < 30  # Should respond within 30 seconds
        assert response.content is not None
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test concurrent request handling."""
        
        async def single_request(client, prompt):
            messages = [{"role": "user", "content": prompt}]
            return await client.chat_completion(
                messages=messages,
                model="qwen3-30b-instruct:q4_k_m"
            )
        
        client = LocalAIClient("http://localhost:8080")
        prompts = [f"Count to {i}" for i in range(1, 6)]
        
        start_time = time.time()
        async with client:
            responses = await asyncio.gather(*[
                single_request(client, prompt) for prompt in prompts
            ])
        end_time = time.time()
        
        total_time = end_time - start_time
        assert len(responses) == 5
        assert all(response.content is not None for response in responses)
        assert total_time < 60  # All requests should complete within 60 seconds
```

### Phase 6: Migration Strategy and Rollout

**3.6.1 Backward Compatibility Layer**

Create compatibility layer to ensure smooth migration:

```python
# runner/compatibility/legacy_adapter.py
from typing import List, Optional, Dict, Any
from models import Model, ModelProfile, Message
from ..pipelines.base import BasePipelineCore, PipeReturn
from ..pipelines.localai.text_pipeline import LocalAITextPipeline

class LegacyPipelineAdapter:
    """Adapter to maintain compatibility with existing pipeline interfaces."""
    
    def __init__(self, localai_pipeline: LocalAITextPipeline):
        self.localai_pipeline = localai_pipeline
    
    # Legacy method mappings
    def generate_response(self, *args, **kwargs):
        """Legacy method - redirects to LocalAI pipeline."""
        return asyncio.run(self.localai_pipeline.run_pipeline(*args, **kwargs))
    
    def stream_response(self, *args, **kwargs):
        """Legacy streaming method - redirects to LocalAI pipeline."""
        return self.localai_pipeline.stream_pipeline(*args, **kwargs)

# Migration flag in environment configuration
ENABLE_LOCALAI_MIGRATION = os.getenv("ENABLE_LOCALAI_MIGRATION", "false").lower() == "true"

def get_pipeline_implementation(model: Model, profile: ModelProfile) -> BasePipelineCore:
    """Factory method with migration flag support."""
    
    if ENABLE_LOCALAI_MIGRATION:
        # Use new LocalAI implementation
        return LocalAITextPipeline(model, profile)
    else:
        # Use legacy implementation (during transition period)
        from ..legacy.openai_gpt_oss import OpenAiGptOssPipe
        return OpenAiGptOssPipe(model, profile)
```

**3.6.2 Deployment Strategy**

Phased deployment approach:

1. **Phase A: Parallel Deployment**
   - Deploy LocalAI alongside existing infrastructure
   - Enable feature flag for gradual testing
   - Monitor performance and functionality

2. **Phase B: Gradual Migration**
   - Enable LocalAI for non-critical operations first
   - Migrate embedding pipelines (lower risk)
   - Migrate text generation pipelines with fallback

3. **Phase C: Full Migration**
   - Switch all operations to LocalAI
   - Remove legacy pipeline implementations
   - Clean up deprecated code and dependencies

## 4. Benefits and Impact Analysis

### 4.1 Technical Benefits

**Code Reduction:**
- **Current Runner Module:** ~2000+ lines across multiple pipeline implementations
- **Post-Migration Runner:** ~300 lines with LocalAI client interfaces
- **Maintenance Reduction:** 85% reduction in pipeline-specific code

**Infrastructure Simplification:**
- **GPU Management:** LocalAI handles automatic backend detection and optimization
- **Model Loading:** LocalAI manages model loading, unloading, and memory optimization
- **Resource Management:** Built-in resource pooling and optimization

**Feature Enhancement:**
- **Unified API:** Single interface for all AI operations
- **Extended Capabilities:** Built-in function calling, grammar constraints, and vision support
- **Distributed Inference:** Support for distributed model execution
- **Model Gallery:** Automatic model discovery and installation

### 4.2 Operational Benefits

**Deployment Simplification:**
- Single LocalAI container replaces complex pipeline configurations
- Automatic GPU optimization eliminates manual tuning
- Built-in model caching reduces storage requirements

**Scalability Improvements:**
- LocalAI's distributed inference capabilities
- Better resource utilization through automatic optimization
- Support for heterogeneous GPU environments

**Development Velocity:**
- Reduced complexity in adding new model support
- Standard OpenAI-compatible interface
- Better testing and debugging capabilities

### 4.3 Risk Mitigation

**Migration Risks:**
- **Compatibility Risk:** Mitigated by backward compatibility layer
- **Performance Risk:** Addressed by comprehensive performance testing
- **Feature Risk:** LocalAI provides superset of current functionality

**Operational Risks:**
- **Single Point of Failure:** Mitigated by LocalAI's built-in reliability features
- **Vendor Lock-in:** LocalAI is open-source with standard OpenAI API
- **Resource Usage:** LocalAI typically uses resources more efficiently

## 5. Implementation Timeline

### Month 1: Infrastructure Setup
- Week 1-2: LocalAI deployment configuration
- Week 3-4: Environment integration and testing

### Month 2: Core Client Development  
- Week 1-2: LocalAI client implementation
- Week 3-4: Basic pipeline migrations (text, embeddings)

### Month 3: Advanced Features
- Week 1-2: Grammar constraint integration
- Week 3-4: Image and vision pipeline migrations

### Month 4: Testing and Validation
- Week 1-2: Comprehensive testing suite
- Week 3-4: Performance optimization and validation

### Month 5: Migration and Rollout
- Week 1-2: Parallel deployment and gradual migration
- Week 3-4: Full migration and legacy cleanup

## 6. Success Metrics

### Technical Metrics
- **Code Reduction:** 80%+ reduction in runner module complexity
- **Performance:** Maintain or improve response times
- **Resource Usage:** 20%+ improvement in GPU utilization
- **Feature Parity:** 100% functionality preservation

### Operational Metrics  
- **Deployment Time:** 50%+ reduction in deployment complexity
- **Bug Reports:** Significant reduction in pipeline-specific issues
- **Developer Velocity:** Faster implementation of new model support
- **System Reliability:** Improved uptime and error handling

## 7. Conclusion

The migration to LocalAI represents a strategic architectural improvement that simplifies the runner module while enhancing capabilities. By leveraging LocalAI's comprehensive feature set and automatic optimizations, the platform will achieve better maintainability, performance, and scalability while reducing technical debt significantly.

The phased approach ensures minimal risk while providing clear benefits at each stage. The resulting architecture will be more robust, easier to maintain, and better positioned for future AI capabilities integration.