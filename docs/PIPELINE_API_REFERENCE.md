# Pipeline API Reference

## Core Abstract Classes

### BasePipelineCore[PipeType]

The foundational abstract base class for all pipelines.

```python
class BasePipelineCore(ABC, Generic[PipeType]):
    """Core pipeline functionality with mandatory LangGraph support."""
    
    # Class attributes
    allowed_return_types: tuple[type, ...] = (str, ChatResponse, list)
    default_return_type: Optional[type] = None
```

#### Constructor

```python
def __init__(
    self,
    model: Model,
    profile: ModelProfile,
    expected_return_type: Optional[type] = None,
)
```

**Parameters:**
- `model`: Model configuration and metadata
- `profile`: Model parameters and settings
- `expected_return_type`: Expected return type for validation

#### Abstract Methods

```python
@abstractmethod
def create_graph(
    self, tools: Optional[List[BaseTool]] = None
) -> CompiledStateGraph[LangGraphState, None, LangGraphState, LangGraphState]

@abstractmethod
async def process_messages(
    self,
    messages: List[Message],
    tools: Optional[List[BaseTool]] = None,
    is_tool_generation: bool = False,
) -> PipeType
```

#### Public Methods

```python
async def run_pipeline(
    self,
    messages: List[Message],
    tools: Optional[List[BaseTool]] = None,
    is_tool_generation: bool = False,
) -> PipeType

async def prompt(self, text: str | List[str]) -> PipeType

def allows_return_type(self, t: type) -> bool

def validate_return_value(self, value: Any) -> None

def get_common_args(self) -> dict
```

#### Streaming Hooks (New Architecture)

```python
def process_streaming_token(self, content: str) -> Optional[ChatResponse]:
    """Process individual streaming tokens with custom routing logic."""

def reset_streaming_state(self) -> None:
    """Reset streaming state before each session."""

def finalize_streaming(self) -> Optional[ChatResponse]:
    """Handle final processing when streaming completes."""
```

### BaseLangGraphPipeline[PipeType]

Extended base class with LangGraph integration, circuit breakers, and timeout protection.

```python
class BaseLangGraphPipeline(BasePipelineCore[PipeType]):
    """Enhanced pipeline with circuit breaker and timeout protection."""
```

#### Constructor

```python
def __init__(
    self,
    model: Model,
    profile: ModelProfile,
    expected_return_type: Optional[type] = None,
    circuit_config: Optional[CircuitBreakerConfig] = None,
    timeout_strategy: str = "medium",
)
```

**Additional Parameters:**
- `circuit_config`: Circuit breaker configuration for fault tolerance
- `timeout_strategy`: Timeout strategy ("conservative", "medium", "aggressive")

#### Circuit Breaker Features

- **Perplexity Guard**: Monitors response quality
- **Repetition Guard**: Detects and prevents repetitive output
- **Timeout Protection**: Prevents hung operations
- **Retry Logic**: Automatic retry with backoff

### BaseLlamaCppPipeline

Specialized base class for llama.cpp-based models with auto-backoff loading.

```python
class BaseLlamaCppPipeline(BaseLangGraphPipeline):
    """Base for llama.cpp models with heuristic auto-backoff loading."""
```

#### Features

- **Auto-backoff Loading**: Automatically adjusts parameters if loading fails
- **GPU Layer Optimization**: Calculates optimal GPU layer distribution
- **Memory Management**: Handles VRAM constraints
- **GGUF Validation**: Validates model files before loading

#### Loading Parameters

```python
@dataclass
class LlamaLoadAttempt:
    n_ctx: int           # Context window size
    n_batch: int         # Batch size for processing
    n_gpu_layers: int    # Number of layers on GPU
    logits_all: bool     # Whether to compute logits for all tokens
    logprobs: int        # Number of logprobs to return
    attempt_index: int   # Attempt number
    error: Optional[str] # Error message if failed
```

## Specialized Pipeline Classes

### ChatPipeline

Base class for text-to-text pipelines that return ChatResponse.

```python
class ChatPipeline(BasePipelineCore, LangGraphCapable):
    """Base class for chat pipelines with LangGraph support."""
    
    allowed_return_types: tuple[type, ...] = (ChatResponse,)
    default_return_type: Optional[type] = ChatResponse
```

### EmbeddingPipeline

Base class for embedding generation pipelines.

```python
class EmbeddingPipeline(BasePipelineCore):
    """Base class for embedding pipelines."""
    
    allowed_return_types: tuple[type, ...] = (list,)
    default_return_type: Optional[type] = list
```

### ImageGenerationPipeline

Base class for text-to-image generation pipelines.

```python
class ImageGenerationPipeline(BasePipelineCore):
    """Base class for image generation pipelines."""
    
    allowed_return_types: tuple[type, ...] = (ChatResponse,)
    default_return_type: Optional[type] = ChatResponse
```

## Data Models

### Message

```python
@dataclass
class Message:
    role: MessageRole
    content: List[MessageContent]
    thinking: Optional[str] = None
    metadata: Optional[dict] = None
```

### MessageContent

```python
@dataclass
class MessageContent:
    type: MessageContentType
    text: Optional[str] = None
    image_url: Optional[str] = None
    image_data: Optional[bytes] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[dict]] = None
```

### MessageContentType

```python
class MessageContentType(Enum):
    TEXT = "text"
    IMAGE = "image"
    IMAGE_URL = "image_url"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
```

### MessageRole

```python
class MessageRole(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    OBSERVER = "observer"
```

### ChatResponse

```python
@dataclass
class ChatResponse:
    done: bool
    message: Optional[Message] = None
    created_at: Optional[datetime] = None
    context: Optional[List[List[float]]] = None
    finish_reason: Optional[str] = None
    total_duration: Optional[float] = None
    load_duration: Optional[float] = None
    prompt_eval_count: Optional[float] = None
    prompt_eval_duration: Optional[float] = None
    eval_count: Optional[float] = None
    eval_duration: Optional[float] = None
    thinking: Optional[str] = None
    channels: Optional[Dict[str, Any]] = None
    observer_messages: Optional[List[str]] = None
```

### Model

```python
@dataclass
class Model:
    id: str
    name: str
    provider: ModelProvider
    model: str
    pipeline: str
    modality: List[str]
    details: Optional[ModelDetails] = None
    lora_weights: Optional[List[LoraWeight]] = None
```

### ModelProfile

```python
@dataclass
class ModelProfile:
    system_prompt: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    repeat_penalty: Optional[float] = None
    num_ctx: Optional[int] = None
    num_predict: Optional[int] = None
    stop: Optional[List[str]] = None
    reasoning_effort: Optional[str] = None
```

### CircuitBreakerConfig

```python
@dataclass
class CircuitBreakerConfig:
    enable_perplexity_guard: bool = False
    perplexity_threshold: float = 10.0
    enable_repetition_guard: bool = True
    max_repetitions: int = 3
    timeout_seconds: float = 120.0
    max_retries: int = 1
    backoff_factor: float = 1.5
    jitter: bool = True
```

## LangGraph Integration

### LangGraphState

```python
@dataclass
class LangGraphState:
    messages: List[Message]
    tools: Optional[List[BaseTool]] = None
    tool_calls: Optional[List[dict]] = None
    response: Optional[str] = None
    thinking: Optional[str] = None
    metadata: Optional[dict] = None
```

### Graph Construction Helpers

```python
def build_langgraph_state(
    messages: List[Message],
    tools: Optional[List[BaseTool]] = None
) -> LangGraphState

def build_lc_messages(messages: List[Message]) -> List[BaseMessage]

def coerce_to_langchain_message_dict(message: Message) -> Dict[str, Any]
```

## Factory and Caching

### PipelineFactory

```python
class PipelineFactory:
    """Factory for creating and managing pipelines."""
    
    def create_pipeline(
        self,
        model_id: str,
        expected_return_type: Optional[type] = None,
        circuit_config: Optional[CircuitBreakerConfig] = None,
        priority: PipelinePriority = PipelinePriority.NORMAL,
        timeout_strategy: str = "medium",
    ) -> BasePipelineCore
    
    def get_model(self, model_id: str) -> Model
    
    def list_available_models(self) -> List[str]
    
    def cleanup_cache(self) -> None
```

### LocalPipelineCacheManager

```python
class LocalPipelineCacheManager:
    """Manages pipeline caching with automatic cleanup."""
    
    def get_or_create_pipeline(
        self,
        cache_key: str,
        factory_func: Callable[[], BasePipelineCore],
        priority: PipelinePriority = PipelinePriority.NORMAL,
    ) -> BasePipelineCore
    
    def cleanup_expired_pipelines(self) -> None
    
    def force_cleanup_all(self) -> None
```

## Streaming and Event Processing

### EventStreamProcessor

```python
class EventStreamProcessor:
    """Processes streaming events with pipeline delegation."""
    
    def __init__(self, config: Optional[StreamingConfig] = None):
        self.pipeline: Optional[BasePipelineCore] = None
    
    def set_pipeline(self, pipeline: Optional[BasePipelineCore]) -> None:
        """Associate pipeline for post-processing."""
    
    def process_event(self, evt: StandardStreamEvent) -> Optional[ChatResponse]:
        """Process streaming event."""
    
    def finalize_pipeline_streaming(self) -> Optional[ChatResponse]:
        """Finalize pipeline streaming."""
```

### Streaming Functions

```python
async def stream_pipeline(
    messages: List[Message],
    model_id: str,
    tools: Optional[List[BaseTool]] = None,
    expected_return_type: Optional[type] = None,
    circuit_config: Optional[CircuitBreakerConfig] = None,
    profile: Optional[ModelProfile] = None,
    timeout_strategy: str = "medium",
) -> AsyncIterator[ChatResponse]

async def run_pipeline(
    messages: List[Message],
    model_id: str,
    tools: Optional[List[BaseTool]] = None,
    expected_return_type: Optional[type] = None,
    circuit_config: Optional[CircuitBreakerConfig] = None,
    profile: Optional[ModelProfile] = None,
    timeout_strategy: str = "medium",
) -> Union[str, ChatResponse, List[List[float]]]

async def embed_pipeline(
    messages: List[Message],
    model_id: str,
    profile: Optional[ModelProfile] = None,
) -> List[List[float]]
```

## Utility Functions

### Message Utilities

```python
def extract_message_text(message: Message) -> str:
    """Extract text content from message."""

def to_lc_message(message: Message) -> BaseMessage:
    """Convert to LangChain message format."""

def from_lc_message(lc_message: BaseMessage) -> Message:
    """Convert from LangChain message format."""
```

### Response Utilities

```python
def create_streaming_chunk(
    text: str,
    done: bool = False,
    role: MessageRole = MessageRole.ASSISTANT
) -> ChatResponse:
    """Create streaming response chunk."""

def create_error_response(error_message: str) -> ChatResponse:
    """Create error response."""
```

### Hardware Management

```python
class HardwareManager:
    """Manages GPU memory and hardware resources."""
    
    def get_available_vram(self) -> int
    def optimize_gpu_layers(self, model_size: int) -> int
    def cleanup_gpu_memory(self) -> None
```

## Environment Variables

### Core Configuration

- `MODELS_FILE_PATH`: Path to models configuration file
- `ALLOW_MISSING_GGUF`: Allow missing GGUF files in dev/test
- `PYTORCH_CUDA_ALLOC_CONF`: CUDA memory allocation configuration

### Circuit Breaker Overrides

- `CIRCUIT_BREAKER_ENABLE_PERPLEXITY_GUARD`: Override perplexity guard
- `CIRCUIT_BREAKER_PERPLEXITY_THRESHOLD`: Override perplexity threshold
- `CIRCUIT_BREAKER_ENABLE_REPETITION_GUARD`: Override repetition guard
- `CIRCUIT_BREAKER_MAX_REPETITIONS`: Override max repetitions
- `CIRCUIT_BREAKER_TIMEOUT_SECONDS`: Override timeout
- `CIRCUIT_BREAKER_MAX_RETRIES`: Override retry count

### Model Loading

- `LLAMA_VERBOSE`: Enable verbose llama.cpp logging
- `CUDA_VISIBLE_DEVICES`: Specify GPU devices
- `OMP_NUM_THREADS`: OpenMP thread count

## Error Handling

### Common Exceptions

```python
class PipelineError(Exception):
    """Base pipeline exception."""

class ModelLoadError(PipelineError):
    """Model loading failed."""

class ValidationError(PipelineError):
    """Input/output validation failed."""

class CircuitBreakerError(PipelineError):
    """Circuit breaker triggered."""

class TimeoutError(PipelineError):
    """Operation timed out."""
```

### Error Response Format

```python
{
    "error": {
        "type": "ModelLoadError",
        "message": "Failed to load model",
        "details": "GGUF file not found",
        "traceback": "..."
    }
}
```

This API reference provides comprehensive documentation for all public interfaces, data models, and utility functions in the pipeline system. Use it alongside the implementation guide for complete understanding of the architecture.