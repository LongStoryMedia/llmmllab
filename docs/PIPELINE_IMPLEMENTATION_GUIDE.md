# Pipeline Implementation Guide for LLM ML Lab Runner Module

## Overview

The LLM ML Lab runner module provides a flexible pipeline architecture for implementing various AI model types including text generation, embeddings, multimodal (vision-language), and image generation. This guide covers all aspects of implementing custom pipelines.

## Architecture Overview

### Core Components

1. **BasePipelineCore** - Abstract base class for all pipelines
2. **BaseLangGraphPipeline** - Extended base with LangGraph integration, circuit breakers, and timeout protection  
3. **BaseLlamaCppPipeline** - Specialized base for llama.cpp-based models with auto-backoff loading
4. **EventStreamProcessor** - Handles streaming and delegates to pipeline-specific post-processing
5. **PipelineFactory** - Creates and manages pipeline instances with caching and resource management

### Pipeline Types

- **Text-to-Text**: Standard language models (chat, completion, reasoning)
- **Embedding**: Vector embedding generation from text
- **Text-to-Image**: Image generation from text prompts  
- **Image-to-Image**: Image transformation/editing
- **Image+Text-to-Text**: Multimodal vision-language models

## Base Classes Hierarchy

```
BasePipelineCore[PipeType]
├── ChatPipeline (text-to-text with ChatResponse)
├── EmbeddingPipeline (text to embeddings)
├── ImageGenerationPipeline (text-to-image)
└── BaseLangGraphPipeline[PipeType]
    ├── BaseLlamaCppPipeline
    │   ├── QwenLangGraphPipe (text-to-text)
    │   ├── OpenAiGptOssPipe (text-to-text with harmony)
    │   ├── Qwen25VLPipeline (multimodal)
    │   └── Qwen3EmbeddingPipe (embeddings)
    ├── FluxPipe (text-to-image)
    └── Custom LangGraph pipelines
```

## Implementation Requirements by Pipeline Type

### 1. Text-to-Text Pipelines

**Base Classes to Inherit From:**
- `BaseLlamaCppPipeline` (for GGUF models)
- `ChatPipeline` (for other backends)

**Required Methods:**
```python
class MyTextPipeline(BaseLlamaCppPipeline):
    # Required return type specification
    allowed_return_types: tuple[type, ...] = (ChatResponse,)
    default_return_type: Optional[type] = ChatResponse
    
    def __init__(self, model: Model, profile: ModelProfile, 
                 expected_return_type: Optional[type] = None,
                 circuit_config: Optional[CircuitBreakerConfig] = None):
        super().__init__(model, profile, expected_return_type, circuit_config)
        # Initialize model-specific state
        
    async def process_messages(self, messages: List[Message], 
                             tools: Optional[List[BaseTool]] = None,
                             is_tool_generation: bool = False) -> ChatResponse:
        \"\"\"Process messages and return ChatResponse.\"\"\"
        # Implementation here
        
    def create_graph(self, tools: Optional[List[BaseTool]] = None) -> CompiledStateGraph:
        \"\"\"Create LangGraph workflow.\"\"\"
        # Build and return compiled graph
        
    async def agent_node(self, state: LangGraphState) -> Dict[str, Any]:
        \"\"\"Process agent node in the graph.\"\"\"
        # Agent processing logic
```

**Optional Streaming Hooks:**
```python
    def reset_streaming_state(self) -> None:
        \"\"\"Reset state before streaming.\"\"\"
        super().reset_streaming_state()
        # Reset custom streaming state
        
    def process_streaming_token(self, content: str) -> Optional[ChatResponse]:
        \"\"\"Process individual tokens during streaming.\"\"\"
        # Custom token routing logic (e.g., harmony channels, think tags)
        # Return ChatResponse with appropriate field routing or None to suppress
        
    def finalize_streaming(self) -> Optional[ChatResponse]:
        \"\"\"Called when streaming completes.\"\"\"
        # Handle any remaining buffered content
```

**Example Implementation:**
```python
class CustomLLMPipeline(BaseLlamaCppPipeline):
    allowed_return_types: tuple[type, ...] = (ChatResponse,)
    default_return_type: Optional[type] = ChatResponse
    
    def __init__(self, model: Model, profile: ModelProfile, **kwargs):
        super().__init__(model, profile, **kwargs)
        # Custom initialization
        
    async def process_messages(self, messages: List[Message], **kwargs) -> ChatResponse:
        # Format messages for your model
        formatted_messages = self._format_messages(messages)
        
        # Generate response using LangGraph
        graph = self.create_graph(kwargs.get('tools'))
        state = build_langgraph_state(messages, tools=kwargs.get('tools'))
        result = await graph.ainvoke(state)
        
        # Convert to ChatResponse
        return self._convert_to_response(result)
```

### 2. Embedding Pipelines

**Base Classes to Inherit From:**
- `BaseLlamaCppCore` (for GGUF embedding models)
- `BasePipelineCore` (for other backends)

**Required Configuration:**
```python
class MyEmbeddingPipeline(BaseLlamaCppCore):
    # Must return list type for embeddings
    allowed_return_types: tuple[type, ...] = (list,)
    default_return_type: Optional[type] = list
    
    def __init__(self, model: Model, profile: ModelProfile):
        # Must specify list as expected return type
        super().__init__(model, profile, expected_return_type=list)
        
    async def process_messages(self, messages: List[Message], **kwargs) -> List[List[float]]:
        \"\"\"Generate embeddings for input messages.\"\"\"
        # Extract text from messages
        texts = [extract_message_text(msg) for msg in messages]
        
        # Generate embeddings
        embeddings = await self._generate_embeddings(texts)
        
        # Validate return value
        self.validate_return_value(embeddings)
        return embeddings
```

**Key Considerations:**
- Return type must be `List[List[float]]` (list of embedding vectors)
- Handle text extraction from various message content types
- Consider batching for efficiency
- Implement text chunking for long inputs

### 3. Text-to-Image Pipelines

**Base Classes to Inherit From:**
- `BasePipelineCore` for custom implementations
- Can integrate with diffusers, DALL-E, Stable Diffusion, etc.

**Required Configuration:**
```python
class MyImagePipeline(BasePipelineCore[ChatResponse]):
    allowed_return_types: tuple[type, ...] = (ChatResponse,)
    default_return_type: Optional[type] = ChatResponse
    
    def __init__(self, model: Model, profile: ModelProfile):
        super().__init__(model, profile, expected_return_type=ChatResponse)
        self._pipeline: Optional[Any] = None  # Your image generation pipeline
        
    async def process_messages(self, messages: List[Message], **kwargs) -> ChatResponse:
        \"\"\"Generate images from text prompts.\"\"\"
        # Extract prompt from messages
        prompt = self._extract_prompt(messages)
        
        # Generate image
        image_data = await self._generate_image(prompt)
        
        # Return as ChatResponse with image content
        return ChatResponse(
            message=Message(
                role=MessageRole.ASSISTANT,
                content=[MessageContent(
                    type=MessageContentType.IMAGE,
                    image_data=image_data
                )]
            ),
            done=True
        )
        
    def create_graph(self, tools: Optional[List[BaseTool]] = None) -> CompiledStateGraph:
        \"\"\"Simple graph for image generation.\"\"\"
        graph = StateGraph(LangGraphState)
        graph.add_node("generate", self.agent_node)
        graph.add_edge(START, "generate")
        graph.add_edge("generate", END)
        return graph.compile(checkpointer=self.memory)
```

**Example with Diffusers:**
```python
from diffusers import StableDiffusionPipeline

class StableDiffusionPipe(BasePipelineCore[ChatResponse]):
    def __init__(self, model: Model, profile: ModelProfile):
        super().__init__(model, profile, expected_return_type=ChatResponse)
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            model.model,
            torch_dtype=torch.float16
        ).to("cuda")
        
    async def _generate_image(self, prompt: str) -> bytes:
        \"\"\"Generate image and return as bytes.\"\"\"
        image = self.pipeline(prompt).images[0]
        # Convert PIL image to bytes
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        return buffer.getvalue()
```

### 4. Multimodal (Vision-Language) Pipelines

**Base Classes to Inherit From:**
- `BaseLlamaCppPipeline` (for GGUF vision models)
- `BasePipelineCore` (for other backends)

**Required Features:**
```python
class MyVisionPipeline(BaseLlamaCppPipeline):
    allowed_return_types: tuple[type, ...] = (ChatResponse,)
    default_return_type: Optional[type] = ChatResponse
    
    def __init__(self, model: Model, profile: ModelProfile, **kwargs):
        super().__init__(model, profile, **kwargs)
        # Initialize vision-specific components
        
    async def process_messages(self, messages: List[Message], **kwargs) -> ChatResponse:
        \"\"\"Process messages containing text and images.\"\"\"
        # Handle both text and image content
        processed_messages = self._process_multimodal_messages(messages)
        
        # Use vision-capable model
        response = await self._generate_response(processed_messages)
        
        return response
        
    def _process_multimodal_messages(self, messages: List[Message]) -> List[Dict]:
        \"\"\"Convert messages with images to model format.\"\"\"
        formatted = []
        for msg in messages:
            content_parts = []
            for content in msg.content:
                if content.type == MessageContentType.TEXT:
                    content_parts.append({"type": "text", "text": content.text})
                elif content.type == MessageContentType.IMAGE:
                    # Handle image data (base64, file path, etc.)
                    content_parts.append({
                        "type": "image_url", 
                        "image_url": {"url": content.image_url}
                    })
            formatted.append({"role": msg.role.value, "content": content_parts})
        return formatted
```

**Image Handling:**
- Support various image formats (base64, URLs, file paths)
- Handle image preprocessing/resizing as needed
- Consider memory management for large images

### 5. Image-to-Image Pipelines

**Base Classes to Inherit From:**
- `BasePipelineCore` for custom implementations

**Key Features:**
```python
class MyImg2ImgPipeline(BasePipelineCore[ChatResponse]):
    allowed_return_types: tuple[type, ...] = (ChatResponse,)
    
    async def process_messages(self, messages: List[Message], **kwargs) -> ChatResponse:
        \"\"\"Transform input images based on prompts.\"\"\"
        # Extract source image and transformation prompt
        source_image = self._extract_source_image(messages)
        prompt = self._extract_prompt(messages)
        
        # Apply transformation
        result_image = await self._transform_image(source_image, prompt)
        
        return self._create_image_response(result_image)
```

## Advanced Features

### Streaming Support with Custom Token Processing

For pipelines that need custom token-level processing (like harmony channels or think tags):

```python
class AdvancedStreamingPipeline(BaseLlamaCppPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._reset_streaming_state()
        
    def _reset_streaming_state(self):
        \"\"\"Initialize custom streaming state.\"\"\"
        self.buffer = ""
        self.mode = "normal"
        
    def reset_streaming_state(self):
        \"\"\"Called before each streaming session.\"\"\"
        super().reset_streaming_state()
        self._reset_streaming_state()
        
    def process_streaming_token(self, content: str) -> Optional[ChatResponse]:
        \"\"\"Process each token and route to appropriate fields.\"\"\"
        self.buffer += content
        
        # Detect special markers
        if "<thinking>" in self.buffer:
            self.mode = "thinking"
            # Process and route to thinking field
            return self._create_thinking_response(content)
        elif "</thinking>" in self.buffer:
            self.mode = "normal"
            return None
            
        # Route based on current mode
        if self.mode == "thinking":
            return None  # Buffer thinking content
        else:
            return self._create_content_response(content)
```

### Circuit Breaker Configuration

For robust production deployments:

```python
circuit_config = CircuitBreakerConfig(
    enable_perplexity_guard=True,
    perplexity_threshold=10.0,
    enable_repetition_guard=True,
    max_repetitions=3,
    timeout_seconds=300,
    max_retries=2
)

pipeline = MyPipeline(model, profile, circuit_config=circuit_config)
```

### Tool Integration

For pipelines that support tool calling:

```python
async def process_messages(self, messages: List[Message], 
                         tools: Optional[List[BaseTool]] = None,
                         **kwargs) -> ChatResponse:
    \"\"\"Process with tool support.\"\"\"
    if tools:
        # Create graph with tool nodes
        graph = self.create_graph(tools)
        # Process with tool capability
    else:
        # Process without tools
        
def create_graph(self, tools: Optional[List[BaseTool]] = None) -> CompiledStateGraph:
    \"\"\"Create graph with optional tool integration.\"\"\"
    graph = StateGraph(LangGraphState)
    graph.add_node("agent", self.agent_node)
    
    if tools:
        tool_node = ToolNode(tools)
        graph.add_node("tools", tool_node)
        graph.add_conditional_edges("agent", tools_condition)
        graph.add_edge("tools", "agent")
    else:
        graph.add_edge("agent", END)
        
    graph.add_edge(START, "agent")
    return graph.compile(checkpointer=self.memory)
```

## Registration and Factory Integration

### Model Configuration

Add your model to the models configuration:

```json
{
  "id": "my-custom-model",
  "name": "My Custom Model",
  "provider": "custom",
  "model": "path/to/model",
  "pipeline": "my_custom_pipeline",
  "modality": ["text"],
  "details": {
    "gguf_file": "model.gguf",
    "context_length": 4096
  }
}
```

### Pipeline Registration

Register your pipeline in the factory:

```python
# In pipeline_factory.py
def create_pipeline(self, model_id: str, **kwargs) -> BasePipelineCore:
    model = self.get_model(model_id)
    
    # Add your pipeline mapping
    if model.pipeline == "my_custom_pipeline":
        from .pipelines.custom.my_pipeline import MyCustomPipeline
        return MyCustomPipeline(model, profile, **kwargs)
    
    # Existing pipeline mappings...
```

## Error Handling and Validation

### Input Validation

```python
def validate_inputs(self, messages: List[Message]) -> None:
    \"\"\"Validate input messages.\"\"\"
    if not messages:
        raise ValueError("At least one message is required")
        
    for msg in messages:
        if not msg.content:
            raise ValueError("Message content cannot be empty")
            
        # Validate content types for your pipeline
        for content in msg.content:
            if content.type not in self.supported_content_types:
                raise ValueError(f"Unsupported content type: {content.type}")
```

### Return Value Validation

```python
async def process_messages(self, messages: List[Message], **kwargs) -> PipeType:
    # Generate response
    response = await self._generate_response(messages)
    
    # Validate against expected return type
    self.validate_return_value(response)
    
    return response
```

## Testing Guidelines

### Unit Testing

```python
import pytest
from unittest.mock import Mock, patch

class TestMyPipeline:
    @pytest.fixture
    def pipeline(self):
        model = Mock(spec=Model)
        profile = Mock(spec=ModelProfile)
        return MyPipeline(model, profile)
        
    async def test_process_messages(self, pipeline):
        messages = [Message(
            role=MessageRole.USER,
            content=[MessageContent(type=MessageContentType.TEXT, text="Hello")]
        )]
        
        response = await pipeline.process_messages(messages)
        
        assert isinstance(response, ChatResponse)
        assert response.message.role == MessageRole.ASSISTANT
```

### Integration Testing

```python
async def test_pipeline_integration():
    \"\"\"Test with real model loading.\"\"\"
    factory = PipelineFactory(models_map)
    
    pipeline = factory.create_pipeline("my-model-id")
    
    messages = [create_test_message("Test prompt")]
    response = await pipeline.process_messages(messages)
    
    assert response is not None
```

## Performance Considerations

### Memory Management

- Use appropriate model quantization
- Implement model unloading when not in use
- Consider GPU memory constraints
- Use batch processing for efficiency

### Caching

- Pipeline instances are cached by the factory
- Implement result caching where appropriate
- Consider memory vs compute tradeoffs

### Async/Await Patterns

- All processing methods should be async
- Use proper async context managers
- Handle cancellation gracefully

## Best Practices

1. **Type Safety**: Use proper type hints and validate return types
2. **Error Handling**: Implement comprehensive error handling
3. **Logging**: Use structured logging for debugging
4. **Configuration**: Make pipelines configurable via ModelProfile
5. **Testing**: Write comprehensive unit and integration tests
6. **Documentation**: Document model-specific requirements and capabilities
7. **Resource Management**: Properly manage GPU memory and model lifecycle
8. **Streaming**: Implement streaming support where beneficial
9. **Tool Integration**: Support tool calling where applicable
10. **Validation**: Validate inputs and outputs thoroughly

## Example Complete Implementation

See the existing pipelines for complete examples:
- **Text Generation**: `openai_gpt_oss.py`, `qwen3moe.py`
- **Embeddings**: `qwen3emb.py`
- **Vision Language**: `qwen25_vl.py`
- **Image Generation**: `flux.py`

Each demonstrates the patterns and best practices outlined in this guide.