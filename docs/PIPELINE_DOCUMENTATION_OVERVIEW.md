# LLM ML Lab Pipeline Documentation Suite

## Documentation Overview

This comprehensive documentation suite covers all aspects of implementing and working with pipelines in the LLM ML Lab runner module. The documentation is organized into three main documents:

### 1. [Pipeline Implementation Guide](./PIPELINE_IMPLEMENTATION_GUIDE.md)

**Purpose**: Comprehensive guide for implementing custom pipelines  
**Audience**: Developers building new pipeline types  
**Covers**:

- Architecture overview and design patterns
- Step-by-step implementation for each pipeline type
- Advanced features like streaming and tool integration
- Testing guidelines and best practices
- Complete example implementations

### 2. [Pipeline API Reference](./PIPELINE_API_REFERENCE.md)

**Purpose**: Complete API documentation for all pipeline interfaces  
**Audience**: Developers working with existing pipelines  
**Covers**:

- All public classes, methods, and interfaces
- Data models and type definitions
- Configuration options and parameters
- Error handling and exceptions
- Utility functions and helpers

### 3. [Runner Architecture Overhaul](./RUNNER_ARCHITECTURE_OVERHAUL.md)

**Purpose**: Documentation of the recent architecture improvements  
**Audience**: Maintainers and contributors understanding the new features  
**Covers**:

- New streaming architecture with pipeline-specific post-processing
- Harmony channel and think tag routing
- EventStreamProcessor delegation
- API compatibility and migration notes

## Quick Start Guide

### For New Pipeline Developers

1. **Start with**: [Pipeline Implementation Guide](./PIPELINE_IMPLEMENTATION_GUIDE.md)
2. **Reference**: [Pipeline API Reference](./PIPELINE_API_REFERENCE.md) for specific interfaces
3. **Examples**: Study existing implementations in `inference/runner/pipelines/`

### For Existing Pipeline Users

1. **API Reference**: [Pipeline API Reference](./PIPELINE_API_REFERENCE.md) for complete interface docs
2. **New Features**: [Runner Architecture Overhaul](./RUNNER_ARCHITECTURE_OVERHAUL.md) for recent improvements

### For Contributors and Maintainers

1. **Architecture**: [Runner Architecture Overhaul](./RUNNER_ARCHITECTURE_OVERHAUL.md) for recent changes
2. **Implementation**: [Pipeline Implementation Guide](./PIPELINE_IMPLEMENTATION_GUIDE.md) for development patterns
3. **API**: [Pipeline API Reference](./PIPELINE_API_REFERENCE.md) for complete interface coverage

## Pipeline Types Supported

### Text-to-Text Pipelines
- **Purpose**: Standard language model interactions (chat, completion, reasoning)
- **Base Classes**: `BaseLlamaCppPipeline`, `ChatPipeline`
- **Return Type**: `ChatResponse`
- **Examples**: OpenAI GPT OSS, Qwen3MoE, LlamaChat
- **Features**: Streaming, tool calling, harmony channels, think tags

### Embedding Pipelines
- **Purpose**: Generate vector embeddings from text
- **Base Classes**: `BaseLlamaCppCore`, `EmbeddingPipeline`
- **Return Type**: `List[List[float]]`
- **Examples**: Qwen3 Embedding, Nomic Embed
- **Features**: Batch processing, text chunking, similarity search

### Text-to-Image Pipelines
- **Purpose**: Generate images from text prompts
- **Base Classes**: `BasePipelineCore`, `ImageGenerationPipeline`
- **Return Type**: `ChatResponse` (with image content)
- **Examples**: Flux, Stable Diffusion XL, SD3
- **Features**: Multiple formats, quality settings, style control

### Image-to-Image Pipelines
- **Purpose**: Transform or edit existing images
- **Base Classes**: `BasePipelineCore`
- **Return Type**: `ChatResponse` (with image content)
- **Examples**: Flux img2img, SDXL img2img
- **Features**: Style transfer, inpainting, upscaling

### Multimodal (Vision-Language) Pipelines
- **Purpose**: Process text and images together
- **Base Classes**: `BaseLlamaCppPipeline`
- **Return Type**: `ChatResponse`
- **Examples**: Qwen2.5-VL, GLM-4V
- **Features**: Image understanding, VQA, multimodal reasoning

## Architecture Highlights

### New Streaming Architecture
- **Pipeline-Specific Processing**: Each pipeline can define custom token routing
- **Real-Time Processing**: Token-level processing during streaming (not post-processing)
- **Field Routing**: Content automatically routed to appropriate response fields
- **Backward Compatibility**: Existing API unchanged

### Advanced Features
- **Circuit Breakers**: Fault tolerance with perplexity and repetition guards
- **Timeout Protection**: Configurable timeouts with automatic retry
- **Memory Management**: GPU memory optimization and automatic cleanup
- **Caching**: Intelligent pipeline caching with lifecycle management
- **Tool Integration**: Native support for function calling and tools

### Production Features
- **Error Handling**: Comprehensive error handling and validation
- **Logging**: Structured logging for debugging and monitoring
- **Metrics**: Performance metrics and telemetry
- **Configuration**: Environment-based configuration override
- **Testing**: Unit and integration testing frameworks

## Key Concepts

### Pipeline Lifecycle
1. **Registration**: Model and pipeline configuration
2. **Factory Creation**: Pipeline instantiation via factory
3. **Caching**: Intelligent caching based on usage patterns
4. **Processing**: Message processing with streaming support
5. **Cleanup**: Automatic resource cleanup and memory management

### Streaming Flow
1. **Event Processing**: Raw streaming events from LLM
2. **Pipeline Delegation**: EventStreamProcessor delegates to pipeline
3. **Token Processing**: Pipeline processes tokens with custom logic
4. **Field Routing**: Content routed to appropriate response fields
5. **Finalization**: Final processing when streaming completes

### Type Safety
- **Generic Base Classes**: Type-safe pipeline implementations
- **Return Type Validation**: Automatic validation of return types
- **Input Validation**: Comprehensive input validation
- **Error Handling**: Typed exceptions and error responses

## Integration Points

### Server Integration
- **FastAPI Endpoints**: REST API endpoints for pipeline operations
- **WebSocket Streaming**: Real-time streaming via WebSocket
- **gRPC Services**: High-performance gRPC interface
- **Background Processing**: Async processing with RabbitMQ

### Model Integration
- **GGUF Support**: Native GGUF model support via llama.cpp
- **HuggingFace**: Integration with HuggingFace transformers
- **Diffusers**: Image generation via diffusers library
- **Custom Backends**: Support for custom model backends

### Tool Integration
- **LangChain Tools**: Native LangChain tool support
- **Custom Tools**: Support for custom tool implementations
- **Function Calling**: Automatic function calling capabilities
- **Tool Routing**: Intelligent tool selection and routing

## Development Workflow

### Adding New Pipeline Types
1. Identify base class (text, image, embedding, multimodal)
2. Implement required abstract methods
3. Add model configuration and registration
4. Implement testing suite
5. Update documentation

### Extending Existing Pipelines
1. Override specific methods for customization
2. Add new streaming hooks for token processing
3. Implement new tool integrations
4. Add configuration options

### Performance Optimization
1. Profile pipeline performance
2. Optimize GPU memory usage
3. Implement batching strategies
4. Add caching where appropriate

## Best Practices Summary

### Development
- Follow type hints and validation patterns
- Implement comprehensive error handling
- Use async/await patterns consistently
- Write testable, modular code

### Production
- Configure circuit breakers appropriately
- Monitor memory usage and performance
- Implement proper logging and metrics
- Use caching for better performance

### Maintenance
- Keep documentation updated
- Run comprehensive test suites
- Monitor for deprecation warnings
- Regular dependency updates

## Getting Help

### Common Issues
- **Model Loading Failures**: Check GGUF file paths and permissions
- **Memory Issues**: Adjust GPU layers and batch sizes
- **Timeout Errors**: Configure appropriate timeout strategies
- **Type Errors**: Ensure proper return type configuration

### Debugging
- Enable verbose logging for detailed information
- Use circuit breaker metrics for fault diagnosis
- Monitor GPU memory usage for resource issues
- Check pipeline cache for performance problems

### Community
- Refer to existing pipeline implementations as examples
- Use the test suites as reference implementations
- Check the API reference for complete interface documentation
- Review the implementation guide for architectural patterns

This documentation suite provides everything needed to understand, implement, and work with pipelines in the LLM ML Lab runner module. Whether you're building new pipeline types, using existing ones, or maintaining the system, these documents provide comprehensive coverage of all aspects of the pipeline architecture.