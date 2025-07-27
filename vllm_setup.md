# vLLM Server Setup

This document explains how to set up and configure the vLLM server for high-performance inference.

## Installation

```bash
# Install vLLM with CUDA support
pip install vllm

# For development mode:
pip install --editable ".[all]"
```

## Hardware Requirements

- NVIDIA GPU with compute capability 7.0+ (Volta, Turing, Ampere, Ada, or Hopper architecture)
- CUDA 11.8+ and cuDNN 8.7.0+ 
- GPU memory depends on model size and configuration:
  - For 7B models: At least 12GB VRAM
  - For 30B MoE models: At least 24GB VRAM
  - For 70B models: Multiple GPUs recommended

## Quantization Options

vLLM supports several quantization methods:

- **AWQ**: High-quality 4-bit quantization
- **SqueezeLLM**: 4-bit NF4 quantization 
- **GPTQ**: Standard 4-bit quantization
- **GGUF** models: Direct loading through llama-cpp backend

## Configuration

For RTX 3090 (24GB) and RTX 3060 (12GB) setup:

```python
# For tensor parallelism across both GPUs
vllm_engine = AsyncLLMEngine.from_engine_args(
    engine_args=EngineArgs(
        model="Qwen/Qwen3-30B-A3B",
        tensor_parallel_size=2,  # Use both GPUs 
        gpu_memory_utilization=0.85,  # Keep some memory free for safety
        quantization="awq",  # Choose quantization method
        max_model_len=8192,  # Maximum context length
    )
)

# For running just on 3090 with a large model
vllm_engine = AsyncLLMEngine.from_engine_args(
    engine_args=EngineArgs(
        model="Qwen/Qwen3-30B-A3B",
        tensor_parallel_size=1, 
        gpu_memory_utilization=0.9,
        quantization="awq",
        max_model_len=4096,
    )
)

# For smaller models like Mixtral-8x7B on single GPU
vllm_engine = AsyncLLMEngine.from_engine_args(
    engine_args=EngineArgs(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        dtype="bfloat16",  # Can use bf16 for Mixtral
        max_model_len=32768,  # Mixtral supports longer contexts
    )
)
```

## Streaming Configuration

For streaming responses:

```python
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=1024,
    presence_penalty=0.1,
)

# Generate with streaming
async for output in vllm_engine.generate(prompt, sampling_params, stream=True):
    # Process streaming output
    print(output.outputs[0].text)
```

## Persistent Server Mode

Run vLLM as a persistent server:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-30B-A3B \
    --tensor-parallel-size 2 \
    --quantization awq \
    --port 8000
```

This exposes an OpenAI-compatible API at http://localhost:8000/v1.

## GGUF Integration

For GGUF models, specify the model path:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model /models/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf \
    --max-model-len 8192 \
    --quantization gguf \
    --port 8000
```
