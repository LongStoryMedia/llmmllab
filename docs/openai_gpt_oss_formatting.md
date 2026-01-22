# OpenAI GPT OSS 20B Output Formatting Guide

## Overview

The OpenAI GPT OSS 20B model requires a specific output format called the **harmony response format**. This is not optional - the model was trained specifically on this format and will not function correctly without it.

## Key Requirements

### 1. Harmony Package Installation
```bash
pip install openai-harmony
```

### 2. Harmony Format Structure

The harmony format uses special tokens and channels:

```
<|start|>system<|message|>...content...<|end|>
<|start|>user<|message|>...content...<|end|>
<|start|>assistant|>channel<|message|>...content...<|end|>
```

### 3. Valid Channels

The model supports multiple output channels:

- **analysis** - Chain of thought reasoning
- **commentary** - Tool calling and function descriptions  
- **final** - Final response to the user

### 4. System Prompt Requirements

Your system prompt must include:

```
Reasoning: [low|medium|high]

# Valid channels: analysis, commentary, final. Channel must be included for every message.
# Use 'analysis' channel for chain-of-thought reasoning
# Use 'commentary' channel for tool calls and function descriptions  
# Use 'final' channel for your response to the user
```

## Implementation in LLM ML Lab

### Updated Pipeline Features

The `OpenAiGptOssPipe` has been updated to:

1. **Automatic Harmony Formatting**: Messages are automatically converted to harmony format
2. **Validation**: Checks for harmony package availability and correct model type
3. **Enhanced System Prompt**: Includes proper channel specifications
4. **Error Handling**: Graceful fallback with clear error messages

### Example Usage

```python
# The pipeline handles harmony formatting automatically
from runner.pipelines.txt2txt.openai_gpt_oss import OpenAiGptOssPipe

# Create pipeline instance
pipeline = OpenAiGptOssPipe(model, profile)

# Process messages (harmony formatting applied automatically)
response = await pipeline.process_messages(messages)
```

### Message Flow

1. **Input**: Standard Message objects with role and content
2. **Conversion**: Automatic conversion to harmony format using openai-harmony package
3. **Processing**: Model receives properly formatted harmony prompt
4. **Output**: Standard ChatResponse object

## Configuration Requirements

### Model Profile Updates

Ensure your model profile includes:

```yaml
system_prompt: |
  You are ChatGPT, a large language model trained by OpenAI.
  Knowledge cutoff: 2024-06
  Current date: 2025-09-16
  
  Reasoning: medium
  
  # Valid channels: analysis, commentary, final. Channel must be included for every message.
  # Use 'analysis' channel for chain-of-thought reasoning
  # Use 'commentary' channel for tool calls and function descriptions  
  # Use 'final' channel for your response to the user
```

### Stop Tokens

Use harmony-specific stop tokens:
```yaml
stop: ["<|end|>", "<|start|>"]
```

## Troubleshooting

### Common Issues

1. **Missing harmony package**: Install with `pip install openai-harmony`
2. **Incorrect model**: Ensure you're using a GPT OSS model (.gguf file)
3. **Wrong format**: The model requires harmony format - standard chat templates won't work

### Error Messages

- **"Harmony format required"**: Install openai-harmony package
- **"Garbled output detected"**: Check model compatibility and formatting
- **"Model path doesn't appear to be GPT OSS"**: Verify you're using the correct model

## Performance Notes

### Context Windows

GPT OSS 20B supports large context windows:
- Default: 131,072 tokens
- Recommended: Use full context for better reasoning

### Reasoning Levels

Configure reasoning level in system prompt:
- **low**: Fast responses for general dialogue
- **medium**: Balanced speed and detail (recommended)
- **high**: Deep and detailed analysis

### Memory Requirements

The model requires significant resources:
- **RAM**: ~16GB minimum for 20B model
- **VRAM**: Depends on quantization and layer configuration
- **Storage**: Model file size varies by quantization

## Reference Links

- [OpenAI GPT OSS Models](https://openai.com/open-models)
- [Harmony Format Documentation](https://github.com/openai/harmony)
- [Model on Hugging Face](https://huggingface.co/openai/gpt-oss-20b)
- [GPT OSS Cookbook](https://cookbook.openai.com/topic/gpt-oss)

## Migration from Standard Format

If migrating from a standard chat model pipeline:

1. Install harmony package
2. Update system prompt with channel specifications
3. Use OpenAiGptOssPipe instead of generic pipeline
4. Test with simple queries first
5. Monitor for garbled output (indicates formatting issues)

The harmony format is essential for GPT OSS models to function correctly. Without it, you may experience poor quality output, repetitive responses, or complete failure to generate coherent text.