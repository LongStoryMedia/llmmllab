# GGUF Model Management Scripts

This directory contains a suite of scripts designed to provide a complete workflow for managing GGUF (GPT-Generated Unified Format) quantized models. These tools allow you to download, quantize, and share models with the community through Hugging Face Hub.

## Available Scripts

### Core Scripts

| Script                                                     | Description                                                    | Documentation                                     |
| ---------------------------------------------------------- | -------------------------------------------------------------- | ------------------------------------------------- |
| [download.py](download.py)                                 | Download models from Hugging Face Hub                          | [Documentation](download_docs.md)                 |
| [download_and_quantize.py](download_and_quantize.py)       | Download and convert models to GGUF format                     | [Documentation](download_and_quantize_docs.md)    |
| [upload_gguf.py](upload_gguf.py)                           | Upload GGUF models to Hugging Face Hub                         | [Documentation](upload_gguf_docs.md)              |
| [process_and_upload_model.py](process_and_upload_model.py) | End-to-end workflow for downloading, quantizing, and uploading | [Documentation](process_and_upload_model_docs.md) |

### Supporting Files

| File                                             | Description                                        |
| ------------------------------------------------ | -------------------------------------------------- |
| [model_card_template.md](model_card_template.md) | Template for generating model cards when uploading |

## Workflow Overview

These scripts are designed to work together as part of a comprehensive model management workflow:

1. **Download Models**: Use `download.py` to obtain models from Hugging Face Hub
2. **Quantize Models**: Use `download_and_quantize.py` to convert models to GGUF format with optional quantization
3. **Share Models**: Use `upload_gguf.py` to share your quantized models on Hugging Face Hub
4. **All-in-One**: Use `process_and_upload_model.py` for a seamless end-to-end experience

## Getting Started

For simple use cases, the end-to-end script is recommended:

```bash
python process_and_upload_model.py --model-id meta-llama/Llama-2-7b-chat-hf --quantize q4_k_m
```

This will download Llama 2, convert it to GGUF with Q4_K_M quantization, and upload it to Hugging Face Hub.

For more complex workflows or finer control, use the individual scripts as needed:

```bash
# Step 1: Download a model
python download.py mistralai/Mistral-7B-v0.1 --dir ./models/mistral

# Step 2: Convert and quantize
python download_and_quantize.py --skip-download --dir ./models/mistral \
  --quantize q4_k_m --output-dir ./gguf-models

# Step 3: Upload the model
python upload_gguf.py --files ./gguf-models/mistral-7B-v0.1_q4_k_m.gguf \
  --repo-name mistral-7b-q4km
```

## Requirements

- Python 3.6+
- Hugging Face Hub library: `pip install huggingface_hub`
- Access to [llama.cpp](https://github.com/ggerganov/llama.cpp) tools (for quantization)
- Valid Hugging Face account and access token (for uploads)

## Advanced Usage

For detailed usage options and examples, refer to the individual documentation files linked in the table above.
