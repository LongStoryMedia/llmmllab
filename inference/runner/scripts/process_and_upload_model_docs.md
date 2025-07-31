# End-to-End Model Processing and Upload Utility

## Overview

The `process_and_upload_model.py` script provides a complete, end-to-end solution for downloading, quantizing, and uploading GGUF models to Hugging Face Hub. It combines the functionality of [download_and_quantize.py](download_and_quantize_docs.md) and [upload_gguf.py](upload_gguf_docs.md) into a single workflow, simplifying the entire process from obtaining a model to sharing it with the community.

This script is ideal for automating the entire model processing pipeline with minimal manual steps, making it perfect for batch processing or regularly updating quantized models.

## Key Features

- **One-Step Processing**: Complete pipeline from model download to upload
- **Smart Defaults**: Automatically derives sensible default values
- **Quantization Options**: Support for various quantization methods
- **Flexible Output**: Control where files are saved and uploaded
- **HF Integration**: Full Hugging Face Hub repository management
- **Cleanup Options**: Automatically manage disk usage

## Requirements

- Python 3.6+
- Hugging Face Hub library (`huggingface_hub`)
- Access to [llama.cpp](https://github.com/ggerganov/llama.cpp) tools (for quantization)
- Valid Hugging Face account and access token (for uploads)

## Installation

No special installation is required beyond the dependencies and the companion scripts. Make sure you have:

1. The Hugging Face Hub library: `pip install huggingface_hub`
2. Both [download_and_quantize.py](download_and_quantize_docs.md) and [upload_gguf.py](upload_gguf_docs.md) available in the same directory
3. The [model_card_template.md](model_card_template.md) file (optional, for custom model cards)

## Usage

### Basic Command Structure

```bash
python process_and_upload_model.py --model-id meta-llama/Llama-2-7b-chat-hf --quantize q4_k_m
```

### Command Line Arguments

| Argument             | Description                                  | Default                            |
| -------------------- | -------------------------------------------- | ---------------------------------- |
| `--model-id`         | The model ID on Hugging Face Hub             | (Required)                         |
| `--output-dir`       | Directory to save the output GGUF file(s)    | `./[model_name]_gguf`              |
| `--output-file-name` | Custom name for the output file              | Model name                         |
| `--quantize`         | Quantization method to apply                 | `q4_k_m`                           |
| `--upload-repo`      | Repository name on Hugging Face Hub          | `[model_name]-[quantization]-gguf` |
| `--organization`     | Organization name for the repository         | None (uses your account)           |
| `--private`          | Make the repository private                  | `False` (public)                   |
| `--description`      | Description of the model                     | None                               |
| `--token`            | Hugging Face token                           | None (uses stored token)           |
| `--token-file`       | Path to file containing Hugging Face token   | None                               |
| `--cleanup`          | Delete original model files after conversion | `False`                            |
| `--no-upload`        | Skip uploading to Hugging Face Hub           | `False` (will upload)              |

## Workflow Steps

The script performs the following steps in sequence:

1. **Download and Quantize**:

   - Downloads the specified model from Hugging Face
   - Converts it to GGUF format
   - Applies the requested quantization method
   - Saves the resulting file(s) to the output directory

2. **Upload to Hugging Face** (unless `--no-upload` is specified):
   - Creates or updates a repository on Hugging Face Hub
   - Uploads the GGUF file(s)
   - Generates and uploads metadata and a model card
   - Returns the URL to the published model

## Related Scripts

This utility combines and builds upon these individual tools:

- [download.py](download_docs.md): Handles downloading models from Hugging Face Hub
- [download_and_quantize.py](download_and_quantize_docs.md): Converts and quantizes models to GGUF format
- [upload_gguf.py](upload_gguf_docs.md): Uploads GGUF files to Hugging Face Hub

## Examples

### Basic Usage with Default Settings

```bash
python process_and_upload_model.py --model-id meta-llama/Llama-2-7b-chat-hf
```

This will download Llama 2, convert it to GGUF, quantize it with the default method (q4_k_m), and upload it to a repository named "Llama-2-7b-chat-hf-q4_k_m-gguf".

### Custom Output and Repository Names

```bash
python process_and_upload_model.py --model-id mistralai/Mistral-7B-v0.1 \
  --output-dir ./models/mistral --output-file-name mistral-optimized \
  --upload-repo mistral-7b-quantized
```

### Quantize, Clean Up, but Skip Upload

```bash
python process_and_upload_model.py --model-id meta-llama/Llama-2-13b \
  --quantize q8_0 --cleanup --no-upload
```

### Upload to an Organization with a Description

```bash
python process_and_upload_model.py --model-id meta-llama/Llama-2-7b-chat-hf \
  --organization my-org --description "Quantized Llama 2 model for efficient inference" \
  --private
```

## Notes

- Processing large models requires significant disk space for both the original and converted models
- Using the `--cleanup` flag can help manage disk usage but removes the original model files
- The script automatically looks for and uses the model card template if available
- For very large models, quantization can take a significant amount of time and memory
