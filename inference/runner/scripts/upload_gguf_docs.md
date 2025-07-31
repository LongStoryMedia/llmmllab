# GGUF Model Upload Utility

## Overview

The `upload_gguf.py` script is a comprehensive utility for uploading GGUF (GPT-Generated Unified Format) quantized model files to [Hugging Face Hub](https://huggingface.co). This tool is particularly valuable for sharing optimized models with the community after using tools like [download_and_quantize.py](download_and_quantize_docs.md) to create them.

This script handles authentication, repository management, file uploads, and automatic generation of model cards and metadata—everything needed to share your quantized models efficiently.

## Key Features

- **HF Authentication**: Multiple authentication methods (direct token, token file, or pre-existing login)
- **Repository Management**: Create new repositories or update existing ones
- **Metadata Generation**: Automatically extract and organize model information
- **Model Card Creation**: Generate professional README files with usage instructions
- **Split File Support**: Handle models split into multiple parts
- **Custom Templates**: Use your own model card templates
- **Private Repositories**: Option to create private repositories for your models

## Requirements

- Python 3.6+
- Hugging Face Hub library (`huggingface_hub`)
- Valid Hugging Face account and access token (for uploads)

## Installation

No special installation is required beyond the dependencies. Make sure you have the Hugging Face Hub library installed:

```bash
pip install huggingface_hub
```

## Usage

### Basic Command Structure

```bash
python upload_gguf.py --files model.gguf --repo-name my-quantized-model
```

### Command Line Arguments

| Argument                | Description                                | Default                  |
| ----------------------- | ------------------------------------------ | ------------------------ |
| `--files`               | Path(s) to GGUF model file(s) to upload    | (Required)               |
| `--repo-name`           | Name of the repository to create/update    | (Required)               |
| `--organization`        | Organization name for the repository       | None (uses your account) |
| `--private`             | Make the repository private                | `False` (public)         |
| `--description`         | Description of the model                   | None                     |
| `--model-name`          | Name for the model                         | Repository name          |
| `--token`               | Hugging Face token                         | None (uses stored token) |
| `--token-file`          | Path to file containing Hugging Face token | None                     |
| `--model-card-template` | Path to a custom model card template       | Default template         |

## Authentication

The script supports three authentication methods (in order of priority):

1. Direct token via `--token` argument
2. Token file via `--token-file` argument
3. Pre-existing token in the default location (from `huggingface-cli login`)

If no authentication is available, the script will exit with an error.

## Model Card Templates

The script comes with a built-in default template but also supports custom templates. Custom templates can include variables in the format `{{variable_name}}` that will be replaced with actual values.

Available template variables:

- `{{name}}`: Model name
- `{{description}}`: Model description
- `{{format}}`: Model format (GGUF)
- `{{quantization}}`: Quantization level (e.g., q4_k_m)
- `{{file.name}}`: Filename
- `{{file.size_mb}}`: File size in MB
- `{{file.md5_hash}}`: MD5 hash of the file
- `{{upload_date}}`: Date of upload

See the [model_card_template.md](model_card_template.md) file for a complete example.

## Integration with Other Utilities

This script is designed to be part of a larger model management workflow:

- Used after [download_and_quantize.py](download_and_quantize_docs.md) to share the resulting GGUF files
- Works with [download.py](download_docs.md) in a pipeline to download, process, and upload models
- Combined with both in the end-to-end workflow of [process_and_upload_model.py](process_and_upload_model_docs.md)

## Examples

### Upload a Single GGUF File

```bash
python upload_gguf.py --files ./models/llama2-7b-q4_k_m.gguf --repo-name llama2-7b-quantized --description "Quantized version of Llama 2 7B"
```

### Upload Split Model Files

```bash
python upload_gguf.py --files ./models/mistral-7b-q4_0-00001-of-00002.gguf ./models/mistral-7b-q4_0-00002-of-00002.gguf --repo-name mistral-7b-quantized
```

### Use Custom Model Card Template

```bash
python upload_gguf.py --files ./models/llama2-7b-q4_k_m.gguf --repo-name llama2-7b-quantized --model-card-template ./my_template.md
```

### Create a Private Repository

```bash
python upload_gguf.py --files ./models/proprietary-model-q8_0.gguf --repo-name proprietary-model-quantized --private
```

## Notes

- The script automatically detects quantization level from the filename when possible
- For large files, uploads may take some time depending on your internet connection
- Consider using the Hugging Face CLI to log in permanently if you frequently upload models
