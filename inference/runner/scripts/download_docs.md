# Hugging Face Model Download Utility

## Overview

The `download.py` script provides a clean and simple utility for downloading models from [Hugging Face Hub](https://huggingface.co). This script is designed as a building block that can be used independently or as part of a larger workflow, such as the more comprehensive [download_and_quantize.py](download_and_quantize_docs.md) script.

It leverages the Hugging Face `snapshot_download` function to efficiently download model files and handles directory management automatically.

## Key Features

- **Simple API**: Focused on just one task: downloading models from Hugging Face Hub
- **Symlink Support**: Option to use symlinks instead of copying files to save disk space
- **Flexible Targeting**: Download specific model revisions/versions
- **Programmable Interface**: Can be imported and used as a module in other scripts
- **Command-Line Interface**: Can be run directly as a standalone utility

## Requirements

- Python 3.6+
- Hugging Face Hub library (`huggingface_hub`)

## Installation

No special installation is required beyond the dependencies. Make sure you have the Hugging Face Hub library installed:

```bash
pip install huggingface_hub
```

## Usage

### As a Standalone Script

```bash
python download.py meta-llama/Llama-2-7b-chat-hf --dir ./models/llama-2 --revision main
```

#### Command Line Arguments

| Argument         | Description                                                      | Default                            |
| ---------------- | ---------------------------------------------------------------- | ---------------------------------- |
| `model_id`       | The model ID on Hugging Face Hub (e.g., 'lmsys/vicuna-13b-v1.5') | (Required)                         |
| `--dir`          | Local directory to save the model to                             | Model name extracted from model_id |
| `--use-symlinks` | Use symlinks for downloaded files to save space                  | `False`                            |
| `--revision`     | Revision/version to download from Hugging Face                   | `"main"`                           |

### As a Module in Other Scripts

```python
from download import download_model

# Download a model
download_model(
    model_id="meta-llama/Llama-2-7b-chat-hf",
    local_dir="./models/llama-2",
    use_symlinks=True,
    revision="main"
)
```

## Integration with Other Utilities

This script is designed to be part of a larger model management workflow:

- Used by [download_and_quantize.py](download_and_quantize_docs.md) to obtain models before conversion to GGUF format
- Can be used with [upload_gguf.py](upload_gguf_docs.md) in a pipeline to download, process, and upload models
- Part of the end-to-end workflow in [process_and_upload_model.py](process_and_upload_model_docs.md)

## Examples

### Download a Model Using the Default Directory

```bash
python download.py lmsys/vicuna-13b-v1.5
```

This will download the model to a directory named `vicuna-13b-v1.5` in the current path.

### Download a Model with a Custom Directory and Using Symlinks

```bash
python download.py meta-llama/Llama-2-7b-chat-hf --dir ./models/llama2 --use-symlinks
```

This will download the model to `./models/llama2` and use symlinks to save disk space.

### Download a Specific Model Revision

```bash
python download.py mistralai/Mistral-7B-v0.1 --revision v0.1 --dir ./models/mistral
```

This will download the specific "v0.1" revision of the Mistral model.

## Notes

- For very large models, consider using the `--use-symlinks` option to save disk space
- The script will handle creating the target directory if it doesn't exist
- This utility is focused on model downloading only - for converting models to GGUF format, see [download_and_quantize.py](download_and_quantize_docs.md)
