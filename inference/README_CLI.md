# llmmll - LLMML Lab CLI Tool

A unified command-line interface for managing machine learning models - download, quantize, and share models with the community easily.

## Installation

### Local Development Installation

```bash
# From the inference directory
pip install -e .
```

### Regular Installation

```bash
pip install .
```

## Usage

The `llmmll` tool provides a unified interface for all model management tasks:

```
llmmll <command> [options]
```

Available commands:

- `download`: Download models from Hugging Face Hub
- `quantize`: Convert and quantize models to GGUF format
- `upload`: Upload GGUF models to Hugging Face Hub
- `process-upload`: End-to-end workflow combining download, quantize and upload
- `help`: Show help for a specific command

### Examples

#### Download a Model

```bash
llmmll download meta-llama/Llama-2-7b-chat-hf --dir ./models/llama2
```

#### Quantize a Model

```bash
llmmll quantize --model-id mistralai/Mistral-7B-v0.1 --quantize q4_k_m
```

#### Upload a Model

```bash
llmmll upload --files ./models/model.gguf --repo-name my-quantized-model
```

#### Complete Workflow

```bash
llmmll process-upload --model-id meta-llama/Llama-2-7b-chat-hf --quantize q4_k_m
```

### Getting Help

To see available options for each command:

```bash
llmmll help download
llmmll help quantize
llmmll help upload
llmmll help process-upload
```

Or use the standard help flag:

```bash
llmmll download --help
```

## Command Details

### `download`

Downloads models from Hugging Face Hub.

```bash
llmmll download <model_id> [options]
```

For full documentation, see [download_docs.md](scripts/download_docs.md).

### `quantize`

Downloads (if needed) and converts models to GGUF format with optional quantization.

```bash
llmmll quantize [options]
```

For full documentation, see [download_and_quantize_docs.md](scripts/download_and_quantize_docs.md).

### `upload`

Uploads GGUF model files to Hugging Face Hub.

```bash
llmmll upload --files model.gguf --repo-name repo-name [options]
```

For full documentation, see [upload_gguf_docs.md](scripts/upload_gguf_docs.md).

### `process-upload`

End-to-end workflow for downloading, quantizing, and uploading models.

```bash
llmmll process-upload --model-id <model_id> [options]
```

For full documentation, see [process_and_upload_model_docs.md](scripts/process_and_upload_model_docs.md).
