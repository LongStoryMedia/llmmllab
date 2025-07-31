# GGUF Model Processing Utility

## Overview

The `download_and_quantize.py` script is a comprehensive utility for processing machine learning models for use with [llama.cpp](https://github.com/ggerganov/llama.cpp). It offers an end-to-end solution for obtaining models from Hugging Face, converting them to the GGUF format, applying various quantization methods, splitting large models, and managing disk space through cleanup operations.

This utility is designed to run in a Kubernetes pod environment with llama.cpp installed at `/llama.cpp`, making it suitable for automated model processing pipelines and deployments.

## Key Features

- **Model Acquisition**: Download models directly from Hugging Face Hub using [download.py](download_docs.md)
- **Local Model Support**: Work with pre-existing model files without re-downloading
- **Format Conversion**: Transform Hugging Face models to the GGUF format
- **Quantization**: Apply various compression methods to reduce model size
- **Model Splitting**: Divide large GGUF files into smaller chunks for easier management
- **Disk Space Management**: Clean up intermediate files to conserve storage

## Related Scripts

This script is part of a larger model management ecosystem:

- [download.py](download_docs.md): Used internally for downloading models from Hugging Face
- [upload_gguf.py](upload_gguf_docs.md): Complement to this script for uploading processed models
- [process_and_upload_model.py](process_and_upload_model_docs.md): End-to-end workflow combining this script with upload capabilities

## Requirements

- Python 3.6+
- Hugging Face Hub library (`huggingface_hub`)
- llama.cpp installed in one of these locations:
  - `/llama.cpp` (default in k8s pod)
  - `/app/llama.cpp` (alternative in container)
  - In the parent directory of the script
  - In the user's home directory (`~/llama.cpp`)

Required llama.cpp tools:

- `convert_hf_to_gguf.py` (may be in main directory or `examples` subdirectory)
- `quantize` (may be in main directory, `bin` or `build/bin` subdirectory)
- `gguf-split` (may be in main directory, `bin` or `build/bin` subdirectory)

Additional requirements for LoRA merging:

- `transformers` library (automatically installed if missing)
- `peft` library (automatically installed if missing)
- `accelerate` library (automatically installed if missing)
- `torch` library (PyTorch, automatically installed if missing)

Note: The script will attempt to automatically install any missing LoRA-related dependencies when the `--lora-model-id` option is used.

## Installation

No special installation is required beyond the dependencies. The script will automatically locate the necessary llama.cpp tools if they are available in any of the expected locations.

## Usage

### Basic Command Structure

```bash
python download_and_quantize.py [MODEL SOURCE OPTIONS] [PROCESSING OPTIONS]
```

### Model Source Options (Mutually Exclusive)

You must choose one of the following approaches:

1. **Download a model from Hugging Face Hub**:

   ```bash
   python download_and_quantize.py --model-id MODEL_ID [OPTIONS]
   ```

2. **Use an existing model**:
   ```bash
   python download_and_quantize.py --skip-download --dir MODEL_DIRECTORY [OPTIONS]
   ```

### Common Options

| Option              | Description                              | Default                                                  |
| ------------------- | ---------------------------------------- | -------------------------------------------------------- |
| `--dir PATH`        | Local directory to save/read model files | Model name if downloading, required if `--skip-download` |
| `--revision REV`    | Revision to download from Hugging Face   | `"main"`                                                 |
| `--use-symlinks`    | Use symlinks for downloaded files        | `False`                                                  |
| `--model-name NAME` | Custom name for the output files         | Model name derived from model ID or directory            |
| `--cleanup`         | Delete original files after conversion   | `False`                                                  |

### LoRA Merging Options

| Option                | Description                                                    | Default  |
| --------------------- | -------------------------------------------------------------- | -------- |
| `--lora-model-id ID`  | LoRA adapter ID from Hugging Face to merge with the base model | None     |
| `--lora-revision REV` | Revision of the LoRA model to download                         | `"main"` |

### Quantization Options

Specify `--quantize TYPE` to apply quantization. Available methods:

| Type     | Description                          | Precision                       |
| -------- | ------------------------------------ | ------------------------------- |
| `q2_k`   | 2-bit quantization (k-quants)        | Lowest precision, smallest size |
| `q3_k_s` | 3-bit quantization (small)           | Very low precision              |
| `q3_k_m` | 3-bit quantization (medium)          | Very low precision              |
| `q3_k_l` | 3-bit quantization (large)           | Very low precision              |
| `q4_0`   | 4-bit quantization (legacy)          | Low precision                   |
| `q4_1`   | 4-bit quantization (improved legacy) | Low precision                   |
| `q4_k_s` | 4-bit quantization (small)           | Low precision                   |
| `q4_k_m` | 4-bit quantization (medium)          | Low precision                   |
| `q5_0`   | 5-bit quantization (legacy)          | Medium precision                |
| `q5_1`   | 5-bit quantization (improved legacy) | Medium precision                |
| `q5_k_s` | 5-bit quantization (small)           | Medium precision                |
| `q5_k_m` | 5-bit quantization (medium)          | Medium precision                |
| `q6_k`   | 6-bit quantization (k-quants)        | Higher precision                |
| `q8_0`   | 8-bit quantization                   | Highest precision, largest size |

### Splitting Options

| Option                    | Description                              | Default |
| ------------------------- | ---------------------------------------- | ------- |
| `--split`                 | Split the GGUF file into multiple chunks | `False` |
| `--split-max-size SIZE`   | Max size per split (e.g., "500M", "2G")  | None    |
| `--split-max-tensors NUM` | Max tensors per split                    | 128     |

## Processing Workflow

The script performs operations in the following sequence:

1. **Model Acquisition**:

   - If `--skip-download` is not set: Download model from Hugging Face
   - If `--skip-download` is set: Use existing model files at `--dir`

2. **GGUF Conversion**:

   - Convert the model to GGUF format using llama.cpp's `convert_hf_to_gguf.py`
   - Output file: `<model_name>.gguf` in the model directory

3. **Quantization** (if `--quantize` is specified):

   - Apply the specified quantization method using llama.cpp's `quantize` tool
   - Output file: `<model_name>_<quant_type>.gguf` in the model directory
   - Remove the unquantized GGUF file to save space

4. **Splitting** (if `--split` is specified):

   - Split the GGUF file using llama.cpp's `gguf-split` tool
   - Output files: `<model_name>[-<quant_type>]-00001-of-XXXXX.gguf` etc.
   - Remove the original unsplit GGUF file to save space

5. **Cleanup** (if `--cleanup` is specified):
   - Remove all original model files except the final processed GGUF file
   - Preserve only the final output (quantized and/or split GGUF file)

## Examples

### 1. Download and Convert to GGUF Only

```bash
python download_and_quantize.py --model-id lmsys/vicuna-13b-v1.5
```

This downloads the Vicuna 13B v1.5 model from Hugging Face and converts it to GGUF format without quantization or splitting.

### 2. Download, Convert, and Quantize

```bash
python download_and_quantize.py --model-id lmsys/vicuna-13b-v1.5 --quantize q4_0
```

This downloads the model, converts to GGUF, and applies 4-bit quantization (q4_0) to reduce size.

### 3. Download, Quantize, and Clean Up Original Files

```bash
python download_and_quantize.py --model-id lmsys/vicuna-13b-v1.5 --quantize q4_0 --cleanup
```

Same as Example 2, but removes the original model files after conversion, keeping only the quantized GGUF file.

### 4. Download, Quantize, and Split

```bash
python download_and_quantize.py --model-id lmsys/vicuna-13b-v1.5 --quantize q4_0 --split --split-max-size 2G
```

Download, convert to GGUF, quantize, and split into multiple files of maximum 2GB each.

### 5. Work with Pre-existing Model

```bash
python download_and_quantize.py --skip-download --dir /path/to/existing/model --quantize q5_k_m
```

Skip downloading and process an existing model at the specified directory with 5-bit k-quants medium quantization.

### 6. Merge LoRA Weights with Base Model

```bash
python download_and_quantize.py --model-id black-forest-labs/FLUX.1-dev \
  --lora-model-id Heartsync/Flux-NSFW-uncensored --quantize q4_k_m --model-name flux-nsfw
```

Download FLUX.1-dev model, merge with the Heartsync/Flux-NSFW-uncensored LoRA weights, convert the merged model to GGUF, and quantize using 4-bit k-quants medium.

### 7. Complete Pipeline with LoRA Merging and All Options

```bash
python download_and_quantize.py --model-id black-forest-labs/FLUX.1-dev \
  --lora-model-id Heartsync/Flux-NSFW-uncensored \
  --dir /models/flux-nsfw --model-name flux-nsfw-merged \
  --quantize q4_k_m --cleanup --split --split-max-size 1G
```

Download FLUX model, merge with NSFW LoRA weights, convert to GGUF, apply q4_k_m quantization, split into 1GB chunks, and clean up original files.

### 8. Complete Pipeline with All Options (without LoRA)

```bash
python download_and_quantize.py --model-id lmsys/vicuna-13b-v1.5 --dir /models/vicuna \
  --revision v1.5 --quantize q4_0 --cleanup --split --split-max-size 1G
```

Download Vicuna model to specific directory, convert to GGUF, apply q4_0 quantization, split into 1GB chunks, and clean up original files.

## Return Values

The script returns and prints:

1. Path to the model directory
2. Path to the final processed GGUF file (or first split file)

## Advanced Usage

### Programmatic Usage

You can import and use the main function in your Python code:

```python
from download_and_quantize import download_and_quantize

model_dir, final_path = download_and_quantize(
    model_id="lmsys/vicuna-13b-v1.5",
    quant_type="q4_0",
    cleanup=True
)
```

### Environmental Considerations

- **Disk Space**: Processing large language models requires significant temporary disk space, especially before quantization
- **Memory Usage**: Converting and quantizing large models may require substantial memory
- **Network Bandwidth**: Downloading models from Hugging Face can be bandwidth-intensive

## Troubleshooting

### Common Issues

1. **Insufficient Disk Space**:

   - Error: `OSError: [Errno 28] No space left on device`
   - Solution: Free up disk space or use a volume with more capacity

2. **Model Not Found**:

   - Error: `huggingface_hub.utils._errors.RepositoryNotFoundError`
   - Solution: Verify the model ID and your internet connection

3. **Conversion Failures**:

   - Error: `subprocess.CalledProcessError` during conversion
   - Solution: Check model compatibility with llama.cpp's converter

4. **Split File Not Found**:

   - Error: `FileNotFoundError: No split files found with prefix`
   - Solution: Check disk permissions and available space

5. **llama.cpp Tools Not Found**:
   - Error: `FileNotFoundError: Could not find the 'quantize' tool in any of the expected locations.`
   - Solution:
     - Make sure llama.cpp is properly installed and compiled
     - Check that tools are in one of the expected paths: `/llama.cpp`, `/app/llama.cpp`, parent directory, or `~/llama.cpp`
     - For conversion script: Look in main directory or `examples` subdirectory
     - For binary tools: Look in main directory, `bin` or `build/bin` subdirectories
     - Make sure binary tools have execution permissions

## Technical Details

### File Naming Conventions

- Original GGUF file: `<model_name>.gguf`
- Quantized file: `<model_name>_<quant_type>.gguf`
- Split files: `<name>-00001-of-XXXXX.gguf`, `<name>-00002-of-XXXXX.gguf`, etc.
- LoRA merged model: When using `--lora-model-id`, the output file naming will be based on the `--model-name` parameter if provided, or automatically generated as `<base_model>-with-<lora_model>`.

### LoRA Merging Process

When merging a LoRA adapter with a base model:

1. The base model is downloaded and loaded using the appropriate Hugging Face model class
2. The LoRA adapter is downloaded and loaded using the PEFT library
3. The weights are merged into the base model
4. The merged model is saved to a new directory
5. The GGUF conversion process is then performed on this merged model
6. The same quantization and splitting options can be applied to the merged model

The script will automatically attempt to detect the appropriate model type for loading:

- First tries to load the model as an image generation model (suitable for FLUX, SD3, etc.)
- If that fails, attempts to load as a vision-language model
- Finally falls back to loading as a text generation model

This allows the script to handle a variety of model architectures for LoRA merging, particularly useful for:

- Text generation models like LLaMA, Mixtral, Qwen
- Image generation models like FLUX, Stable Diffusion
- Multimodal models with both text and image capabilities

### Disk Space Requirements

Approximate space needed for common model sizes:

| Model Size | Original HF | GGUF   | GGUF q4_0 | GGUF q5_k_m |
| ---------- | ----------- | ------ | --------- | ----------- |
| 7B         | ~15GB       | ~14GB  | ~4GB      | ~5GB        |
| 13B        | ~30GB       | ~27GB  | ~8GB      | ~10GB       |
| 70B        | ~140GB      | ~130GB | ~40GB     | ~50GB       |

### Memory Requirements

Converting and quantizing large models may require substantial RAM:

- 7B models: ~16GB RAM
- 13B models: ~32GB RAM
- 70B models: ~150GB RAM or more

## Limitations

- Only supports Hugging Face models compatible with llama.cpp's converter
- Requires llama.cpp to be installed and properly compiled in one of the supported locations
- Designed for transformer-based models (language models, diffusion models, multimodal models)
- LoRA merging requires sufficient RAM to load both the base model and adapter weights
- Some specialized model architectures may require additional customization for LoRA merging
- For image generation models with LoRA adapters, additional image-specific configuration may be needed

## License

This script is distributed under the same license as the llmmllab project.

## Credits

- llama.cpp: https://github.com/ggerganov/llama.cpp
- Hugging Face: https://huggingface.co/
