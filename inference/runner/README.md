# LLM ML Lab Runner

This project provides model running, downloading, and management capabilities for language models.

## Overview

The LLM ML Lab Runner project provides:

- Pipeline implementations for text and image generation
- Scripts for downloading, transforming, and quantizing models
- Configuration management for model settings

## Installation

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running Models Directly

```bash
# Run a text generation model
python -m runner --model model_name --prompt "Your prompt here"

# Run an image generation model
python -m runner --model stable_diffusion --prompt "An image of a cat" --output cat.png
```

### Downloading and Processing Models

```bash
# Download a model from Hugging Face
python -m scripts.download --model meta-llama/Llama-2-7b-chat-hf --output ./models

# Download and quantize a model
python -m scripts.download_and_quantize --model meta-llama/Llama-2-7b-chat-hf --bits 4
```

## Project Structure

- `pipelines/`: Model inference pipelines for different tasks
  - `txt2txt/`: Text generation pipelines
  - `txt2img/`: Text-to-image generation pipelines
  - `img2img/`: Image-to-image generation pipelines
  - `imgtxt2txt/`: Multimodal pipelines
- `scripts/`: Utility scripts for model management
- `config/`: Configuration files for models

## Configuration

The system uses configuration files in the `config/` directory:

- `models.json`: Model configurations and parameters

## License

[License information]
