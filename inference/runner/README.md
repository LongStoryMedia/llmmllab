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

Note: Model download and processing scripts have been removed. Use standard Hugging Face tools for model management.

## Project Structure

- `pipelines/`: Model inference pipelines for different tasks
  - `txt2txt/`: Text generation pipelines
  - `txt2img/`: Text-to-image generation pipelines
  - `img2img/`: Image-to-image generation pipelines
  - `imgtxt2txt/`: Multimodal pipelines
- `config/`: Configuration files for models

## Configuration

The system uses configuration files in the `config/` directory:

- `models.json`: Model configurations and parameters

## License

[License information]
