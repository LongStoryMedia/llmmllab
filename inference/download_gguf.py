#!/usr/bin/env python3
"""
Script to download GGUF model files from Hugging Face Hub.
This script downloads specific GGUF files for the Qwen3-30B-A3B model
and ensures GPU support is enabled.
"""

import os
import sys
import torch
from huggingface_hub import hf_hub_download

# Define the model repository
REPO_ID = "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF"
LOCAL_DIR = "/models/Mixtral-8x7B-Instruct-v0.1-GGUF"
MODEL_FILE = "mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf"  # The specific quantization we want


def check_gpu():
    """Check for GPU availability and report details."""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"✅ CUDA is available with {device_count} device(s):")
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            mem_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"   GPU {i}: {device_name} with {mem_total:.2f} GB memory")
        return True
    else:
        print("❌ CUDA is NOT available! Models will run on CPU only, which will be much slower.")
        return False


def download_model():
    """Download the GGUF model file."""
    print(f"Downloading {MODEL_FILE} from {REPO_ID}...")

    # Check GPU availability first
    has_gpu = check_gpu()
    if not has_gpu:
        print("WARNING: No GPU detected. Model will run on CPU, which will be very slow.")
        response = input("Do you want to continue anyway? (y/n): ").lower()
        if response != 'y':
            print("Download cancelled.")
            return False

    try:
        # Create directory if it doesn't exist
        os.makedirs(LOCAL_DIR, exist_ok=True)

        # Download the specific GGUF file
        file_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=MODEL_FILE,
            local_dir=LOCAL_DIR,
            local_dir_use_symlinks=False,
            resume_download=True,
        )

        print(f"Successfully downloaded model to: {file_path}")
        print(f"File size: {os.path.getsize(file_path) / (1024 * 1024 * 1024):.2f} GB")

        if has_gpu:
            print("\n✅ Model will use GPU acceleration for inference.")
            print("   For optimal performance with llama-cpp-python, ensure you have:")
            print("   1. Installed CUDA toolkit")
            print("   2. Installed llama-cpp-python with CUDA support:")
            print("      pip uninstall -y llama-cpp-python")
            print("      CMAKE_ARGS=\"-DLLAMA_CUBLAS=on\" pip install llama-cpp-python --no-cache-dir")
            print("   3. Set n_gpu_layers=-1 in your code (already configured)")

        return True
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        return False


if __name__ == "__main__":
    success = download_model()
    sys.exit(0 if success else 1)
