#!/usr/bin/env python3
"""
Test script to verify CUDA support in llama-cpp-python and measure performance.
"""

import os
import time
import torch
from llama_cpp import Llama


def main():
    print("CUDA Check for llama-cpp-python")
    print("=" * 40)

    # Check CUDA availability
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        # Get CUDA version using nvidia-smi instead
        import subprocess
        try:
            cuda_version = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]).decode().strip()
            print(f"CUDA driver version: {cuda_version}")
        except (subprocess.SubprocessError, FileNotFoundError):
            print("CUDA version: Unknown (nvidia-smi not available)")

        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("WARNING: CUDA is not available in PyTorch!")

    print("\nAttempting to load a small model with GGUF to test CUDA support...")

    # Get model path - default to the Qwen model if available
    model_path = "/models/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please specify a valid model path with MODEL_PATH env var")
        model_path = os.environ.get("MODEL_PATH", None)
        if not model_path or not os.path.exists(model_path):
            print("No model available for testing")
            return

    print(f"Loading model from: {model_path}")

    # Test loading model with GPU layers
    try:
        # Measure loading time
        start_time = time.time()
        # Proportional split: 24GB / (24GB + 12GB) = 0.666...
        # We define the proportion of layers for each GPU. The rest will go to the last one.
        # This gives you precise control over the memory distribution.
        tensor_split_proportions = [2/3]

        model = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_gpu_layers=-1,         # Offload all layers
            tensor_split=tensor_split_proportions,  # Add this line!
            verbose=True,
            seed=42,
            offload_kqv=True
        )

        load_time = time.time() - start_time
        print(f"Model loaded with GPU support in {load_time:.2f} seconds")

        # Basic inference test
        prompt = "Write a single sentence about machine learning."

        print("\nRunning inference test...")
        start_time = time.time()
        inference_time = 0
        try:
            output = model.create_completion(
                prompt=prompt,
                max_tokens=20,
                temperature=0.7,
                stop=[".", "\n"],
                echo=True,
                stream=False  # Make sure we get a direct result, not a stream
            )
            inference_time = time.time() - start_time
            print(f"Inference completed in {inference_time:.2f} seconds")

            # Print the output safely
            print("Output:")
            print(output)

            # Try to extract the text output
            try:
                if isinstance(output, dict):
                    choices = output.get('choices', [])
                    if choices and len(choices) > 0:
                        text = choices[0].get('text', '')
                        print(f"Generated text: {text}")
            except:
                pass
        except Exception as e:
            print(f"Inference error: {str(e)}")

        # Check if CUDA was actually used
        if inference_time > 0 and inference_time < 5:  # This is a rough heuristic - GPU should be much faster
            print("\n✅ GPU acceleration is working correctly!")
        else:
            print("\n⚠️  Inference was slow. GPU acceleration might not be working.")
            print("    Verify that llama-cpp-python was built with CUDA support.")

    except Exception as e:
        print(f"Error: {str(e)}")
        if "n_gpu_layers > 0 but GGML_USE_CUBLAS is not defined" in str(e):
            print("\n❌ llama-cpp-python was NOT compiled with CUDA support!")
            print("   Please reinstall with: CMAKE_ARGS=\"-DLLAMA_CUBLAS=on\" pip install --no-cache-dir llama-cpp-python")
        else:
            print("An unexpected error occurred")


if __name__ == "__main__":
    main()
