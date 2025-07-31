#!/usr/bin/env python3
import argparse
from huggingface_hub import snapshot_download


def download_model(model_id, local_dir=None, use_symlinks=False, revision="main"):
    """
    Download a model from Hugging Face Hub

    Args:
        model_id (str): The model ID on Hugging Face Hub (e.g., 'lmsys/vicuna-13b-v1.5')
        local_dir (str): Local directory to save the model to. If None, uses the model name
        use_symlinks (bool): Whether to use symlinks for the downloaded files
        revision (str): The revision to download
    """
    if local_dir is None:
        # Extract model name from model_id for the directory name
        local_dir = model_id.split('/')[-1] if '/' in model_id else model_id

    print(f"Downloading model {model_id} to {local_dir}...")
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=use_symlinks,
        revision=revision
    )
    print(f"Model {model_id} downloaded successfully to {local_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download models from Hugging Face Hub")
    parser.add_argument("model_id", help="The model ID on Hugging Face Hub (e.g., 'lmsys/vicuna-13b-v1.5')")
    parser.add_argument("--dir", help="Local directory to save the model to (defaults to model name)")
    parser.add_argument("--use-symlinks", action="store_true", help="Enable use of symlinks for downloaded files")
    parser.add_argument("--revision", default="main", help="The revision to download (default: 'main')")

    args = parser.parse_args()

    download_model(
        model_id=args.model_id,
        local_dir=args.dir,
        use_symlinks=args.use_symlinks,
        revision=args.revision
    )
