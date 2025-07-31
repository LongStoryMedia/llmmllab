#!/usr/bin/env python3
"""
Enhanced startup script for the integrated Stable Diffusion + vLLM server
"""

import argparse
import os
import sys
import uvicorn
import logging
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            (
                logging.FileHandler("/var/log/inference-server.log", mode="a")
                if os.path.exists("/var/log")
                else logging.NullHandler()
            ),
        ],
    )


def check_requirements():
    """Check if required dependencies are available."""
    try:
        import torch

        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    except ImportError:
        print("Warning: PyTorch not available")

    try:
        import vllm

        print(f"vLLM version: {vllm.__version__}")
    except ImportError:
        print("Warning: vLLM not available - OpenAI endpoints will not work")

    try:
        import diffusers

        print(f"Diffusers version: {diffusers.__version__}")
    except ImportError:
        print("Warning: Diffusers not available - Image generation may not work")


def setup_environment():
    """Setup environment variables and directories."""
    # Create required directories
    os.makedirs("/root/images", exist_ok=True)
    os.makedirs("/app/config", exist_ok=True)
    os.makedirs("/var/log", exist_ok=True)

    # Set default environment variables if not already set
    env_defaults = {
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "HF_HOME": "/root/.cache/huggingface",
        "VLLM_MODEL": "microsoft/DialoGPT-medium",
        "VLLM_GPU_MEMORY_UTILIZATION": "0.8",
        "VLLM_MAX_MODEL_LEN": "2048",
        "IMAGE_RETENTION_HOURS": "24",
        "LOG_LEVEL": "INFO",
    }

    for key, value in env_defaults.items():
        if key not in os.environ:
            os.environ[key] = value
            print(f"Set {key}={value}")


def main():
    parser = argparse.ArgumentParser(
        description="Integrated Stable Diffusion + vLLM Server"
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of worker processes"
    )
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    parser.add_argument("--vllm-model", type=str, help="vLLM model to use")
    parser.add_argument(
        "--vllm-preset",
        type=str,
        choices=["chat", "code", "instruct", "llama2-7b", "llama2-13b"],
        help="Use a predefined vLLM model preset",
    )
    parser.add_argument(
        "--disable-vllm",
        action="store_true",
        help="Disable vLLM/OpenAI compatibility (saves memory)",
    )
    parser.add_argument(
        "--check-requirements",
        action="store_true",
        help="Check system requirements and exit",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Check requirements if requested
    if args.check_requirements:
        check_requirements()
        return

    # Setup environment
    setup_environment()

    # Override environment variables with command line arguments
    if args.vllm_model:
        os.environ["VLLM_MODEL"] = args.vllm_model
        logger.info(f"Using vLLM model: {args.vllm_model}")

    if args.vllm_preset:
        # Set preset-specific environment variables
        from .config import VLLM_MODEL_PRESETS

        if args.vllm_preset in VLLM_MODEL_PRESETS:
            preset = VLLM_MODEL_PRESETS[args.vllm_preset]
            os.environ["VLLM_MODEL"] = preset["model"]
            os.environ["VLLM_MAX_MODEL_LEN"] = str(preset["max_model_len"])
            os.environ["VLLM_TENSOR_PARALLEL_SIZE"] = str(
                preset["tensor_parallel_size"]
            )
            os.environ["VLLM_GPU_MEMORY_UTILIZATION"] = str(
                preset["gpu_memory_utilization"]
            )
            logger.info(f"Using vLLM preset: {args.vllm_preset} -> {preset['model']}")

    if args.disable_vllm:
        os.environ["DISABLE_VLLM"] = "true"
        logger.info("vLLM/OpenAI compatibility disabled")

    # Log startup configuration
    logger.info("=== Server Configuration ===")
    logger.info(f"Host: {args.host}:{args.port}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Reload: {args.reload}")
    logger.info(f"Log Level: {args.log_level}")
    logger.info(f"vLLM Model: {os.environ.get('VLLM_MODEL', 'Not set')}")
    logger.info(f"vLLM Disabled: {os.environ.get('DISABLE_VLLM', 'false')}")
    logger.info("=" * 30)

    # Check system requirements
    check_requirements()

    # Import the app after environment setup
    try:
        from app import app

        logger.info("Application imported successfully")
    except ImportError as e:
        logger.error(f"Failed to import application: {e}")
        sys.exit(1)

    # Configure uvicorn
    uvicorn_config = {
        "app": "app:app",
        "host": args.host,
        "port": args.port,
        "log_level": args.log_level.lower(),
        "access_log": True,
        "workers": args.workers if not args.reload else 1,  # Single worker for reload
        "reload": args.reload,
    }

    # Add SSL configuration if certificates are available
    cert_file = os.environ.get("SSL_CERT_FILE")
    key_file = os.environ.get("SSL_KEY_FILE")
    if (
        cert_file
        and key_file
        and os.path.exists(cert_file)
        and os.path.exists(key_file)
    ):
        uvicorn_config.update(
            {
                "ssl_certfile": cert_file,
                "ssl_keyfile": key_file,
            }
        )
        logger.info("SSL enabled")

    logger.info("Starting server...")

    try:
        uvicorn.run(**uvicorn_config)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
