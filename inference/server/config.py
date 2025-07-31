import os
import logging
from typing import Any, Dict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("inference-service")

# Image storage configuration
IMAGE_DIR = "/root/images"

# Model configuration
DEFAULT_MODEL_ID = "black-forest-labs-flux.1-dev"
DEFAULT_MODEL_SOURCE = "black-forest-labs-flux.1-dev"
MODEL_NAME = DEFAULT_MODEL_SOURCE  # For backwards compatibility

# Default HuggingFace cache location
DEFAULT_HF_CACHE = "/root/.cache/huggingface"
# Use environment variable if set, otherwise use default
HF_HOME = os.environ.get("HF_HOME", DEFAULT_HF_CACHE)

# Config storage
CONFIG_DIR = "/app/config"
MODELS_CONFIG_PATH = os.path.join(CONFIG_DIR, "models.json")
LORAS_CONFIG_PATH = os.path.join(CONFIG_DIR, "loras.json")

# Internal API key for secure communication with maistro
MAISTRO_INTERNAL_API_KEY = os.environ.get("MAISTRO_INTERNAL_API_KEY", "")
# Base URL for internal maistro communication
MAISTRO_BASE_URL = os.environ.get(
    "MAISTRO_BASE_URL", "http://maistro.maistro.svc.cluster.local:8080"
)

# Image retention period (in hours)
IMAGE_RETENTION_HOURS = int(os.environ.get("IMAGE_RETENTION_HOURS", "1"))

# Stable Diffusion generation parameters
DEFAULT_NUM_INFERENCE_STEPS = int(os.environ.get("DEFAULT_NUM_INFERENCE_STEPS", "50"))
DEFAULT_GUIDANCE_SCALE = float(os.environ.get("DEFAULT_GUIDANCE_SCALE", "7.5"))
USE_SAFETY_CHECKER = os.environ.get("USE_SAFETY_CHECKER", "").lower() == "true"

# SDXL default sizes
SDXL_DEFAULT_WIDTH = int(os.environ.get("SDXL_DEFAULT_WIDTH", "1024"))
SDXL_DEFAULT_HEIGHT = int(os.environ.get("SDXL_DEFAULT_HEIGHT", "1024"))
SDXL_AESTHETIC_SCORE = float(os.environ.get("SDXL_AESTHETIC_SCORE", "6.0"))

# Memory optimization settings
# Don't use xformers by default since it's not installed
ENABLE_MEMORY_EFFICIENT_ATTENTION = (
    os.environ.get("ENABLE_MEMORY_EFFICIENT_ATTENTION", "").lower() == "true"
)
# Enable VAE slicing for memory efficiency
ENABLE_VAE_SLICING = os.environ.get("ENABLE_VAE_SLICING", "true").lower() == "true"
# Enable CPU offloading for low memory (slower but uses less VRAM)
ENABLE_MODEL_CPU_OFFLOAD = (
    os.environ.get("ENABLE_MODEL_CPU_OFFLOAD", "").lower() == "true"
)
# Enable sequential CPU offload (even less VRAM but slower)
ENABLE_SEQUENTIAL_CPU_OFFLOAD = (
    os.environ.get("ENABLE_SEQUENTIAL_CPU_OFFLOAD", "").lower() == "true"
)
# Use FP16 precision to reduce memory usage (only on CUDA/GPU devices)
USE_FP16_PRECISION = os.environ.get("USE_FP16_PRECISION", "true").lower() == "true"
# Check if we should actually use FP16 based on device availability
FORCE_FP32_ON_CPU = os.environ.get("FORCE_FP32_ON_CPU", "true").lower() == "true"
# Lower inference steps when memory is constrained
INFERENCE_STEPS_LOW_MEMORY = int(os.environ.get("INFERENCE_STEPS_LOW_MEMORY", "20"))
# Smaller height and width when memory is constrained
HEIGHT_LOW_MEMORY = int(os.environ.get("HEIGHT_LOW_MEMORY", "512"))
WIDTH_LOW_MEMORY = int(os.environ.get("WIDTH_LOW_MEMORY", "512"))

# Default image dimensions (can be reduced for memory efficiency)
DEFAULT_HEIGHT = int(os.environ.get("DEFAULT_HEIGHT", "768"))
DEFAULT_WIDTH = int(os.environ.get("DEFAULT_WIDTH", "768"))
MAX_HEIGHT = int(os.environ.get("MAX_HEIGHT", "1024"))
MAX_WIDTH = int(os.environ.get("MAX_WIDTH", "1024"))

# LoRA configuration
LORAS_CONFIG_PATH = os.path.join(CONFIG_DIR, "loras.json")
# Default weight for LoRA adaptation
DEFAULT_LORA_WEIGHT = float(os.environ.get("DEFAULT_LORA_WEIGHT", "0.75"))
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

os.environ.setdefault("PYTORCH_NO_CUDA_MEMORY_CACHING", "1")
# os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
#                       "expandable_segments:True")

RABBITMQ_HOST = os.environ.get(
    "RABBITMQ_HOST", "rabbitmq-0.rabbitmq.rabbitmq.svc.cluster.local"
)
RABBITMQ_PORT = os.environ.get("RABBITMQ_PORT", 5672)
RABBITMQ_USER = os.environ.get("RABBITMQ_USER", "lsm")
RABBITMQ_PASSWORD = os.environ.get("RABBITMQ_PASSWORD", "")
RABBITMQ_VHOST = os.environ.get("RABBITMQ_VHOST", "/")


# gRPC server configuration
GRPC_PORT = int(os.environ.get("GRPC_PORT", "50051"))
GRPC_MAX_WORKERS = int(os.environ.get("GRPC_MAX_WORKERS", "10"))
GRPC_MAX_MESSAGE_SIZE = int(
    os.environ.get("GRPC_MAX_MESSAGE_SIZE", "104857600")
)  # 100MB
GRPC_MAX_CONCURRENT_RPCS = int(os.environ.get("GRPC_MAX_CONCURRENT_RPCS", "100"))
GRPC_ENABLE_REFLECTION = (
    os.environ.get("GRPC_ENABLE_REFLECTION", "true").lower() == "true"
)

# gRPC authentication
GRPC_REQUIRE_API_KEY = os.environ.get("GRPC_REQUIRE_API_KEY", "false").lower() == "true"
GRPC_API_KEY = os.environ.get("GRPC_API_KEY", "")  # API key for authentication

# TLS/SSL configuration
GRPC_USE_TLS = os.environ.get("GRPC_USE_TLS", "false").lower() == "true"
GRPC_CERT_FILE = os.environ.get("GRPC_CERT_FILE", "/etc/inference/certs/server.crt")
GRPC_KEY_FILE = os.environ.get("GRPC_KEY_FILE", "/etc/inference/certs/server.key")

# vLLM configuration for OpenAI compatibility
VLLM_MODEL = os.environ.get("VLLM_MODEL", "microsoft/DialoGPT-medium")
VLLM_TENSOR_PARALLEL_SIZE = int(os.environ.get("VLLM_TENSOR_PARALLEL_SIZE", "1"))
VLLM_GPU_MEMORY_UTILIZATION = float(
    os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.8")
)
VLLM_MAX_MODEL_LEN = int(os.environ.get("VLLM_MAX_MODEL_LEN", "2048"))
VLLM_DTYPE = os.environ.get("VLLM_DTYPE", "auto")
VLLM_TRUST_REMOTE_CODE = (
    os.environ.get("VLLM_TRUST_REMOTE_CODE", "true").lower() == "true"
)

# OpenAI API compatibility settings
OPENAI_DEFAULT_MAX_TOKENS = int(os.environ.get("OPENAI_DEFAULT_MAX_TOKENS", "512"))
OPENAI_DEFAULT_TEMPERATURE = float(os.environ.get("OPENAI_DEFAULT_TEMPERATURE", "0.7"))
OPENAI_MAX_CONCURRENT_REQUESTS = int(
    os.environ.get("OPENAI_MAX_CONCURRENT_REQUESTS", "10")
)

# Model presets for different use cases
VLLM_MODEL_PRESETS = {
    "chat": {
        "model": "microsoft/DialoGPT-medium",
        "max_model_len": 1024,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.8,
    },
    "code": {
        "model": "codellama/CodeLlama-7b-Instruct-hf",
        "max_model_len": 16384,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.9,
    },
    "instruct": {
        "model": "mistralai/Mistral-7B-Instruct-v0.1",
        "max_model_len": 8192,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.85,
    },
    "llama2-7b": {
        "model": "meta-llama/Llama-2-7b-chat-hf",
        "max_model_len": 4096,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.9,
    },
    "llama2-13b": {
        "model": "meta-llama/Llama-2-13b-chat-hf",
        "max_model_len": 4096,
        "tensor_parallel_size": 2,  # Requires 2 GPUs
        "gpu_memory_utilization": 0.9,
    },
}


def get_grpc_config() -> Dict[str, Any]:
    """Get the gRPC server configuration."""
    return {
        "port": GRPC_PORT,
        "max_workers": GRPC_MAX_WORKERS,
        "max_message_size": GRPC_MAX_MESSAGE_SIZE,
        "max_concurrent_rpcs": GRPC_MAX_CONCURRENT_RPCS,
        "enable_reflection": GRPC_ENABLE_REFLECTION,
        "require_api_key": GRPC_REQUIRE_API_KEY,
        "api_key": GRPC_API_KEY,
        "use_tls": GRPC_USE_TLS,
        "cert_file": GRPC_CERT_FILE,
        "key_file": GRPC_KEY_FILE,
    }


def get_vllm_config(preset: str) -> Dict[str, Any]:
    """Get vLLM configuration, optionally using a preset."""
    if preset and preset in VLLM_MODEL_PRESETS:
        config = VLLM_MODEL_PRESETS[preset].copy()
        # Override with environment variables if set
        config.update(
            {
                "model": os.environ.get("VLLM_MODEL", config["model"]),
                "tensor_parallel_size": int(
                    os.environ.get(
                        "VLLM_TENSOR_PARALLEL_SIZE", str(config["tensor_parallel_size"])
                    )
                ),
                "gpu_memory_utilization": float(
                    os.environ.get(
                        "VLLM_GPU_MEMORY_UTILIZATION",
                        str(config["gpu_memory_utilization"]),
                    )
                ),
                "max_model_len": int(
                    os.environ.get(
                        "VLLM_MAX_MODEL_LEN", str(config.get("max_model_len", 2048))
                    )
                ),
                "dtype": os.environ.get("VLLM_DTYPE", VLLM_DTYPE),
                "trust_remote_code": os.environ.get(
                    "VLLM_TRUST_REMOTE_CODE", str(VLLM_TRUST_REMOTE_CODE)
                ).lower()
                == "true",
            }
        )
        return config

    # Default configuration from environment variables
    return {
        "model": VLLM_MODEL,
        "tensor_parallel_size": VLLM_TENSOR_PARALLEL_SIZE,
        "gpu_memory_utilization": VLLM_GPU_MEMORY_UTILIZATION,
        "max_model_len": VLLM_MAX_MODEL_LEN,
        "dtype": VLLM_DTYPE,
        "trust_remote_code": VLLM_TRUST_REMOTE_CODE,
    }


def get_openai_config() -> Dict[str, Any]:
    """Get OpenAI compatibility configuration."""
    return {
        "default_max_tokens": OPENAI_DEFAULT_MAX_TOKENS,
        "default_temperature": OPENAI_DEFAULT_TEMPERATURE,
        "max_concurrent_requests": OPENAI_MAX_CONCURRENT_REQUESTS,
    }
