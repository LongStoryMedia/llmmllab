import os
from typing import Any, Dict
from utils.logging import llmmllogger


logger = llmmllogger.logger.bind(component="Server")

# API versioning configuration
API_VERSION = os.environ.get("API_VERSION", "v1")
logger.info(f"API version: {API_VERSION}")

# Authentication configuration
AUTH_ISSUER = os.environ.get("AUTH_ISSUER", "https://auth.longstorymedia.com")
AUTH_AUDIENCE = os.environ.get("AUTH_AUDIENCE", "lsm-client")
AUTH_CLIENT_ID = os.environ.get("AUTH_CLIENT_ID", "lsm-client")
AUTH_CLIENT_SECRET = os.environ.get("AUTH_CLIENT_SECRET", "")
AUTH_JWKS_URI = os.environ.get("AUTH_JWKS_URI", "https://auth.longstorymedia.com/keys")

# Image storage configuration
IMAGE_DIR = os.environ.get("IMAGE_DIR", "/root/images")

# Database configuration
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = int(os.environ.get("DB_PORT", "5432"))
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "")
DB_NAME = os.environ.get("DB_NAME", "llmmll")
DB_SSLMODE = os.environ.get("DB_SSLMODE", "disable")

# Build proper PostgreSQL connection string
# Format: postgresql://user:password@host:port/dbname?sslmode=value
DB_CONNECTION_STRING = os.environ.get(
    "DB_CONNECTION_STRING",
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?sslmode={DB_SSLMODE}",
)

# Log connection availability without exposing credentials
if DB_CONNECTION_STRING:
    parts = DB_CONNECTION_STRING.split("@")
    if len(parts) > 1:
        masked_conn = f"***@{parts[1]}"
        logger.debug(f"Database connection string available: {masked_conn}")
else:
    logger.warning("No database connection string available")

# Model configuration
DEFAULT_MODEL_ID = "black-forest-labs-flux.1-dev"
DEFAULT_MODEL_SOURCE = "black-forest-labs-flux.1-dev"
MODEL_NAME = DEFAULT_MODEL_SOURCE  # For backwards compatibility

# Redis configuration
REDIS_ENABLED = os.environ.get("REDIS_ENABLED", "true").lower() == "true"
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "")
REDIS_DB = int(os.environ.get("REDIS_DB", "0"))
REDIS_CONVERSATION_TTL = int(
    os.environ.get("REDIS_CONVERSATION_TTL", "360")
)  # in minutes
REDIS_MESSAGE_TTL = int(os.environ.get("REDIS_MESSAGE_TTL", "180"))  # in minutes
REDIS_SUMMARY_TTL = int(os.environ.get("REDIS_SUMMARY_TTL", "720"))  # in minutes
REDIS_POOL_SIZE = int(os.environ.get("REDIS_POOL_SIZE", "10"))
REDIS_MIN_IDLE_CONNECTIONS = int(os.environ.get("REDIS_MIN_IDLE_CONNECTIONS", "2"))
REDIS_CONNECT_TIMEOUT = int(os.environ.get("REDIS_CONNECT_TIMEOUT", "5"))

# Default HuggingFace cache location
DEFAULT_HF_CACHE = "/root/.cache/huggingface"
# Use environment variable if set, otherwise use default
HF_HOME = os.environ.get("HF_HOME", DEFAULT_HF_CACHE)

# Summarization configuration
MESSAGES_BEFORE_SUMMARY = int(os.environ.get("MESSAGES_BEFORE_SUMMARY", "6"))
SUMMARIES_BEFORE_CONSOLIDATION = int(
    os.environ.get("SUMMARIES_BEFORE_CONSOLIDATION", "3")
)
SUMMARY_MODEL = os.environ.get("SUMMARY_MODEL", "qwen3:0.6b")
SUMMARY_SYSTEM_PROMPT = os.environ.get(
    "SUMMARY_SYSTEM_PROMPT",
    "compress the following conversation so far into a concise paragraph. Include key points and conclusions, "
    "but omit redundant details. The summary will be used as context for future interaction. "
    "It should be as small as possible and does not need to be human readable.",
)
MAX_SUMMARY_LEVELS = int(os.environ.get("MAX_SUMMARY_LEVELS", "3"))

# Config storage
CONFIG_DIR = os.environ.get("CONFIG_DIR", "/app/config")
MODELS_CONFIG_PATH = os.path.join("/app", ".models.json")

# Inference services configuration
INFERENCE_SERVICES_OLLAMA_BASE_URL = os.environ.get(
    "INFERENCE_SERVICES_OLLAMA_BASE_URL", "http://localhost:11434"
)
INFERENCE_SERVICES_SD_BASE_URL = os.environ.get(
    "INFERENCE_SERVICES_SD_BASE_URL", "http://localhost:8000"
)
INFERENCE_SERVICES_HOST = os.environ.get("INFERENCE_SERVICES_HOST", "localhost")
INFERENCE_SERVICES_PORT = int(os.environ.get("INFERENCE_SERVICES_PORT", "50051"))

# Internal API key for secure communication with maistro
MAISTRO_INTERNAL_API_KEY = os.environ.get("MAISTRO_INTERNAL_API_KEY", "")
# Base URL for internal maistro communication
MAISTRO_BASE_URL = os.environ.get(
    "MAISTRO_BASE_URL", "http://maistro.maistro.svc.cluster.local:8080"
)

# Image generation configuration
IMAGE_GENERATION_ENABLED = (
    os.environ.get("IMAGE_GENERATION_ENABLED", "true").lower() == "true"
)
IMAGE_STORAGE_DIRECTORY = os.environ.get("IMAGE_STORAGE_DIRECTORY", IMAGE_DIR)
MAX_IMAGE_SIZE = int(os.environ.get("MAX_IMAGE_SIZE", "2048"))
# Image retention period (in hours)
IMAGE_RETENTION_HOURS = int(os.environ.get("IMAGE_RETENTION_HOURS", "24"))

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

# Default weight for LoRA adaptation
DEFAULT_LORA_WEIGHT = float(os.environ.get("DEFAULT_LORA_WEIGHT", "0.75"))
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

os.environ.setdefault("PYTORCH_NO_CUDA_MEMORY_CACHING", "1")
# os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
#                       "expandable_segments:True")

# Inference service configuration

# Internal security configuration
INTERNAL_API_KEY = os.environ.get(
    "INTERNAL_API_KEY", "4a6204749cf376a6f4feaa529760fc52"
)
INTERNAL_ALLOWED_IPS = os.environ.get(
    "INTERNAL_ALLOWED_IPS", "192.168.0.0/24,10.43.0.0/16"
)

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


def get_redis_config() -> Dict[str, Any]:
    """Get Redis configuration."""
    return {
        "enabled": REDIS_ENABLED,
        "host": REDIS_HOST,
        "port": REDIS_PORT,
        "password": REDIS_PASSWORD,
        "db": REDIS_DB,
        "conversation_ttl": REDIS_CONVERSATION_TTL,
        "message_ttl": REDIS_MESSAGE_TTL,
        "summary_ttl": REDIS_SUMMARY_TTL,
        "pool_size": REDIS_POOL_SIZE,
        "min_idle_connections": REDIS_MIN_IDLE_CONNECTIONS,
        "connect_timeout": REDIS_CONNECT_TIMEOUT,
    }


def get_database_config() -> Dict[str, Any]:
    """Get database configuration."""
    return {
        "host": DB_HOST,
        "port": DB_PORT,
        "user": DB_USER,
        "password": DB_PASSWORD,
        "dbname": DB_NAME,
        "sslmode": DB_SSLMODE,
        "connection_string": DB_CONNECTION_STRING,
    }


def get_auth_config() -> Dict[str, Any]:
    """Get authentication configuration."""
    return {
        "issuer": AUTH_ISSUER,
        "audience": AUTH_AUDIENCE,
        "client_id": AUTH_CLIENT_ID,
        "client_secret": AUTH_CLIENT_SECRET,
        "jwks_uri": AUTH_JWKS_URI,
    }


def get_inference_services_config() -> Dict[str, Any]:
    """Get inference services configuration."""
    return {
        "ollama": {
            "base_url": INFERENCE_SERVICES_OLLAMA_BASE_URL,
        },
        "stable_diffusion": {
            "base_url": INFERENCE_SERVICES_SD_BASE_URL,
        },
        "host": INFERENCE_SERVICES_HOST,
        "port": INFERENCE_SERVICES_PORT,
    }
