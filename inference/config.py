import os
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("inference-service")

# Image storage configuration
IMAGE_DIR = "/root/images"

# Model configuration
DEFAULT_MODEL_ID = "black-forest-labs-flux.1-dev"
DEFAULT_MODEL_SOURCE = "stabilityai/stable-diffusion-3.5-large"
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
MAISTRO_BASE_URL = os.environ.get("MAISTRO_BASE_URL", "http://maistro.maistro.svc.cluster.local:8080")

# Image retention period (in hours)
IMAGE_RETENTION_HOURS = int(os.environ.get("IMAGE_RETENTION_HOURS", "1"))

# Stable Diffusion generation parameters
DEFAULT_NUM_INFERENCE_STEPS = int(
    os.environ.get("DEFAULT_NUM_INFERENCE_STEPS", "50"))
DEFAULT_GUIDANCE_SCALE = float(os.environ.get("DEFAULT_GUIDANCE_SCALE", "7.5"))
USE_SAFETY_CHECKER = os.environ.get("USE_SAFETY_CHECKER", "").lower() == "true"

# SDXL default sizes
SDXL_DEFAULT_WIDTH = int(os.environ.get("SDXL_DEFAULT_WIDTH", "1024"))
SDXL_DEFAULT_HEIGHT = int(os.environ.get("SDXL_DEFAULT_HEIGHT", "1024"))
SDXL_AESTHETIC_SCORE = float(os.environ.get("SDXL_AESTHETIC_SCORE", "6.0"))

# Memory optimization settings
# Don't use xformers by default since it's not installed
ENABLE_MEMORY_EFFICIENT_ATTENTION = os.environ.get(
    "ENABLE_MEMORY_EFFICIENT_ATTENTION", "").lower() == "true"
# Enable VAE slicing for memory efficiency
ENABLE_VAE_SLICING = os.environ.get(
    "ENABLE_VAE_SLICING", "true").lower() == "true"
# Enable CPU offloading for low memory (slower but uses less VRAM)
ENABLE_MODEL_CPU_OFFLOAD = os.environ.get(
    "ENABLE_MODEL_CPU_OFFLOAD", "").lower() == "true"
# Enable sequential CPU offload (even less VRAM but slower)
ENABLE_SEQUENTIAL_CPU_OFFLOAD = os.environ.get(
    "ENABLE_SEQUENTIAL_CPU_OFFLOAD", "").lower() == "true"
# Use FP16 precision to reduce memory usage (only on CUDA/GPU devices)
USE_FP16_PRECISION = os.environ.get(
    "USE_FP16_PRECISION", "true").lower() == "true"
# Check if we should actually use FP16 based on device availability
FORCE_FP32_ON_CPU = os.environ.get(
    "FORCE_FP32_ON_CPU", "true").lower() == "true"
# Lower inference steps when memory is constrained
INFERENCE_STEPS_LOW_MEMORY = int(
    os.environ.get("INFERENCE_STEPS_LOW_MEMORY", "20"))
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

RABBITMQ_HOST = os.environ.get("RABBITMQ_HOST", "rabbitmq-0.rabbitmq.rabbitmq.svc.cluster.local")
RABBITMQ_PORT = os.environ.get("RABBITMQ_PORT", 5672)
RABBITMQ_USER = os.environ.get("RABBITMQ_USER", "lsm")
RABBITMQ_PASSWORD = os.environ.get("RABBITMQ_PASSWORD", "")
RABBITMQ_VHOST = os.environ.get("RABBITMQ_VHOST", "/")
