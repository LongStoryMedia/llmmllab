import os

# Image storage configuration
IMAGE_DIR = "/app/images"

# Model configuration
DEFAULT_MODEL_ID = "default"
DEFAULT_MODEL_SOURCE = "stable-diffusion-v1-5/stable-diffusion-v1-5"
MODEL_NAME = DEFAULT_MODEL_SOURCE  # For backwards compatibility

# Default HuggingFace cache location
DEFAULT_HF_CACHE = "/root/.cache/huggingface"
# Use environment variable if set, otherwise use default
HF_HOME = os.environ.get("HF_HOME", DEFAULT_HF_CACHE)

# Config storage
CONFIG_DIR = "/app/config"
MODELS_CONFIG_PATH = os.path.join(CONFIG_DIR, "models.json")

# Image retention period (in hours)
IMAGE_RETENTION_HOURS = 1

# Stable Diffusion generation parameters
DEFAULT_NUM_INFERENCE_STEPS = 50
DEFAULT_GUIDANCE_SCALE = 7.5
USE_SAFETY_CHECKER = False

# SDXL default sizes
SDXL_DEFAULT_WIDTH = 1024
SDXL_DEFAULT_HEIGHT = 1024
SDXL_AESTHETIC_SCORE = 6.0

# Memory optimization settings
ENABLE_MEMORY_EFFICIENT_ATTENTION = True  # Use memory efficient attention
# Enable VAE slicing for memory efficiency
ENABLE_VAE_SLICING = True
# Enable CPU offloading for low memory (slower but uses less VRAM)
ENABLE_MODEL_CPU_OFFLOAD = False
# Enable sequential CPU offload (even less VRAM but slower)
ENABLE_SEQUENTIAL_CPU_OFFLOAD = False
# Use FP16 precision to reduce memory usage
USE_FP16_PRECISION = True
# Lower inference steps when memory is constrained
INFERENCE_STEPS_LOW_MEMORY = 20
# Smaller height when memory is constrained
HEIGHT_LOW_MEMORY = 512
WIDTH_LOW_MEMORY = 512                    # Smaller width when memory is constrained

# Default image dimensions (can be reduced for memory efficiency)
DEFAULT_HEIGHT = 768
DEFAULT_WIDTH = 768
MAX_HEIGHT = 1024
MAX_WIDTH = 1024

# LoRA configuration
LORAS_CONFIG_PATH = os.path.join(CONFIG_DIR, "loras.json")
DEFAULT_LORA_WEIGHT = 0.75  # Default weight for LoRA adaptation
