"""Configuration utilities for resolving per-profile vs per-user config."""

from typing import Optional

from models.model_profile import ModelProfile
from models.user_config import UserConfig
from models.gpu_config import GPUConfig
from models.default_configs import DEFAULT_GPU_CONFIG
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="ConfigUtils")


def resolve_gpu_config(
    profile: ModelProfile, user_config: Optional[UserConfig] = None
) -> GPUConfig:
    """Resolve GPU configuration: profile override → user global → default."""
    if profile.gpu_config:
        return profile.gpu_config
    if user_config and user_config.gpu_config:
        return user_config.gpu_config
    return DEFAULT_GPU_CONFIG
