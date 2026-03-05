"""
Configuration utilities for hierarchical configuration patterns.
"""

from typing import Optional

from models.parameter_optimization_config import ParameterOptimizationConfig
from models.model_profile import ModelProfile
from models.user_config import UserConfig, ParameterOptimizationConfig
from models.gpu_config import GPUConfig
from models.default_configs import DEFAULT_GPU_CONFIG
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="ConfigUtils")


def resolve_parameter_optimization_config(
    profile: ModelProfile, user_config: Optional[UserConfig] = None
) -> ParameterOptimizationConfig:
    """
    Resolve parameter optimization configuration using hierarchical pattern.

    Priority (highest to lowest):
    1. Model profile specific configuration (profile.parameter_optimization)
    2. Global user configuration (user_config.parameter_optimization)
    3. None (disabled)

    Args:
        profile: Model profile that may have parameter_optimization override
        user_config: User configuration with global parameter_optimization settings

    Returns:
        Resolved parameter optimization configuration or None if disabled
    """
    # Check for profile-specific override first
    profile_config = getattr(profile, "parameter_optimization", None)
    if profile_config is not None:
        logger.info(
            f"🎯 Using profile-specific parameter optimization for {profile.name}"
        )
        return profile_config

    # Fall back to global user configuration
    if user_config and hasattr(user_config, "parameter_optimization"):
        global_config = user_config.parameter_optimization
        if global_config and global_config.enabled:
            logger.info(f"🎯 Using global parameter optimization for {profile.name}")
            return global_config

    # No configuration found
    logger.debug(f"❌ Parameter optimization disabled for {profile.name}")
    return None


def resolve_gpu_config(
    profile: ModelProfile, user_config: Optional[UserConfig] = None
) -> GPUConfig:
    """
    Resolve GPU configuration using hierarchical precedence.

    Priority order:
    1. Profile-specific gpu_config (highest priority)
    2. Global user_config.gpu_config
    3. DEFAULT_GPU_CONFIG (fallback)

    Args:
        profile: Model profile that may contain GPU config override
        user_config: User configuration that may contain global GPU config

    Returns:
        GPUConfig: Resolved GPU configuration (never None due to default fallback)
    """
    logger = llmmllogger.bind(component="ConfigUtils")

    # Check profile-specific override first (highest priority)
    if profile.gpu_config:
        logger.info(f"🎯 Using profile-specific GPU config for {profile.name}")
        return profile.gpu_config

    # Fall back to global user configuration
    if user_config and hasattr(user_config, "gpu_config") and user_config.gpu_config:
        logger.info(f"🎯 Using global GPU config for {profile.name}")
        return user_config.gpu_config

    # Use system default as final fallback
    logger.debug(f"🎯 Using default GPU config for {profile.name}")
    return DEFAULT_GPU_CONFIG
