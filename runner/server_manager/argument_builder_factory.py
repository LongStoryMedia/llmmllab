"""
Argument Builder Factory - Creates argument builders for different server types.

This module provides a factory function for creating appropriate argument
builders based on the server type.
"""

from typing import Optional

from runner.models import Model, ModelProfile, UserConfig
from .base_argument_builder import BaseArgumentBuilder
from .llamacpp_argument_builder import LlamaCppArgumentBuilder


def create_argument_builder(
    server_type: str,
    model: Model,
    profile: ModelProfile,
    user_config: Optional[UserConfig] = None,
    port: Optional[int] = None,
    is_embedding: bool = False,
) -> BaseArgumentBuilder:
    """Create an argument builder for the specified server type."""
    builders = {
        "llamacpp": LlamaCppArgumentBuilder,
    }

    if server_type not in builders:
        raise ValueError(
            f"Unknown server type: {server_type}. Available: {list(builders.keys())}"
        )

    return builders[server_type](
        model=model,
        profile=profile,
        user_config=user_config,
        port=port,
        is_embedding=is_embedding,
    )
